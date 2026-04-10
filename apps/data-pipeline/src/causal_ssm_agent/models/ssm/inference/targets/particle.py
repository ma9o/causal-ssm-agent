"""Particle filter likelihood backend via cuthbert bootstrap PF.

Computes log p(y|θ) by running a bootstrap particle filter and returning
the log normalizing constant. Differentiable via JAX autodiff — resampling
uses jnp.searchsorted (integer output, zero gradient), so gradients flow
through particle weights and propagation only.

With a fixed RNG key the PF likelihood is a deterministic function of θ,
making it suitable for NUTS sampling via numpyro.factor().

Use when:
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
- Any dynamics (linear or nonlinear)
- This is the universal backend for all SSM inference
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.artifacts.model_spec import DistributionFamily, LinkFunction
from causal_ssm_agent.models.ssm.discretization import discretize_system, discretize_system_batched
from causal_ssm_agent.models.ssm.inference.targets.kernels import (
    build_transition_kernel,
    compile_measurement_semantics,
    compile_transition_semantics,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    advance_support_observation_state,
    compile_observation_operator,
    summarize_support_observation,
    support_observation_log_prob,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cuthbert.smc.types import InitSample, LogPotential, PropagateSample
    from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray
    from jax import Array
    from jax.typing import ArrayLike

    from causal_ssm_agent.models.ssm.inference.targets.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
# =============================================================================
# JAX-native systematic resampling (gradient-safe on all platforms)
# =============================================================================
#
# cuthbert's built-in systematic resampling uses jax.pure_callback + numba on
# CPU, which does not support JVP.  We replace it with a pure-JAX version that
# uses jnp.searchsorted so that jax.grad can trace through the full PF
# (resampling indices are integers → zero gradient, which is correct).


def _systematic_resampling(key: KeyArray, logits: ArrayLike, n: int) -> Array:  # noqa: ARG001
    """Systematic resampling using pure JAX ops (no pure_callback).

    cuthbert's built-in systematic resampling uses jax.pure_callback + numba
    on CPU, which blocks JVP and therefore jax.grad / NUTS.  This version uses
    jnp.searchsorted directly, producing integer indices with zero gradient so
    that the full PF log-normalizing-constant is differentiable.

    Args:
        key: JAX PRNG key.
        logits: Log-weights, possibly un-normalized.  Shape (N,).
        n: Number of indices to sample (must equal logits.shape[0]).

    Returns:
        Integer index array of shape (n,).
    """
    logits_ = jnp.asarray(logits)
    N = logits_.shape[0]  # use static shape, not the traced `n` arg
    weights = jnp.exp(logits_ - jax.nn.logsumexp(logits_))
    cumsum = jnp.cumsum(weights)
    us = (random.uniform(key, ()) + jnp.arange(N)) / N
    idx = jnp.searchsorted(cumsum, us)
    return jnp.clip(idx, 0, N - 1).astype(jnp.int64)


# =============================================================================
# SSMAdapter -- maps CTParams to PF-compatible functions
# =============================================================================


class SSMAdapter:
    """Adapts CTParams into particle filter-compatible functions.

    Maps the continuous-time structure (drift, diffusion, measurement) into
    initial_sample, transition_sample, and observation_log_prob.

    Used by the bootstrap PF fallback (non-Gaussian dynamics, block_rb disabled).
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dists: Sequence[DistributionFamily | str] | None = None,
        diffusion_dists: Sequence[DistributionFamily | str] | None = None,
        manifest_links: Sequence[LinkFunction | str | None] | None = None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dists = (
            [
                dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
                for dist in manifest_dists
            ]
            if manifest_dists is not None
            else [DistributionFamily.GAUSSIAN] * n_manifest
        )
        self.diffusion_dists = (
            [
                dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
                for dist in diffusion_dists
            ]
            if diffusion_dists is not None
            else [DistributionFamily.GAUSSIAN] * n_latent
        )
        self.transition_semantics = compile_transition_semantics(self.diffusion_dists, n_latent)
        self.manifest_links: list[LinkFunction | str | None] | None = (
            list(manifest_links) if manifest_links is not None else None
        )

    def initial_sample(self, key: jax.Array, params: dict) -> jax.Array:
        """Sample eta_0 ~ N(t0_mean, t0_cov)."""
        t0_mean = params["t0_mean"]
        t0_cov = params["t0_cov"]
        chol = jla.cholesky(t0_cov + jnp.eye(self.n_latent) * 1e-6, lower=True)
        return t0_mean + chol @ random.normal(key, (self.n_latent,))

    def transition_sample(
        self, key: jax.Array, x_prev: jax.Array, params: dict, dt: float
    ) -> jax.Array:
        """Sample eta_t | eta_{t-1} via CT->DT discretization.

        For gaussian: eta_t ~ N(Ad * eta_{t-1} + cd, Qd)
        For student_t: same mean, but multivariate Student-t noise.
        """
        Ad, Qd, cd = discretize_system(
            params["drift"], params["diffusion_cov"], params.get("cint"), dt
        )
        mean = Ad @ x_prev
        if cd is not None:
            mean = mean + cd.flatten()
        chol = jla.cholesky(Qd + jnp.eye(self.n_latent) * 1e-6, lower=True)
        extra_params = {"proc_df": params.get("proc_df", 5.0)}
        trans_kernel = build_transition_kernel(self.transition_semantics, extra_params)
        return mean + trans_kernel.sample_noise_fn(key, chol)

    def observation_log_prob(
        self, y: jax.Array, x: jax.Array, params: dict, obs_mask: jax.Array
    ) -> float:
        """Compute log p(y | x) under measurement model.

        Compiles point-observation semantics on-the-fly for the bootstrap PF path.
        """
        H = params["lambda_mat"]
        d = params["manifest_means"]
        R = params["manifest_cov"]
        mask_float = obs_mask.astype(jnp.float64)
        extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        measurement_semantics = compile_measurement_semantics(
            self.manifest_dists,
            manifest_cov=R,
            extra_params=extra,
            manifest_links=self.manifest_links,
        )
        return measurement_semantics.obs_kernel.emission_fn(y, x, H, d, R, mask_float)


class SupportAwareParticleState(NamedTuple):
    """Per-particle state carrying interval-summary accumulators."""

    latent: jnp.ndarray
    response: jnp.ndarray
    accum_sum: jnp.ndarray
    accum_sumsq: jnp.ndarray
    accum_weight: jnp.ndarray
    obs_sum: jnp.ndarray
    obs_sumsq: jnp.ndarray
    obs_weight: jnp.ndarray


# =============================================================================
# ParticleLikelihood — LikelihoodBackend via cuthbert bootstrap PF
# =============================================================================


class ParticleLikelihood:
    """Particle filter likelihood backend via cuthbert bootstrap PF.

    Computes log p(y|theta) by running a bootstrap particle filter with a
    fixed RNG key, returning the log normalizing constant. Differentiable
    via JAX autodiff for use with NUTS.

    Args:
        n_latent: Number of latent states
        n_manifest: Number of manifest indicators
        n_particles: Number of particles (default 200)
        rng_key: Fixed JAX random key for deterministic PF
        manifest_dists: Per-channel observation noise families
        diffusion_dists: Per-latent process noise families
        ess_threshold: ESS/N threshold for resampling
    """

    checkpoint_loglik = True

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        n_particles: int = 200,
        rng_key: jax.Array | None = None,
        manifest_dists: Sequence[DistributionFamily | str] | None = None,
        diffusion_dists: Sequence[DistributionFamily | str] | None = None,
        ess_threshold: float = 0.5,
        block_rb: bool = True,
        manifest_links: Sequence[LinkFunction | str | None] | None = None,
        observation_support=None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.n_particles = n_particles
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.manifest_dists = (
            [
                dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
                for dist in manifest_dists
            ]
            if manifest_dists is not None
            else [DistributionFamily.GAUSSIAN] * n_manifest
        )
        self.diffusion_dists = (
            [
                dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
                for dist in diffusion_dists
            ]
            if diffusion_dists is not None
            else [DistributionFamily.GAUSSIAN] * n_latent
        )
        self.manifest_links: list[LinkFunction | str | None] | None = (
            list(manifest_links) if manifest_links is not None else None
        )
        self.ess_threshold = ess_threshold
        self.observation_support = observation_support

        self._block_rb = block_rb
        self.observation_operator = compile_observation_operator(observation_support)

        self.transition_semantics = compile_transition_semantics(self.diffusion_dists, n_latent)
        self.transition_dispatch_mode = self.transition_semantics.dispatch_mode

        # Pre-compute partition indices for mixed mode (static, not traced)
        if self.transition_semantics.is_mixed:
            if self._block_rb and self.transition_semantics.sampled_block_dist is None:
                raise ValueError(
                    "Mixed block-RB requires all sampled diffusion coordinates to share one family."
                )
            self._g_idx = jnp.asarray(self.transition_semantics.gaussian_idx, dtype=jnp.int64)
            self._s_idx = jnp.asarray(self.transition_semantics.sampled_idx, dtype=jnp.int64)
            self._sampled_block_dist = (
                self.transition_semantics.sampled_block_dist
                if self.transition_semantics.sampled_block_dist is not None
                else DistributionFamily.STUDENT_T
            )

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> jnp.ndarray:
        """Compute log-likelihood via bootstrap particle filter.

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals BEFORE each observation
            obs_mask: (T, n_manifest) boolean mask for observed values
            extra_params: Noise family hyperparameters (obs_df, obs_shape, proc_df)

        Returns:
            (T,) cumulative log-normalizing constants from the particle filter.
        """
        from cuthbert.filtering import filter as cuthbert_filter
        from cuthbert.smc.particle_filter import build_filter

        n = self.n_latent

        observations = jnp.asarray(observations, dtype=jnp.float64)
        time_intervals = jnp.asarray(time_intervals, dtype=jnp.float64)
        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)

        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        if self.observation_operator.requires_interval_summary_handling:
            return self._compute_support_aware_log_likelihood(
                ct_params,
                measurement_params,
                initial_state,
                clean_obs,
                time_intervals,
                obs_mask,
                extra_params,
            )

        # --- Pre-discretize CT→DT for all T timesteps (once, not per particle) ---
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))

        # Pre-compute Cholesky of Qd for all timesteps
        jitter = jnp.eye(n) * 1e-6
        chol_Qd = jax.vmap(lambda Q: jla.cholesky(Q + jitter, lower=True))(Qd)

        # Build params dict for observation model + initial state
        params = {
            "lambda_mat": measurement_params.lambda_mat,
            "manifest_means": measurement_params.manifest_means,
            "manifest_cov": measurement_params.manifest_cov,
            "t0_mean": initial_state.mean,
            "t0_cov": initial_state.cov,
        }
        if extra_params:
            params.update(extra_params)

        # --- Build measurement/transition semantics once ---
        obs_extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        measurement_semantics = compile_measurement_semantics(
            self.manifest_dists,
            manifest_cov=measurement_params.manifest_cov,
            extra_params=obs_extra,
            manifest_links=self.manifest_links,
            observation_support=self.observation_support,
        )
        obs_kernel = measurement_semantics.obs_kernel

        # Build Feynman-Kac model closures.
        if self.transition_semantics.is_gaussian and self._block_rb:
            from causal_ssm_agent.models.ssm.inference.targets.rao_blackwell import (
                make_rb_callbacks,
            )

            init_sample, propagate_sample, log_potential = make_rb_callbacks(
                params=params,
                m0=initial_state.mean,
                P0=initial_state.cov,
                obs_kernel=obs_kernel,
            )
        elif self.transition_semantics.is_mixed and self._block_rb:
            from causal_ssm_agent.models.ssm.inference.targets.block_rb import (
                make_block_rb_callbacks,
            )

            trans_extra = {k: v for k, v in params.items() if k.startswith("proc_")}
            trans_kernel = build_transition_kernel([self._sampled_block_dist], trans_extra)

            init_sample, propagate_sample, log_potential = make_block_rb_callbacks(
                n_latent=n,
                params=params,
                m0=initial_state.mean,
                P0=initial_state.cov,
                g_idx=self._g_idx,
                s_idx=self._s_idx,
                obs_kernel=obs_kernel,
                trans_kernel=trans_kernel,
            )
        else:
            trans_extra = {k: v for k, v in params.items() if k.startswith("proc_")}
            trans_kernel = build_transition_kernel(self.transition_semantics, trans_extra)

            H = measurement_params.lambda_mat
            d_meas = measurement_params.manifest_means
            R = measurement_params.manifest_cov

            def init_sample(key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree:  # noqa: ARG001
                chol = jla.cholesky(initial_state.cov + jnp.eye(self.n_latent) * 1e-6, lower=True)
                return initial_state.mean + chol @ random.normal(key, (self.n_latent,))

            def propagate_sample(
                key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
            ) -> ArrayTree:
                Ad_t = model_inputs["Ad"]
                cd_t = model_inputs["cd"]
                chol_Qd_t = model_inputs["chol_Qd"]
                mean = Ad_t @ state + cd_t
                return mean + trans_kernel.sample_noise_fn(key, chol_Qd_t)

            def log_potential(
                state_prev: ArrayTreeLike,  # noqa: ARG001 (required by filter protocol)
                state: ArrayTreeLike,
                model_inputs: ArrayTreeLike,
            ) -> ScalarArray:
                obs = model_inputs["observation"]
                mask = model_inputs["obs_mask"]
                mask_float = mask.astype(jnp.float64)
                return obs_kernel.emission_fn(obs, state, H, d_meas, R, mask_float)

        # Build model_inputs with leading temporal dimension T.
        model_inputs = {
            "observation": clean_obs,
            "obs_mask": obs_mask.astype(jnp.float64),
            "Ad": Ad,
            "cd": cd,
            "Qd": Qd,
            "chol_Qd": chol_Qd,
        }

        # Build and run filter
        filter_obj = build_filter(
            init_sample=cast("InitSample", init_sample),
            propagate_sample=cast("PropagateSample", propagate_sample),
            log_potential=cast("LogPotential", log_potential),
            n_filter_particles=self.n_particles,
            resampling_fn=_systematic_resampling,
            ess_threshold=self.ess_threshold,
        )

        states = cuthbert_filter(filter_obj, model_inputs, key=self.rng_key)

        return states.log_normalizing_constant

    def _compute_support_aware_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray,
        extra_params: dict | None,
    ) -> jnp.ndarray:
        """Compute PF likelihood for interval-summary observation semantics."""
        from cuthbert.filtering import filter as cuthbert_filter
        from cuthbert.smc.particle_filter import build_filter

        n = self.n_latent

        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))

        jitter = jnp.eye(n) * 1e-6
        chol_Qd = jax.vmap(lambda Q: jla.cholesky(Q + jitter, lower=True))(Qd)

        params = {
            "lambda_mat": measurement_params.lambda_mat,
            "manifest_means": measurement_params.manifest_means,
            "manifest_cov": measurement_params.manifest_cov,
            "t0_mean": initial_state.mean,
            "t0_cov": initial_state.cov,
        }
        if extra_params:
            params.update(extra_params)

        obs_extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        measurement_semantics = compile_measurement_semantics(
            self.manifest_dists,
            manifest_cov=measurement_params.manifest_cov,
            extra_params=obs_extra,
            manifest_links=self.manifest_links,
            observation_support=self.observation_support,
        )
        obs_kernel = measurement_semantics.obs_kernel
        if measurement_semantics.mean_log_prob_fn is None:
            raise NotImplementedError(
                "Interval-summary observations are not supported for the current measurement setup."
            )
        mean_log_prob_fn = measurement_semantics.mean_log_prob_fn
        observation_operator = measurement_semantics.observation_operator

        trans_extra = {k: v for k, v in params.items() if k.startswith("proc_")}
        trans_kernel = build_transition_kernel(self.transition_semantics, trans_extra)

        H = measurement_params.lambda_mat
        d_meas = measurement_params.manifest_means
        R = measurement_params.manifest_cov
        assert observation_operator.requires_interval_summary_handling
        assert observation_operator.prev_coeffs is not None
        assert observation_operator.curr_coeffs is not None
        assert observation_operator.interval_weights is not None
        assert observation_operator.emission_slots is not None

        def init_sample(key: KeyArray, _model_inputs: ArrayTreeLike) -> ArrayTree:
            chol = jla.cholesky(initial_state.cov + jnp.eye(self.n_latent) * 1e-6, lower=True)
            latent = initial_state.mean + chol @ random.normal(key, (self.n_latent,))
            response = obs_kernel.response_fn(H @ latent + d_meas)
            zeros = observation_operator.empty_accumulators(response.dtype)
            return SupportAwareParticleState(
                latent,
                response,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            )

        def propagate_sample(
            key: KeyArray,
            state: ArrayTreeLike,
            model_inputs: ArrayTreeLike,
        ) -> ArrayTree:
            Ad_t = model_inputs["Ad"]
            cd_t = model_inputs["cd"]
            chol_Qd_t = model_inputs["chol_Qd"]
            mean = Ad_t @ state.latent + cd_t
            latent_new = mean + trans_kernel.sample_noise_fn(key, chol_Qd_t)
            response_new = obs_kernel.response_fn(H @ latent_new + d_meas)

            prev_coeff = model_inputs["support_prev_coeff"]
            curr_coeff = model_inputs["support_curr_coeff"]
            interval_weight = model_inputs["support_weight"]
            emission_slots = model_inputs["support_emission_slot"]
            obs_mask_float = model_inputs["obs_mask"].astype(response_new.dtype)

            step_result = advance_support_observation_state(
                observation_operator,
                state.response,
                state.accum_sum,
                state.accum_sumsq,
                state.accum_weight,
                response_new,
                obs_mask_float,
                prev_coeff,
                curr_coeff,
                interval_weight,
                emission_slots,
            )

            return SupportAwareParticleState(
                latent_new,
                response_new,
                step_result.next_accum_sum,
                step_result.next_accum_sumsq,
                step_result.next_accum_weight,
                step_result.obs_sum,
                step_result.obs_sumsq,
                step_result.obs_weight,
            )

        def log_potential(
            state_prev: ArrayTreeLike,  # noqa: ARG001
            state: ArrayTreeLike,
            model_inputs: ArrayTreeLike,
        ) -> ScalarArray:
            obs = model_inputs["observation"]
            mask_float = model_inputs["obs_mask"].astype(jnp.float64)
            emission_slots = model_inputs["support_emission_slot"]
            summary = summarize_support_observation(
                observation_operator,
                state.response,
                state.obs_sum,
                state.obs_sumsq,
                state.obs_weight,
                mask_float,
                emission_slots,
            )
            return support_observation_log_prob(
                observation_operator,
                obs_kernel,
                mean_log_prob_fn,
                obs,
                mask_float,
                state.latent,
                H,
                d_meas,
                R,
                summary,
            )

        model_inputs = {
            "observation": observations,
            "obs_mask": obs_mask.astype(jnp.float64),
            "Ad": Ad,
            "cd": cd,
            "Qd": Qd,
            "chol_Qd": chol_Qd,
            "support_prev_coeff": jnp.asarray(observation_operator.prev_coeffs, dtype=jnp.float64),
            "support_curr_coeff": jnp.asarray(observation_operator.curr_coeffs, dtype=jnp.float64),
            "support_weight": jnp.asarray(observation_operator.interval_weights, dtype=jnp.float64),
            "support_emission_slot": jnp.asarray(
                observation_operator.emission_slots, dtype=jnp.int64
            ),
        }

        filter_obj = build_filter(
            init_sample=cast("InitSample", init_sample),
            propagate_sample=cast("PropagateSample", propagate_sample),
            log_potential=cast("LogPotential", log_potential),
            n_filter_particles=self.n_particles,
            resampling_fn=_systematic_resampling,
            ess_threshold=self.ess_threshold,
        )

        states = cuthbert_filter(filter_obj, model_inputs, key=self.rng_key)
        return states.log_normalizing_constant
