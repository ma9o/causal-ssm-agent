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

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.likelihoods.kernels import (
    build_observation_kernel,
    build_transition_kernel,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system, discretize_system_batched

if TYPE_CHECKING:
    from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray
    from jax import Array
    from jax.typing import ArrayLike

    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

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
    return jnp.clip(idx, 0, N - 1)


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
        manifest_dist: DistributionFamily,
        diffusion_dist: DistributionFamily,
        manifest_link: LinkFunction,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.diffusion_dist = diffusion_dist
        self.manifest_link = manifest_link

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

        if self.diffusion_dist == "student_t":
            df = params.get("proc_df", 5.0)
            df = jnp.maximum(df, 2.1)
            key_z, key_chi2 = random.split(key)
            z = random.normal(key_z, (self.n_latent,))
            chi2_sample = jnp.maximum(random.gamma(key_chi2, df / 2.0) * 2.0, 1e-8)
            scale = jnp.sqrt((df - 2.0) / chi2_sample)
            return mean + chol @ (z * scale)
        return mean + chol @ random.normal(key, (self.n_latent,))

    def observation_log_prob(
        self, y: jax.Array, x: jax.Array, params: dict, obs_mask: jax.Array
    ) -> float:
        """Compute log p(y | x) under measurement model.

        Builds an ObservationKernel on-the-fly for the bootstrap PF path.
        """
        H = params["lambda_mat"]
        d = params["manifest_means"]
        R = params["manifest_cov"]
        mask_float = obs_mask.astype(jnp.float32)
        extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        obs_kernel = build_observation_kernel(
            self.manifest_dist, self.manifest_link, extra, manifest_cov=R
        )
        return obs_kernel.emission_fn(y, x, H, d, R, mask_float)


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
        manifest_dist: Observation noise family (DistributionFamily enum)
        diffusion_dist: Process noise family (DistributionFamily enum or list)
        ess_threshold: ESS/N threshold for resampling
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        n_particles: int = 200,
        rng_key: jax.Array | None = None,
        manifest_dist: DistributionFamily | str = "gaussian",
        diffusion_dist: DistributionFamily | str | list = "gaussian",
        ess_threshold: float = 0.5,
        block_rb: bool = True,
        manifest_link: LinkFunction | str = "identity",
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.n_particles = n_particles
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.manifest_dist = manifest_dist
        self.manifest_link = manifest_link
        self.ess_threshold = ess_threshold

        self._block_rb = block_rb

        # Normalize diffusion_dist to per-variable list
        if isinstance(diffusion_dist, (str, type(manifest_dist))):
            # Scalar — broadcast to all latents
            self.diffusion_dist = diffusion_dist
            self._per_var_diffusion = [diffusion_dist] * n_latent
        else:
            self._per_var_diffusion = list(diffusion_dist)
            # Determine dispatch mode
            unique = {str(d) for d in self._per_var_diffusion}
            if unique == {"gaussian"}:
                self.diffusion_dist = "gaussian"
            elif "gaussian" not in unique:
                self.diffusion_dist = self._per_var_diffusion[0]
            else:
                self.diffusion_dist = "mixed"

        # When block_rb is disabled and mixed, collapse to first non-Gaussian
        if not block_rb and self.diffusion_dist == "mixed":
            s_types = [d for d in self._per_var_diffusion if str(d) != "gaussian"]
            self.diffusion_dist = s_types[0] if s_types else "student_t"

        # Pre-compute partition indices for mixed mode (static, not traced)
        if self.diffusion_dist == "mixed":
            from causal_ssm_agent.models.likelihoods.block_rb import partition_indices

            self._g_idx, self._s_idx = partition_indices([str(d) for d in self._per_var_diffusion])
            s_types = [self._per_var_diffusion[int(i)] for i in self._s_idx]
            self._diffusion_dist_s = s_types[0] if s_types else "student_t"

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

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)

        clean_obs = jnp.nan_to_num(observations, nan=0.0)

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

        # --- Build kernels once ---
        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

        dist = DistributionFamily(self.manifest_dist)
        link = LinkFunction(self.manifest_link)
        obs_extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        obs_kernel = build_observation_kernel(
            dist, link, obs_extra, manifest_cov=measurement_params.manifest_cov
        )

        # Build Feynman-Kac model closures.
        if self.diffusion_dist == "gaussian" and self._block_rb:
            from causal_ssm_agent.models.likelihoods.rao_blackwell import make_rb_callbacks

            init_sample, propagate_sample, log_potential = make_rb_callbacks(
                params=params,
                m0=initial_state.mean,
                P0=initial_state.cov,
                obs_kernel=obs_kernel,
            )
        elif self.diffusion_dist == "mixed":
            from causal_ssm_agent.models.likelihoods.block_rb import make_block_rb_callbacks

            trans_extra = {k: v for k, v in params.items() if k.startswith("proc_")}
            trans_kernel = build_transition_kernel(
                DistributionFamily(self._diffusion_dist_s), trans_extra
            )

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
            adapter = SSMAdapter(
                self.n_latent,
                self.n_manifest,
                dist,
                DistributionFamily(self.diffusion_dist),
                manifest_link=link,
            )

            # Build transition kernel for bootstrap PF (non-Gaussian dynamics)
            trans_extra = {k: v for k, v in params.items() if k.startswith("proc_")}
            trans_kernel = build_transition_kernel(
                DistributionFamily(self.diffusion_dist), trans_extra
            )

            H = measurement_params.lambda_mat
            d_meas = measurement_params.manifest_means
            R = measurement_params.manifest_cov

            def init_sample(key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree:  # noqa: ARG001
                return adapter.initial_sample(key, params)

            def propagate_sample(
                key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
            ) -> ArrayTree:
                Ad_t = model_inputs["Ad"]
                cd_t = model_inputs["cd"]
                chol_Qd_t = model_inputs["chol_Qd"]
                mean = Ad_t @ state + cd_t
                return mean + trans_kernel.sample_noise_fn(key, chol_Qd_t)

            def log_potential(
                state_prev: ArrayTreeLike, state: ArrayTreeLike, model_inputs: ArrayTreeLike  # noqa: ARG001
            ) -> ScalarArray:
                obs = model_inputs["observation"]
                mask = model_inputs["obs_mask"]
                mask_float = mask.astype(jnp.float32)
                return obs_kernel.emission_fn(obs, state, H, d_meas, R, mask_float)

        # Build model_inputs with leading temporal dimension T.
        model_inputs = {
            "observation": clean_obs,
            "obs_mask": obs_mask.astype(jnp.float32),
            "Ad": Ad,
            "cd": cd,
            "Qd": Qd,
            "chol_Qd": chol_Qd,
        }

        # Build and run filter
        filter_obj = build_filter(
            init_sample=init_sample,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            n_filter_particles=self.n_particles,
            resampling_fn=_systematic_resampling,
            ess_threshold=self.ess_threshold,
        )

        states = cuthbert_filter(filter_obj, model_inputs, key=self.rng_key)

        return states.log_normalizing_constant
