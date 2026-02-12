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

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.emissions import get_emission_fn
from dsem_agent.models.ssm.discretization import discretize_system, discretize_system_batched

# =============================================================================
# JAX-native systematic resampling (gradient-safe on all platforms)
# =============================================================================
#
# cuthbert's built-in systematic resampling uses jax.pure_callback + numba on
# CPU, which does not support JVP.  We replace it with a pure-JAX version that
# uses jnp.searchsorted so that jax.grad can trace through the full PF
# (resampling indices are integers → zero gradient, which is correct).


def _systematic_resampling(key: jax.Array, logits: jnp.ndarray, n: int) -> jnp.ndarray:  # noqa: ARG001
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
    N = logits.shape[0]  # use static shape, not the traced `n` arg
    weights = jnp.exp(logits - jax.nn.logsumexp(logits))
    cumsum = jnp.cumsum(weights)
    us = (random.uniform(key, ()) + jnp.arange(N)) / N
    idx = jnp.searchsorted(cumsum, us)
    return jnp.clip(idx, 0, N - 1)


# =============================================================================
# SSMAdapter — maps CT-SEM parameters to PF-compatible functions
# =============================================================================


class SSMAdapter:
    """Adapts CT-SEM parameters into particle filter-compatible functions.

    Maps the CT-SEM structure (drift, diffusion, measurement) into
    initial_sample, transition_sample, and observation_log_prob.

    Supports non-Gaussian observation and process noise families:
        - manifest_dist: "gaussian", "poisson", "student_t", "gamma"
        - diffusion_dist: "gaussian", "student_t"
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: str = "gaussian",
        diffusion_dist: str = "gaussian",
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.diffusion_dist = diffusion_dist

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
            chi2_sample = random.gamma(key_chi2, df / 2.0) * 2.0
            scale = jnp.sqrt((df - 2.0) / chi2_sample)
            return mean + chol @ (z * scale)
        else:
            return mean + chol @ random.normal(key, (self.n_latent,))

    def observation_log_prob(
        self, y: jax.Array, x: jax.Array, params: dict, obs_mask: jax.Array
    ) -> float:
        """Compute log p(y | x) under measurement model.

        Delegates to canonical emission functions in emissions.py.
        """
        H = params["lambda_mat"]
        d = params["manifest_means"]
        R = params["manifest_cov"]
        mask_float = obs_mask.astype(jnp.float32)
        extra = {k: v for k, v in params.items() if k.startswith("obs_")}
        emission_fn = get_emission_fn(self.manifest_dist, extra)
        return emission_fn(y, x, H, d, R, mask_float)


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
        manifest_dist: Observation noise family
        diffusion_dist: Process noise family
        ess_threshold: ESS/N threshold for resampling
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        n_particles: int = 200,
        rng_key: jax.Array | None = None,
        manifest_dist: str = "gaussian",
        diffusion_dist: str | list[str] = "gaussian",
        ess_threshold: float = 0.5,
        block_rb: bool = True,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.n_particles = n_particles
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.manifest_dist = manifest_dist
        self.ess_threshold = ess_threshold

        self._block_rb = block_rb

        # Normalize diffusion_dist to per-variable list
        if isinstance(diffusion_dist, str):
            self.diffusion_dist = diffusion_dist
            self._per_var_diffusion = [diffusion_dist] * n_latent
        else:
            self._per_var_diffusion = list(diffusion_dist)
            # Determine dispatch mode
            unique = set(self._per_var_diffusion)
            if unique == {"gaussian"}:
                self.diffusion_dist = "gaussian"
            elif "gaussian" not in unique:
                # All non-Gaussian — pick the first non-Gaussian type
                self.diffusion_dist = self._per_var_diffusion[0]
            else:
                self.diffusion_dist = "mixed"

        # When block_rb is disabled and mixed, collapse to first non-Gaussian
        # type to skip block RBPF dispatch. Gaussian stays Gaussian — the
        # dispatch in compute_log_likelihood checks _block_rb to skip RBPF.
        if not block_rb and self.diffusion_dist == "mixed":
            s_types = [d for d in self._per_var_diffusion if d != "gaussian"]
            self.diffusion_dist = s_types[0] if s_types else "student_t"

        # Pre-compute partition indices for mixed mode (static, not traced)
        if self.diffusion_dist == "mixed":
            from dsem_agent.models.likelihoods.block_rb import partition_indices

            self._g_idx, self._s_idx = partition_indices(self._per_var_diffusion)
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
    ) -> float:
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
            Log-likelihood p(y|theta) as a scalar
        """
        from cuthbert.filtering import filter as cuthbert_filter
        from cuthbert.smc.particle_filter import build_filter

        n = self.n_latent

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)

        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        # --- Pre-discretize CT→DT for all T timesteps (once, not per particle) ---
        # This is the key optimization: matrix exponential + Lyapunov solve is
        # O(n³) per timestep and identical across particles. Pre-computing avoids
        # redundant work inside propagate_sample.
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))

        # Pre-compute Cholesky of Qd for all timesteps (once, not per particle).
        # cuthbert vmaps propagate_sample over N_PF particles with model_inputs
        # broadcast — without this, each particle redundantly recomputes Cholesky.
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

        # Build Feynman-Kac model closures.
        # When dynamics are Gaussian and block_rb is enabled, use
        # Rao-Blackwellized callbacks (Kalman filter inside each particle).
        # Mixed + block_rb: block RBPF marginalizes Gaussian subset, samples rest.
        # Otherwise fall back to bootstrap PF callbacks.
        if self.diffusion_dist == "gaussian" and self._block_rb:
            from dsem_agent.models.likelihoods.rao_blackwell import make_rb_callbacks

            init_sample, propagate_sample, log_potential = make_rb_callbacks(
                n_latent=n,
                n_manifest=self.n_manifest,
                manifest_dist=self.manifest_dist,
                params=params,
                extra_params=extra_params or {},
                m0=initial_state.mean,
                P0=initial_state.cov,
            )
        elif self.diffusion_dist == "mixed":
            from dsem_agent.models.likelihoods.block_rb import make_block_rb_callbacks

            block_extra = dict(extra_params or {})
            block_extra["diffusion_dist_s"] = self._diffusion_dist_s

            init_sample, propagate_sample, log_potential = make_block_rb_callbacks(
                n_latent=n,
                n_manifest=self.n_manifest,
                manifest_dist=self.manifest_dist,
                params=params,
                extra_params=block_extra,
                m0=initial_state.mean,
                P0=initial_state.cov,
                g_idx=self._g_idx,
                s_idx=self._s_idx,
            )
        else:
            adapter = SSMAdapter(
                self.n_latent,
                self.n_manifest,
                self.manifest_dist,
                self.diffusion_dist,
            )

            def init_sample(key, _model_inputs):
                return adapter.initial_sample(key, params)

            def propagate_sample(key, state, model_inputs):
                Ad_t = model_inputs["Ad"]
                cd_t = model_inputs["cd"]
                chol_Qd_t = model_inputs["chol_Qd"]
                mean = Ad_t @ state + cd_t

                df = params.get("proc_df", 5.0)
                df = jnp.maximum(df, 2.1)
                key_z, key_chi2 = random.split(key)
                z = random.normal(key_z, (n,))
                chi2_sample = random.gamma(key_chi2, df / 2.0) * 2.0
                scale = jnp.sqrt((df - 2.0) / chi2_sample)
                return mean + chol_Qd_t @ (z * scale)

            def log_potential(_state_prev, state, model_inputs):
                obs = model_inputs["observation"]
                mask = model_inputs["obs_mask"]
                return adapter.observation_log_prob(obs, state, params, mask)

        # Build model_inputs with leading temporal dimension T.
        # cuthbert convention: model_inputs[0] → init_prepare (sample particles
        # and weight against obs[0]); model_inputs[k] for k=1..T-1 → propagate
        # with dt[k] and weight against obs[k].
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

        return states.log_normalizing_constant[-1]
