"""Base protocol and parameter types for likelihood computation.

Defines the interface that likelihood backends must implement:
compute_log_likelihood(params, observations, times) -> jnp.ndarray

Returns the (T,) cumulative log-normalizing-constant array from the filter.
The total log-likelihood is lnc[-1]; per-timestep one-step-ahead predictive
log-likelihoods are jnp.diff(lnc, prepend=0.0).

Used by ParticleLikelihood to integrate out latent states via bootstrap PF
and inject the result into NumPyro models via numpyro.factor().
"""

from typing import NamedTuple, Protocol

import jax.numpy as jnp

MISSING_DATA_LARGE_VAR = 1e10
CHOL_JITTER = 1e-8
NUMERICAL_EPSILON = 1e-10
PROB_CLIP_MIN = 1e-7
ETA_CLIP_MIN = 1e-6

LIKELIHOOD_SOLVER_KIND_KALMAN_EXACT = 0
LIKELIHOOD_SOLVER_KIND_POINT_IEKS = 1
LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS = 2
LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT = 3


class CTParams(NamedTuple):
    """Continuous-time state-space parameters.

    Represents the continuous-time SDE:
        dη = (A*η + c) dt + G dW

    where:
        A = drift matrix (n_latent x n_latent)
        G*G' = diffusion covariance (n_latent x n_latent)
        c = continuous intercept (n_latent,)

    Note: diffusion_cov stores the covariance G*G', not the Cholesky factor G.
    """

    drift: jnp.ndarray  # (n_latent, n_latent)
    diffusion_cov: jnp.ndarray  # (n_latent, n_latent) - G @ G.T
    cint: jnp.ndarray | None  # (n_latent,) or None


class MeasurementParams(NamedTuple):
    """Measurement model parameters.

    Represents the observation equation:
        y = Λ*η + μ + ε, ε ~ N(0, Σ_R)

    where:
        Λ = factor loadings (n_manifest x n_latent)
        μ = manifest intercepts (n_manifest,)
        Σ_R = measurement error covariance (n_manifest x n_manifest)

    Note: the higher-level SSMSpec stores ``manifest_var = L_R`` as a
    Cholesky factor. ``MeasurementParams.manifest_cov`` stores the derived
    covariance ``Σ_R = L_R L_Rᵀ`` used by likelihood backends.
    """

    lambda_mat: jnp.ndarray  # (n_manifest, n_latent)
    manifest_means: jnp.ndarray  # (n_manifest,)
    manifest_cov: jnp.ndarray  # (n_manifest, n_manifest) = Σ_R


class InitialStateParams(NamedTuple):
    """Initial state distribution parameters.

    η_0 ~ N(m_0, P_0)
    """

    mean: jnp.ndarray  # (n_latent,)
    cov: jnp.ndarray  # (n_latent, n_latent)


class LikelihoodBackend(Protocol):
    """Protocol for state-space likelihood computation backends.

    Each backend must implement compute_log_likelihood() which integrates
    out latent states and returns cumulative log-normalizing constants.

    The returned (T,) array lnc satisfies:
    - lnc[-1] = total log p(y|θ), used in numpyro.factor()
    - diff(lnc) = per-timestep one-step-ahead predictive log p(y_t|y_{1:t-1},θ),
      used for LOO-CV via the innovation decomposition

    Implementation:
    - ParticleLikelihood: Universal backend via differentiable bootstrap PF (cuthbert SMC)
    """

    checkpoint_loglik: bool

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute log-likelihood by marginalizing out latent states.

        Args:
            ct_params: Continuous-time dynamics parameters (drift, diffusion, cint)
            measurement_params: Observation model parameters (Λ, μ, R)
            initial_state: Initial state distribution (m_0, P_0)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            (T,) cumulative log-normalizing constants from the filter.
            lnc[-1] is the total log-likelihood p(y|θ).
        """
        ...


def build_likelihood_eval_aux(
    dtype,
    *,
    solver_kind: int,
    **overrides,
) -> dict[str, jnp.ndarray]:
    """Build a fixed-shape backend diagnostic payload for host-side progress logs."""
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    aux = {
        "solver_kind": jnp.asarray(solver_kind, dtype=jnp.int32),
        "n_iterations": jnp.asarray(0, dtype=jnp.int32),
        "n_accepted_steps": jnp.asarray(0, dtype=jnp.int32),
        "init_log_joint": nan,
        "final_log_joint": nan,
        "final_rel_change": nan,
        "final_damping": nan,
        "final_step_alpha": nan,
        "final_step_norm": nan,
        "laplace_logdet": nan,
        "min_chol_diag": nan,
    }
    for key, value in overrides.items():
        if key not in aux:
            raise KeyError(f"Unknown likelihood-eval aux field: {key}")
        aux[key] = jnp.asarray(value, dtype=aux[key].dtype)
    return aux


def preprocess_missing_data(
    observations: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    obs_mask: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Preprocess observations and measurement covariance for missing data.

    Centralizes the large-variance injection pattern used across all backends.
    Missing observations are replaced with 0 and their corresponding measurement
    variance is inflated so the filter effectively ignores them.

    Args:
        observations: (T, n_manifest) raw observations (may contain NaN)
        manifest_cov: (n_manifest, n_manifest) measurement covariance R
        obs_mask: (T, n_manifest) boolean mask (True = observed), or None

    Returns:
        clean_obs: (T, n_manifest) observations with NaN replaced by 0
        R_adjusted: (T, n_manifest, n_manifest) per-timestep adjusted R
        obs_mask: (T, n_manifest) boolean mask
    """
    if obs_mask is None:
        obs_mask = ~jnp.isnan(observations)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)

    # Build per-timestep R with inflated variance for missing entries
    T, n_manifest = observations.shape
    mask_float = obs_mask.astype(jnp.float64)  # (T, n_manifest)
    # (T, n_manifest) diagonal inflation values
    inflation = (1.0 - mask_float) * MISSING_DATA_LARGE_VAR
    # Broadcast R to (T, n_manifest, n_manifest) and add diagonal inflation
    R_base = jnp.broadcast_to(manifest_cov, (T, n_manifest, n_manifest))
    R_adjusted = R_base + jnp.eye(n_manifest) * inflation[:, :, None]

    return clean_obs, R_adjusted, obs_mask
