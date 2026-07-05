"""Base protocol and parameter types for likelihood computation.

Defines the interface that likelihood backends must implement:
compute_log_likelihood(params, observations, times) -> jnp.ndarray

Returns the (T,) cumulative log-normalizing-constant array from the filter.
The total log-likelihood is lnc[-1]; per-timestep one-step-ahead predictive
log-likelihoods are jnp.diff(lnc, prepend=0.0).

Used by Laplace likelihood backends to inject marginalized state likelihoods
into NumPyro models via numpyro.factor().
"""

from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.shapes import Array, Float, Shaped

MISSING_DATA_LARGE_VAR = 1e10
CHOL_JITTER = 1e-8
NUMERICAL_EPSILON = 1e-10
PROB_CLIP_MIN = 1e-7

LIKELIHOOD_SOLVER_KIND_POINT_IEKS = 1
LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS = 2
LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT = 3


class RuntimeDynamics(NamedTuple):
    """Continuous-time dynamics expressed as a vector field plus parameters.

    This is the model-facing drift representation. Inference backends may derive
    specialized internal parameterizations from it, but ``SSMModel`` always
    hands off dynamics through this vector-field surface.
    """

    vector_field: Any
    vf_params: tuple[dict[str, Array], ...]
    diffusion_cov: Float[Array, "D D"]
    input_effect: Float[Array, "D I"] | None = None


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

    lambda_mat: Float[Array, "M D"]
    manifest_means: Float[Array, " M"]
    manifest_cov: Float[Array, "M M"]  # Σ_R


class InitialStateParams(NamedTuple):
    """Initial state distribution parameters.

    η_0 ~ N(m_0, P_0)
    """

    mean: Float[Array, " D"]
    cov: Float[Array, "D D"]


class TrajectoryTarget(Protocol):
    """Latent path prior contract exposed to inference runtimes."""

    kind: str
    supports_affine_prefix_marginals: bool

    def initial_moments(self, context) -> tuple[jnp.ndarray, jnp.ndarray]: ...

    def initial_log_prob(self, context, particle0: jnp.ndarray) -> jnp.ndarray: ...

    def predictive_latent_init(self, context) -> jnp.ndarray: ...

    def sample_transition(
        self,
        key: jnp.ndarray,
        context,
        previous_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray: ...

    def transition_log_prob(
        self,
        context,
        previous_state: jnp.ndarray,
        current_state: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray: ...

    def transition_log_probs_for_pairs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray: ...

    def pairwise_transition_log_probs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray: ...

    def trajectory_prior_log_prob(
        self,
        context,
        latent_trajectory: jnp.ndarray,
        prior_terms: Any | None = None,
    ) -> jnp.ndarray: ...


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
    observations: Float[Array, "T M"],
    manifest_cov: Float[Array, "M M"],
    obs_mask: Shaped[Array, "T M"] | None,
) -> tuple[Float[Array, "T M"], Float[Array, "T M M"], Shaped[Array, "T M"]]:
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
    mask_float = obs_mask.astype(jnp.float32)  # (T, n_manifest)
    # (T, n_manifest) diagonal inflation values
    inflation = (1.0 - mask_float) * MISSING_DATA_LARGE_VAR
    # Broadcast R to (T, n_manifest, n_manifest) and add diagonal inflation
    R_base = jnp.broadcast_to(manifest_cov, (T, n_manifest, n_manifest))
    R_adjusted = R_base + jnp.eye(n_manifest) * inflation[:, :, None]

    return clean_obs, R_adjusted, obs_mask
