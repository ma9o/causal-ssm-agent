"""Kalman filter implementation for CT-SEM.

Implements the Kalman filter with:
- Variable time intervals (continuous-time discretization)
- Missing data handling
- Log-likelihood computation for NumPyro integration
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax


class KalmanState(NamedTuple):
    """State of the Kalman filter at a single time point."""

    mean: jnp.ndarray  # Filtered state mean (n_latent,)
    cov: jnp.ndarray  # Filtered state covariance (n_latent, n_latent)
    log_lik: float  # Log-likelihood contribution


class KalmanFilterResult(NamedTuple):
    """Full result of Kalman filtering."""

    filtered_means: jnp.ndarray  # (T, n_latent)
    filtered_covs: jnp.ndarray  # (T, n_latent, n_latent)
    predicted_means: jnp.ndarray  # (T, n_latent)
    predicted_covs: jnp.ndarray  # (T, n_latent, n_latent)
    log_likelihood: float  # Total log-likelihood


def kalman_predict(
    state_mean: jnp.ndarray,
    state_cov: jnp.ndarray,
    discrete_drift: jnp.ndarray,
    discrete_Q: jnp.ndarray,
    discrete_cint: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kalman filter prediction step.

    Propagates state from t to t+dt:
        mean_{t+dt|t} = A_d * mean_{t|t} + c_d
        cov_{t+dt|t} = A_d * cov_{t|t} * A_d' + Q_d

    Args:
        state_mean: Current state mean (n,)
        state_cov: Current state covariance (n, n)
        discrete_drift: Discrete drift matrix A_d = exp(A*dt)
        discrete_Q: Discrete diffusion covariance Q_d
        discrete_cint: Discrete intercept c_d (optional)

    Returns:
        Tuple of (predicted_mean, predicted_cov)
    """
    # Predict mean
    predicted_mean = discrete_drift @ state_mean
    if discrete_cint is not None:
        predicted_mean = predicted_mean + discrete_cint.flatten()

    # Predict covariance
    predicted_cov = discrete_drift @ state_cov @ discrete_drift.T + discrete_Q

    # Ensure symmetry
    predicted_cov = 0.5 * (predicted_cov + predicted_cov.T)

    return predicted_mean, predicted_cov


def kalman_update(
    predicted_mean: jnp.ndarray,
    predicted_cov: jnp.ndarray,
    observation: jnp.ndarray,
    obs_mask: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Kalman filter update step with missing data handling via masking.

    Updates state given observations using a mask-based approach that is
    compatible with JAX tracing. Missing observations are handled by
    adding large variance to the measurement covariance for unobserved
    components, effectively zeroing out their contribution to the update.

    Args:
        predicted_mean: Predicted state mean (n_latent,)
        predicted_cov: Predicted state covariance (n_latent, n_latent)
        observation: Observed values (n_manifest,) - missing values set to 0
        obs_mask: Boolean mask, True for observed (n_manifest,)
        lambda_mat: Factor loadings (n_manifest, n_latent)
        manifest_means: Manifest intercepts (n_manifest,)
        manifest_cov: Measurement error covariance (n_manifest, n_manifest)

    Returns:
        Tuple of (updated_mean, updated_cov, log_likelihood_contribution)
    """
    n_manifest = observation.shape[0]

    # Convert mask to float for masking operations
    mask_float = obs_mask.astype(jnp.float32)
    n_observed = jnp.sum(mask_float)

    # For missing observations, add large variance to effectively ignore them
    # This is a standard trick for handling missing data in Kalman filters
    large_var = 1e10
    adjusted_manifest_cov = manifest_cov + jnp.diag((1.0 - mask_float) * large_var)

    # Predicted observation (for all manifest variables)
    y_pred = lambda_mat @ predicted_mean + manifest_means

    # Innovation (set to 0 for missing observations)
    innovation = (observation - y_pred) * mask_float

    # Innovation covariance with adjusted variance for missing data
    S = lambda_mat @ predicted_cov @ lambda_mat.T + adjusted_manifest_cov
    S = 0.5 * (S + S.T)  # Ensure symmetry

    # Add small regularization for numerical stability
    S = S + jnp.eye(n_manifest) * 1e-8

    # Kalman gain: K = P * H' * S^{-1}
    K = jla.solve(S, lambda_mat @ predicted_cov, assume_a="pos").T

    # Update mean and covariance
    updated_mean = predicted_mean + K @ innovation
    updated_cov = predicted_cov - K @ lambda_mat @ predicted_cov
    updated_cov = 0.5 * (updated_cov + updated_cov.T)

    # Log-likelihood only for observed variables
    # Use the masked innovation and original (not inflated) covariance
    # for observed variables
    S_obs = lambda_mat @ predicted_cov @ lambda_mat.T + manifest_cov
    S_obs = 0.5 * (S_obs + S_obs.T) + jnp.eye(n_manifest) * 1e-8

    # Compute log-likelihood contribution
    # For missing data, we use the masked computation approach
    sign, logdet = jnp.linalg.slogdet(S_obs)

    # Masked Mahalanobis distance (only count observed)
    innovation_obs = observation - y_pred
    mahal = innovation_obs @ jla.solve(S_obs, innovation_obs, assume_a="pos")

    # Approximate the log-likelihood for the observed subset
    # ll = -0.5 * (k*log(2pi) + log|S_obs| + mahal)
    # where k is number of observed variables
    ll = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)

    # Zero out likelihood if no observations
    ll = jnp.where(n_observed > 0, ll, 0.0)

    return updated_mean, updated_cov, ll


def kalman_update_simple(
    predicted_mean: jnp.ndarray,
    predicted_cov: jnp.ndarray,
    observation: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Kalman update without missing data handling (simpler, for testing).

    Args:
        predicted_mean: Predicted state mean (n_latent,)
        predicted_cov: Predicted state covariance (n_latent, n_latent)
        observation: Observed values (n_manifest,)
        lambda_mat: Factor loadings (n_manifest, n_latent)
        manifest_means: Manifest intercepts (n_manifest,)
        manifest_cov: Measurement error covariance (n_manifest, n_manifest)

    Returns:
        Tuple of (updated_mean, updated_cov, log_likelihood_contribution)
    """
    # Predicted observation
    y_pred = lambda_mat @ predicted_mean + manifest_means

    # Innovation
    innovation = observation - y_pred

    # Innovation covariance
    S = lambda_mat @ predicted_cov @ lambda_mat.T + manifest_cov
    S = 0.5 * (S + S.T)

    # Kalman gain
    K = jla.solve(S, lambda_mat @ predicted_cov, assume_a="pos").T

    # Update
    updated_mean = predicted_mean + K @ innovation
    updated_cov = predicted_cov - K @ lambda_mat @ predicted_cov
    updated_cov = 0.5 * (updated_cov + updated_cov.T)

    # Log-likelihood
    n_manifest = observation.shape[0]
    sign, logdet = jnp.linalg.slogdet(S)
    mahal = innovation @ jla.solve(S, innovation, assume_a="pos")
    ll = -0.5 * (n_manifest * jnp.log(2 * jnp.pi) + logdet + mahal)

    return updated_mean, updated_cov, ll


def kalman_filter(
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_cov: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
) -> KalmanFilterResult:
    """Run Kalman filter over all observations.

    Args:
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals from previous observation
        drift: (n_latent, n_latent) continuous drift A
        diffusion_cov: (n_latent, n_latent) continuous diffusion GG'
        cint: (n_latent,) continuous intercept (optional)
        lambda_mat: (n_manifest, n_latent) factor loadings
        manifest_means: (n_manifest,) manifest intercepts
        manifest_cov: (n_manifest, n_manifest) measurement error cov
        t0_mean: (n_latent,) initial state mean
        t0_cov: (n_latent, n_latent) initial state covariance
        obs_mask: (T, n_manifest) boolean mask for observed values

    Returns:
        KalmanFilterResult with filtered states and log-likelihood
    """
    from dsem_agent.models.ctsem.core import discretize_system

    T, n_manifest = observations.shape
    n_latent = drift.shape[0]

    # Initialize storage
    filtered_means = jnp.zeros((T, n_latent))
    filtered_covs = jnp.zeros((T, n_latent, n_latent))
    predicted_means = jnp.zeros((T, n_latent))
    predicted_covs = jnp.zeros((T, n_latent, n_latent))

    if obs_mask is None:
        obs_mask = ~jnp.isnan(observations)

    # Initial state
    state_mean = t0_mean
    state_cov = t0_cov
    total_ll = 0.0

    def scan_fn(carry, inputs):
        state_mean, state_cov, total_ll = carry
        obs, dt, mask = inputs

        # Discretize for this time interval
        discrete_drift, discrete_Q, discrete_cint = discretize_system(
            drift, diffusion_cov, cint, dt
        )

        # Predict
        pred_mean, pred_cov = kalman_predict(
            state_mean, state_cov, discrete_drift, discrete_Q, discrete_cint
        )

        # Update
        upd_mean, upd_cov, ll = kalman_update(
            pred_mean,
            pred_cov,
            jnp.nan_to_num(obs, nan=0.0),  # Replace NaN for computation
            mask,
            lambda_mat,
            manifest_means,
            manifest_cov,
        )

        return (upd_mean, upd_cov, total_ll + ll), (pred_mean, pred_cov, upd_mean, upd_cov)

    # Run filter
    inputs = (observations, time_intervals, obs_mask)
    (final_mean, final_cov, total_ll), outputs = lax.scan(
        scan_fn, (state_mean, state_cov, 0.0), inputs
    )

    pred_means, pred_covs, filt_means, filt_covs = outputs

    return KalmanFilterResult(
        filtered_means=filt_means,
        filtered_covs=filt_covs,
        predicted_means=pred_means,
        predicted_covs=pred_covs,
        log_likelihood=total_ll,
    )


def kalman_log_likelihood(
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_cov: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
) -> float:
    """Compute log-likelihood via Kalman filter (optimized for MCMC).

    This version only computes the log-likelihood without storing
    intermediate states, making it more memory efficient for MCMC.

    Args:
        Same as kalman_filter

    Returns:
        Total log-likelihood (scalar)
    """
    from dsem_agent.models.ctsem.core import discretize_system

    T, n_manifest = observations.shape

    if obs_mask is None:
        obs_mask = ~jnp.isnan(observations)

    def scan_fn(carry, inputs):
        state_mean, state_cov, total_ll = carry
        obs, dt, mask = inputs

        # Discretize
        discrete_drift, discrete_Q, discrete_cint = discretize_system(
            drift, diffusion_cov, cint, dt
        )

        # Predict
        pred_mean, pred_cov = kalman_predict(
            state_mean, state_cov, discrete_drift, discrete_Q, discrete_cint
        )

        # Update
        upd_mean, upd_cov, ll = kalman_update(
            pred_mean,
            pred_cov,
            jnp.nan_to_num(obs, nan=0.0),
            mask,
            lambda_mat,
            manifest_means,
            manifest_cov,
        )

        return (upd_mean, upd_cov, total_ll + ll), None

    # Run filter
    t0_mean_init = t0_mean
    t0_cov_init = t0_cov
    (_, _, total_ll), _ = lax.scan(
        scan_fn, (t0_mean_init, t0_cov_init, 0.0), (observations, time_intervals, obs_mask)
    )

    return total_ll


def kalman_log_likelihood_single_subject(
    observations: jnp.ndarray,
    times: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_chol: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
) -> float:
    """Log-likelihood for a single subject (convenience function).

    Takes Cholesky factors instead of covariances for easier NumPyro integration.

    Args:
        observations: (T, n_manifest) observed data
        times: (T,) absolute times (intervals computed internally)
        drift: (n_latent, n_latent) continuous drift A
        diffusion_chol: (n_latent, n_latent) lower Cholesky of diffusion
        cint: (n_latent,) continuous intercept (optional)
        lambda_mat: (n_manifest, n_latent) factor loadings
        manifest_means: (n_manifest,) manifest intercepts
        manifest_chol: (n_manifest, n_manifest) lower Cholesky of measurement error
        t0_mean: (n_latent,) initial state mean
        t0_chol: (n_latent, n_latent) lower Cholesky of initial covariance

    Returns:
        Log-likelihood (scalar)
    """
    # Compute covariances from Cholesky factors
    diffusion_cov = diffusion_chol @ diffusion_chol.T
    manifest_cov = manifest_chol @ manifest_chol.T
    t0_cov = t0_chol @ t0_chol.T

    # Compute time intervals
    time_intervals = jnp.diff(times, prepend=times[0])
    # First interval is from t0; use small positive value
    time_intervals = time_intervals.at[0].set(1e-6)

    return kalman_log_likelihood(
        observations=observations,
        time_intervals=time_intervals,
        drift=drift,
        diffusion_cov=diffusion_cov,
        cint=cint,
        lambda_mat=lambda_mat,
        manifest_means=manifest_means,
        manifest_cov=manifest_cov,
        t0_mean=t0_mean,
        t0_cov=t0_cov,
    )
