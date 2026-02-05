"""Kalman filter likelihood backend for linear-Gaussian models.

Wraps the existing Kalman filter implementation to provide a unified
interface for state-space likelihood computation.

Use when:
- Dynamics are linear (drift matrix is constant)
- Process noise is Gaussian
- Measurement model is linear
- Observation noise is Gaussian

This is the exact solution - no approximation needed.
"""

import jax.numpy as jnp
from jax import lax

from dsem_agent.models.ctsem.discretization import discretize_system
from dsem_agent.models.ctsem.kalman import (
    kalman_predict,
    kalman_update,
)
from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)


class KalmanLikelihood:
    """Kalman filter likelihood backend for linear-Gaussian models.

    Computes exact log-likelihood by running a Kalman filter forward pass,
    integrating out the latent states analytically.

    Example usage in NumPyro:
        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
    ) -> float:
        """Compute log-likelihood via Kalman filter.

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Total log-likelihood (scalar)
        """
        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)

        # Extract parameters
        drift = ct_params.drift
        diffusion_cov = ct_params.diffusion_cov
        cint = ct_params.cint
        lambda_mat = measurement_params.lambda_mat
        manifest_means = measurement_params.manifest_means
        manifest_cov = measurement_params.manifest_cov
        t0_mean = initial_state.mean
        t0_cov = initial_state.cov

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
                jnp.nan_to_num(obs, nan=0.0),
                mask,
                lambda_mat,
                manifest_means,
                manifest_cov,
            )

            return (upd_mean, upd_cov, total_ll + ll), None

        # Run filter
        (_, _, total_ll), _ = lax.scan(
            scan_fn, (t0_mean, t0_cov, 0.0), (observations, time_intervals, obs_mask)
        )

        return total_ll


# Convenience function for backwards compatibility
def compute_kalman_log_likelihood(
    ct_params: CTParams,
    measurement_params: MeasurementParams,
    initial_state: InitialStateParams,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
) -> float:
    """Functional interface to Kalman filter likelihood.

    Args:
        ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
        measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
        initial_state: Initial state distribution (mean, cov)
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals between observations
        obs_mask: (T, n_manifest) boolean mask for observed values

    Returns:
        Total log-likelihood (scalar)
    """
    backend = KalmanLikelihood()
    return backend.compute_log_likelihood(
        ct_params, measurement_params, initial_state, observations, time_intervals, obs_mask
    )
