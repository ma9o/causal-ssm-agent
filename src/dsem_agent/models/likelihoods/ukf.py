"""Unscented Kalman Filter likelihood backend for mildly nonlinear models.

**Status: Stub implementation - requires dynamax dependency**

This module will wrap dynamax's UKF implementation to provide approximate
inference for models with mildly nonlinear dynamics but Gaussian noise.

Use when:
- Dynamics are nonlinear but smooth (no discontinuities)
- Process noise is Gaussian
- Observation noise is Gaussian
- Measurement model may be nonlinear

The UKF uses sigma point propagation to capture second-order effects
that the Extended Kalman Filter (EKF) misses, without requiring
explicit Jacobian computation.

Dependencies:
    pip install dynamax

References:
    - Julier & Uhlmann (1997): New extension of the Kalman filter
    - Särkkä (2013): Bayesian Filtering and Smoothing, Ch. 5
    - https://probml.github.io/dynamax/
"""

import jax.numpy as jnp

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)


class UKFLikelihood:
    """Unscented Kalman Filter likelihood backend.

    Computes approximate log-likelihood for models with mildly nonlinear
    dynamics using sigma point propagation.

    **Not yet implemented** - requires dynamax integration.

    Example usage (future):
        backend = UKFLikelihood(alpha=1e-3, beta=2.0, kappa=0.0)
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)

    Args:
        alpha: Spread of sigma points around mean (default: 1e-3)
        beta: Prior knowledge about distribution (default: 2.0 for Gaussian)
        kappa: Secondary scaling parameter (default: 0.0)
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """Initialize UKF with tuning parameters.

        Args:
            alpha: Spread of sigma points (small positive, e.g., 1e-3)
            beta: Distribution parameter (2.0 optimal for Gaussian)
            kappa: Secondary scaling (typically 0 or 3-n)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
    ) -> float:
        """Compute log-likelihood via Unscented Kalman Filter.

        **Not yet implemented.**

        To implement:
        1. Add dynamax to dependencies
        2. Convert CTParams to dynamax NonlinearGaussianSSM
        3. Run dynamax's UKF forward pass
        4. Return accumulated log-likelihood

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Total log-likelihood (scalar)

        Raises:
            NotImplementedError: Always (pending dynamax integration)
        """
        raise NotImplementedError(
            "UKF backend requires dynamax. Install with: pip install dynamax\n"
            "See docs/modeling/inference-strategies.md for implementation notes."
        )


def compute_ukf_log_likelihood(
    ct_params: CTParams,
    measurement_params: MeasurementParams,
    initial_state: InitialStateParams,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> float:
    """Functional interface to UKF likelihood.

    **Not yet implemented.**

    Args:
        ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
        measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
        initial_state: Initial state distribution (mean, cov)
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals between observations
        obs_mask: (T, n_manifest) boolean mask for observed values
        alpha: UKF sigma point spread parameter
        beta: UKF distribution parameter
        kappa: UKF secondary scaling parameter

    Returns:
        Total log-likelihood (scalar)

    Raises:
        NotImplementedError: Always (pending dynamax integration)
    """
    backend = UKFLikelihood(alpha=alpha, beta=beta, kappa=kappa)
    return backend.compute_log_likelihood(
        ct_params, measurement_params, initial_state, observations, time_intervals, obs_mask
    )
