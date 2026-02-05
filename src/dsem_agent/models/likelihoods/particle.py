"""Particle Filter likelihood backend for general nonlinear/non-Gaussian models.

**Status: Stub implementation - requires cuthbert or blackjax dependency**

This module will wrap cuthbert's bootstrap particle filter to provide
approximate inference for models with arbitrary nonlinearity and non-Gaussian
distributions.

Use when:
- Dynamics are strongly nonlinear
- Process noise is non-Gaussian
- Observation noise is non-Gaussian (Poisson, Student-t, etc.)
- UKF approximation is insufficient

The particle filter uses Sequential Monte Carlo (SMC) to approximate
the filtering distribution with weighted samples, handling arbitrary
model structures at the cost of increased computation.

Dependencies:
    pip install cuthbert  # or implement directly with blackjax

References:
    - Gordon, Salmond, Smith (1993): Novel approach to nonlinear/non-Gaussian
      Bayesian state estimation
    - Doucet & Johansen (2011): A tutorial on particle filtering
    - Chopin & Papaspiliopoulos (2020): An Introduction to Sequential Monte Carlo
    - https://github.com/probml/cuthbert
"""

import jax.numpy as jnp

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)


class ParticleLikelihood:
    """Bootstrap Particle Filter likelihood backend.

    Computes approximate log-likelihood for general nonlinear and
    non-Gaussian models using Sequential Monte Carlo.

    **Not yet implemented** - requires cuthbert or direct blackjax integration.

    Example usage (future):
        backend = ParticleLikelihood(n_particles=1000)
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)

    Args:
        n_particles: Number of particles (default: 1000)
        resampling: Resampling strategy ('systematic', 'multinomial', 'residual')
        ess_threshold: Effective sample size threshold for resampling (0-1)
    """

    def __init__(
        self,
        n_particles: int = 1000,
        resampling: str = "systematic",
        ess_threshold: float = 0.5,
    ):
        """Initialize particle filter with configuration.

        Args:
            n_particles: Number of particles for SMC approximation
            resampling: Resampling method ('systematic' recommended)
            ess_threshold: Resample when ESS/N drops below this (default 0.5)
        """
        self.n_particles = n_particles
        self.resampling = resampling
        self.ess_threshold = ess_threshold

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
    ) -> float:
        """Compute log-likelihood via Bootstrap Particle Filter.

        **Not yet implemented.**

        To implement:
        1. Add cuthbert (or blackjax) to dependencies
        2. Convert CTParams to particle filter state-space model
        3. Run bootstrap filter forward pass with adaptive resampling
        4. Return accumulated log-likelihood estimate

        Note: The log-likelihood estimate is unbiased but has variance
        that decreases with particle count. For MCMC, this produces
        a valid (pseudo-marginal) posterior.

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Log-likelihood estimate (scalar)

        Raises:
            NotImplementedError: Always (pending cuthbert integration)
        """
        raise NotImplementedError(
            "Particle filter backend requires cuthbert or blackjax.\n"
            "Install with: pip install cuthbert\n"
            "See docs/modeling/inference-strategies.md for implementation notes."
        )


def compute_particle_log_likelihood(
    ct_params: CTParams,
    measurement_params: MeasurementParams,
    initial_state: InitialStateParams,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
    n_particles: int = 1000,
    resampling: str = "systematic",
    ess_threshold: float = 0.5,
) -> float:
    """Functional interface to particle filter likelihood.

    **Not yet implemented.**

    Args:
        ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
        measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
        initial_state: Initial state distribution (mean, cov)
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals between observations
        obs_mask: (T, n_manifest) boolean mask for observed values
        n_particles: Number of particles for SMC
        resampling: Resampling strategy
        ess_threshold: ESS threshold for adaptive resampling

    Returns:
        Log-likelihood estimate (scalar)

    Raises:
        NotImplementedError: Always (pending cuthbert integration)
    """
    backend = ParticleLikelihood(
        n_particles=n_particles,
        resampling=resampling,
        ess_threshold=ess_threshold,
    )
    return backend.compute_log_likelihood(
        ct_params, measurement_params, initial_state, observations, time_intervals, obs_mask
    )
