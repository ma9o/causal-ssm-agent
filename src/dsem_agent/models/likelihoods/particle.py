"""Bootstrap Particle Filter likelihood backend for general nonlinear/non-Gaussian models.

Implements Sequential Monte Carlo (SMC) directly in JAX for models with
arbitrary nonlinearity and non-Gaussian distributions.

Use when:
- Dynamics are strongly nonlinear
- Process noise is non-Gaussian
- Observation noise is non-Gaussian (Poisson, Student-t, etc.)
- UKF approximation is insufficient

References:
    - Gordon, Salmond, Smith (1993): Novel approach to nonlinear/non-Gaussian
      Bayesian state estimation
    - Doucet & Johansen (2011): A tutorial on particle filtering
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from jax import lax

from dsem_agent.models.ctsem.discretization import discretize_system
from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)


class ParticleState(NamedTuple):
    """State of the particle filter at a single time point."""

    particles: jnp.ndarray  # (n_particles, n_latent)
    log_weights: jnp.ndarray  # (n_particles,)
    log_likelihood: float  # Accumulated log-likelihood


def systematic_resample(
    key: random.PRNGKey,
    particles: jnp.ndarray,
    log_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Systematic resampling of particles.

    Args:
        key: JAX random key
        particles: (n_particles, n_latent) particle positions
        log_weights: (n_particles,) log weights (unnormalized)

    Returns:
        resampled_particles: (n_particles, n_latent)
        new_log_weights: (n_particles,) uniform log weights
    """
    n_particles = particles.shape[0]

    # Normalize weights
    log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)

    # Cumulative sum
    cumsum = jnp.cumsum(weights)

    # Systematic resampling positions
    u = random.uniform(key, shape=()) / n_particles
    positions = u + jnp.arange(n_particles) / n_particles

    # Find indices
    indices = jnp.searchsorted(cumsum, positions)
    indices = jnp.clip(indices, 0, n_particles - 1)

    # Resample
    resampled = particles[indices]
    new_log_weights = jnp.zeros(n_particles) - jnp.log(n_particles)

    return resampled, new_log_weights


def compute_ess(log_weights: jnp.ndarray) -> float:
    """Compute effective sample size from log weights.

    Args:
        log_weights: (n_particles,) unnormalized log weights

    Returns:
        ESS as fraction of n_particles (0 to 1)
    """
    n_particles = log_weights.shape[0]
    log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)
    ess = 1.0 / jnp.sum(weights**2)
    return ess / n_particles


def gaussian_log_likelihood(
    observation: jnp.ndarray,
    predicted: jnp.ndarray,
    cov: jnp.ndarray,
    obs_mask: jnp.ndarray,
) -> float:
    """Compute Gaussian log-likelihood with missing data handling.

    Args:
        observation: (n_manifest,) observed values
        predicted: (n_manifest,) predicted values
        cov: (n_manifest, n_manifest) covariance
        obs_mask: (n_manifest,) boolean mask for observed

    Returns:
        Log-likelihood (scalar)
    """
    n_manifest = observation.shape[0]
    mask_float = obs_mask.astype(jnp.float32)
    n_observed = jnp.sum(mask_float)

    # Mask innovation
    innovation = (observation - predicted) * mask_float

    # Add large variance for missing observations
    large_var = 1e10
    adjusted_cov = cov + jnp.diag((1.0 - mask_float) * large_var)
    adjusted_cov = 0.5 * (adjusted_cov + adjusted_cov.T) + jnp.eye(n_manifest) * 1e-8

    # Compute log-likelihood
    _, logdet = jnp.linalg.slogdet(adjusted_cov)
    mahal = innovation @ jla.solve(adjusted_cov, innovation, assume_a="pos")
    ll = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)

    return jnp.where(n_observed > 0, ll, 0.0)


class ParticleLikelihood:
    """Bootstrap Particle Filter likelihood backend.

    Computes approximate log-likelihood for general nonlinear and
    non-Gaussian models using Sequential Monte Carlo.

    The log-likelihood estimate is unbiased but has variance that
    decreases with particle count. For MCMC, this produces a valid
    (pseudo-marginal) posterior.

    Example:
        backend = ParticleLikelihood(n_particles=1000)
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

    def __init__(
        self,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
        seed: int = 0,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray, float, random.PRNGKey], jnp.ndarray]
        | None = None,
        log_likelihood_fn: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray, MeasurementParams], float
        ]
        | None = None,
    ):
        """Initialize particle filter with configuration.

        Args:
            n_particles: Number of particles for SMC approximation
            ess_threshold: Resample when ESS/N drops below this (default 0.5)
            seed: Random seed for reproducibility
            dynamics_fn: Custom dynamics with signature
                f(x, params, dt, key) -> x_next. If None, uses linear Gaussian.
            log_likelihood_fn: Custom observation log-likelihood with signature
                ll(x, observation, obs_mask, meas_params) -> float.
                If None, uses linear Gaussian.
        """
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.seed = seed
        self.custom_dynamics_fn = dynamics_fn
        self.custom_log_likelihood_fn = log_likelihood_fn

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

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Log-likelihood estimate (scalar)
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

        n_latent = t0_mean.shape[0]
        key = random.PRNGKey(self.seed)

        # Initialize particles from prior
        key, subkey = random.split(key)
        t0_chol = jla.cholesky(t0_cov + jnp.eye(n_latent) * 1e-8, lower=True)
        particles = t0_mean + random.normal(subkey, shape=(self.n_particles, n_latent)) @ t0_chol.T
        log_weights = jnp.zeros(self.n_particles) - jnp.log(self.n_particles)

        def scan_fn(carry, inputs):
            particles, log_weights, total_ll, key = carry
            obs, dt, mask = inputs

            # Discretize for this time interval
            discrete_drift, discrete_Q, discrete_cint = discretize_system(
                drift, diffusion_cov, cint, dt
            )

            # Propagate particles through dynamics
            key, subkey = random.split(key)
            Q_chol = jla.cholesky(discrete_Q + jnp.eye(n_latent) * 1e-8, lower=True)

            if self.custom_dynamics_fn is not None:
                # Custom nonlinear dynamics
                keys = random.split(subkey, self.n_particles)
                new_particles = jnp.array(
                    [self.custom_dynamics_fn(p, ct_params, dt, k) for p, k in zip(particles, keys)]
                )
            else:
                # Linear Gaussian dynamics
                noise = random.normal(subkey, shape=(self.n_particles, n_latent)) @ Q_chol.T
                new_particles = particles @ discrete_drift.T + noise
                if discrete_cint is not None:
                    new_particles = new_particles + discrete_cint.flatten()

            # Update weights based on observation likelihood
            obs_clean = jnp.nan_to_num(obs, nan=0.0)

            if self.custom_log_likelihood_fn is not None:
                # Custom observation model
                new_log_weights = jnp.array(
                    [
                        log_weights[i]
                        + self.custom_log_likelihood_fn(
                            new_particles[i], obs_clean, mask, measurement_params
                        )
                        for i in range(self.n_particles)
                    ]
                )
            else:
                # Linear Gaussian observation model
                predicted_obs = new_particles @ lambda_mat.T + manifest_means
                particle_lls = jnp.array(
                    [
                        gaussian_log_likelihood(obs_clean, predicted_obs[i], manifest_cov, mask)
                        for i in range(self.n_particles)
                    ]
                )
                new_log_weights = log_weights + particle_lls

            # Compute log-likelihood increment (log of mean weight)
            log_mean_weight = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(
                self.n_particles
            )

            # Normalize weights
            new_log_weights_norm = new_log_weights - jax.scipy.special.logsumexp(new_log_weights)

            # Adaptive resampling based on ESS
            key, subkey = random.split(key)
            ess = compute_ess(new_log_weights_norm)

            # Resample if ESS is low
            def do_resample(_):
                return systematic_resample(subkey, new_particles, new_log_weights_norm)

            def no_resample(_):
                return new_particles, new_log_weights_norm

            final_particles, final_log_weights = lax.cond(
                ess < self.ess_threshold, do_resample, no_resample, None
            )

            return (final_particles, final_log_weights, total_ll + log_mean_weight, key), None

        # Run filter
        init_state = (particles, log_weights, 0.0, key)
        (_, _, total_ll, _), _ = lax.scan(
            scan_fn, init_state, (observations, time_intervals, obs_mask)
        )

        return total_ll


def compute_particle_log_likelihood(
    ct_params: CTParams,
    measurement_params: MeasurementParams,
    initial_state: InitialStateParams,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    seed: int = 0,
) -> float:
    """Functional interface to particle filter likelihood.

    Args:
        ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
        measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
        initial_state: Initial state distribution (mean, cov)
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals between observations
        obs_mask: (T, n_manifest) boolean mask for observed values
        n_particles: Number of particles for SMC
        ess_threshold: ESS threshold for adaptive resampling
        seed: Random seed

    Returns:
        Log-likelihood estimate (scalar)
    """
    backend = ParticleLikelihood(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        seed=seed,
    )
    return backend.compute_log_likelihood(
        ct_params, measurement_params, initial_state, observations, time_intervals, obs_mask
    )
