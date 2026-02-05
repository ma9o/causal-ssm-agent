"""Base protocol and parameter types for likelihood computation.

Defines the interface that all likelihood backends must implement:
compute_log_likelihood(params, observations, times) -> float

This allows plugging different state-space marginalization strategies
(Kalman, UKF, particle filter) into NumPyro models via numpyro.factor().
"""

from typing import NamedTuple, Protocol

import jax.numpy as jnp


class CTParams(NamedTuple):
    """Continuous-time state-space parameters.

    Represents the continuous-time SDE:
        dη = (A*η + c) dt + G dW

    where:
        A = drift matrix (n_latent x n_latent)
        G = diffusion Cholesky factor (n_latent x n_latent)
        c = continuous intercept (n_latent,)
    """

    drift: jnp.ndarray  # (n_latent, n_latent)
    diffusion_cov: jnp.ndarray  # (n_latent, n_latent) - G @ G.T
    cint: jnp.ndarray | None  # (n_latent,) or None


class DTParams(NamedTuple):
    """Discrete-time state-space parameters.

    Represents the discretized system:
        η_{t+dt} = A_d * η_t + c_d + ε, ε ~ N(0, Q_d)

    where:
        A_d = exp(A*dt)
        Q_d = discretized process noise covariance
        c_d = discretized intercept
    """

    drift: jnp.ndarray  # (n_latent, n_latent) - exp(A*dt)
    process_cov: jnp.ndarray  # (n_latent, n_latent) - Q_d
    cint: jnp.ndarray | None  # (n_latent,) or None


class MeasurementParams(NamedTuple):
    """Measurement model parameters.

    Represents the observation equation:
        y = Λ*η + μ + ε, ε ~ N(0, R)

    where:
        Λ = factor loadings (n_manifest x n_latent)
        μ = manifest intercepts (n_manifest,)
        R = measurement error covariance (n_manifest x n_manifest)
    """

    lambda_mat: jnp.ndarray  # (n_manifest, n_latent)
    manifest_means: jnp.ndarray  # (n_manifest,)
    manifest_cov: jnp.ndarray  # (n_manifest, n_manifest)


class InitialStateParams(NamedTuple):
    """Initial state distribution parameters.

    η_0 ~ N(m_0, P_0)
    """

    mean: jnp.ndarray  # (n_latent,)
    cov: jnp.ndarray  # (n_latent, n_latent)


class LikelihoodBackend(Protocol):
    """Protocol for state-space likelihood computation backends.

    Each backend must implement compute_log_likelihood() which integrates
    out latent states and returns log p(y|θ) as a scalar.

    The returned value is used in NumPyro via:
        numpyro.factor("ssm", backend.compute_log_likelihood(...))

    Example implementations:
    - KalmanLikelihood: Exact for linear-Gaussian (dynamax)
    - UKFLikelihood: Approximate for mildly nonlinear (dynamax)
    - ParticleLikelihood: General nonlinear/non-Gaussian (cuthbert)
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
        """Compute log-likelihood by marginalizing out latent states.

        Args:
            ct_params: Continuous-time dynamics parameters (drift, diffusion, cint)
            measurement_params: Observation model parameters (Λ, μ, R)
            initial_state: Initial state distribution (m_0, P_0)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals between observations
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Log-likelihood p(y|θ) as a scalar, suitable for numpyro.factor()
        """
        ...
