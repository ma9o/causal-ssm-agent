"""Unscented Kalman Filter likelihood backend for mildly nonlinear models.

Implements the UKF algorithm directly in JAX for models with nonlinear
dynamics but Gaussian noise. Uses sigma point propagation to capture
second-order effects without requiring Jacobian computation.

Use when:
- Dynamics are nonlinear but smooth (no discontinuities)
- Process noise is Gaussian
- Observation noise is Gaussian
- Measurement model may be nonlinear

References:
    - Julier & Uhlmann (1997): New extension of the Kalman filter
    - Särkkä (2013): Bayesian Filtering and Smoothing, Ch. 5
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.ssm.discretization import discretize_system


class UKFHyperParams(NamedTuple):
    """Hyperparameters for Unscented Kalman Filter.

    Controls the spread and weighting of sigma points.
    """

    alpha: float = 1e-3  # Spread of sigma points (small positive)
    beta: float = 2.0  # Prior knowledge (2.0 optimal for Gaussian)
    kappa: float = 0.0  # Secondary scaling parameter


def compute_sigma_points(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute sigma points and weights for unscented transform.

    Args:
        mean: State mean (n,)
        cov: State covariance (n, n)
        alpha: Spread parameter
        beta: Distribution parameter
        kappa: Secondary scaling

    Returns:
        sigma_points: (2n+1, n) sigma points
        wm: (2n+1,) weights for mean
        wc: (2n+1,) weights for covariance
    """
    n = mean.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

    # Compute square root of covariance via Cholesky
    # Add small regularization for numerical stability
    cov_reg = cov + jnp.eye(n) * 1e-8
    sqrt_cov = jla.cholesky(cov_reg, lower=True)

    # Scaling factor
    scale = jnp.sqrt(n + lambda_)

    # Generate sigma points: [mean, mean + scale*sqrt_cov, mean - scale*sqrt_cov]
    sigma_points = jnp.zeros((2 * n + 1, n))
    sigma_points = sigma_points.at[0].set(mean)

    for i in range(n):
        sigma_points = sigma_points.at[i + 1].set(mean + scale * sqrt_cov[:, i])
        sigma_points = sigma_points.at[n + i + 1].set(mean - scale * sqrt_cov[:, i])

    # Weights for mean
    wm = jnp.zeros(2 * n + 1)
    wm = wm.at[0].set(lambda_ / (n + lambda_))
    wm = wm.at[1:].set(1.0 / (2 * (n + lambda_)))

    # Weights for covariance
    wc = jnp.zeros(2 * n + 1)
    wc = wc.at[0].set(lambda_ / (n + lambda_) + (1 - alpha**2 + beta))
    wc = wc.at[1:].set(1.0 / (2 * (n + lambda_)))

    return sigma_points, wm, wc


def unscented_transform(
    sigma_points: jnp.ndarray,
    wm: jnp.ndarray,
    wc: jnp.ndarray,
    transform_fn: Callable[[jnp.ndarray], jnp.ndarray],
    noise_cov: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply unscented transform to propagate distribution through nonlinear function.

    Args:
        sigma_points: (2n+1, n) sigma points
        wm: (2n+1,) weights for mean
        wc: (2n+1,) weights for covariance
        transform_fn: Function to apply to each sigma point
        noise_cov: Additive noise covariance (optional)

    Returns:
        mean: Transformed mean
        cov: Transformed covariance
    """
    # Propagate sigma points
    transformed = jnp.array([transform_fn(sp) for sp in sigma_points])

    # Compute mean
    mean = jnp.sum(wm[:, None] * transformed, axis=0)

    # Compute covariance
    diff = transformed - mean
    cov = jnp.sum(wc[:, None, None] * jnp.einsum("ij,ik->ijk", diff, diff), axis=0)

    if noise_cov is not None:
        cov = cov + noise_cov

    # Ensure symmetry
    cov = 0.5 * (cov + cov.T)

    return mean, cov


def ukf_predict(
    state_mean: jnp.ndarray,
    state_cov: jnp.ndarray,
    dynamics_fn: Callable[[jnp.ndarray], jnp.ndarray],
    process_cov: jnp.ndarray,
    hyperparams: UKFHyperParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """UKF prediction step.

    Args:
        state_mean: Current state mean (n,)
        state_cov: Current state covariance (n, n)
        dynamics_fn: State transition function x_{t+1} = f(x_t)
        process_cov: Process noise covariance (n, n)
        hyperparams: UKF tuning parameters

    Returns:
        predicted_mean, predicted_cov
    """
    sigma_points, wm, wc = compute_sigma_points(
        state_mean, state_cov, hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    )

    pred_mean, pred_cov = unscented_transform(
        sigma_points, wm, wc, dynamics_fn, process_cov
    )

    return pred_mean, pred_cov


def ukf_update(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    observation: jnp.ndarray,
    obs_mask: jnp.ndarray,
    measurement_fn: Callable[[jnp.ndarray], jnp.ndarray],
    measurement_cov: jnp.ndarray,
    hyperparams: UKFHyperParams,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """UKF update step with missing data handling.

    Args:
        pred_mean: Predicted state mean (n_latent,)
        pred_cov: Predicted state covariance (n_latent, n_latent)
        observation: Observed values (n_manifest,)
        obs_mask: Boolean mask for observed values (n_manifest,)
        measurement_fn: Measurement function y = h(x)
        measurement_cov: Measurement noise covariance (n_manifest, n_manifest)
        hyperparams: UKF tuning parameters

    Returns:
        updated_mean, updated_cov, log_likelihood
    """
    n_manifest = observation.shape[0]

    # Generate sigma points from predicted state
    sigma_points, wm, wc = compute_sigma_points(
        pred_mean, pred_cov, hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    )

    # Transform sigma points through measurement function
    transformed_obs = jnp.array([measurement_fn(sp) for sp in sigma_points])

    # Predicted observation mean
    obs_mean = jnp.sum(wm[:, None] * transformed_obs, axis=0)

    # Innovation covariance
    obs_diff = transformed_obs - obs_mean
    S = jnp.sum(wc[:, None, None] * jnp.einsum("ij,ik->ijk", obs_diff, obs_diff), axis=0)

    # Handle missing data by inflating variance
    mask_float = obs_mask.astype(jnp.float32)
    large_var = 1e10
    adjusted_measurement_cov = measurement_cov + jnp.diag((1.0 - mask_float) * large_var)
    S = S + adjusted_measurement_cov
    S = 0.5 * (S + S.T) + jnp.eye(n_manifest) * 1e-8

    # Cross-covariance
    state_diff = sigma_points - pred_mean
    Pxy = jnp.sum(wc[:, None, None] * jnp.einsum("ij,ik->ijk", state_diff, obs_diff), axis=0)

    # Kalman gain
    K = jla.solve(S, Pxy.T, assume_a="pos").T

    # Innovation (masked for missing data)
    innovation = (observation - obs_mean) * mask_float

    # Update
    updated_mean = pred_mean + K @ innovation
    updated_cov = pred_cov - K @ S @ K.T
    updated_cov = 0.5 * (updated_cov + updated_cov.T)

    # Log-likelihood (only for observed variables)
    n_observed = jnp.sum(mask_float)
    S_obs = S  # Use the non-inflated version for likelihood
    _, logdet = jnp.linalg.slogdet(S_obs)
    innovation_obs = observation - obs_mean
    mahal = innovation_obs @ jla.solve(S_obs, innovation_obs, assume_a="pos")
    ll = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)
    ll = jnp.where(n_observed > 0, ll, 0.0)

    return updated_mean, updated_cov, ll


class UKFLikelihood:
    """Unscented Kalman Filter likelihood backend.

    Computes approximate log-likelihood for models with mildly nonlinear
    dynamics using sigma point propagation.

    For linear models, this is equivalent to the standard Kalman filter
    but with slightly higher computational cost.

    Example:
        backend = UKFLikelihood(alpha=1e-3, beta=2.0)
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray] | None = None,
        measurement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    ):
        """Initialize UKF with tuning parameters and optional custom functions.

        Args:
            alpha: Spread of sigma points (small positive, e.g., 1e-3)
            beta: Distribution parameter (2.0 optimal for Gaussian)
            kappa: Secondary scaling (typically 0 or 3-n)
            dynamics_fn: Custom dynamics x_{t+1} = f(x_t, params, dt). If None,
                uses linear dynamics from ct_params.
            measurement_fn: Custom measurement y = h(x, params). If None,
                uses linear measurement from measurement_params.
        """
        self.hyperparams = UKFHyperParams(alpha=alpha, beta=beta, kappa=kappa)
        self.custom_dynamics_fn = dynamics_fn
        self.custom_measurement_fn = measurement_fn

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

            # Define dynamics function for this timestep
            if self.custom_dynamics_fn is not None:

                def dynamics_fn(x):
                    return self.custom_dynamics_fn(x, ct_params, dt)
            else:
                # Linear dynamics: x_{t+1} = A @ x_t + c
                def dynamics_fn(x):
                    result = discrete_drift @ x
                    if discrete_cint is not None:
                        result = result + discrete_cint.flatten()
                    return result

            # Define measurement function
            if self.custom_measurement_fn is not None:

                def measurement_fn(x):
                    return self.custom_measurement_fn(x, measurement_params)
            else:
                # Linear measurement: y = H @ x + mu
                def measurement_fn(x):
                    return lambda_mat @ x + manifest_means

            # Predict
            pred_mean, pred_cov = ukf_predict(
                state_mean, state_cov, dynamics_fn, discrete_Q, self.hyperparams
            )

            # Update
            upd_mean, upd_cov, ll = ukf_update(
                pred_mean,
                pred_cov,
                jnp.nan_to_num(obs, nan=0.0),
                mask,
                measurement_fn,
                manifest_cov,
                self.hyperparams,
            )

            return (upd_mean, upd_cov, total_ll + ll), None

        # Run filter
        (_, _, total_ll), _ = lax.scan(
            scan_fn, (t0_mean, t0_cov, 0.0), (observations, time_intervals, obs_mask)
        )

        return total_ll


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
    """
    backend = UKFLikelihood(alpha=alpha, beta=beta, kappa=kappa)
    return backend.compute_log_likelihood(
        ct_params, measurement_params, initial_state, observations, time_intervals, obs_mask
    )
