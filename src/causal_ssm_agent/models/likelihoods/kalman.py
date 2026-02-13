"""Kalman filter likelihood backend via cuthbert moments filter.

Computes log p(y|θ) analytically for linear Gaussian SSMs using the
non-associative moments filter. No particles, no noise — the marginal
likelihood is computed in closed form via the prediction error decomposition.

Uses gaussian.moments with associative=False (not gaussian.kalman) because
cuthbert's associative Kalman filter uses QR decomposition internally
(tria()) which produces NaN gradients when matrices are ill-conditioned.
The non-associative moments filter uses predict/update operations that have
stable gradients, and is exact for linear models.

Use when:
- Linear dynamics (drift matrix, not nonlinear transition)
- Gaussian observation and process noise
- This gives the exact marginal likelihood, ideal for SVI and NUTS
"""

import jax.numpy as jnp
import jax.scipy.linalg as jla

from causal_ssm_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
    preprocess_missing_data,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched


class KalmanLikelihood:
    """Exact Kalman filter likelihood backend via cuthbert moments filter.

    Computes log p(y|theta) using the non-associative moments filter for
    linear Gaussian state-space models. Pre-discretizes CT parameters for
    all time intervals, then runs cuthbert's sequential Kalman filter.

    Args:
        n_latent: Number of latent states
        n_manifest: Number of manifest indicators
    """

    def __init__(self, n_latent: int, n_manifest: int):
        self.n_latent = n_latent
        self.n_manifest = n_manifest

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,  # noqa: ARG002
    ) -> float:
        """Compute exact log-likelihood via Kalman filter.

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals BEFORE each observation
            obs_mask: (T, n_manifest) boolean mask for observed values
            extra_params: Ignored (no noise family hyperparameters for Kalman)

        Returns:
            Log-likelihood p(y|theta) as a scalar
        """
        from cuthbert.filtering import filter as cuthbert_filter
        from cuthbert.gaussian.moments import build_filter

        n = self.n_latent
        m = self.n_manifest

        # Handle missing data: inflate R for missing observations
        clean_obs, R_adjusted, obs_mask = preprocess_missing_data(
            observations, measurement_params.manifest_cov, obs_mask
        )

        # Pre-discretize CT params for all time intervals
        # Ad: (T, n, n), Qd: (T, n, n), cd: (T, n) or None
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))

        # Cholesky factors for cuthbert (square-root form)
        jitter = jnp.eye(n) * 1e-6
        chol_Qd = jla.cholesky(Qd + jitter, lower=True)  # (T, n, n)

        m_jitter = jnp.eye(m) * 1e-6
        chol_R = jla.cholesky(R_adjusted + m_jitter, lower=True)  # (T, m, m)

        chol_P0 = jla.cholesky(initial_state.cov + jitter, lower=True)  # (n, n)

        # Build model_inputs with leading temporal dimension T.
        # cuthbert convention: model_inputs[0] → init_prepare (initial state + first obs)
        # model_inputs[k] for k≥1 → dynamics from k-1→k + obs k
        H = measurement_params.lambda_mat  # (m, n)
        d = measurement_params.manifest_means  # (m,)

        model_inputs = {
            "m0": jnp.broadcast_to(initial_state.mean, (len(time_intervals), n)),
            "chol_P0": jnp.broadcast_to(chol_P0, (len(time_intervals), n, n)),
            "F": Ad,
            "c": cd,
            "chol_Q": chol_Qd,
            "H": jnp.broadcast_to(H, (len(time_intervals), m, n)),
            "d": jnp.broadcast_to(d, (len(time_intervals), m)),
            "chol_R": chol_R,
            "y": clean_obs,
        }

        # Moments filter callbacks for linear model.
        # GetDynamicsMoments returns (mean_and_chol_cov_func, linearization_point).
        # For linear dynamics: f(x) = F@x + c, with Cholesky noise chol_Q.
        # GetObservationMoments returns (mean_and_chol_cov_func, lin_point, y).
        # For linear obs: g(x) = H@x + d, with Cholesky noise chol_R.

        def get_init_params(inputs):
            return inputs["m0"], inputs["chol_P0"]

        def get_dynamics_params(state, inputs):
            F_t = inputs["F"]
            c_t = inputs["c"]
            chol_Q_t = inputs["chol_Q"]

            def dynamics_fn(x):
                return F_t @ x + c_t, chol_Q_t

            # Linearization point: previous filter mean (from state).
            # For linear models the linearization is exact regardless.
            return dynamics_fn, state.mean

        def get_observation_params(state, inputs):
            H_t = inputs["H"]
            d_t = inputs["d"]
            chol_R_t = inputs["chol_R"]
            y_t = inputs["y"]

            def obs_fn(x):
                return H_t @ x + d_t, chol_R_t

            return obs_fn, state.mean, y_t

        # Build non-associative moments filter (stable gradients)
        filter_obj = build_filter(
            get_init_params=get_init_params,
            get_dynamics_params=get_dynamics_params,
            get_observation_params=get_observation_params,
            associative=False,
        )

        states = cuthbert_filter(filter_obj, model_inputs)
        return states.log_normalizing_constant[-1]
