"""Laplace-EM: Iterated EKF Smoother + Laplace-approximated marginal likelihood.

Implements Method 1 from the algorithmic specification:
1. Inner loop: Iterated Extended Kalman Smoother (IEKS) finds the mode of
   p(z_{1:T} | y_{1:T}, theta) via Newton iterations on the joint state posterior.
2. Laplace approximation: Gaussian approximation around the mode gives an
   approximate marginal likelihood log p(y_{1:T} | theta).
3. Outer loop: Optimize theta via gradient descent (MLE/MAP) or sample via NUTS,
   using the Laplace-approximated marginal likelihood as the log-density.

Works for any exponential-family emission (Gaussian, Poisson, Bernoulli, Gamma,
Student-t) with linear dynamics. The key requirement is twice-differentiable
log-emission density, which holds for all supported noise families.

The block-tridiagonal structure of the state-space Hessian makes the IEKS
O(T D^3) per iteration, and typically 3-8 iterations suffice for convergence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from causal_ssm_agent.models.likelihoods.kernels import build_observation_kernel
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.tempered_core import run_tempered_smc

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction


# ---------------------------------------------------------------------------
# Iterated Extended Kalman Smoother (IEKS)
# ---------------------------------------------------------------------------


def _ieks_smooth(
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H,
    d,
    R,
    init_mean,
    init_cov,
    emission_log_prob_fn,
    n_ieks_iters=5,
):
    """Run the Iterated Extended Kalman Smoother to find the MAP state trajectory.

    Args:
        observations: (T, n_manifest) observed data
        obs_mask: (T, n_manifest) boolean observation mask
        Ad: (T, D, D) discrete-time transition matrices
        Qd: (T, D, D) discrete-time process noise covariances
        cd: (T, D) discrete-time intercepts
        H: (n_manifest, D) measurement matrix
        d: (n_manifest,) measurement intercept
        R: (n_manifest, n_manifest) measurement noise covariance
        init_mean: (D,) initial state mean
        init_cov: (D, D) initial state covariance
        emission_log_prob_fn: callable(y_t, z_t, H, d, R, mask_t) -> scalar
        n_ieks_iters: number of IEKS iterations

    Returns:
        z_smooth: (T, D) smoothed state means (MAP trajectory)
        P_smooth: (T, D, D) smoothed state covariances
        log_lik_approx: scalar approximate log-likelihood
    """
    T = observations.shape[0]
    D = init_mean.shape[0]
    jitter = jnp.eye(D) * 1e-6

    # Compute gradient and Hessian of emission log-prob w.r.t. z_t
    def _emission_grad_hess(y_t, z_t, mask_t):
        """Compute gradient and negative Hessian of log p(y_t|z_t) w.r.t. z_t."""

        def _log_prob(z):
            return emission_log_prob_fn(y_t, z, H, d, R, mask_t)

        grad_fn = jax.grad(_log_prob)
        hess_fn = jax.hessian(_log_prob)
        g = grad_fn(z_t)
        neg_H = -hess_fn(z_t)
        # Ensure neg_H is symmetric PSD (it should be for exp-family with canonical link)
        neg_H = 0.5 * (neg_H + neg_H.T)
        # Clamp eigenvalues to be non-negative
        eigvals, eigvecs = jnp.linalg.eigh(neg_H)
        neg_H = eigvecs @ jnp.diag(jnp.maximum(eigvals, 0.0)) @ eigvecs.T
        return g, neg_H

    # Initialize state estimates (zeros or prior mean)
    z_est = jnp.broadcast_to(init_mean, (T, D)).copy()

    cd_scan = cd if cd is not None else jnp.zeros((T, D))
    obs_mask_float = obs_mask.astype(jnp.float32)

    # Forward pass body for lax.scan (used inside each IEKS iteration)
    def _forward_step(carry, inputs):
        z_f_prev, P_f_prev = carry
        Ad_t, Qd_t, cd_t, J_tt, ty_t = inputs

        # Predict
        z_pred = Ad_t @ z_f_prev + cd_t
        P_pred = Ad_t @ P_f_prev @ Ad_t.T + Qd_t
        P_pred = 0.5 * (P_pred + P_pred.T) + jitter

        # Update in information form
        P_pred_inv = jla.solve(P_pred + jitter, jnp.eye(D), assume_a="pos")
        P_f_inv = P_pred_inv + J_tt
        P_f = jla.solve(P_f_inv + jitter, jnp.eye(D), assume_a="pos")
        P_f = 0.5 * (P_f + P_f.T) + jitter
        z_f = P_f @ (P_pred_inv @ z_pred + ty_t)

        return (z_f, P_f), (z_f, P_f)

    # Backward pass body for lax.scan (RTS smoother)
    def _backward_step(carry, inputs):
        z_s_next, P_s_next = carry
        z_f_t, P_f_t, Ad_tp1, Qd_tp1, cd_tp1 = inputs

        z_pred_tp1 = Ad_tp1 @ z_f_t + cd_tp1
        P_pred_tp1 = Ad_tp1 @ P_f_t @ Ad_tp1.T + Qd_tp1
        P_pred_tp1 = 0.5 * (P_pred_tp1 + P_pred_tp1.T) + jitter

        G_t = P_f_t @ Ad_tp1.T @ jla.solve(P_pred_tp1 + jitter, jnp.eye(D), assume_a="pos")

        z_s_t = z_f_t + G_t @ (z_s_next - z_pred_tp1)
        P_s_t = P_f_t + G_t @ (P_s_next - P_pred_tp1) @ G_t.T
        P_s_t = 0.5 * (P_s_t + P_s_t.T) + jitter

        return (z_s_t, P_s_t), (z_s_t, P_s_t)

    def _ieks_body(_, z_est):
        """Single IEKS iteration: linearize + forward filter + backward smooth."""
        # Linearize emissions around current state estimates
        grads_and_hess = jax.vmap(_emission_grad_hess)(observations, z_est, obs_mask_float)
        grads = grads_and_hess[0]  # (T, D)
        J_t = grads_and_hess[1]  # (T, D, D) negative Hessian

        # Pseudo-observations in information form
        tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)

        # Time 0: predict from prior
        z_pred_0 = Ad[0] @ init_mean + cd_scan[0]
        P_pred_0 = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
        P_pred_0 = 0.5 * (P_pred_0 + P_pred_0.T) + jitter

        P_pred_inv_0 = jla.solve(P_pred_0 + jitter, jnp.eye(D), assume_a="pos")
        P_filt_inv_0 = P_pred_inv_0 + J_t[0]
        P_filt_0 = jla.solve(P_filt_inv_0 + jitter, jnp.eye(D), assume_a="pos")
        P_filt_0 = 0.5 * (P_filt_0 + P_filt_0.T) + jitter
        z_filt_0 = P_filt_0 @ (P_pred_inv_0 @ z_pred_0 + tilde_y[0])

        # Forward scan for t=1..T-1
        _, (z_filt_rest, P_filt_rest) = jax.lax.scan(
            _forward_step,
            (z_filt_0, P_filt_0),
            (Ad[1:], Qd[1:], cd_scan[1:], J_t[1:], tilde_y[1:]),
        )
        z_filt = jnp.concatenate([z_filt_0[None], z_filt_rest], axis=0)
        P_filt = jnp.concatenate([P_filt_0[None], P_filt_rest], axis=0)

        # Backward scan (RTS smoother)
        bwd_inputs_rev = (
            z_filt[: T - 1][::-1],
            P_filt[: T - 1][::-1],
            Ad[1:][::-1],
            Qd[1:][::-1],
            cd_scan[1:][::-1],
        )
        _, (z_smooth_rest_rev, _P_smooth_rest_rev) = jax.lax.scan(
            _backward_step,
            (z_filt[T - 1], P_filt[T - 1]),
            bwd_inputs_rev,
        )
        z_smooth = jnp.concatenate([z_smooth_rest_rev[::-1], z_filt[T - 1][None]], axis=0)

        return z_smooth

    z_est = jax.lax.fori_loop(0, n_ieks_iters, _ieks_body, z_est)

    # Final iteration to extract P_smooth for log-likelihood computation
    grads_and_hess = jax.vmap(_emission_grad_hess)(observations, z_est, obs_mask_float)
    grads_final = grads_and_hess[0]
    J_t_final = grads_and_hess[1]
    tilde_y_final = jax.vmap(lambda J, z, g: J @ z + g)(J_t_final, z_est, grads_final)

    z_pred_0 = Ad[0] @ init_mean + cd_scan[0]
    P_pred_0 = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P_pred_0 = 0.5 * (P_pred_0 + P_pred_0.T) + jitter
    P_pred_inv_0 = jla.solve(P_pred_0 + jitter, jnp.eye(D), assume_a="pos")
    P_filt_inv_0 = P_pred_inv_0 + J_t_final[0]
    P_filt_0 = jla.solve(P_filt_inv_0 + jitter, jnp.eye(D), assume_a="pos")
    P_filt_0 = 0.5 * (P_filt_0 + P_filt_0.T) + jitter

    z_filt_0 = P_filt_0 @ (P_pred_inv_0 @ z_pred_0 + tilde_y_final[0])

    _, (z_filt_rest, P_filt_rest) = jax.lax.scan(
        _forward_step,
        (z_filt_0, P_filt_0),
        (Ad[1:], Qd[1:], cd_scan[1:], J_t_final[1:], tilde_y_final[1:]),
    )
    z_filt = jnp.concatenate([z_filt_0[None], z_filt_rest], axis=0)
    P_filt = jnp.concatenate([P_filt_0[None], P_filt_rest], axis=0)

    bwd_inputs_rev = (
        z_filt[: T - 1][::-1],
        P_filt[: T - 1][::-1],
        Ad[1:][::-1],
        Qd[1:][::-1],
        cd_scan[1:][::-1],
    )
    _, (z_smooth_rest_rev, P_smooth_rest_rev) = jax.lax.scan(
        _backward_step,
        (z_filt[T - 1], P_filt[T - 1]),
        bwd_inputs_rev,
    )
    z_smooth = jnp.concatenate([z_smooth_rest_rev[::-1], z_filt[T - 1][None]], axis=0)
    P_smooth = jnp.concatenate([P_smooth_rest_rev[::-1], P_filt[T - 1][None]], axis=0)

    # Compute approximate log-likelihood via prediction error decomposition
    # Sum of log p(y_t | y_{1:t-1}, theta) from the final filter pass
    log_lik = _compute_laplace_log_lik(
        observations,
        obs_mask,
        z_smooth,
        P_smooth,
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
        emission_log_prob_fn,
        H,
        d,
        R,
    )

    return z_smooth, P_smooth, log_lik


def _compute_laplace_log_lik(
    observations,
    obs_mask,
    z_smooth,
    _P_smooth,
    Ad,
    Qd,
    cd,
    init_mean,
    init_cov,
    emission_log_prob_fn,
    H,
    d,
    R,
):
    """Compute Laplace-approximated log-likelihood via prediction error decomposition.

    Uses the one-step-ahead Laplace formula. Linearizes the emission model
    around z_smooth (from the IEKS), runs a forward Kalman filter pass with
    the linearized model, and accumulates the per-step contribution:

        ll_t = l(z_filt_t) - 0.5||z_filt_t - z_pred_t||^2_{P_pred^-1}
             + 0.5 log(det P_filt_t / det P_pred_t)

    where z_filt_t is the filter mode (the correct z* for the one-step Laplace),
    NOT the smoother output. For the linear Gaussian case this reduces to the
    exact Kalman prediction error decomposition.
    """
    T, D = z_smooth.shape
    jitter = jnp.eye(D) * 1e-6
    mask_float = obs_mask.astype(jnp.float32)

    # Compute emission gradients and Hessians at the smoothed states (linearization point)
    def _emission_grad_hess(y_t, z_t, mask_t):
        def _log_prob(z):
            return emission_log_prob_fn(y_t, z, H, d, R, mask_t)

        g = jax.grad(_log_prob)(z_t)
        neg_H = -jax.hessian(_log_prob)(z_t)
        neg_H = 0.5 * (neg_H + neg_H.T)
        eigvals, eigvecs = jnp.linalg.eigh(neg_H)
        neg_H = eigvecs @ jnp.diag(jnp.maximum(eigvals, 0.0)) @ eigvecs.T
        return g, neg_H

    all_grads_hess = jax.vmap(_emission_grad_hess)(observations, z_smooth, mask_float)
    grads = all_grads_hess[0]  # (T, D)
    J_t = all_grads_hess[1]  # (T, D, D) — emission info matrices

    def _step_ll(y_t, mask_t, z_pred, P_pred, J_obs, grad_obs, z_lin):
        """Single-step Laplace log-likelihood contribution.

        Returns (ll_t, z_filt, P_filt) for chaining the filter forward.
        """
        P_pred_reg = P_pred + jitter

        # Filter update (information form): P_filt^{-1} = P_pred^{-1} + J_obs
        P_pred_inv = jla.solve(P_pred_reg, jnp.eye(D), assume_a="pos")
        P_filt_inv = P_pred_inv + J_obs
        P_filt = jla.solve(P_filt_inv + jitter, jnp.eye(D), assume_a="pos")
        P_filt = 0.5 * (P_filt + P_filt.T) + jitter

        # Filter mean: z_filt = P_filt @ (P_pred_inv @ z_pred + tilde_y)
        tilde_y = J_obs @ z_lin + grad_obs
        z_filt = P_filt @ (P_pred_inv @ z_pred + tilde_y)

        # Emission log-prob at the filter mode z_filt (the correct z* for one-step Laplace)
        emission_ll = emission_log_prob_fn(y_t, z_filt, H, d, R, mask_t)

        # Log-determinant ratio: log(det P_filt / det P_pred)
        _, ld_filt = jnp.linalg.slogdet(P_filt)
        _, ld_pred = jnp.linalg.slogdet(P_pred_reg)
        log_det_ratio = ld_filt - ld_pred

        # Prior penalty: -0.5 * (z_filt - z_pred)^T P_pred^{-1} (z_filt - z_pred)
        diff = z_filt - z_pred
        mahal = diff @ jla.solve(P_pred_reg, diff, assume_a="pos")

        ll_t = emission_ll - 0.5 * mahal + 0.5 * log_det_ratio
        return ll_t, z_filt, P_filt

    # Time 0: predict from initial state
    z_pred_0 = Ad[0] @ init_mean + cd[0]
    P_pred_0 = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P_pred_0 = 0.5 * (P_pred_0 + P_pred_0.T)

    ll_0, z_filt_0, P_filt_0 = _step_ll(
        observations[0], mask_float[0], z_pred_0, P_pred_0, J_t[0], grads[0], z_smooth[0]
    )

    if T == 1:
        return ll_0

    # Forward scan for t=1..T-1: predict, then compute ll_t
    def _forward_ll_step(carry, inputs):
        z_filt_prev, P_filt_prev = carry
        Ad_t, Qd_t, cd_t, z_lin_t, J_obs_t, grad_t, y_t, mask_t = inputs

        # Predict
        z_pred = Ad_t @ z_filt_prev + cd_t
        P_pred = Ad_t @ P_filt_prev @ Ad_t.T + Qd_t
        P_pred = 0.5 * (P_pred + P_pred.T)

        ll_t, z_filt, P_filt = _step_ll(
            y_t, mask_t, z_pred, P_pred, J_obs_t, grad_t, z_lin_t
        )

        return (z_filt, P_filt), ll_t

    _, ll_rest = jax.lax.scan(
        _forward_ll_step,
        (z_filt_0, P_filt_0),
        (Ad[1:], Qd[1:], cd[1:], z_smooth[1:], J_t[1:], grads[1:],
         observations[1:], mask_float[1:]),
    )

    return ll_0 + jnp.sum(ll_rest)


# ---------------------------------------------------------------------------
# Laplace likelihood backend (for use in NumPyro model)
# ---------------------------------------------------------------------------


class LaplaceLikelihood:
    """Laplace-approximated likelihood backend.

    Computes log p(y|theta) via IEKS + Laplace approximation.
    Drop-in replacement for KalmanLikelihood / ParticleLikelihood.
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: DistributionFamily | str = "gaussian",
        manifest_link: LinkFunction | str = "identity",
        n_ieks_iters: int = 5,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.manifest_link = manifest_link
        self.n_ieks_iters = n_ieks_iters

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> float:
        """Compute Laplace-approximated log-likelihood."""
        n = self.n_latent

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        # Pre-discretize CT -> DT
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))

        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

        obs_kernel = build_observation_kernel(
            DistributionFamily(self.manifest_dist),
            LinkFunction(self.manifest_link),
            extra_params,
        )

        _, _, log_lik = _ieks_smooth(
            clean_obs,
            obs_mask,
            Ad,
            Qd,
            cd,
            measurement_params.lambda_mat,
            measurement_params.manifest_means,
            measurement_params.manifest_cov,
            initial_state.mean,
            initial_state.cov,
            obs_kernel.emission_fn,
            n_ieks_iters=self.n_ieks_iters,
        )

        return log_lik


# ---------------------------------------------------------------------------
# fit_laplace_em: outer loop for parameter estimation
# ---------------------------------------------------------------------------


def fit_laplace_em(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    n_ieks_iters: int = 5,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via Laplace-EM with tempered SMC outer loop.

    Uses the Laplace-approximated marginal likelihood (via IEKS) as the
    log-density for a tempered SMC sampler over the parameter space.

    If the model has an explicit likelihood override (e.g. likelihood="kalman"),
    that backend is used instead of the Laplace approximation.
    """
    if model.likelihood == "kalman":
        backend = model.make_likelihood_backend()
    else:
        backend = LaplaceLikelihood(
            n_latent=model.spec.n_latent,
            n_manifest=model.spec.n_manifest,
            manifest_dist=model.spec.manifest_dist,
            manifest_link=model.spec.manifest_link,
            n_ieks_iters=n_ieks_iters,
        )
    return run_tempered_smc(
        model,
        observations,
        times,
        n_outer=n_outer,
        n_csmc_particles=n_csmc_particles,
        n_mh_steps=n_mh_steps,
        param_step_size=param_step_size,
        n_warmup=n_warmup,
        target_accept=target_accept,
        seed=seed,
        adaptive_tempering=adaptive_tempering,
        target_ess_ratio=target_ess_ratio,
        waste_free=waste_free,
        n_leapfrog=n_leapfrog,
        method_name="laplace_em",
        likelihood_backend=backend,
        extra_diagnostics={"n_ieks_iters": n_ieks_iters},
        print_prefix="Laplace-EM",
    )
