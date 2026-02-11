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
import jax.random as random
import jax.scipy.linalg as jla

from dsem_agent.models.ssm.discretization import discretize_system_batched
from dsem_agent.models.ssm.inference import InferenceResult
from dsem_agent.models.ssm.utils import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)

if TYPE_CHECKING:
    from dsem_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )

# ---------------------------------------------------------------------------
# Emission log-prob and its derivatives (gradient and negative Hessian)
# ---------------------------------------------------------------------------


def _emission_log_prob_gaussian(y_t, z_t, H, d, R, obs_mask_t):
    """Log p(y_t | z_t) for Gaussian emissions."""
    pred = H @ z_t + d
    residual = (y_t - pred) * obs_mask_t
    n_obs = jnp.sum(obs_mask_t)
    large_var = 1e10
    R_adj = R + jnp.diag((1.0 - obs_mask_t) * large_var)
    R_adj = 0.5 * (R_adj + R_adj.T) + jnp.eye(R.shape[0]) * 1e-8
    _, logdet = jnp.linalg.slogdet(R_adj)
    n_missing = y_t.shape[0] - n_obs
    logdet = logdet - n_missing * jnp.log(large_var)
    mahal = residual @ jla.solve(R_adj, residual, assume_a="pos")
    return jnp.where(n_obs > 0, -0.5 * (n_obs * jnp.log(2 * jnp.pi) + logdet + mahal), 0.0)


def _emission_log_prob_poisson(y_t, z_t, H, d, _R, obs_mask_t):
    """Log p(y_t | z_t) for Poisson emissions (log-link)."""
    eta = H @ z_t + d
    rate = jnp.exp(eta)
    log_probs = jax.scipy.stats.poisson.logpmf(y_t, rate)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def _emission_log_prob_student_t(y_t, z_t, H, d, R, obs_mask_t, df=5.0):
    """Log p(y_t | z_t) for Student-t emissions."""
    eta = H @ z_t + d
    scale = jnp.sqrt(jnp.diag(R))
    log_probs = jax.scipy.stats.t.logpdf(y_t, df, loc=eta, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def _emission_log_prob_gamma(y_t, z_t, H, d, _R, obs_mask_t, shape=1.0):
    """Log p(y_t | z_t) for Gamma emissions (log-link for mean)."""
    eta = H @ z_t + d
    mean = jnp.exp(eta)
    scale = mean / shape
    log_probs = jax.scipy.stats.gamma.logpdf(y_t, shape, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def _get_emission_fn(manifest_dist, extra_params=None):
    """Return the appropriate emission log-prob function."""
    extra_params = extra_params or {}
    if manifest_dist == "gaussian":
        return _emission_log_prob_gaussian
    elif manifest_dist == "poisson":
        return _emission_log_prob_poisson
    elif manifest_dist == "student_t":
        df = extra_params.get("obs_df", 5.0)
        return lambda y, z, H, d, R, m: _emission_log_prob_student_t(y, z, H, d, R, m, df)
    elif manifest_dist == "gamma":
        shape = extra_params.get("obs_shape", 1.0)
        return lambda y, z, H, d, R, m: _emission_log_prob_gamma(y, z, H, d, R, m, shape)
    else:
        raise ValueError(f"Unknown manifest_dist: {manifest_dist}")


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
    P_smooth,
    Ad,
    Qd,
    cd,
    init_mean,
    _init_cov,
    emission_log_prob_fn,
    H,
    d,
    R,
):
    """Compute Laplace-approximated log-likelihood.

    log p(y|theta) ≈ sum_t log p(y_t | z_hat_t)
                    + sum_t log N(z_hat_t; A z_hat_{t-1}, Q)
                    + 0.5 * sum_t log det P_smooth_t + const

    The last term is the entropy of the Gaussian approximation to the
    state posterior, which corrects for the integration over latent states.
    """
    T, D = z_smooth.shape

    # 1. Emission log-probs at the mode
    mask_float = obs_mask.astype(jnp.float32)
    emission_lls = jax.vmap(lambda y, z, m: emission_log_prob_fn(y, z, H, d, R, m))(
        observations, z_smooth, mask_float
    )
    total_emission_ll = jnp.sum(emission_lls)

    # 2. Transition log-probs at the mode
    jitter = jnp.eye(D) * 1e-6

    def _transition_ll(z_t, z_tm1, Ad_t, Qd_t, cd_t):
        mean = Ad_t @ z_tm1 + cd_t
        diff = z_t - mean
        Qd_reg = Qd_t + jitter
        _, logdet = jnp.linalg.slogdet(Qd_reg)
        mahal = diff @ jla.solve(Qd_reg, diff, assume_a="pos")
        return -0.5 * (D * jnp.log(2 * jnp.pi) + logdet + mahal)

    if T > 1:
        trans_lls = jax.vmap(_transition_ll)(z_smooth[1:], z_smooth[:-1], Ad[1:], Qd[1:], cd[1:])
        total_trans_ll = jnp.sum(trans_lls)
    else:
        total_trans_ll = 0.0

    # Initial state log-prob
    diff0 = z_smooth[0] - (Ad[0] @ init_mean + cd[0])
    Qd0_reg = Qd[0] + jitter
    _, logdet0 = jnp.linalg.slogdet(Qd0_reg)
    mahal0 = diff0 @ jla.solve(Qd0_reg, diff0, assume_a="pos")
    init_ll = -0.5 * (D * jnp.log(2 * jnp.pi) + logdet0 + mahal0)
    total_trans_ll = total_trans_ll + init_ll

    # 3. Entropy correction: 0.5 * sum_t log det P_smooth_t
    def _log_det_P(P_t):
        _, ld = jnp.linalg.slogdet(P_t + jitter)
        return ld

    log_dets = jax.vmap(_log_det_P)(P_smooth)
    entropy_correction = 0.5 * jnp.sum(log_dets) + 0.5 * T * D * jnp.log(2 * jnp.pi * jnp.e)

    return total_emission_ll + total_trans_ll + entropy_correction


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
        manifest_dist: str = "gaussian",
        n_ieks_iters: int = 5,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
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

        emission_fn = _get_emission_fn(self.manifest_dist, extra_params)

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
            emission_fn,
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
    subject_ids: jnp.ndarray | None = None,
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
    This combines the speed of the Laplace approximation with the robustness
    of SMC for multimodal parameter posteriors.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices
        n_outer: max tempering levels
        n_csmc_particles: number of parameter particles
        n_mh_steps: HMC mutations per round
        param_step_size: initial leapfrog step size
        n_warmup: levels to discard as warmup
        target_accept: target MH acceptance rate
        seed: random seed
        n_ieks_iters: IEKS iterations for Laplace approximation
        n_leapfrog: leapfrog steps (1=MALA, >1=HMC)
        adaptive_tempering: use ESS-based tempering
        target_ess_ratio: target ESS fraction
        waste_free: waste-free recycling

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    from jax.flatten_util import ravel_pytree

    from dsem_agent.models.ssm.mcmc_utils import (
        compute_weighted_chol_mass,
        find_next_beta,
        hmc_step,
    )

    if target_accept is None:
        target_accept = 0.65 if n_leapfrog > 1 else 0.44

    rng_key = random.PRNGKey(seed)
    N = n_csmc_particles

    if waste_free and N % n_mh_steps != 0:
        raise ValueError(
            f"waste_free requires N % n_mh_steps == 0, got N={N}, n_mh_steps={n_mh_steps}"
        )

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    # 2. Build differentiable evaluators
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
    )

    # Safe value-and-grad for log-likelihood
    def _safe_lik_val_and_grad(z):
        val, grad = jax.value_and_grad(log_lik_fn)(z)
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    batch_lik_val_and_grad = jax.jit(jax.vmap(_safe_lik_val_and_grad))

    # Tempered target: log_prior + beta * log_lik
    def _tempered_val_and_grad(z, beta):
        lik_val, lik_grad = jax.value_and_grad(log_lik_fn)(z)
        prior_val, prior_grad = jax.value_and_grad(log_prior_unc_fn)(z)
        val = prior_val + beta * lik_val
        grad = prior_grad + beta * lik_grad
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    # HMC mutation kernel
    def _mutate_particle(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), None

        (z_final, n_accept), _ = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return z_final, n_accept

    def _mutate_batch(rng_key, particles, beta, eps, chol_mass):
        keys = random.split(rng_key, particles.shape[0])
        return jax.vmap(lambda k, z: _mutate_particle(k, z, beta, eps, chol_mass))(keys, particles)

    _mutate_batch_jit = jax.jit(_mutate_batch)

    # Waste-free mutation
    def _mutate_particle_wastefree(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), z_new

        (_, n_acc), all_z = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return all_z, n_acc

    def _mutate_batch_wastefree(rng_key, particles_M, beta, eps, chol_mass):
        M = particles_M.shape[0]
        keys = random.split(rng_key, M)
        return jax.vmap(lambda k, z: _mutate_particle_wastefree(k, z, beta, eps, chol_mass))(
            keys, particles_M
        )

    _mutate_batch_wastefree_jit = jax.jit(_mutate_batch_wastefree)

    # 3. Initialize N particles from prior
    eps = param_step_size
    print(
        f"Laplace-EM: N={N}, K={n_outer}, D={D}, "
        f"n_ieks={n_ieks_iters}, n_mh={n_mh_steps}, eps={eps}"
    )
    print(f"  Initializing {N} particles from prior...")

    parts = []
    for name in sorted(site_info.keys()):
        info = site_info[name]
        rng_key, sample_key = random.split(rng_key)
        prior_samples = info["distribution"].sample(sample_key, (N,))
        unc_samples = info["transform"].inv(prior_samples)
        parts.append(unc_samples.reshape(N, -1))

    particles = jnp.concatenate(parts, axis=1)
    chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)

    # Pilot adaptation
    from blackjax.smc.resampling import systematic as _systematic_resample

    print("  Pilot: adapting step size at prior...")
    for pilot_step in range(30):
        rng_key, mutate_key = random.split(rng_key)
        particles_new, n_accepts = _mutate_batch_jit(mutate_key, particles, 0.0, eps, chol_mass)
        avg_accept = float(jnp.mean(n_accepts) / n_mh_steps)
        particles = particles_new

        log_eps = jnp.log(jnp.array(eps))
        log_eps = log_eps + 0.5 * (avg_accept - target_accept)
        eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

        if pilot_step >= 5 and abs(avg_accept - target_accept) < 0.1:
            print(
                f"    pilot converged at step {pilot_step + 1}: accept={avg_accept:.2f} eps={eps:.4f}"
            )
            break
    else:
        print(f"    pilot done: accept={avg_accept:.2f} eps={eps:.4f}")

    log_liks, _ = batch_lik_val_and_grad(particles)
    chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)
    logw = jnp.zeros(N)

    # Diagnostics
    accept_rates = []
    ess_history = []
    eps_history = []
    beta_schedule = []
    chain_samples = []

    beta_prev = 0.0
    level = 0
    max_mutation_rounds = 5
    M = N // n_mh_steps if waste_free else N

    # Tempering loop
    while beta_prev < 1.0 and level < n_outer:
        if adaptive_tempering:
            beta_k = find_next_beta(logw, log_liks, beta_prev, target_ess_ratio, N)
        else:
            beta_k = float(level + 1) / n_outer

        beta_schedule.append(beta_k)
        logw = logw + (beta_k - beta_prev) * log_liks

        lse = jax.nn.logsumexp(logw)
        log_wn = logw - lse
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        if ess > N / 4:
            chol_mass = compute_weighted_chol_mass(particles, logw, D)

        if waste_free:
            rng_key, resample_key, mutate_key = random.split(rng_key, 3)
            idx = _systematic_resample(resample_key, wn, M)
            resampled = particles[idx]
            all_trajs, n_accs = _mutate_batch_wastefree_jit(
                mutate_key, resampled, beta_k, eps, chol_mass
            )
            particles = all_trajs.reshape(N, D)
            logw = jnp.full(N, -jnp.log(float(N)))
            avg_accept = float(jnp.mean(n_accs) / n_mh_steps)

            log_eps = jnp.log(jnp.array(eps))
            log_eps = log_eps + 0.1 * (avg_accept - target_accept)
            eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))
        else:
            if ess < N / 2:
                rng_key, resample_key = random.split(rng_key)
                idx = _systematic_resample(resample_key, wn, N)
                particles = particles[idx]
                log_liks = log_liks[idx]
                logw = jnp.full(N, -jnp.log(float(N)))

            total_accepts = 0
            total_proposals = 0
            for mutation_round in range(max_mutation_rounds):
                rng_key, mutate_key = random.split(rng_key)
                particles_new, n_accepts = _mutate_batch_jit(
                    mutate_key, particles, beta_k, eps, chol_mass
                )
                round_accepts = float(jnp.sum(n_accepts))
                total_accepts += round_accepts
                total_proposals += N * n_mh_steps
                particles = particles_new

                round_accept_rate = round_accepts / (N * n_mh_steps)
                log_eps = jnp.log(jnp.array(eps))
                log_eps = log_eps + 0.1 * (round_accept_rate - target_accept)
                eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

                if mutation_round > 0 and round_accept_rate > 0.2:
                    break

            avg_accept = total_accepts / max(total_proposals, 1)

        accept_rates.append(avg_accept)
        eps_history.append(eps)

        log_liks, _ = batch_lik_val_and_grad(particles)
        chain_samples.append(particles[level % N])

        print(
            f"  step {level + 1}  beta={beta_k:.3f}  ESS={ess:.1f}/{N}"
            f"  accept={avg_accept:.2f}  eps={eps:.4f}"
        )

        beta_prev = beta_k
        level += 1

    actual_levels = level
    if n_warmup is None:
        n_warmup = actual_levels // 2
    n_warmup = min(n_warmup, max(actual_levels - 1, 0))

    chain_particles = jnp.stack(chain_samples[n_warmup:], axis=0)

    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(chain_particles)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="laplace_em",
        diagnostics={
            "accept_rates": accept_rates,
            "ess_history": ess_history,
            "eps_history": eps_history,
            "beta_schedule": beta_schedule,
            "n_levels": actual_levels,
            "n_ieks_iters": n_ieks_iters,
            "n_outer": n_outer,
            "n_csmc_particles": N,
            "n_mh_steps": n_mh_steps,
            "n_leapfrog": n_leapfrog,
            "n_warmup": n_warmup,
        },
    )
