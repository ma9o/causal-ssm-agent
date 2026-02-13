"""Shared MCMC utilities for tempered SMC and PGAS samplers.

Provides:
  - hmc_step: Generalized HMC/MALA step with full mass matrix preconditioning.
    When n_leapfrog=1, identical to preconditioned MALA.
  - compute_weighted_chol_mass: Weighted precision Cholesky from particle cloud.
  - find_next_beta: ESS-based bisection for adaptive tempering schedules.
  - DualAveragingState / dual_averaging_*: Nesterov dual averaging for step size
    adaptation (Hoffman & Gelman 2014).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla


def hmc_step(rng_key, z, log_target_val_and_grad, step_size, chol_mass, n_leapfrog=1):
    """Preconditioned HMC step with full mass matrix M = L L^T.

    When n_leapfrog=1, this is exactly preconditioned MALA (1-step leapfrog).
    When n_leapfrog>1, runs L-step leapfrog with MH correction.

    Args:
        rng_key: PRNG key
        z: current position (D,)
        log_target_val_and_grad: fn(z) -> (scalar, (D,))
        step_size: leapfrog epsilon (scalar JAX value)
        chol_mass: Cholesky factor of mass matrix (D, D), lower triangular
        n_leapfrog: number of leapfrog steps (1 = MALA)

    Returns:
        z_new: accepted position (D,)
        accepted: bool scalar
        log_target_new: log target at accepted position
    """
    noise_key, accept_key = random.split(rng_key)
    D = z.shape[0]

    # Current value and gradient
    log_pi, grad = log_target_val_and_grad(z)
    grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # Sample momentum: p ~ N(0, M) where M = L L^T
    u = random.normal(noise_key, (D,))
    p = chol_mass @ u

    # Half-step momentum
    p = p + 0.5 * step_size * grad

    # Leapfrog integration: L-1 interior steps (no-op when L=1)
    def leapfrog_body(carry, _):
        z_i, p_i = carry
        # Full position step
        z_i = z_i + step_size * jla.cho_solve((chol_mass, True), p_i)
        # Evaluate gradient at new position
        _, grad_i = log_target_val_and_grad(z_i)
        grad_i = jnp.nan_to_num(grad_i, nan=0.0, posinf=0.0, neginf=0.0)
        # Full momentum step
        p_i = p_i + step_size * grad_i
        return (z_i, p_i), None

    (z_prop, p_prop), _ = jax.lax.scan(leapfrog_body, (z, p), jnp.arange(n_leapfrog - 1))

    # Final position step
    z_prop = z_prop + step_size * jla.cho_solve((chol_mass, True), p_prop)

    # Final gradient and half-step momentum
    log_pi_prop, grad_prop = log_target_val_and_grad(z_prop)
    grad_prop = jnp.nan_to_num(grad_prop, nan=0.0, posinf=0.0, neginf=0.0)
    p_prop = p_prop + 0.5 * step_size * grad_prop

    # Kinetic energy: 0.5 * p^T M^{-1} p = 0.5 * ||L^{-1} p||^2
    p_init = chol_mass @ u + 0.5 * step_size * grad  # reconstruct initial momentum
    Linv_p_init = jla.solve_triangular(chol_mass, p_init, lower=True)
    Linv_p_prop = jla.solve_triangular(chol_mass, p_prop, lower=True)
    kinetic_old = 0.5 * jnp.dot(Linv_p_init, Linv_p_init)
    kinetic_new = 0.5 * jnp.dot(Linv_p_prop, Linv_p_prop)

    log_alpha = (log_pi_prop - kinetic_new) - (log_pi - kinetic_old)
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)

    accept_u = random.uniform(accept_key)
    accepted = jnp.log(accept_u) < log_alpha

    z_new = jnp.where(accepted, z_prop, z)
    log_target_new = jnp.where(accepted, log_pi_prop, log_pi)

    return z_new, accepted, log_target_new


def compute_weighted_chol_mass(particles, logw, D, reg=1e-3):
    """Compute Cholesky of precision (= inverse covariance) for HMC mass matrix.

    With M = precision, the leapfrog dynamics are isotropic in posterior-
    standardized space: noise becomes eps * N(0, I). This matches the
    Stan/NUTS convention.

    Args:
        particles: (N, D) particle positions
        logw: (N,) log-weights
        D: dimensionality
        reg: regularization for covariance matrix

    Returns:
        chol_mass: (D, D) lower-triangular Cholesky factor of precision
    """
    wn = jnp.exp(logw - jax.nn.logsumexp(logw))
    mean = jnp.sum(wn[:, None] * particles, axis=0)
    centered = particles - mean
    cov = (centered * wn[:, None]).T @ centered
    cov_reg = cov + reg * jnp.eye(D)
    # Compute precision = cov^{-1} via Cholesky solve
    L_cov = jla.cholesky(cov_reg, lower=True)
    prec = jla.cho_solve((L_cov, True), jnp.eye(D))
    return jla.cholesky(prec, lower=True)


def find_next_beta(logw, log_liks, beta_prev, target_ess_ratio, N):
    """Find next tempering beta via bisection on ESS target.

    Finds delta_beta such that ESS(w * lik^delta) = target * N.
    Returns min(beta_prev + delta, 1.0).

    Args:
        logw: (N,) current log-weights
        log_liks: (N,) log-likelihoods at current particles
        beta_prev: current beta value
        target_ess_ratio: target ESS as fraction of N (e.g. 0.5)
        N: number of particles

    Returns:
        beta_next: next tempering beta (float)
    """
    target_ess = target_ess_ratio * N

    def _compute_ess(delta_beta):
        logw_new = logw + delta_beta * log_liks
        lse = jax.nn.logsumexp(logw_new)
        log_wn = logw_new - lse
        wn = jnp.exp(log_wn)
        return 1.0 / jnp.sum(wn**2)

    # Check if jumping to beta=1.0 still keeps ESS above target
    delta_max = 1.0 - beta_prev
    ess_at_one = _compute_ess(delta_max)
    if float(ess_at_one) >= target_ess:
        return 1.0

    # Bisection: find delta_beta in [0, delta_max] where ESS = target
    lo, hi = 0.0, delta_max
    for _ in range(50):
        mid = (lo + hi) / 2.0
        ess_mid = float(_compute_ess(mid))
        if ess_mid > target_ess:
            lo = mid
        else:
            hi = mid

    beta_next = beta_prev + lo
    return min(beta_next, 1.0)


# ---------------------------------------------------------------------------
# Dual averaging for step size adaptation (Hoffman & Gelman 2014, Algorithm 5)
# ---------------------------------------------------------------------------


class DualAveragingState:
    """State for Nesterov dual averaging step size adaptation.

    Tracks running statistics to converge to the optimal step size that
    achieves a target MH acceptance rate. After warmup, use `eps_bar`
    (the averaged step size) for stable sampling.

    Attributes:
        log_eps: current (non-averaged) log step size
        log_eps_bar: averaged log step size (use after warmup)
        h_bar: running average of (target - accept_prob)
        step: number of updates so far
        mu: shrinkage target = log(10 * eps_init)
    """

    __slots__ = ("h_bar", "log_eps", "log_eps_bar", "mu", "step")

    def __init__(self, log_eps, log_eps_bar, h_bar, step, mu):
        self.log_eps = log_eps
        self.log_eps_bar = log_eps_bar
        self.h_bar = h_bar
        self.step = step
        self.mu = mu

    @property
    def eps(self):
        """Current step size (use during warmup)."""
        import math

        return math.exp(self.log_eps)

    @property
    def eps_bar(self):
        """Averaged step size (use after warmup)."""
        import math

        return math.exp(self.log_eps_bar)


def dual_averaging_init(eps_init):
    """Initialize dual averaging state.

    Args:
        eps_init: initial step size guess

    Returns:
        DualAveragingState
    """
    import math

    return DualAveragingState(
        log_eps=math.log(eps_init),
        log_eps_bar=0.0,
        h_bar=0.0,
        step=0,
        mu=math.log(10.0 * eps_init),
    )


def dual_averaging_update(state, accept_prob, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75):
    """Update dual averaging state with new acceptance probability.

    Implements Algorithm 5 from Hoffman & Gelman (2014). Converges to
    the step size achieving `target_accept` acceptance rate.

    Args:
        state: current DualAveragingState
        accept_prob: observed MH acceptance probability (0-1)
        target_accept: target acceptance rate (0.65 for HMC, 0.574 for MALA)
        gamma: adaptation regularization (default 0.05)
        t0: early stabilization offset (default 10)
        kappa: forgetting rate for step size averaging (default 0.75)

    Returns:
        updated DualAveragingState
    """
    import math

    m = state.step + 1
    w = 1.0 / (m + t0)
    h_bar = (1.0 - w) * state.h_bar + w * (target_accept - accept_prob)
    log_eps = state.mu - math.sqrt(m) / gamma * h_bar
    m_kappa = m ** (-kappa)
    log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * state.log_eps_bar

    return DualAveragingState(
        log_eps=log_eps,
        log_eps_bar=log_eps_bar,
        h_bar=h_bar,
        step=m,
        mu=state.mu,
    )
