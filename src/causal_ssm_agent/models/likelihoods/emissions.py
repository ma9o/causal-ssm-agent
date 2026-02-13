"""Canonical emission log-probability functions for all noise families.

Each function computes log p(y_t | z_t) for a single time step given
the measurement model parameters (H, d, R) and an observation mask.

Used by: Laplace-EM, Structured VI, DPF, Rao-Blackwell PF, bootstrap PF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


def emission_log_prob_gaussian(y_t, z_t, H, d, R, obs_mask_t):
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


def emission_log_prob_poisson(y_t, z_t, H, d, _R, obs_mask_t):
    """Log p(y_t | z_t) for Poisson emissions (log-link)."""
    eta = H @ z_t + d
    rate = jnp.exp(eta)
    log_probs = jax.scipy.stats.poisson.logpmf(y_t, rate)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_student_t(y_t, z_t, H, d, R, obs_mask_t, df=5.0):
    """Log p(y_t | z_t) for Student-t emissions."""
    eta = H @ z_t + d
    scale = jnp.sqrt(jnp.diag(R))
    log_probs = jax.scipy.stats.t.logpdf(y_t, df, loc=eta, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_gamma(y_t, z_t, H, d, _R, obs_mask_t, shape=1.0):
    """Log p(y_t | z_t) for Gamma emissions (log-link for mean)."""
    eta = H @ z_t + d
    mean = jnp.exp(eta)
    scale = mean / shape
    log_probs = jax.scipy.stats.gamma.logpdf(y_t, shape, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def get_emission_fn(manifest_dist, extra_params=None):
    """Return the appropriate emission log-prob function.

    Args:
        manifest_dist: One of "gaussian", "poisson", "student_t", "gamma".
        extra_params: Optional dict with "obs_df" (student_t) or "obs_shape" (gamma).

    Returns:
        Callable(y_t, z_t, H, d, R, obs_mask_t) -> scalar log-prob.
    """
    extra_params = extra_params or {}
    if manifest_dist == "gaussian":
        return emission_log_prob_gaussian
    elif manifest_dist == "poisson":
        return emission_log_prob_poisson
    elif manifest_dist == "student_t":
        df = extra_params.get("obs_df", 5.0)
        return lambda y, z, H, d, R, m: emission_log_prob_student_t(y, z, H, d, R, m, df)
    elif manifest_dist == "gamma":
        shape = extra_params.get("obs_shape", 1.0)
        return lambda y, z, H, d, R, m: emission_log_prob_gamma(y, z, H, d, R, m, shape)
    else:
        raise ValueError(f"Unknown manifest_dist: {manifest_dist}")
