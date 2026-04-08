"""Canonical emission log-probability functions for all noise families.

Each function computes log p(y_t | z_t) for a single time step given
the measurement model parameters (H, d, R) and an observation mask.

Used by: Laplace-EM, Structured VI, DPF, Rao-Blackwell PF, bootstrap PF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax.scipy.special
import jax.scipy.stats as jstats

from causal_ssm_agent.models.ssm.inference.targets.base import (
    CHOL_JITTER,
    ETA_CLIP_MIN,
    MISSING_DATA_LARGE_VAR,
    NUMERICAL_EPSILON,
    PROB_CLIP_MIN,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _logistic_pdf(x: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.sigmoid(x)
    return p * (1.0 - p)


def _logistic_pdf_prime(x: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.sigmoid(x)
    pdf = p * (1.0 - p)
    return pdf * (1.0 - 2.0 * p)


def _normalize_discrete_observation(
    y_t: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rounded = jnp.rint(y_t)
    y_idx = rounded.astype(jnp.int32)
    valid = (
        jnp.isfinite(y_t)
        & jnp.isclose(y_t, rounded, atol=1e-4)
        & (y_idx >= 0)
        & (y_idx < level_counts)
    )
    safe_idx = jnp.clip(y_idx, 0, jnp.maximum(level_counts - 1, 0))
    return safe_idx, valid


def _select_rowwise(values: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(values, indices[:, None], axis=1).squeeze(axis=1)


def _discrete_moments_from_probs(probs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    class_values = jnp.arange(probs.shape[1], dtype=probs.dtype)
    mean = jnp.sum(probs * class_values[None, :], axis=1)
    second_moment = jnp.sum(probs * (class_values[None, :] ** 2), axis=1)
    variance = jnp.maximum(second_moment - mean**2, NUMERICAL_EPSILON)
    return mean, variance


def ordered_logistic_probabilities(
    eta: jnp.ndarray,
    cutpoints: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-channel ordered-logistic probabilities over encoded categories."""
    eta = jnp.asarray(eta)
    cutpoints = jnp.asarray(cutpoints)
    level_counts = jnp.asarray(level_counts, dtype=jnp.int32)

    n_manifest = eta.shape[0]
    max_levels = cutpoints.shape[1] + 1
    boundary_idx = jnp.arange(1, max_levels)
    valid_boundaries = boundary_idx[None, :] < level_counts[:, None]

    cdf_mid = jax.nn.sigmoid(cutpoints - eta[:, None])
    cdf_mid = jnp.where(valid_boundaries, cdf_mid, 1.0)

    cdf = jnp.concatenate(
        [jnp.zeros((n_manifest, 1)), cdf_mid, jnp.ones((n_manifest, 1))],
        axis=1,
    )
    probs = jnp.diff(cdf, axis=1)

    class_mask = jnp.arange(max_levels)[None, :] < level_counts[:, None]
    probs = jnp.where(class_mask, jnp.maximum(probs, 0.0), 0.0)
    norm = jnp.sum(probs, axis=1, keepdims=True)
    fallback = jax.nn.one_hot(jnp.zeros(n_manifest, dtype=jnp.int32), max_levels)
    return jnp.where(
        norm > NUMERICAL_EPSILON,
        probs / jnp.maximum(norm, NUMERICAL_EPSILON),
        fallback,
    )


def categorical_probabilities(
    eta: jnp.ndarray,
    intercepts: jnp.ndarray,
    slopes: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-channel softmax probabilities over encoded categories."""
    eta = jnp.asarray(eta)
    intercepts = jnp.asarray(intercepts)
    slopes = jnp.asarray(slopes)
    level_counts = jnp.asarray(level_counts, dtype=jnp.int32)

    max_levels = intercepts.shape[1] + 1
    nonbaseline_mask = jnp.arange(1, max_levels)[None, :] < level_counts[:, None]
    logits_extra = intercepts + slopes * eta[:, None]
    logits_extra = jnp.where(nonbaseline_mask, logits_extra, -1e30)
    logits = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), logits_extra], axis=1)

    probs = jax.nn.softmax(logits, axis=1)
    class_mask = jnp.arange(max_levels)[None, :] < level_counts[:, None]
    probs = jnp.where(class_mask, probs, 0.0)
    norm = jnp.sum(probs, axis=1, keepdims=True)
    fallback = jax.nn.one_hot(jnp.zeros(eta.shape[0], dtype=jnp.int32), max_levels)
    return jnp.where(
        norm > NUMERICAL_EPSILON,
        probs / jnp.maximum(norm, NUMERICAL_EPSILON),
        fallback,
    )


def ordered_logistic_moments(
    eta: jnp.ndarray,
    cutpoints: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _discrete_moments_from_probs(
        ordered_logistic_probabilities(eta, cutpoints, level_counts)
    )


def categorical_moments(
    eta: jnp.ndarray,
    intercepts: jnp.ndarray,
    slopes: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _discrete_moments_from_probs(
        categorical_probabilities(eta, intercepts, slopes, level_counts)
    )


def get_ordered_logistic_extra_params(
    extra_params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)
    cutpoints = jnp.asarray(extra_params["obs_ordered_cutpoints"])
    return level_counts, cutpoints


def get_categorical_extra_params(
    extra_params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)
    intercepts = jnp.asarray(extra_params["obs_cat_intercepts"])
    slopes = jnp.asarray(extra_params["obs_cat_slopes"])
    return level_counts, intercepts, slopes


def emission_log_prob_gaussian(y_t, z_t, H, d, R, obs_mask_t):
    """Log p(y_t | z_t) for Gaussian emissions."""
    pred = H @ z_t + d
    residual = (y_t - pred) * obs_mask_t
    n_obs = jnp.sum(obs_mask_t)
    R_adj = R + jnp.diag((1.0 - obs_mask_t) * MISSING_DATA_LARGE_VAR)
    R_adj = 0.5 * (R_adj + R_adj.T) + jnp.eye(R.shape[0]) * CHOL_JITTER
    _, logdet = jnp.linalg.slogdet(R_adj)
    n_missing = y_t.shape[0] - n_obs
    logdet = logdet - n_missing * jnp.log(MISSING_DATA_LARGE_VAR)
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
    # Clamp y_t away from 0 so gamma.logpdf doesn't produce -inf/+inf from
    # log(0), which causes NaN gradients via JAX autodiff even when masked.
    safe_y = jnp.maximum(y_t, NUMERICAL_EPSILON)
    log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_bernoulli(y_t, z_t, H, d, _R, obs_mask_t):
    """Log p(y_t | z_t) for Bernoulli emissions (logit-link)."""
    eta = H @ z_t + d
    logit_p = eta
    log_probs = y_t * jax.nn.log_sigmoid(logit_p) + (1.0 - y_t) * jax.nn.log_sigmoid(-logit_p)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_negative_binomial(y_t, z_t, H, d, _R, obs_mask_t, r=5.0):
    """Log p(y_t | z_t) for Negative Binomial emissions (log-link).

    Parameterisation: mean = exp(eta), overdispersion r.
    Var = mu + mu^2/r.  As r -> inf this converges to Poisson.
    """
    eta = H @ z_t + d
    mu = jnp.exp(eta)
    log_probs = (
        jax.lax.lgamma(y_t + r)
        - jax.lax.lgamma(r)
        - jax.lax.lgamma(y_t + 1.0)
        + r * jnp.log(r / (r + mu))
        + y_t * jnp.log(mu / (r + mu) + NUMERICAL_EPSILON)
    )
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_beta(y_t, z_t, H, d, _R, obs_mask_t, concentration=10.0):
    """Log p(y_t | z_t) for Beta emissions (logit-link).

    mean = sigmoid(eta), concentration phi.
    alpha = mean * phi, beta_ = (1 - mean) * phi.
    """
    eta = H @ z_t + d
    mean = jax.nn.sigmoid(eta)
    alpha = mean * concentration
    beta_ = (1.0 - mean) * concentration
    # Clamp y_t into (0, 1) so beta.logpdf doesn't produce -inf from log(0)
    # or log(1-0), which causes NaN gradients via JAX autodiff even when masked.
    safe_y = jnp.clip(y_t, NUMERICAL_EPSILON, 1.0 - NUMERICAL_EPSILON)
    log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_ordered_logistic(
    y_t,
    z_t,
    H,
    d,
    _R,
    obs_mask_t,
    cutpoints,
    level_counts,
):
    """Log p(y_t | z_t) for ordered-logistic emissions."""
    eta = H @ z_t + d
    probs = ordered_logistic_probabilities(eta, cutpoints, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    chosen_probs = _select_rowwise(probs, y_idx)
    log_probs = jnp.where(valid_obs, jnp.log(jnp.maximum(chosen_probs, NUMERICAL_EPSILON)), -1e30)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_categorical(
    y_t,
    z_t,
    H,
    d,
    _R,
    obs_mask_t,
    intercepts,
    slopes,
    level_counts,
):
    """Log p(y_t | z_t) for categorical softmax emissions."""
    eta = H @ z_t + d
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    chosen_probs = _select_rowwise(probs, y_idx)
    log_probs = jnp.where(valid_obs, jnp.log(jnp.maximum(chosen_probs, NUMERICAL_EPSILON)), -1e30)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_bernoulli_probit(y_t, z_t, H, d, _R, obs_mask_t):
    """Log p(y_t | z_t) for Bernoulli emissions (probit-link).

    Uses the normal CDF (Phi) as the inverse link instead of sigmoid.
    """
    eta = H @ z_t + d
    p = jstats.norm.cdf(eta)
    p = jnp.clip(p, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    log_probs = y_t * jnp.log(p) + (1.0 - y_t) * jnp.log(1.0 - p)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_gamma_inverse(y_t, z_t, H, d, _R, obs_mask_t, shape=1.0):
    """Log p(y_t | z_t) for Gamma emissions (inverse-link for mean).

    mean = 1 / eta (canonical link for Gamma).
    """
    eta = H @ z_t + d
    mean = 1.0 / jnp.clip(eta, ETA_CLIP_MIN, None)
    scale = mean / shape
    safe_y = jnp.maximum(y_t, NUMERICAL_EPSILON)
    log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_beta_probit(y_t, z_t, H, d, _R, obs_mask_t, concentration=10.0):
    """Log p(y_t | z_t) for Beta emissions (probit-link).

    mean = Phi(eta), concentration phi.
    alpha = mean * phi, beta_ = (1 - mean) * phi.
    """
    eta = H @ z_t + d
    mean = jstats.norm.cdf(eta)
    mean = jnp.clip(mean, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = mean * concentration
    beta_ = (1.0 - mean) * concentration
    safe_y = jnp.clip(y_t, NUMERICAL_EPSILON, 1.0 - NUMERICAL_EPSILON)
    log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


# =============================================================================
# Analytical score (d log p/d eta_j) and neg-Hessian diag (-d^2 log p/d eta_j^2)
# for IEKS linearization. Eliminates jax.hessian from the inner loop.
# All functions: (y_t, eta, obs_mask_t) -> (g_eta, w_eta), shape (n_manifest,).
# =============================================================================


def _score_weight_poisson(y_t, eta, obs_mask_t):
    """Poisson (log link): score = y - lambda, neg-Hessian = lambda."""
    lam = jnp.exp(eta)
    return (y_t - lam) * obs_mask_t, lam * obs_mask_t


def _score_weight_bernoulli_logit(y_t, eta, obs_mask_t):
    """Bernoulli (logit link): score = y - p, neg-Hessian = p(1-p)."""
    p = jax.nn.sigmoid(eta)
    return (y_t - p) * obs_mask_t, (p * (1.0 - p)) * obs_mask_t


def _score_weight_beta_logit(y_t, eta, obs_mask_t, concentration):
    """Beta (logit link): exact score and neg-Hessian via digamma/polygamma."""
    phi = concentration
    mu = jax.nn.sigmoid(eta)
    mu_c = jnp.clip(mu, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu_c
    beta_ = phi * (1.0 - mu_c)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    sig_deriv = mu_c * (1.0 - mu_c)
    psi_diff = jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha)
    score_mu = phi * (logit_y + psi_diff)
    g = sig_deriv * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w_raw = phi * sig_deriv * (phi * sig_deriv * psi1_sum - (1.0 - 2.0 * mu_c) * score_mu)
    return g, jnp.maximum(w_raw, 0.0) * obs_mask_t


def _score_weight_gamma_log(y_t, eta, obs_mask_t, shape):
    """Gamma (log link): score = a*(y/mu - 1), neg-Hessian = a*y/mu."""
    mu = jnp.maximum(jnp.exp(eta), 1e-8)
    ratio = y_t / mu
    return shape * (ratio - 1.0) * obs_mask_t, shape * ratio * obs_mask_t


def _score_weight_gamma_inverse(y_t, eta, obs_mask_t, shape):
    """Gamma (inverse link): score = a*(mu - y), neg-Hessian = a*mu^2."""
    mu = 1.0 / jnp.clip(eta, ETA_CLIP_MIN, None)
    return shape * (mu - y_t) * obs_mask_t, (shape * mu**2) * obs_mask_t


def _score_weight_negative_binomial(y_t, eta, obs_mask_t, r):
    """Negative Binomial (log link): exact score and neg-Hessian."""
    mu = jnp.exp(eta)
    denom = r + mu
    g = r * (y_t - mu) / denom * obs_mask_t
    w = r * (r + y_t) * mu / (denom**2) * obs_mask_t
    return g, w


def _score_weight_ordered_logistic(y_t, eta, obs_mask_t, cutpoints, level_counts):
    """Ordered-logistic score and neg-Hessian w.r.t. the linear predictor."""
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    max_cutpoints = cutpoints.shape[1]
    lower_idx = jnp.clip(y_idx - 1, 0, max_cutpoints - 1)
    upper_idx = jnp.clip(y_idx, 0, max_cutpoints - 1)

    lower_arg = _select_rowwise(cutpoints, lower_idx) - eta
    upper_arg = _select_rowwise(cutpoints, upper_idx) - eta

    has_lower = y_idx > 0
    has_upper = y_idx < (level_counts - 1)

    lower_cdf = jnp.where(has_lower, jax.nn.sigmoid(lower_arg), 0.0)
    upper_cdf = jnp.where(has_upper, jax.nn.sigmoid(upper_arg), 1.0)
    lower_pdf = jnp.where(has_lower, _logistic_pdf(lower_arg), 0.0)
    upper_pdf = jnp.where(has_upper, _logistic_pdf(upper_arg), 0.0)
    lower_pdf_prime = jnp.where(has_lower, _logistic_pdf_prime(lower_arg), 0.0)
    upper_pdf_prime = jnp.where(has_upper, _logistic_pdf_prime(upper_arg), 0.0)

    prob = jnp.maximum(upper_cdf - lower_cdf, NUMERICAL_EPSILON)
    dprob = lower_pdf - upper_pdf
    d2prob = upper_pdf_prime - lower_pdf_prime

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = dprob / prob
    weight = jnp.maximum(score**2 - d2prob / prob, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_categorical(y_t, eta, obs_mask_t, intercepts, slopes, level_counts):
    """Categorical softmax score and neg-Hessian w.r.t. the linear predictor."""
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    slope_matrix = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), slopes], axis=1)
    chosen_slope = _select_rowwise(slope_matrix, y_idx)
    mean_slope = jnp.sum(probs * slope_matrix, axis=1)
    second_moment = jnp.sum(probs * (slope_matrix**2), axis=1)

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = chosen_slope - mean_slope
    weight = jnp.maximum(second_moment - mean_slope**2, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_bernoulli_probit(y_t, eta, obs_mask_t):
    """Bernoulli (probit link): exact score; Fisher information Hessian."""
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    var = mu * (1.0 - mu)
    return (y_t - mu) * phi_eta / var * obs_mask_t, (phi_eta**2 / var) * obs_mask_t


def _score_weight_beta_probit(y_t, eta, obs_mask_t, concentration):
    """Beta (probit link): exact score; Gauss-Newton Hessian."""
    phi = concentration
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu
    beta_ = phi * (1.0 - mu)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    score_mu = phi * (logit_y + jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha))
    g = phi_eta * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w = jnp.maximum(phi_eta**2 * phi**2 * psi1_sum, 0.0) * obs_mask_t
    return g, w


def get_emission_score_weight_fn(manifest_dist, extra_params=None, *, link=None):
    """Return analytical (score, neg_hess_diag) w.r.t. linear predictor η.

    Returns Callable(y_t, eta, obs_mask_t) → (g_eta, w_eta) of shape (n_manifest,),
    or None for gaussian/student_t which require special handling in kernels.py.
    """
    from causal_ssm_agent.models.ssm.inference.targets.observation_families import FAMILY_REGISTRY
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)
    family_spec = FAMILY_REGISTRY.get(dist)
    if family_spec is None:
        return None
    link_key = str(link) if link else "default"
    factory = family_spec.score_weight_fns.get(link_key) or family_spec.score_weight_fns.get(
        "default"
    )
    if factory is None:
        return None
    return factory(extra_params)


def get_emission_fn(manifest_dist, extra_params=None, *, link=None):
    """Return the appropriate emission log-prob function.

    Args:
        manifest_dist: Distribution family (DistributionFamily enum or string).
        extra_params: Optional dict with distribution-specific hyperparameters.
        link: Link function (LinkFunction enum or string, e.g. "logit", "probit",
            "inverse"). When None, uses the default link for the distribution.

    Returns:
        Callable(y_t, z_t, H, d, R, obs_mask_t) -> scalar log-prob.
    """
    from causal_ssm_agent.models.ssm.inference.targets.observation_families import FAMILY_REGISTRY
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    try:
        dist = DistributionFamily(manifest_dist)
    except ValueError as exc:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}'. "
            "Supported: gaussian, student_t, poisson, gamma, bernoulli, "
            "negative_binomial, beta, ordered_logistic, categorical."
        ) from exc
    family_spec = FAMILY_REGISTRY.get(dist)
    if family_spec is None:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}'. "
            f"Supported: gaussian, student_t, poisson, gamma, bernoulli, "
            "negative_binomial, beta, ordered_logistic, categorical."
        )
    link_key = str(link) if link else "default"
    factory = family_spec.emission_fns.get(link_key) or family_spec.emission_fns.get("default")
    if factory is None:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}', link='{link}'."
        )
    return factory(extra_params)


def get_mean_param_log_prob_fn(manifest_dist, extra_params=None):
    """Return log-prob(y | mean-parameter) for one observation vector.

    Unlike ``get_emission_fn()``, this operates directly on the expected mean /
    location in observation space. It is used for interval-summary measurement
    semantics where the mean is aggregated over a support window after applying
    the link function.
    """
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)

    def gaussian(y_t, mean_t, R, obs_mask_t):
        residual = (y_t - mean_t) * obs_mask_t
        n_obs = jnp.sum(obs_mask_t)
        R_adj = R + jnp.diag((1.0 - obs_mask_t) * MISSING_DATA_LARGE_VAR)
        R_adj = 0.5 * (R_adj + R_adj.T) + jnp.eye(R.shape[0]) * CHOL_JITTER
        _, logdet = jnp.linalg.slogdet(R_adj)
        n_missing = y_t.shape[0] - n_obs
        logdet = logdet - n_missing * jnp.log(MISSING_DATA_LARGE_VAR)
        mahal = residual @ jla.solve(R_adj, residual, assume_a="pos")
        return jnp.where(
            n_obs > 0,
            -0.5 * (n_obs * jnp.log(2 * jnp.pi) + logdet + mahal),
            0.0,
        )

    def student_t(y_t, mean_t, R, obs_mask_t):
        df = extra_params.get("obs_df", 5.0)
        scale = jnp.sqrt(jnp.diag(R))
        log_probs = jax.scipy.stats.t.logpdf(y_t, df, loc=mean_t, scale=scale)
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    def poisson(y_t, mean_t, _R, obs_mask_t):
        rate = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        log_probs = jax.scipy.stats.poisson.logpmf(y_t, rate)
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    def gamma(y_t, mean_t, _R, obs_mask_t):
        shape = extra_params.get("obs_shape", 1.0)
        safe_mean = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        scale = safe_mean / shape
        safe_y = jnp.maximum(y_t, NUMERICAL_EPSILON)
        log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    def bernoulli(y_t, mean_t, _R, obs_mask_t):
        p = jnp.clip(mean_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
        log_probs = y_t * jnp.log(p) + (1.0 - y_t) * jnp.log(1.0 - p)
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    def negative_binomial(y_t, mean_t, _R, obs_mask_t):
        r = extra_params.get("obs_r", 5.0)
        mu = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        log_probs = (
            jax.lax.lgamma(y_t + r)
            - jax.lax.lgamma(r)
            - jax.lax.lgamma(y_t + 1.0)
            + r * jnp.log(r / (r + mu))
            + y_t * jnp.log(mu / (r + mu) + NUMERICAL_EPSILON)
        )
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    def beta(y_t, mean_t, _R, obs_mask_t):
        concentration = extra_params.get("obs_concentration", 10.0)
        clipped_mean = jnp.clip(mean_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
        alpha = clipped_mean * concentration
        beta_ = (1.0 - clipped_mean) * concentration
        safe_y = jnp.clip(y_t, NUMERICAL_EPSILON, 1.0 - NUMERICAL_EPSILON)
        log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
        return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))

    mean_log_prob_fns = {
        DistributionFamily.GAUSSIAN: gaussian,
        DistributionFamily.STUDENT_T: student_t,
        DistributionFamily.POISSON: poisson,
        DistributionFamily.GAMMA: gamma,
        DistributionFamily.BERNOULLI: bernoulli,
        DistributionFamily.NEGATIVE_BINOMIAL: negative_binomial,
        DistributionFamily.BETA: beta,
    }
    if dist not in mean_log_prob_fns:
        raise ValueError(
            f"Mean-parameter log-prob is not defined for manifest_dist='{manifest_dist}'."
        )
    return mean_log_prob_fns[dist]


def get_mean_param_sample_fn(manifest_dist, extra_params=None):
    """Return a sampler operating directly in observation mean-parameter space."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)

    def gaussian(key, mean_t, R):
        R_adj = 0.5 * (R + R.T) + jnp.eye(R.shape[0], dtype=R.dtype) * CHOL_JITTER
        chol = jnp.linalg.cholesky(R_adj)
        return mean_t + chol @ jax.random.normal(key, mean_t.shape)

    def student_t(key, mean_t, R):
        df = extra_params.get("obs_df", 5.0)
        scale = jnp.sqrt(jnp.maximum(jnp.diag(R), NUMERICAL_EPSILON))
        key_num, key_den = jax.random.split(key)
        z = jax.random.normal(key_num, mean_t.shape)
        chi2 = 2.0 * jax.random.gamma(key_den, df / 2.0, shape=mean_t.shape)
        t_val = z * jnp.sqrt(df / jnp.maximum(chi2, NUMERICAL_EPSILON))
        return mean_t + scale * t_val

    def poisson(key, mean_t, _R):
        rate = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        return jax.random.poisson(key, rate).astype(jnp.float32)

    def gamma(key, mean_t, _R):
        shape = extra_params.get("obs_shape", 1.0)
        safe_mean = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        scale = safe_mean / jnp.maximum(shape, NUMERICAL_EPSILON)
        return jax.random.gamma(key, shape, shape=mean_t.shape) * scale

    def bernoulli(key, mean_t, _R):
        p = jnp.clip(mean_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
        return jax.random.bernoulli(key, p).astype(jnp.float32)

    def negative_binomial(key, mean_t, _R):
        r = extra_params.get("obs_r", 5.0)
        safe_mean = jnp.maximum(mean_t, NUMERICAL_EPSILON)
        key_gamma, key_poisson = jax.random.split(key)
        gamma_draw = (
            jax.random.gamma(key_gamma, r, shape=mean_t.shape) * safe_mean / jnp.maximum(r, 1e-8)
        )
        return jax.random.poisson(
            key_poisson,
            jnp.maximum(gamma_draw, NUMERICAL_EPSILON),
        ).astype(jnp.float32)

    def beta(key, mean_t, _R):
        concentration = extra_params.get("obs_concentration", 10.0)
        clipped_mean = jnp.clip(mean_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
        alpha = jnp.maximum(clipped_mean * concentration, 1e-4)
        beta_param = jnp.maximum((1.0 - clipped_mean) * concentration, 1e-4)
        key_alpha, key_beta = jax.random.split(key)
        gamma_alpha = jax.random.gamma(key_alpha, alpha)
        gamma_beta = jax.random.gamma(key_beta, beta_param)
        return gamma_alpha / jnp.maximum(gamma_alpha + gamma_beta, NUMERICAL_EPSILON)

    mean_sample_fns = {
        DistributionFamily.GAUSSIAN: gaussian,
        DistributionFamily.STUDENT_T: student_t,
        DistributionFamily.POISSON: poisson,
        DistributionFamily.GAMMA: gamma,
        DistributionFamily.BERNOULLI: bernoulli,
        DistributionFamily.NEGATIVE_BINOMIAL: negative_binomial,
        DistributionFamily.BETA: beta,
    }
    if dist not in mean_sample_fns:
        raise ValueError(
            f"Mean-parameter sampler is not defined for manifest_dist='{manifest_dist}'."
        )
    return mean_sample_fns[dist]


@dataclass(frozen=True)
class PredictiveObservationSampler:
    """Compiled predictive sampler shared by posterior/prior predictive paths."""

    sample_point_trajectory: Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
    sample_mean_trajectory: Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
    all_gaussian: bool
    manifest_dists: tuple[str, ...]


def _slice_per_channel_extra_params(
    extra_params: dict | None,
    ch_indices: list[int],
) -> dict | None:
    if extra_params is None:
        return None

    sliced: dict = {}
    idx = jnp.array(ch_indices, dtype=jnp.int32)
    for key, value in extra_params.items():
        if (
            hasattr(value, "ndim")
            and hasattr(value, "shape")
            and value.ndim >= 1
            and value.shape[0] == len(idx)
        ):
            sliced[key] = value
            continue
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim >= 1:
            try:
                if value.shape[0] >= len(ch_indices):
                    sliced[key] = value[idx]
                    continue
            except TypeError:
                pass
        sliced[key] = value
    return sliced


def build_predictive_observation_sampler(
    manifest_dists,
    manifest_cov: jnp.ndarray,
    *,
    manifest_links=None,
    extra_params: dict | None = None,
) -> PredictiveObservationSampler:
    """Compile predictive samplers for point observations and mean-space summaries."""
    from causal_ssm_agent.models.ssm.inference.targets.observation_families import (
        POSTERIOR_PREDICTIVE_SWITCH_BRANCHES,
        get_posterior_predictive_switch_index,
        resolve_manifest_families_and_links,
    )
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    dists, links = resolve_manifest_families_and_links(
        manifest_dists,
        manifest_links=manifest_links,
    )
    n_manifest = len(dists)
    all_gaussian = all(dist == DistributionFamily.GAUSSIAN for dist in dists)
    manifest_dist_values = tuple(dist.value for dist in dists)
    try:
        mean_sample_fn = build_composite_mean_sample_fn(manifest_dist_values, extra_params)
    except ValueError as exc:
        mean_sample_fn = None
        mean_sampler_error = exc
    else:
        mean_sampler_error = None

    def _sample_mean_vector(key, mean_t):
        if mean_sample_fn is None:
            raise ValueError(
                f"Mean-parameter sampler is not defined for manifest_dists={manifest_dist_values}."
            ) from mean_sampler_error
        return mean_sample_fn(key, mean_t, manifest_cov)

    def _sample_mean_trajectory(key, mean_trajectory):
        mean_keys = jax.random.split(key, mean_trajectory.shape[0])
        return jax.vmap(_sample_mean_vector)(mean_keys, mean_trajectory)

    if all_gaussian:
        manifest_cov_adj = (
            0.5 * (manifest_cov + manifest_cov.T)
            + jnp.eye(n_manifest, dtype=manifest_cov.dtype) * CHOL_JITTER
        )
        manifest_chol = jnp.linalg.cholesky(manifest_cov_adj)

        def _sample_point_vector(key, linear_predictor):
            return linear_predictor + manifest_chol @ jax.random.normal(key, linear_predictor.shape)

        def _sample_point_trajectory(key, linear_predictors):
            point_keys = jax.random.split(key, linear_predictors.shape[0])
            return jax.vmap(_sample_point_vector)(point_keys, linear_predictors)

        return PredictiveObservationSampler(
            sample_point_trajectory=_sample_point_trajectory,
            sample_mean_trajectory=_sample_mean_trajectory,
            all_gaussian=True,
            manifest_dists=manifest_dist_values,
        )

    dist_indices = jnp.asarray(
        [
            get_posterior_predictive_switch_index(dist, link=link)
            for dist, link in zip(dists, links, strict=False)
        ],
        dtype=jnp.int32,
    )
    manifest_std = jnp.sqrt(jnp.maximum(jnp.diag(manifest_cov), NUMERICAL_EPSILON))
    params = extra_params or {}
    level_counts = params.get("obs_level_counts")
    if level_counts is None:
        level_counts = jnp.ones((n_manifest,), dtype=jnp.int32)
    else:
        level_counts = jnp.asarray(level_counts, dtype=jnp.int32)
    ordered_cutpoints = params.get("obs_ordered_cutpoints")
    if ordered_cutpoints is None:
        ordered_cutpoints = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    cat_intercepts = params.get("obs_cat_intercepts")
    if cat_intercepts is None:
        cat_intercepts = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    cat_slopes = params.get("obs_cat_slopes")
    if cat_slopes is None:
        cat_slopes = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    obs_df = jnp.asarray(params.get("obs_df", 5.0), dtype=manifest_cov.dtype)
    obs_shape = jnp.asarray(params.get("obs_shape", 2.0), dtype=manifest_cov.dtype)
    obs_r = jnp.asarray(params.get("obs_r", 5.0), dtype=manifest_cov.dtype)
    obs_concentration = jnp.asarray(
        params.get("obs_concentration", 10.0),
        dtype=manifest_cov.dtype,
    )

    def _sample_channel(
        loc_j,
        key,
        dist_idx,
        std_j,
        df,
        shape_p,
        r_p,
        phi_p,
        level_count,
        cutpoints,
        cat_intercepts_j,
        cat_slopes_j,
    ):
        return jax.lax.switch(
            dist_idx,
            POSTERIOR_PREDICTIVE_SWITCH_BRANCHES,
            loc_j,
            key,
            std_j,
            df,
            shape_p,
            r_p,
            phi_p,
            level_count,
            cutpoints,
            cat_intercepts_j,
            cat_slopes_j,
        )

    def _sample_point_vector(key, linear_predictor):
        channel_keys = jax.random.split(key, n_manifest)
        return jax.vmap(_sample_channel)(
            linear_predictor,
            channel_keys,
            dist_indices,
            manifest_std,
            jnp.full((n_manifest,), obs_df),
            jnp.full((n_manifest,), obs_shape),
            jnp.full((n_manifest,), obs_r),
            jnp.full((n_manifest,), obs_concentration),
            level_counts,
            ordered_cutpoints,
            cat_intercepts,
            cat_slopes,
        )

    def _sample_point_trajectory(key, linear_predictors):
        point_keys = jax.random.split(key, linear_predictors.shape[0])
        return jax.vmap(_sample_point_vector)(point_keys, linear_predictors)

    return PredictiveObservationSampler(
        sample_point_trajectory=_sample_point_trajectory,
        sample_mean_trajectory=_sample_mean_trajectory,
        all_gaussian=False,
        manifest_dists=manifest_dist_values,
    )


def build_composite_mean_log_prob_fn(
    manifest_dists,
    extra_params: dict | None = None,
):
    """Build an observation-space log-prob for heterogeneous manifest families."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    dists = [DistributionFamily(dist) for dist in manifest_dists]
    if len(set(dists)) == 1:
        return get_mean_param_log_prob_fn(dists[0], extra_params)

    from collections import defaultdict

    groups: dict[DistributionFamily, list[int]] = defaultdict(list)
    for ch_idx, dist in enumerate(dists):
        groups[dist].append(ch_idx)

    group_fns: list[tuple[list[int], Callable]] = []
    for dist, ch_indices in groups.items():
        group_fns.append(
            (
                ch_indices,
                get_mean_param_log_prob_fn(
                    dist,
                    _slice_per_channel_extra_params(extra_params, ch_indices),
                ),
            )
        )

    def composite_mean_log_prob(y_t, mean_t, R, obs_mask_t):
        total_ll = 0.0
        for ch_indices, group_fn in group_fns:
            idx = jnp.array(ch_indices)
            y_g = y_t[idx]
            mean_g = mean_t[idx]
            R_g = R[jnp.ix_(idx, idx)]
            mask_g = obs_mask_t[idx]
            total_ll = total_ll + group_fn(y_g, mean_g, R_g, mask_g)
        return total_ll

    return composite_mean_log_prob


def build_composite_mean_sample_fn(
    manifest_dists,
    extra_params: dict | None = None,
):
    """Build an observation-space sampler for heterogeneous manifest families."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    dists = [DistributionFamily(dist) for dist in manifest_dists]
    if len(set(dists)) == 1:
        return get_mean_param_sample_fn(dists[0], extra_params)

    from collections import defaultdict

    groups: dict[DistributionFamily, list[int]] = defaultdict(list)
    for ch_idx, dist in enumerate(dists):
        groups[dist].append(ch_idx)

    group_fns: list[tuple[list[int], Callable]] = []
    for dist, ch_indices in groups.items():
        group_fns.append(
            (
                ch_indices,
                get_mean_param_sample_fn(
                    dist,
                    _slice_per_channel_extra_params(extra_params, ch_indices),
                ),
            )
        )

    def composite_mean_sample(key, mean_t, R):
        sampled = jnp.zeros_like(mean_t)
        keys = jax.random.split(key, len(group_fns))
        for subkey, (ch_indices, group_fn) in zip(keys, group_fns, strict=False):
            idx = jnp.array(ch_indices)
            sampled = sampled.at[idx].set(group_fn(subkey, mean_t[idx], R[jnp.ix_(idx, idx)]))
        return sampled

    return composite_mean_sample
