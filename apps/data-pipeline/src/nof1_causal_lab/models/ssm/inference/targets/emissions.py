"""Canonical emission log-probability functions for all noise families.

Each function computes log p(y_t | z_t) for a single time step given
the measurement model parameters (H, d, R) and an observation mask.

Used by: MAP and blocked MCMC.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax.scipy.special
import jax.scipy.stats as jstats

from nof1_causal_lab.models.ssm.covariance_utils import (
    inflate_missing_variance,
    symmetrize_with_jitter,
)
from nof1_causal_lab.models.ssm.inference.targets.base import (
    MISSING_DATA_LARGE_VAR,
    NUMERICAL_EPSILON,
    PROB_CLIP_MIN,
)
from nof1_causal_lab.models.ssm.shapes import Array, Bool, Float, FloatScalar, Int, Shaped

if TYPE_CHECKING:
    from collections.abc import Callable


def _logistic_pdf(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    return p * (1.0 - p)


def _logistic_pdf_prime(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    pdf = p * (1.0 - p)
    return pdf * (1.0 - 2.0 * p)


def _normalize_discrete_observation(
    y_t: Float[Array, " M"],
    level_counts: Int[Array, " M"],
) -> tuple[Int[Array, " M"], Bool[Array, " M"]]:
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


def _select_rowwise(values: Float[Array, "M C"], indices: Int[Array, " M"]) -> Float[Array, " M"]:
    return jnp.take_along_axis(values, indices[:, None], axis=1).squeeze(axis=1)


def _sum_masked_log_probs(
    log_probs: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    *,
    valid_obs: Bool[Array, " M"] | None = None,
) -> FloatScalar:
    """Sum observed-channel log-probs, returning ``-inf`` on support violations."""
    observed = obs_mask_t > 0.5
    valid = jnp.ones_like(observed, dtype=bool) if valid_obs is None else valid_obs
    invalid_observed = observed & ~valid
    total = jnp.sum(jnp.where(observed & valid, log_probs, 0.0))
    return jnp.where(jnp.any(invalid_observed), -jnp.inf, total)


def _discrete_moments_from_probs(
    probs: Float[Array, "M C"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    class_values = jnp.arange(probs.shape[1], dtype=probs.dtype)
    mean = jnp.sum(probs * class_values[None, :], axis=1)
    second_moment = jnp.sum(probs * (class_values[None, :] ** 2), axis=1)
    variance = jnp.maximum(second_moment - mean**2, NUMERICAL_EPSILON)
    return mean, variance


def ordered_logistic_probabilities(
    eta: Float[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> Float[Array, "M C"]:
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
    eta: Float[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> Float[Array, "M C"]:
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
    eta: Float[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    return _discrete_moments_from_probs(
        ordered_logistic_probabilities(eta, cutpoints, level_counts)
    )


def categorical_moments(
    eta: Float[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
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


def emission_log_prob_gaussian(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for Gaussian emissions."""
    pred = H @ z_t + d
    residual = (y_t - pred) * obs_mask_t
    n_obs = jnp.sum(obs_mask_t)
    R_adj = symmetrize_with_jitter(inflate_missing_variance(R, obs_mask_t))
    # One Cholesky yields both the log-det (2·Σlog diag) and the whitened solve, vs the
    # prior slogdet(LU) + solve(pos) which factored R_adj twice (and the LU storm).
    chol = jnp.linalg.cholesky(R_adj)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    n_missing = y_t.shape[0] - n_obs
    logdet = logdet - n_missing * jnp.log(MISSING_DATA_LARGE_VAR)
    whitened = jla.solve_triangular(chol, residual, lower=True)
    mahal = jnp.sum(whitened * whitened)
    return jnp.where(n_obs > 0, -0.5 * (n_obs * jnp.log(2 * jnp.pi) + logdet + mahal), 0.0)


def emission_log_prob_poisson(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for Poisson emissions (log-link)."""
    eta = H @ z_t + d
    rate = jnp.exp(eta)
    log_probs = jax.scipy.stats.poisson.logpmf(y_t, rate)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_student_t(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    df=5.0,
) -> FloatScalar:
    """Log p(y_t | z_t) for Student-t emissions."""
    eta = H @ z_t + d
    scale = jnp.sqrt(jnp.diag(R))
    log_probs = jax.scipy.stats.t.logpdf(y_t, df, loc=eta, scale=scale)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_gamma(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    shape=1.0,
) -> FloatScalar:
    """Log p(y_t | z_t) for Gamma emissions (log-link for mean)."""
    eta = H @ z_t + d
    mean = jnp.exp(eta)
    scale = mean / shape
    valid_y = jnp.isfinite(y_t) & (y_t > 0.0)
    safe_y = jnp.where(valid_y, y_t, 1.0)
    log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
    return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_y)


def emission_log_prob_bernoulli(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for Bernoulli emissions (logit-link)."""
    eta = H @ z_t + d
    logit_p = eta
    log_probs = y_t * jax.nn.log_sigmoid(logit_p) + (1.0 - y_t) * jax.nn.log_sigmoid(-logit_p)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_negative_binomial(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    r=5.0,
) -> FloatScalar:
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


def emission_log_prob_beta(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration=10.0,
) -> FloatScalar:
    """Log p(y_t | z_t) for Beta emissions (logit-link).

    mean = sigmoid(eta), concentration phi.
    alpha = mean * phi, beta_ = (1 - mean) * phi.
    """
    eta = H @ z_t + d
    mean = jax.nn.sigmoid(eta)
    alpha = mean * concentration
    beta_ = (1.0 - mean) * concentration
    valid_y = jnp.isfinite(y_t) & (y_t > 0.0) & (y_t < 1.0)
    safe_y = jnp.where(valid_y, y_t, 0.5)
    log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
    return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_y)


def emission_log_prob_ordered_logistic(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for ordered-logistic emissions."""
    eta = H @ z_t + d
    probs = ordered_logistic_probabilities(eta, cutpoints, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    chosen_probs = _select_rowwise(probs, y_idx)
    log_probs = jnp.where(valid_obs, jnp.log(jnp.maximum(chosen_probs, NUMERICAL_EPSILON)), -1e30)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_categorical(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for categorical softmax emissions."""
    eta = H @ z_t + d
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    chosen_probs = _select_rowwise(probs, y_idx)
    log_probs = jnp.where(valid_obs, jnp.log(jnp.maximum(chosen_probs, NUMERICAL_EPSILON)), -1e30)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_bernoulli_probit(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | z_t) for Bernoulli emissions (probit-link).

    Uses the normal CDF (Phi) as the inverse link instead of sigmoid.
    """
    eta = H @ z_t + d
    p = jstats.norm.cdf(eta)
    p = jnp.clip(p, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    log_probs = y_t * jnp.log(p) + (1.0 - y_t) * jnp.log(1.0 - p)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_gamma_inverse(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    shape=1.0,
) -> FloatScalar:
    """Log p(y_t | z_t) for Gamma emissions (inverse-link for mean).

    mean = 1 / eta (canonical link for Gamma).
    """
    eta = H @ z_t + d
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    safe_eta = jnp.where(valid_eta, eta, 1.0)
    mean = 1.0 / safe_eta
    scale = mean / shape
    valid_y = jnp.isfinite(y_t) & (y_t > 0.0)
    safe_y = jnp.where(valid_y, y_t, 1.0)
    log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
    return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_y & valid_eta)


def emission_log_prob_beta_probit(
    y_t: Float[Array, " M"],
    z_t: Float[Array, " D"],
    H: Float[Array, "M D"],
    d: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration=10.0,
) -> FloatScalar:
    """Log p(y_t | z_t) for Beta emissions (probit-link).

    mean = Phi(eta), concentration phi.
    alpha = mean * phi, beta_ = (1 - mean) * phi.
    """
    eta = H @ z_t + d
    mean = jstats.norm.cdf(eta)
    mean = jnp.clip(mean, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = mean * concentration
    beta_ = (1.0 - mean) * concentration
    valid_y = jnp.isfinite(y_t) & (y_t > 0.0) & (y_t < 1.0)
    safe_y = jnp.where(valid_y, y_t, 0.5)
    log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
    return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_y)


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
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    safe_eta = jnp.where(valid_eta, eta, 1.0)
    mu = 1.0 / safe_eta
    g = shape * (mu - y_t) * obs_mask_t
    w = (shape * mu**2) * obs_mask_t
    invalid_observed = (obs_mask_t > 0.5) & ~valid_eta
    g = jnp.where(invalid_observed, jnp.nan, g)
    w = jnp.where(invalid_observed, jnp.nan, w)
    return g, w


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


def get_mean_param_log_prob_fn(manifest_dist, extra_params=None):
    """Return log-prob(y | mean-parameter) for one observation vector.

    Unlike ``get_emission_fn()``, this operates directly on the expected mean /
    location in observation space. It is used for interval-summary measurement
    semantics where the mean is aggregated over a support window after applying
    the link function.
    """
    from nof1_causal_lab.artifacts.model_spec import DistributionFamily

    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)

    def gaussian(y_t, mean_t, R, obs_mask_t):
        residual = (y_t - mean_t) * obs_mask_t
        n_obs = jnp.sum(obs_mask_t)
        R_adj = symmetrize_with_jitter(inflate_missing_variance(R, obs_mask_t))
        chol = jnp.linalg.cholesky(R_adj)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        n_missing = y_t.shape[0] - n_obs
        logdet = logdet - n_missing * jnp.log(MISSING_DATA_LARGE_VAR)
        whitened = jla.solve_triangular(chol, residual, lower=True)
        mahal = jnp.sum(whitened * whitened)
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
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0)
        rate = jnp.where(valid_mean, mean_t, 1.0)
        log_probs = jax.scipy.stats.poisson.logpmf(y_t, rate)
        return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_mean)

    def gamma(y_t, mean_t, _R, obs_mask_t):
        shape = extra_params.get("obs_shape", 1.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t > 0.0)
        safe_mean = jnp.where(valid_mean, mean_t, 1.0)
        scale = safe_mean / shape
        valid_y = jnp.isfinite(y_t) & (y_t > 0.0)
        safe_y = jnp.where(valid_y, y_t, 1.0)
        log_probs = jax.scipy.stats.gamma.logpdf(safe_y, shape, scale=scale)
        return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_mean & valid_y)

    def bernoulli(y_t, mean_t, _R, obs_mask_t):
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0) & (mean_t <= 1.0)
        valid_y = jnp.isfinite(y_t) & (jnp.isclose(y_t, 0.0) | jnp.isclose(y_t, 1.0))
        p = jnp.where(valid_mean, mean_t, 0.5)
        log_probs = jnp.where(jnp.isclose(y_t, 1.0), jnp.log(p), jnp.log1p(-p))
        return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_mean & valid_y)

    def negative_binomial(y_t, mean_t, _R, obs_mask_t):
        r = extra_params.get("obs_r", 5.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0)
        mu = jnp.where(valid_mean, mean_t, 1.0)
        log_probs = (
            jax.lax.lgamma(y_t + r)
            - jax.lax.lgamma(r)
            - jax.lax.lgamma(y_t + 1.0)
            + r * jnp.log(r / (r + mu))
            + y_t * jnp.log(mu / (r + mu) + NUMERICAL_EPSILON)
        )
        return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_mean)

    def beta(y_t, mean_t, _R, obs_mask_t):
        concentration = extra_params.get("obs_concentration", 10.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t > 0.0) & (mean_t < 1.0)
        safe_mean = jnp.where(valid_mean, mean_t, 0.5)
        alpha = safe_mean * concentration
        beta_ = (1.0 - safe_mean) * concentration
        valid_y = jnp.isfinite(y_t) & (y_t > 0.0) & (y_t < 1.0)
        safe_y = jnp.where(valid_y, y_t, 0.5)
        log_probs = jax.scipy.stats.beta.logpdf(safe_y, alpha, beta_)
        return _sum_masked_log_probs(log_probs, obs_mask_t, valid_obs=valid_mean & valid_y)

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
    from nof1_causal_lab.artifacts.model_spec import DistributionFamily

    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)

    def gaussian(key, mean_t, R):
        R_adj = symmetrize_with_jitter(R)
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
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0)
        safe_rate = jnp.where(valid_mean, mean_t, 1.0)
        draw = jax.random.poisson(key, safe_rate).astype(jnp.float32)
        return jnp.where(valid_mean, draw, jnp.nan)

    def gamma(key, mean_t, _R):
        shape = extra_params.get("obs_shape", 1.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t > 0.0)
        safe_mean = jnp.where(valid_mean, mean_t, 1.0)
        scale = safe_mean / jnp.maximum(shape, NUMERICAL_EPSILON)
        draw = jax.random.gamma(key, shape, shape=mean_t.shape) * scale
        return jnp.where(valid_mean, draw, jnp.nan)

    def bernoulli(key, mean_t, _R):
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0) & (mean_t <= 1.0)
        safe_p = jnp.where(valid_mean, mean_t, 0.5)
        draw = jax.random.bernoulli(key, safe_p).astype(jnp.float32)
        return jnp.where(valid_mean, draw, jnp.nan)

    def negative_binomial(key, mean_t, _R):
        r = extra_params.get("obs_r", 5.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t >= 0.0)
        safe_mean = jnp.where(valid_mean, mean_t, 1.0)
        key_gamma, key_poisson = jax.random.split(key)
        gamma_draw = (
            jax.random.gamma(key_gamma, r, shape=mean_t.shape) * safe_mean / jnp.maximum(r, 1e-8)
        )
        draw = jax.random.poisson(
            key_poisson,
            jnp.maximum(gamma_draw, NUMERICAL_EPSILON),
        ).astype(jnp.float32)
        return jnp.where(valid_mean, draw, jnp.nan)

    def beta(key, mean_t, _R):
        concentration = extra_params.get("obs_concentration", 10.0)
        valid_mean = jnp.isfinite(mean_t) & (mean_t > 0.0) & (mean_t < 1.0)
        safe_mean = jnp.where(valid_mean, mean_t, 0.5)
        alpha = jnp.maximum(safe_mean * concentration, 1e-4)
        beta_param = jnp.maximum((1.0 - safe_mean) * concentration, 1e-4)
        key_alpha, key_beta = jax.random.split(key)
        gamma_alpha = jax.random.gamma(key_alpha, alpha)
        gamma_beta = jax.random.gamma(key_beta, beta_param)
        draw = gamma_alpha / jnp.maximum(gamma_alpha + gamma_beta, NUMERICAL_EPSILON)
        return jnp.where(valid_mean, draw, jnp.nan)

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


def build_heterogeneous_mean_log_prob_fn(
    manifest_dists,
    extra_params: dict | None = None,
):
    """Build an observation-space log-prob for heterogeneous manifest families."""
    from nof1_causal_lab.artifacts.model_spec import DistributionFamily

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

    def heterogeneous_mean_log_prob(y_t, mean_t, R, obs_mask_t):
        total_ll = 0.0
        for ch_indices, group_fn in group_fns:
            idx = jnp.array(ch_indices)
            y_g = y_t[idx]
            mean_g = mean_t[idx]
            R_g = R[jnp.ix_(idx, idx)]
            mask_g = obs_mask_t[idx]
            total_ll = total_ll + group_fn(y_g, mean_g, R_g, mask_g)
        return total_ll

    return heterogeneous_mean_log_prob


def build_heterogeneous_mean_sample_fn(
    manifest_dists,
    extra_params: dict | None = None,
):
    """Build an observation-space sampler for heterogeneous manifest families."""
    from nof1_causal_lab.artifacts.model_spec import DistributionFamily

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

    def heterogeneous_mean_sample(key, mean_t, R):
        sampled = jnp.zeros_like(mean_t)
        keys = jax.random.split(key, len(group_fns))
        for subkey, (ch_indices, group_fn) in zip(keys, group_fns, strict=False):
            idx = jnp.array(ch_indices)
            sampled = sampled.at[idx].set(group_fn(subkey, mean_t[idx], R[jnp.ix_(idx, idx)]))
        return sampled

    return heterogeneous_mean_sample
