"""Canonical emission log-probability functions for all noise families.

Each function computes log p(y_t | z_t) for a single time step given
the measurement model parameters (H, d, R) and an observation mask.

Used by: Laplace-EM, Structured VI, DPF, Rao-Blackwell PF, bootstrap PF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax.scipy.special
import jax.scipy.stats as jstats

from causal_ssm_agent.models.likelihoods.base import (
    CHOL_JITTER,
    ETA_CLIP_MIN,
    MISSING_DATA_LARGE_VAR,
    NUMERICAL_EPSILON,
    PROB_CLIP_MIN,
)


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
    log_probs = jax.scipy.stats.gamma.logpdf(y_t, shape, scale=scale)
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
    log_probs = jax.scipy.stats.beta.logpdf(y_t, alpha, beta_)
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
    log_probs = jax.scipy.stats.gamma.logpdf(y_t, shape, scale=scale)
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
    log_probs = jax.scipy.stats.beta.logpdf(y_t, alpha, beta_)
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
    from causal_ssm_agent.models.likelihoods.observation_families import FAMILY_REGISTRY
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    family_spec = FAMILY_REGISTRY.get(DistributionFamily(manifest_dist))
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
    from causal_ssm_agent.models.likelihoods.observation_families import FAMILY_REGISTRY
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    extra_params = extra_params or {}
    family_spec = FAMILY_REGISTRY.get(DistributionFamily(manifest_dist))
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
