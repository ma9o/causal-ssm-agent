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
    # NB log-pmf via the gamma-Poisson mixture identity:
    # log P(y|r,mu) = gammaln(y+r) - gammaln(r) - gammaln(y+1)
    #                 + r*log(r/(r+mu)) + y*log(mu/(r+mu))
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
    """Beta (logit link): exact score and neg-Hessian via digamma/polygamma.

    Derivation: eta = H z + d, mu = sigmoid(eta), alpha = phi*mu, beta = phi*(1-mu).
    d log p/d eta = phi * mu*(1-mu) * [logit(y) + psi(beta) - psi(alpha)]
    -d^2 log p/d eta^2 = phi * mu*(1-mu) * [phi*mu*(1-mu)*(psi1(alpha)+psi1(beta))
                         - (1-2*mu)*(logit(y)+psi(beta)-psi(alpha))]
    """
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
    extra_params = extra_params or {}
    if manifest_dist == "poisson":
        return _score_weight_poisson
    if manifest_dist == "bernoulli":
        if link == "probit":
            return _score_weight_bernoulli_probit
        return _score_weight_bernoulli_logit
    if manifest_dist == "beta":
        conc = extra_params.get("obs_concentration", 10.0)
        if link == "probit":
            return lambda y, eta, m: _score_weight_beta_probit(y, eta, m, conc)
        return lambda y, eta, m: _score_weight_beta_logit(y, eta, m, conc)
    if manifest_dist == "gamma":
        shape = extra_params.get("obs_shape", 1.0)
        if link == "inverse":
            return lambda y, eta, m: _score_weight_gamma_inverse(y, eta, m, shape)
        return lambda y, eta, m: _score_weight_gamma_log(y, eta, m, shape)
    if manifest_dist == "negative_binomial":
        r = extra_params.get("obs_r", 5.0)
        return lambda y, eta, m: _score_weight_negative_binomial(y, eta, m, r)
    return None  # gaussian and student_t handled separately


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
    extra_params = extra_params or {}
    if manifest_dist == "gaussian":
        return emission_log_prob_gaussian
    if manifest_dist == "poisson":
        return emission_log_prob_poisson
    if manifest_dist == "student_t":
        df = extra_params.get("obs_df", 5.0)
        return lambda y, z, H, d, R, m: emission_log_prob_student_t(y, z, H, d, R, m, df)
    if manifest_dist == "gamma":
        shape = extra_params.get("obs_shape", 1.0)
        if link == "inverse":
            return lambda y, z, H, d, R, m: emission_log_prob_gamma_inverse(y, z, H, d, R, m, shape)
        return lambda y, z, H, d, R, m: emission_log_prob_gamma(y, z, H, d, R, m, shape)
    if manifest_dist == "bernoulli":
        if link == "probit":
            return emission_log_prob_bernoulli_probit
        return emission_log_prob_bernoulli
    if manifest_dist == "negative_binomial":
        r = extra_params.get("obs_r", 5.0)
        return lambda y, z, H, d, R, m: emission_log_prob_negative_binomial(y, z, H, d, R, m, r)
    if manifest_dist == "beta":
        conc = extra_params.get("obs_concentration", 10.0)
        if link == "probit":
            return lambda y, z, H, d, R, m: emission_log_prob_beta_probit(y, z, H, d, R, m, conc)
        return lambda y, z, H, d, R, m: emission_log_prob_beta(y, z, H, d, R, m, conc)
    raise ValueError(
        f"No emission function for manifest_dist='{manifest_dist}'. "
        f"Supported: gaussian, student_t, poisson, gamma, bernoulli, "
        f"negative_binomial, beta."
    )
