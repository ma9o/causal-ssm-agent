"""Canonical emission log-probability functions for all noise families.

Each function computes log p(y_t | z_t) for a single time step given
the measurement model parameters (H, d, R) and an observation mask.

Used by: Laplace-EM, Structured VI, DPF, Rao-Blackwell PF, bootstrap PF.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax.scipy.stats as jstats


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
        + y_t * jnp.log(mu / (r + mu) + 1e-10)
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
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    log_probs = y_t * jnp.log(p) + (1.0 - y_t) * jnp.log(1.0 - p)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


def emission_log_prob_gamma_inverse(y_t, z_t, H, d, _R, obs_mask_t, shape=1.0):
    """Log p(y_t | z_t) for Gamma emissions (inverse-link for mean).

    mean = 1 / eta (canonical link for Gamma).
    """
    eta = H @ z_t + d
    mean = 1.0 / jnp.clip(eta, 1e-6, None)
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
    mean = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
    alpha = mean * concentration
    beta_ = (1.0 - mean) * concentration
    log_probs = jax.scipy.stats.beta.logpdf(y_t, alpha, beta_)
    return jnp.sum(jnp.where(obs_mask_t > 0.5, log_probs, 0.0))


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
    elif manifest_dist == "poisson":
        return emission_log_prob_poisson
    elif manifest_dist == "student_t":
        df = extra_params.get("obs_df", 5.0)
        return lambda y, z, H, d, R, m: emission_log_prob_student_t(y, z, H, d, R, m, df)
    elif manifest_dist == "gamma":
        shape = extra_params.get("obs_shape", 1.0)
        if link == "inverse":
            return lambda y, z, H, d, R, m: emission_log_prob_gamma_inverse(y, z, H, d, R, m, shape)
        return lambda y, z, H, d, R, m: emission_log_prob_gamma(y, z, H, d, R, m, shape)
    elif manifest_dist == "bernoulli":
        if link == "probit":
            return emission_log_prob_bernoulli_probit
        return emission_log_prob_bernoulli
    elif manifest_dist == "negative_binomial":
        r = extra_params.get("obs_r", 5.0)
        return lambda y, z, H, d, R, m: emission_log_prob_negative_binomial(y, z, H, d, R, m, r)
    elif manifest_dist == "beta":
        conc = extra_params.get("obs_concentration", 10.0)
        if link == "probit":
            return lambda y, z, H, d, R, m: emission_log_prob_beta_probit(y, z, H, d, R, m, conc)
        return lambda y, z, H, d, R, m: emission_log_prob_beta(y, z, H, d, R, m, conc)
    else:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}'. "
            f"Supported: gaussian, student_t, poisson, gamma, bernoulli, "
            f"negative_binomial, beta."
        )
