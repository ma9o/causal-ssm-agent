"""Tests for emission log-probability functions.

Covers: Gaussian, Poisson, Student-t, Gamma, Bernoulli, NegBin, Beta,
        probit-link variants, inverse-link Gamma, and get_emission_fn dispatcher.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import pytest

from nof1_causal_lab.models.ssm.inference.targets.emissions import (
    emission_log_prob_bernoulli,
    emission_log_prob_bernoulli_probit,
    emission_log_prob_beta,
    emission_log_prob_beta_probit,
    emission_log_prob_categorical,
    emission_log_prob_gamma_inverse,
    emission_log_prob_gaussian,
    emission_log_prob_negative_binomial,
    emission_log_prob_ordered_logistic,
    emission_log_prob_poisson,
    emission_log_prob_student_t,
    get_mean_param_log_prob_fn,
)
from nof1_causal_lab.models.ssm.inference.targets.observation_dispatch import get_emission_fn

# =============================================================================
# Helpers
# =============================================================================


def _simple_params(n_latent=2, n_manifest=2):
    """Identity measurement structure: H=I, d=0, R=0.5*I."""
    H = jnp.eye(n_manifest, n_latent)
    d = jnp.zeros(n_manifest)
    R = jnp.eye(n_manifest) * 0.5
    return H, d, R


# =============================================================================
# Gaussian
# =============================================================================


class TestGaussianEmission:
    def test_missing_channel_ignored(self):
        """Masked-out channels should not affect log-prob."""
        H, d, R = _simple_params()
        z = jnp.array([1.0, 2.0])
        y_close = jnp.array([1.0, 2.0])
        y_far = jnp.array([1.0, 999.0])
        mask = jnp.array([1.0, 0.0])
        lp_close = emission_log_prob_gaussian(y_close, z, H, d, R, mask)
        lp_far = emission_log_prob_gaussian(y_far, z, H, d, R, mask)
        assert jnp.isclose(lp_close, lp_far, atol=1e-3)

    def test_all_missing_returns_zero(self):
        """When all channels are missing, log-prob should be 0."""
        H, d, R = _simple_params()
        z = jnp.array([1.0, 2.0])
        y = jnp.array([999.0, 999.0])
        mask = jnp.zeros(2)
        lp = emission_log_prob_gaussian(y, z, H, d, R, mask)
        assert jnp.isclose(lp, 0.0)


# =============================================================================
# Poisson
# =============================================================================


class TestPoissonEmission:
    def test_matches_scipy(self):
        """Log-prob should match jax.scipy.stats.poisson.logpmf."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(5.0)])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        lp = emission_log_prob_poisson(y, z, H, d, R, mask)
        expected = jstats.poisson.logpmf(3.0, 5.0)
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_masked_channel_zero(self):
        """Masked channels contribute 0 to log-prob."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        z = jnp.array([jnp.log(5.0), jnp.log(10.0)])
        y = jnp.array([3.0, 999.0])
        mask = jnp.array([1.0, 0.0])
        lp = emission_log_prob_poisson(y, z, H, d, R, mask)
        expected = jstats.poisson.logpmf(3.0, 5.0)
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Student-t
# =============================================================================


class TestStudentTEmission:
    def test_heavier_tails_than_gaussian(self):
        """Student-t should give higher log-prob for outliers than Gaussian."""
        H, d, R = _simple_params(1, 1)
        z = jnp.array([0.0])
        y = jnp.array([5.0])
        mask = jnp.ones(1)
        lp_t = emission_log_prob_student_t(y, z, H, d, R, mask, df=3.0)
        lp_g = emission_log_prob_gaussian(y, z, H, d, R, mask)
        assert lp_t > lp_g

    def test_matches_scipy_univariate(self):
        """Should match scipy t.logpdf for scalar case."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1) * 2.0
        z = jnp.array([1.0])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        df = 5.0
        lp = emission_log_prob_student_t(y, z, H, d, R, mask, df=df)
        scale = jnp.sqrt(2.0)
        expected = jstats.t.logpdf(3.0, df, loc=1.0, scale=scale)
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Gamma
# =============================================================================


class TestGammaEmission:
    def test_inverse_link(self):
        """Gamma with inverse link: mean = 1/eta, scale = mean/shape."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.5])
        y = jnp.array([1.5])
        mask = jnp.ones(1)
        lp = emission_log_prob_gamma_inverse(y, z, H, d, R, mask, shape=2.0)
        # mean = 1/0.5 = 2.0, scale = 2.0/2.0 = 1.0
        expected = jstats.gamma.logpdf(1.5, a=2.0, scale=1.0)
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_log_link_invalid_observation_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(2.0)])
        y = jnp.array([0.0])
        mask = jnp.ones(1)
        fn = get_emission_fn("gamma", extra_params={"obs_shape": 2.0})
        lp = fn(y, z, H, d, R, mask)
        assert jnp.isneginf(lp)

    def test_inverse_link_invalid_linear_predictor_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([-0.5])
        y = jnp.array([1.5])
        mask = jnp.ones(1)
        lp = emission_log_prob_gamma_inverse(y, z, H, d, R, mask, shape=2.0)
        assert jnp.isneginf(lp)


# =============================================================================
# Bernoulli
# =============================================================================


class TestBernoulliEmission:
    def test_probit_link(self):
        """Probit link should give log(0.5) at eta=0."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        lp_logit = emission_log_prob_bernoulli(y, z, H, d, R, mask)
        lp_probit = emission_log_prob_bernoulli_probit(y, z, H, d, R, mask)
        assert jnp.isclose(lp_logit, jnp.log(0.5), atol=1e-5)
        assert jnp.isclose(lp_probit, jnp.log(0.5), atol=1e-5)


# =============================================================================
# Negative Binomial
# =============================================================================


class TestNegBinEmission:
    def test_overdispersion_increases_with_lower_r(self):
        """Lower r means more overdispersion, so NB should be more spread."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([jnp.log(5.0)])
        y = jnp.array([3.0])
        mask = jnp.ones(1)
        lp_low_r = emission_log_prob_negative_binomial(y, z, H, d, R, mask, r=2.0)
        lp_high_r = emission_log_prob_negative_binomial(y, z, H, d, R, mask, r=100.0)
        # Higher r (less overdispersion) should give higher log-prob near the mean
        assert lp_high_r > lp_low_r


# =============================================================================
# Ordered Logistic / Categorical
# =============================================================================


class TestDiscreteEmission:
    def test_ordered_logistic_matches_manual_probability(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        cutpoints = jnp.array([[-1.0, 1.0]])
        level_counts = jnp.array([3])

        lp = emission_log_prob_ordered_logistic(y, z, H, d, R, mask, cutpoints, level_counts)
        expected = jnp.log(jax.nn.sigmoid(1.0) - jax.nn.sigmoid(-1.0))
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_categorical_matches_manual_softmax(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.7])
        y = jnp.array([2.0])
        mask = jnp.ones(1)
        intercepts = jnp.array([[-1.0, 0.5]])
        slopes = jnp.array([[0.2, -0.4]])
        level_counts = jnp.array([3])

        lp = emission_log_prob_categorical(
            y,
            z,
            H,
            d,
            R,
            mask,
            intercepts,
            slopes,
            level_counts,
        )
        logits = jnp.array([0.0, -1.0 + 0.2 * 0.7, 0.5 - 0.4 * 0.7])
        expected = jax.nn.log_softmax(logits)[2]
        assert jnp.isclose(lp, expected, atol=1e-5)


# =============================================================================
# Beta
# =============================================================================


class TestBetaEmission:
    def test_probit_link_at_center(self):
        """Beta probit at eta=0: Phi(0)=0.5 → Beta(0.5|5,5), must be a valid positive density."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        mask = jnp.ones(1)
        lp = emission_log_prob_beta_probit(y, z, H, d, R, mask, concentration=10.0)
        # Phi(0)=0.5, concentration=10 → alpha=beta=5, y=0.5 is mode → high density
        assert lp > 0.0, f"Log-prob at mode of symmetric Beta should be positive, got {lp}"

    def test_logit_vs_probit_at_center(self):
        """At eta=0, logit and probit both give mean=0.5."""
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        mask = jnp.ones(1)
        conc = 10.0
        lp_logit = emission_log_prob_beta(y, z, H, d, R, mask, concentration=conc)
        lp_probit = emission_log_prob_beta_probit(y, z, H, d, R, mask, concentration=conc)
        assert jnp.isclose(lp_logit, lp_probit, atol=1e-4)

    def test_invalid_observation_returns_negative_infinity(self):
        H = jnp.eye(1)
        d = jnp.zeros(1)
        R = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        mask = jnp.ones(1)
        lp = emission_log_prob_beta(y, z, H, d, R, mask, concentration=10.0)
        assert jnp.isneginf(lp)


class TestMeanParamLogProb:
    def test_gamma_invalid_support_returns_negative_infinity(self):
        fn = get_mean_param_log_prob_fn("gamma", extra_params={"obs_shape": 2.0})
        lp = fn(
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
            jnp.eye(1, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert jnp.isneginf(lp)

    def test_beta_invalid_mean_returns_negative_infinity(self):
        fn = get_mean_param_log_prob_fn("beta", extra_params={"obs_concentration": 10.0})
        lp = fn(
            jnp.array([0.5], dtype=jnp.float32),
            jnp.array([1.2], dtype=jnp.float32),
            jnp.eye(1, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert jnp.isneginf(lp)


# =============================================================================
# get_emission_fn dispatcher
# =============================================================================


class TestGetEmissionFn:
    def test_bernoulli_probit(self):
        fn = get_emission_fn("bernoulli", link="probit")
        assert fn is emission_log_prob_bernoulli_probit

    def test_student_t_wraps_df(self):
        fn = get_emission_fn("student_t", extra_params={"obs_df": 10.0})
        H = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp = fn(y, z, H, d, R, mask)
        expected = emission_log_prob_student_t(y, z, H, d, R, mask, df=10.0)
        assert jnp.isclose(lp, expected)

    def test_gamma_default_log_matches_direct(self):
        fn = get_emission_fn("gamma", extra_params={"obs_shape": 2.0})
        H = jnp.eye(1)
        z = jnp.array([jnp.log(3.0)])
        y = jnp.array([2.0])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp = fn(y, z, H, d, R, mask)
        # Log link: mean = exp(eta) = 3.0, scale = 3.0/2.0 = 1.5
        expected = jstats.gamma.logpdf(2.0, a=2.0, scale=1.5)
        assert jnp.isclose(lp, expected, atol=1e-5)

    def test_gamma_inverse_matches_direct(self):
        fn = get_emission_fn("gamma", extra_params={"obs_shape": 2.0}, link="inverse")
        H = jnp.eye(1)
        z = jnp.array([0.5])
        y = jnp.array([1.5])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_gamma_inverse(y, z, H, d, R, mask, shape=2.0)
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_negative_binomial_matches_direct(self):
        fn = get_emission_fn("negative_binomial", extra_params={"obs_r": 5.0})
        H = jnp.eye(1)
        z = jnp.array([jnp.log(3.0)])
        y = jnp.array([2.0])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_negative_binomial(y, z, H, d, R, mask, r=5.0)
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_beta_default_logit_matches_direct(self):
        fn = get_emission_fn("beta", extra_params={"obs_concentration": 10.0})
        H = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_beta(y, z, H, d, R, mask, concentration=10.0)
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_beta_probit_matches_direct(self):
        fn = get_emission_fn("beta", extra_params={"obs_concentration": 10.0}, link="probit")
        H = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([0.5])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_beta_probit(y, z, H, d, R, mask, concentration=10.0)
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_ordered_logistic_matches_direct(self):
        cutpoints = jnp.array([[-1.0, 1.0]])
        level_counts = jnp.array([3])
        fn = get_emission_fn(
            "ordered_logistic",
            extra_params={
                "obs_level_counts": level_counts,
                "obs_ordered_cutpoints": cutpoints,
            },
            link="cumulative_logit",
        )
        H = jnp.eye(1)
        z = jnp.array([0.0])
        y = jnp.array([1.0])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_ordered_logistic(y, z, H, d, R, mask, cutpoints, level_counts)
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_categorical_matches_direct(self):
        intercepts = jnp.array([[-1.0, 0.5]])
        slopes = jnp.array([[0.2, -0.4]])
        level_counts = jnp.array([3])
        fn = get_emission_fn(
            "categorical",
            extra_params={
                "obs_level_counts": level_counts,
                "obs_cat_intercepts": intercepts,
                "obs_cat_slopes": slopes,
            },
            link="softmax",
        )
        H = jnp.eye(1)
        z = jnp.array([0.7])
        y = jnp.array([2.0])
        d = jnp.zeros(1)
        R = jnp.eye(1)
        mask = jnp.ones(1)
        lp_dispatch = fn(y, z, H, d, R, mask)
        lp_direct = emission_log_prob_categorical(
            y, z, H, d, R, mask, intercepts, slopes, level_counts
        )
        assert jnp.isclose(lp_dispatch, lp_direct)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="No emission function"):
            get_emission_fn("unsupported_distribution")
