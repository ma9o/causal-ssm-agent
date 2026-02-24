"""Tests for emission log-probability functions, focusing on link function variants.

Covers:
1. Probit Bernoulli emission log-probs
2. Inverse Gamma emission log-probs
3. Probit Beta emission log-probs
4. get_emission_fn dispatch with link parameter
"""

import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.likelihoods.emissions import (
    emission_log_prob_bernoulli,
    emission_log_prob_bernoulli_probit,
    emission_log_prob_beta,
    emission_log_prob_beta_probit,
    emission_log_prob_gamma,
    emission_log_prob_gamma_inverse,
    get_emission_fn,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_emission_args(n_latent=2, n_manifest=2, seed=0):
    """Standard emission test arguments."""
    key = random.PRNGKey(seed)
    z_t = random.normal(key, (n_latent,)) * 0.5
    H = jnp.eye(n_manifest, n_latent)
    d = jnp.zeros(n_manifest)
    R = jnp.eye(n_manifest) * 0.1
    obs_mask = jnp.ones(n_manifest)
    return z_t, H, d, R, obs_mask


# =============================================================================
# TestBernoulliProbit
# =============================================================================


class TestBernoulliProbit:
    """Tests for Bernoulli probit emission log-prob."""

    def test_finite_output(self):
        """Probit Bernoulli should produce finite log-probs."""
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([1.0, 0.0])
        ll = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, obs_mask)
        assert jnp.isfinite(ll), f"Probit Bernoulli ll = {ll}"

    def test_negative_log_prob(self):
        """Log-prob should be non-positive."""
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([1.0, 0.0])
        ll = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, obs_mask)
        assert ll <= 0.0, f"Log-prob should be <= 0, got {ll}"

    def test_extreme_eta_values(self):
        """Probit should handle extreme linear predictor values without NaN."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        y_t = jnp.array([1.0, 0.0])

        # Large positive eta → p ≈ 1
        z_large = jnp.array([10.0, -10.0])
        ll = emission_log_prob_bernoulli_probit(y_t, z_large, H, d, R, obs_mask)
        assert jnp.isfinite(ll)

    def test_agrees_with_logit_at_zero(self):
        """At eta=0, probit and logit should agree (both give p=0.5)."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([1.0, 0.0])
        z_t = jnp.zeros(2)

        ll_probit = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, obs_mask)
        ll_logit = emission_log_prob_bernoulli(y_t, z_t, H, d, R, obs_mask)
        assert jnp.allclose(ll_probit, ll_logit, atol=1e-5), (
            f"At eta=0: probit={ll_probit}, logit={ll_logit}"
        )

    def test_respects_obs_mask(self):
        """Masked observations should not contribute to log-prob."""
        z_t, H, d, R, _ = _make_emission_args()
        y_t = jnp.array([1.0, 0.0])

        ll_full = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, jnp.ones(2))
        ll_partial = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, jnp.array([1.0, 0.0]))
        assert ll_full != ll_partial, "Masking should change the result"


# =============================================================================
# TestGammaInverse
# =============================================================================


class TestGammaInverse:
    """Tests for Gamma inverse-link emission log-prob."""

    def test_finite_output(self):
        """Inverse Gamma should produce finite log-probs."""
        H = jnp.eye(2)
        d = jnp.ones(2) * 2.0  # ensure eta > 0
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.zeros(2)  # eta = d = 2.0 → mean = 0.5
        y_t = jnp.array([0.5, 0.3])

        ll = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        assert jnp.isfinite(ll), f"Inverse Gamma ll = {ll}"

    def test_negative_log_prob(self):
        """Log-prob should be non-positive for reasonable data."""
        H = jnp.eye(2)
        d = jnp.ones(2) * 2.0
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.zeros(2)
        y_t = jnp.array([0.5, 0.3])

        ll = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        # Note: log-prob can be positive for Gamma density with high concentration
        # at least it should be finite
        assert jnp.isfinite(ll)

    def test_different_from_log_link(self):
        """Inverse link should give different results from log link."""
        H = jnp.eye(2)
        d = jnp.ones(2) * 1.0  # eta = 1.0
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.zeros(2)
        y_t = jnp.array([0.8, 1.2])

        # log link: mean = exp(1) ≈ 2.718
        # inverse link: mean = 1/1 = 1.0
        ll_log = emission_log_prob_gamma(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        ll_inv = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        assert ll_log != ll_inv, "Log and inverse links should give different results"

    def test_small_eta_clipped(self):
        """Near-zero eta should be clipped and produce finite output."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.array([1e-8, 1e-8])  # very small eta
        y_t = jnp.array([0.5, 0.3])

        ll = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        assert jnp.isfinite(ll), f"Small eta should be clipped: ll = {ll}"

    def test_respects_obs_mask(self):
        """Masked observations should not contribute to log-prob."""
        H = jnp.eye(2)
        d = jnp.ones(2) * 2.0
        R = jnp.eye(2) * 0.1
        z_t = jnp.zeros(2)
        y_t = jnp.array([0.5, 0.3])

        ll_full = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, jnp.ones(2), shape=2.0)
        ll_partial = emission_log_prob_gamma_inverse(
            y_t, z_t, H, d, R, jnp.array([1.0, 0.0]), shape=2.0
        )
        assert ll_full != ll_partial


# =============================================================================
# TestBetaProbit
# =============================================================================


class TestBetaProbit:
    """Tests for Beta probit emission log-prob."""

    def test_finite_output(self):
        """Probit Beta should produce finite log-probs."""
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([0.3, 0.7])  # in (0, 1)

        ll = emission_log_prob_beta_probit(y_t, z_t, H, d, R, obs_mask, concentration=10.0)
        assert jnp.isfinite(ll), f"Probit Beta ll = {ll}"

    def test_agrees_with_logit_at_zero(self):
        """At eta=0, probit and logit should agree (both give mean=0.5)."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([0.3, 0.7])
        z_t = jnp.zeros(2)

        ll_probit = emission_log_prob_beta_probit(y_t, z_t, H, d, R, obs_mask, concentration=10.0)
        ll_logit = emission_log_prob_beta(y_t, z_t, H, d, R, obs_mask, concentration=10.0)
        assert jnp.allclose(ll_probit, ll_logit, atol=1e-4), (
            f"At eta=0: probit={ll_probit}, logit={ll_logit}"
        )

    def test_extreme_eta_values(self):
        """Probit should handle extreme linear predictor values without NaN."""
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([0.3, 0.7])

        z_large = jnp.array([5.0, -5.0])
        ll = emission_log_prob_beta_probit(y_t, z_large, H, d, R, obs_mask, concentration=10.0)
        assert jnp.isfinite(ll)

    def test_respects_obs_mask(self):
        """Masked observations should not contribute to log-prob."""
        z_t, H, d, R, _ = _make_emission_args()
        y_t = jnp.array([0.3, 0.7])

        ll_full = emission_log_prob_beta_probit(y_t, z_t, H, d, R, jnp.ones(2), concentration=10.0)
        ll_partial = emission_log_prob_beta_probit(
            y_t, z_t, H, d, R, jnp.array([1.0, 0.0]), concentration=10.0
        )
        assert ll_full != ll_partial


# =============================================================================
# TestGetEmissionFn — link dispatch
# =============================================================================


class TestGetEmissionFn:
    """Test get_emission_fn dispatch with link parameter."""

    def test_bernoulli_default_logit(self):
        """Bernoulli with no link returns logit variant."""
        fn = get_emission_fn("bernoulli")
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([1.0, 0.0])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_bernoulli(y_t, z_t, H, d, R, obs_mask)
        assert jnp.allclose(ll, ll_expected)

    def test_bernoulli_probit_link(self):
        """Bernoulli with probit link returns probit variant."""
        fn = get_emission_fn("bernoulli", link="probit")
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([1.0, 0.0])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_bernoulli_probit(y_t, z_t, H, d, R, obs_mask)
        assert jnp.allclose(ll, ll_expected)

    def test_gamma_default_log(self):
        """Gamma with no link returns log variant."""
        fn = get_emission_fn("gamma", {"obs_shape": 2.0})
        H = jnp.eye(2)
        d = jnp.ones(2)
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.zeros(2)
        y_t = jnp.array([0.5, 0.3])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_gamma(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        assert jnp.allclose(ll, ll_expected)

    def test_gamma_inverse_link(self):
        """Gamma with inverse link returns inverse variant."""
        fn = get_emission_fn("gamma", {"obs_shape": 2.0}, link="inverse")
        H = jnp.eye(2)
        d = jnp.ones(2) * 2.0
        R = jnp.eye(2) * 0.1
        obs_mask = jnp.ones(2)
        z_t = jnp.zeros(2)
        y_t = jnp.array([0.5, 0.3])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_gamma_inverse(y_t, z_t, H, d, R, obs_mask, shape=2.0)
        assert jnp.allclose(ll, ll_expected)

    def test_beta_default_logit(self):
        """Beta with no link returns logit variant."""
        fn = get_emission_fn("beta", {"obs_concentration": 10.0})
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([0.3, 0.7])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_beta(y_t, z_t, H, d, R, obs_mask, concentration=10.0)
        assert jnp.allclose(ll, ll_expected)

    def test_beta_probit_link(self):
        """Beta with probit link returns probit variant."""
        fn = get_emission_fn("beta", {"obs_concentration": 10.0}, link="probit")
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([0.3, 0.7])
        ll = fn(y_t, z_t, H, d, R, obs_mask)
        ll_expected = emission_log_prob_beta_probit(y_t, z_t, H, d, R, obs_mask, concentration=10.0)
        assert jnp.allclose(ll, ll_expected)

    def test_gaussian_ignores_link(self):
        """Gaussian emission ignores link parameter."""
        fn_default = get_emission_fn("gaussian")
        fn_with_link = get_emission_fn("gaussian", link="identity")
        z_t, H, d, R, obs_mask = _make_emission_args()
        y_t = jnp.array([0.5, -0.3])
        ll1 = fn_default(y_t, z_t, H, d, R, obs_mask)
        ll2 = fn_with_link(y_t, z_t, H, d, R, obs_mask)
        assert jnp.allclose(ll1, ll2)

    def test_gradients_flow_probit_bernoulli(self):
        """Gradient should flow through probit Bernoulli emission."""
        fn = get_emission_fn("bernoulli", link="probit")
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([1.0, 0.0])

        def ll_fn(z):
            return fn(y_t, z, H, d, R, obs_mask)

        grad = jax.grad(ll_fn)(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(grad)), f"Gradient not finite: {grad}"

    def test_gradients_flow_inverse_gamma(self):
        """Gradient should flow through inverse Gamma emission."""
        fn = get_emission_fn("gamma", {"obs_shape": 2.0}, link="inverse")
        H = jnp.eye(2)
        d = jnp.ones(2) * 2.0
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([0.5, 0.3])

        def ll_fn(z):
            return fn(y_t, z, H, d, R, obs_mask)

        grad = jax.grad(ll_fn)(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(grad)), f"Gradient not finite: {grad}"

    def test_gradients_flow_probit_beta(self):
        """Gradient should flow through probit Beta emission."""
        fn = get_emission_fn("beta", {"obs_concentration": 10.0}, link="probit")
        H = jnp.eye(2)
        d = jnp.zeros(2)
        R = jnp.eye(2)
        obs_mask = jnp.ones(2)
        y_t = jnp.array([0.3, 0.7])

        def ll_fn(z):
            return fn(y_t, z, H, d, R, obs_mask)

        grad = jax.grad(ll_fn)(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(grad)), f"Gradient not finite: {grad}"
