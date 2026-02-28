"""Tests for SSM model prior construction helpers.

Covers: _make_prior_dist, _make_prior_batch.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from causal_ssm_agent.models.ssm.model import _make_prior_batch, _make_prior_dist


class TestMakePriorDist:
    def test_normal_with_mu_sigma(self):
        """Prior with mu and sigma should give Normal."""
        d = _make_prior_dist({"mu": 0.0, "sigma": 1.0})
        assert isinstance(d, dist.Normal)

    def test_half_normal_with_sigma_only(self):
        """Prior with only sigma should give HalfNormal."""
        d = _make_prior_dist({"sigma": 2.0})
        assert isinstance(d, dist.HalfNormal)

    def test_truncated_with_bounds(self):
        """Prior with lower/upper should give a truncated distribution."""
        d = _make_prior_dist({"mu": 0.0, "sigma": 1.0, "lower": -1.0, "upper": 1.0})
        # NumPyro wraps TruncatedNormal → TwoSidedTruncatedDistribution
        assert hasattr(d, "low") or hasattr(d, "base_dist")

    def test_normal_mean(self):
        """Normal prior should have correct mean."""
        d = _make_prior_dist({"mu": 5.0, "sigma": 2.0})
        assert jnp.isclose(d.mean, 5.0)

    def test_array_valued_params(self):
        """Should handle array-valued mu/sigma."""
        d = _make_prior_dist({"mu": jnp.array([1.0, 2.0]), "sigma": jnp.array([0.5, 1.0])})
        assert d.batch_shape == (2,)


class TestMakePriorBatch:
    def test_scalar_expanded_to_n(self):
        """Scalar prior should expand to batch shape (n,)."""
        d = _make_prior_batch({"mu": 0.0, "sigma": 1.0}, n=3)
        assert d.batch_shape == (3,)

    def test_array_matching_n(self):
        """Array prior matching n should pass through."""
        d = _make_prior_batch(
            {"mu": jnp.array([1.0, 2.0, 3.0]), "sigma": jnp.array([0.5, 0.5, 0.5])},
            n=3,
        )
        assert d.batch_shape == (3,)

    def test_mismatched_shape_raises(self):
        """Array prior with wrong size should raise."""
        with pytest.raises(ValueError, match="does not match"):
            _make_prior_batch(
                {"mu": jnp.array([1.0, 2.0]), "sigma": jnp.array([0.5, 0.5])},
                n=3,
            )

    def test_half_normal_expanded(self):
        """HalfNormal prior should also expand."""
        d = _make_prior_batch({"sigma": 1.0}, n=4)
        assert d.batch_shape == (4,)
        assert isinstance(d.base_dist if hasattr(d, "base_dist") else d, dist.HalfNormal)
