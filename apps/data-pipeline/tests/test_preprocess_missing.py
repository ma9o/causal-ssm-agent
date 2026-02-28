"""Tests for base.py preprocess_missing_data utility.

Covers: NaN handling, mask creation, variance inflation for missing entries.
"""

import jax.numpy as jnp

from causal_ssm_agent.models.likelihoods.base import (
    MISSING_DATA_LARGE_VAR,
    preprocess_missing_data,
)


class TestPreprocessMissingData:
    def test_no_missing_data(self):
        """All observed: observations unchanged, R unchanged, mask all True."""
        obs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        R = jnp.eye(2) * 0.5
        clean, R_adj, mask = preprocess_missing_data(obs, R, obs_mask=None)

        assert jnp.allclose(clean, obs)
        assert mask.all()
        # R should be broadcast but not inflated
        assert R_adj.shape == (2, 2, 2)
        assert jnp.allclose(R_adj[0], R)

    def test_nan_replaced_with_zero(self):
        """NaN values should be replaced with 0."""
        obs = jnp.array([[1.0, float("nan")], [float("nan"), 4.0]])
        R = jnp.eye(2) * 0.5
        clean, _, _ = preprocess_missing_data(obs, R, obs_mask=None)

        assert jnp.isclose(clean[0, 0], 1.0)
        assert jnp.isclose(clean[0, 1], 0.0)
        assert jnp.isclose(clean[1, 0], 0.0)
        assert jnp.isclose(clean[1, 1], 4.0)

    def test_mask_created_from_nan(self):
        """When obs_mask is None, mask should be True where obs is not NaN."""
        obs = jnp.array([[1.0, float("nan")], [3.0, 4.0]])
        R = jnp.eye(2)
        _, _, mask = preprocess_missing_data(obs, R, obs_mask=None)

        assert mask.shape == (2, 2)
        assert bool(mask[0, 0]) is True
        assert bool(mask[0, 1]) is False
        assert bool(mask[1, 0]) is True
        assert bool(mask[1, 1]) is True

    def test_variance_inflated_for_missing(self):
        """Missing entries should have inflated diagonal variance."""
        obs = jnp.array([[1.0, float("nan")]])
        R = jnp.eye(2) * 0.5
        _, R_adj, _ = preprocess_missing_data(obs, R, obs_mask=None)

        # Channel 0: observed → R unchanged
        assert jnp.isclose(R_adj[0, 0, 0], 0.5)
        # Channel 1: missing → R inflated
        assert R_adj[0, 1, 1] > 1e5
        assert jnp.isclose(R_adj[0, 1, 1], 0.5 + MISSING_DATA_LARGE_VAR)

    def test_explicit_mask_used(self):
        """Explicit obs_mask should override NaN-based detection."""
        obs = jnp.array([[1.0, 2.0]])  # No NaN
        R = jnp.eye(2) * 0.5
        mask = jnp.array([[True, False]])  # Mark channel 1 as missing
        _clean, R_adj, out_mask = preprocess_missing_data(obs, R, obs_mask=mask)

        # Channel 1 should be inflated even though value is not NaN
        assert R_adj[0, 1, 1] > 1e5
        assert jnp.allclose(out_mask, mask)

    def test_output_shapes(self):
        """Output shapes should match expected dimensions."""
        T, n = 5, 3
        obs = jnp.ones((T, n))
        R = jnp.eye(n)
        clean, R_adj, mask = preprocess_missing_data(obs, R, obs_mask=None)

        assert clean.shape == (T, n)
        assert R_adj.shape == (T, n, n)
        assert mask.shape == (T, n)

    def test_mixed_missing_pattern(self):
        """Different missing patterns across timesteps."""
        obs = jnp.array(
            [
                [1.0, float("nan"), 3.0],
                [float("nan"), 2.0, float("nan")],
            ]
        )
        R = jnp.eye(3) * 0.1
        _, R_adj, mask = preprocess_missing_data(obs, R, obs_mask=None)

        # t=0: channels 0,2 observed; channel 1 missing
        assert bool(mask[0, 0]) is True
        assert bool(mask[0, 1]) is False
        assert bool(mask[0, 2]) is True
        assert jnp.isclose(R_adj[0, 0, 0], 0.1)
        assert R_adj[0, 1, 1] > 1e5

        # t=1: channel 1 observed; channels 0,2 missing
        assert bool(mask[1, 0]) is False
        assert bool(mask[1, 1]) is True
        assert bool(mask[1, 2]) is False
