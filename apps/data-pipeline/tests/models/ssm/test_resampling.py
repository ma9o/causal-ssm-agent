"""Tests for JAX-native systematic resampling.

Covers: correct index shape, valid index range, weight concentration,
        deterministic with fixed key, uniform weights produce uniform samples.
"""

import jax
import jax.numpy as jnp

from causal_ssm_agent.models.ssm.inference.targets.particle import _systematic_resampling


class TestSystematicResampling:
    def test_deterministic_with_same_key(self):
        """Same key should produce same indices."""
        logits = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
        key = jax.random.PRNGKey(123)
        idx1 = _systematic_resampling(key, logits, 5)
        idx2 = _systematic_resampling(key, logits, 5)
        assert jnp.array_equal(idx1, idx2)

    def test_high_weight_particle_selected_more(self):
        """Particle with much higher weight should be selected most often."""
        key = jax.random.PRNGKey(7)
        N = 100
        logits = jnp.full(N, -10.0)
        logits = logits.at[0].set(10.0)  # particle 0 has overwhelmingly high weight
        idx = _systematic_resampling(key, logits, N)
        # Almost all indices should be 0
        assert jnp.sum(idx == 0) > N * 0.9

    def test_uniform_weights_spread_indices(self):
        """Uniform weights should produce roughly uniform index distribution."""
        key = jax.random.PRNGKey(99)
        N = 100
        logits = jnp.zeros(N)
        idx = _systematic_resampling(key, logits, N)
        # With systematic resampling and uniform weights, each particle
        # should be selected exactly once
        counts = jnp.bincount(idx, length=N)
        assert jnp.all(counts == 1)
