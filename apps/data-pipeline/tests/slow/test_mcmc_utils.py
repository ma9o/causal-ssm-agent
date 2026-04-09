"""Slow MCMC utility tests."""

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from causal_ssm_agent.models.ssm.inference.engines.mcmc_utils import hmc_step

pytestmark = pytest.mark.slow


class TestHMCStep:
    def _make_gaussian_target(self, mean, prec):
        def log_target(z):
            diff = z - mean
            return -0.5 * diff @ prec @ diff

        return jax.value_and_grad(log_target)

    def test_mala_moves_toward_mode(self):
        mean = jnp.array([3.0, -2.0])
        target_fn = self._make_gaussian_target(mean, jnp.eye(2) * 4.0)
        chol_mass = jnp.eye(2) * 2.0

        z = jnp.zeros(2)
        key = random.PRNGKey(42)
        positions = []
        for i in range(500):
            key, step_key = random.split(key)
            z, _accepted, _ = hmc_step(
                step_key, z, target_fn, step_size=0.1, chol_mass=chol_mass, n_leapfrog=1
            )
            if i >= 200:
                positions.append(z)

        sample_mean = jnp.mean(jnp.stack(positions), axis=0)
        assert jnp.allclose(sample_mean, mean, atol=0.5)
