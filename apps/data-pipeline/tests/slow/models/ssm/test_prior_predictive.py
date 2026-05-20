"""Slow prior-predictive tests."""

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.parameterization import compile_prior_semantics
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    sample_prior_predictive_from_compiled_semantics,
)
from tests.models.ssm._support import complex_mixed_runtime_spec

pytestmark = pytest.mark.slow


class TestCompiledPriorPredictiveRuntime:
    def test_mixed_likelihood_samples_are_finite(self):
        spec = complex_mixed_runtime_spec()
        semantics = compile_prior_semantics(spec)
        samples = sample_prior_predictive_from_compiled_semantics(
            spec,
            semantics,
            jnp.linspace(0.0, 5.5, 12, dtype=jnp.float32),
            num_samples=10,
            seed=7,
        )

        assert samples["observations"].shape == (10, 12, 10)
        assert samples["observations_mask"].shape == (10, 12, 10)
        assert bool(jnp.isfinite(samples["drift"]).all())
        assert bool(jnp.isfinite(samples["diffusion"]).all())
        assert bool(jnp.isfinite(samples["observations"]).all())
        assert bool(samples["observations_mask"].all())
        assert bool(
            (
                (samples["observations"][:, :, 1] == 0) | (samples["observations"][:, :, 1] == 1)
            ).all()
        )
        assert bool((samples["observations"][:, :, 2] >= 0).all())
        assert bool((samples["observations"][:, :, 4] > 0).all())
        assert bool(
            (
                (samples["observations"][:, :, 5] >= 0) & (samples["observations"][:, :, 5] <= 1)
            ).all()
        )
        assert bool(
            (
                (samples["observations"][:, :, 6] >= 0) & (samples["observations"][:, :, 6] <= 3)
            ).all()
        )
        assert bool(
            (
                (samples["observations"][:, :, 7] >= 0) & (samples["observations"][:, :, 7] <= 3)
            ).all()
        )
        assert bool((samples["observations"][:, :, 8] >= 0).all())
