"""Slow posterior predictive tests."""

import jax.numpy as jnp
import jax.random
import pytest

from nof1_causal_lab.models.posterior_predictive import PPCResult, run_posterior_predictive_checks
from nof1_causal_lab.models.predictive_simulation import (
    sample_predictive_observations_from_linear_predictors,
)
from tests.models.ssm._support import (
    complex_mixed_family_config,
    complex_mixed_runtime_spec,
    make_complex_mixed_samples,
)

pytestmark = pytest.mark.slow


class TestForwardSimulation:
    def test_forward_simulate_large_mixed_family_model(self):
        manifest_dists, manifest_links, manifest_level_counts, _manifest_names = (
            complex_mixed_family_config()
        )
        samples = make_complex_mixed_samples()
        times = jnp.linspace(0.0, 5.5, 12, dtype=jnp.float32)
        # Exercise every emission family from a neutral observation linear predictor.
        linear_predictors = jnp.zeros((8, 12, 10), dtype=jnp.float32)

        y_sim, _, _ = sample_predictive_observations_from_linear_predictors(
            linear_predictors,
            samples,
            times,
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            manifest_level_counts=manifest_level_counts,
            n_subsample=8,
            rng_key=jax.random.PRNGKey(3),
        )

        assert y_sim.shape == (8, 12, 10)
        assert bool(jnp.isfinite(y_sim).all())
        assert bool(((y_sim[:, :, 1] == 0) | (y_sim[:, :, 1] == 1)).all())
        assert bool((y_sim[:, :, 2] >= 0).all())
        assert bool((jnp.mod(y_sim[:, :, 2], 1.0) == 0).all())
        assert bool((y_sim[:, :, 4] > 0).all())
        assert bool(((y_sim[:, :, 5] >= 0) & (y_sim[:, :, 5] <= 1)).all())
        assert bool(((y_sim[:, :, 6] >= 0) & (y_sim[:, :, 6] <= 3)).all())
        assert bool(((y_sim[:, :, 7] >= 0) & (y_sim[:, :, 7] <= 3)).all())
        assert bool((y_sim[:, :, 8] >= 0).all())
        assert bool((jnp.mod(y_sim[:, :, 8], 1.0) == 0).all())


class TestRunPPC:
    def test_basic_run(self):
        manifest_dists, manifest_links, manifest_level_counts, manifest_names = (
            complex_mixed_family_config()
        )
        spec = complex_mixed_runtime_spec()
        samples = make_complex_mixed_samples(seed=7)
        times = jnp.linspace(0.0, 5.5, 12, dtype=jnp.float32)
        # Reference "observed" series: one neutral-predictor emission draw.
        reference_y, _, _ = sample_predictive_observations_from_linear_predictors(
            jnp.zeros((8, 12, 10), dtype=jnp.float32),
            samples,
            times,
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            manifest_level_counts=manifest_level_counts,
            n_subsample=8,
            rng_key=jax.random.PRNGKey(11),
        )
        observations = reference_y[3]

        result = run_posterior_predictive_checks(
            samples=samples,
            observations=observations,
            times=times,
            manifest_names=manifest_names,
            spec=spec,
            n_subsample=20,
        )

        assert isinstance(result, PPCResult)
        assert result.checked is True
        assert result.n_subsample == 12
        assert isinstance(result.per_variable_warnings, list)
        assert len(result.overlays) == len(manifest_names)
        assert {overlay.variable for overlay in result.overlays} == set(manifest_names)
        assert len(result.test_stats) >= len(manifest_names) * 2
