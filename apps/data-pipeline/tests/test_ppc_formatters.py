"""Tests for posterior_predictive.py pure formatter functions.

Covers: _compute_overlays, _compute_test_stats.
"""

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.posterior_predictive import (
    _compute_overlays,
    _compute_test_stats,
)

# =============================================================================
# _compute_overlays
# =============================================================================


class TestComputeOverlays:
    def _make_data(self, n_draws=10, T=5, n_manifest=2):
        """Create synthetic simulation data."""
        rng = np.random.default_rng(42)
        y_sim = jnp.array(rng.normal(0, 1, (n_draws, T, n_manifest)))
        observations = jnp.array(rng.normal(0, 1, (T, n_manifest)))
        return y_sim, observations

    def test_returns_one_overlay_per_variable(self):
        y_sim, obs = self._make_data(n_manifest=3)
        result = _compute_overlays(y_sim, obs, ["x", "y", "z"])
        assert len(result) == 3
        assert result[0].variable == "x"
        assert result[1].variable == "y"
        assert result[2].variable == "z"

    def test_overlay_has_correct_length(self):
        y_sim, obs = self._make_data(T=7, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        assert len(result[0].q025) == 7
        assert len(result[0].median) == 7
        assert len(result[0].observed) == 7

    def test_quantile_ordering(self):
        """q025 <= q25 <= median <= q75 <= q975 at each time step."""
        y_sim, obs = self._make_data(n_draws=50, T=10, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        overlay = result[0]
        for t in range(10):
            assert overlay.q025[t] <= overlay.q25[t]
            assert overlay.q25[t] <= overlay.median[t]
            assert overlay.median[t] <= overlay.q75[t]
            assert overlay.q75[t] <= overlay.q975[t]

    def test_spaghetti_draws_count(self):
        y_sim, obs = self._make_data(n_draws=30, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"], n_spaghetti=5)
        assert len(result[0].spaghetti_draws) == 5

    def test_spaghetti_count_capped_at_n_draws(self):
        y_sim, obs = self._make_data(n_draws=3, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"], n_spaghetti=100)
        assert len(result[0].spaghetti_draws) == 3

    def test_spaghetti_draw_length_matches_T(self):
        y_sim, obs = self._make_data(n_draws=5, T=8, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        for draw in result[0].spaghetti_draws:
            assert len(draw) == 8

    def test_nan_observations_become_none(self):
        y_sim, obs = self._make_data(T=3, n_manifest=1)
        obs_with_nan = obs.at[1, 0].set(float("nan"))
        result = _compute_overlays(y_sim, obs_with_nan, ["x"])
        assert result[0].observed[1] is None
        assert result[0].observed[0] is not None

    def test_fallback_variable_name(self):
        """When manifest_names is shorter than n_manifest, use fallback."""
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_overlays(y_sim, obs, ["x"])  # only 1 name for 2 vars
        assert result[0].variable == "x"
        assert result[1].variable == "var_1"

    def test_single_draw(self):
        y_sim, obs = self._make_data(n_draws=1, T=3, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        # All quantiles should be the same value
        assert result[0].q025 == result[0].q975


# =============================================================================
# _compute_test_stats
# =============================================================================


class TestComputeTestStats:
    def _make_data(self, n_draws=10, T=20, n_manifest=2):
        rng = np.random.default_rng(42)
        y_sim = jnp.array(rng.normal(0, 1, (n_draws, T, n_manifest)))
        observations = jnp.array(rng.normal(0, 1, (T, n_manifest)))
        return y_sim, observations

    def test_returns_4_stats_per_variable(self):
        y_sim, obs = self._make_data(n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        assert len(result) == 4
        stat_names = {r.stat_name for r in result}
        assert stat_names == {"mean", "sd", "min", "max"}

    def test_multiple_variables(self):
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_test_stats(y_sim, obs, ["x", "y"])
        assert len(result) == 8  # 4 stats * 2 variables

    def test_rep_values_length_matches_n_draws(self):
        y_sim, obs = self._make_data(n_draws=15, n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        for stat in result:
            assert len(stat.rep_values) == 15

    def test_observed_value_is_float(self):
        y_sim, obs = self._make_data(n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        for stat in result:
            assert isinstance(stat.observed_value, float)

    def test_skips_variables_with_too_few_valid_obs(self):
        """Variables with < 3 valid observations should be skipped."""
        y_sim, obs = self._make_data(T=5, n_manifest=1)
        # Make all but 2 observations NaN
        obs_sparse = jnp.full_like(obs, float("nan"))
        obs_sparse = obs_sparse.at[0, 0].set(1.0)
        obs_sparse = obs_sparse.at[1, 0].set(2.0)
        result = _compute_test_stats(y_sim, obs_sparse, ["x"])
        assert len(result) == 0

    def test_handles_nan_masking(self):
        """NaN values in observations should be excluded from stats."""
        y_sim, obs = self._make_data(T=10, n_manifest=1)
        obs_with_nan = obs.at[0, 0].set(float("nan"))
        result = _compute_test_stats(y_sim, obs_with_nan, ["x"])
        # Should still produce results (9 valid obs > 3)
        assert len(result) == 4

    def test_mean_stat_is_reasonable(self):
        """Mean of replicated data should be near 0 for N(0,1) draws."""
        rng = np.random.default_rng(0)
        y_sim = jnp.array(rng.normal(0, 1, (50, 100, 1)))
        obs = jnp.array(rng.normal(0, 1, (100, 1)))
        result = _compute_test_stats(y_sim, obs, ["x"])
        mean_stat = next(r for r in result if r.stat_name == "mean")
        # Observed mean should be within range of replicated means
        rep_min = min(mean_stat.rep_values)
        rep_max = max(mean_stat.rep_values)
        # Generous bounds since data is random
        assert rep_min < 0.5
        assert rep_max > -0.5

    def test_fallback_variable_name(self):
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_test_stats(y_sim, obs, ["x"])
        vars_seen = {r.variable for r in result}
        assert "x" in vars_seen
        assert "var_1" in vars_seen
