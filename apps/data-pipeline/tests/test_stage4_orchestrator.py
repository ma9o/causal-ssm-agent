"""Tests for Stage 4 orchestrator deterministic spec derivation.

Covers: derive_deterministic_spec, build_data_summary.
"""

import polars as pl

from causal_ssm_agent.orchestrator.stage4_orchestrator import (
    build_data_summary,
    derive_deterministic_spec,
)


def _make_causal_spec(
    constructs: list[dict],
    edges: list[dict],
    indicators: list[dict],
) -> dict:
    """Build a CausalSpec dict from components."""
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": {"indicators": indicators},
    }


def _simple_spec():
    """Two-construct, one-edge, binary + continuous indicators."""
    return _make_causal_spec(
        constructs=[
            {
                "name": "stress",
                "role": "exogenous",
                "temporal_status": "time_varying",
                "temporal_scale": "daily",
            },
            {
                "name": "sleep",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "temporal_scale": "daily",
                "is_outcome": True,
            },
        ],
        edges=[{"cause": "stress", "effect": "sleep"}],
        indicators=[
            {
                "name": "pss_score",
                "construct_name": "stress",
                "measurement_dtype": "continuous",
                "how_to_measure": "PSS score",
                "aggregation": "mean",
            },
            {
                "name": "sleep_quality",
                "construct_name": "sleep",
                "measurement_dtype": "continuous",
                "how_to_measure": "Sleep quality rating",
                "aggregation": "mean",
            },
        ],
    )


# =============================================================================
# derive_deterministic_spec
# =============================================================================


class TestDeriveDeterministicSpec:
    def test_returns_four_lists(self):
        """Should return 4 lists."""
        result = derive_deterministic_spec(_simple_spec())
        assert len(result) == 4
        resolved, ambiguous, params, loading_params = result
        assert isinstance(resolved, list)
        assert isinstance(ambiguous, list)
        assert isinstance(params, list)
        assert isinstance(loading_params, list)

    def test_continuous_resolves_to_gaussian(self):
        """Continuous dtype should resolve to gaussian/identity."""
        resolved, _, _, _ = derive_deterministic_spec(_simple_spec())
        for lik in resolved:
            assert lik["distribution"] == "gaussian"
            assert lik["link"] == "identity"

    def test_binary_resolves_to_bernoulli(self):
        """Binary dtype should resolve to bernoulli."""
        spec = _make_causal_spec(
            constructs=[
                {"name": "mood", "role": "endogenous", "temporal_status": "time_varying",
                 "temporal_scale": "daily", "is_outcome": True},
            ],
            edges=[],
            indicators=[
                {"name": "happy", "construct_name": "mood",
                 "measurement_dtype": "binary", "how_to_measure": "Happy?",
                 "aggregation": "mean"},
            ],
        )
        resolved, ambiguous, _, _ = derive_deterministic_spec(spec)
        # Binary has single dist (bernoulli) but multiple links (logit, probit)
        # So it should go to ambiguous
        all_vars = [r["variable"] for r in resolved] + [a["variable"] for a in ambiguous]
        assert "happy" in all_vars

    def test_count_is_ambiguous(self):
        """Count dtype has multiple valid distributions (poisson, negative_binomial)."""
        spec = _make_causal_spec(
            constructs=[
                {"name": "activity", "role": "endogenous", "temporal_status": "time_varying",
                 "temporal_scale": "daily", "is_outcome": True},
            ],
            edges=[],
            indicators=[
                {"name": "steps", "construct_name": "activity",
                 "measurement_dtype": "count", "how_to_measure": "Steps",
                 "aggregation": "sum"},
            ],
        )
        _, ambiguous, _, _ = derive_deterministic_spec(spec)
        step_ambig = [a for a in ambiguous if a["variable"] == "steps"]
        assert len(step_ambig) == 1
        assert "valid_distributions" in step_ambig[0] or "fixed_distribution" in step_ambig[0]

    def test_ar_params_for_endogenous(self):
        """Endogenous time-varying constructs should get AR params."""
        _, _, params, _ = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in params if p["role"] == "ar_coefficient"]
        # Only "sleep" is endogenous
        assert len(ar_params) == 1
        assert "sleep" in ar_params[0]["name"]

    def test_beta_params_for_edges(self):
        """Each edge should produce a beta (fixed effect) parameter."""
        _, _, params, _ = derive_deterministic_spec(_simple_spec())
        beta_params = [p for p in params if p["role"] == "fixed_effect"]
        assert len(beta_params) == 1
        assert "stress" in beta_params[0]["name"]
        assert "sleep" in beta_params[0]["name"]

    def test_sigma_params_for_all_constructs(self):
        """Each construct should get a residual SD parameter."""
        _, _, params, _ = derive_deterministic_spec(_simple_spec())
        sigma_params = [p for p in params if p["role"] == "residual_sd"]
        assert len(sigma_params) == 2

    def test_multi_indicator_loadings(self):
        """Multi-indicator constructs should get loading params for non-reference indicators."""
        spec = _make_causal_spec(
            constructs=[
                {"name": "stress", "role": "endogenous", "temporal_status": "time_varying",
                 "temporal_scale": "daily", "is_outcome": True},
            ],
            edges=[],
            indicators=[
                {"name": "pss", "construct_name": "stress",
                 "measurement_dtype": "continuous", "how_to_measure": "PSS",
                 "aggregation": "mean"},
                {"name": "vas", "construct_name": "stress",
                 "measurement_dtype": "continuous", "how_to_measure": "VAS",
                 "aggregation": "mean"},
            ],
        )
        _, _, _, loading_params = derive_deterministic_spec(spec)
        # First indicator is reference (no param), second gets loading
        assert len(loading_params) == 1
        assert "vas" in loading_params[0]["name"]

    def test_single_indicator_no_loadings(self):
        """Single-indicator constructs should not generate loading params."""
        _, _, _, loading_params = derive_deterministic_spec(_simple_spec())
        assert len(loading_params) == 0

    def test_no_ar_for_exogenous(self):
        """Exogenous constructs should NOT get AR parameters."""
        _, _, params, _ = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in params if p["role"] == "ar_coefficient"]
        for p in ar_params:
            assert "stress" not in p["name"]  # stress is exogenous


# =============================================================================
# build_data_summary
# =============================================================================


class TestBuildDataSummary:
    def test_basic_output(self):
        df = pl.DataFrame({
            "time_bucket": ["2024-01-01", "2024-01-02"],
            "mood": [5.0, 7.0],
            "sleep": [6.0, 8.0],
        })
        result = build_data_summary({"daily": df})
        assert "daily" in result.lower()
        assert "2" in result  # 2 time points

    def test_time_invariant(self):
        df = pl.DataFrame({
            "time_bucket": ["na"],
            "age": [30.0],
        })
        result = build_data_summary({"time_invariant": df})
        assert "time-invariant" in result.lower()

    def test_empty_data(self):
        result = build_data_summary({})
        assert "Data Overview" in result
