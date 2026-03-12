"""Tests for causal_spec.py make_extraction_context.

Trivial accessor helpers (get_constructs, get_edges, get_indicators, etc.)
are exercised through higher-level tests. This file covers make_extraction_context
which performs real data transformation.
"""

from causal_ssm_agent.utils.causal_spec import make_extraction_context


def _full_spec():
    """Minimal valid CausalSpec dict."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous", "temporal_scale": "daily"},
                {
                    "name": "mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_scale": "daily",
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "mood", "description": "Stress affects mood"},
            ],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "pss_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Extract PSS score",
                },
                {
                    "name": "mood_rating",
                    "construct_name": "mood",
                    "measurement_dtype": "ordinal",
                    "how_to_measure": "Rate mood 1-5",
                },
            ],
        },
    }


# =============================================================================
# make_extraction_context
# =============================================================================


class TestMakeExtractionContext:
    def test_strips_to_worker_fields(self):
        spec = _full_spec()
        # Add extra fields that workers don't need
        spec["measurement"]["indicators"][0]["aggregation"] = "mean"
        spec["measurement"]["indicators"][0]["construct_name"] = "stress"
        spec["measurement"]["indicators"][0]["source_columns"] = ["pss_col"]
        ctx = make_extraction_context(spec)
        ind = ctx["measurement"]["indicators"][0]
        assert set(ind.keys()) == {"name", "measurement_dtype", "how_to_measure", "source_columns"}
        assert "aggregation" not in ind
        assert "construct_name" not in ind

    def test_outcome_slimmed_to_name_and_description(self):
        spec = _full_spec()
        ctx = make_extraction_context(spec)
        constructs = ctx["latent"]["constructs"]
        assert len(constructs) == 1
        outcome = constructs[0]
        assert set(outcome.keys()) == {"name", "description"}

    def test_no_outcome_gives_empty_constructs(self):
        spec = {
            "latent": {"constructs": [{"name": "x", "role": "exogenous"}]},
            "measurement": {"indicators": [{"name": "ind", "measurement_dtype": "continuous"}]},
        }
        ctx = make_extraction_context(spec)
        assert ctx["latent"]["constructs"] == []

    def test_source_columns_included_when_present(self):
        spec = _full_spec()
        spec["measurement"]["indicators"][0]["source_columns"] = ["col_a", "col_b"]
        ctx = make_extraction_context(spec)
        assert ctx["measurement"]["indicators"][0]["source_columns"] == ["col_a", "col_b"]
