"""Tests for stage helper functions (pure data transformations).

Covers: build_raw_data_summary (stage4), build_causal_spec (agents).
"""

import polars as pl
import pytest

from causal_ssm_agent.flows.stages.stage4_model import build_raw_data_summary
from causal_ssm_agent.orchestrator.agents import build_causal_spec

# =============================================================================
# build_raw_data_summary
# =============================================================================


class TestBuildRawDataSummary:
    def test_empty_dataframe(self):
        df = pl.DataFrame({"indicator": [], "value": [], "timestamp": []})
        result = build_raw_data_summary(df)
        assert result == "No data available."

    def test_basic_summary(self):
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0, 1.0, 2.0],
                "indicator": ["x", "x", "x", "y", "y"],
                "value": [10.0, 20.0, 30.0, 5.0, 15.0],
            }
        )
        result = build_raw_data_summary(df)
        assert "Total observations: 5" in result
        assert "x:" in result
        assert "y:" in result
        assert "n=" in result
        assert "mean=" in result

    def test_time_bucket_column(self):
        df = pl.DataFrame(
            {
                "time_bucket": [1.0, 2.0],
                "indicator": ["x", "x"],
                "value": [10.0, 20.0],
            }
        )
        result = build_raw_data_summary(df)
        assert "time_bucket" in result

    def test_timestamp_column(self):
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "indicator": ["x", "x"],
                "value": [10.0, 20.0],
            }
        )
        result = build_raw_data_summary(df)
        assert "timestamp" in result

    def test_string_values(self):
        """String values should be cast to float for stats."""
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "indicator": ["x", "x"],
                "value": ["10.5", "20.3"],
            }
        )
        result = build_raw_data_summary(df)
        assert "x:" in result
        assert "mean=" in result

    def test_single_indicator(self):
        df = pl.DataFrame(
            {
                "timestamp": [1.0],
                "indicator": ["x"],
                "value": [42.0],
            }
        )
        result = build_raw_data_summary(df)
        assert "Total observations: 1" in result
        assert "x:" in result

    def test_sorted_indicators(self):
        """Indicators should appear in sorted order."""
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 1.0, 1.0],
                "indicator": ["z", "a", "m"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        result = build_raw_data_summary(df)
        lines = result.split("\n")
        indicator_lines = [line for line in lines if ":" in line and "n=" in line]
        names = [line.strip().split(":")[0] for line in indicator_lines]
        assert names == sorted(names)


# =============================================================================
# build_causal_spec
# =============================================================================


def _valid_latent():
    return {
        "constructs": [
            {
                "name": "stress",
                "description": "Perceived psychological stress",
                "role": "exogenous",
                "temporal_status": "time_varying",
                "temporal_scale": "daily",
            },
            {
                "name": "sleep",
                "description": "Sleep quality",
                "role": "endogenous",
                "is_outcome": True,
                "temporal_status": "time_varying",
                "temporal_scale": "daily",
            },
        ],
        "edges": [
            {
                "cause": "stress",
                "effect": "sleep",
                "description": "Stress impairs sleep",
            }
        ],
    }


def _valid_measurement():
    return {
        "indicators": [
            {
                "name": "pss",
                "construct_name": "stress",
                "how_to_measure": "Self-reported PSS score",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "hrs",
                "construct_name": "sleep",
                "how_to_measure": "Hours of sleep recorded",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


class TestBuildCausalSpec:
    def test_basic_combination(self):
        result = build_causal_spec(_valid_latent(), _valid_measurement())
        assert "latent" in result
        assert "measurement" in result
        assert result["latent"]["constructs"][0]["name"] == "stress"
        assert result["measurement"]["indicators"][0]["name"] == "pss"

    def test_includes_identifiability_when_provided(self):
        id_status = {
            "identifiable_treatments": {
                "stress": {
                    "method": "do_calculus",
                    "estimand": "P(sleep|do(stress))",
                    "marginalized_confounders": [],
                }
            },
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": ["stress", "sleep"],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 4,
            },
        }
        result = build_causal_spec(_valid_latent(), _valid_measurement(), id_status)
        assert result.get("identifiability") is not None

    def test_identifiability_none_when_omitted(self):
        result = build_causal_spec(_valid_latent(), _valid_measurement())
        assert result.get("identifiability") is None

    def test_invalid_latent_raises(self):
        with pytest.raises(ValueError):
            build_causal_spec({"constructs": []}, _valid_measurement())

    def test_roundtrip_preserves_data(self):
        lm = _valid_latent()
        mm = _valid_measurement()
        result = build_causal_spec(lm, mm)
        # Should be serializable dict
        assert isinstance(result, dict)
        assert len(result["latent"]["edges"]) == 1
        assert len(result["measurement"]["indicators"]) == 2
