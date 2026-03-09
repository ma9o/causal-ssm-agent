"""Tests for causal_spec.py accessor helpers.

Covers: get_constructs, get_edges, get_indicators, get_indicator_info,
        get_indicator_dtypes, get_outcome_construct.
"""

from causal_ssm_agent.utils.causal_spec import (
    get_constructs,
    get_edges,
    get_indicator_dtypes,
    get_indicator_info,
    get_indicators,
    get_outcome_construct,
)


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
# get_constructs
# =============================================================================


class TestGetConstructs:
    def test_returns_constructs(self):
        result = get_constructs(_full_spec())
        assert len(result) == 2
        assert result[0]["name"] == "stress"

    def test_empty_spec(self):
        assert get_constructs({}) == []

    def test_missing_constructs_key(self):
        assert get_constructs({"latent": {}}) == []

    def test_missing_latent_key(self):
        assert get_constructs({"measurement": {}}) == []


# =============================================================================
# get_edges
# =============================================================================


class TestGetEdges:
    def test_returns_edges(self):
        result = get_edges(_full_spec())
        assert len(result) == 1
        assert result[0]["cause"] == "stress"

    def test_empty_spec(self):
        assert get_edges({}) == []

    def test_no_edges(self):
        spec = {"latent": {"constructs": [{"name": "x"}]}}
        assert get_edges(spec) == []


# =============================================================================
# get_indicators
# =============================================================================


class TestGetIndicators:
    def test_returns_indicators(self):
        result = get_indicators(_full_spec())
        assert len(result) == 2
        names = {ind["name"] for ind in result}
        assert names == {"pss_score", "mood_rating"}

    def test_empty_spec(self):
        assert get_indicators({}) == []

    def test_missing_measurement_key(self):
        assert get_indicators({"latent": {}}) == []

    def test_missing_indicators_key(self):
        assert get_indicators({"measurement": {}}) == []


# =============================================================================
# get_indicator_info
# =============================================================================


class TestGetIndicatorInfo:
    def test_returns_info_dict(self):
        result = get_indicator_info(_full_spec())
        assert "pss_score" in result
        assert result["pss_score"]["dtype"] == "continuous"
        assert result["pss_score"]["construct_name"] == "stress"

    def test_ordinal_dtype(self):
        result = get_indicator_info(_full_spec())
        assert result["mood_rating"]["dtype"] == "ordinal"
        assert result["mood_rating"]["construct_name"] == "mood"

    def test_empty_spec(self):
        assert get_indicator_info({}) == {}

    def test_missing_dtype_is_none(self):
        spec = {
            "measurement": {
                "indicators": [{"name": "x", "construct_name": "c"}],
            }
        }
        result = get_indicator_info(spec)
        assert result["x"]["dtype"] is None


# =============================================================================
# get_indicator_dtypes
# =============================================================================


class TestGetIndicatorDtypes:
    def test_returns_dtype_mapping(self):
        result = get_indicator_dtypes(_full_spec())
        assert result == {"pss_score": "continuous", "mood_rating": "ordinal"}

    def test_defaults_to_continuous(self):
        spec = {
            "measurement": {
                "indicators": [{"name": "x"}],
            }
        }
        result = get_indicator_dtypes(spec)
        assert result["x"] == "continuous"

    def test_empty_spec(self):
        assert get_indicator_dtypes({}) == {}


# =============================================================================
# get_outcome_construct
# =============================================================================


class TestGetOutcomeConstruct:
    def test_from_full_spec(self):
        result = get_outcome_construct(_full_spec())
        assert result is not None
        assert result["name"] == "mood"
        assert result["is_outcome"] is True

    def test_from_bare_latent_model(self):
        latent = _full_spec()["latent"]
        result = get_outcome_construct(latent)
        assert result is not None
        assert result["name"] == "mood"

    def test_no_outcome(self):
        spec = {
            "latent": {
                "constructs": [
                    {"name": "stress", "role": "exogenous"},
                ],
            }
        }
        assert get_outcome_construct(spec) is None

    def test_empty_spec(self):
        assert get_outcome_construct({}) is None

    def test_bare_latent_no_outcome(self):
        latent = {"constructs": [{"name": "x"}]}
        assert get_outcome_construct(latent) is None

    def test_is_outcome_false(self):
        spec = {
            "latent": {
                "constructs": [{"name": "x", "is_outcome": False}],
            }
        }
        assert get_outcome_construct(spec) is None
