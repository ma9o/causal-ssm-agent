"""Tests for utils/causal_spec.py accessor helpers.

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
    """Minimal full CausalSpec dict."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous"},
                {"name": "sleep", "role": "endogenous", "is_outcome": True},
            ],
            "edges": [
                {"cause": "stress", "effect": "sleep"},
            ],
        },
        "measurement": {
            "indicators": [
                {"name": "pss", "construct_name": "stress", "measurement_dtype": "continuous"},
                {"name": "hrs", "construct_name": "sleep", "measurement_dtype": "continuous"},
            ]
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

    def test_missing_edges_key(self):
        assert get_edges({"latent": {"constructs": []}}) == []


# =============================================================================
# get_indicators
# =============================================================================


class TestGetIndicators:
    def test_returns_indicators(self):
        result = get_indicators(_full_spec())
        assert len(result) == 2
        assert result[0]["name"] == "pss"

    def test_empty_spec(self):
        assert get_indicators({}) == []

    def test_missing_indicators_key(self):
        assert get_indicators({"measurement": {}}) == []


# =============================================================================
# get_indicator_info
# =============================================================================


class TestGetIndicatorInfo:
    def test_returns_mapping(self):
        result = get_indicator_info(_full_spec())
        assert "pss" in result
        assert result["pss"]["dtype"] == "continuous"
        assert result["pss"]["construct_name"] == "stress"

    def test_empty_spec(self):
        assert get_indicator_info({}) == {}

    def test_missing_optional_fields(self):
        spec = {"measurement": {"indicators": [{"name": "x"}]}}
        result = get_indicator_info(spec)
        assert result["x"]["dtype"] is None
        assert result["x"]["construct_name"] is None


# =============================================================================
# get_indicator_dtypes
# =============================================================================


class TestGetIndicatorDtypes:
    def test_returns_mapping(self):
        result = get_indicator_dtypes(_full_spec())
        assert result == {"pss": "continuous", "hrs": "continuous"}

    def test_empty_spec(self):
        assert get_indicator_dtypes({}) == {}

    def test_defaults_to_continuous(self):
        spec = {"measurement": {"indicators": [{"name": "x"}]}}
        result = get_indicator_dtypes(spec)
        assert result["x"] == "continuous"

    def test_binary_dtype(self):
        spec = {"measurement": {"indicators": [{"name": "smoke", "measurement_dtype": "binary"}]}}
        result = get_indicator_dtypes(spec)
        assert result["smoke"] == "binary"


# =============================================================================
# get_outcome_construct
# =============================================================================


class TestGetOutcomeConstruct:
    def test_finds_outcome_from_full_spec(self):
        result = get_outcome_construct(_full_spec())
        assert result is not None
        assert result["name"] == "sleep"

    def test_finds_outcome_from_bare_latent(self):
        latent = {
            "constructs": [
                {"name": "Y", "is_outcome": True},
            ],
        }
        result = get_outcome_construct(latent)
        assert result is not None
        assert result["name"] == "Y"

    def test_no_outcome_returns_none(self):
        spec = {"latent": {"constructs": [{"name": "X", "role": "exogenous"}]}}
        assert get_outcome_construct(spec) is None

    def test_empty_spec_returns_none(self):
        assert get_outcome_construct({}) is None

    def test_is_outcome_false(self):
        spec = {"latent": {"constructs": [{"name": "X", "is_outcome": False}]}}
        assert get_outcome_construct(spec) is None
