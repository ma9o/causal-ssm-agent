"""Tests for CausalSpec accessor helpers.

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


def _make_spec():
    """Minimal CausalSpec dict for testing."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "is_outcome": False},
                {"name": "sleep", "is_outcome": True},
            ],
            "edges": [
                {"from": "stress", "to": "sleep"},
            ],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "pss_score",
                    "measurement_dtype": "continuous",
                    "construct_name": "stress",
                },
                {
                    "name": "sleep_quality",
                    "measurement_dtype": "ordinal",
                    "construct_name": "sleep",
                },
            ],
        },
    }


# =============================================================================
# get_constructs
# =============================================================================


class TestGetConstructs:
    def test_returns_constructs(self):
        spec = _make_spec()
        result = get_constructs(spec)
        assert len(result) == 2
        assert result[0]["name"] == "stress"

    def test_empty_spec(self):
        assert get_constructs({}) == []

    def test_no_latent_key(self):
        assert get_constructs({"measurement": {}}) == []

    def test_no_constructs_key(self):
        assert get_constructs({"latent": {}}) == []


# =============================================================================
# get_edges
# =============================================================================


class TestGetEdges:
    def test_returns_edges(self):
        spec = _make_spec()
        result = get_edges(spec)
        assert len(result) == 1
        assert result[0]["from"] == "stress"
        assert result[0]["to"] == "sleep"

    def test_empty_spec(self):
        assert get_edges({}) == []

    def test_no_edges(self):
        assert get_edges({"latent": {"constructs": []}}) == []


# =============================================================================
# get_indicators
# =============================================================================


class TestGetIndicators:
    def test_returns_indicators(self):
        spec = _make_spec()
        result = get_indicators(spec)
        assert len(result) == 2
        assert result[0]["name"] == "pss_score"

    def test_empty_spec(self):
        assert get_indicators({}) == []

    def test_no_measurement_key(self):
        assert get_indicators({"latent": {}}) == []


# =============================================================================
# get_indicator_info
# =============================================================================


class TestGetIndicatorInfo:
    def test_extracts_info(self):
        spec = _make_spec()
        info = get_indicator_info(spec)
        assert "pss_score" in info
        assert info["pss_score"]["dtype"] == "continuous"
        assert info["pss_score"]["construct_name"] == "stress"

    def test_maps_all_indicators(self):
        spec = _make_spec()
        info = get_indicator_info(spec)
        assert len(info) == 2
        assert "sleep_quality" in info

    def test_empty_spec(self):
        assert get_indicator_info({}) == {}

    def test_missing_dtype_returns_none(self):
        spec = {
            "measurement": {
                "indicators": [{"name": "x"}],
            }
        }
        info = get_indicator_info(spec)
        assert info["x"]["dtype"] is None


# =============================================================================
# get_indicator_dtypes
# =============================================================================


class TestGetIndicatorDtypes:
    def test_extracts_dtypes(self):
        spec = _make_spec()
        dtypes = get_indicator_dtypes(spec)
        assert dtypes["pss_score"] == "continuous"
        assert dtypes["sleep_quality"] == "ordinal"

    def test_default_dtype_is_continuous(self):
        spec = {
            "measurement": {
                "indicators": [{"name": "x"}],
            }
        }
        dtypes = get_indicator_dtypes(spec)
        assert dtypes["x"] == "continuous"

    def test_empty_spec(self):
        assert get_indicator_dtypes({}) == {}


# =============================================================================
# get_outcome_construct
# =============================================================================


class TestGetOutcomeConstruct:
    def test_finds_outcome_in_full_spec(self):
        spec = _make_spec()
        outcome = get_outcome_construct(spec)
        assert outcome is not None
        assert outcome["name"] == "sleep"

    def test_finds_outcome_in_bare_latent(self):
        latent = {
            "constructs": [
                {"name": "a", "is_outcome": False},
                {"name": "b", "is_outcome": True},
            ],
        }
        outcome = get_outcome_construct(latent)
        assert outcome is not None
        assert outcome["name"] == "b"

    def test_no_outcome_returns_none(self):
        spec = {
            "latent": {
                "constructs": [{"name": "x", "is_outcome": False}],
            }
        }
        assert get_outcome_construct(spec) is None

    def test_empty_spec_returns_none(self):
        assert get_outcome_construct({}) is None

    def test_empty_constructs_returns_none(self):
        spec = {"latent": {"constructs": []}}
        assert get_outcome_construct(spec) is None
