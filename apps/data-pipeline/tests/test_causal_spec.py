"""Tests for causal_spec.py accessor helpers."""

from causal_ssm_agent.utils.causal_spec import (
    get_constructs,
    get_edges,
    get_indicator_dtypes,
    get_indicator_info,
    get_indicators,
    get_outcome_construct,
)

SAMPLE_SPEC = {
    "latent": {
        "constructs": [
            {"name": "X", "is_outcome": False},
            {"name": "Y", "is_outcome": True},
        ],
        "edges": [{"cause": "X", "effect": "Y"}],
    },
    "measurement": {
        "indicators": [
            {"name": "x1", "construct_name": "X", "measurement_dtype": "continuous"},
            {"name": "y1", "construct_name": "Y", "measurement_dtype": "binary"},
        ]
    },
}


class TestGetConstructs:
    def test_returns_constructs(self):
        assert len(get_constructs(SAMPLE_SPEC)) == 2

    def test_empty_on_missing(self):
        assert get_constructs({}) == []


class TestGetEdges:
    def test_returns_edges(self):
        edges = get_edges(SAMPLE_SPEC)
        assert len(edges) == 1
        assert edges[0]["cause"] == "X"

    def test_empty_on_missing(self):
        assert get_edges({}) == []


class TestGetIndicators:
    def test_returns_indicators(self):
        assert len(get_indicators(SAMPLE_SPEC)) == 2

    def test_empty_on_missing(self):
        assert get_indicators({}) == []


class TestGetIndicatorInfo:
    def test_returns_mapping(self):
        info = get_indicator_info(SAMPLE_SPEC)
        assert "x1" in info
        assert info["x1"]["dtype"] == "continuous"
        assert info["y1"]["construct_name"] == "Y"


class TestGetIndicatorDtypes:
    def test_returns_dtypes(self):
        dtypes = get_indicator_dtypes(SAMPLE_SPEC)
        assert dtypes["x1"] == "continuous"
        assert dtypes["y1"] == "binary"

    def test_defaults_to_continuous(self):
        spec = {"measurement": {"indicators": [{"name": "z"}]}}
        dtypes = get_indicator_dtypes(spec)
        assert dtypes["z"] == "continuous"


class TestGetOutcomeConstruct:
    def test_finds_outcome_from_causal_spec(self):
        result = get_outcome_construct(SAMPLE_SPEC)
        assert result is not None
        assert result["name"] == "Y"

    def test_finds_outcome_from_bare_latent(self):
        latent = SAMPLE_SPEC["latent"]
        result = get_outcome_construct(latent)
        assert result is not None
        assert result["name"] == "Y"

    def test_returns_none_when_missing(self):
        spec = {"latent": {"constructs": [{"name": "X"}]}}
        assert get_outcome_construct(spec) is None

    def test_returns_none_on_empty(self):
        assert get_outcome_construct({}) is None
