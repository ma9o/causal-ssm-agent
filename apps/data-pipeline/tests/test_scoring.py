"""Tests for latent model scoring function."""

import json

from causal_ssm_agent.orchestrator.schemas import (
    CausalEdge,
    LatentModel,
    Role,
)
from causal_ssm_agent.orchestrator.scoring import (
    _count_rule_points,
    score_latent_model,
    score_latent_model_normalized,
)
from tests.helpers import MockPrediction


class TestScoreLatentModel:
    """Tests for the main scoring function."""

    def test_invalid_json_returns_zero(self):
        """Invalid JSON should return 0."""
        pred = MockPrediction(structure="not valid json")
        assert score_latent_model(None, pred) == 0.0

    def test_missing_structure_field_returns_zero(self):
        """Missing structure field should return 0."""

        class BadPred:
            pass

        assert score_latent_model(None, BadPred()) == 0.0

    def test_schema_validation_failure_returns_zero(self):
        """Schema validation failure should return 0."""
        # Missing outcome
        pred = MockPrediction(structure='{"constructs": [], "edges": []}')
        assert score_latent_model(None, pred) == 0.0

        # Invalid: time_varying without temporal_scale
        invalid = {
            "constructs": [
                {
                    "name": "mood",
                    "description": "test",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    # missing temporal_scale
                }
            ],
            "edges": [],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_latent_model(None, pred) == 0.0

    def test_invalid_edge_returns_zero(self):
        """Edge referencing non-existent construct should return 0."""
        invalid = {
            "constructs": [
                {
                    "name": "mood",
                    "description": "test",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                }
            ],
            "edges": [{"cause": "nonexistent", "effect": "mood", "description": "Test"}],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_latent_model(None, pred) == 0.0

    def test_valid_simple_structure_scores_positive(self):
        """Valid simple structure should score > 0."""
        valid = {
            "constructs": [
                {
                    "name": "stress",
                    "description": "Daily stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "mood",
                    "description": "Daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [
                {
                    "cause": "stress",
                    "effect": "mood",
                    "description": "Stress affects mood",
                    "lagged": False,
                }
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_latent_model(None, pred)
        assert score > 0

    def test_complex_structure_scores_higher(self):
        """More complex valid structure should score higher."""
        simple = {
            "constructs": [
                {
                    "name": "X",
                    "description": "input",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }

        complex_struct = {
            "constructs": [
                {
                    "name": "stress",
                    "description": "hourly stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "hourly",
                },
                {
                    "name": "sleep",
                    "description": "daily sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "mood",
                    "description": "daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "age",
                    "description": "participant age",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "mood", "description": "Stress affects mood"},
                {
                    "cause": "sleep",
                    "effect": "mood",
                    "description": "Sleep affects mood",
                },
                {"cause": "mood", "effect": "sleep", "description": "Mood affects sleep"},
            ],
        }

        simple_score = score_latent_model(None, MockPrediction(json.dumps(simple)))
        complex_score = score_latent_model(None, MockPrediction(json.dumps(complex_struct)))

        assert complex_score > simple_score


class TestCountRulePoints:
    """Tests for the internal point counting function."""

    def test_endogenous_time_varying_construct_points(self, construct_factory):
        """Endogenous time-varying construct with all correct fields scores points."""
        structure = LatentModel(
            constructs=[
                construct_factory("stress", "daily", Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="stress", effect="mood", description="Test")],
        )
        points = _count_rule_points(structure)
        # +1 role, +1 temporal_status, +1 temporal_scale present, +1 valid temporal_scale (per construct)
        assert points >= 4

    def test_time_invariant_construct_points(self, construct_factory):
        """Time-invariant construct scores points for correct constraints."""
        structure = LatentModel(
            constructs=[
                construct_factory("age", None, Role.EXOGENOUS),
                construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="age", effect="mood", description="Test")],
        )
        points = _count_rule_points(structure)
        # age: +1 role, +1 temporal_status, +1 no granularity = 3
        # mood: +1 role, +1 temporal_status, +1 granularity, +1 valid granularity = 4
        assert points >= 7

    def test_edge_points(self, construct_factory):
        """Edge between valid constructs scores points."""
        structure = LatentModel(
            constructs=[
                construct_factory("X", "daily", Role.EXOGENOUS),
                construct_factory("Y", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="X", effect="Y", description="X causes Y")],
        )
        points = _count_rule_points(structure)
        # Construct points (4+4=8) + edge points (cause exists, effect exists, effect endogenous, same timescale)
        assert points >= 8 + 4

    def test_cross_timescale_edge_bonus(self, construct_factory):
        """Cross-timescale edge gets bonus points."""
        structure = LatentModel(
            constructs=[
                construct_factory("hourly_stress", "hourly", Role.EXOGENOUS),
                construct_factory("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[
                CausalEdge(
                    cause="hourly_stress", effect="daily_mood", description="Stress affects mood"
                )
            ],
        )
        points = _count_rule_points(structure)
        # Cross-timescale gives +2 instead of +1
        assert points >= 8 + 5  # constructs + edge with bonus


class TestNormalizedScoring:
    """Tests for normalized scoring function."""

    def test_invalid_returns_zero(self):
        """Invalid structure returns 0."""
        pred = MockPrediction(structure="invalid")
        assert score_latent_model_normalized(None, pred) == 0.0

    def test_valid_returns_between_zero_and_one(self):
        """Valid structure returns score in [0, 1]."""
        valid = {
            "constructs": [
                {
                    "name": "X",
                    "description": "input",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_latent_model_normalized(None, pred)
        assert 0 < score <= 1.0


class TestExogenousEffectViolation:
    """Test that exogenous constructs as effects return 0."""

    def test_exogenous_as_effect_returns_zero(self):
        """Exogenous construct as edge effect should fail validation."""
        invalid = {
            "constructs": [
                {
                    "name": "mood",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "weather",
                    "description": "input",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [
                {"cause": "mood", "effect": "weather", "description": "Invalid"}
            ],  # Invalid: exogenous effect
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_latent_model(None, pred) == 0.0


class TestCrossScaleEdges:
    """Test cross-scale edges are valid."""

    def test_finer_to_coarser_valid(self):
        """Finer->coarser edge is valid."""
        valid = {
            "constructs": [
                {
                    "name": "hourly_stress",
                    "description": "hourly",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "hourly",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [
                {
                    "cause": "hourly_stress",
                    "effect": "daily_mood",
                    "description": "Hourly affects daily",
                }
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_latent_model(None, pred) > 0.0

    def test_coarser_to_finer_valid(self):
        """Coarser->finer edge is valid."""
        valid = {
            "constructs": [
                {
                    "name": "weekly_stress",
                    "description": "weekly",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "weekly",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
            ],
            "edges": [
                {
                    "cause": "weekly_stress",
                    "effect": "daily_mood",
                    "description": "Weekly affects daily",
                }
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_latent_model(None, pred) > 0.0
