"""Tests for structural model scoring function."""

import json
from dataclasses import dataclass

import pytest

from causal_agent.orchestrator.scoring import (
    _count_rule_points,
    score_structural_model,
    score_structural_model_normalized,
)
from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Construct,
    Role,
    StructuralModel,
    TemporalStatus,
)


@dataclass
class MockPrediction:
    """Mock DSPy prediction object."""

    structure: str


class TestScoreStructuralModel:
    """Tests for the main scoring function."""

    def test_invalid_json_returns_zero(self):
        """Invalid JSON should return 0."""
        pred = MockPrediction(structure="not valid json")
        assert score_structural_model(None, pred) == 0.0

    def test_missing_structure_field_returns_zero(self):
        """Missing structure field should return 0."""

        class BadPred:
            pass

        assert score_structural_model(None, BadPred()) == 0.0

    def test_schema_validation_failure_returns_zero(self):
        """Schema validation failure should return 0."""
        # Missing outcome
        pred = MockPrediction(structure='{"constructs": [], "edges": []}')
        assert score_structural_model(None, pred) == 0.0

        # Invalid: time_varying without causal_granularity
        invalid = {
            "constructs": [
                {
                    "name": "mood",
                    "description": "test",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    # missing causal_granularity
                }
            ],
            "edges": [],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structural_model(None, pred) == 0.0

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
                    "causal_granularity": "daily",
                }
            ],
            "edges": [{"cause": "nonexistent", "effect": "mood", "description": "Test"}],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structural_model(None, pred) == 0.0

    def test_valid_simple_structure_scores_positive(self):
        """Valid simple structure should score > 0."""
        valid = {
            "constructs": [
                {
                    "name": "stress",
                    "description": "Daily stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "mood",
                    "description": "Daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
            ],
            "edges": [{"cause": "stress", "effect": "mood", "description": "Stress affects mood", "lagged": False}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structural_model(None, pred)
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
                    "causal_granularity": "daily",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
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
                    "causal_granularity": "hourly",
                },
                {
                    "name": "sleep",
                    "description": "daily sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "mood",
                    "description": "daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
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
                {"cause": "sleep", "effect": "mood", "description": "Sleep affects mood", "lagged": False},
                {"cause": "mood", "effect": "sleep", "description": "Mood affects sleep"},
            ],
        }

        simple_score = score_structural_model(None, MockPrediction(json.dumps(simple)))
        complex_score = score_structural_model(None, MockPrediction(json.dumps(complex_struct)))

        assert complex_score > simple_score


class TestCountRulePoints:
    """Tests for the internal point counting function."""

    def _make_construct(self, name, granularity, role, is_outcome=False):
        """Helper to create a construct."""
        temporal_status = TemporalStatus.TIME_VARYING if granularity else TemporalStatus.TIME_INVARIANT
        return Construct(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            temporal_status=temporal_status,
            causal_granularity=granularity,
        )

    def _make_structure(self, constructs, edges):
        """Helper to create StructuralModel."""
        return StructuralModel(constructs=constructs, edges=edges)

    def test_endogenous_time_varying_construct_points(self):
        """Endogenous time-varying construct with all correct fields scores points."""
        structure = self._make_structure(
            constructs=[self._make_construct("mood", "daily", Role.ENDOGENOUS, is_outcome=True)],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 role, +1 temporal_status, +1 causal_granularity present, +1 valid causal_granularity
        assert points >= 4

    def test_time_invariant_construct_points(self):
        """Time-invariant construct scores points for correct constraints."""
        structure = self._make_structure(
            constructs=[
                self._make_construct("age", None, Role.EXOGENOUS),
                self._make_construct("mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # age: +1 role, +1 temporal_status, +1 no granularity = 3
        # mood: +1 role, +1 temporal_status, +1 granularity, +1 valid granularity = 4
        assert points >= 7

    def test_edge_points(self):
        """Edge between valid constructs scores points."""
        structure = self._make_structure(
            constructs=[
                self._make_construct("X", "daily", Role.EXOGENOUS),
                self._make_construct("Y", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="X", effect="Y", description="X causes Y")],
        )
        points = _count_rule_points(structure)
        # Construct points (4+4=8) + edge points (cause exists, effect exists, effect endogenous, same timescale)
        assert points >= 8 + 4

    def test_cross_timescale_edge_bonus(self):
        """Cross-timescale edge gets bonus points."""
        structure = self._make_structure(
            constructs=[
                self._make_construct("hourly_stress", "hourly", Role.EXOGENOUS),
                self._make_construct("daily_mood", "daily", Role.ENDOGENOUS, is_outcome=True),
            ],
            edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", description="Stress affects mood")],
        )
        points = _count_rule_points(structure)
        # Cross-timescale gives +2 instead of +1
        assert points >= 8 + 5  # constructs + edge with bonus


class TestNormalizedScoring:
    """Tests for normalized scoring function."""

    def test_invalid_returns_zero(self):
        """Invalid structure returns 0."""
        pred = MockPrediction(structure="invalid")
        assert score_structural_model_normalized(None, pred) == 0.0

    def test_valid_returns_between_zero_and_one(self):
        """Valid structure returns score in [0, 1]."""
        valid = {
            "constructs": [
                {
                    "name": "X",
                    "description": "input",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structural_model_normalized(None, pred)
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
                    "causal_granularity": "daily",
                },
                {
                    "name": "weather",
                    "description": "input",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
            ],
            "edges": [{"cause": "mood", "effect": "weather", "description": "Invalid"}],  # Invalid: exogenous effect
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structural_model(None, pred) == 0.0


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
                    "causal_granularity": "hourly",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
            ],
            "edges": [
                {"cause": "hourly_stress", "effect": "daily_mood", "description": "Hourly affects daily"}
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_structural_model(None, pred) > 0.0

    def test_coarser_to_finer_valid(self):
        """Coarser->finer edge is valid."""
        valid = {
            "constructs": [
                {
                    "name": "weekly_stress",
                    "description": "weekly",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "causal_granularity": "weekly",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
            ],
            "edges": [
                {"cause": "weekly_stress", "effect": "daily_mood", "description": "Weekly affects daily"}
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_structural_model(None, pred) > 0.0
