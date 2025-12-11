"""Tests for DSPy scoring function."""

import json
from dataclasses import dataclass

import pytest

from causal_agent.orchestrator.scoring import (
    _count_rule_points,
    score_structure_proposal,
    score_structure_proposal_normalized,
)
from causal_agent.orchestrator.schemas import (
    CausalEdge,
    Dimension,
    DSEMStructure,
    Observability,
    Role,
    TemporalStatus,
)


@dataclass
class MockPrediction:
    """Mock DSPy prediction object."""

    structure: str


class TestScoreStructureProposal:
    """Tests for the main scoring function."""

    def test_invalid_json_returns_zero(self):
        """Invalid JSON should return 0."""
        pred = MockPrediction(structure="not valid json")
        assert score_structure_proposal(None, pred) == 0.0

    def test_missing_structure_field_returns_zero(self):
        """Missing structure field should return 0."""

        class BadPred:
            pass

        assert score_structure_proposal(None, BadPred()) == 0.0

    def test_schema_validation_failure_returns_zero(self):
        """Schema validation failure should return 0."""
        # Missing required fields
        pred = MockPrediction(structure='{"dimensions": [], "edges": []}')
        # This is actually valid (empty structure)
        assert score_structure_proposal(None, pred) == 0.0  # No points for empty

        # Invalid: time_varying without causal_granularity
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "test",
                    "role": "endogenous",
                    "observability": "observed",
                    "temporal_status": "time_varying",
                    "measurement_dtype": "continuous",
                    # missing causal_granularity
                }
            ],
            "edges": [],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0

    def test_invalid_edge_returns_zero(self):
        """Edge referencing non-existent variable should return 0."""
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "test",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
            "edges": [{"cause": "nonexistent", "effect": "mood", "description": "Test"}],
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0

    def test_valid_simple_structure_scores_positive(self):
        """Valid simple structure should score > 0."""
        valid = {
            "dimensions": [
                {
                    "name": "stress",
                    "description": "Daily stress",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract stress from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "mood",
                    "description": "Daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "stress", "effect": "mood", "description": "Stress affects mood", "lagged": False}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structure_proposal(None, pred)
        assert score > 0

    def test_complex_structure_scores_higher(self):
        """More complex valid structure should score higher."""
        simple = {
            "dimensions": [
                {
                    "name": "X",
                    "description": "input",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract X from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract Y from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }

        complex_struct = {
            "dimensions": [
                {
                    "name": "stress",
                    "description": "hourly stress",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract stress from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "hourly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep",
                    "description": "daily sleep",
                    "role": "endogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract sleep from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "mood",
                    "description": "daily mood",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "age",
                    "description": "participant age",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract age from data",
                    "temporal_status": "time_invariant",
                    "measurement_dtype": "continuous",
                },
                {
                    "name": "intercept",
                    "description": "person baseline",
                    "role": "exogenous",
                    "observability": "latent",
                    "temporal_status": "time_invariant",
                    "measurement_dtype": "continuous",
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "mood", "description": "Stress affects mood", "aggregation": "mean"},
                {"cause": "sleep", "effect": "mood", "description": "Sleep affects mood", "lagged": False},
                {"cause": "mood", "effect": "sleep", "description": "Mood affects sleep"},
            ],
        }

        simple_score = score_structure_proposal(None, MockPrediction(json.dumps(simple)))
        complex_score = score_structure_proposal(None, MockPrediction(json.dumps(complex_struct)))

        assert complex_score > simple_score


class TestCountRulePoints:
    """Tests for the internal point counting function."""

    def _make_structure(self, dims, edges):
        """Helper to create DSEMStructure."""
        return DSEMStructure(dimensions=dims, edges=edges)

    def test_endogenous_time_varying_dimension_points(self):
        """Endogenous time-varying dimension with all correct fields scores points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="mood",
                    description="test",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract mood from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                )
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 role, +1 observability, +1 temporal_status, +1 granularity present,
        # +1 valid granularity, +1 aggregation present, +1 valid aggregation, +1 valid dtype
        assert points >= 8

    def test_time_invariant_dimension_points(self):
        """Time-invariant dimension scores points for correct constraints."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="age",
                    description="test",
                    role=Role.EXOGENOUS,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract age from data",
                    temporal_status=TemporalStatus.TIME_INVARIANT,
                    measurement_dtype="continuous",
                ),
                Dimension(
                    name="mood",
                    description="outcome",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract mood from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 role, +1 observability, +1 temporal_status, +1 no granularity, +1 no aggregation, +1 valid dtype
        assert points >= 6

    def test_latent_variable_bonus(self):
        """Latent variable gets bonus point for modeling heterogeneity."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="intercept",
                    description="test",
                    role=Role.EXOGENOUS,
                    observability=Observability.LATENT,
                    temporal_status=TemporalStatus.TIME_INVARIANT,
                    measurement_dtype="continuous",
                ),
                Dimension(
                    name="mood",
                    description="outcome",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract mood from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[],
        )
        points = _count_rule_points(structure)
        # +1 role, +1 observability, +1 temporal_status, +1 no granularity, +1 no aggregation, +1 valid dtype, +1 latent bonus
        assert points >= 7

    def test_edge_points(self):
        """Edge between valid dimensions scores points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="X",
                    description="input",
                    role=Role.EXOGENOUS,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract X from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                Dimension(
                    name="Y",
                    description="outcome",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract Y from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[CausalEdge(cause="X", effect="Y", description="X causes Y")],
        )
        points = _count_rule_points(structure)
        # Dimension points + edge points (cause exists, effect exists, effect endogenous, same timescale)
        assert points >= 16 + 4

    def test_cross_timescale_edge_bonus(self):
        """Cross-timescale edge gets bonus points."""
        structure = self._make_structure(
            dims=[
                Dimension(
                    name="hourly_stress",
                    description="hourly",
                    role=Role.EXOGENOUS,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract stress from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="hourly",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                Dimension(
                    name="daily_mood",
                    description="daily",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    observability=Observability.OBSERVED,
                    how_to_measure="Extract mood from data",
                    temporal_status=TemporalStatus.TIME_VARYING,
                    causal_granularity="daily",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
            edges=[CausalEdge(cause="hourly_stress", effect="daily_mood", description="Stress affects mood", aggregation="mean")],
        )
        points = _count_rule_points(structure)
        # Cross-timescale gives +2 instead of +1
        assert points >= 16 + 5  # dims + edge with bonus


class TestNormalizedScoring:
    """Tests for normalized scoring function."""

    def test_invalid_returns_zero(self):
        """Invalid structure returns 0."""
        pred = MockPrediction(structure="invalid")
        assert score_structure_proposal_normalized(None, pred) == 0.0

    def test_valid_returns_between_zero_and_one(self):
        """Valid structure returns score in [0, 1]."""
        valid = {
            "dimensions": [
                {
                    "name": "X",
                    "description": "input",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract X from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract Y from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "X", "effect": "Y", "description": "X causes Y"}],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        score = score_structure_proposal_normalized(None, pred)
        assert 0 < score <= 1.0


class TestExogenousEffectViolation:
    """Test that exogenous variables as effects return 0."""

    def test_exogenous_as_effect_returns_zero(self):
        """Exogenous variable as edge effect should fail validation."""
        invalid = {
            "dimensions": [
                {
                    "name": "mood",
                    "description": "outcome",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "weather",
                    "description": "input",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract weather from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [{"cause": "mood", "effect": "weather", "description": "Invalid"}],  # Invalid: exogenous effect
        }
        pred = MockPrediction(structure=json.dumps(invalid))
        assert score_structure_proposal(None, pred) == 0.0


class TestCrossScaleEdges:
    """Test cross-scale edges are valid."""

    def test_finer_to_coarser_valid(self):
        """Finer->coarser edge is valid."""
        valid = {
            "dimensions": [
                {
                    "name": "hourly_stress",
                    "description": "hourly",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract stress from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "hourly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [
                {"cause": "hourly_stress", "effect": "daily_mood", "description": "Hourly affects daily"}
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_structure_proposal(None, pred) > 0.0

    def test_coarser_to_finer_valid(self):
        """Coarser->finer edge is valid."""
        valid = {
            "dimensions": [
                {
                    "name": "weekly_stress",
                    "description": "weekly",
                    "role": "exogenous",
                    "observability": "observed",
                    "how_to_measure": "Extract stress from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "weekly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "daily_mood",
                    "description": "daily",
                    "role": "endogenous",
                    "is_outcome": True,
                    "observability": "observed",
                    "how_to_measure": "Extract mood from data",
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
            "edges": [
                {"cause": "weekly_stress", "effect": "daily_mood", "description": "Weekly affects daily"}
            ],
        }
        pred = MockPrediction(structure=json.dumps(valid))
        assert score_structure_proposal(None, pred) > 0.0
