"""Tests for latent model scoring functions.

Covers: _count_rule_points, score_latent_model, score_latent_model_normalized.
"""

import json
from types import SimpleNamespace

from causal_ssm_agent.orchestrator.schemas import LatentModel
from causal_ssm_agent.orchestrator.scoring import (
    _count_rule_points,
    score_latent_model,
    score_latent_model_normalized,
)


def _simple_model_json():
    """Minimal valid latent model JSON."""
    return json.dumps(
        {
            "constructs": [
                {
                    "name": "stress",
                    "description": "Perceived stress level",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "sleep", "description": "Stress disrupts sleep"},
            ],
        }
    )


def _model_with_invariant_json():
    """Model with a time-invariant construct."""
    return json.dumps(
        {
            "constructs": [
                {
                    "name": "trait",
                    "description": "Stable personality trait",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
                {
                    "name": "mood",
                    "description": "Daily mood state",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            "edges": [
                {"cause": "trait", "effect": "mood", "description": "Trait affects mood"},
            ],
        }
    )


# =============================================================================
# _count_rule_points
# =============================================================================


class TestCountRulePoints:
    def test_more_constructs_more_points(self):
        """Adding constructs should increase score."""
        small = LatentModel(**json.loads(_simple_model_json()))
        large_data = json.loads(_simple_model_json())
        large_data["constructs"].append(
            {
                "name": "exercise",
                "description": "Physical activity",
                "role": "exogenous",
                "temporal_status": "time_varying",
            }
        )
        large_data["edges"].append(
            {
                "cause": "exercise",
                "effect": "sleep",
                "description": "Exercise improves sleep",
            }
        )
        large = LatentModel(**large_data)

        assert _count_rule_points(large) > _count_rule_points(small)

    def test_time_invariant_gets_points(self):
        """Time-invariant construct should get points."""
        structure = LatentModel(**json.loads(_model_with_invariant_json()))
        points = _count_rule_points(structure)
        assert points > 0

    def test_exact_simple_model_points(self):
        """Verify exact point count for a simple 2-construct, 1-edge model."""
        structure = LatentModel(**json.loads(_simple_model_json()))
        points = _count_rule_points(structure)
        # stress: role(1) + temporal_status(1) = 2
        # sleep:  role(1) + temporal_status(1) = 2
        # edge:   cause_exists(1) + effect_exists(1) + endogenous(1) = 3
        assert points == 7.0

    def test_time_invariant_construct_points(self):
        """Time-invariant construct gets 2 points (role + temporal_status)."""
        structure = LatentModel(**json.loads(_model_with_invariant_json()))
        points = _count_rule_points(structure)
        # trait (time_invariant): role(1) + temporal_status(1) = 2
        # mood (time_varying): role(1) + temporal_status(1) = 2
        # edge: cause(1) + effect(1) + endogenous(1) = 3
        assert points == 7.0


# =============================================================================
# score_latent_model
# =============================================================================


class TestScoreLatentModel:
    def test_valid_model_positive_score(self):
        pred = SimpleNamespace(structure=_simple_model_json())
        score = score_latent_model(None, pred)
        assert score > 0

    def test_no_structure_attr_zero(self):
        pred = SimpleNamespace(other="field")
        score = score_latent_model(None, pred)
        assert score == 0.0


# =============================================================================
# score_latent_model_normalized
# =============================================================================


class TestScoreLatentModelNormalized:
    def test_in_zero_one_range(self):
        pred = SimpleNamespace(structure=_simple_model_json())
        score = score_latent_model_normalized(None, pred)
        assert 0.0 < score <= 1.0

    def test_invalid_returns_zero(self):
        pred = SimpleNamespace(structure="not json")
        score = score_latent_model_normalized(None, pred)
        assert score == 0.0

    def test_consistent_with_raw(self):
        """Normalized should be positive exactly when raw is positive."""
        pred = SimpleNamespace(structure=_simple_model_json())
        raw = score_latent_model(None, pred)
        norm = score_latent_model_normalized(None, pred)
        assert (raw > 0) == (norm > 0)
