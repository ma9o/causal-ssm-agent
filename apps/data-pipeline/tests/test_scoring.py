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

    def test_construct_points_scale_with_count(self):
        """Each construct contributes exactly 2 points (role + temporal_status)."""
        small = LatentModel(**json.loads(_simple_model_json()))
        small_points = _count_rule_points(small)

        # Add a third construct with no new edges
        data = json.loads(_simple_model_json())
        data["constructs"].append(
            {
                "name": "exercise",
                "description": "Physical activity",
                "role": "exogenous",
                "temporal_status": "time_varying",
            }
        )
        larger = LatentModel(**data)
        # Exactly 2 more points from the new construct, 0 from edges
        assert _count_rule_points(larger) == small_points + 2

    def test_edge_points_scale_with_count(self):
        """Each well-formed edge contributes exactly 3 points (cause + effect + endogenous)."""
        base = LatentModel(**json.loads(_simple_model_json()))
        base_points = _count_rule_points(base)

        data = json.loads(_simple_model_json())
        data["constructs"].append(
            {
                "name": "exercise",
                "description": "Physical activity",
                "role": "exogenous",
                "temporal_status": "time_varying",
            }
        )
        data["edges"].append(
            {"cause": "exercise", "effect": "sleep", "description": "Exercise improves sleep"}
        )
        with_edge = LatentModel(**data)
        # +2 for new construct, +3 for new well-formed edge
        assert _count_rule_points(with_edge) == base_points + 2 + 3


# =============================================================================
# score_latent_model
# =============================================================================


class TestScoreLatentModel:
    def test_valid_model_equals_rule_points(self):
        """Raw score equals the rule-point count for a valid model."""
        pred = SimpleNamespace(structure=_simple_model_json())
        structure = LatentModel(**json.loads(_simple_model_json()))
        score = score_latent_model(None, pred)
        assert score == _count_rule_points(structure)

    def test_no_structure_attr_zero(self):
        pred = SimpleNamespace(other="field")
        score = score_latent_model(None, pred)
        assert score == 0.0


# =============================================================================
# score_latent_model_normalized
# =============================================================================


class TestScoreLatentModelNormalized:
    def test_perfect_model_scores_one(self):
        """A model where every edge's effect is endogenous should achieve 1.0."""
        pred = SimpleNamespace(structure=_simple_model_json())
        score = score_latent_model_normalized(None, pred)
        # stress→sleep edge: sleep is endogenous, so all 3 edge points awarded
        # 2 constructs * 2 + 1 edge * 3 = 7 points out of max 7
        assert score == 1.0

    def test_invalid_returns_zero(self):
        pred = SimpleNamespace(structure="not json")
        score = score_latent_model_normalized(None, pred)
        assert score == 0.0

    def test_equals_raw_over_max(self):
        """Normalized = raw / (n_constructs*2 + n_edges*3)."""
        pred = SimpleNamespace(structure=_simple_model_json())
        raw = score_latent_model(None, pred)
        norm = score_latent_model_normalized(None, pred)
        data = json.loads(_simple_model_json())
        max_points = len(data["constructs"]) * 2 + len(data["edges"]) * 3
        assert norm == raw / max_points
