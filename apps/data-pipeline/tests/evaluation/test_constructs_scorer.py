"""Tests for latent structure scoring functions.

Covers: count_rule_points, score_latent_structure, score_latent_structure_normalized.
"""

import json
from types import SimpleNamespace

from evaluation.scorers.constructs import (
    count_rule_points,
    score_latent_structure,
    score_latent_structure_normalized,
)

from nof1_causal_lab.artifacts import LatentStructure


def _simple_model_json():
    """Minimal valid latent structure JSON."""
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
# count_rule_points
# =============================================================================


class TestCountRulePoints:
    def test_more_constructs_more_points(self):
        """Adding constructs should increase score."""
        small = LatentStructure(**json.loads(_simple_model_json()))
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
        large = LatentStructure(**large_data)

        assert count_rule_points(large) > count_rule_points(small)

    def test_time_invariant_gets_points(self):
        """Time-invariant construct should get points."""
        structure = LatentStructure(**json.loads(_model_with_invariant_json()))
        points = count_rule_points(structure)
        assert points > 0

    def test_construct_points_scale_with_count(self):
        """Each construct contributes exactly 2 points (role + temporal_status)."""
        small = LatentStructure(**json.loads(_simple_model_json()))
        small_points = count_rule_points(small)

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
        larger = LatentStructure(**data)
        # Exactly 2 more points from the new construct, 0 from edges
        assert count_rule_points(larger) == small_points + 2

    def test_edge_points_scale_with_count(self):
        """Each well-formed edge contributes exactly 3 points (cause + effect + endogenous)."""
        base = LatentStructure(**json.loads(_simple_model_json()))
        base_points = count_rule_points(base)

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
        with_edge = LatentStructure(**data)
        # +2 for new construct, +3 for new well-formed edge
        assert count_rule_points(with_edge) == base_points + 2 + 3


# =============================================================================
# score_latent_structure
# =============================================================================


class TestScoreLatentStructure:
    def test_valid_model_equals_rule_points(self):
        """Raw score equals the rule-point count for a valid model."""
        pred = SimpleNamespace(structure=_simple_model_json())
        structure = LatentStructure(**json.loads(_simple_model_json()))
        score = score_latent_structure(None, pred)
        assert score == count_rule_points(structure)

    def test_no_structure_attr_zero(self):
        pred = SimpleNamespace(other="field")
        score = score_latent_structure(None, pred)
        assert score == 0.0


# =============================================================================
# score_latent_structure_normalized
# =============================================================================


class TestScoreLatentStructureNormalized:
    def test_perfect_model_scores_one(self):
        """A model where every edge's effect is endogenous should achieve 1.0."""
        pred = SimpleNamespace(structure=_simple_model_json())
        score = score_latent_structure_normalized(None, pred)
        # stress→sleep edge: sleep is endogenous, so all 3 edge points awarded
        # 2 constructs * 2 + 1 edge * 3 = 7 points out of max 7
        assert score == 1.0

    def test_invalid_returns_zero(self):
        pred = SimpleNamespace(structure="not json")
        score = score_latent_structure_normalized(None, pred)
        assert score == 0.0

    def test_equals_raw_over_max(self):
        """Normalized = raw / (n_constructs*2 + n_edges*3)."""
        pred = SimpleNamespace(structure=_simple_model_json())
        raw = score_latent_structure(None, pred)
        norm = score_latent_structure_normalized(None, pred)
        data = json.loads(_simple_model_json())
        max_points = len(data["constructs"]) * 2 + len(data["edges"]) * 3
        assert norm == raw / max_points
