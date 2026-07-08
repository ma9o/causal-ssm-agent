"""Latent-model (Target 1a constructs) scoring.

Lifted out of the deleted ``orchestrator.scoring`` so the rule-based latent-structure
score is a single source shared by the Inspect Target 1a eval and any registry
constructs row — not duplicated. ``count_rule_points`` is the core logic; the
``score_latent_structure*`` functions are the DSPy-metric interface; and
``LatentStructureScorer`` is the registry-shaped wrapper.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from nof1_causal_lab.artifacts.latent_structure import LatentStructure, Role

_logger = logging.getLogger(__name__)


def count_rule_points(structure: LatentStructure) -> float:
    """Count points for each rule instance correctly applied.

    Points per construct: +1 valid role, +1 valid temporal_status.
    Points per edge: +1 cause exists, +1 effect exists, +1 effect is endogenous.
    """
    points = 0.0
    construct_map = {c.name: c for c in structure.constructs}

    for _construct in structure.constructs:
        points += 1  # valid role (schema-validated)
        points += 1  # valid temporal_status

    for edge in structure.edges:
        if edge.cause in construct_map:
            points += 1
        if edge.effect in construct_map:
            points += 1
        effect_construct = construct_map.get(edge.effect)
        if effect_construct and effect_construct.role == Role.ENDOGENOUS:
            points += 1

    return points


def score_latent_structure(_example, pred, _trace=None) -> float:
    """Score a latent structure proposal (DSPy-metric compatible; usable standalone).

    Returns 0 if any rule is violated, otherwise the sum of rule-instance points.
    """
    try:
        structure_json = pred.structure
    except AttributeError:
        _logger.info("Prediction missing 'structure' field")
        return 0.0

    try:
        data = json.loads(structure_json)
    except json.JSONDecodeError as e:
        _logger.info("Invalid JSON in prediction structure: %s", e)
        return 0.0

    try:
        structure = LatentStructure(**data)
    except (ValidationError, ValueError, TypeError) as e:
        _logger.info("Prediction failed schema validation: %s", e)
        return 0.0

    return count_rule_points(structure)


def score_latent_structure_normalized(example, pred, trace=None) -> float:
    """Normalized (0-1) version of :func:`score_latent_structure`."""
    raw_score = score_latent_structure(example, pred, trace)
    if raw_score == 0:
        return 0.0

    try:
        data = json.loads(pred.structure)
        n_constructs = len(data.get("constructs", []))
        n_edges = len(data.get("edges", []))
    except (json.JSONDecodeError, AttributeError):
        return 0.0

    max_points = n_constructs * 2 + n_edges * 3
    if max_points == 0:
        return 0.0

    return min(1.0, raw_score / max_points)
