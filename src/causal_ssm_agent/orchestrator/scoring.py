"""Scoring function for latent model proposals.

Scoring strategy:
- Award points for each INSTANCE of a hard rule being respected
- Complex valid structures score higher (more constructs/edges = more rule instances)
- Return 0 immediately if ANY validation rule is violated

Can be used for manual evaluation or with DSPy optimization.
"""

import json

from pydantic import ValidationError

from causal_ssm_agent.orchestrator.schemas import (
    GRANULARITY_HOURS,
    LatentModel,
    TemporalStatus,
)


def score_latent_model(_example, pred, _trace=None) -> float:
    """Score a latent model proposal.

    Compatible with DSPy metric interface but can be used standalone.

    Args:
        _example: Context/reference (unused, for DSPy compatibility)
        pred: Object with 'structure' field containing JSON string
        _trace: Optional trace (unused, for DSPy compatibility)

    Returns:
        Float score: 0 if any rule violated, otherwise sum of rule instance points
    """
    try:
        structure_json = pred.structure
    except AttributeError:
        return 0.0

    # Parse JSON
    try:
        data = json.loads(structure_json)
    except json.JSONDecodeError:
        return 0.0

    # Validate against schema (returns 0 if any hard rule violated)
    try:
        structure = LatentModel(**data)
    except (ValidationError, ValueError, TypeError):
        return 0.0

    # Count points for each rule instance respected
    return _count_rule_points(structure)


def _count_rule_points(structure: LatentModel) -> float:
    """Count points for each rule instance correctly applied.

    Points per construct:
    - +1 valid role
    - +1 valid temporal_status
    - +1 correct causal_granularity constraint (required for time_varying, forbidden for time_invariant)
    - +1 valid causal_granularity value (if specified)

    Points per edge:
    - +1 cause exists in constructs
    - +1 effect exists in constructs
    - +1 effect is endogenous
    - +1 correct timescale handling (same-scale or cross-scale rules)

    Bonus points:
    - +2 per cross-timescale edge (more complex modeling)
    """
    from causal_ssm_agent.orchestrator.schemas import Role

    points = 0.0
    construct_map = {c.name: c for c in structure.constructs}

    # Points for constructs
    for construct in structure.constructs:
        # Valid role (already validated by schema, but count it)
        points += 1

        # Valid temporal_status
        points += 1

        is_time_varying = construct.temporal_status == TemporalStatus.TIME_VARYING

        # Correct causal_granularity constraint
        if is_time_varying:
            if construct.causal_granularity is not None:
                points += 1
                # Valid granularity value
                if construct.causal_granularity in GRANULARITY_HOURS:
                    points += 1
        else:  # time_invariant
            if construct.causal_granularity is None:
                points += 1

    # Points for edges
    for edge in structure.edges:
        # Cause exists
        if edge.cause in construct_map:
            points += 1

        # Effect exists
        if edge.effect in construct_map:
            points += 1

        cause_construct = construct_map.get(edge.cause)
        effect_construct = construct_map.get(edge.effect)

        if cause_construct and effect_construct:
            # Effect is endogenous
            if effect_construct.role == Role.ENDOGENOUS:
                points += 1

            cause_gran = cause_construct.causal_granularity
            effect_gran = effect_construct.causal_granularity

            # Timescale handling
            if cause_gran == effect_gran:
                # Same timescale: valid lag handling
                points += 1
            else:
                # Cross-timescale: bonus for complexity
                points += 2

    return points


def score_latent_model_normalized(example, pred, trace=None) -> float:
    """Normalized version of score_latent_model (0-1 range).

    Useful when comparing across very different structure complexities.
    Divides by theoretical maximum for the given structure size.
    """
    raw_score = score_latent_model(example, pred, trace)
    if raw_score == 0:
        return 0.0

    # Parse to get structure size for normalization
    try:
        data = json.loads(pred.structure)
        n_constructs = len(data.get("constructs", []))
        n_edges = len(data.get("edges", []))
    except (json.JSONDecodeError, AttributeError):
        return 0.0

    # Theoretical max per construct: ~4 points (2 classification + 2 causal_granularity)
    # Theoretical max per edge: ~4 points (cause + effect + endogenous + timescale/cross-timescale bonus)
    max_construct_points = n_constructs * 4
    max_edge_points = n_edges * 4
    max_points = max_construct_points + max_edge_points

    if max_points == 0:
        return 0.0

    return min(1.0, raw_score / max_points)
