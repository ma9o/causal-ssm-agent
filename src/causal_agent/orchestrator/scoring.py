"""DSPy scoring function for DSEM structure proposals.

Scoring strategy:
- Award points for each INSTANCE of a hard rule being respected
- Complex valid structures score higher (more dimensions/edges = more rule instances)
- Return 0 immediately if ANY validation rule is violated
"""

import json

from pydantic import ValidationError

from causal_agent.orchestrator.schemas import (
    GRANULARITY_HOURS,
    CausalEdge,
    Dimension,
    DSEMStructure,
    Observability,
    Role,
    TemporalStatus,
)
from causal_agent.utils.aggregations import AGGREGATION_REGISTRY


def score_structure_proposal(example, pred, trace=None) -> float:
    """Score a DSEM structure proposal for DSPy optimization.

    Args:
        example: DSPy example (unused, required by DSPy interface)
        pred: DSPy prediction with 'structure' field containing JSON
        trace: Optional trace (unused, required by DSPy interface)

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
        structure = DSEMStructure(**data)
    except (ValidationError, ValueError, TypeError):
        return 0.0

    # Count points for each rule instance respected
    return _count_rule_points(structure)


def _count_rule_points(structure: DSEMStructure) -> float:
    """Count points for each rule instance correctly applied.

    Points per dimension:
    - +1 valid role
    - +1 valid observability
    - +1 valid temporal_status
    - +1 correct causal_granularity constraint (required for time_varying, forbidden for time_invariant)
    - +1 correct aggregation constraint (required for time_varying, forbidden for time_invariant)
    - +1 valid base_dtype
    - +1 valid aggregation name (if specified)
    - +1 valid causal_granularity value (if specified)

    Points per edge:
    - +1 cause exists in dimensions
    - +1 effect exists in dimensions
    - +1 effect is endogenous
    - +1 correct timescale handling (same-scale or cross-scale rules)
    - +1 correct aggregation constraint (finer->coarser requires, coarser->finer forbids)
    - +1 valid aggregation name (if specified)

    Bonus points:
    - +2 per cross-timescale edge (more complex modeling)
    - +1 per latent variable (captures unobserved heterogeneity)
    """
    points = 0.0
    dim_map = {d.name: d for d in structure.dimensions}

    # Points for dimensions
    for dim in structure.dimensions:
        # Valid role (already validated by schema, but count it)
        points += 1

        # Valid observability
        points += 1

        # Valid temporal_status
        points += 1

        is_time_varying = dim.temporal_status == TemporalStatus.TIME_VARYING

        # Correct causal_granularity constraint
        if is_time_varying:
            if dim.causal_granularity is not None:
                points += 1
                # Valid granularity value
                if dim.causal_granularity in GRANULARITY_HOURS:
                    points += 1
        else:  # time_invariant
            if dim.causal_granularity is None:
                points += 1

        # Correct aggregation constraint
        if is_time_varying:
            if dim.aggregation is not None:
                points += 1
                # Valid aggregation name
                if dim.aggregation in AGGREGATION_REGISTRY:
                    points += 1
        else:  # time_invariant
            if dim.aggregation is None:
                points += 1

        # Valid base_dtype
        if dim.base_dtype in ("continuous", "binary", "count", "ordinal", "categorical"):
            points += 1

        # Bonus for latent variables (modeling unobserved heterogeneity)
        if dim.observability == Observability.LATENT:
            points += 1

    # Points for edges
    for edge in structure.edges:
        # Cause exists
        if edge.cause in dim_map:
            points += 1

        # Effect exists
        if edge.effect in dim_map:
            points += 1

        cause_dim = dim_map.get(edge.cause)
        effect_dim = dim_map.get(edge.effect)

        if cause_dim and effect_dim:
            # Effect is endogenous
            if effect_dim.role == Role.ENDOGENOUS:
                points += 1

            cause_gran = cause_dim.causal_granularity
            effect_gran = effect_dim.causal_granularity

            # Timescale handling
            if cause_gran == effect_gran:
                # Same timescale: valid lag handling
                points += 1
            else:
                # Cross-timescale: bonus for complexity
                points += 2

                # Check aggregation rules for cross-timescale
                if cause_gran and effect_gran:
                    cause_hours = GRANULARITY_HOURS.get(cause_gran, 0)
                    effect_hours = GRANULARITY_HOURS.get(effect_gran, 0)

                    if cause_hours < effect_hours:
                        # Finer->coarser: aggregation required
                        if edge.aggregation is not None:
                            points += 1
                    else:
                        # Coarser->finer: aggregation forbidden
                        if edge.aggregation is None:
                            points += 1

        # Valid edge aggregation name (if specified)
        if edge.aggregation is not None and edge.aggregation in AGGREGATION_REGISTRY:
            points += 1

    return points


def score_structure_proposal_normalized(example, pred, trace=None) -> float:
    """Normalized version of score_structure_proposal (0-1 range).

    Useful when comparing across very different structure complexities.
    Divides by theoretical maximum for the given structure size.
    """
    raw_score = score_structure_proposal(example, pred, trace)
    if raw_score == 0:
        return 0.0

    # Parse to get structure size for normalization
    try:
        data = json.loads(pred.structure)
        n_dims = len(data.get("dimensions", []))
        n_edges = len(data.get("edges", []))
    except (json.JSONDecodeError, AttributeError):
        return 0.0

    # Theoretical max per dimension: ~8-9 points (3 classification + 2 granularity + 2 aggregation + 1 dtype + 1 latent bonus)
    # Theoretical max per edge: ~5-6 points
    max_dim_points = n_dims * 9
    max_edge_points = n_edges * 6
    max_points = max_dim_points + max_edge_points

    if max_points == 0:
        return 0.0

    return min(1.0, raw_score / max_points)
