"""Deterministic estimation-time projection from the user-facing causal DAG."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from causal_ssm_agent.artifacts.latent_model import Role, TemporalStatus
from causal_ssm_agent.utils.identifiability import (
    analyze_unobserved_constructs,
    get_observed_constructs,
)


def build_estimation_projection(
    latent_model: dict,
    measurement_model: dict,
    identifiability_result: dict | None,
    known_inputs: list[dict] | None = None,
) -> dict[str, Any]:
    """Project the user-facing DAG into the retained estimation-time state graph.

    The external contract remains a latent-variable DAG with explicit unobserved
    constructs. This helper derives the executable state-space view used by the
    compiler and downstream model-facing surfaces.

    Current compiler/runtime rule:
    - Retain measured constructs unless they are declared known inputs.
    - Compile declared known inputs as observed transition drivers, not latent states.
    - Marginalize only unobserved root exogenous constructs that are safe to
      marginalize per the deterministic identifiability analysis.
    - Convert marginalized shared confounders into pairwise induced
      dependencies among retained states.
    - Leave other unmeasured constructs in the user-facing latent DAG only;
      they do not remain in the executable estimation state vector.
    """

    identifiability = identifiability_result or {}
    construct_lookup = {
        construct["name"]: construct
        for construct in latent_model.get("constructs", [])
        if isinstance(construct, dict) and isinstance(construct.get("name"), str)
    }

    parents_by_construct: dict[str, set[str]] = defaultdict(set)
    children_by_construct: dict[str, set[str]] = defaultdict(set)
    for edge in latent_model.get("edges", []):
        cause = edge.get("cause")
        effect = edge.get("effect")
        if isinstance(cause, str) and isinstance(effect, str):
            parents_by_construct[effect].add(cause)
            children_by_construct[cause].add(effect)

    observed_constructs = get_observed_constructs(measurement_model)
    known_input_payloads = [dict(item) for item in (known_inputs or [])]
    known_input_names = {
        item["construct"]
        for item in known_input_payloads
        if isinstance(item.get("construct"), str)
    }

    analysis = analyze_unobserved_constructs(
        latent_model,
        measurement_model,
        identifiability,
    )
    can_marginalize = set(analysis.get("can_marginalize", set()))
    confounders = set(
        (identifiability.get("graph_info") or {}).get("unobserved_confounders", []) or []
    )

    marginalizable_roots: set[str] = set()
    for name in sorted(can_marginalize):
        construct = construct_lookup.get(name)
        if construct is None:
            continue
        if construct.get("role") != Role.EXOGENOUS.value:
            continue
        if parents_by_construct.get(name):
            continue
        marginalizable_roots.add(name)

    retained_names = set(observed_constructs) - known_input_names
    state_order = _build_state_order(latent_model, retained_names)

    retained_edges = [
        edge
        for edge in latent_model.get("edges", [])
        if edge.get("effect") in retained_names
        and edge.get("cause") in (retained_names | known_input_names)
    ]

    dependency_sources: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for confounder in sorted(marginalizable_roots & confounders):
        construct = construct_lookup.get(confounder)
        if construct is None:
            continue

        retained_children = sorted(
            {
                effect
                for effect in children_by_construct.get(confounder, set())
                if effect in retained_names
            }
        )
        if len(retained_children) < 2:
            continue

        dependency_kind = (
            "initial_state_correlation"
            if construct.get("temporal_status") == TemporalStatus.TIME_INVARIANT.value
            else "innovation_correlation"
        )
        for state_1, state_2 in combinations(retained_children, 2):
            dependency_sources[(state_1, state_2, dependency_kind)].append(confounder)

    induced_dependencies = [
        {
            "between": [state_1, state_2],
            "kind": dependency_kind,
            "source_confounders": sorted(source_confounders),
        }
        for (state_1, state_2, dependency_kind), source_confounders in sorted(
            dependency_sources.items()
        )
    ]

    return {
        "state_order": state_order,
        "edges": retained_edges,
        "induced_dependencies": induced_dependencies,
        "known_inputs": known_input_payloads,
    }


def _build_state_order(
    latent_model: dict,
    retained_names: set[str],
) -> list[str]:
    """Canonical state ordering: retained time-varying, then retained time-invariant."""
    time_varying: list[str] = []
    time_invariant: list[str] = []
    for construct in latent_model.get("constructs", []):
        name = construct.get("name")
        if not isinstance(name, str) or name not in retained_names:
            continue
        if construct.get("temporal_status") == TemporalStatus.TIME_INVARIANT.value:
            time_invariant.append(name)
        else:
            time_varying.append(name)
    return [*time_varying, *time_invariant]
