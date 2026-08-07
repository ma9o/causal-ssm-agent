"""Read helpers and deterministic restrictions for StructuralPlan artifacts."""

from __future__ import annotations

from collections import defaultdict

from nof1_causal_lab.artifacts.structural_plan import (
    StructuralDisposition,
    StructuralPlan,
)
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


def get_state_ids(plan: StructuralPlan) -> list[str]:
    return list(plan.state_order)


def get_state_names(plan: StructuralPlan) -> list[str]:
    return [plan.semantics.constructs[source_id].name for source_id in plan.state_order]


def get_plan_constructs(plan: StructuralPlan) -> list[UncheckedJsonObject]:
    return [
        {"source_id": source_id, **construct.model_dump(mode="json")}
        for source_id, construct in plan.semantics.constructs.items()
    ]


def get_plan_indicators(plan: StructuralPlan) -> list[UncheckedJsonObject]:
    return [
        {"source_id": source_id, **indicator.model_dump(mode="json")}
        for source_id, indicator in plan.semantics.indicators.items()
    ]


def get_manifest_indicators(plan: StructuralPlan) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": source_id,
            **plan.semantics.indicators[source_id].model_dump(mode="json"),
        }
        for source_id in plan.manifest_indicator_order
    ]


def get_reference_indicator_lookup(plan: StructuralPlan) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    return {
        plan.semantics.constructs[construct_id].name: (plan.semantics.indicators[indicator_id].name)
        for construct_id, indicator_id in plan.reference_indicator_ids.items()
    }


def get_reference_indicator_polarities(plan: StructuralPlan) -> dict[str, str]:
    """Return retained construct name to its planned reference polarity."""
    return {
        plan.semantics.constructs[construct_id].name: (
            plan.semantics.indicators[indicator_id].construct_polarity
        )
        for construct_id, indicator_id in plan.reference_indicator_ids.items()
    }


def get_edges(plan: StructuralPlan) -> list[UncheckedJsonObject]:
    result: list[UncheckedJsonObject] = []
    for edge in plan.edges:
        source_edge = plan.semantics.edges[edge.source_id]
        result.append(
            {
                "source_id": edge.source_id,
                "cause_id": edge.cause_id,
                "effect_id": edge.effect_id,
                "cause": plan.semantics.constructs[edge.cause_id].name,
                "effect": plan.semantics.constructs[edge.effect_id].name,
                "description": source_edge.description,
                "lagged": edge.lagged,
                "sources": [source.model_dump(mode="json") for source in source_edge.sources],
            }
        )
    return result


def get_known_inputs(plan: StructuralPlan) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": item.source_id,
            "construct_id": item.construct_id,
            "source_indicator_id": item.source_indicator_id,
            "construct": plan.semantics.constructs[item.construct_id].name,
            "source_indicator": plan.semantics.indicators[item.source_indicator_id].name,
            "scale": item.scale,
            "missing_policy": item.missing_policy,
        }
        for item in plan.known_inputs
    ]


def get_known_input_source_indicators(
    plan: StructuralPlan,
) -> set[str]:
    return {str(item["source_indicator"]) for item in get_known_inputs(plan)}


def get_induced_dependencies(
    plan: StructuralPlan,
) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": dependency.source_id,
            "between_ids": list(dependency.between),
            "between": [
                plan.semantics.constructs[source_id].name for source_id in dependency.between
            ],
            "kind": dependency.kind,
            "source_confounder_ids": list(dependency.source_confounder_ids),
            "source_confounders": [
                plan.semantics.constructs[source_id].name
                for source_id in dependency.source_confounder_ids
            ],
        }
        for dependency in plan.induced_dependencies
    ]


def get_marginalized_scales(
    plan: StructuralPlan,
) -> list[UncheckedJsonObject]:
    """Return identifiable marginalized-confounder scale equivalence classes."""
    dependencies = get_induced_dependencies(plan)
    footprint_by_confounder: dict[str, set[str]] = defaultdict(set)
    source_name_by_id: dict[str, str] = {}
    kind_by_confounder: dict[str, str] = {}
    directions_by_confounder: dict[str, list[tuple[str, str]]] = defaultdict(list)
    dependency_ids_by_confounder: dict[str, list[str]] = defaultdict(list)

    for dependency in dependencies:
        kind = str(dependency["kind"])
        between = tuple(str(name) for name in dependency["between"])
        if len(between) != 2:
            raise ValueError(
                f"Malformed induced dependency {dependency['source_id']!r}: {between!r}"
            )
        for source_id, source_name in zip(
            dependency["source_confounder_ids"],
            dependency["source_confounders"],
            strict=True,
        ):
            source_id = str(source_id)
            source_name_by_id[source_id] = str(source_name)
            footprint_by_confounder[source_id].update(between)
            directions_by_confounder[source_id].append(between)
            dependency_ids_by_confounder[source_id].append(str(dependency["source_id"]))
            prior_kind = kind_by_confounder.setdefault(source_id, kind)
            if prior_kind != kind:
                raise ValueError(
                    f"Confounder source {source_id!r} has inconsistent dependency kinds"
                )

    members_by_footprint: dict[tuple[str, frozenset[str]], list[str]] = defaultdict(list)
    for source_id, footprint in footprint_by_confounder.items():
        members_by_footprint[(kind_by_confounder[source_id], frozenset(footprint))].append(
            source_id
        )

    scales: list[UncheckedJsonObject] = []
    for (kind, footprint), source_ids in sorted(
        members_by_footprint.items(),
        key=lambda item: (
            item[0][0],
            sorted(item[0][1]),
            sorted(item[1]),
        ),
    ):
        source_ids = sorted(source_ids)
        source_names = sorted(source_name_by_id[source_id] for source_id in source_ids)
        directions: set[tuple[str, str]] = set()
        dependency_ids: set[str] = set()
        for source_id in source_ids:
            directions.update(directions_by_confounder[source_id])
            dependency_ids.update(dependency_ids_by_confounder[source_id])
        scales.append(
            {
                "parameter": "tau_" + "__".join(source_names),
                "kind": kind,
                "source_ids": source_ids,
                "sources": source_names,
                "affected_states": sorted(footprint),
                "directions": sorted(directions),
                "dependency_ids": sorted(dependency_ids),
            }
        )
    return scales


def get_model_clock(plan: StructuralPlan) -> str:
    return plan.semantics.model_clock


def restrict_structural_plan(
    plan: StructuralPlan,
    keep_names: set[str],
) -> StructuralPlan:
    """Restrict one plan consistently for cumulative construct admission."""
    construct_id_by_name = {
        construct.name: source_id for source_id, construct in plan.semantics.constructs.items()
    }
    keep_ids = {construct_id_by_name[name] for name in keep_names if name in construct_id_by_name}
    relevant_edges = tuple(
        edge
        for edge in plan.edges
        if edge.effect_id in keep_ids
        and (
            edge.cause_id in keep_ids
            or any(item.construct_id == edge.cause_id for item in plan.known_inputs)
        )
    )
    relevant_input_ids = {
        edge.cause_id
        for edge in relevant_edges
        if any(item.construct_id == edge.cause_id for item in plan.known_inputs)
    }
    known_inputs = tuple(
        item for item in plan.known_inputs if item.construct_id in relevant_input_ids
    )
    manifest_order = tuple(
        indicator_id
        for indicator_id in plan.manifest_indicator_order
        if plan.semantics.indicators[indicator_id].construct_name in keep_names
    )
    dependencies = tuple(
        dependency
        for dependency in plan.induced_dependencies
        if set(dependency.between) <= keep_ids
    )
    retained_edge_ids = {edge.source_id for edge in relevant_edges}
    manifest_ids = set(manifest_order)
    input_indicator_ids = {item.source_indicator_id for item in known_inputs}
    dispositions = []
    for disposition in plan.dispositions:
        update: UncheckedJsonObject = {}
        if disposition.source_kind == "construct":
            if disposition.source_id in keep_ids:
                update["disposition"] = StructuralDisposition.RETAINED_STATE
            elif disposition.source_id in relevant_input_ids:
                update["disposition"] = StructuralDisposition.KNOWN_INPUT
            elif disposition.disposition in {
                StructuralDisposition.RETAINED_STATE,
                StructuralDisposition.KNOWN_INPUT,
            }:
                update["disposition"] = StructuralDisposition.IDENTIFICATION_ONLY
                update["reason"] = "Excluded from this cumulative admission prefix."
        elif disposition.source_kind == "edge":
            update["disposition"] = (
                StructuralDisposition.RETAINED_EDGE
                if disposition.source_id in retained_edge_ids
                else StructuralDisposition.PROJECTED_EDGE
            )
        elif disposition.source_id in manifest_ids:
            update["disposition"] = StructuralDisposition.MANIFEST
        elif disposition.source_id in input_indicator_ids:
            update["disposition"] = StructuralDisposition.KNOWN_INPUT_SOURCE
        else:
            update["disposition"] = StructuralDisposition.EXCLUDED_INDICATOR
        dispositions.append(disposition.model_copy(update=update))
    return StructuralPlan(
        semantics=plan.semantics,
        state_order=tuple(source_id for source_id in plan.state_order if source_id in keep_ids),
        edges=relevant_edges,
        manifest_indicator_order=manifest_order,
        reference_indicator_ids={
            construct_id: indicator_id
            for construct_id, indicator_id in plan.reference_indicator_ids.items()
            if construct_id in keep_ids
        },
        known_inputs=known_inputs,
        induced_dependencies=dependencies,
        dispositions=tuple(dispositions),
    )


__all__ = [
    "get_edges",
    "get_induced_dependencies",
    "get_known_input_source_indicators",
    "get_known_inputs",
    "get_manifest_indicators",
    "get_marginalized_scales",
    "get_model_clock",
    "get_plan_constructs",
    "get_plan_indicators",
    "get_reference_indicator_lookup",
    "get_reference_indicator_polarities",
    "get_state_ids",
    "get_state_names",
    "restrict_structural_plan",
]
