"""Deterministic CausalDesign -> StructuralPlan projection."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast

from nof1_causal_lab.artifacts.latent_structure import Role, TemporalStatus
from nof1_causal_lab.artifacts.structural_plan import (
    StructuralDisposition,
    StructuralEdge,
    StructuralInducedDependency,
    StructuralItemDisposition,
    StructuralKnownInput,
    StructuralPlan,
    StructuralSemanticCatalog,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.utils.causal_design import build_reference_indicator_lookup
from nof1_causal_lab.utils.identifiability import (
    analyze_unobserved_constructs,
    dag_to_admg,
    get_observed_constructs,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.causal_design import CausalDesign


class StructuralPlanningError(AggregatedCompileError):
    """Aggregate independent source-to-plan compilation failures."""

    header = "Structural planning failed"


def _source_id(kind: str, identity: str) -> str:
    """Mint an opaque ID from an item's semantic identity, independent of ordering."""
    digest = sha256(f"{kind}\0{identity}".encode()).hexdigest()[:20]
    return f"{kind}:{digest}"


def build_structural_plan(causal_design: CausalDesign) -> StructuralPlan:
    """Project a scientific causal design into one typed executable plan.

    Every source construct, edge, and indicator receives an explicit disposition.
    Unsupported retained semantics fail here; no downstream translator may silently
    erase a retained structural item.
    """
    design_data = causal_design.model_dump(mode="json")
    latent = design_data["latent"]
    measurement = design_data["measurement"]
    identifiability = design_data.get("identifiability") or {}
    known_input_declarations = design_data.get("known_inputs") or []
    scientific_only_declarations = design_data.get("scientific_only_constructs") or []

    constructs = list(causal_design.latent.constructs)
    edges = list(causal_design.latent.edges)
    indicators = list(causal_design.measurement.indicators)

    duplicate_construct_names = sorted(
        name for name, count in Counter(item.name for item in constructs).items() if count > 1
    )
    duplicate_indicator_names = sorted(
        name for name, count in Counter(item.name for item in indicators).items() if count > 1
    )
    if duplicate_construct_names or duplicate_indicator_names:
        raise StructuralPlanningError(
            [
                *(
                    [
                        "CausalDesign contains duplicate construct names: "
                        f"{duplicate_construct_names}."
                    ]
                    if duplicate_construct_names
                    else []
                ),
                *(
                    [
                        "CausalDesign contains duplicate indicator names: "
                        f"{duplicate_indicator_names}."
                    ]
                    if duplicate_indicator_names
                    else []
                ),
            ]
        )
    duplicate_edge_endpoints = sorted(
        endpoints
        for endpoints, count in Counter((item.cause, item.effect) for item in edges).items()
        if count > 1
    )
    if duplicate_edge_endpoints:
        raise StructuralPlanningError(
            [
                "CausalDesign contains multiple structural edges for the same cause/effect "
                f"pair: {duplicate_edge_endpoints}. One executable edge cannot carry multiple "
                "lag classes."
            ]
        )

    construct_id_by_name = {
        construct.name: _source_id("construct", construct.name) for construct in constructs
    }
    edge_id_by_index = {
        index: _source_id(
            "edge",
            f"{edge.cause}\0{edge.effect}\0{'lagged' if edge.lagged else 'contemporaneous'}",
        )
        for index, edge in enumerate(edges)
    }
    indicator_id_by_name = {
        indicator.name: _source_id("indicator", indicator.name) for indicator in indicators
    }
    semantics = StructuralSemanticCatalog(
        constructs={construct_id_by_name[construct.name]: construct for construct in constructs},
        edges={edge_id_by_index[index]: edge for index, edge in enumerate(edges)},
        indicators={indicator_id_by_name[indicator.name]: indicator for indicator in indicators},
        model_clock=causal_design.measurement.model_clock,
    )
    reference_indicator_names = build_reference_indicator_lookup(
        [indicator.model_dump(mode="json") for indicator in indicators]
    )

    parents_by_construct: dict[str, set[str]] = defaultdict(set)
    children_by_construct: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        parents_by_construct[edge.effect].add(edge.cause)
        children_by_construct[edge.cause].add(edge.effect)

    observed_constructs = get_observed_constructs(measurement)
    known_input_names = {item["construct"] for item in known_input_declarations}
    scientific_only_names = {item["construct"] for item in scientific_only_declarations}
    analysis = analyze_unobserved_constructs(latent, measurement, identifiability)
    can_marginalize = set(analysis.get("can_marginalize", set()))
    _, confounders = dag_to_admg(latent, observed_constructs)

    marginalizable_roots: set[str] = set()
    construct_lookup = {construct.name: construct for construct in constructs}
    for name in sorted(can_marginalize):
        construct = construct_lookup.get(name)
        if construct is None:
            continue
        if construct.role != Role.EXOGENOUS:
            continue
        if parents_by_construct.get(name):
            continue
        marginalizable_roots.add(name)

    retained_names = set(observed_constructs) - known_input_names - scientific_only_names
    state_names = [
        *[
            construct.name
            for construct in constructs
            if construct.name in retained_names
            and construct.temporal_status != TemporalStatus.TIME_INVARIANT
        ],
        *[
            construct.name
            for construct in constructs
            if construct.name in retained_names
            and construct.temporal_status == TemporalStatus.TIME_INVARIANT
        ],
    ]
    state_order = tuple(construct_id_by_name[name] for name in state_names)

    errors: list[str] = []
    retained_edges: list[StructuralEdge] = []
    retained_edge_ids: set[str] = set()
    permitted_causes = retained_names | known_input_names
    for index, edge in enumerate(edges):
        if edge.effect not in retained_names or edge.cause not in permitted_causes:
            continue
        source_id = edge_id_by_index[index]
        effect_construct = construct_lookup[edge.effect]
        if effect_construct.temporal_status == TemporalStatus.TIME_INVARIANT:
            errors.append(
                f"Unsupported retained static-target edge {source_id} "
                f"({edge.cause!r} -> {edge.effect!r}). The executable SSM has no "
                "baseline structural-equation semantics. Convert observed baseline "
                "quantities to known inputs, reduce the static chain before compilation, "
                "or retain the relation only in the scientific DAG."
            )
            continue
        retained_edge_ids.add(source_id)
        retained_edges.append(
            StructuralEdge(
                source_id=source_id,
                cause_id=construct_id_by_name[edge.cause],
                effect_id=construct_id_by_name[edge.effect],
                lagged=edge.lagged,
            )
        )

    known_inputs: list[StructuralKnownInput] = []
    for declaration in known_input_declarations:
        construct_name = declaration["construct"]
        source_indicator = declaration["source_indicator"]
        construct_id = construct_id_by_name.get(construct_name)
        indicator_id = indicator_id_by_name.get(source_indicator)
        if construct_id is None or indicator_id is None:
            errors.append(
                f"Known input {construct_name!r} references unknown construct or indicator "
                f"{source_indicator!r}."
            )
            continue
        known_inputs.append(
            StructuralKnownInput(
                source_id=_source_id(
                    "known_input",
                    f"{construct_name}\0{source_indicator}",
                ),
                construct_id=construct_id,
                source_indicator_id=indicator_id,
                scale=declaration.get("scale", 1.0),
                missing_policy=declaration.get("missing_policy", "zero"),
            )
        )
    used_known_input_names = {
        edge.cause
        for edge in edges
        if edge.cause in known_input_names and edge.effect in retained_names
    }
    unused_known_inputs = sorted(known_input_names - used_known_input_names)
    if unused_known_inputs:
        errors.append(
            "Known inputs have no outgoing edge into a retained state: "
            f"{unused_known_inputs}. Mark scientific-context-only constructs explicitly "
            "instead of compiling unused transition inputs."
        )

    known_input_source_ids = {item.source_indicator_id for item in known_inputs}
    manifest_indicator_ids = tuple(
        indicator_id_by_name[indicator.name]
        for indicator in indicators
        if indicator.construct_name in retained_names
        and indicator_id_by_name[indicator.name] not in known_input_source_ids
    )

    manifest_construct_counts = Counter(
        semantics.indicators[indicator_id].construct_name for indicator_id in manifest_indicator_ids
    )
    uncovered = sorted(name for name in state_names if manifest_construct_counts[name] == 0)
    if uncovered:
        errors.append(
            "Retained structural-plan states have no manifest indicators: "
            f"{uncovered}. Add proxy indicators or exclude them from the executable plan."
        )
    if len(manifest_indicator_ids) < len(state_order):
        errors.append(
            "Loading matrix is rank-deficient at structural-plan construction: "
            f"n_manifest ({len(manifest_indicator_ids)}) < n_latent ({len(state_order)})."
        )

    dependency_sources: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for confounder in sorted(marginalizable_roots & confounders):
        construct = construct_lookup.get(confounder)
        if construct is None:
            continue
        retained_children = sorted(
            child
            for child in children_by_construct.get(confounder, set())
            if child in retained_names
        )
        if len(retained_children) < 2:
            continue
        dependency_kind: Literal["innovation_correlation", "initial_state_correlation"] = (
            "initial_state_correlation"
            if construct.temporal_status == TemporalStatus.TIME_INVARIANT
            else "innovation_correlation"
        )
        for state_1, state_2 in combinations(retained_children, 2):
            dependency_sources[(state_1, state_2, dependency_kind)].append(confounder)

    induced_dependencies = tuple(
        StructuralInducedDependency(
            source_id=_source_id(
                "dependency",
                "\0".join(
                    (
                        dependency_kind,
                        state_1,
                        state_2,
                        *sorted(source_confounders),
                    )
                ),
            ),
            between=(construct_id_by_name[state_1], construct_id_by_name[state_2]),
            kind=cast(
                'Literal["innovation_correlation", "initial_state_correlation"]',
                dependency_kind,
            ),
            source_confounder_ids=tuple(
                construct_id_by_name[name] for name in sorted(source_confounders)
            ),
        )
        for (
            (state_1, state_2, dependency_kind),
            source_confounders,
        ) in sorted(dependency_sources.items())
    )

    dispositions: list[StructuralItemDisposition] = []
    for construct in constructs:
        source_id = construct_id_by_name[construct.name]
        if construct.name in retained_names:
            disposition = StructuralDisposition.RETAINED_STATE
            reason = "Measured construct retained as an executable latent state."
        elif construct.name in known_input_names:
            disposition = StructuralDisposition.KNOWN_INPUT
            reason = "Observed construct compiled as a deterministic transition input."
        elif construct.name in marginalizable_roots:
            disposition = StructuralDisposition.MARGINALIZED
            reason = "Safe unobserved exogenous root projected from the executable state vector."
        elif construct.name in scientific_only_names:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = (
                "Author explicitly retained this measured construct for scientific context only."
            )
        else:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = "Scientific-DAG construct not retained in the executable state."
        dispositions.append(
            StructuralItemDisposition(
                source_id=source_id,
                source_kind="construct",
                disposition=disposition,
                reason=reason,
            )
        )

    for index, _edge in enumerate(edges):
        source_id = edge_id_by_index[index]
        dispositions.append(
            StructuralItemDisposition(
                source_id=source_id,
                source_kind="edge",
                disposition=(
                    StructuralDisposition.RETAINED_EDGE
                    if source_id in retained_edge_ids
                    else StructuralDisposition.PROJECTED_EDGE
                ),
                reason=(
                    "Both endpoints survive the executable projection."
                    if source_id in retained_edge_ids
                    else "At least one endpoint is not an executable retained state or known input."
                ),
            )
        )

    for indicator in indicators:
        source_id = indicator_id_by_name[indicator.name]
        if source_id in manifest_indicator_ids:
            disposition = StructuralDisposition.MANIFEST
            reason = "Indicator retained as a manifest likelihood channel."
        elif source_id in known_input_source_ids:
            disposition = StructuralDisposition.KNOWN_INPUT_SOURCE
            reason = "Indicator supplies a deterministic known-input trajectory."
        else:
            disposition = StructuralDisposition.EXCLUDED_INDICATOR
            reason = "Indicator belongs to a construct outside the executable state vector."
        dispositions.append(
            StructuralItemDisposition(
                source_id=source_id,
                source_kind="indicator",
                disposition=disposition,
                reason=reason,
            )
        )

    if errors:
        raise StructuralPlanningError(errors)

    return StructuralPlan(
        semantics=semantics,
        state_order=state_order,
        edges=tuple(retained_edges),
        manifest_indicator_order=manifest_indicator_ids,
        reference_indicator_ids={
            construct_id_by_name[construct_name]: indicator_id_by_name[indicator_name]
            for construct_name, indicator_name in reference_indicator_names.items()
            if construct_id_by_name[construct_name] in set(state_order)
        },
        known_inputs=tuple(known_inputs),
        induced_dependencies=induced_dependencies,
        dispositions=tuple(dispositions),
    )
