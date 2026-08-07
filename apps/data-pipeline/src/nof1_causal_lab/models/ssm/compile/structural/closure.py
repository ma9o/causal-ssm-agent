"""Total StructuralPlan -> SSMSpec closure and identification certificates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.compile.contracts import (
    AnchorCertificate,
    CompiledStructuralBinding,
)
from nof1_causal_lab.models.ssm.dynamics.spec import HillEdgeSpec, LinearEdgeSpec, NodePotentialSpec
from nof1_causal_lab.models.ssm.structure.parameters import Fixed
from nof1_causal_lab.utils.structural_plan import (
    get_edges,
    get_induced_dependencies,
    get_known_inputs,
    get_manifest_indicators,
    get_marginalized_scales,
    get_state_ids,
    get_state_names,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm import SSMSpec


class StructuralClosureError(AggregatedCompileError):
    """Aggregate failures in plan-to-runtime forward or backward closure."""

    header = "Structural closure failed"


def compile_anchor_certificates(
    spec: SSMSpec,
    structural_plan: StructuralPlan,
) -> list[AnchorCertificate]:
    """Prove location and scale identification for every retained state."""
    latent_names = list(spec.latent_names or [])
    manifest_names = list(spec.manifest_names or [])
    state_ids = get_state_ids(structural_plan)
    indicators = get_manifest_indicators(structural_plan)
    indicator_by_name = {str(item["name"]): item for item in indicators}
    channels_by_construct: dict[str, list[int]] = {}
    for manifest_index, manifest_name in enumerate(manifest_names):
        indicator = indicator_by_name.get(manifest_name)
        if indicator is not None:
            channels_by_construct.setdefault(str(indicator["construct_name"]), []).append(
                manifest_index
            )

    standardized = list(spec.manifest_standardized or [False] * len(manifest_names))
    categorical_anchors = list(spec.manifest_cat_anchor or [False] * len(manifest_names))
    loading_template = np.asarray(spec.lambda_block.template, dtype=float)
    loading_support = np.asarray(spec.lambda_block.free_support, dtype=bool)
    time_invariant = np.asarray(
        spec.diffusion_block.time_invariant_mask
        if spec.diffusion_block.time_invariant_mask is not None
        else np.zeros(len(latent_names), dtype=bool),
        dtype=bool,
    )
    t0_mean_support = np.asarray(spec.t0_means_block.free_support, dtype=bool)
    fixed_dynamics_centers = {
        int(component.target)
        for component in spec.dynamics_spec.components
        if isinstance(component, NodePotentialSpec) and isinstance(component.center, Fixed)
    }

    errors: list[str] = []
    certificates: list[AnchorCertificate] = []
    for latent_index, (construct_id, construct_name) in enumerate(
        zip(state_ids, latent_names, strict=True)
    ):
        channels = channels_by_construct.get(construct_name, [])
        if not channels:
            errors.append(
                f"Construct {construct_name!r} retains no manifest channel; its latent "
                "location and scale are unidentified."
            )
            continue

        standardized_channels = [index for index in channels if standardized[index]]
        if standardized_channels:
            location_index = standardized_channels[0]
            location_anchor = "standardized_manifest"
            location_source_id = str(indicator_by_name[manifest_names[location_index]]["source_id"])
        elif time_invariant[latent_index] and not t0_mean_support[latent_index]:
            location_anchor = "fixed_initial_mean"
            location_source_id = None
        elif not time_invariant[latent_index] and latent_index in fixed_dynamics_centers:
            location_anchor = "fixed_dynamics_center"
            location_source_id = None
        else:
            location_parameter = "t0 mean" if time_invariant[latent_index] else "equilibrium center"
            errors.append(
                f"Construct {construct_name!r} has no location anchor: its free "
                f"{location_parameter} rides an exact additive ridge with channel-side "
                "location parameters."
            )
            continue

        fixed_loading_channels = [
            index
            for index in channels
            if spec.manifest_dists[index] != DistributionFamily.CATEGORICAL
            and loading_template[index, latent_index] != 0.0
            and not loading_support[index, latent_index]
        ]
        categorical_anchor_channels = [index for index in channels if categorical_anchors[index]]
        if fixed_loading_channels:
            scale_index = fixed_loading_channels[0]
            scale_anchor = "fixed_manifest_loading"
        elif categorical_anchor_channels:
            scale_index = categorical_anchor_channels[0]
            scale_anchor = "categorical_slope_pin"
        else:
            errors.append(
                f"Construct {construct_name!r} has no scale anchor: no non-categorical "
                "channel carries a fixed loading and no categorical anchor slope is pinned."
            )
            continue

        free_categorical_channels = [
            manifest_names[index]
            for index in channels
            if spec.manifest_dists[index] == DistributionFamily.CATEGORICAL
            and loading_support[index, latent_index]
        ]
        if free_categorical_channels:
            errors.append(
                "Categorical channels have free loadings that are exactly redundant "
                f"with their class slopes: {sorted(free_categorical_channels)}."
            )
            continue

        certificates.append(
            AnchorCertificate(
                construct_id=construct_id,
                construct_name=construct_name,
                location_anchor=location_anchor,
                location_source_id=location_source_id,
                scale_anchor=scale_anchor,
                scale_source_id=str(indicator_by_name[manifest_names[scale_index]]["source_id"]),
            )
        )

    if errors:
        raise StructuralClosureError(errors)
    return certificates


def compile_structural_bindings(
    spec: SSMSpec,
    structural_plan: StructuralPlan,
) -> list[CompiledStructuralBinding]:
    """Bind every executable plan item and reject unplanned runtime structure."""
    state_ids = get_state_ids(structural_plan)
    state_names = get_state_names(structural_plan)
    manifests = get_manifest_indicators(structural_plan)
    known_inputs = get_known_inputs(structural_plan)
    edges = get_edges(structural_plan)
    dependencies = get_induced_dependencies(structural_plan)
    errors: list[str] = []

    if list(spec.latent_names or []) != state_names:
        errors.append(
            "Compiled latent order does not exactly match StructuralPlan state_order: "
            f"{list(spec.latent_names or [])!r} != {state_names!r}."
        )
    manifest_names = [str(item["name"]) for item in manifests]
    if list(spec.manifest_names or []) != manifest_names:
        errors.append(
            "Compiled manifest order does not exactly match StructuralPlan: "
            f"{list(spec.manifest_names or [])!r} != {manifest_names!r}."
        )
    input_names = [str(item["construct"]) for item in known_inputs]
    if list(spec.input_names or []) != input_names:
        errors.append(
            "Compiled input order does not exactly match StructuralPlan: "
            f"{list(spec.input_names or [])!r} != {input_names!r}."
        )

    bindings = [
        CompiledStructuralBinding(
            source_id=source_id,
            source_kind="state",
            target_kind="latent_state",
            target_indices=(index,),
            target_name=state_names[index],
        )
        for index, source_id in enumerate(state_ids)
    ]
    bindings.extend(
        CompiledStructuralBinding(
            source_id=str(indicator["source_id"]),
            source_kind="manifest",
            target_kind="manifest_channel",
            target_indices=(index,),
            target_name=str(indicator["name"]),
        )
        for index, indicator in enumerate(manifests)
    )
    bindings.extend(
        CompiledStructuralBinding(
            source_id=str(known_input["source_id"]),
            source_kind="known_input",
            target_kind="transition_input",
            target_indices=(index,),
            target_name=str(known_input["construct"]),
        )
        for index, known_input in enumerate(known_inputs)
    )

    latent_idx = {name: index for index, name in enumerate(state_names)}
    input_idx = {name: index for index, name in enumerate(input_names)}
    expected_dynamics_edges: dict[tuple[int, int], UncheckedJsonObject] = {}
    expected_input_edges: dict[tuple[int, int], UncheckedJsonObject] = {}
    for edge in edges:
        effect_index = latent_idx[str(edge["effect"])]
        cause_name = str(edge["cause"])
        if cause_name in input_idx:
            expected_input_edges[(effect_index, input_idx[cause_name])] = edge
        else:
            expected_dynamics_edges[(effect_index, latent_idx[cause_name])] = edge

    emitted_dynamics_edges: list[tuple[int, int]] = [
        (int(component.target), int(component.source))
        for component in spec.dynamics_spec.components
        if isinstance(component, (LinearEdgeSpec, HillEdgeSpec))
    ]
    if len(emitted_dynamics_edges) != len(set(emitted_dynamics_edges)):
        errors.append("Compiled dynamics contains duplicate components for one structural edge.")
    if set(emitted_dynamics_edges) != set(expected_dynamics_edges):
        errors.append(
            "Structural edge closure mismatch for dynamics components: "
            f"planned={sorted(expected_dynamics_edges)}, "
            f"compiled={sorted(set(emitted_dynamics_edges))}."
        )
    for target in sorted(set(emitted_dynamics_edges) & set(expected_dynamics_edges)):
        edge = expected_dynamics_edges[target]
        bindings.append(
            CompiledStructuralBinding(
                source_id=str(edge["source_id"]),
                source_kind="edge",
                target_kind="dynamics_edge",
                target_indices=target,
                target_name=f"{edge['cause']} -> {edge['effect']}",
            )
        )

    emitted_input_edges = {
        (int(effect_index), int(input_index))
        for effect_index, input_index in zip(
            *np.where(np.asarray(spec.input_effect_block.free_support, dtype=bool)),
            strict=True,
        )
    }
    if emitted_input_edges != set(expected_input_edges):
        errors.append(
            "Structural edge closure mismatch for input effects: "
            f"planned={sorted(expected_input_edges)}, compiled={sorted(emitted_input_edges)}."
        )
    for target in sorted(emitted_input_edges & set(expected_input_edges)):
        edge = expected_input_edges[target]
        bindings.append(
            CompiledStructuralBinding(
                source_id=str(edge["source_id"]),
                source_kind="edge",
                target_kind="input_effect",
                target_indices=target,
                target_name=f"{edge['cause']} -> {edge['effect']}",
            )
        )

    innovation_dependencies: dict[tuple[int, int], list[UncheckedJsonObject]] = {}
    for dependency in dependencies:
        if dependency["kind"] != "innovation_correlation":
            continue
        first, second = (latent_idx[str(name)] for name in dependency["between"])
        target = (max(first, second), min(first, second))
        innovation_dependencies.setdefault(target, []).append(dependency)
    diffusion_support = np.asarray(spec.diffusion_block.diffusion_chol_support, dtype=bool)
    emitted_innovation_targets = {
        (row, col) for row, col in zip(*np.where(diffusion_support), strict=True) if row > col
    }
    if emitted_innovation_targets != set(innovation_dependencies):
        errors.append(
            "Induced innovation-dependency closure mismatch: "
            f"planned={sorted(innovation_dependencies)}, "
            f"compiled={sorted(emitted_innovation_targets)}."
        )
    for target in sorted(emitted_innovation_targets & set(innovation_dependencies)):
        for dependency in innovation_dependencies[target]:
            bindings.append(
                CompiledStructuralBinding(
                    source_id=str(dependency["source_id"]),
                    source_kind="induced_dependency",
                    target_kind="diffusion_correlation",
                    target_indices=target,
                    target_name=(
                        "correlated innovations: "
                        + ", ".join(str(name) for name in dependency["between"])
                    ),
                )
            )

    initial_dependencies = {
        str(dependency["source_id"]): dependency
        for dependency in dependencies
        if dependency["kind"] == "initial_state_correlation"
    }
    initial_scales = [
        scale
        for scale in get_marginalized_scales(structural_plan)
        if scale["kind"] == "initial_state_correlation"
    ]
    expected_factor_names = [str(scale["parameter"]) for scale in initial_scales]
    emitted_factor_names = list(spec.static_factor_names or [])
    if emitted_factor_names != expected_factor_names:
        errors.append(
            "Induced initial-state dependency closure mismatch: "
            f"planned factors={expected_factor_names!r}, compiled={emitted_factor_names!r}."
        )
    for factor_index, scale in enumerate(initial_scales):
        for dependency_id in scale["dependency_ids"]:
            dependency = initial_dependencies.get(str(dependency_id))
            if dependency is None:
                errors.append(
                    f"Static factor {scale['parameter']!r} references unknown dependency "
                    f"{dependency_id!r}."
                )
                continue
            bindings.append(
                CompiledStructuralBinding(
                    source_id=str(dependency_id),
                    source_kind="induced_dependency",
                    target_kind="static_factor",
                    target_indices=(factor_index,),
                    target_name=str(scale["parameter"]),
                )
            )

    expected_source_ids = {
        *state_ids,
        *(str(item["source_id"]) for item in manifests),
        *(str(item["source_id"]) for item in known_inputs),
        *(str(item["source_id"]) for item in edges),
        *(str(item["source_id"]) for item in dependencies),
    }
    bound_source_ids = {binding.source_id for binding in bindings}
    if bound_source_ids != expected_source_ids:
        errors.append(
            "Structural forward closure mismatch: "
            f"unbound={sorted(expected_source_ids - bound_source_ids)}, "
            f"unplanned={sorted(bound_source_ids - expected_source_ids)}."
        )

    if errors:
        raise StructuralClosureError(errors)
    return bindings


__all__ = [
    "StructuralClosureError",
    "compile_anchor_certificates",
    "compile_structural_bindings",
]
