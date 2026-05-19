"""Shared inference-structure planning for backend routing and UI payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import RBPartition
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm_observation_metadata import ObservationSupportRuntime

StructuralBackend = Literal["kalman", "composed", "particle"]
RequestedMethod = Literal[
    "particle_mgrad",
    "aux_gibbs",
    "map",
    "svi",
]
ResolvedMethod = Literal[
    "particle_mgrad",
    "aux_gibbs",
    "map",
    "svi",
]


@dataclass(frozen=True)
class InferenceStructurePlan:
    """Canonical structural plan shared across runtime prep and inference."""

    structural_backend: StructuralBackend
    resolved_method: ResolvedMethod
    method_override: ResolvedMethod | None
    first_pass_partition: RBPartition | None = None


def _normalize_method_override(
    method_override: RequestedMethod | None,
) -> ResolvedMethod | None:
    if method_override is None:
        return None
    if method_override not in {"particle_mgrad", "aux_gibbs", "map", "svi"}:
        raise ValueError(
            "Unsupported inference method override "
            f"{method_override!r}; expected 'particle_mgrad', 'aux_gibbs', 'map', or 'svi'."
        )
    return cast("ResolvedMethod", method_override)


def _resolve_structural_backend(
    spec: SSMSpec,
    *,
    likelihood: Literal["particle", "kalman"],
    observation_support: ObservationSupportRuntime | None,
) -> tuple[StructuralBackend, RBPartition | None]:
    from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import analyze_first_pass_rb

    if observation_support is not None and observation_support.requires_interval_summary_handling:
        return "particle", None

    if likelihood == "kalman":
        return "kalman", None

    partition = analyze_first_pass_rb(spec)
    if not partition.has_particle_block:
        return "kalman", None

    if (
        spec.first_pass_rb
        and partition.has_kalman_block
        and len(partition.particle_idx) > 0
        and len(partition.obs_kalman_idx) > 0
    ):
        return "composed", partition

    return "particle", None


def _resolve_default_method(
    *,
    structural_backend: StructuralBackend,
    observation_support: ObservationSupportRuntime | None,
    n_timepoints: int | None,
) -> ResolvedMethod:
    del structural_backend, observation_support, n_timepoints
    return "aux_gibbs"


def plan_inference_structure(
    spec: SSMSpec,
    *,
    likelihood: Literal["particle", "kalman"] = "particle",
    observation_support: ObservationSupportRuntime | None = None,
    method_override: RequestedMethod | None = None,
    n_timepoints: int | None = None,
) -> InferenceStructurePlan:
    """Resolve the active likelihood path and default inference plan once."""
    structural_backend, first_pass_partition = _resolve_structural_backend(
        spec,
        likelihood=likelihood,
        observation_support=observation_support,
    )
    normalized_override = _normalize_method_override(method_override)
    resolved_method = normalized_override or _resolve_default_method(
        structural_backend=structural_backend,
        observation_support=observation_support,
        n_timepoints=n_timepoints,
    )
    return InferenceStructurePlan(
        structural_backend=structural_backend,
        resolved_method=resolved_method,
        method_override=normalized_override,
        first_pass_partition=first_pass_partition,
    )
