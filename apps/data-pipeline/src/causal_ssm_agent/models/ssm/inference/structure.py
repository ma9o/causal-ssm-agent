"""Shared inference-structure planning for backend routing and UI payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import RBPartition
    from causal_ssm_agent.models.ssm.model import SSMSpec
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

StructuralBackend = Literal["kalman", "composed", "particle"]
RequestedMethod = Literal[
    "auto",
    "aux_csmc",
    "particle_mgrad",
    "aux_gibbs",
    "nuts",
    "map",
    "svi",
]
ResolvedMethod = Literal[
    "aux_csmc",
    "particle_mgrad",
    "aux_gibbs",
    "nuts",
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
    if method_override in {None, "auto"}:
        return None
    return cast("ResolvedMethod", method_override)


def _resolve_structural_backend(
    spec: SSMSpec,
    *,
    likelihood: Literal["particle", "kalman"],
    observation_support: ObservationSupportRuntime | None,
) -> tuple[StructuralBackend, RBPartition | None]:
    from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import analyze_first_pass_rb

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


def _resolve_auto_method(
    *,
    structural_backend: StructuralBackend,
    observation_support: ObservationSupportRuntime | None,
    n_timepoints: int | None,
) -> ResolvedMethod:
    del structural_backend, observation_support, n_timepoints
    return "nuts"


def _payload_partition_for_plan(
    spec: SSMSpec,
    plan: InferenceStructurePlan,
    *,
    likelihood: Literal["particle", "kalman"],
    observation_support: ObservationSupportRuntime | None,
) -> RBPartition | None:
    if plan.first_pass_partition is not None:
        return plan.first_pass_partition

    if (
        plan.structural_backend != "kalman"
        or not spec.first_pass_rb
        or likelihood == "kalman"
        or (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        )
    ):
        return None

    from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import analyze_first_pass_rb

    partition = analyze_first_pass_rb(spec)
    return partition if not partition.has_particle_block else None


def plan_inference_structure(
    spec: SSMSpec,
    *,
    likelihood: Literal["particle", "kalman"] = "particle",
    observation_support: ObservationSupportRuntime | None = None,
    method_override: RequestedMethod | None = None,
    n_timepoints: int | None = None,
) -> InferenceStructurePlan:
    """Resolve the active likelihood path and auto-routing plan once."""
    structural_backend, first_pass_partition = _resolve_structural_backend(
        spec,
        likelihood=likelihood,
        observation_support=observation_support,
    )
    normalized_override = _normalize_method_override(method_override)
    resolved_method = normalized_override or _resolve_auto_method(
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


def build_inference_structure_payload(
    spec: SSMSpec,
    plan: InferenceStructurePlan,
    *,
    likelihood: Literal["particle", "kalman"] = "particle",
    observation_support: ObservationSupportRuntime | None = None,
) -> dict:
    """Serialize an inference-structure plan for stage payloads."""
    latent_names = spec.latent_names or [f"latent_{i}" for i in range(spec.n_latent)]
    manifest_names = spec.manifest_names or [f"obs_{i}" for i in range(spec.n_manifest)]

    latent_variables: list[dict[str, str]] = []
    obs_variables: list[dict[str, str]] = []
    partition = _payload_partition_for_plan(
        spec,
        plan,
        likelihood=likelihood,
        observation_support=observation_support,
    )
    if partition is not None:
        latent_variables = [
            {
                "name": latent_names[i],
                "method": "kalman" if i in partition.kalman_idx else "particle",
            }
            for i in range(spec.n_latent)
        ]
        obs_variables = [
            {
                "name": manifest_names[i],
                "method": "kalman" if i in partition.obs_kalman_idx else "particle",
            }
            for i in range(spec.n_manifest)
        ]

    auto_method = _resolve_auto_method(
        structural_backend=plan.structural_backend,
        observation_support=observation_support,
        n_timepoints=None,
    )

    return {
        "likelihood_path": plan.structural_backend,
        "auto_method": auto_method,
        "first_pass_rb": {
            "status": "active" if partition is not None else "inactive",
            "latent_variables": latent_variables,
            "obs_variables": obs_variables,
        },
    }
