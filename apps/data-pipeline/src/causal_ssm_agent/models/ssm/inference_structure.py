"""Shared inference-structure planning for backend routing and UI payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.graph_analysis import RBPartition
    from causal_ssm_agent.models.ssm.model import SSMSpec
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

LikelihoodPath = Literal["kalman", "composed", "particle"]
AutoMethod = Literal["nuts", "laplace_em", "svi"]
FirstPassRBStatus = Literal["active", "inactive"]
FirstPassRBInactiveReason = Literal[
    "disabled_in_spec",
    "interval_summary_support",
    "no_executable_partition",
    "likelihood_override",
]


@dataclass(frozen=True)
class FirstPassRBPlan:
    """Whether first-pass Rao-Blackwellization is active in the runtime path."""

    status: FirstPassRBStatus
    inactive_reason: FirstPassRBInactiveReason | None = None
    partition: RBPartition | None = None

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class InferenceStructurePlan:
    """Canonical structural plan shared across runtime prep and inference."""

    likelihood_path: LikelihoodPath
    auto_method: AutoMethod
    first_pass_rb: FirstPassRBPlan


def plan_inference_structure(
    spec: SSMSpec,
    *,
    likelihood: Literal["particle", "kalman"] = "particle",
    observation_support: ObservationSupportRuntime | None = None,
) -> InferenceStructurePlan:
    """Resolve the active likelihood path and auto-routing plan once."""
    from causal_ssm_agent.models.likelihoods.graph_analysis import analyze_first_pass_rb

    if observation_support is not None and observation_support.requires_interval_summary_handling:
        return InferenceStructurePlan(
            likelihood_path="particle",
            auto_method="laplace_em",
            first_pass_rb=FirstPassRBPlan(
                status="inactive",
                inactive_reason="interval_summary_support",
            ),
        )

    if likelihood == "kalman":
        return InferenceStructurePlan(
            likelihood_path="kalman",
            auto_method="nuts",
            first_pass_rb=FirstPassRBPlan(
                status="inactive",
                inactive_reason="likelihood_override",
            ),
        )

    partition = analyze_first_pass_rb(spec)
    auto_method: AutoMethod = "nuts" if not partition.has_particle_block else "laplace_em"

    if not spec.first_pass_rb:
        return InferenceStructurePlan(
            likelihood_path="particle",
            auto_method=auto_method,
            first_pass_rb=FirstPassRBPlan(
                status="inactive",
                inactive_reason="disabled_in_spec",
            ),
        )

    if not partition.has_particle_block:
        return InferenceStructurePlan(
            likelihood_path="kalman",
            auto_method=auto_method,
            first_pass_rb=FirstPassRBPlan(status="active", partition=partition),
        )

    if (
        partition.has_kalman_block
        and len(partition.particle_idx) > 0
        and len(partition.obs_kalman_idx) > 0
    ):
        return InferenceStructurePlan(
            likelihood_path="composed",
            auto_method=auto_method,
            first_pass_rb=FirstPassRBPlan(status="active", partition=partition),
        )

    return InferenceStructurePlan(
        likelihood_path="particle",
        auto_method=auto_method,
        first_pass_rb=FirstPassRBPlan(
            status="inactive",
            inactive_reason="no_executable_partition",
        ),
    )


def build_inference_structure_payload(
    spec: SSMSpec,
    plan: InferenceStructurePlan,
) -> dict:
    """Serialize an inference-structure plan for stage payloads."""
    latent_names = spec.latent_names or [f"latent_{i}" for i in range(spec.n_latent)]
    manifest_names = spec.manifest_names or [f"obs_{i}" for i in range(spec.n_manifest)]

    latent_variables: list[dict[str, str]] = []
    obs_variables: list[dict[str, str]] = []
    partition = plan.first_pass_rb.partition
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

    return {
        "likelihood_path": plan.likelihood_path,
        "auto_method": plan.auto_method,
        "first_pass_rb": {
            "status": plan.first_pass_rb.status,
            "inactive_reason": plan.first_pass_rb.inactive_reason,
            "latent_variables": latent_variables,
            "obs_variables": obs_variables,
        },
    }
