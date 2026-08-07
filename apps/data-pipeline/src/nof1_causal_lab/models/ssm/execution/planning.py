"""Shared exact execution-method planning for runtime prep and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

StructuralBackend = Literal["laplace"]
RequestedMethod = Literal["marginal_particle_gibbs"]
ResolvedMethod = Literal["marginal_particle_gibbs"]


@dataclass(frozen=True)
class InferenceStructurePlan:
    """Canonical structural plan shared across runtime prep and inference."""

    structural_backend: StructuralBackend
    resolved_method: ResolvedMethod
    method_override: ResolvedMethod | None


def _normalize_method_override(
    method_override: RequestedMethod | None,
) -> ResolvedMethod | None:
    if method_override is None:
        return None
    if method_override != "marginal_particle_gibbs":
        raise ValueError(
            "Unsupported inference method override "
            f"{method_override!r}; expected 'marginal_particle_gibbs'."
        )
    return method_override


def _resolve_default_method(
    *,
    structural_backend: StructuralBackend,
    observation_support: ObservationSupportRuntime | None,
    n_timepoints: int | None,
) -> ResolvedMethod:
    del structural_backend, observation_support, n_timepoints
    return "marginal_particle_gibbs"


def plan_inference_structure(
    spec: SSMSpec,
    *,
    observation_support: ObservationSupportRuntime | None = None,
    method_override: RequestedMethod | None = None,
    n_timepoints: int | None = None,
) -> InferenceStructurePlan:
    """Resolve the default inference plan once."""
    del spec
    structural_backend: StructuralBackend = "laplace"
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
    )
