"""Canonical runtime construction for component-owned vector fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpyro.distributions as dist
    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField
    from nof1_causal_lab.models.ssm.model import SSMSpec

    PriorFn = Callable[[str], dist.Distribution]
    ComponentParams = tuple[dict[str, Array], ...]


@dataclass(frozen=True)
class VectorFieldRuntime:
    """Executable vector field plus the parameter tuple it consumes."""

    vector_field: VectorField
    vf_params: ComponentParams


def sample_vector_field_runtime(
    spec: SSMSpec,
    prior_fn: PriorFn,
    *,
    prefix: str = "vf",
) -> VectorFieldRuntime:
    """Sample component parameters inside a NumPyro model context."""
    compiled = compile_dynamics(spec.dynamics_spec, prefix=prefix)
    return VectorFieldRuntime(
        vector_field=compiled.vector_field,
        vf_params=compiled.sample_params(prior_fn),
    )


def pack_vector_field_params_from_samples(
    spec: SSMSpec,
    samples: Mapping[str, Array],
    deterministics: Mapping[str, Array] | None = None,
    *,
    prefix: str = "vf",
) -> ComponentParams:
    """Pack flat runtime site values into component parameter slices."""
    sample_dict = dict(samples)
    deterministic_dict = sample_dict if deterministics is None else dict(deterministics)
    return pack_component_params_from_samples(
        spec.dynamics_spec,
        sample_dict,
        deterministic_dict,
        prefix=prefix,
    )


def build_vector_field_runtime_from_samples(
    spec: SSMSpec,
    samples: Mapping[str, Array],
    deterministics: Mapping[str, Array] | None = None,
    *,
    prefix: str = "vf",
) -> VectorFieldRuntime:
    """Build executable vector-field runtime from constrained site values."""
    compiled = compile_dynamics(spec.dynamics_spec, prefix=prefix)
    return VectorFieldRuntime(
        vector_field=compiled.vector_field,
        vf_params=pack_vector_field_params_from_samples(
            spec,
            samples,
            deterministics,
            prefix=prefix,
        ),
    )
