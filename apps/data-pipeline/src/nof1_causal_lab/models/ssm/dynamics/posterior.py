"""Posterior sample assembly for vector-field dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from .composite import compile_composite, pack_component_params_from_samples

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.models.ssm.model import SSMSpec

    from .vector_field import CompositeVectorField


@dataclass(frozen=True)
class PosteriorDynamicsSamples:
    """Vector-field posterior draws reconstructed from canonical sites."""

    vector_field: CompositeVectorField
    param_samples: list[tuple[dict[str, Array], ...]]


def _posterior_draw_count(samples: dict[str, Any]) -> int:
    if not samples:
        return 0
    for values in samples.values():
        if hasattr(values, "shape") and len(values.shape) > 0:
            return int(values.shape[0])
    return 0


def component_param_samples_from_site_samples(
    spec: SSMSpec,
    samples: dict[str, Any],
    *,
    prefix: str = "vf",
) -> list[tuple[dict[str, Array], ...]]:
    """Pack posterior site samples into per-draw vector-field params."""
    n_draws = _posterior_draw_count(samples)
    param_samples: list[tuple[dict[str, Array], ...]] = []
    for draw_idx in range(n_draws):
        draw = {
            name: jnp.asarray(values)[draw_idx]
            for name, values in samples.items()
            if hasattr(values, "shape") and len(values.shape) > 0
        }
        param_samples.append(
            pack_component_params_from_samples(
                spec.dynamics_spec,
                draw,
                draw,
                prefix=prefix,
            )
        )
    return param_samples


def posterior_dynamics_from_samples(
    spec: SSMSpec,
    samples: dict[str, Any],
    *,
    prefix: str = "vf",
) -> PosteriorDynamicsSamples:
    """Rebuild posterior vector-field draws from an ``SSMSpec`` and samples."""
    compiled = compile_composite(spec.dynamics_spec, prefix=prefix)
    return PosteriorDynamicsSamples(
        vector_field=compiled.vector_field,
        param_samples=component_param_samples_from_site_samples(
            spec,
            samples,
            prefix=prefix,
        ),
    )


def posterior_dynamics_from_result(
    spec: SSMSpec,
    result,
    *,
    prefix: str = "vf",
) -> PosteriorDynamicsSamples:
    """Rebuild posterior vector-field draws from an inference result."""
    return posterior_dynamics_from_samples(
        spec,
        result.get_samples() or {},
        prefix=prefix,
    )
