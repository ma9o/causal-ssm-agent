"""Affine views derived from vector-field runtime dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.dynamics.edges import (
    DenseLinear,
    DiagonalDecay,
    Intercept,
    LinearEdge,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.base import RuntimeDynamics


class AffineDynamicsParams(NamedTuple):
    """Inference-internal affine dynamics view."""

    drift: jnp.ndarray
    diffusion_cov: jnp.ndarray
    cint: jnp.ndarray | None
    input_effect: jnp.ndarray | None = None


def derive_affine_dynamics(dynamics: RuntimeDynamics) -> AffineDynamicsParams:
    """Derive an exact affine view when every vector-field component is affine."""
    n_latent = int(dynamics.vector_field.n_latent)
    dtype = dynamics.diffusion_cov.dtype
    drift = jnp.zeros((n_latent, n_latent), dtype=dtype)
    cint = jnp.zeros((n_latent,), dtype=dtype)
    has_cint = False

    components = dynamics.vector_field.components
    if len(components) != len(dynamics.vf_params):
        raise ValueError(
            "RuntimeDynamics component count does not match parameter tuple: "
            f"{len(components)} components vs {len(dynamics.vf_params)} parameter slices."
        )

    for component, params in zip(components, dynamics.vf_params, strict=True):
        if isinstance(component, DenseLinear):
            drift = drift + params["drift"]
            if "cint" in params:
                cint = cint + params["cint"]
                has_cint = True
        elif isinstance(component, DiagonalDecay):
            drift = drift - jnp.diag(params["decay"])
        elif isinstance(component, Intercept):
            cint = cint + params["cint"]
            has_cint = True
        elif isinstance(component, LinearEdge):
            drift = drift.at[component.target, component.source].add(params["weight"])
        else:
            raise NotImplementedError(
                "This inference backend requires affine vector-field dynamics. "
                f"{type(component).__name__} needs a nonlinear likelihood backend."
            )

    return AffineDynamicsParams(
        drift=drift,
        diffusion_cov=dynamics.diffusion_cov,
        cint=cint if has_cint else None,
        input_effect=dynamics.input_effect,
    )
