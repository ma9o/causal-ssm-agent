"""Serializable dynamics-spec config for the runtime ``DynamicsSpec``.

Dynamics component specs are structural only. Priors live in the
canonical site-prior registry and are serialized separately from this
component topology.

Schema
------

Each component is a single dict with a ``kind`` discriminator plus
component-specific indices.

Example::

    {
        "n_latent": 5,
        "components": [
            {"kind": "DiagonalDecay"},
            {"kind": "HillEdge", "source": 3, "target": 4}
        ]
    }
"""

from __future__ import annotations

from typing import Any

from .spec import (
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
    InterceptSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    NodePotentialSpec,
    StateDecaySpec,
    StateInterceptSpec,
)

_COMPONENT_KIND_REGISTRY: dict[str, str] = {
    "StateDecay": "state_decay",
    "DiagonalDecay": "diagonal_decay",
    "StateIntercept": "state_intercept",
    "Intercept": "intercept",
    "NodePotential": "node_potential",
    "LinearEdge": "linear_edge",
    "HillEdge": "hill_edge",
    "MultiplicativeEdge": "multiplicative_edge",
}


def dynamics_spec_to_dict(spec: DynamicsSpec) -> dict[str, Any]:
    """Inverse of :func:`dynamics_spec_from_dict`."""
    components: list[dict[str, Any]] = []
    for component in spec.components:
        if isinstance(component, StateDecaySpec):
            entry: dict[str, Any] = {
                "kind": "StateDecay",
                "target": int(component.target),
            }
        elif isinstance(component, DiagonalDecaySpec):
            entry = {
                "kind": "DiagonalDecay",
            }
        elif isinstance(component, StateInterceptSpec):
            entry = {
                "kind": "StateIntercept",
                "target": int(component.target),
            }
        elif isinstance(component, InterceptSpec):
            entry = {
                "kind": "Intercept",
            }
        elif isinstance(component, NodePotentialSpec):
            entry = {
                "kind": "NodePotential",
                "target": int(component.target),
            }
            if component.fixed_center is not None:
                entry["fixed_center"] = float(component.fixed_center)
            if component.fixed_stiffness is not None:
                entry["fixed_stiffness"] = float(component.fixed_stiffness)
            if component.fixed_quartic is not None:
                entry["fixed_quartic"] = float(component.fixed_quartic)
        elif isinstance(component, LinearEdgeSpec):
            entry = {
                "kind": "LinearEdge",
                "source": int(component.source),
                "target": int(component.target),
            }
        elif isinstance(component, HillEdgeSpec):
            entry = {
                "kind": "HillEdge",
                "source": int(component.source),
                "target": int(component.target),
            }
            if component.fixed_emax is not None:
                entry["fixed_emax"] = float(component.fixed_emax)
            if component.fixed_ec50 is not None:
                entry["fixed_ec50"] = float(component.fixed_ec50)
            if component.fixed_n is not None:
                entry["fixed_n"] = float(component.fixed_n)
        elif isinstance(component, MultiplicativeEdgeSpec):
            entry = {
                "kind": "MultiplicativeEdge",
                "source_a": int(component.source_a),
                "source_b": int(component.source_b),
                "target": int(component.target),
            }
        else:
            raise ValueError(
                f"dynamics_spec_to_dict: unsupported component type {type(component).__name__}"
            )
        components.append(entry)
    return {"n_latent": int(spec.n_latent), "components": components}


def _spec_from_component_dict(component: dict[str, Any]) -> Any:
    """Build a component spec from a dict-config."""
    kind = component["kind"]

    if kind == "StateDecay":
        return StateDecaySpec(target=int(component["target"]))
    if kind == "DiagonalDecay":
        return DiagonalDecaySpec()
    if kind == "StateIntercept":
        return StateInterceptSpec(target=int(component["target"]))
    if kind == "Intercept":
        return InterceptSpec()
    if kind == "NodePotential":
        return NodePotentialSpec(
            target=int(component["target"]),
            fixed_center=(
                None if "fixed_center" not in component else float(component["fixed_center"])
            ),
            fixed_stiffness=(
                None if "fixed_stiffness" not in component else float(component["fixed_stiffness"])
            ),
            fixed_quartic=(
                None if "fixed_quartic" not in component else float(component["fixed_quartic"])
            ),
        )
    if kind == "LinearEdge":
        return LinearEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
        )
    if kind == "HillEdge":
        return HillEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
            fixed_emax=(None if "fixed_emax" not in component else float(component["fixed_emax"])),
            fixed_ec50=(None if "fixed_ec50" not in component else float(component["fixed_ec50"])),
            fixed_n=None if "fixed_n" not in component else float(component["fixed_n"]),
        )
    if kind == "MultiplicativeEdge":
        return MultiplicativeEdgeSpec(
            source_a=int(component["source_a"]),
            source_b=int(component["source_b"]),
            target=int(component["target"]),
        )
    raise ValueError(
        f"Unknown dynamics component kind {kind!r}; known: {sorted(_COMPONENT_KIND_REGISTRY)}"
    )


def dynamics_spec_from_dict(config: dict[str, Any]) -> DynamicsSpec:
    """Build a ``DynamicsSpec`` from a nested dict-config (see module docstring)."""
    if "n_latent" not in config:
        raise ValueError("dynamics_spec_from_dict requires 'n_latent'")
    components = tuple(_spec_from_component_dict(c) for c in config.get("components", ()))
    return DynamicsSpec(n_latent=int(config["n_latent"]), components=components)
