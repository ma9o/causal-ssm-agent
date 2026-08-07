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
            {
                "kind": "HillEdge",
                "source": 3,
                "target": 4,
                "parameters": {
                    "emax": {"kind": "free"},
                    "ec50": {"kind": "free"},
                    "n": {"kind": "fixed", "value": 2.0}
                }
            }
        ]
    }
"""

from __future__ import annotations

from typing import Any

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.structure.parameters import Fixed, Free, ParameterSlot

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

type ParameterSlotConfig = dict[str, Any]
type ParameterizedComponentConfig = dict[str, Any]

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


def _parameter_slot_to_dict(slot: ParameterSlot) -> ParameterSlotConfig:
    """Serialize one explicit free-or-fixed parameter slot."""
    if isinstance(slot, Fixed):
        return {"kind": "fixed", "value": float(slot.value)}
    return {"kind": "free"}


def _parameter_slot_from_dict(payload: ParameterSlotConfig) -> ParameterSlot:
    """Deserialize one explicit free-or-fixed parameter slot."""
    kind = payload["kind"]
    if kind == "free":
        return Free()
    if kind == "fixed":
        return Fixed(float(payload["value"]))
    raise ValueError(f"Unknown parameter slot kind {kind!r}; expected 'free' or 'fixed'.")


def _component_parameter_slots(
    component: ParameterizedComponentConfig,
    names: tuple[str, ...],
) -> dict[str, ParameterSlot]:
    """Read the complete parameter-slot map for one serialized component."""
    parameters = component["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Dynamics component 'parameters' must be an object.")
    actual = set(parameters)
    expected = set(names)
    if actual != expected:
        raise ValueError(
            "Dynamics component parameter slots must exactly match its vocabulary: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )
    return {name: _parameter_slot_from_dict(parameters[name]) for name in names}


def dynamics_spec_to_dict(spec: DynamicsSpec) -> UncheckedJsonObject:
    """Inverse of :func:`dynamics_spec_from_dict`."""
    components: list[UncheckedJsonObject] = []
    for component in spec.components:
        if isinstance(component, StateDecaySpec):
            entry: UncheckedJsonObject = {
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
                "parameters": {
                    "center": _parameter_slot_to_dict(component.center),
                    "stiffness": _parameter_slot_to_dict(component.stiffness),
                    "quartic": _parameter_slot_to_dict(component.quartic),
                },
            }
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
                "parameters": {
                    "emax": _parameter_slot_to_dict(component.emax),
                    "ec50": _parameter_slot_to_dict(component.ec50),
                    "n": _parameter_slot_to_dict(component.n),
                },
            }
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


def _spec_from_component_dict(component: UncheckedJsonObject) -> Any:
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
        slots = _component_parameter_slots(component, ("center", "stiffness", "quartic"))
        return NodePotentialSpec(
            target=int(component["target"]),
            center=slots["center"],
            stiffness=slots["stiffness"],
            quartic=slots["quartic"],
        )
    if kind == "LinearEdge":
        return LinearEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
        )
    if kind == "HillEdge":
        slots = _component_parameter_slots(component, ("emax", "ec50", "n"))
        return HillEdgeSpec(
            source=int(component["source"]),
            target=int(component["target"]),
            emax=slots["emax"],
            ec50=slots["ec50"],
            n=slots["n"],
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


def dynamics_spec_from_dict(config: UncheckedJsonObject) -> DynamicsSpec:
    """Build a ``DynamicsSpec`` from a nested dict-config (see module docstring)."""
    if "n_latent" not in config:
        raise ValueError("dynamics_spec_from_dict requires 'n_latent'")
    components = tuple(_spec_from_component_dict(c) for c in config.get("components", ()))
    return DynamicsSpec(n_latent=int(config["n_latent"]), components=components)
