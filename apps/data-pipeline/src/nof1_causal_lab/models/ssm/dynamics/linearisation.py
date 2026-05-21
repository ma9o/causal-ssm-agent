"""Vector-field linearisation classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .edges import DenseLinear, DiagonalDecay, Intercept, LinearEdge, StateDecay, StateIntercept

if TYPE_CHECKING:
    from .vector_field import CompositeVectorField

Linearisation = Literal["constant", "trajectory"]

_CONSTANT_JACOBIAN_COMPONENTS = (
    DenseLinear,
    DiagonalDecay,
    StateDecay,
    Intercept,
    StateIntercept,
    LinearEdge,
)


def infer_linearisation(vector_field: CompositeVectorField) -> Linearisation:
    """Classify whether a composite vector field has a state-independent Jacobian."""
    return (
        "constant"
        if all(isinstance(c, _CONSTANT_JACOBIAN_COMPONENTS) for c in vector_field.components)
        else "trajectory"
    )
