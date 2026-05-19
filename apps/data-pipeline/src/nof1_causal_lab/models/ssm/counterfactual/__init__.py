"""Counterfactual simulation for the SSM.

One vector-field class (``CompositeVectorField``) composes drift
components into the system's dynamics. ``DenseLinear`` recovers the
existing Stage 5b dense-posterior shape ``f(η) = A·η + c`` in one
matmul; per-edge primitives (``LinearEdge``, ``HillEdge``,
``MultiplicativeEdge``) plus ``DiagonalDecay`` and ``Intercept`` are
how the LLM-elicited non-linear vocabulary plugs in. Either case goes
through the same Diffrax simulator and Optimistix steady-state
root-finder.

Public API:

- ``CompositeVectorField``, ``VectorFieldArgs``, ``VectorField`` — dynamics
- ``DriftComponent``, ``DenseLinear``, ``DiagonalDecay``, ``Intercept``,
  ``LinearEdge``, ``HillEdge``, ``MultiplicativeEdge`` — components
- ``linear_vector_field`` — factory for the dense-linear case
- ``Intervention``, ``VariableOverride``, ``EdgeInputOverride``,
  ``constant_value``, ``linear_ramp`` — intervention DSL
- ``simulate``, ``simulate_pair``, ``SimulationConfig`` — Diffrax forward
- ``compute_steady_state`` — Optimistix root-find
- ``summarize_draws``, ``summarize_temporal_effect``, ``resolve_action_value``,
  ``build_time_grid``, ``project_to_manifest`` — estimand helpers
- ``compute_interventions`` — Stage-6 orchestrator
- ``vmap_steady_state_effect``, ``vmap_simulate_action_from_state`` —
  ``tool_server`` vmapped entry points
- ``approximate_abducted_state`` — rung-3 abduction (Kalman smoother)
"""

from __future__ import annotations

from .abduction import approximate_abducted_state
from .discretization import discretize_at_state
from .edges import (
    DenseLinear,
    DiagonalDecay,
    DriftComponent,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
)
from .estimands import (
    build_time_grid,
    project_to_manifest,
    resolve_action_value,
    summarize_draws,
    summarize_temporal_effect,
)
from .intervention import (
    EdgeInputOverride,
    Intervention,
    Override,
    ValueFn,
    VariableOverride,
    constant_value,
    linear_ramp,
)
from .orchestration import (
    compute_interventions,
    linear_vector_field,
    vmap_simulate_action_from_state,
    vmap_steady_state_effect,
)
from .simulator import SimulationConfig, simulate, simulate_pair
from .steady_state import compute_steady_state
from .vector_field import CompositeVectorField, VectorField, VectorFieldArgs

__all__ = [
    "CompositeVectorField",
    "DenseLinear",
    "DiagonalDecay",
    "DriftComponent",
    "EdgeInputOverride",
    "HillEdge",
    "Intercept",
    "Intervention",
    "LinearEdge",
    "MultiplicativeEdge",
    "Override",
    "SimulationConfig",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "approximate_abducted_state",
    "build_time_grid",
    "compute_interventions",
    "compute_steady_state",
    "constant_value",
    "discretize_at_state",
    "linear_ramp",
    "linear_vector_field",
    "project_to_manifest",
    "resolve_action_value",
    "simulate",
    "simulate_pair",
    "summarize_draws",
    "summarize_temporal_effect",
    "vmap_simulate_action_from_state",
    "vmap_steady_state_effect",
]
