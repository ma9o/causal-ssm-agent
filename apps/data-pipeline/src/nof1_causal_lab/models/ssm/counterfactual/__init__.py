"""Counterfactual simulation for the SSM.

Single Diffrax-backed trajectory path with an Optimistix steady-state
root-finder behind a vector-field protocol. The protocol is the seam where
the non-linear edge-primitive vocabulary will plug in later: a new
``VectorField`` implementation slots in behind the same simulator, steady
state, and orchestration code without touching call sites.

Public API:

- ``VectorField``, ``LinearVectorField``, ``VectorFieldArgs`` — dynamics
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
    vmap_simulate_action_from_state,
    vmap_steady_state_effect,
)
from .simulator import SimulationConfig, simulate, simulate_pair
from .steady_state import compute_steady_state
from .vector_field import LinearVectorField, VectorField, VectorFieldArgs

__all__ = [
    "EdgeInputOverride",
    "Intervention",
    "LinearVectorField",
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
    "linear_ramp",
    "project_to_manifest",
    "resolve_action_value",
    "simulate",
    "simulate_pair",
    "summarize_draws",
    "summarize_temporal_effect",
    "vmap_simulate_action_from_state",
    "vmap_steady_state_effect",
]
