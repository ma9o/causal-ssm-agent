"""Counterfactual API — Pearl rung-2 / rung-3 estimands.

This module is a *consumer* of ``models/ssm/dynamics``. It uses the
vector-field substrate to compute intervention effects (compute_interventions)
and summarise effect distributions (summarize_draws,
summarize_temporal_effect, resolve_action_value, ...).

Rung-3 abduction (recovering the latent state at the evidence boundary)
is *not* done here: it reads the exact particle-smoother posterior latent
paths produced by the fit, never a linearised Kalman/RTS smoother.

The dynamics framework itself — vector fields, edges, intervention DSL,
compilation, discretisation, priors, stability, simulators — lives one
directory up in ``dynamics/``. Inference consumes from there directly,
not via this module.

Public API:

- ``compute_interventions`` — Stage-6 orchestrator
- ``summarize_draws``, ``summarize_temporal_effect``, ``build_time_grid``,
  ``resolve_action_value`` — estimand helpers
"""

from __future__ import annotations

from .estimands import (
    build_time_grid,
    resolve_action_value,
    summarize_draws,
    summarize_temporal_effect,
)
from .orchestration import (
    ClampSpec,
    build_segment_bounds,
    compute_interventions,
    vmap_simulate_clamps_from_state,
    vmap_steady_state_effect_dynamics,
)

__all__ = [
    "ClampSpec",
    "build_segment_bounds",
    "build_time_grid",
    "compute_interventions",
    "resolve_action_value",
    "summarize_draws",
    "summarize_temporal_effect",
    "vmap_simulate_clamps_from_state",
    "vmap_steady_state_effect_dynamics",
]
