"""Counterfactual API — Pearl rung-2 / rung-3 estimands.

This module is a *consumer* of ``models/ssm/dynamics``. It uses the
vector-field substrate to compute intervention effects (compute_interventions),
abduct rung-3 latent states from observed history (approximate_abducted_state),
and summarise effect distributions (summarize_draws,
summarize_temporal_effect, resolve_action_value, ...).

The dynamics framework itself — vector fields, edges, intervention DSL,
compilation, discretisation, priors, stability, simulators — lives one
directory up in ``dynamics/``. Inference consumes from there directly,
not via this module.

Public API:

- ``compute_interventions`` — Stage-6 orchestrator
- ``approximate_abducted_state`` — rung-3 abduction (Kalman smoother)
- ``summarize_draws``, ``summarize_temporal_effect``, ``build_time_grid``,
  ``resolve_action_value``, ``project_to_manifest`` — estimand helpers
"""

from __future__ import annotations

from .abduction import (
    approximate_abducted_state,
)
from .estimands import (
    build_time_grid,
    project_to_manifest,
    resolve_action_value,
    summarize_draws,
    summarize_temporal_effect,
)
from .orchestration import (
    compute_interventions,
    compute_interventions_composite,
    vmap_simulate_action_from_state_composite,
    vmap_steady_state_effect_composite,
)

__all__ = [
    "approximate_abducted_state",
    "build_time_grid",
    "compute_interventions",
    "compute_interventions_composite",
    "project_to_manifest",
    "resolve_action_value",
    "summarize_draws",
    "summarize_temporal_effect",
    "vmap_simulate_action_from_state_composite",
    "vmap_steady_state_effect_composite",
]
