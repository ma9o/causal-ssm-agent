"""Diffrax-backed forward simulator for counterfactual trajectories.

Single integration path used by all rung-2/rung-3 estimands. Same code
handles the deterministic mean trajectory (no diffusion) and stochastic
SDE samples, the linear and (eventually) non-linear vector fields, and
constant or time-varying interventions.

The simulator is intentionally minimal: it owns the Diffrax call and the
intervention initial-condition handoff, nothing else. Estimands
(treatment effects, summaries, manifest projections) live in
``estimands.py``; orchestration over posterior draws lives in
``orchestration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffrax as dfx
import jax.numpy as jnp

from .vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from jax import Array

    from .intervention import Intervention
    from .vector_field import VectorField


@dataclass(frozen=True)
class SimulationConfig:
    rtol: float = 1e-4
    atol: float = 1e-6
    max_steps: int = 4096


def simulate(
    vector_field: VectorField,
    params: dict[str, Array],
    intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
) -> Array:
    """Forward-simulate the SSM mean trajectory under ``intervention``.

    Args:
        vector_field: Drift callable for the SSM.
        params: Parameter pytree for the field.
        intervention: Override set active over the integration window.
        initial_state: ``(n_latent,)`` state at ``time_grid[0]``. Hard
            variable overrides are applied to this state before integration.
        time_grid: ``(T,)`` monotonically increasing array of evaluation
            times. ``time_grid[0]`` is the integration start.
        config: Solver tolerances; defaults are conservative for stable
            linear systems and adequate for moderately stiff non-linear
            fields.

    Returns:
        ``(T, n_latent)`` state trajectory at the requested grid.
    """
    cfg = config or SimulationConfig()
    args = VectorFieldArgs(params=params, intervention=intervention)

    y0 = vector_field.initial_condition(initial_state, args)
    t0 = time_grid[0]
    t1 = time_grid[-1]
    initial_dt = jnp.maximum((t1 - t0) / 256.0, 1e-6)

    term = dfx.ODETerm(lambda t, y, a: vector_field(t, y, a))
    solver = dfx.Tsit5()
    controller = dfx.PIDController(rtol=cfg.rtol, atol=cfg.atol)

    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=initial_dt,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(ts=time_grid),
        stepsize_controller=controller,
        max_steps=cfg.max_steps,
        throw=False,
    )
    return solution.ys


def simulate_pair(
    vector_field: VectorField,
    params: dict[str, Array],
    baseline_intervention: Intervention,
    action_intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
) -> tuple[Array, Array, Array]:
    """Simulate baseline and action paths and return ``(baseline, action,
    effect)`` where ``effect = action - baseline``.

    Sharing ``initial_state`` and ``time_grid`` between the two integrations
    makes the contrast a pure subtraction at matching grid points.
    """
    baseline = simulate(
        vector_field,
        params,
        baseline_intervention,
        initial_state,
        time_grid,
        config,
    )
    action = simulate(
        vector_field,
        params,
        action_intervention,
        initial_state,
        time_grid,
        config,
    )
    return baseline, action, action - baseline
