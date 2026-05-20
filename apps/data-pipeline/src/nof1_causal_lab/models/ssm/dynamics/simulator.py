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
    sde_dt: float | None = None
    """Constant step size for the SDE solver. ``None`` → ``(t1 - t0) / 200``."""
    sde_brownian_tol: float = 1e-3
    """Tolerance for ``VirtualBrownianTree``; smaller = finer Brownian path."""


def simulate(
    vector_field: VectorField,
    params: dict[str, Array],
    intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
    *,
    key: Array | None = None,
    diffusion_cov: Array | None = None,
) -> Array:
    """Forward-simulate the SSM trajectory under ``intervention``.

    Two modes:

    - **Deterministic mean trajectory** (default, ``key=None``): the ODE
      ``dy/dt = f(t, y; θ)`` is integrated with an adaptive Tsit5 solver.
      This is the cheap path used by Stage-6 counterfactual estimands.

    - **Single SDE sample** (``key`` and ``diffusion_cov`` both provided):
      the SDE ``dy = f(t, y; θ) dt + L dW`` is integrated with a Heun
      SDE solver, where ``L = chol(diffusion_cov + jitter·I)`` and ``dW``
      is a ``VirtualBrownianTree``-backed Wiener process. Each call with
      a fresh ``key`` produces an independent sample path. Use this for
      distributional counterfactuals: draw many samples to get a
      distribution of trajectories under intervention, rather than just
      the conditional mean.

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
        key: Optional JAX PRNG key. When provided alongside
            ``diffusion_cov``, the simulator returns one SDE sample.
        diffusion_cov: Optional ``(n_latent, n_latent)`` PSD diffusion
            covariance ``G·G'``. State-independent (additive Wiener).

    Returns:
        ``(T, n_latent)`` state trajectory at the requested grid. Mean
        trajectory in deterministic mode, one sample path in SDE mode.
    """
    cfg = config or SimulationConfig()
    args = VectorFieldArgs(params=params, intervention=intervention)

    y0 = vector_field.initial_condition(initial_state, args)
    t0 = time_grid[0]
    t1 = time_grid[-1]

    if key is None or diffusion_cov is None:
        if (key is None) != (diffusion_cov is None):
            raise ValueError(
                "SDE mode requires both 'key' and 'diffusion_cov'; got "
                f"key={'set' if key is not None else 'None'}, "
                f"diffusion_cov={'set' if diffusion_cov is not None else 'None'}."
            )
        # Deterministic ODE path.
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

    # SDE path: f(t,y) dt + L dW with L = chol(diffusion_cov).
    n_latent = vector_field.n_latent
    dtype = y0.dtype
    diffusion_cov = jnp.asarray(diffusion_cov, dtype=dtype)
    jitter = jnp.asarray(1e-8, dtype=dtype)
    chol_G = jnp.linalg.cholesky(
        diffusion_cov + jitter * jnp.eye(n_latent, dtype=dtype)
    )

    brownian = dfx.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=cfg.sde_brownian_tol,
        shape=(n_latent,),
        key=key,
    )
    ode_term = dfx.ODETerm(lambda t, y, a: vector_field(t, y, a))
    diffusion_term = dfx.ControlTerm(lambda _t, _y, _a: chol_G, brownian)
    term = dfx.MultiTerm(ode_term, diffusion_term)
    solver = dfx.Heun()
    dt0 = cfg.sde_dt if cfg.sde_dt is not None else float((t1 - t0) / 200.0)
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(ts=time_grid),
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
