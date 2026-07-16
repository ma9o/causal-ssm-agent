"""Stage-6 entry point: rank treatments by intervention effect."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import Array

from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VariableOverride,
    VectorField,
    compute_steady_state,
    constant_value,
    linear_ramp,
    precomputed_value,
    simulate,
    simulate_pair,
)

from .estimands import (
    build_time_grid,
    summarize_temporal_effect,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _steady_state_treatment_effect_canonical(
    vector_field: VectorField,
    params: tuple[dict[str, Array], ...],
    treat_idx: int,
    outcome_idx: int,
    shift_size: float,
) -> Array:
    """Equilibrium contrast: ``effect = (do(treat = baseline+shift) - baseline)[outcome]``.

    Canonical implementation: takes a per-component ``params`` tuple
    directly. A dense affine component passes ``({"drift": A, "cint": c},)``;
    component-owned vector fields pass the per-component tuple they already
    have.
    """
    baseline = compute_steady_state(vector_field, params, Intervention.none())
    do_value = baseline[treat_idx] + shift_size
    intervention = Intervention(
        overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
    )
    intervened = compute_steady_state(vector_field, params, intervention, initial_guess=baseline)
    return intervened[outcome_idx] - baseline[outcome_idx]


def _temporal_treatment_effect_canonical(
    vector_field: VectorField,
    params: tuple[dict[str, Array], ...],
    treat_idx: int,
    shift_size: float,
    time_grid: Array,
) -> Array:
    """Per-time effect trajectory: ``(action - baseline)`` over ``time_grid``.

    Canonical implementation: see :func:`_steady_state_treatment_effect_canonical`.
    """
    baseline_state = compute_steady_state(vector_field, params, Intervention.none())
    do_value = baseline_state[treat_idx] + shift_size
    action_intervention = Intervention(
        overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
    )
    _, _, effect_path = simulate_pair(
        vector_field,
        params,
        Intervention.none(),
        action_intervention,
        baseline_state,
        time_grid,
    )
    return effect_path


def compute_interventions(
    param_samples: Sequence[tuple[dict[str, Array], ...]],
    vector_field: VectorField,
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_design: dict | None = None,
    manifest_names: list[str] | None = None,
    times: jnp.ndarray | None = None,
    shift_size: float = 1.0,
    lambda_mean: Array | None = None,
) -> list[dict[str, Any]]:
    """Compute interventions from vector-field posterior parameter draws."""
    name_to_idx = {name: i for i, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)

    def _skeleton(treatment_name: str) -> dict[str, Any]:
        return {"treatment": treatment_name}

    if outcome_idx is None:
        logger.warning("Outcome '%s' not found in latent names %s", outcome, latent_names)
        return [_skeleton(t) for t in treatments]

    if not param_samples:
        logger.warning("No posterior parameter samples for vector-field intervention")
        return [_skeleton(t) for t in treatments]

    time_grid = _build_horizon_grid(causal_design, times)

    results: list[dict[str, Any]] = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            logger.warning("'%s' not in latent structure — skipping", treatment_name)
            results.append(_skeleton(treatment_name))
            continue

        effects = jnp.stack(
            [
                _steady_state_treatment_effect_canonical(
                    vector_field, p, treat_idx, outcome_idx, shift_size
                )
                for p in param_samples
            ]
        )
        mean_effect = float(jnp.mean(effects))
        entry: dict[str, Any] = {
            "treatment": treatment_name,
            "posterior_draws": effects.tolist(),
        }
        if time_grid is not None:
            try:
                trajectories = jnp.stack(
                    [
                        _temporal_treatment_effect_canonical(
                            vector_field, p, treat_idx, shift_size, time_grid
                        )
                        for p in param_samples
                    ]
                )
                mean_traj = jnp.mean(trajectories[:, :, outcome_idx], axis=0)
                entry["temporal"] = summarize_temporal_effect(mean_traj, time_grid)

                if lambda_mean is not None and lambda_mean.ndim == 2:
                    m_names = manifest_names or []
                    loadings = lambda_mean[:, outcome_idx]
                    manifest_effects = {}
                    for mi in range(len(loadings)):
                        loading_val = float(loadings[mi])
                        if abs(loading_val) > 1e-6:
                            name = m_names[mi] if mi < len(m_names) else f"manifest_{mi}"
                            manifest_effects[name] = loading_val * mean_effect
                    if manifest_effects:
                        entry["manifest_effects"] = manifest_effects
            except (ValueError, RuntimeError, FloatingPointError):
                logger.warning(
                    "Vector-field simulation failed for '%s'; temporal effects unavailable",
                    treatment_name,
                    exc_info=True,
                )
        results.append(entry)

    def _abs_mean(entry: dict) -> float:
        draws = entry.get("posterior_draws")
        return abs(sum(draws) / len(draws)) if draws else 0.0

    results.sort(key=_abs_mean, reverse=True)
    return results


def _build_horizon_grid(
    causal_design: dict | None,
    times: jnp.ndarray | None,
    horizon_days: float = 30.0,
) -> Array | None:
    """Derive a forward simulation grid from the measurement clock or
    median observation spacing. Returns ``None`` when no usable step size
    is available."""
    dt_days: float | None = None
    model_clock_str = (causal_design or {}).get("measurement", {}).get("model_clock")
    if model_clock_str:
        from nof1_causal_lab.artifacts.duration import parse_duration_to_hours

        dt_days = parse_duration_to_hours(model_clock_str) / 24.0
    elif times is not None and len(times) > 1:
        diffs = jnp.diff(times)
        dt_days = float(jnp.median(diffs))

    if dt_days is None or dt_days <= 0:
        return None

    return build_time_grid(0.0, horizon_days, dt_days)


def _stack_dynamics_params(
    param_samples: list[tuple[dict[str, Array], ...]],
) -> tuple[dict[str, Array], ...]:
    """Stack per-draw component parameters into a batched pytree."""
    import jax

    return jax.tree.map(lambda *values: jnp.stack(values), *param_samples)


@dataclass(frozen=True)
class ClampSpec:
    """A resolved latent clamp (do-operator) for orchestration.

    ``index`` is the latent dimension; window ``[from_day, to_day)`` is in days
    relative to the rollout start (``to_day=None`` runs through the horizon).
    Value parameters are static (from the validated tool input); ``shift`` is the
    one mode resolved per-draw against the start state.
    """

    index: int
    mode: str  # "set" | "shift" | "ramp" | "trajectory"
    from_day: float = 0.0
    to_day: float | None = None
    value: float | None = None
    amount: float | None = None
    value_start: float | None = None
    value_end: float | None = None
    values: tuple[float, ...] | None = None


def _nearest_grid_index(time_grid: Array, day: float) -> int:
    return int(jnp.argmin(jnp.abs(jnp.asarray(time_grid) - day)))


def build_segment_bounds(time_grid: Array, clamps: list[ClampSpec]) -> list[tuple[int, int]]:
    """Grid-index segment ranges split at every clamp window boundary.

    Segmenting at window edges makes segment membership equal clamp activation, so
    a ``set``/``shift`` clamp whose window opens mid-rollout is pinned exactly at the
    segment start (via the simulator's initial condition) rather than only its slope.
    """
    grid = [float(x) for x in time_grid]
    start, end = grid[0], grid[-1]
    horizon = end - start
    boundary_days = {start, end}
    for clamp in clamps:
        boundary_days.add(start + clamp.from_day)
        if clamp.to_day is not None:
            boundary_days.add(start + min(clamp.to_day, horizon))
    indices = sorted({_nearest_grid_index(time_grid, d) for d in boundary_days})
    return [(indices[k], indices[k + 1]) for k in range(len(indices) - 1)]


def _segment_overrides(
    clamps: list[ClampSpec],
    start_state: Array,
    grid_start: float,
    grid_end: float,
    seg_start_day: float,
) -> tuple[VariableOverride, ...]:
    """VariableOverrides for clamps active over a segment beginning at ``seg_start_day``
    (days relative to the rollout start). Value functions use absolute integration time."""
    horizon = grid_end - grid_start
    eps = 1e-9
    overrides: list[VariableOverride] = []
    for clamp in clamps:
        upper = float("inf") if clamp.to_day is None else clamp.to_day
        if not (clamp.from_day - eps <= seg_start_day < upper - eps):
            continue
        if clamp.mode == "set":
            value_fn = constant_value(jnp.asarray(clamp.value, dtype=start_state.dtype))
        elif clamp.mode == "shift":
            value_fn = constant_value(
                start_state[clamp.index] + jnp.asarray(clamp.amount, dtype=start_state.dtype)
            )
        elif clamp.mode == "ramp":
            eff_to = clamp.to_day if clamp.to_day is not None else horizon
            value_fn = linear_ramp(
                t_start=jnp.asarray(grid_start + clamp.from_day),
                t_end=jnp.asarray(grid_start + eff_to),
                value_start=jnp.asarray(clamp.value_start, dtype=start_state.dtype),
                value_end=jnp.asarray(clamp.value_end, dtype=start_state.dtype),
            )
        else:  # trajectory
            eff_to = clamp.to_day if clamp.to_day is not None else horizon
            n = len(clamp.values or ())
            times = jnp.linspace(grid_start + clamp.from_day, grid_start + eff_to, num=n)
            value_fn = precomputed_value(times, jnp.asarray(clamp.values, dtype=start_state.dtype))
        overrides.append(VariableOverride(index=clamp.index, value_fn=value_fn))
    return tuple(overrides)


def vmap_simulate_clamps_from_state(
    vector_field: VectorField,
    param_samples: list[tuple[dict[str, Array], ...]],
    initial_states: Array | None,
    clamps: list[ClampSpec],
    *,
    time_grid: Array,
) -> tuple[Array, Array, Array]:
    """Vmapped baseline / clamped / effect trajectories under a composed clamp list.

    The clamped path is simulated in segments split at clamp window boundaries
    (see :func:`build_segment_bounds`); each segment re-applies the active clamps'
    initial conditions, so windowed and staggered-onset clamps are exact at grid
    resolution. The baseline (no-clamp) path is one rollout; effect is their
    difference. Start = baseline steady state (``initial_states=None``) or an
    abducted state per draw.
    """
    import jax

    n_latent = vector_field.n_latent
    if not param_samples:
        empty = jnp.zeros((0, time_grid.shape[0], n_latent))
        return empty, empty, empty

    grid_start = float(time_grid[0])
    grid_end = float(time_grid[-1])
    segments = [
        (i0, i1, float(time_grid[i0]) - grid_start)
        for (i0, i1) in build_segment_bounds(time_grid, clamps)
    ]
    stacked = _stack_dynamics_params(param_samples)

    n_segments = len(segments)

    def clamped_path(params: tuple[dict[str, Array], ...], y0: Array) -> Array:
        segment_paths: list[Array] = []
        state = y0
        for seg_idx, (i0, i1, seg_start_day) in enumerate(segments):
            seg_grid = time_grid[i0 : i1 + 1]
            intervention = Intervention(
                overrides=_segment_overrides(clamps, y0, grid_start, grid_end, seg_start_day)
            )
            seg_ys = simulate(vector_field, params, intervention, state, seg_grid)
            # Carry the integrated boundary state forward, but emit the *next* segment's
            # pinned boundary point so a window opening mid-rollout shows its jump exactly.
            segment_paths.append(seg_ys[:-1] if seg_idx < n_segments - 1 else seg_ys)
            state = seg_ys[-1]
        return jnp.concatenate(segment_paths, axis=0)

    def per_draw(params: tuple[dict[str, Array], ...], y0: Array) -> tuple[Array, Array, Array]:
        baseline_path = simulate(vector_field, params, Intervention.none(), y0, time_grid)
        action_path = clamped_path(params, y0)
        return baseline_path, action_path, action_path - baseline_path

    if initial_states is None:

        def per_draw_steady(params):
            y0 = compute_steady_state(vector_field, params, Intervention.none())
            return per_draw(params, y0)

        return jax.vmap(per_draw_steady)(stacked)

    return jax.vmap(per_draw)(stacked, initial_states)
