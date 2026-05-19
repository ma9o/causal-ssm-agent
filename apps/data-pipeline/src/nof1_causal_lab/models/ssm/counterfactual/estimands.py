"""Estimand helpers: posterior summaries and temporal extracts.

Pure functions over trajectories and posterior draws. No JAX primitives
specific to the linear case; safe to use unchanged once the non-linear
vector fields land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array


def summarize_draws(draws: Array) -> dict[str, float]:
    """Posterior summary statistics for a 1-D array of effect draws."""
    return {
        "mean": float(jnp.mean(draws)),
        "median": float(jnp.median(draws)),
        "lower_95": float(jnp.quantile(draws, 0.025)),
        "upper_95": float(jnp.quantile(draws, 0.975)),
        "prob_positive": float(jnp.mean(draws > 0)),
    }


def summarize_temporal_effect(
    effect_trajectory: Array,
    time_grid: Array,
) -> dict[str, float]:
    """Effect at 1-day, 7-day, 30-day horizons plus peak and time-to-peak.

    ``effect_trajectory[k]`` is the effect at ``time_grid[k]`` measured in
    days from the simulation start. Linear interpolation between grid
    points keeps the summary stable to grid resolution.
    """
    t = jnp.asarray(time_grid, dtype=jnp.float32)
    traj = jnp.asarray(effect_trajectory, dtype=jnp.float32)
    t0 = t[0]
    days = t - t0

    def _at(target_day: float) -> float:
        return float(jnp.interp(jnp.asarray(target_day), days, traj))

    abs_traj = jnp.abs(traj)
    peak_idx = int(jnp.argmax(abs_traj))
    peak_effect = float(traj[peak_idx])
    time_to_peak_days = float(days[peak_idx])

    return {
        "effect_1d": _at(1.0),
        "effect_7d": _at(7.0),
        "effect_30d": _at(30.0),
        "peak_effect": peak_effect,
        "time_to_peak_days": time_to_peak_days,
    }


ActionMode = Literal["set", "shift"]


def resolve_action_value(
    baseline_value: Array,
    *,
    mode: ActionMode,
    value: float | None = None,
    amount: float | None = None,
) -> Array:
    """Map a stage-6 set/shift action onto an absolute latent-space value."""
    baseline = jnp.asarray(baseline_value)
    if mode == "set":
        if value is None:
            raise ValueError("mode='set' requires value")
        return jnp.asarray(value, dtype=baseline.dtype)
    if mode == "shift":
        if amount is None:
            raise ValueError("mode='shift' requires amount")
        return baseline + jnp.asarray(amount, dtype=baseline.dtype)
    raise ValueError(f"Unsupported action mode: {mode}")


def build_time_grid(t_start: float, t_end: float, dt: float) -> Array:
    """Inclusive uniform grid from ``t_start`` to ``t_end`` at spacing ``dt``."""
    n_steps = int(jnp.ceil((t_end - t_start) / dt)) + 1
    return jnp.linspace(t_start, t_end, num=n_steps)


def project_to_manifest(
    latent_effects: Array,
    lambda_mat: Array,
    outcome_latent_idx: int,
) -> Array:
    """Manifest-level effect: ``lambda[:, outcome] * effect_on_outcome_latent``.

    ``latent_effects`` is the per-latent effect (e.g., at steady state or a
    horizon endpoint). Only the outcome-latent loading column contributes
    because we project a scalar latent effect, not a full vector.
    """
    loadings = lambda_mat[:, outcome_latent_idx]
    return loadings * latent_effects[outcome_latent_idx]


def per_draw_effect_summary(
    effect_draws: Array,
) -> Callable[[], dict[str, float]]:
    """Compatibility shim — kept as a function so callers can lazily call
    on demand without pre-computing a dict at trace time."""
    return lambda: summarize_draws(effect_draws)
