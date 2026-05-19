"""Stage-6 entry point: rank treatments by intervention effect.

Single trajectory path (Diffrax) for both steady-state contrasts and
temporal trajectories. Steady-state effects come from
``compute_steady_state`` on the baseline and on the intervened system;
temporal effects come from ``simulate_pair`` over a horizon grid.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, vmap

from nof1_causal_lab.flows import get_prefect_logger

from .estimands import (
    build_time_grid,
    summarize_temporal_effect,
)
from .intervention import Intervention, VariableOverride, constant_value
from .simulator import simulate, simulate_pair
from .steady_state import compute_steady_state
from .vector_field import LinearVectorField, VectorField

logger = get_prefect_logger(__name__)


def _steady_state_treatment_effect(
    vector_field: VectorField,
    drift: Array,
    cint: Array,
    treat_idx: int,
    outcome_idx: int,
    shift_size: float,
) -> Array:
    """Equilibrium contrast: ``effect = (do(treat = baseline+shift) - baseline)[outcome]``."""
    params = {"drift": drift, "cint": cint}
    baseline = compute_steady_state(vector_field, params, Intervention.none())
    do_value = baseline[treat_idx] + shift_size
    intervention = Intervention(
        overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
    )
    intervened = compute_steady_state(
        vector_field, params, intervention, initial_guess=baseline
    )
    return intervened[outcome_idx] - baseline[outcome_idx]


def _temporal_treatment_effect(
    vector_field: VectorField,
    drift: Array,
    cint: Array,
    treat_idx: int,
    shift_size: float,
    time_grid: Array,
) -> Array:
    """Per-time effect trajectory: ``(action - baseline)`` over ``time_grid``."""
    params = {"drift": drift, "cint": cint}
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
    samples: dict[str, jnp.ndarray],
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_spec: dict | None = None,
    manifest_names: list[str] | None = None,
    times: jnp.ndarray | None = None,
    shift_size: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute intervention effects for all treatments from posterior samples.

    Pure function over posterior samples. Returns ranked result dicts
    matching the existing Stage-6 contract.
    """
    name_to_idx = {name: i for i, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)

    def _skeleton(treatment_name: str) -> dict[str, Any]:
        return {"treatment": treatment_name}

    if outcome_idx is None:
        logger.warning("Outcome '%s' not found in latent names %s", outcome, latent_names)
        return [_skeleton(t) for t in treatments]

    drift_draws = samples.get("drift")
    if drift_draws is None:
        logger.warning("No 'drift' in posterior samples")
        return [_skeleton(t) for t in treatments]

    n_latent = drift_draws.shape[-1]
    cint_draws = samples.get("cint")
    if cint_draws is None:
        cint_draws = jnp.zeros((drift_draws.shape[0], n_latent))

    vector_field = LinearVectorField(n_latent=n_latent)

    time_grid = _build_horizon_grid(causal_spec, times)

    lambda_draws = samples.get("lambda")
    lambda_mean: jnp.ndarray | None = None
    if lambda_draws is not None:
        lambda_mean = jnp.mean(lambda_draws, axis=0) if lambda_draws.ndim == 3 else lambda_draws

    results: list[dict[str, Any]] = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            logger.warning("'%s' not in latent model — skipping", treatment_name)
            results.append(_skeleton(treatment_name))
            continue

        effects = vmap(
            lambda d, c, ti=treat_idx, oi=outcome_idx: _steady_state_treatment_effect(
                vector_field, d, c, ti, oi, shift_size
            )
        )(drift_draws, cint_draws)

        mean_effect = float(jnp.mean(effects))

        entry: dict[str, Any] = {
            "treatment": treatment_name,
            "posterior_draws": effects.tolist(),
        }

        if time_grid is not None:
            try:
                trajectories = vmap(
                    lambda d, c, ti=treat_idx: _temporal_treatment_effect(
                        vector_field, d, c, ti, shift_size, time_grid
                    )
                )(drift_draws, cint_draws)
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
                    "Forward simulation failed for '%s'; temporal effects unavailable",
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
    causal_spec: dict | None,
    times: jnp.ndarray | None,
    horizon_days: float = 30.0,
) -> Array | None:
    """Derive a forward simulation grid from the measurement clock or
    median observation spacing. Returns ``None`` when no usable step size
    is available."""
    dt_days: float | None = None
    model_clock_str = (causal_spec or {}).get("measurement", {}).get("model_clock")
    if model_clock_str:
        from nof1_causal_lab.artifacts.duration import parse_duration_to_hours

        dt_days = parse_duration_to_hours(model_clock_str) / 24.0
    elif times is not None and len(times) > 1:
        diffs = jnp.diff(times)
        dt_days = float(jnp.median(diffs))

    if dt_days is None or dt_days <= 0:
        return None

    return build_time_grid(0.0, horizon_days, dt_days)


def vmap_steady_state_effect(
    vector_field: VectorField,
    drift_draws: Array,
    cint_draws: Array,
    treat_idx: int,
    outcome_idx: int,
    *,
    mode: str,
    value: float | None = None,
    amount: float | None = None,
) -> Array:
    """Vmapped steady-state set/shift effect for tool_server."""
    from .estimands import resolve_action_value

    def _per_draw(d: Array, c: Array) -> Array:
        params = {"drift": d, "cint": c}
        baseline = compute_steady_state(vector_field, params, Intervention.none())
        do_value = resolve_action_value(
            baseline[treat_idx], mode=mode, value=value, amount=amount
        )
        intervention = Intervention(
            overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
        )
        intervened = compute_steady_state(
            vector_field, params, intervention, initial_guess=baseline
        )
        return intervened[outcome_idx] - baseline[outcome_idx]

    return vmap(_per_draw)(drift_draws, cint_draws)


def vmap_simulate_action_from_state(
    vector_field: VectorField,
    drift_draws: Array,
    cint_draws: Array,
    initial_states: Array | None,
    treat_idx: int,
    *,
    mode: str,
    value: float | None = None,
    amount: float | None = None,
    time_grid: Array,
) -> tuple[Array, Array, Array]:
    """Vmapped baseline / action / effect trajectories.

    If ``initial_states`` is ``None`` (rung 2), the per-draw baseline steady
    state is used as the starting point. If provided (rung 3), each draw
    uses the same abducted state.
    """
    from .estimands import resolve_action_value

    def _per_draw(d: Array, c: Array, y0_seed: Array) -> tuple[Array, Array, Array]:
        params = {"drift": d, "cint": c}
        if initial_states is None:
            y0 = compute_steady_state(vector_field, params, Intervention.none())
        else:
            y0 = y0_seed
        do_value = resolve_action_value(
            y0[treat_idx], mode=mode, value=value, amount=amount
        )
        action_intervention = Intervention(
            overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
        )
        baseline_path = simulate(vector_field, params, Intervention.none(), y0, time_grid)
        action_path = simulate(vector_field, params, action_intervention, y0, time_grid)
        return baseline_path, action_path, action_path - baseline_path

    seed = (
        initial_states
        if initial_states is not None
        else jnp.zeros((drift_draws.shape[0], vector_field.n_latent))
    )
    return jax.vmap(_per_draw)(drift_draws, cint_draws, seed)
