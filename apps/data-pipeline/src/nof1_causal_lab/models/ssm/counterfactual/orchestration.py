"""Stage-6 entry point: rank treatments by intervention effect."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    DenseLinear,
    Intervention,
    VariableOverride,
    compute_steady_state,
    constant_value,
    simulate,
    simulate_pair,
)

from .estimands import (
    build_time_grid,
    summarize_temporal_effect,
)

logger = get_prefect_logger(__name__)


def linear_vector_field(n_latent: int) -> CompositeVectorField:
    """Factory for a single-component dense-linear vector field."""
    return CompositeVectorField(n_latent=n_latent, components=(DenseLinear(),))


def _steady_state_treatment_effect_canonical(
    vector_field: CompositeVectorField,
    params: tuple[dict[str, Array], ...],
    treat_idx: int,
    outcome_idx: int,
    shift_size: float,
) -> Array:
    """Equilibrium contrast: ``effect = (do(treat = baseline+shift) - baseline)[outcome]``.

    Canonical implementation: takes a per-component ``params`` tuple
    directly. A dense affine component passes ``({"drift": A, "cint": c},)``;
    other composites pass the per-component tuple they already have.
    """
    baseline = compute_steady_state(vector_field, params, Intervention.none())
    do_value = baseline[treat_idx] + shift_size
    intervention = Intervention(
        overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
    )
    intervened = compute_steady_state(
        vector_field, params, intervention, initial_guess=baseline
    )
    return intervened[outcome_idx] - baseline[outcome_idx]


def _temporal_treatment_effect_canonical(
    vector_field: CompositeVectorField,
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
    param_samples: list[tuple[dict[str, Array], ...]],
    vector_field: CompositeVectorField,
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_spec: dict | None = None,
    manifest_names: list[str] | None = None,
    times: jnp.ndarray | None = None,
    shift_size: float = 1.0,
    lambda_mean: Array | None = None,
) -> list[dict[str, Any]]:
    """Compute intervention effects for all treatments from posterior samples.

    Thin public wrapper over :func:`compute_interventions_composite`.
    """
    return compute_interventions_composite(
        param_samples=param_samples,
        vector_field=vector_field,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        causal_spec=causal_spec,
        manifest_names=manifest_names,
        times=times,
        shift_size=shift_size,
        lambda_mean=lambda_mean,
    )


def compute_interventions_composite(
    param_samples: list[tuple[dict[str, Array], ...]],
    vector_field: CompositeVectorField,
    treatments: list[str],
    outcome: str,
    latent_names: list[str],
    causal_spec: dict | None = None,
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
        logger.warning("No posterior parameter samples for composite intervention")
        return [_skeleton(t) for t in treatments]

    time_grid = _build_horizon_grid(causal_spec, times)

    results: list[dict[str, Any]] = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            logger.warning("'%s' not in latent model — skipping", treatment_name)
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
                    "Composite simulation failed for '%s'; temporal effects unavailable",
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


def _stack_composite_params(
    param_samples: list[tuple[dict[str, Array], ...]],
) -> tuple[dict[str, Array], ...]:
    """Convert a list of per-draw composite parameter tuples into a single
    batched pytree where every leaf has a leading ``(N, ...)`` axis.

    The composite per-draw shape ``tuple[dict[str, Array], ...]`` has
    heterogeneous shapes across components, but each leaf has the same
    shape across draws — so ``jax.tree.map(stack, *...)`` produces a
    well-formed batched pytree consumable by ``jax.vmap``.
    """
    import jax

    return jax.tree.map(lambda *xs: jnp.stack(xs), *param_samples)


def vmap_steady_state_effect_composite(
    vector_field: CompositeVectorField,
    param_samples: list[tuple[dict[str, Array], ...]],
    treat_idx: int,
    outcome_idx: int,
    *,
    mode: str,
    value: float | None = None,
    amount: float | None = None,
) -> Array:
    """Vmapped steady-state set/shift effect for vector-field params.

    Uses ``jax.vmap`` over the stacked per-draw pytree (see
    :func:`_stack_composite_params`). Heterogeneity across composite
    components is irrelevant for vmap — each leaf has consistent shape
    across draws, which is all vmap requires.

    The ``Intervention`` is constructed *inside* the vmapped function so
    the closure over the traced ``do_value`` participates in the vmap
    trace correctly; constructing it outside would close over the whole
    batched array and produce wrong semantics.
    """
    import jax

    from .estimands import resolve_action_value

    if not param_samples:
        return jnp.zeros((0,))

    stacked = _stack_composite_params(param_samples)

    def per_draw(params: tuple[dict[str, Array], ...]) -> Array:
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

    return jax.vmap(per_draw)(stacked)


def vmap_simulate_action_from_state_composite(
    vector_field: CompositeVectorField,
    param_samples: list[tuple[dict[str, Array], ...]],
    initial_states: Array | None,
    treat_idx: int,
    *,
    mode: str,
    value: float | None = None,
    amount: float | None = None,
    time_grid: Array,
) -> tuple[Array, Array, Array]:
    """Vmapped baseline / action / effect trajectories for vector-field params.

    ``jax.vmap`` over the stacked per-draw pytree (and optionally
    ``initial_states``). Same vmap argument as
    :func:`vmap_steady_state_effect_composite` — heterogeneity is per
    component, not per draw.
    """
    import jax

    from .estimands import resolve_action_value

    n_latent = vector_field.n_latent
    if not param_samples:
        empty = jnp.zeros((0, time_grid.shape[0], n_latent))
        return empty, empty, empty

    stacked = _stack_composite_params(param_samples)

    def _per_draw_paths(
        params: tuple[dict[str, Array], ...], y0: Array
    ) -> tuple[Array, Array, Array]:
        do_value = resolve_action_value(
            y0[treat_idx], mode=mode, value=value, amount=amount
        )
        action_intervention = Intervention(
            overrides=(VariableOverride(index=treat_idx, value_fn=constant_value(do_value)),)
        )
        baseline_path = simulate(
            vector_field, params, Intervention.none(), y0, time_grid
        )
        action_path = simulate(
            vector_field, params, action_intervention, y0, time_grid
        )
        return baseline_path, action_path, action_path - baseline_path

    if initial_states is None:

        def per_draw_steady(params):
            y0 = compute_steady_state(vector_field, params, Intervention.none())
            return _per_draw_paths(params, y0)

        return jax.vmap(per_draw_steady)(stacked)

    return jax.vmap(_per_draw_paths)(stacked, initial_states)
