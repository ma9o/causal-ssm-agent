"""Stage-6 entry point: rank treatments by intervention effect.

Single trajectory path (Diffrax) for both steady-state contrasts and
temporal trajectories. The vector field is a ``CompositeVectorField``
with a single ``DenseLinear`` component for the existing Stage 5b
dense-posterior shape; the same code path will work for explicit
primitive composition once Stage 4 emits non-linear edges.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, vmap

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
    """Factory: single-component ``DenseLinear`` vector field — TEST UTILITY ONLY.

    Production Stage 6 uses the two-component
    (``DenseLinear`` + ``Intercept``) shape via
    :func:`_two_component_linear_vector_field` and the vmap-ified composite
    consumers. This single-component factory remains as a test utility
    for code paths that build params as ``({"drift": A, "cint": c},)``
    (the legacy dense posterior shape). New code should use the
    two-component shape to stay aligned with the auto-built ``drift_spec``.

    The returned field has one ``DenseLinear`` component; callers pass
    ``params=({"drift": A, "cint": c},)`` via :func:`_linear_params`.

    Note: this is intentionally *not* the same shape as the two-component
    ``drift_spec`` that :meth:`SSMSpec.__post_init__` auto-builds. The
    two coexist by design — this factory matches the legacy dense
    posterior shape consumed by ``compute_interventions`` and
    Stage 6 ``tool_server`` dispatch, where ``samples["drift"]`` and
    ``samples["cint"]`` come back as separate stacked arrays and are
    vmapped together as a single dict.
    """
    return CompositeVectorField(n_latent=n_latent, components=(DenseLinear(),))


def _linear_params(drift: Array, cint: Array) -> tuple[dict[str, Array]]:
    return ({"drift": drift, "cint": cint},)


def _steady_state_treatment_effect_canonical(
    vector_field: CompositeVectorField,
    params: tuple[dict[str, Array], ...],
    treat_idx: int,
    outcome_idx: int,
    shift_size: float,
) -> Array:
    """Equilibrium contrast: ``effect = (do(treat = baseline+shift) - baseline)[outcome]``.

    Canonical implementation: takes a per-component ``params`` tuple
    directly. The dense-linear path passes ``({"drift": A, "cint": c},)``;
    composite paths pass the per-component tuple they already have.
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


def _linear_samples_to_composite_params(
    drift_draws: Array, cint_draws: Array
) -> list[tuple[dict[str, Array], ...]]:
    """Adapt linear-path posterior samples to the canonical per-draw
    composite param shape (``({"drift": A}, {"cint": c})``) — matching
    the auto-built two-component ``drift_spec`` that ``SSMSpec.__post_init__``
    produces for linear models."""
    return [({"drift": d}, {"cint": c}) for d, c in zip(drift_draws, cint_draws, strict=False)]


def _two_component_linear_vector_field(n_latent: int) -> CompositeVectorField:
    """The two-component (``DenseLinear`` + ``Intercept``) vector field
    matching the auto-built linear ``drift_spec``."""
    from nof1_causal_lab.models.ssm.dynamics.edges import Intercept as _Intercept

    return CompositeVectorField(n_latent=n_latent, components=(DenseLinear(), _Intercept()))


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

    Thin adapter over :func:`compute_interventions_composite`. Converts
    linear posterior samples (stacked ``(drift, cint)`` arrays) into the
    canonical per-draw composite param shape and delegates. The composite
    path uses ``jax.vmap`` internally so the linear case preserves
    vectorisation.
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

    n_latent = int(drift_draws.shape[-1])
    cint_draws = samples.get("cint")
    if cint_draws is None:
        cint_draws = jnp.zeros((drift_draws.shape[0], n_latent))

    lambda_draws = samples.get("lambda")
    lambda_mean: jnp.ndarray | None = None
    if lambda_draws is not None:
        lambda_mean = jnp.mean(lambda_draws, axis=0) if lambda_draws.ndim == 3 else lambda_draws

    return compute_interventions_composite(
        param_samples=_linear_samples_to_composite_params(drift_draws, cint_draws),
        vector_field=_two_component_linear_vector_field(n_latent),
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
    """Composite-spec counterpart of :func:`compute_interventions`.

    Consumes a posterior parameter-tuple list (the shape
    ``composite_aux_kalman.fit_*`` returns as ``param_samples``) plus the
    compiled vector field, and returns Stage-6-style intervention dicts.

    This closes the integration gap where Stage 6 was hardcoded to the
    linear ``(drift, cint)`` posterior shape. The output schema matches
    :func:`compute_interventions` so downstream UI / summarisation code
    needs no special case.
    """
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
    """Composite analogue of :func:`vmap_steady_state_effect`.

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
    """Composite analogue of :func:`vmap_simulate_action_from_state`.

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


def vmap_steady_state_effect(
    vector_field: CompositeVectorField,
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
        params = _linear_params(d, c)
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
    vector_field: CompositeVectorField,
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
        params = _linear_params(d, c)
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
