"""Lightweight tool execution server for the refinement proxy.

Exposes pipeline tool schemas and execution over HTTP so the Next.js
refinement route can proxy LLM tool calls to the same Python validation
logic the pipeline uses.

Run alongside Prefect::

    cd apps/data-pipeline
    uv run uvicorn nof1_causal_lab.tool_server:app --port 8100
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.flows.run_store import load_parquet, load_pickle
from nof1_causal_lab.flows.stage_contracts import STAGE_TOOLS
from nof1_causal_lab.flows.stage_persistence import persist_web_patch
from nof1_causal_lab.flows.stages.stage1a.grounding import stage1a_grounding
from nof1_causal_lab.flows.stages.stage1b.grounding import stage1b_grounding
from nof1_causal_lab.flows.stages.stage4.tool_registry import (
    execute_public_search_literature as _execute_search_literature,
)
from nof1_causal_lab.flows.stages.stage4.tool_registry import (
    execute_public_submit_model_spec as _execute_submit_model_spec,
)
from nof1_causal_lab.flows.stages.stage4.tool_registry import (
    execute_public_submit_priors as _execute_submit_priors,
)
from nof1_causal_lab.flows.stages.stage4.tool_registry import (
    execute_public_validate_composite_spec as _execute_validate_composite_spec,
)
from nof1_causal_lab.models.ssm.builder import SSMModelBuilder, prepare_model_runtime
from nof1_causal_lab.models.ssm.counterfactual import (
    summarize_draws,
    vmap_simulate_action_from_state_composite,
    vmap_steady_state_effect_composite,
)
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    Intervention,
    compute_steady_state,
    infer_linearisation,
    posterior_dynamics_from_result,
)
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.causal_spec import (
    get_estimable_treatments,
    get_estimation_constructs,
    get_estimation_state_order,
    get_indicators,
    get_outcome_name,
)
from nof1_causal_lab.utils.data import runs_dir

logger = logging.getLogger(__name__)

app = FastAPI(title="Tool Server", docs_url="/api/tools/docs")

app.add_middleware(
    cast("Any", CORSMiddleware),
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _load_stage_result(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted stage result from storage."""
    path = storage.join(runs_dir(workspace_id), f"{stage_id}.json")
    if not storage.exists(path):
        raise HTTPException(404, f"Stage result not found: {path}")
    return storage.read_json(path)


def _load_optional_stage_result(workspace_id: str, stage_id: str) -> dict[str, Any]:
    try:
        return _load_stage_result(workspace_id, stage_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return {}
        raise


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _extract_observation_timestamps(observation_data: Any) -> list[datetime]:
    import polars as pl

    if observation_data is None or observation_data.is_empty():
        return []

    anchor = pl.col("anchor_time")
    if observation_data.schema.get("anchor_time") == pl.Utf8:
        anchor = anchor.str.to_datetime(strict=False, time_zone="UTC")

    values = (
        observation_data.select(anchor.alias("anchor_time"))
        .drop_nulls()
        .unique()
        .sort("anchor_time")
        .get_column("anchor_time")
        .to_list()
    )
    out: list[datetime] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            out.append(dt.astimezone(UTC))
    return out


def _stage6_time_config(
    causal_spec: dict[str, Any],
    times: Any,
    horizon_days: int,
) -> tuple[float, int]:
    model_clock = ((causal_spec or {}).get("measurement") or {}).get("model_clock")
    dt_days: float | None = None
    if model_clock:
        dt_days = parse_duration_to_hours(model_clock) / 24.0
    elif times is not None and len(times) > 1:
        dt_days = float(jnp.median(jnp.diff(times)))

    if dt_days is None or not bool(jnp.isfinite(dt_days)) or dt_days <= 0:
        dt_days = 1.0

    horizon_steps = max(1, int(float(jnp.ceil(horizon_days / dt_days))))
    return dt_days, horizon_steps


def _manifest_effects(
    samples: dict[str, Any],
    outcome_idx: int,
    effect_mean: float,
    manifest_names: list[str],
) -> dict[str, float] | None:
    lambda_draws = samples.get("lambda")
    if lambda_draws is None:
        return None
    lambda_mean = (
        jnp.mean(lambda_draws, axis=0) if getattr(lambda_draws, "ndim", 0) == 3 else lambda_draws
    )
    if getattr(lambda_mean, "ndim", 0) != 2:
        return None

    effects: dict[str, float] = {}
    for idx, loading in enumerate(lambda_mean[:, outcome_idx]):
        loading_value = float(loading)
        if abs(loading_value) <= 1e-9:
            continue
        name = manifest_names[idx] if idx < len(manifest_names) else f"manifest_{idx}"
        effects[name] = loading_value * effect_mean
    return effects or None


def _serialize_effect_trajectory(
    trajectory: jnp.ndarray, time_grid_days: jnp.ndarray
) -> list[dict[str, float]]:
    days = time_grid_days.tolist()
    values = trajectory.tolist()
    return [
        {"day": round(float(day), 3), "effect": float(value)}
        for day, value in zip(days, values, strict=False)
    ]


def _serialize_node_trajectories(
    state_paths: jnp.ndarray,
    latent_names: list[str],
) -> dict[str, list[float]]:
    mean_paths = jnp.mean(state_paths, axis=0)
    return {
        name: [float(value) for value in mean_paths[:, idx].tolist()]
        for idx, name in enumerate(latent_names)
    }


def _serialize_latent_state(state: jnp.ndarray, latent_names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(latent_names, state.tolist(), strict=False)}


def _fitted_latent_paths_from_result(result: Any) -> jnp.ndarray | None:
    """Return retained per-draw fitted latent paths as ``(draw, time, latent)``."""
    paths = result.get_latent_paths() if hasattr(result, "get_latent_paths") else None
    if paths is None:
        diagnostics = getattr(result, "diagnostics", {}) or {}
        paths = diagnostics.get("latent_paths")
    if paths is None:
        return None

    latent_paths = jnp.asarray(paths)
    if latent_paths.ndim == 4:
        latent_paths = latent_paths.reshape(
            (
                latent_paths.shape[0] * latent_paths.shape[1],
                latent_paths.shape[2],
                latent_paths.shape[3],
            )
        )
    if latent_paths.ndim != 3:
        raise HTTPException(
            400,
            "Persisted fitted latent paths must have shape "
            "(draw, time, latent) or (chain, draw, time, latent).",
        )
    return latent_paths


def _resolve_counterfactual_start(
    ctx: dict[str, Any],
    start: dict[str, Any],
    *,
    n_timepoints: int,
) -> tuple[int, dict[str, Any]]:
    if n_timepoints <= 0:
        raise HTTPException(400, "Persisted fitted latent paths contain no timepoints.")

    raw_time_index = start.get("time_index")
    raw_time = start.get("time")
    timestamps = list(ctx.get("_observation_timestamps") or [])

    if raw_time_index is not None:
        time_index = int(raw_time_index)
    elif raw_time:
        if not timestamps:
            raise HTTPException(
                400, "start.time requires observed timestamps in the fitted workspace."
            )
        requested = _parse_iso_datetime(str(raw_time))
        matches = [
            idx for idx, timestamp in enumerate(timestamps[:n_timepoints]) if timestamp == requested
        ]
        if not matches:
            raise HTTPException(
                400, "start.time must exactly match a retained fitted-state timestamp."
            )
        time_index = matches[0]
    else:
        time_index = n_timepoints - 1

    if time_index < 0 or time_index >= n_timepoints:
        raise HTTPException(
            400,
            f"start.time_index must be between 0 and {n_timepoints - 1}; got {time_index}.",
        )

    time = timestamps[time_index].isoformat() if time_index < len(timestamps) else None
    return (
        time_index,
        {
            "time_index": time_index,
            "time": time,
            "state_source": "fitted_latent_paths",
        },
    )


def _build_stage6_context(workspace_id: str) -> dict[str, Any]:
    stage1b = _load_stage_result(workspace_id, "stage-1b")
    stage4 = _load_optional_stage_result(workspace_id, "stage-4")
    stage5b = _load_stage_result(workspace_id, "stage-5b")
    stage6 = _load_optional_stage_result(workspace_id, "stage-6")

    from nof1_causal_lab.flows.run_store import (
        STAGE2_MODEL_PARQUET_FILENAMES,
        STAGE5B_PICKLE_FILENAMES,
        find_run_artifact,
    )

    data_for_model_path = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    fitted_result_path = find_run_artifact(workspace_id, STAGE5B_PICKLE_FILENAMES)
    fitted_artifact = load_pickle(fitted_result_path)
    data_for_model = load_parquet(data_for_model_path)
    persisted_builder = getattr(fitted_artifact, "builder", None)
    fitted_spec = getattr(persisted_builder, "spec", None)
    if fitted_spec is None:
        raise HTTPException(
            500,
            "Stage 5b fitted artifact is missing the compiled SSMSpec required for stage-6 tools.",
        )
    runtime = prepare_model_runtime(
        data_for_model=data_for_model,
        builder=SSMModelBuilder(ssm_spec=fitted_spec),
    )
    fitted_artifact.builder = runtime.builder
    fitted_artifact.observation_support = runtime.observation_support

    causal_spec = stage1b.get("causal_spec", {})
    non_identifiable = (causal_spec.get("identifiability") or {}).get(
        "non_identifiable_treatments"
    ) or {}
    outcome_name = get_outcome_name(causal_spec)
    treatments = get_estimable_treatments(causal_spec)

    return {
        "_workspace_id": workspace_id,
        "stage-1b": stage1b,
        "stage-4": stage4,
        "stage-5b": stage5b,
        "stage-6": stage6,
        "_fitted_artifact": fitted_artifact,
        "_prepared_runtime": runtime,
        "_observation_timestamps": _extract_observation_timestamps(runtime.observation_data),
        "_outcome_name": outcome_name,
        "_identifiable_treatments": [t for t in treatments if t not in non_identifiable],
    }


@dataclass(frozen=True)
class Stage6SimulationSetup:
    fitted_artifact: Any
    runtime: Any
    samples: dict[str, Any]
    causal_spec: dict[str, Any]
    query: dict[str, Any]
    action: dict[str, Any]
    treatment: str
    outcome: str
    spec: Any
    latent_names: list[str]
    manifest_names: list[str]
    treat_idx: int
    outcome_idx: int
    dt_days: float
    horizon_steps: int
    time_grid: jnp.ndarray
    vector_field: CompositeVectorField
    # ``param_samples`` is the canonical per-draw composite-shape
    # parameter list rebuilt from ``SSMSpec`` and posterior sample sites.
    param_samples: list[tuple[dict[str, Any], ...]] | None = None

    @property
    def is_composite(self) -> bool:
        """True when dynamics require trajectory-local linearisation."""
        return infer_linearisation(self.vector_field) == "trajectory"


@dataclass(frozen=True)
class Stage6EffectOutputs:
    summary: dict[str, float]
    effect_trajectory: list[dict[str, float]] | None
    visualization: dict[str, Any] | None
    manifest_effects: dict[str, float] | None


def _tool_error_result(
    message: str,
    *,
    identifiable_treatments: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"error": message}
    if identifiable_treatments is not None:
        result["identifiable_treatments"] = identifiable_treatments
    return {"result": result}


def _prepare_stage6_simulation(
    ctx: dict[str, Any],
    args: dict[str, Any],
) -> tuple[Stage6SimulationSetup | None, dict[str, Any] | None]:
    fitted_artifact = ctx["_fitted_artifact"]
    runtime = ctx["_prepared_runtime"]
    if fitted_artifact.result is None or fitted_artifact.builder is None:
        return None, _tool_error_result("Fitted model artifact is unavailable for simulation.")
    samples = fitted_artifact.result.get_samples() or {}

    action = dict(args.get("action") or {})
    treatment = str(action.get("variable", ""))
    if treatment not in ctx["_identifiable_treatments"]:
        return None, _tool_error_result(
            f"Treatment '{treatment}' is not an identifiable stage-6 intervention target.",
            identifiable_treatments=ctx["_identifiable_treatments"],
        )

    causal_spec = ctx["stage-1b"].get("causal_spec", {})
    outcome = str(args.get("outcome") or ctx.get("_outcome_name") or "")
    spec = fitted_artifact.builder.spec
    latent_names = list(spec.latent_names or [])
    manifest_names = list(spec.manifest_names or [])
    name_to_idx = {name: idx for idx, name in enumerate(latent_names)}
    treat_idx = name_to_idx.get(treatment)
    outcome_idx = name_to_idx.get(outcome)
    if treat_idx is None or outcome_idx is None:
        return None, _tool_error_result("Treatment or outcome not present in fitted latent model.")

    query = dict(args.get("query") or {})
    dt_days, horizon_steps = _stage6_time_config(
        causal_spec,
        runtime.times,
        int(query.get("horizon_days") or 30),
    )
    time_grid = jnp.linspace(0.0, dt_days * horizon_steps, horizon_steps + 1)

    posterior_dynamics = posterior_dynamics_from_result(spec, fitted_artifact.result)
    vector_field = posterior_dynamics.vector_field
    param_samples = posterior_dynamics.param_samples
    if not param_samples:
        return None, _tool_error_result("Posterior dynamics samples are unavailable.")

    return (
        Stage6SimulationSetup(
            fitted_artifact=fitted_artifact,
            runtime=runtime,
            samples=samples,
            causal_spec=causal_spec,
            query=query,
            action=action,
            treatment=treatment,
            outcome=outcome,
            spec=spec,
            latent_names=latent_names,
            manifest_names=manifest_names,
            treat_idx=treat_idx,
            outcome_idx=outcome_idx,
            dt_days=dt_days,
            horizon_steps=horizon_steps,
            time_grid=time_grid,
            vector_field=vector_field,
            param_samples=param_samples,
        ),
        None,
    )


def _build_visualization_payload(
    latent_names: list[str],
    *,
    reference_node_paths: jnp.ndarray | None = None,
    action_node_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    start_state: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    reference_node_trajectories = (
        _serialize_node_trajectories(reference_node_paths[:, 1:], latent_names)
        if reference_node_paths is not None
        else None
    )
    action_node_trajectories = (
        _serialize_node_trajectories(action_node_paths[:, 1:], latent_names)
        if action_node_paths is not None
        else None
    )
    node_effect_trajectories = (
        _serialize_node_trajectories(node_effect_paths[:, 1:], latent_names)
        if node_effect_paths is not None
        else None
    )
    if (
        reference_node_trajectories is None
        and action_node_trajectories is None
        and node_effect_trajectories is None
        and start_state is None
    ):
        return None
    return {
        "reference_node_trajectories": reference_node_trajectories,
        "action_node_trajectories": action_node_trajectories,
        "node_effect_trajectories": node_effect_trajectories,
        "start_state": start_state,
    }


def _build_effect_outputs(
    setup: Stage6SimulationSetup,
    *,
    effect_draws: jnp.ndarray | None = None,
    effect_paths: jnp.ndarray | None = None,
    reference_node_paths: jnp.ndarray | None = None,
    action_node_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    start_state: dict[str, float] | None = None,
) -> Stage6EffectOutputs:
    if effect_paths is not None:
        effect_draws = effect_paths[:, -1]
        mean_effect_trajectory = jnp.mean(effect_paths, axis=0)
        effect_trajectory = _serialize_effect_trajectory(
            mean_effect_trajectory[1:], setup.time_grid[1:]
        )
    else:
        effect_trajectory = None

    if effect_draws is None:
        raise ValueError("Either effect_draws or effect_paths must be provided.")

    summary = summarize_draws(effect_draws)
    manifest_effects = None
    if setup.query.get("projection", "latent") in {"manifest", "both"}:
        manifest_effects = _manifest_effects(
            setup.samples,
            setup.outcome_idx,
            summary["mean"],
            setup.manifest_names,
        )

    return Stage6EffectOutputs(
        summary=summary,
        effect_trajectory=effect_trajectory,
        visualization=_build_visualization_payload(
            setup.latent_names,
            reference_node_paths=reference_node_paths,
            action_node_paths=action_node_paths,
            node_effect_paths=node_effect_paths,
            start_state=start_state,
        ),
        manifest_effects=manifest_effects,
    )


def _collect_stage6_warnings(
    ctx: dict[str, Any],
    *,
    treatment: str | None = None,
    include_diagnostic_warnings: bool = False,
    extra_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if include_diagnostic_warnings and treatment is not None:
        stage5b = ctx.get("stage-5b", {})
        for item in stage5b.get("power_scaling", []):
            if item.get("diagnosis") == "prior_dominated":
                param = item.get("parameter", "")
                if treatment in param or param.startswith("drift_offdiag"):
                    warnings.append(
                        f"Effect may be prior-driven: parameter {param} "
                        f"is prior-dominated per power-scaling diagnostic"
                    )
        for item in stage5b.get("ppc", {}).get("per_variable_warnings", []) or []:
            message = item.get("message")
            if message:
                warnings.append(str(message))

    for warning in extra_warnings or []:
        if warning:
            warnings.append(str(warning))
    return warnings


# ---------------------------------------------------------------------------
# Tool implementations — map (stage_id, tool_name) → execute(context, input)
# ---------------------------------------------------------------------------


def _run_compute(
    args: dict[str, Any],
    param_name: str,
    compute_fn: Any,
) -> dict[str, Any]:
    """Parse JSON arg, run compute function, return result + stage_output."""
    raw = args.get(param_name, "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"result": f"JSON parse error: {e}", "stage_output": None}

    stage_output, feedback = compute_fn(data)
    return {"result": feedback, "stage_output": stage_output}


def _execute_validate_latent_model(_ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return _run_compute(args, "structure_json", stage1a_grounding)


def _execute_validate_measurement_model(
    ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    stage1a = ctx.get("stage-1a", {})
    latent_model = stage1a["latent_model"]
    return _run_compute(
        args,
        "measurement_json",
        lambda data: stage1b_grounding(data, latent_model),
    )


def _execute_validate_extractions(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, str]:
    from nof1_causal_lab.utils.llm import _validate_json_and_format
    from nof1_causal_lab.workers.schemas import validate_worker_output

    schema = ctx.get("_extraction_schema", {})
    result = _validate_json_and_format(
        args["output_json"],
        lambda data: validate_worker_output(data, schema),
    )
    return {"result": result}


def _build_model_info_payload(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    sections = list(args.get("sections") or ["overview", "variables", "capabilities"])
    focused = {str(name) for name in (args.get("names") or [])}
    stage1b = ctx["stage-1b"]
    stage5b = ctx.get("stage-5b", {})
    stage6 = ctx.get("stage-6", {})
    runtime = ctx["_prepared_runtime"]
    fitted_artifact = ctx["_fitted_artifact"]
    causal_spec = stage1b.get("causal_spec", {})
    measurement = causal_spec.get("measurement") or {}
    retained_state_names = set(get_estimation_state_order(causal_spec))
    constructs = get_estimation_constructs(causal_spec)
    indicators = [
        indicator
        for indicator in get_indicators(causal_spec)
        if indicator.get("construct_name") in retained_state_names
    ]

    if focused:
        constructs = [item for item in constructs if item.get("name") in focused]
        indicators = [
            item
            for item in indicators
            if item.get("name") in focused or item.get("construct_name") in focused
        ]

    payload: dict[str, Any] = {}
    if "overview" in sections:
        payload["overview"] = {
            "outcome": ctx.get("_outcome_name"),
            "treatments": ctx["_identifiable_treatments"],
            "n_latent": len(getattr(fitted_artifact.builder.spec, "latent_names", []) or []),
            "n_manifest": len(runtime.manifest_names),
            "inference_method": (stage5b.get("inference_metadata") or {}).get("method"),
            "observed_time_range": {
                "start": ctx["_observation_timestamps"][0].isoformat()
                if ctx["_observation_timestamps"]
                else None,
                "end": ctx["_observation_timestamps"][-1].isoformat()
                if ctx["_observation_timestamps"]
                else None,
            },
        }
    if "variables" in sections:
        payload["variables"] = {
            "constructs": [
                {
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "role": item.get("role"),
                    "temporal_status": item.get("temporal_status"),
                    "is_outcome": item.get("is_outcome"),
                }
                for item in constructs
            ],
            "indicators": [
                {
                    "name": item.get("name"),
                    "construct_name": item.get("construct_name"),
                    "measurement_dtype": item.get("measurement_dtype"),
                    "support_kind": item.get("support_kind"),
                    "summary_operator": item.get("summary_operator"),
                    "observation_window": item.get("observation_window"),
                }
                for item in indicators
            ],
        }
    if "measurement" in sections:
        payload["measurement"] = {
            "model_clock": measurement.get("model_clock"),
            "manifest_names": runtime.manifest_names,
        }
    if "identifiability" in sections:
        payload["identifiability"] = {
            "identifiable_treatments": ctx["_identifiable_treatments"],
            "non_identifiable_treatments": (
                (causal_spec.get("identifiability") or {}).get("non_identifiable_treatments") or {}
            ),
        }
    if "diagnostics" in sections:
        power_scaling = list(stage5b.get("power_scaling", []) or [])
        payload["diagnostics"] = {
            "ppc_warning_count": len(
                (stage5b.get("ppc") or {}).get("per_variable_warnings", []) or []
            ),
            "power_scaling_issues": [
                entry
                for entry in power_scaling
                if entry.get("diagnosis") in {"prior_dominated", "prior_data_conflict"}
            ],
        }
    if "baseline_effects" in sections:
        baseline = list(stage6.get("intervention_results", []) or [])
        if focused:
            baseline = [entry for entry in baseline if entry.get("treatment") in focused]

        def _draws_summary(draws):
            if not draws:
                return None, None
            return sum(draws) / len(draws), sum(1 for d in draws if d > 0) / len(draws)

        payload["baseline_effects"] = [
            {
                "treatment": entry.get("treatment"),
                "effect_size": _draws_summary(entry.get("posterior_draws"))[0],
                "prob_positive": _draws_summary(entry.get("posterior_draws"))[1],
            }
            for entry in baseline[:10]
        ]
    if "capabilities" in sections:
        payload["capabilities"] = {
            "intervention": {
                "rung": 2,
                "supported_treatments": ctx["_identifiable_treatments"],
                "actions": ["set", "shift"],
                "estimands": ["steady_state", "trajectory"],
            },
            "counterfactual": {
                "rung": 3,
                "start_state": "retained_fitted_latent_paths",
                "estimands": ["end_state", "trajectory"],
            },
        }
    return payload


def _execute_get_model_info(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return {"result": _build_model_info_payload(ctx, args)}


def _execute_simulate_intervention(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    setup, error = _prepare_stage6_simulation(ctx, args)
    if error is not None:
        return error
    assert setup is not None

    # ``setup.param_samples`` and ``setup.vector_field`` are reconstructed
    # from the fitted ``SSMSpec`` and canonical posterior sample sites.
    assert setup.param_samples is not None
    _stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *setup.param_samples)
    baseline_states = jax.vmap(
        lambda params: compute_steady_state(setup.vector_field, params, Intervention.none())
    )(_stacked_params)
    baseline_treatment_mean = float(jnp.mean(baseline_states[:, setup.treat_idx]))

    if setup.query.get("estimand", "steady_state") == "trajectory":
        baseline_state_paths, action_state_paths, effect_state_paths = (
            vmap_simulate_action_from_state_composite(
                setup.vector_field,
                setup.param_samples,
                initial_states=baseline_states,
                treat_idx=setup.treat_idx,
                mode=str(setup.action["mode"]),
                value=setup.action.get("value"),
                amount=setup.action.get("amount"),
                time_grid=setup.time_grid,
            )
        )
        outputs = _build_effect_outputs(
            setup,
            effect_paths=effect_state_paths[:, :, setup.outcome_idx],
            reference_node_paths=baseline_state_paths,
            action_node_paths=action_state_paths,
            node_effect_paths=effect_state_paths,
        )
    else:
        effect_draws = vmap_steady_state_effect_composite(
            setup.vector_field,
            setup.param_samples,
            treat_idx=setup.treat_idx,
            outcome_idx=setup.outcome_idx,
            mode=str(setup.action["mode"]),
            value=setup.action.get("value"),
            amount=setup.action.get("amount"),
        )
        outputs = _build_effect_outputs(setup, effect_draws=effect_draws)

    return {
        "result": {
            "rung": 2,
            "action": setup.action,
            "outcome": setup.outcome,
            "estimand": setup.query.get("estimand", "steady_state"),
            "baseline_treatment_mean": baseline_treatment_mean,
            "summary": outputs.summary,
            "effect_trajectory": outputs.effect_trajectory,
            "visualization": outputs.visualization,
            "manifest_effects": outputs.manifest_effects,
            "warnings": _collect_stage6_warnings(
                ctx,
                treatment=setup.treatment,
                include_diagnostic_warnings=True,
            ),
        }
    }


def _execute_simulate_counterfactual(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    setup, error = _prepare_stage6_simulation(ctx, args)
    if error is not None:
        return error
    assert setup is not None

    latent_paths = _fitted_latent_paths_from_result(setup.fitted_artifact.result)
    if latent_paths is None:
        return _tool_error_result(
            "Stage 5b fitted artifact is missing persisted latent state paths required "
            "for counterfactual simulation."
        )

    start_index, start_meta = _resolve_counterfactual_start(
        ctx,
        dict(args.get("start") or {}),
        n_timepoints=int(latent_paths.shape[1]),
    )

    estimand = str(setup.query.get("estimand", "end_state"))
    assert setup.param_samples is not None
    n_draws = len(setup.param_samples)
    initial_states = latent_paths[:, start_index, :]
    if int(initial_states.shape[0]) != n_draws:
        return _tool_error_result(
            "Persisted fitted latent path draw count does not match posterior dynamics "
            f"draw count ({int(initial_states.shape[0])} != {n_draws})."
        )
    start_state = _serialize_latent_state(jnp.mean(initial_states, axis=0), setup.latent_names)

    baseline_state_paths, counterfactual_state_paths, effect_state_paths = (
        vmap_simulate_action_from_state_composite(
            setup.vector_field,
            setup.param_samples,
            initial_states=initial_states,
            treat_idx=setup.treat_idx,
            mode=str(setup.action["mode"]),
            value=setup.action.get("value"),
            amount=setup.action.get("amount"),
            time_grid=setup.time_grid,
        )
    )
    baseline_paths = baseline_state_paths[:, :, setup.outcome_idx]

    if estimand == "trajectory":
        outputs = _build_effect_outputs(
            setup,
            effect_paths=effect_state_paths[:, :, setup.outcome_idx],
            reference_node_paths=baseline_state_paths,
            action_node_paths=counterfactual_state_paths,
            node_effect_paths=effect_state_paths,
            start_state=start_state,
        )
    else:
        outputs = _build_effect_outputs(
            setup,
            effect_draws=effect_state_paths[:, -1, setup.outcome_idx],
            start_state=start_state,
        )

    mean_baseline = jnp.mean(baseline_paths[:, -1])

    return {
        "result": {
            "rung": 3,
            "start": start_meta,
            "action": setup.action,
            "outcome": setup.outcome,
            "estimand": estimand,
            "baseline_forecast_mean": float(mean_baseline),
            "summary": outputs.summary,
            "effect_trajectory": outputs.effect_trajectory,
            "visualization": outputs.visualization,
            "manifest_effects": outputs.manifest_effects,
            "warnings": _collect_stage6_warnings(ctx),
        }
    }


# Registry: (stage_id, tool_name) → implementation function
_TOOL_IMPLS: dict[tuple[str, str], Any] = {
    ("stage-1a", "validate_latent_model"): _execute_validate_latent_model,
    ("stage-1b", "validate_measurement_model"): _execute_validate_measurement_model,
    ("stage-2", "validate_extractions"): _execute_validate_extractions,
    ("stage-4", "submit_model_spec"): _execute_submit_model_spec,
    ("stage-4", "submit_priors"): _execute_submit_priors,
    ("stage-4", "validate_composite_spec"): _execute_validate_composite_spec,
    ("stage-4", "search_literature"): _execute_search_literature,
    ("stage-6", "get_model_info"): _execute_get_model_info,
    ("stage-6", "simulate_intervention"): _execute_simulate_intervention,
    ("stage-6", "simulate_counterfactual"): _execute_simulate_counterfactual,
}

# Upstream dependencies: which stage results need to be loaded for context
_STAGE_CONTEXT_DEPS: dict[str, list[str]] = {
    "stage-1a": [],
    "stage-1b": ["stage-1a"],
    "stage-2": [],
    "stage-4": ["stage-1b"],
    "stage-6": [],
}


def _build_context(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load upstream stage results needed for tool execution context."""
    if stage_id == "stage-6":
        return _build_stage6_context(workspace_id)
    ctx: dict[str, Any] = {"_workspace_id": workspace_id}
    for dep_stage in _STAGE_CONTEXT_DEPS.get(stage_id, []):
        ctx[dep_stage] = _load_stage_result(workspace_id, dep_stage)
    return ctx


def _get_tool_contract(stage_id: str, tool_name: str) -> Any | None:
    contracts = STAGE_TOOLS.get(stage_id) or []
    return next((contract for contract in contracts if contract.name == tool_name), None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    workspace_id: str
    input: dict[str, Any]


class PersistStagePatchRequest(BaseModel):
    workspace_id: str
    patch: dict[str, Any]


@app.get("/api/tools/{stage_id}")
def get_tool_schemas(stage_id: str) -> list[dict[str, Any]]:
    """Return tool definitions for a stage (name, description, JSON Schema parameters/results)."""
    contracts = STAGE_TOOLS.get(stage_id)
    if contracts is None:
        raise HTTPException(404, f"No tools defined for stage {stage_id}")
    return [
        {
            "name": tc.name,
            "description": tc.description,
            "parameters": tc.parameters_json_schema(),
            "result": tc.result_json_schema(),
        }
        for tc in contracts
    ]


@app.post("/api/tools/{stage_id}/{tool_name}")
async def execute_tool(stage_id: str, tool_name: str, request: ToolCallRequest) -> dict[str, Any]:
    """Execute a pipeline tool and return its result."""
    contract = _get_tool_contract(stage_id, tool_name)
    if contract is None:
        raise HTTPException(404, f"No tool contract for tool {tool_name!r} in stage {stage_id!r}")

    impl = _TOOL_IMPLS.get((stage_id, tool_name))
    if impl is None:
        raise HTTPException(404, f"No implementation for tool {tool_name!r} in stage {stage_id!r}")

    try:
        validated_input = contract.input_schema.model_validate(request.input).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors()) from exc

    try:
        ctx = _build_context(request.workspace_id, stage_id)
        import inspect

        payload = (
            await impl(ctx, validated_input)
            if inspect.iscoroutinefunction(impl)
            else impl(ctx, validated_input)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Tool execution failed for %s/%s", stage_id, tool_name)
        raise HTTPException(
            500,
            detail={
                "message": str(exc) or repr(exc),
                "exception_type": exc.__class__.__name__,
                "stage_id": stage_id,
                "tool_name": tool_name,
            },
        ) from exc

    if contract.output_schema is None:
        return payload

    try:
        payload["result"] = contract.output_schema.model_validate(payload.get("result")).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise HTTPException(
            500,
            detail={
                "message": (
                    f"Tool {tool_name!r} in stage {stage_id!r} returned a payload "
                    "that violates its declared result contract."
                ),
                "errors": exc.errors(),
            },
        ) from exc
    return payload


@app.post("/api/stages/{stage_id}/persist-web-patch")
def persist_stage_web_patch(stage_id: str, request: PersistStagePatchRequest) -> dict[str, Any]:
    """Persist a validated patch to a stage's public payload and refresh snapshot web state."""
    return {
        "ok": True,
        "payload": persist_web_patch(stage_id, request.patch, request.workspace_id),
    }
