"""Lightweight tool execution server for the refinement proxy.

Exposes pipeline tool schemas and execution over HTTP so the Next.js
refinement route can proxy LLM tool calls to the same Python validation
logic the pipeline uses.

Run alongside Prefect::

    cd apps/data-pipeline
    uv run uvicorn causal_ssm_agent.tool_server:app --port 8100
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from causal_ssm_agent.flows.run_store import load_parquet, load_pickle, load_stage_snapshot
from causal_ssm_agent.flows.stages.contracts import STAGE_TOOLS
from causal_ssm_agent.flows.stages.persist import persist_web_patch
from causal_ssm_agent.flows.stages.stage_tools import (
    search_literature,
    stage1a_grounding,
    stage1b_grounding,
    stage4_grounding,
)
from causal_ssm_agent.models.ssm.counterfactual import (
    approximate_abducted_state,
    forward_simulate_action_from_state,
    forward_simulate_latent_action_from_state,
    steady_state,
    summarize_draws,
    treatment_effect_for_action,
)
from causal_ssm_agent.models.ssm_builder import prepare_model_runtime
from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours
from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.causal_spec import (
    get_estimable_treatments,
    get_estimation_constructs,
    get_estimation_state_order,
    get_indicators,
    get_outcome_name,
)
from causal_ssm_agent.utils.data import runs_dir

logger = logging.getLogger(__name__)

app = FastAPI(title="Tool Server", docs_url="/api/tools/docs")

app.add_middleware(
    CORSMiddleware,
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


def _load_data_for_model(workspace_id: str) -> Any:
    """Load data_for_model parquet for prior predictive checks."""
    import polars as pl

    path = storage.join(runs_dir(workspace_id), "stage-4-data.parquet")
    if storage.exists(path):
        return pl.read_parquet(path, storage_options=storage.polars_storage_options())
    return None


def _load_runtime_stage_result(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load the internal persisted stage result (with artifact paths)."""
    try:
        snapshot = load_stage_snapshot(workspace_id, stage_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Stage snapshot not found for {stage_id}") from exc
    result = snapshot.get("result")
    if not isinstance(result, dict):
        raise HTTPException(500, f"Stage snapshot for {stage_id} is malformed")
    return result


def _load_optional_stage_result(workspace_id: str, stage_id: str) -> dict[str, Any]:
    try:
        return _load_stage_result(workspace_id, stage_id)
    except HTTPException:
        return {}


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


def _point_like_manifest_names(observation_support: Any) -> set[str] | None:
    if observation_support is None:
        return None
    return {
        name
        for name, support_kind in zip(
            observation_support.manifest_names,
            observation_support.support_kinds,
            strict=False,
        )
        if support_kind in (None, "point")
    }


def _manifest_effects(
    samples: dict[str, Any],
    outcome_idx: int,
    effect_mean: float,
    manifest_names: list[str],
    observation_support: Any,
) -> dict[str, float] | None:
    lambda_draws = samples.get("lambda")
    if lambda_draws is None:
        return None
    lambda_mean = (
        jnp.mean(lambda_draws, axis=0) if getattr(lambda_draws, "ndim", 0) == 3 else lambda_draws
    )
    if getattr(lambda_mean, "ndim", 0) != 2:
        return None

    point_like = _point_like_manifest_names(observation_support)
    effects: dict[str, float] = {}
    for idx, loading in enumerate(lambda_mean[:, outcome_idx]):
        loading_value = float(loading)
        if abs(loading_value) <= 1e-9:
            continue
        name = manifest_names[idx] if idx < len(manifest_names) else f"manifest_{idx}"
        if point_like is not None and name not in point_like:
            continue
        effects[name] = loading_value * effect_mean
    return effects or None


def _serialize_effect_trajectory(trajectory: jnp.ndarray, dt_days: float) -> list[dict[str, float]]:
    return [
        {
            "day": round((idx + 1) * dt_days, 3),
            "effect": float(value),
        }
        for idx, value in enumerate(trajectory.tolist())
    ]


def _serialize_node_effect_trajectories(
    effect_paths: jnp.ndarray,
    latent_names: list[str],
) -> dict[str, list[float]]:
    mean_paths = jnp.mean(effect_paths, axis=0)
    return {
        name: [float(value) for value in mean_paths[:, idx].tolist()]
        for idx, name in enumerate(latent_names)
    }


def _serialize_latent_state(state: jnp.ndarray, latent_names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(latent_names, state.tolist(), strict=False)}


def _select_evidence_window(
    timestamps: list[datetime],
    evidence: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    if not timestamps:
        raise HTTPException(
            400, "No observed history is available for counterfactual conditioning."
        )

    start = _parse_iso_datetime(evidence.get("start_time")) or timestamps[0]
    end = _parse_iso_datetime(evidence.get("end_time")) or timestamps[-1]
    if start > end:
        raise HTTPException(400, "evidence.start_time must be <= evidence.end_time")

    matching = [idx for idx, ts in enumerate(timestamps) if start <= ts <= end]
    if not matching:
        raise HTTPException(400, "Evidence window does not overlap the observed history.")

    return (
        matching[0],
        matching[-1],
        {
            "start_time": timestamps[matching[0]].isoformat(),
            "end_time": timestamps[matching[-1]].isoformat(),
            "n_timepoints": len(matching),
            "variables": list(evidence.get("variables", []) or []),
        },
    )


def _build_stage6_context(workspace_id: str) -> dict[str, Any]:
    stage1b = _load_stage_result(workspace_id, "stage-1b")
    stage4 = _load_optional_stage_result(workspace_id, "stage-4")
    stage4b = _load_optional_stage_result(workspace_id, "stage-4b")
    stage5b = _load_stage_result(workspace_id, "stage-5b")
    stage6 = _load_optional_stage_result(workspace_id, "stage-6")

    stage2_runtime = _load_runtime_stage_result(workspace_id, "stage-2")
    stage5b_runtime = _load_runtime_stage_result(workspace_id, "stage-5b")
    fitted_artifact = load_pickle(stage5b_runtime["_fitted_result_path"])
    data_for_model = load_parquet(stage2_runtime["_data_for_model_path"])
    runtime = prepare_model_runtime(data_for_model=data_for_model, builder=fitted_artifact.builder)

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
        "stage-4b": stage4b,
        "stage-5b": stage5b,
        "stage-6": stage6,
        "_stage-2-runtime": stage2_runtime,
        "_stage-5b-runtime": stage5b_runtime,
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
    drift_draws: Any
    cint_draws: Any
    dt_days: float
    horizon_steps: int


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
    samples = fitted_artifact.result.get_samples() if fitted_artifact.result is not None else None
    if fitted_artifact.result is None or fitted_artifact.builder is None or not samples:
        return None, _tool_error_result("Fitted model artifact is unavailable for simulation.")

    action = dict(args.get("action") or {})
    treatment = str(action.get("variable", ""))
    if treatment not in ctx["_identifiable_treatments"]:
        return None, _tool_error_result(
            f"Treatment '{treatment}' is not an identifiable stage-6 intervention target.",
            identifiable_treatments=ctx["_identifiable_treatments"],
        )

    causal_spec = ctx["stage-1b"].get("causal_spec", {})
    outcome = str(args.get("outcome") or ctx.get("_outcome_name") or "")
    spec = fitted_artifact.builder._spec
    latent_names = list(spec.latent_names or [])
    manifest_names = list(spec.manifest_names or [])
    name_to_idx = {name: idx for idx, name in enumerate(latent_names)}
    treat_idx = name_to_idx.get(treatment)
    outcome_idx = name_to_idx.get(outcome)
    if treat_idx is None or outcome_idx is None:
        return None, _tool_error_result("Treatment or outcome not present in fitted latent model.")

    drift_draws = samples.get("drift")
    if drift_draws is None:
        return None, _tool_error_result("Posterior drift samples are unavailable.")
    cint_draws = samples.get("cint")
    if cint_draws is None:
        cint_draws = jnp.zeros((drift_draws.shape[0], drift_draws.shape[-1]))

    query = dict(args.get("query") or {})
    dt_days, horizon_steps = _stage6_time_config(
        causal_spec,
        runtime.times,
        int(query.get("horizon_days") or 30),
    )

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
            drift_draws=drift_draws,
            cint_draws=cint_draws,
            dt_days=dt_days,
            horizon_steps=horizon_steps,
        ),
        None,
    )


def _build_visualization_payload(
    latent_names: list[str],
    *,
    node_effect_paths: jnp.ndarray | None = None,
    abducted_state: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    node_effect_trajectories = (
        _serialize_node_effect_trajectories(node_effect_paths, latent_names)
        if node_effect_paths is not None
        else None
    )
    if node_effect_trajectories is None and abducted_state is None:
        return None
    return {
        "node_effect_trajectories": node_effect_trajectories,
        "abducted_state": abducted_state,
    }


def _build_effect_outputs(
    setup: Stage6SimulationSetup,
    *,
    effect_draws: jnp.ndarray | None = None,
    effect_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    abducted_state: dict[str, float] | None = None,
) -> Stage6EffectOutputs:
    if effect_paths is not None:
        effect_draws = effect_paths[:, -1]
        mean_effect_trajectory = jnp.mean(effect_paths, axis=0)
        effect_trajectory = _serialize_effect_trajectory(mean_effect_trajectory, setup.dt_days)
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
            setup.fitted_artifact.observation_support,
        )

    return Stage6EffectOutputs(
        summary=summary,
        effect_trajectory=effect_trajectory,
        visualization=_build_visualization_payload(
            setup.latent_names,
            node_effect_paths=node_effect_paths,
            abducted_state=abducted_state,
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


def _execute_validate_model(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    from causal_ssm_agent.flows.stages.stage_tools import should_capture_stage4_output

    workspace_id = ctx["_workspace_id"]
    stage1b = ctx.get("stage-1b", {})
    causal_spec = stage1b.get("causal_spec", {})
    current = _load_stage4_current(workspace_id)
    data_for_model = _load_data_for_model(workspace_id)
    result = _run_compute(
        args,
        "model_json",
        lambda data: stage4_grounding(
            data, causal_spec, current=current, data_for_model=data_for_model
        ),
    )
    if not should_capture_stage4_output(result.get("stage_output"), result.get("result", "")):
        result["stage_output"] = None
    return result


async def _execute_search_literature(_ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Execute search_literature via Exa API (async)."""
    query = args.get("query", "")
    if not query:
        return {"result": "Error: query is required"}
    result = await search_literature(query)
    return {"result": result}


def _execute_validate_extractions(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, str]:
    from causal_ssm_agent.utils.llm import _validate_json_and_format
    from causal_ssm_agent.workers.schemas import validate_worker_output

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
    stage4b = ctx.get("stage-4b", {})
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
            "n_latent": len(getattr(fitted_artifact.builder._spec, "latent_names", []) or []),
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
            "inference_structure": (stage4b.get("inference_structure") or {}),
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
                "evidence_mode": "observed_window",
                "conditioning_methods": ["kalman_smoother", "observation_pseudoinverse"],
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

    baseline_states = jax.vmap(lambda d, c: steady_state(d, c))(setup.drift_draws, setup.cint_draws)
    baseline_treatment_mean = float(jnp.mean(baseline_states[:, setup.treat_idx]))

    if setup.query.get("estimand", "steady_state") == "trajectory":
        _, _, node_effect_paths = jax.vmap(
            lambda d, c, s: forward_simulate_latent_action_from_state(
                d,
                c,
                s,
                setup.treat_idx,
                mode=str(setup.action["mode"]),
                value=setup.action.get("value"),
                amount=setup.action.get("amount"),
                dt=setup.dt_days,
                horizon_steps=setup.horizon_steps,
            )
        )(setup.drift_draws, setup.cint_draws, baseline_states)
        outputs = _build_effect_outputs(
            setup,
            effect_paths=node_effect_paths[:, :, setup.outcome_idx],
            node_effect_paths=node_effect_paths,
        )
    else:
        outputs = _build_effect_outputs(
            setup,
            effect_draws=jax.vmap(
                lambda d, c: treatment_effect_for_action(
                    d,
                    c,
                    setup.treat_idx,
                    setup.outcome_idx,
                    mode=str(setup.action["mode"]),
                    value=setup.action.get("value"),
                    amount=setup.action.get("amount"),
                )
            )(setup.drift_draws, setup.cint_draws),
        )

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

    evidence = dict(args.get("evidence") or {})
    evidence_start_idx, evidence_end_idx, evidence_meta = _select_evidence_window(
        ctx["_observation_timestamps"],
        evidence,
    )
    abducted = approximate_abducted_state(
        setup.samples,
        setup.fitted_artifact.builder._model,
        setup.spec,
        setup.runtime.observations,
        setup.runtime.times,
        evidence_start_idx,
        evidence_end_idx,
    )
    initial_state = abducted["state"]
    abducted_state = _serialize_latent_state(initial_state, setup.latent_names)
    estimand = str(setup.query.get("estimand", "end_state"))

    if estimand == "trajectory":
        baseline_state_paths, _counterfactual_state_paths, effect_state_paths = jax.vmap(
            lambda d, c: forward_simulate_latent_action_from_state(
                d,
                c,
                initial_state,
                setup.treat_idx,
                mode=str(setup.action["mode"]),
                value=setup.action.get("value"),
                amount=setup.action.get("amount"),
                dt=setup.dt_days,
                horizon_steps=setup.horizon_steps,
            )
        )(setup.drift_draws, setup.cint_draws)
        baseline_paths = baseline_state_paths[:, :, setup.outcome_idx]
        outputs = _build_effect_outputs(
            setup,
            effect_paths=effect_state_paths[:, :, setup.outcome_idx],
            node_effect_paths=effect_state_paths,
            abducted_state=abducted_state,
        )
    else:
        baseline_paths, _counterfactual_paths, effect_paths = jax.vmap(
            lambda d, c: forward_simulate_action_from_state(
                d,
                c,
                initial_state,
                setup.treat_idx,
                setup.outcome_idx,
                mode=str(setup.action["mode"]),
                value=setup.action.get("value"),
                amount=setup.action.get("amount"),
                dt=setup.dt_days,
                horizon_steps=setup.horizon_steps,
            )
        )(setup.drift_draws, setup.cint_draws)
        outputs = _build_effect_outputs(
            setup,
            effect_draws=effect_paths[:, -1],
            abducted_state=abducted_state,
        )

    mean_baseline = jnp.mean(baseline_paths[:, -1])

    return {
        "result": {
            "rung": 3,
            "evidence": {
                **evidence_meta,
                "conditioning_method": abducted["method"],
            },
            "action": setup.action,
            "outcome": setup.outcome,
            "estimand": estimand,
            "baseline_forecast_mean": float(mean_baseline),
            "summary": outputs.summary,
            "effect_trajectory": outputs.effect_trajectory,
            "visualization": outputs.visualization,
            "manifest_effects": outputs.manifest_effects,
            "warnings": _collect_stage6_warnings(
                ctx,
                extra_warnings=[str(abducted["warning"])] if abducted.get("warning") else None,
            ),
        }
    }


# Registry: (stage_id, tool_name) → implementation function
_TOOL_IMPLS: dict[tuple[str, str], Any] = {
    ("stage-1a", "validate_latent_model"): _execute_validate_latent_model,
    ("stage-1b", "validate_measurement_model"): _execute_validate_measurement_model,
    ("stage-2", "validate_extractions"): _execute_validate_extractions,
    ("stage-4", "validate_model"): _execute_validate_model,
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


def _load_stage4_current(workspace_id: str) -> dict[str, Any] | None:
    """Load stage-4 result with draft overlay for state accumulation.

    During refinement, priors are submitted incrementally. Each successful
    tool call saves a draft; subsequent calls merge new proposals with the
    accumulated state (original result + draft overlay).
    """
    path = storage.join(runs_dir(workspace_id), "stage-4.json")
    if not storage.exists(path):
        return None
    state = storage.read_json(path)
    draft_path = storage.join(runs_dir(workspace_id), "stage-4-draft.json")
    if storage.exists(draft_path):
        state.update(storage.read_json(draft_path))
    return state


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

    ctx = _build_context(request.workspace_id, stage_id)
    import inspect

    payload = (
        await impl(ctx, validated_input)
        if inspect.iscoroutinefunction(impl)
        else impl(ctx, validated_input)
    )
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
