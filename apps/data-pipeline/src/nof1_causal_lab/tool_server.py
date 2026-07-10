"""Lightweight tool execution server for the refinement proxy.

Exposes pipeline tool schemas and execution over HTTP so the Next.js
refinement route can proxy LLM tool calls to the same Python validation
logic the stages use, plus the episode facade (moves via the Temporal
workflow, reads via the append-only transition log).

Run alongside the Temporal dev server and episode worker::

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
from nof1_causal_lab.episode_api import (
    capabilities_router,
    machine_router,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.episode_api import router as episode_router
from nof1_causal_lab.flows.artifact_contracts import CONTEXT_TOOLS
from nof1_causal_lab.flows.transitions.latent_structure.grounding import latent_structure_grounding
from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
    measurement_structure_grounding,
)
from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    execute_public_search_literature as _execute_search_literature,
)
from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    execute_public_submit_priors as _execute_submit_priors,
)
from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    execute_public_submit_statistical_model_spec as _execute_submit_statistical_model_spec,
)
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename, pickle_filename
from nof1_causal_lab.models.ssm import SSMModel
from nof1_causal_lab.models.ssm.counterfactual import (
    ClampSpec,
    summarize_draws,
    vmap_simulate_clamps_from_state,
)
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VectorField,
    compute_steady_state,
    posterior_dynamics_from_result,
)
from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime
from nof1_causal_lab.utils.causal_design import (
    get_estimation_constructs,
    get_estimation_state_order,
    get_indicators,
    get_outcome_name,
)

logger = logging.getLogger(__name__)

_API_DESCRIPTION = """\
The episode machine is the single interface to an N-of-1 causal analysis. An
external agent drives it entirely over this HTTP API — the same surface the web
viewer uses. There is no SDK and no MCP server: `curl` is the interface.

## Orientation

Call `GET /api/machine` once. It returns the static artifact graph — every
transition with what it consumes, produces, and optionally co-produces — plus
each transition's creation class and the derivation graph:

- `deterministic` — pure compute, no credentials (e.g. identification).
- `batch_llm` — bulk LLM compute on the service's ambient key. You trigger it
  with a `run` move; you never supply a key.
- `judgment` — proposal work you can do yourself by writing the produced
  artifact directly. These transitions are flagged `writable`.

## The loop

1. `GET /api/machine` once, then `GET /api/episodes/{workspace_id}` for the live
   state: per-artifact freshness, the legal moves, and whether an auto-run is
   active.
2. Propose a move at `POST /api/episodes/{workspace_id}/moves` — either
   `{"move": {"kind": "run", "artifact_id": "latent_structure"}}` to run a transition, or
   `{"move": {"kind": "write", "artifact_id": "latent_structure", "provenance": "llm"}, "payload": {...}}`
   to author a judgment artifact directly.
3. Long transitions (`statistical_model_spec`, `posterior` — minutes to hours) can outlive a client
   timeout. Prefer `POST /api/episodes/{workspace_id}/auto` (a background driver
   that runs enabled transitions in dependency order) and poll the state.
4. Read what happened at `GET /api/episodes/{workspace_id}/timeline`: `applied`,
   `rejected` (illegal, state unchanged), or `raised` (typed transition error).

## Staleness

A `write` becomes a new provenance root and marks everything downstream stale
until re-run. Numeric tools (`simulate`, `get_model_info`) hard-flag
stale provenance chains in their warnings — never report numbers past those
flags.

## Data in, results out

Upload raw data at `POST /api/upload` (`multipart/form-data` with `workspaceId`
and `file`) before running the `raw_data` transition. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files
(parquet, pickle) are served individually from `.../files/{filename}`.

## Read-only deployments

The hosted viewer's backend serves these same read endpoints against a published
store with no move plane. `GET /api/capabilities` reports `moves_enabled`; every
move returns 403 when it is `false`.
"""

app = FastAPI(
    title="nof1-causal-lab episode API",
    description=_API_DESCRIPTION,
    docs_url="/api/tools/docs",
)

app.add_middleware(
    cast("Any", CORSMiddleware),
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(episode_router)
app.include_router(capabilities_router)
app.include_router(workspaces_router)
app.include_router(uploads_router)
app.include_router(machine_router)


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


def _analysis_time_config(
    causal_design: dict[str, Any],
    times: Any,
    horizon_days: int,
) -> tuple[float, int]:
    model_clock = ((causal_design or {}).get("measurement") or {}).get("model_clock")
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


def _build_ranking_context(workspace_id: str) -> dict[str, Any]:
    """Query-plane context: pinned artifact versions + provenance freshness.

    Everything is read from the versioned store at the episode's *current*
    versions; ``panel`` comes from the version the posterior was
    actually fitted on (its ``derived_from`` pin). Freshness of the whole
    serving chain rides along so simulations from a superseded model are
    hard-flagged rather than silently served.
    """
    from nof1_causal_lab.machine.moves import freshness_report
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    state = derive_current_state(workspace_id)
    posterior_info = state.get("posterior")
    if posterior_info is None:
        raise HTTPException(404, f"No fitted posterior for workspace {workspace_id}")
    store = ArtifactStore(workspace_id)

    fitted_artifact = store.read_pickle_file(
        "posterior", posterior_info.version, pickle_filename("posterior", "fitted")
    )
    posterior_payload = store.read_json_file(
        "posterior", posterior_info.version, json_filename("posterior", "diagnostics")
    )

    panel_pin = posterior_info.derived_from.get("panel")
    if panel_pin is None:
        raise HTTPException(500, "posterior artifact is missing its panel pin")
    data_for_model = store.read_parquet_file("panel", panel_pin, parquet_filename("panel", "panel"))

    spec_info = state.get("causal_design")
    if spec_info is None:
        raise HTTPException(404, f"No causal_design for workspace {workspace_id}")
    causal_design_payload = store.read_json_file(
        "causal_design", spec_info.version, json_filename("causal_design", "causal_design")
    )

    compiled_info = state.get("compiled_ssm")
    compiled_report = (
        store.read_json_file(
            "compiled_ssm", compiled_info.version, json_filename("compiled_ssm", "report")
        )
        if compiled_info is not None
        else {}
    )
    ranking_info = state.get("baseline_report")
    baseline_report = (
        store.read_json_file(
            "baseline_report",
            ranking_info.version,
            json_filename("baseline_report", "baseline_report"),
        )
        if ranking_info is not None
        else {}
    )
    identification_report_info = state.get("identification_report")
    if identification_report_info is None:
        raise HTTPException(404, f"No identification_report for workspace {workspace_id}")
    identification_report = store.read_json_file(
        "identification_report",
        identification_report_info.version,
        json_filename("identification_report", "identification_report"),
    )

    fitted_spec = getattr(fitted_artifact, "spec", None)
    if fitted_spec is None:
        raise HTTPException(
            500,
            "Posterior fitted artifact is missing the compiled SSMSpec required for ranking tools.",
        )
    runtime = prepare_model_runtime(
        data_for_model=data_for_model,
        model=SSMModel(fitted_spec),
    )
    fitted_artifact.observation_support = runtime.observation_support

    causal_design = causal_design_payload.get("causal_design", {})
    outcome_name = get_outcome_name(causal_design)

    serving_chain = (
        "posterior",
        "baseline_report",
        "causal_design",
        "identification_report",
        "compiled_ssm",
    )
    stale_artifacts = [
        status.artifact_id
        for status in freshness_report(state)
        if status.stale and status.artifact_id in serving_chain
    ]

    return {
        "_workspace_id": workspace_id,
        "causal_design": causal_design_payload,
        "compiled_ssm": compiled_report,
        "posterior": posterior_payload,
        "baseline_report": baseline_report,
        "_fitted_artifact": fitted_artifact,
        "_prepared_runtime": runtime,
        "_observation_timestamps": _extract_observation_timestamps(runtime.observation_data),
        "_outcome_name": outcome_name,
        "_identifiable_treatments": list(identification_report["estimable_treatments"]),
        "_stale_artifacts": stale_artifacts,
    }


@dataclass(frozen=True)
class AnalysisSimulationSetup:
    fitted_artifact: Any
    runtime: Any
    samples: dict[str, Any]
    causal_design: dict[str, Any]
    query: dict[str, Any]
    clamps: list[ClampSpec]
    outcome: str
    spec: Any
    latent_names: list[str]
    manifest_names: list[str]
    outcome_idx: int
    dt_days: float
    horizon_steps: int
    time_grid: jnp.ndarray
    vector_field: VectorField
    # ``param_samples`` is the canonical per-draw component-shape
    # parameter list rebuilt from ``SSMSpec`` and posterior sample sites.
    param_samples: list[tuple[dict[str, Any], ...]] | None = None


@dataclass(frozen=True)
class AnalysisEffectOutputs:
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


def _prepare_analysis_simulation(
    ctx: dict[str, Any],
    args: dict[str, Any],
) -> tuple[AnalysisSimulationSetup | None, dict[str, Any] | None]:
    fitted_artifact = ctx["_fitted_artifact"]
    runtime = ctx["_prepared_runtime"]
    if fitted_artifact.result is None or fitted_artifact.spec is None:
        return None, _tool_error_result("Fitted model artifact is unavailable for simulation.")
    samples = fitted_artifact.result.get_samples() or {}

    causal_design = ctx["causal_design"].get("causal_design", {})
    outcome = str(args.get("outcome") or ctx.get("_outcome_name") or "")
    spec = fitted_artifact.spec
    latent_names = list(spec.latent_names or [])
    manifest_names = list(spec.manifest_names or [])
    name_to_idx = {name: idx for idx, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)
    if outcome_idx is None:
        return None, _tool_error_result("Outcome not present in fitted latent structure.")

    raw_clamps = list(args.get("clamps") or [])
    if not raw_clamps:
        return None, _tool_error_result("At least one clamp is required.")
    identifiable = ctx["_identifiable_treatments"]
    clamps: list[ClampSpec] = []
    for raw in raw_clamps:
        variable = str(raw.get("variable", ""))
        if variable not in identifiable:
            return None, _tool_error_result(
                f"Clamp target '{variable}' is not an identifiable ranking target.",
                identifiable_treatments=identifiable,
            )
        index = name_to_idx.get(variable)
        if index is None:
            return None, _tool_error_result(
                f"Clamp target '{variable}' is not present in the fitted latent structure."
            )
        values = raw.get("values")
        clamps.append(
            ClampSpec(
                index=index,
                mode=str(raw.get("mode")),
                from_day=float(raw.get("from_day") or 0.0),
                to_day=None if raw.get("to_day") is None else float(raw["to_day"]),
                value=raw.get("value"),
                amount=raw.get("amount"),
                value_start=raw.get("value_start"),
                value_end=raw.get("value_end"),
                values=tuple(values) if values is not None else None,
            )
        )

    query = dict(args.get("query") or {})
    dt_days, horizon_steps = _analysis_time_config(
        causal_design,
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
        AnalysisSimulationSetup(
            fitted_artifact=fitted_artifact,
            runtime=runtime,
            samples=samples,
            causal_design=causal_design,
            query=query,
            clamps=clamps,
            outcome=outcome,
            spec=spec,
            latent_names=latent_names,
            manifest_names=manifest_names,
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
    setup: AnalysisSimulationSetup,
    *,
    effect_draws: jnp.ndarray | None = None,
    effect_paths: jnp.ndarray | None = None,
    reference_node_paths: jnp.ndarray | None = None,
    action_node_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    start_state: dict[str, float] | None = None,
) -> AnalysisEffectOutputs:
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

    return AnalysisEffectOutputs(
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


def _collect_analysis_warnings(
    ctx: dict[str, Any],
    *,
    treatments: list[str] | None = None,
    include_diagnostic_warnings: bool = False,
    extra_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    stale_artifacts = ctx.get("_stale_artifacts") or []
    if stale_artifacts:
        warnings.append(
            "STALE PROVENANCE: "
            + ", ".join(stale_artifacts)
            + " were superseded after this posterior was produced; results below "
            "reflect the old model. Re-run the fit chain to refresh."
        )
    if include_diagnostic_warnings and treatments:
        posterior = ctx.get("posterior", {})
        for item in posterior.get("ppc", {}).get("per_variable_warnings", []) or []:
            message = item.get("message")
            if message:
                warnings.append(str(message))

    for warning in extra_warnings or []:
        if warning:
            warnings.append(str(warning))
    return warnings


# ---------------------------------------------------------------------------
# Tool implementations — map (context_id, tool_name) -> execute(context, input)
# ---------------------------------------------------------------------------


def _run_compute(
    args: dict[str, Any],
    param_name: str,
    compute_fn: Any,
) -> dict[str, Any]:
    """Parse JSON arg, run compute function, return result + context_output."""
    raw = args.get(param_name, "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"result": f"JSON parse error: {e}", "context_output": None}

    context_output, feedback = compute_fn(data)
    return {"result": feedback, "context_output": context_output}


def _execute_validate_latent_structure(
    _ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    return _run_compute(args, "structure_json", latent_structure_grounding)


def _execute_validate_measurement_structure(
    ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    latent_structure_payload = ctx.get("latent_structure", {})
    latent_structure = latent_structure_payload["latent_structure"]
    return _run_compute(
        args,
        "measurement_json",
        lambda data: measurement_structure_grounding(data, latent_structure),
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
    causal_design_payload = ctx["causal_design"]
    posterior = ctx.get("posterior", {})
    baseline_report = ctx.get("baseline_report", {})
    runtime = ctx["_prepared_runtime"]
    fitted_artifact = ctx["_fitted_artifact"]
    causal_design = causal_design_payload.get("causal_design", {})
    measurement = causal_design.get("measurement") or {}
    retained_state_names = set(get_estimation_state_order(causal_design))
    constructs = get_estimation_constructs(causal_design)
    indicators = [
        indicator
        for indicator in get_indicators(causal_design)
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
            "n_latent": len(getattr(fitted_artifact.spec, "latent_names", []) or []),
            "n_manifest": len(runtime.manifest_names),
            "inference_method": (posterior.get("inference_metadata") or {}).get("method"),
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
                (causal_design.get("identifiability") or {}).get("non_identifiable_treatments")
                or {}
            ),
        }
    if "diagnostics" in sections:
        payload["diagnostics"] = {
            "ppc_warning_count": len(
                (posterior.get("ppc") or {}).get("per_variable_warnings", []) or []
            ),
        }
    if "baseline_effects" in sections:
        baseline = list(baseline_report.get("intervention_results", []) or [])
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
            "simulate": {
                "supported_targets": ctx["_identifiable_treatments"],
                "start": ["baseline", "abducted"],
                "clamp_modes": ["set", "shift", "ramp", "trajectory"],
                "estimands": ["end_state", "trajectory"],
                "composable": True,
            },
        }
    return payload


def _execute_get_model_info(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return {"result": _build_model_info_payload(ctx, args)}


def _execute_simulate(ctx: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Run a composable scenario: a start state + a list of timed latent clamps.

    The start is the population baseline steady state (interventional) or an abducted
    fitted latent state (counterfactual); the clamps are do-operators over time windows.
    The Pearl rung is emergent from the start rather than a separate query type.
    """
    setup, error = _prepare_analysis_simulation(ctx, args)
    if error is not None:
        return error
    assert setup is not None
    assert setup.param_samples is not None

    start_input = dict(args.get("start") or {})
    kind = str(start_input.get("kind") or "baseline")
    estimand = str(setup.query.get("estimand", "trajectory"))

    if kind == "abducted":
        latent_paths = _fitted_latent_paths_from_result(setup.fitted_artifact.result)
        if latent_paths is None:
            return _tool_error_result(
                "Posterior fitted artifact is missing persisted latent state paths required "
                "for an abducted start."
            )
        start_index, start_meta = _resolve_counterfactual_start(
            ctx, start_input, n_timepoints=int(latent_paths.shape[1])
        )
        initial_states = latent_paths[:, start_index, :]
        n_draws = len(setup.param_samples)
        if int(initial_states.shape[0]) != n_draws:
            return _tool_error_result(
                "Persisted fitted latent path draw count does not match posterior dynamics "
                f"draw count ({int(initial_states.shape[0])} != {n_draws})."
            )
        start_result = {
            "kind": "abducted",
            "time_index": start_meta["time_index"],
            "time": start_meta.get("time"),
            "state_source": "fitted_latent_paths",
        }
    else:
        stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *setup.param_samples)
        initial_states = jax.vmap(
            lambda params: compute_steady_state(setup.vector_field, params, Intervention.none())
        )(stacked_params)
        start_result = {
            "kind": "baseline",
            "time_index": None,
            "time": None,
            "state_source": "baseline_steady_state",
        }

    start_state = _serialize_latent_state(jnp.mean(initial_states, axis=0), setup.latent_names)

    baseline_state_paths, action_state_paths, effect_state_paths = vmap_simulate_clamps_from_state(
        setup.vector_field,
        setup.param_samples,
        initial_states=initial_states,
        clamps=setup.clamps,
        time_grid=setup.time_grid,
    )
    outcome_effect = effect_state_paths[:, :, setup.outcome_idx]
    reference_mean = float(jnp.mean(baseline_state_paths[:, -1, setup.outcome_idx]))

    common_viz: dict[str, Any] = {
        "reference_node_paths": baseline_state_paths,
        "action_node_paths": action_state_paths,
        "node_effect_paths": effect_state_paths,
        "start_state": start_state,
    }
    if estimand == "trajectory":
        outputs = _build_effect_outputs(setup, effect_paths=outcome_effect, **common_viz)
    else:
        outputs = _build_effect_outputs(setup, effect_draws=outcome_effect[:, -1], **common_viz)

    clamp_variables = [setup.latent_names[clamp.index] for clamp in setup.clamps]

    return {
        "result": {
            "start": start_result,
            "clamps": list(args.get("clamps") or []),
            "outcome": setup.outcome,
            "estimand": estimand,
            "summary": outputs.summary,
            "effect_trajectory": outputs.effect_trajectory,
            "visualization": outputs.visualization,
            "manifest_effects": outputs.manifest_effects,
            "reference_mean": reference_mean,
            "warnings": _collect_analysis_warnings(
                ctx,
                treatments=clamp_variables,
                include_diagnostic_warnings=True,
            ),
        }
    }


# Registry: (context_id, tool_name) -> implementation function
_TOOL_IMPLS: dict[tuple[str, str], Any] = {
    ("latent-structure", "validate_latent_structure"): _execute_validate_latent_structure,
    (
        "measurement-structure",
        "validate_measurement_structure",
    ): _execute_validate_measurement_structure,
    ("measurement", "validate_extractions"): _execute_validate_extractions,
    (
        "statistical-model-spec",
        "submit_statistical_model_spec",
    ): _execute_submit_statistical_model_spec,
    ("statistical-model-spec", "submit_priors"): _execute_submit_priors,
    ("statistical-model-spec", "search_literature"): _execute_search_literature,
    ("ranking", "get_model_info"): _execute_get_model_info,
    ("ranking", "simulate"): _execute_simulate,
}

# Upstream dependencies: which context results need to be loaded for execution.
_CONTEXT_DEPS: dict[str, list[str]] = {
    "latent-structure": [],
    "measurement-structure": ["latent_structure"],
    "measurement": [],
    "statistical-model-spec": ["causal_design"],
    "ranking": [],
}


def _load_context_result(workspace_id: str, artifact_id: str) -> dict[str, Any]:
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    state = derive_current_state(workspace_id)
    store = ArtifactStore(workspace_id)
    if artifact_id == "latent_structure":
        info = state.get("latent_structure")
        if info is None:
            raise HTTPException(404, f"No latent_structure for workspace {workspace_id}")
        return store.read_json_file(
            "latent_structure",
            info.version,
            json_filename("latent_structure", "latent_structure"),
        )
    if artifact_id == "causal_design":
        info = state.get("causal_design")
        if info is None:
            raise HTTPException(404, f"No causal_design for workspace {workspace_id}")
        return store.read_json_file(
            "causal_design",
            info.version,
            json_filename("causal_design", "causal_design"),
        )
    raise KeyError(f"No canonical tool context loader for {artifact_id}")


def _build_context(workspace_id: str, context_id: str) -> dict[str, Any]:
    """Load upstream results needed for tool execution context."""
    if context_id == "ranking":
        return _build_ranking_context(workspace_id)
    ctx: dict[str, Any] = {"_workspace_id": workspace_id}
    for artifact_id in _CONTEXT_DEPS.get(context_id, []):
        ctx[artifact_id] = _load_context_result(workspace_id, artifact_id)
    return ctx


def _get_tool_contract(context_id: str, tool_name: str) -> Any | None:
    contracts = CONTEXT_TOOLS.get(context_id) or []
    return next((contract for contract in contracts if contract.name == tool_name), None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    workspace_id: str
    input: dict[str, Any]


@app.get("/api/tools/{context_id}")
def get_tool_schemas(context_id: str) -> list[dict[str, Any]]:
    """List a context's validation/query tools — the same tools the in-service LLM loops use.

    Each entry is `{name, description, parameters, result}` where `parameters`
    and `result` are JSON Schemas. Fetch this first to learn a tool's argument
    shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
    ranking `simulate` / `get_model_info`, statistical-model-spec `submit_statistical_model_spec`.
    """
    contracts = CONTEXT_TOOLS.get(context_id)
    if contracts is None:
        raise HTTPException(404, f"No tools defined for context {context_id}")
    return [
        {
            "name": tc.name,
            "description": tc.description,
            "parameters": tc.parameters_json_schema(),
            "result": tc.result_json_schema(),
        }
        for tc in contracts
    ]


@app.post("/api/tools/{context_id}/{tool_name}")
async def execute_tool(context_id: str, tool_name: str, request: ToolCallRequest) -> dict[str, Any]:
    """Execute a context tool against the workspace's current artifact-store versions.

    Body is `{"workspace_id": "...", "input": {...}}` where `input` matches the
    tool's `parameters` schema from `GET /api/tools/{context_id}`; 422 on a schema
    violation. Numeric tools hard-flag stale provenance chains in their result
    warnings — do not report numbers past those flags.
    """
    contract = _get_tool_contract(context_id, tool_name)
    if contract is None:
        raise HTTPException(
            404, f"No tool contract for tool {tool_name!r} in context {context_id!r}"
        )

    impl = _TOOL_IMPLS.get((context_id, tool_name))
    if impl is None:
        raise HTTPException(
            404, f"No implementation for tool {tool_name!r} in context {context_id!r}"
        )

    try:
        validated_input = contract.input_schema.model_validate(request.input).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors()) from exc

    try:
        ctx = _build_context(request.workspace_id, context_id)
        import inspect

        payload = (
            await impl(ctx, validated_input)
            if inspect.iscoroutinefunction(impl)
            else impl(ctx, validated_input)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Tool execution failed for %s/%s", context_id, tool_name)
        raise HTTPException(
            500,
            detail={
                "message": str(exc) or repr(exc),
                "exception_type": exc.__class__.__name__,
                "context_id": context_id,
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
                    f"Tool {tool_name!r} in context {context_id!r} returned a payload "
                    "that violates its declared result contract."
                ),
                "errors": exc.errors(),
            },
        ) from exc
    return payload
