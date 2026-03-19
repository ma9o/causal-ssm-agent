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
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from causal_ssm_agent.flows.stages.contracts import STAGE_TOOLS
from causal_ssm_agent.flows.stages.stage_tools import (
    search_literature,
    stage1a_grounding,
    stage1b_grounding,
    stage4_grounding,
)
from causal_ssm_agent.utils import storage
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


def _load_stage_result(user_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted stage result from storage."""
    path = storage.join(runs_dir(user_id), f"{stage_id}.json")
    if not storage.exists(path):
        raise HTTPException(404, f"Stage result not found: {path}")
    return storage.read_json(path)


def _load_raw_data(user_id: str) -> Any:
    """Load raw_data parquet for prior predictive checks."""
    import polars as pl

    path = storage.join(runs_dir(user_id), "stage-4-data.parquet")
    if storage.exists(path):
        return pl.read_parquet(path, storage_options=storage.polars_storage_options())
    return None


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
    user_id = ctx["_user_id"]
    stage1b = ctx.get("stage-1b", {})
    causal_spec = stage1b.get("causal_spec", {})
    current = _load_stage4_current(user_id)
    raw_data = _load_raw_data(user_id)
    return _run_compute(
        args,
        "model_json",
        lambda data: stage4_grounding(data, causal_spec, current=current, raw_data=raw_data),
    )


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


# Registry: (stage_id, tool_name) → implementation function
_TOOL_IMPLS: dict[tuple[str, str], Any] = {
    ("stage-1a", "validate_latent_model"): _execute_validate_latent_model,
    ("stage-1b", "validate_measurement_model"): _execute_validate_measurement_model,
    ("stage-2", "validate_extractions"): _execute_validate_extractions,
    ("stage-4", "validate_model"): _execute_validate_model,
    ("stage-4", "search_literature"): _execute_search_literature,
}

# Upstream dependencies: which stage results need to be loaded for context
_STAGE_CONTEXT_DEPS: dict[str, list[str]] = {
    "stage-1a": [],
    "stage-1b": ["stage-1a"],
    "stage-2": [],
    "stage-4": ["stage-1b"],
}


def _load_stage4_current(user_id: str) -> dict[str, Any] | None:
    """Load stage-4 result with draft overlay for state accumulation.

    During refinement, priors are submitted incrementally. Each successful
    tool call saves a draft; subsequent calls merge new proposals with the
    accumulated state (original result + draft overlay).
    """
    path = storage.join(runs_dir(user_id), "stage-4.json")
    if not storage.exists(path):
        return None
    state = storage.read_json(path)
    draft_path = storage.join(runs_dir(user_id), "stage-4-draft.json")
    if storage.exists(draft_path):
        state.update(storage.read_json(draft_path))
    return state


def _build_context(user_id: str, stage_id: str) -> dict[str, Any]:
    """Load upstream stage results needed for tool execution context."""
    ctx: dict[str, Any] = {"_user_id": user_id}
    for dep_stage in _STAGE_CONTEXT_DEPS.get(stage_id, []):
        ctx[dep_stage] = _load_stage_result(user_id, dep_stage)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    user_id: str
    input: dict[str, Any]


@app.get("/api/tools/{stage_id}")
def get_tool_schemas(stage_id: str) -> list[dict[str, Any]]:
    """Return tool definitions for a stage (name, description, JSON Schema parameters)."""
    contracts = STAGE_TOOLS.get(stage_id)
    if contracts is None:
        raise HTTPException(404, f"No tools defined for stage {stage_id}")
    return [
        {
            "name": tc.name,
            "description": tc.description,
            "parameters": tc.parameters_json_schema(),
        }
        for tc in contracts
    ]


@app.post("/api/tools/{stage_id}/{tool_name}")
async def execute_tool(stage_id: str, tool_name: str, request: ToolCallRequest) -> dict[str, Any]:
    """Execute a pipeline tool and return its result."""
    impl = _TOOL_IMPLS.get((stage_id, tool_name))
    if impl is None:
        raise HTTPException(404, f"No implementation for tool {tool_name!r} in stage {stage_id!r}")

    ctx = _build_context(request.user_id, stage_id)
    import inspect

    if inspect.iscoroutinefunction(impl):
        return await impl(ctx, request.input)
    return impl(ctx, request.input)
