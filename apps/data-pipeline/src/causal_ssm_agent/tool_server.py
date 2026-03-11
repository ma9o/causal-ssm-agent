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
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from causal_ssm_agent.flows.stages.contracts import STAGE_TOOLS

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

_RESULTS_DIR = Path("results")


def _load_stage_result(run_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted stage result from disk."""
    path = _RESULTS_DIR / run_id / f"{stage_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Stage result not found: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tool implementations — map (stage_id, tool_name) → execute(context, input)
# ---------------------------------------------------------------------------


def _execute_validate_latent_model(
    _ctx: dict[str, Any], args: dict[str, Any]
) -> str:
    from causal_ssm_agent.orchestrator.schemas import validate_latent_model
    from causal_ssm_agent.utils.llm import _validate_json_and_format

    return _validate_json_and_format(args["structure_json"], validate_latent_model)


def _execute_validate_measurement_model(
    ctx: dict[str, Any], args: dict[str, Any]
) -> str:
    from causal_ssm_agent.models.ssm_compiler import (
        validate_measurement_model_for_compilation,
    )
    from causal_ssm_agent.orchestrator.schemas import LatentModel
    from causal_ssm_agent.utils.llm import _validate_json_and_format

    stage1a = ctx.get("stage-1a", {})
    latent_model = LatentModel.model_validate(stage1a["latent_model"])
    return _validate_json_and_format(
        args["measurement_json"],
        lambda data: validate_measurement_model_for_compilation(data, latent_model),
    )


def _execute_validate_model_spec(
    ctx: dict[str, Any], args: dict[str, Any]
) -> str:
    from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.llm import _validate_json_and_format

    stage1b = ctx.get("stage-1b", {})
    causal_spec = stage1b.get("causal_spec", {})
    indicators = get_indicators(causal_spec)
    return _validate_json_and_format(
        args["model_spec_json"],
        lambda data: validate_model_spec_dict(data, indicators=indicators or None),
    )


def _execute_validate_extractions(
    ctx: dict[str, Any], args: dict[str, Any]
) -> str:
    from causal_ssm_agent.utils.llm import _validate_json_and_format
    from causal_ssm_agent.workers.schemas import validate_worker_output

    schema = ctx.get("_extraction_schema", {})
    return _validate_json_and_format(
        args["output_json"],
        lambda data: validate_worker_output(data, schema),
    )


# Registry: (stage_id, tool_name) → implementation function
_TOOL_IMPLS: dict[tuple[str, str], Any] = {
    ("stage-1a", "validate_latent_model_tool"): _execute_validate_latent_model,
    ("stage-1b", "validate_measurement_model_tool"): _execute_validate_measurement_model,
    ("stage-2", "validate_extractions"): _execute_validate_extractions,
    ("stage-4", "validate_model_spec_tool"): _execute_validate_model_spec,
}

# Upstream dependencies: which stage results need to be loaded for context
_STAGE_CONTEXT_DEPS: dict[str, list[str]] = {
    "stage-1a": [],
    "stage-1b": ["stage-1a"],
    "stage-2": [],
    "stage-4": ["stage-1b"],
}


def _build_context(run_id: str, stage_id: str) -> dict[str, Any]:
    """Load upstream stage results needed for tool execution context."""
    ctx: dict[str, Any] = {}
    for dep_stage in _STAGE_CONTEXT_DEPS.get(stage_id, []):
        ctx[dep_stage] = _load_stage_result(run_id, dep_stage)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    run_id: str
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
def execute_tool(
    stage_id: str, tool_name: str, request: ToolCallRequest
) -> dict[str, str]:
    """Execute a pipeline tool and return its result."""
    impl = _TOOL_IMPLS.get((stage_id, tool_name))
    if impl is None:
        raise HTTPException(
            404, f"No implementation for tool {tool_name!r} in stage {stage_id!r}"
        )

    ctx = _build_context(request.run_id, stage_id)
    result = impl(ctx, request.input)
    return {"result": result}
