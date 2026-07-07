"""Single-source Stage 4 public tool registry.

Centralizes the Stage 4 *public* refinement-tool surface exposed through the tool
server: contract metadata (:func:`build_stage4_public_tool_contracts`) and the
execution handlers. Each public submission is validated by ``stage4_grounding``
against the persisted Stage 4 state; the batch construct-admission flow owns
incremental admission and does not go through this registry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.run_store import load_parquet
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename

if TYPE_CHECKING:
    from nof1_causal_lab.flows.stage_contracts import ToolContract


Stage4PublicToolImpl = Callable[[dict[str, Any], dict[str, Any]], Any]


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for empirical literature about effect sizes.")
    parameter_name: str = Field(
        description="Name of the parameter this search is for (e.g. 'beta_stress_sleep')."
    )


class SubmitStatisticalModelSpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statistical_model_spec_json: str = Field(
        description=(
            "The JSON string containing the complete StatisticalModelSpec to lock for Stage 4."
        ),
    )


class SubmitPriorsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priors_json: str = Field(
        description="The JSON string containing prior proposals keyed by parameter name.",
    )


@dataclass(frozen=True)
class Stage4ToolSpec:
    """Single-source metadata for one public Stage 4 tool."""

    name: str
    description: str
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    public_impl: Stage4PublicToolImpl | None = None


def _parse_json_arg(args: dict[str, Any], param_name: str) -> tuple[Any | None, str | None]:
    """Parse one JSON-encoded public tool argument."""
    raw = args.get(param_name, "")
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def _load_stage2_data_for_model(workspace_id: str) -> Any:
    from nof1_causal_lab.machine.store import current_artifact_file

    try:
        path = current_artifact_file(
            workspace_id, "model_data", parquet_filename("model_data", "model_data")
        )
    except FileNotFoundError:
        return None
    return load_parquet(path)


def _load_stage4_current(workspace_id: str) -> dict[str, Any] | None:
    """Load the current accepted Stage 4 report, if one exists."""
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal

    info = EpisodeJournal(workspace_id).latest_state().get("compiled_ssm")
    if info is None:
        return None
    return ArtifactStore(workspace_id).read_json_file(
        "compiled_ssm", info.version, json_filename("compiled_ssm", "report")
    )


def _execute_stage4_submission(
    ctx: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Execute one public Stage 4 refinement submission."""
    from nof1_causal_lab.flows.stages.stage4.grounding import (
        should_capture_stage4_output,
        stage4_grounding,
    )

    workspace_id = ctx["_workspace_id"]
    stage1b = ctx.get("stage-1b", {})
    causal_design = stage1b.get("causal_design", {})
    current = _load_stage4_current(workspace_id)
    data_for_model = _load_stage2_data_for_model(workspace_id)

    grounding_result = stage4_grounding(
        data,
        causal_design,
        current=current,
        data_for_model=data_for_model,
    )
    stage_output = (
        grounding_result.stage_output if should_capture_stage4_output(grounding_result) else None
    )
    return {"result": grounding_result.feedback, "stage_output": stage_output}


def _execute_public_json_submission(
    ctx: dict[str, Any],
    args: dict[str, Any],
    *,
    arg_name: str,
    payload_key: str,
) -> dict[str, Any]:
    """Execute one JSON-encoded Stage 4 public submission."""
    value, error = _parse_json_arg(args, arg_name)
    if error is not None:
        return {"result": error, "stage_output": None}
    return _execute_stage4_submission(ctx, {payload_key: value})


async def execute_public_search_literature(
    _ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """Execute the public Stage 4 literature-search tool."""
    from nof1_causal_lab.flows.stages.stage4.tools import search_literature

    query = args.get("query", "")
    if not query:
        return {"result": "Error: query is required"}
    result = await search_literature(query)
    return {"result": result}


execute_public_submit_statistical_model_spec = partial(
    _execute_public_json_submission,
    arg_name="statistical_model_spec_json",
    payload_key="statistical_model_spec",
)
execute_public_submit_priors = partial(
    _execute_public_json_submission,
    arg_name="priors_json",
    payload_key="priors",
)


STAGE4_TOOL_SPECS: tuple[Stage4ToolSpec, ...] = (
    Stage4ToolSpec(
        name="search_literature",
        description="Search for empirical literature about effect sizes for model parameters.",
        input_schema=SearchLiteratureInput,
        public_impl=execute_public_search_literature,
    ),
    Stage4ToolSpec(
        name="submit_statistical_model_spec",
        description="Submit the full Stage 4 StatisticalModelSpec for compile-only locking and validation.",
        input_schema=SubmitStatisticalModelSpecInput,
        public_impl=execute_public_submit_statistical_model_spec,
    ),
    Stage4ToolSpec(
        name="submit_priors",
        description="Submit Stage 4 prior proposals for schema, compile, and prior-predictive validation.",
        input_schema=SubmitPriorsInput,
        public_impl=execute_public_submit_priors,
    ),
)


def build_stage4_public_tool_contracts() -> list[ToolContract]:
    """Materialize Stage 4 public tool contracts from the shared registry."""
    from nof1_causal_lab.flows.contracts_base import ToolContract

    return [
        ToolContract(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
        )
        for spec in STAGE4_TOOL_SPECS
        if spec.input_schema is not None
    ]
