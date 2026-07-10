"""Single-source model-spec public tool registry.

Centralizes the model-spec *public* refinement-tool surface exposed through the tool
server: contract metadata (:func:`build_model_spec_public_tool_contracts`) and the
execution handlers. Each public submission is validated by ``model_spec_grounding``
against the persisted model-spec state; the batch construct-admission flow owns
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
    from nof1_causal_lab.flows.artifact_contracts import ToolContract


ModelSpecPublicToolImpl = Callable[[dict[str, Any], dict[str, Any]], Any]


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
            "The JSON string containing the complete StatisticalModelSpec to lock for model-spec."
        ),
    )


class SubmitPriorsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priors_json: str = Field(
        description="The JSON string containing prior proposals keyed by parameter name.",
    )


@dataclass(frozen=True)
class ModelSpecToolSpec:
    """Single-source metadata for one public model-spec tool."""

    name: str
    description: str
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    public_impl: ModelSpecPublicToolImpl | None = None


def _parse_json_arg(args: dict[str, Any], param_name: str) -> tuple[Any | None, str | None]:
    """Parse one JSON-encoded public tool argument."""
    raw = args.get(param_name, "")
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def _load_extraction_data_for_model(workspace_id: str) -> Any:
    from nof1_causal_lab.machine.store import current_artifact_file

    try:
        path = current_artifact_file(workspace_id, "panel", parquet_filename("panel", "panel"))
    except FileNotFoundError:
        return None
    return load_parquet(path)


def _load_model_spec_current(workspace_id: str) -> dict[str, Any] | None:
    """Load the current accepted model-spec report, if one exists."""
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    info = derive_current_state(workspace_id).get("statistical_model_spec")
    if info is None:
        return None
    return ArtifactStore(workspace_id).read_json_file(
        "statistical_model_spec",
        info.version,
        json_filename("statistical_model_spec", "statistical_model_spec"),
    )


def _execute_model_spec_submission(
    ctx: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Execute one public model-spec refinement submission."""
    from nof1_causal_lab.flows.transitions.model_spec.grounding import (
        model_spec_grounding,
        should_capture_model_spec_output,
    )

    workspace_id = ctx["_workspace_id"]
    causal_design_payload = ctx.get("causal_design", {})
    causal_design = causal_design_payload.get("causal_design", {})
    current = _load_model_spec_current(workspace_id)
    data_for_model = _load_extraction_data_for_model(workspace_id)

    grounding_result = model_spec_grounding(
        data,
        causal_design,
        current=current,
        data_for_model=data_for_model,
    )
    context_output = (
        grounding_result.context_output
        if should_capture_model_spec_output(grounding_result)
        else None
    )
    return {"result": grounding_result.feedback, "context_output": context_output}


def _execute_public_json_submission(
    ctx: dict[str, Any],
    args: dict[str, Any],
    *,
    arg_name: str,
    payload_key: str,
) -> dict[str, Any]:
    """Execute one JSON-encoded model-spec public submission."""
    value, error = _parse_json_arg(args, arg_name)
    if error is not None:
        return {"result": error, "context_output": None}
    return _execute_model_spec_submission(ctx, {payload_key: value})


async def execute_public_search_literature(
    _ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """Execute the public model-spec literature-search tool."""
    from nof1_causal_lab.flows.transitions.model_spec.tools import search_literature

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


MODEL_SPEC_TOOL_SPECS: tuple[ModelSpecToolSpec, ...] = (
    ModelSpecToolSpec(
        name="search_literature",
        description="Search for empirical literature about effect sizes for model parameters.",
        input_schema=SearchLiteratureInput,
        public_impl=execute_public_search_literature,
    ),
    ModelSpecToolSpec(
        name="submit_statistical_model_spec",
        description="Submit the full model-spec StatisticalModelSpec for compile-only locking and validation.",
        input_schema=SubmitStatisticalModelSpecInput,
        public_impl=execute_public_submit_statistical_model_spec,
    ),
    ModelSpecToolSpec(
        name="submit_priors",
        description="Submit model-spec prior proposals for schema, compile, and prior-predictive validation.",
        input_schema=SubmitPriorsInput,
        public_impl=execute_public_submit_priors,
    ),
)


def build_model_spec_public_tool_contracts() -> list[ToolContract]:
    """Materialize model-spec public tool contracts from the shared registry."""
    from nof1_causal_lab.flows.contracts_base import ToolContract

    return [
        ToolContract(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
        )
        for spec in MODEL_SPEC_TOOL_SPECS
        if spec.input_schema is not None
    ]
