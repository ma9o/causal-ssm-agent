"""Single-source Stage 4 tool registry.

This module centralizes Stage 4 tool metadata across:
- public refinement-tool contracts exposed through the tool server,
- public refinement-tool execution handlers, and
- reducer-owned agentic tool construction plus block-level permissions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from causal_ssm_agent.flows.run_store import load_parquet
from causal_ssm_agent.flows.stages.stage4.tools import (
    make_search_tool,
    make_submit_indicator_choice_tool,
    make_submit_model_configuration_tool,
    make_submit_model_review_tool,
    make_submit_prior_block_tool,
)
from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.data import runs_dir

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stage_contracts import ToolContract
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_session import Stage4Session


Stage4SessionToolFactory = Callable[["Stage4Session", "Stage4SessionToolConfig"], Any]
Stage4PublicToolImpl = Callable[[dict[str, Any], dict[str, Any]], Any]

_COMMON_AGENTIC_PRIOR_BLOCK_KINDS = frozenset(
    {
        "measurement_prior",
        "observation_prior",
        "dynamics_prior",
        "effect_prior",
        "correlation_prior",
    }
)
_ALL_AGENTIC_PRIOR_BLOCK_KINDS = frozenset(
    {*_COMMON_AGENTIC_PRIOR_BLOCK_KINDS, "global_prior_review"}
)


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for empirical literature about effect sizes.")
    parameter_name: str = Field(
        description="Name of the parameter this search is for (e.g. 'beta_stress_sleep')."
    )


class SubmitModelSpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_spec_json: str = Field(
        description=("The JSON string containing the complete ModelSpec to lock for Stage 4."),
    )


class SubmitPriorsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priors_json: str = Field(
        description="The JSON string containing prior proposals keyed by parameter name.",
    )


@dataclass(frozen=True)
class Stage4SessionToolConfig:
    """Runtime config for reducer-owned Stage 4 tool construction."""

    question: str
    enable_literature: bool
    enable_paraphrasing: bool
    n_paraphrases: int
    paraphrase_session_factory: Any  # StageSessionFactory, optional
    max_tool_turns: int


@dataclass(frozen=True)
class Stage4ToolSpec:
    """Single-source metadata for one Stage 4 tool."""

    name: str
    description: str
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    public_impl: Stage4PublicToolImpl | None = None
    session_factory: Stage4SessionToolFactory | None = None
    session_enabled: Callable[[Stage4SessionToolConfig], bool] = field(default=lambda _config: True)
    allowed_block_kinds: frozenset[str] = field(default_factory=frozenset)


def _parse_json_arg(args: dict[str, Any], param_name: str) -> tuple[Any | None, str | None]:
    """Parse one JSON-encoded public tool argument."""
    raw = args.get(param_name, "")
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def _load_stage2_data_for_model(workspace_id: str) -> Any:
    """Load Stage 2 canonical modeling rows for public Stage 4 tool execution."""
    from causal_ssm_agent.flows.run_store import (
        STAGE2_MODEL_PARQUET_FILENAMES,
        find_run_artifact,
    )

    try:
        path = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    except FileNotFoundError as exc:
        raise HTTPException(500, "Stage 2 model data parquet not found") from exc
    return load_parquet(path)


def _load_stage4_current(workspace_id: str) -> dict[str, Any] | None:
    """Load persisted Stage 4 state with draft overlay for refinement accumulation."""
    path = storage.join(runs_dir(workspace_id), "stage-4.json")
    if not storage.exists(path):
        return None
    state = storage.read_json(path)
    draft_path = storage.join(runs_dir(workspace_id), "stage-4-draft.json")
    if storage.exists(draft_path):
        state.update(storage.read_json(draft_path))
    return state


def _execute_stage4_submission(
    ctx: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Execute one public Stage 4 refinement submission."""
    from causal_ssm_agent.flows.stages.stage4.grounding import (
        should_capture_stage4_output,
        stage4_grounding,
    )

    workspace_id = ctx["_workspace_id"]
    stage1b = ctx.get("stage-1b", {})
    causal_spec = stage1b.get("causal_spec", {})
    current = _load_stage4_current(workspace_id)
    data_for_model = _load_stage2_data_for_model(workspace_id)

    grounding_result = stage4_grounding(
        data,
        causal_spec,
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
    from causal_ssm_agent.flows.stages.stage4.tools import search_literature

    query = args.get("query", "")
    if not query:
        return {"result": "Error: query is required"}
    result = await search_literature(query)
    return {"result": result}


execute_public_submit_model_spec = partial(
    _execute_public_json_submission,
    arg_name="model_spec_json",
    payload_key="model_spec",
)
execute_public_submit_priors = partial(
    _execute_public_json_submission,
    arg_name="priors_json",
    payload_key="priors",
)


def _session_tool_factory(factory: Callable[[Stage4Session], Any]) -> Stage4SessionToolFactory:
    """Adapt a session-only Stage 4 tool builder to the shared config signature."""
    return lambda session, _config: factory(session)


def _make_elicit_prior_gmm_session_tool(
    _session: Stage4Session,
    config: Stage4SessionToolConfig,
) -> Any:
    from causal_ssm_agent.flows.stages.stage4.tools import make_elicit_prior_gmm_tool

    return make_elicit_prior_gmm_tool(
        question=config.question,
        paraphrase_session_factory=config.paraphrase_session_factory,
        n_paraphrases=config.n_paraphrases,
    )


def _config_flag(name: str) -> Callable[[Stage4SessionToolConfig], bool]:
    """Return a predicate that reads one boolean flag from the session config."""
    return lambda config: bool(getattr(config, name))


STAGE4_TOOL_SPECS: tuple[Stage4ToolSpec, ...] = (
    Stage4ToolSpec(
        name="submit_indicator_choice",
        description="Submit one distribution/link choice for the active Stage 4 indicator block.",
        session_factory=_session_tool_factory(make_submit_indicator_choice_tool),
        allowed_block_kinds=frozenset({"indicator_decision"}),
    ),
    Stage4ToolSpec(
        name="submit_model_configuration",
        description=(
            "Submit the global initialization, observation-intercept, and "
            "equilibrium-forcing decision."
        ),
        session_factory=_session_tool_factory(make_submit_model_configuration_tool),
        allowed_block_kinds=frozenset({"model_configuration"}),
    ),
    Stage4ToolSpec(
        name="submit_model_review",
        description="Submit the active Stage 4 model-review decision.",
        session_factory=_session_tool_factory(make_submit_model_review_tool),
        allowed_block_kinds=frozenset({"global_review"}),
    ),
    Stage4ToolSpec(
        name="submit_prior_block",
        description="Submit prior proposals for the active Stage 4 prior block only.",
        session_factory=_session_tool_factory(make_submit_prior_block_tool),
        allowed_block_kinds=_ALL_AGENTIC_PRIOR_BLOCK_KINDS,
    ),
    Stage4ToolSpec(
        name="search_literature",
        description="Search for empirical literature about effect sizes for model parameters.",
        input_schema=SearchLiteratureInput,
        public_impl=execute_public_search_literature,
        session_factory=_session_tool_factory(make_search_tool),
        session_enabled=_config_flag("enable_literature"),
        allowed_block_kinds=frozenset({"effect_prior"}),
    ),
    Stage4ToolSpec(
        name="elicit_prior_gmm",
        description=(
            "Run robust paraphrased prior elicitation with GMM aggregation "
            "for a single parameter. Returns an aggregated prior estimate."
        ),
        session_factory=_make_elicit_prior_gmm_session_tool,
        session_enabled=_config_flag("enable_paraphrasing"),
        allowed_block_kinds=_COMMON_AGENTIC_PRIOR_BLOCK_KINDS,
    ),
    Stage4ToolSpec(
        name="submit_model_spec",
        description="Submit the full Stage 4 ModelSpec for compile-only locking and validation.",
        input_schema=SubmitModelSpecInput,
        public_impl=execute_public_submit_model_spec,
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
    from causal_ssm_agent.flows.contracts_base import ToolContract

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


def build_stage4_session_tool_map(
    session: Stage4Session,
    *,
    question: str,
    enable_literature: bool,
    enable_paraphrasing: bool,
    n_paraphrases: int,
    paraphrase_session_factory: Any,
    max_tool_turns: int,
) -> dict[str, Any]:
    """Build the reducer-owned Stage 4 session tool map from the shared registry."""
    config = Stage4SessionToolConfig(
        question=question,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
        n_paraphrases=n_paraphrases,
        paraphrase_session_factory=paraphrase_session_factory,
        max_tool_turns=max_tool_turns,
    )
    tool_map: dict[str, Any] = {}
    for spec in STAGE4_TOOL_SPECS:
        if spec.session_factory is None or not spec.session_enabled(config):
            continue
        tool_map[spec.name] = spec.session_factory(session, config)
    return tool_map


def allowed_stage4_tool_names(block_kind: str) -> tuple[str, ...]:
    """Return the declared reducer-owned tools allowed for one Stage 4 block kind."""
    return tuple(
        spec.name
        for spec in STAGE4_TOOL_SPECS
        if spec.session_factory is not None and block_kind in spec.allowed_block_kinds
    )
