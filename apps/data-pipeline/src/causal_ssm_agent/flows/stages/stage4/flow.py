"""Stage 4: Model Specification & Prior Elicitation (Prefect wrapper).

Thin Prefect wrapper around the Stage 4 agent loop and runtime projections.
This module manages config, Prefect lifecycle, and materialization.
"""

import polars as pl
from prefect import flow

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.runtime_events import (
    emit_nested_stage_running_event,
    emit_stage4_block_transition_event,
    emit_stage4_graph_event,
    emit_stage4_snapshot_event,
)
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.llm import LLMStageContext, get_generate_config
from causal_ssm_agent.utils.openrouter_client import GenerateConfig, use_openrouter_api_key

logger = get_prefect_logger(__name__)


def _stage4_generate_config() -> GenerateConfig:
    """Return the Stage 4 LLM config.

    Stage 4 intentionally removes the shared max-token cap and tool-output
    truncation so the model can continue beyond the default global ceiling on
    long prior-authoring turns and retain full literature/validator payloads.
    It also enforces a bounded per-request timeout so hung provider calls do
    not stall the whole stage indefinitely.
    """
    base = get_generate_config()
    return GenerateConfig(
        max_tokens=None,
        timeout=180,
        reasoning_effort=base.reasoning_effort,
        max_tool_output=None,
    )


@flow(name="stage4-agentic", log_prints=True, persist_result=True, result_serializer="json")
async def stage4_agentic_flow(
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict],
    enable_literature: bool = True,
    workspace_id: str | None = None,
    openrouter_api_key: str | None = None,
    root_run_id: str | None = None,
) -> dict:
    """Stage 4 LLM flow.

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        enable_literature: Whether to offer the search_literature tool

    Returns:
        Full grounded Stage 4 result (same shape as before).
    """
    from causal_ssm_agent.flows.run_store import clear_stage4_checkpoint, save_stage4_checkpoint
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop import run_stage4
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_navigation import (
        project_stage4_graph,
        project_stage4_snapshot,
    )

    from .assembly import materialize_stage4_result

    if root_run_id:
        emit_nested_stage_running_event(root_run_id, "stage-4")

    config = get_config()
    s4 = config.stage4_prior_elicitation

    with use_openrouter_api_key(openrouter_api_key):
        async with LLMStageContext("stage-4") as ctx:
            generate = ctx.make_generate(
                s4.model,
                config=_stage4_generate_config(),
                max_tool_turns=s4.max_tool_turns,
            )

            def _on_state_change(plan, runtime, transitions):
                if root_run_id:
                    graph = project_stage4_graph(plan)
                    emit_stage4_graph_event(root_run_id, graph=graph)
                    for transition in transitions:
                        emit_stage4_block_transition_event(root_run_id, transition=transition)
                    snapshot = project_stage4_snapshot(plan, runtime)
                    emit_stage4_snapshot_event(root_run_id, snapshot=snapshot)

            result = await run_stage4(
                causal_spec=causal_spec,
                question=question,
                data_for_model=data_for_model,
                indicator_audits=indicator_audits,
                generate=generate,
                enable_literature=enable_literature and s4.literature_search.enabled,
                enable_paraphrasing=s4.paraphrasing.enabled,
                n_paraphrases=s4.paraphrasing.n_paraphrases,
                gmm_model=s4.paraphrasing.gmm_model or s4.model,
                max_tool_turns=s4.max_tool_turns,
                load_checkpoint=(
                    None
                    if workspace_id is None
                    else lambda: _load_stage4_checkpoint_or_none(workspace_id)
                ),
                save_checkpoint=(
                    None
                    if workspace_id is None
                    else lambda runtime: save_stage4_checkpoint(runtime, workspace_id)
                ),
                clear_checkpoint=(
                    None if workspace_id is None else lambda: clear_stage4_checkpoint(workspace_id)
                ),
                on_state_change=_on_state_change if root_run_id else None,
            )

            materialized = materialize_stage4_result(
                model_spec=result.model_spec,
                authored_priors=result.authored_priors,
                data_for_model=data_for_model,
                indicator_audits=indicator_audits,
                causal_spec=causal_spec,
                validation=result.validation,
                search_queries=result.search_queries,
            )
            return ctx.finalize(materialized)


def _load_stage4_checkpoint_or_none(workspace_id: str):
    """Load a Stage 4 checkpoint when present, otherwise return ``None``."""
    from causal_ssm_agent.flows.run_store import load_stage4_checkpoint

    try:
        return load_stage4_checkpoint(workspace_id)
    except FileNotFoundError:
        return None
