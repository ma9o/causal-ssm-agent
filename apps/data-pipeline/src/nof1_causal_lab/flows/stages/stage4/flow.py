"""Stage 4: Model Specification & Prior Elicitation (Prefect wrapper).

Thin Prefect wrapper around the Stage 4 agent loop and runtime projections.
This module manages config, Prefect lifecycle, and materialization.
"""

import logging
from dataclasses import replace
from pathlib import Path

import polars as pl

from nof1_causal_lab.flows.llm_stage_runtime import (
    LLMStageRuntimeConfig,
    attach_trace,
    build_stage_session_factory,
    open_llm_stage,
)
from nof1_causal_lab.flows.runtime_events import (
    emit_stage4_block_transition_event,
    emit_stage4_graph_event,
    emit_stage4_snapshot_event,
)
from nof1_causal_lab.utils.agent_session import (
    StageSessionFactory,  # noqa: TC001 — runtime-annotated local
)
from nof1_causal_lab.utils.config import get_config, get_secret
from nof1_causal_lab.utils.data import runs_dir
from nof1_causal_lab.utils.llm import get_generate_config
from nof1_causal_lab.utils.openrouter_client import GenerateConfig

logger = logging.getLogger(__name__)


def _stage4_generate_config() -> GenerateConfig:
    """Return the bounded generation config historically used by Stage 4 tests."""
    config = get_generate_config()
    return GenerateConfig(
        max_tokens=None,
        timeout=min(int(config.timeout or 180), 180),
        reasoning_effort=config.reasoning_effort,
        max_tool_output=None,
    )


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
    from nof1_causal_lab.flows.run_store import clear_stage4_checkpoint, save_stage4_checkpoint
    from nof1_causal_lab.flows.stage4_compile_cache import (
        dispatch_stage4_model_compile_warmup,
    )
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_agent_loop import run_stage4
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_megaprompt import (
        run_stage4_megaprompt,
    )
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_runtime_projections import (
        project_stage4_graph,
        project_stage4_snapshot,
    )

    from .assembly import materialize_stage4_result

    config = get_config()
    s4 = config.stage4_prior_elicitation
    literature_enabled = enable_literature and bool(get_secret("EXA_API_KEY"))
    if enable_literature and not literature_enabled:
        logger.warning(
            "search_literature disabled: EXA_API_KEY is not set; tool will not be exposed to Stage 4."
        )

    # Per-turn / per-request timeout is configured in ``config.yaml``
    # (``stage4_prior_elicitation.llm.timeout``) and honoured by each
    # backend. It is not clamped here — callers that need a tighter
    # ceiling for hung provider calls should set it in config.
    stage4_llm = s4.llm

    runtime_config = LLMStageRuntimeConfig(
        stage_id="stage-4",
        stage_llm=stage4_llm,
        llm_defaults=config.llm,
        max_tool_turns=s4.max_tool_turns,
    )
    async with open_llm_stage(
        config=runtime_config,
        openrouter_api_key=openrouter_api_key,
        logger=logger,
    ) as factory:
        paraphrase_factory: StageSessionFactory | None = None
        if s4.paraphrasing.enabled:
            paraphrase_llm = stage4_llm
            if s4.paraphrasing.gmm_model:
                paraphrase_llm = replace(stage4_llm, model=s4.paraphrasing.gmm_model)
            paraphrase_factory = build_stage_session_factory(
                LLMStageRuntimeConfig(
                    stage_id="stage-4/paraphrase",
                    stage_llm=paraphrase_llm,
                    llm_defaults=config.llm,
                    max_tool_turns=s4.max_tool_turns,
                )
            )

        def _on_state_change(plan, runtime, transitions):
            if root_run_id:
                graph = project_stage4_graph(plan)
                emit_stage4_graph_event(root_run_id, graph=graph)
                for transition in transitions:
                    emit_stage4_block_transition_event(root_run_id, transition=transition)
                snapshot = project_stage4_snapshot(plan, runtime)
                emit_stage4_snapshot_event(root_run_id, snapshot=snapshot)

        def _on_model_spec_locked(runtime):
            accepted_model_spec = runtime.domain.accepted.model_spec
            if workspace_id is None or accepted_model_spec is None:
                return
            try:
                dispatch_stage4_model_compile_warmup(
                    workspace_id,
                    accepted_model_spec,
                    causal_spec,
                )
            except Exception:
                logger.exception("Stage 4 compile-cache warmup dispatch failed")

        if s4.state_machine_enabled:
            result = await run_stage4(
                causal_spec=causal_spec,
                question=question,
                data_for_model=data_for_model,
                indicator_audits=indicator_audits,
                session_factory=factory,
                paraphrase_session_factory=paraphrase_factory,
                enable_literature=literature_enabled,
                enable_paraphrasing=s4.paraphrasing.enabled,
                n_paraphrases=s4.paraphrasing.n_paraphrases,
                max_tool_turns=s4.max_tool_turns,
                load_checkpoint=(
                    None
                    if workspace_id is None
                    else lambda: _load_stage4_checkpoint_if_present(workspace_id)
                ),
                save_checkpoint=(
                    None
                    if workspace_id is None
                    else lambda runtime: save_stage4_checkpoint(runtime, workspace_id)
                ),
                clear_checkpoint=(
                    None if workspace_id is None else lambda: clear_stage4_checkpoint(workspace_id)
                ),
                on_model_spec_locked=_on_model_spec_locked,
                on_state_change=_on_state_change if root_run_id else None,
            )
        else:
            logger.info("Stage 4 state machine disabled via config; running megaprompt mode.")
            checkpoint_path = (
                None
                if workspace_id is None
                else Path(runs_dir(workspace_id)) / "stage-4-megaprompt.json"
            )
            result = await run_stage4_megaprompt(
                causal_spec=causal_spec,
                question=question,
                data_for_model=data_for_model,
                indicator_audits=indicator_audits,
                session_factory=factory,
                paraphrase_session_factory=paraphrase_factory,
                enable_literature=literature_enabled,
                enable_paraphrasing=s4.paraphrasing.enabled,
                n_paraphrases=s4.paraphrasing.n_paraphrases,
                max_tool_turns=s4.max_tool_turns,
                max_outer_turns=s4.megaprompt_max_outer_turns,
                checkpoint_path=checkpoint_path,
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
        attach_trace(materialized, factory.accumulated_trace)
        return materialized


def _load_stage4_checkpoint_if_present(workspace_id: str):
    """Load a Stage 4 checkpoint when the cursor exists; otherwise return ``None``.

    Unlike a try/except around :func:`load_stage4_checkpoint`, this only skips
    the load when the cursor is absent — any other I/O or parse error surfaces.
    """
    from nof1_causal_lab.flows.run_store import (
        load_stage4_checkpoint,
        stage4_checkpoint_exists,
    )

    if not stage4_checkpoint_exists(workspace_id):
        return None
    return load_stage4_checkpoint(workspace_id)
