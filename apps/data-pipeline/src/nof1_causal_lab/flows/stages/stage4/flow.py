"""Stage 4: Model Specification & Prior Elicitation.

Thin wrapper around the gradual construct-by-construct admission loop. Manages
config + the LLM stage runtime, then materializes the grounded result. The model
spec and priors are produced by :func:`run_stage4_construct_build`; the exact
prior-predictive reachability battery gates each construct as it is admitted.
"""

import logging

import polars as pl

from nof1_causal_lab.flows.llm_stage_runtime import (
    LLMStageRuntimeConfig,
    attach_trace,
    open_llm_stage,
)
from nof1_causal_lab.utils.config import get_config, get_secret
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
) -> dict:
    """Stage 4 LLM flow: gradual construct admission → grounded result.

    Args:
        causal_spec: Full CausalSpec dict.
        question: Research question.
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support).
        indicator_audits: Per-indicator audit hints keyed by indicator name.
        enable_literature: Whether to offer the search_literature tool per construct.
        workspace_id: When set, stream construct-admission telemetry for the live web view.

    Returns:
        The full grounded Stage 4 result (``model_spec``, ``authored_priors``,
        ``resolved_priors``, ``_compiled_ssm``, …).
    """
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_construct_flow import (
        run_stage4_construct_build,
    )

    from .assembly import materialize_stage4_result

    config = get_config()
    s4 = config.stage4_prior_elicitation
    literature_enabled = enable_literature and bool(get_secret("EXA_API_KEY"))
    if enable_literature and not literature_enabled:
        logger.warning(
            "search_literature disabled: EXA_API_KEY is not set; tool will not be exposed to Stage 4."
        )

    runtime_config = LLMStageRuntimeConfig(
        stage_id="stage-4",
        stage_llm=s4.llm,
        llm_defaults=config.llm,
        max_tool_turns=s4.max_tool_turns,
    )
    async with open_llm_stage(config=runtime_config, logger=logger) as factory:
        result = await run_stage4_construct_build(
            causal_spec=causal_spec,
            question=question,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            session_factory=factory,
            enable_literature=literature_enabled,
            workspace_id=workspace_id,
        )
        materialized = materialize_stage4_result(
            model_spec=result.model_spec,
            authored_priors=result.authored_priors,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            causal_spec=causal_spec,
            validation=result.validation,
            search_queries=result.search_queries,
            skip_ppc=True,  # reachability battery already gated every construct
        )
        attach_trace(materialized, factory.accumulated_trace)
        return materialized
