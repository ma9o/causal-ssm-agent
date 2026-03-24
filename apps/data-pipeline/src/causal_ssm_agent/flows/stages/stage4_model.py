"""Stage 4: Model Specification & Prior Elicitation (Prefect wrapper).

Thin Prefect wrapper around ``orchestrator.stage4.run_stage4``.
Follows the same two-layer pattern as stages 1a/1b:
- ``orchestrator/stage4.py`` contains the pure agentic logic
- This module manages config, Prefect lifecycle, and materialization
"""

import polars as pl
from prefect import flow

from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.litellm_client import use_openrouter_api_key
from causal_ssm_agent.utils.llm import LLMStageContext

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


@flow(name="stage4-agentic", log_prints=True, persist_result=True, result_serializer="json")
async def stage4_agentic_flow(
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict],
    enable_literature: bool = True,
    openrouter_api_key: str | None = None,
) -> dict:
    """Stage 4 agentic flow: single multi-turn LLM conversation.

    The LLM proposes model spec decisions + all priors in one conversation,
    using tools (search_literature / elicit_prior_gmm) as needed.  The
    grounding tool validates compile + prior predictive on each submission.

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        enable_literature: Whether to offer the search_literature tool

    Returns:
        Full grounded Stage 4 result (same shape as before).
    """
    from causal_ssm_agent.orchestrator.stage4 import run_stage4

    from .stage4_assembly import materialize_stage4_result

    config = get_config()
    s4 = config.stage4_prior_elicitation

    with use_openrouter_api_key(openrouter_api_key):
        async with LLMStageContext("stage-4") as ctx:
            generate = ctx.make_generate(s4.model)

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
