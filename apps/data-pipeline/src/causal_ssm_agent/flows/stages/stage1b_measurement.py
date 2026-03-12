"""Stage 1b: Measurement Model Proposal (Prefect wrapper).

Wraps the core Stage 1b logic for use in Prefect pipelines.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.agents import build_causal_spec as _build_causal_spec_core
from causal_ssm_agent.orchestrator.stage1b import run_stage1b
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.llm import LLMStageContext


@task(cache_policy=INPUTS, result_serializer="json")
def build_causal_spec(
    latent_model: dict, measurement_model: dict, identifiability_status: dict | None = None
) -> dict:
    """Combine latent and measurement models into full CausalSpec with identifiability."""
    return _build_causal_spec_core(latent_model, measurement_model, identifiability_status)


@task(
    cache_policy=INPUTS,
    persist_result=True,
    result_serializer="json",
)
async def propose_measurement_with_identifiability_fix(
    question: str,
    latent_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Run Stage 1b: propose measurements and fix identifiability issues.

    This is the Prefect task wrapper around the core Stage 1b logic.

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        data_sample: Dataset schema and sample (formatted by format_schema_for_llm)
        dataset_summary: Brief overview of the full dataset

    Returns:
        Stage1bData dict matching the web frontend contract.
    """
    async with LLMStageContext("stage-1b") as ctx:
        generate = ctx.make_generate(get_config().stage1_structure_proposal.model)
        result = await run_stage1b(
            question=question,
            latent_model=latent_model,
            chunks=data_sample,
            generate=generate,
            dataset_summary=dataset_summary,
        )
        # causal_spec is already built by stage1b_grounding — pass through directly
        return ctx.finalize({"causal_spec": result.causal_spec})
