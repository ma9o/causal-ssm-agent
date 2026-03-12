"""Stage 1a: Latent Model Proposal (Prefect wrapper).

Wraps the core Stage 1a logic for use in Prefect pipelines.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.stage1a import run_stage1a
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.llm import LLMStageContext


@task(
    cache_policy=INPUTS,
    persist_result=True,
    retries=2,
    retry_delay_seconds=30,
)
async def propose_latent_model(question: str) -> dict:
    """Orchestrator proposes theoretical constructs and causal edges (latent model).

    This is Stage 1a - reasoning from domain knowledge only, no data.

    Args:
        question: The causal research question

    Returns:
        Stage1aData dict matching the web frontend contract.
    """
    async with LLMStageContext("stage-1a") as ctx:
        generate = ctx.make_generate(get_config().stage1_structure_proposal.model)
        result = await run_stage1a(question=question, generate=generate)

        return ctx.finalize(
            {
                "latent_model": result.latent_model,
                "outcome_name": result.outcome_name,
                "treatments": result.treatments,
            }
        )
