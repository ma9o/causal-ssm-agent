"""Stage 1a: Latent Model Proposal (Prefect wrapper).

Wraps the core Stage 1a logic for use in Prefect pipelines.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.stage1a import run_stage1a
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.effects import get_all_treatments, get_outcome_from_latent_model
from causal_ssm_agent.utils.llm import StageContext


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
    ctx = StageContext("stage-1a")
    generate = ctx.make_generate(get_config().stage1_structure_proposal.model)
    result = await run_stage1a(question=question, generate=generate)
    latent_model = result.latent_model

    outcome = get_outcome_from_latent_model(latent_model)
    treatments = get_all_treatments(latent_model)

    return ctx.finalize({
        "latent_model": latent_model,
        "outcome_name": outcome or "",
        "treatments": treatments,
    })
