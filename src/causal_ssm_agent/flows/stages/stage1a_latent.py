"""Stage 1a: Latent Model Proposal (Prefect wrapper).

Wraps the core Stage 1a logic for use in Prefect pipelines.
"""

from inspect_ai.model import get_model
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.stage1a import run_stage1a
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_latent_model(question: str) -> dict:
    """Orchestrator proposes theoretical constructs and causal edges (latent model).

    This is Stage 1a - reasoning from domain knowledge only, no data.

    Args:
        question: The causal research question

    Returns:
        LatentModel as a dictionary with 'constructs' and 'edges'
    """
    import asyncio

    async def run():
        model = get_model(get_config().stage1_structure_proposal.model)
        generate = make_orchestrator_generate_fn(model)
        result = await run_stage1a(question=question, generate=generate)
        return result.latent_model

    return asyncio.run(run())
