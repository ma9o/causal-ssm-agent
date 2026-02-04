"""Stage 1b: Measurement Model Proposal (Prefect wrapper).

Wraps the core Stage 1b logic for use in Prefect pipelines.
"""

from pathlib import Path

from inspect_ai.model import get_model
from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.orchestrator.agents import build_dsem_model as build_dsem_model_agent
from dsem_agent.orchestrator.stage1b import run_stage1b
from dsem_agent.utils.config import get_config
from dsem_agent.utils.data import (
    get_orchestrator_chunk_size,
)
from dsem_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
)
from dsem_agent.utils.llm import make_orchestrator_generate_fn


@task(cache_policy=INPUTS)
def load_orchestrator_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for orchestrator (stage 1b)."""
    return load_text_chunks_util(input_path, chunk_size=get_orchestrator_chunk_size())


@task(cache_policy=INPUTS)
def build_dsem_model(
    latent_model: dict, measurement_model: dict, identifiability_status: dict | None = None
) -> dict:
    """Combine latent and measurement models into full DSEMModel with identifiability."""
    dsem = build_dsem_model_agent(latent_model, measurement_model)
    dsem["identifiability"] = identifiability_status
    return dsem


@task(cache_policy=INPUTS)
def propose_measurement_with_identifiability_fix(
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
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        Dict with 'measurement_model' and 'identifiability_status'
    """
    import asyncio

    async def run():
        model = get_model(get_config().stage1_structure_proposal.model)
        generate = make_orchestrator_generate_fn(model)
        result = await run_stage1b(
            question=question,
            latent_model=latent_model,
            chunks=data_sample,
            generate=generate,
            dataset_summary=dataset_summary,
        )
        return {
            "measurement_model": result.measurement_model,
            "identifiability_status": result.identifiability_status,
        }

    return asyncio.run(run())
