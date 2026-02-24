"""Stage 1b: Measurement Model Proposal (Prefect wrapper).

Wraps the core Stage 1b logic for use in Prefect pipelines.
"""

from inspect_ai.model import get_model
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.agents import build_causal_spec as _build_causal_spec_core
from causal_ssm_agent.orchestrator.stage1b import run_stage1b
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.data import chunk_lines, get_orchestrator_chunk_size
from causal_ssm_agent.utils.llm import (
    attach_trace,
    make_live_trace_path,
    make_orchestrator_generate_fn,
)


@task(cache_policy=INPUTS, result_serializer="json")
def load_orchestrator_chunks(lines: list[str]) -> list[str]:
    """Group preprocessed lines into orchestrator-sized chunks."""
    return chunk_lines(lines, chunk_size=get_orchestrator_chunk_size())


@task(cache_policy=INPUTS, result_serializer="json")
def build_causal_spec(
    latent_model: dict, measurement_model: dict, identifiability_status: dict | None = None
) -> dict:
    """Combine latent and measurement models into full CausalSpec with identifiability."""
    return _build_causal_spec_core(latent_model, measurement_model, identifiability_status)


@task(
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
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        Stage1bData dict matching the web frontend contract.
    """
    model = get_model(get_config().stage1_structure_proposal.model)
    trace_capture: dict = {}
    generate = make_orchestrator_generate_fn(
        model, trace_capture=trace_capture, trace_path=make_live_trace_path("stage-1b")
    )
    result = await run_stage1b(
        question=question,
        latent_model=latent_model,
        chunks=data_sample,
        generate=generate,
        dataset_summary=dataset_summary,
    )
    causal_spec = _build_causal_spec_core(
        latent_model, result.measurement_model, result.identifiability_status
    )
    out: dict = {
        "causal_spec": causal_spec,
        "measurement_model": result.measurement_model,
        "identifiability_status": result.identifiability_status,
        "context": (
            "Stage 1b proposes indicators and checks nonparametric identification "
            "via do-calculus (Pearl/Shpitser-Pearl ID algorithm)."
        ),
    }
    attach_trace(out, trace_capture)
    return out
