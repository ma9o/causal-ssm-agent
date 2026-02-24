"""Stage 1a: Latent Model Proposal (Prefect wrapper).

Wraps the core Stage 1a logic for use in Prefect pipelines.
"""

from inspect_ai.model import get_model
from prefect import task

from causal_ssm_agent.orchestrator.stage1a import run_stage1a
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.effects import get_all_treatments, get_outcome_from_latent_model
from causal_ssm_agent.utils.llm import (
    attach_trace,
    make_live_trace_path,
    make_orchestrator_generate_fn,
)


@task(
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
    model = get_model(get_config().stage1_structure_proposal.model)
    trace_capture: dict = {}
    generate = make_orchestrator_generate_fn(
        model, trace_capture=trace_capture, trace_path=make_live_trace_path("stage-1a")
    )
    result = await run_stage1a(question=question, generate=generate)
    latent_model = result.latent_model

    outcome = get_outcome_from_latent_model(latent_model)
    treatments = get_all_treatments(latent_model)

    out: dict = {
        "latent_model": latent_model,
        "outcome_name": outcome or "",
        "treatments": treatments,
        "context": (
            "Stage 1a proposes a latent causal model based on domain knowledge alone. "
            "The model specifies theoretical constructs and their causal relationships."
        ),
    }
    attach_trace(out, trace_capture)
    return out
