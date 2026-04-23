"""Orchestrator agents using OpenRouter-backed runtime clients.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model (Stage 1a) - theoretical constructs + causal edges, NO DATA
2. Measurement Model (Stage 1b) - operationalize constructs into indicators, WITH DATA
"""

import asyncio

from causal_ssm_agent.flows.stages.stage1a.run import run_stage1a
from causal_ssm_agent.flows.stages.stage1b.run import run_stage1b
from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.llm import make_generate_fn

__all__ = [
    "propose_latent_model",
    "propose_latent_model_async",
    "propose_measurement_model",
    "propose_measurement_model_async",
]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1a: LATENT MODEL (theory-driven, no data)
# ══════════════════════════════════════════════════════════════════════════════


async def propose_latent_model_async(question: str) -> dict:
    """
    Use the orchestrator LLM to propose a theoretical causal structure (latent model).

    This is Stage 1a - the LLM reasons from domain knowledge only, without seeing data.

    Two-step process:
    1. Initial proposal: Generate structure from question
    2. Self-review: Check theoretical coherence

    Args:
        question: The causal research question (natural language)

    Returns:
        LatentModel as a dictionary
    """
    stage1 = get_config().stage1_structure_proposal
    generate = make_generate_fn(stage1.llm.model, max_tool_turns=stage1.stage1a_max_tool_turns)
    result = await run_stage1a(question=question, generate=generate)
    return result.latent_model


def propose_latent_model(question: str) -> dict:
    """
    Synchronous wrapper for propose_latent_model_async.

    Args:
        question: The causal research question

    Returns:
        LatentModel as a dictionary
    """
    return asyncio.run(propose_latent_model_async(question))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1b: MEASUREMENT MODEL (data-driven operationalization)
# ══════════════════════════════════════════════════════════════════════════════


async def propose_measurement_model_async(
    question: str,
    latent_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Use the orchestrator LLM to propose a measurement model for the latent model.

    This is Stage 1b - the LLM sees data and operationalizes constructs into indicators.

    Two-step process:
    1. Initial proposal: Generate indicators from latent model + data
    2. Self-review: Check operationalization coherence

    Args:
        question: The causal research question (natural language)
        latent_model: The latent model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        MeasurementModel as a dictionary
    """
    stage1 = get_config().stage1_structure_proposal
    generate = make_generate_fn(stage1.llm.model, max_tool_turns=stage1.stage1b_max_tool_turns)
    result = await run_stage1b(
        question=question,
        latent_model=latent_model,
        chunks=data_sample,
        generate=generate,
        dataset_summary=dataset_summary,
    )
    return result.measurement_model


def propose_measurement_model(
    question: str,
    latent_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Synchronous wrapper for propose_measurement_model_async.

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        MeasurementModel as a dictionary
    """
    return asyncio.run(
        propose_measurement_model_async(question, latent_model, data_sample, dataset_summary)
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED: FULL CAUSAL SPEC
# ══════════════════════════════════════════════════════════════════════════════
