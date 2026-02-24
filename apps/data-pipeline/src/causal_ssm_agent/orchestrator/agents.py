"""Orchestrator agents using Inspect AI with OpenRouter.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model (Stage 1a) - theoretical constructs + causal edges, NO DATA
2. Measurement Model (Stage 1b) - operationalize constructs into indicators, WITH DATA
"""

from inspect_ai.model import get_model

from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn

from .schemas import CausalSpec, LatentModel, MeasurementModel
from .stage1a import run_stage1a
from .stage1b import run_stage1b

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
    model = get_model(get_config().stage1_structure_proposal.model)
    generate = make_orchestrator_generate_fn(model)
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
    import asyncio

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
    model = get_model(get_config().stage1_structure_proposal.model)
    generate = make_orchestrator_generate_fn(model)
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
    import asyncio

    return asyncio.run(
        propose_measurement_model_async(question, latent_model, data_sample, dataset_summary)
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED: FULL CAUSAL SPEC
# ══════════════════════════════════════════════════════════════════════════════


def build_causal_spec(
    latent_model: dict, measurement_model: dict, identifiability_status: dict | None = None
) -> dict:
    """
    Combine latent and measurement models into a full CausalSpec with identifiability.

    Args:
        latent_model: The latent model dict from Stage 1a
        measurement_model: The measurement model dict from Stage 1b
        identifiability_status: Identifiability status dict from Stage 1b

    Returns:
        CausalSpec as a dictionary (includes identifiability key)
    """
    from .schemas import IdentifiabilityStatus

    causal_spec = CausalSpec(
        latent=LatentModel.model_validate(latent_model),
        measurement=MeasurementModel.model_validate(measurement_model),
        identifiability=(
            IdentifiabilityStatus.model_validate(identifiability_status)
            if identifiability_status
            else None
        ),
    )
    return causal_spec.model_dump()
