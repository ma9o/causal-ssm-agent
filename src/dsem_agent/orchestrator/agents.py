"""Orchestrator agents using Inspect AI with OpenRouter.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model (Stage 1a) - theoretical constructs + causal edges, NO DATA
2. Measurement Model (Stage 1b) - operationalize constructs into indicators, WITH DATA
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from dsem_agent.utils.config import get_config
from dsem_agent.utils.llm import (
    make_validate_measurement_model_tool,
    multi_turn_generate,
    parse_json_response,
    validate_latent_model_tool,
)
from .prompts import (
    LATENT_MODEL_SYSTEM,
    LATENT_MODEL_USER,
    LATENT_MODEL_REVIEW,
    MEASUREMENT_MODEL_SYSTEM,
    MEASUREMENT_MODEL_USER,
    MEASUREMENT_MODEL_REVIEW,
    PROXY_REQUEST_SYSTEM,
    PROXY_REQUEST_USER,
)
from .schemas import DSEMModel, MeasurementModel, LatentModel

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


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
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    messages = [
        ChatMessageSystem(content=LATENT_MODEL_SYSTEM),
        ChatMessageUser(content=LATENT_MODEL_USER.format(question=question)),
    ]

    # Run multi-turn: initial proposal + self-review, with validation tool available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        follow_ups=[LATENT_MODEL_REVIEW],
        tools=[validate_latent_model_tool()],
    )

    # Parse and validate final result
    data = parse_json_response(completion)
    latent_model = LatentModel.model_validate(data)

    return latent_model.model_dump()


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
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    # Parse latent model for validation tool
    latent = LatentModel.model_validate(latent_model)

    # Format the chunks for the prompt
    chunks_text = "\n".join(data_sample)

    messages = [
        ChatMessageSystem(content=MEASUREMENT_MODEL_SYSTEM),
        ChatMessageUser(
            content=MEASUREMENT_MODEL_USER.format(
                question=question,
                latent_model_json=json.dumps(latent_model, indent=2),
                dataset_summary=dataset_summary or "Not provided",
                chunks=chunks_text,
            )
        ),
    ]

    # Run multi-turn: initial proposal + self-review, with validation tool available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        follow_ups=[MEASUREMENT_MODEL_REVIEW],
        tools=[make_validate_measurement_model_tool(latent)],
    )

    # Parse and validate final result
    data = parse_json_response(completion)
    measurement_model = MeasurementModel.model_validate(data)

    return measurement_model.model_dump()


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
# COMBINED: FULL DSEM MODEL
# ══════════════════════════════════════════════════════════════════════════════


def build_dsem_model(latent_model: dict, measurement_model: dict) -> dict:
    """
    Combine latent and measurement models into a full DSEMModel.

    Args:
        latent_model: The latent model dict from Stage 1a
        measurement_model: The measurement model dict from Stage 1b

    Returns:
        DSEMModel as a dictionary
    """
    dsem = DSEMModel(
        latent=LatentModel.model_validate(latent_model),
        measurement=MeasurementModel.model_validate(measurement_model),
    )
    return dsem.model_dump()


# ══════════════════════════════════════════════════════════════════════════════
# PROXY REQUEST: Fix non-identifiable effects
# ══════════════════════════════════════════════════════════════════════════════


async def request_proxy_measurements_async(
    question: str,
    latent_model: dict,
    current_measurement: dict,
    blocking_info: str,
    confounders_to_operationalize: list[str],
    data_sample: list[str],
) -> dict:
    """
    Request proxy measurements for specific blocking confounders.

    Args:
        question: The causal research question
        latent_model: The latent model dict
        current_measurement: Current measurement model
        blocking_info: Description of which effects are blocked
        confounders_to_operationalize: List of confounder names needing proxies
        data_sample: Sample chunks from the dataset

    Returns:
        Dict with 'new_proxies' and 'unfeasible_confounders'
    """
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    # Format the data sample
    data_text = "\n".join(data_sample[:5])  # Limit to first 5 chunks

    messages = [
        ChatMessageSystem(content=PROXY_REQUEST_SYSTEM),
        ChatMessageUser(
            content=PROXY_REQUEST_USER.format(
                blocking_info=blocking_info,
                confounders_to_operationalize=", ".join(confounders_to_operationalize),
                latent_model_json=json.dumps(latent_model, indent=2),
                current_measurements_json=json.dumps(current_measurement, indent=2),
                data_sample=data_text,
            )
        ),
    ]

    # Single-turn generation for proxy request
    response = await model.generate(messages)

    # Parse response
    data = parse_json_response(response.completion)

    return data


def request_proxy_measurements(
    question: str,
    latent_model: dict,
    current_measurement: dict,
    blocking_info: str,
    confounders_to_operationalize: list[str],
    data_sample: list[str],
) -> dict:
    """
    Synchronous wrapper for request_proxy_measurements_async.

    Args:
        question: The causal research question
        latent_model: The latent model dict
        current_measurement: Current measurement model
        blocking_info: Description of which effects are blocked
        confounders_to_operationalize: List of confounder names needing proxies
        data_sample: Sample chunks from the dataset

    Returns:
        Dict with 'new_proxies' and 'unfeasible_confounders'
    """
    import asyncio

    return asyncio.run(
        request_proxy_measurements_async(
            question,
            latent_model,
            current_measurement,
            blocking_info,
            confounders_to_operationalize,
            data_sample,
        )
    )
