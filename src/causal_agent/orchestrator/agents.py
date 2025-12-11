"""Orchestrator agents using Inspect AI with OpenRouter."""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from .prompts import (
    STRUCTURE_PROPOSER_SYSTEM,
    STRUCTURE_PROPOSER_USER,
    STRUCTURE_REVIEW_REQUEST,
)
from .schemas import DSEMStructure

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


def _build_json_schema() -> dict:
    """Build JSON schema from Pydantic model for structured output."""
    return DSEMStructure.model_json_schema()


def _parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e


async def propose_structure_async(
    question: str,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Use the orchestrator LLM to propose a causal model structure.

    Two-stage process:
    1. Initial proposal: Generate structure from question and data
    2. Self-review: Double-check measurement_dtype, aggregation, and how_to_measure

    Args:
        question: The causal research question (natural language)
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        DSEMStructure as a dictionary
    """
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    # Format the chunks for the prompt
    chunks_text = "\n".join(data_sample)

    # Build conversation as a list we'll extend
    messages = [
        ChatMessageSystem(content=STRUCTURE_PROPOSER_SYSTEM),
        ChatMessageUser(
            content=STRUCTURE_PROPOSER_USER.format(
                question=question,
                dataset_summary=dataset_summary or "Not provided",
                chunks=chunks_text,
            )
        ),
    ]

    # Stage 1: Initial proposal
    proposal_response = await model.generate(messages)
    proposal_data = _parse_json_response(proposal_response.completion)

    # Validate initial proposal
    initial_structure = DSEMStructure.model_validate(proposal_data)

    # Append assistant response to conversation history
    messages.append(ChatMessageAssistant(content=proposal_response.completion))

    # Stage 2: Self-review (continues the conversation)
    messages.append(ChatMessageUser(content=STRUCTURE_REVIEW_REQUEST))

    review_response = await model.generate(messages)
    reviewed_data = _parse_json_response(review_response.completion)

    # Validate reviewed structure
    reviewed_structure = DSEMStructure.model_validate(reviewed_data)

    return reviewed_structure.model_dump(by_alias=True)


def propose_structure(
    question: str,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Synchronous wrapper for propose_structure_async.

    Args:
        question: The causal research question
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        DSEMStructure as a dictionary
    """
    import asyncio

    return asyncio.run(propose_structure_async(question, data_sample, dataset_summary))
