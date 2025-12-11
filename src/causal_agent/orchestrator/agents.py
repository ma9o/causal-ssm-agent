"""Orchestrator agents using Inspect AI with OpenRouter."""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from .prompts import STRUCTURE_PROPOSER_SYSTEM, STRUCTURE_PROPOSER_USER
from .schemas import DSEMStructure

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


def _build_json_schema() -> dict:
    """Build JSON schema from Pydantic model for structured output."""
    return DSEMStructure.model_json_schema()


async def propose_structure_async(
    question: str,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Use the orchestrator LLM to propose a causal model structure.

    Args:
        question: The causal research question (natural language)
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        DSEMStructure as a dictionary
    """
    model_name = get_config().structure_proposal.model
    model = get_model(model_name)

    # Format the chunks for the prompt - show them as they appear in the data
    chunks_text = "\n".join(data_sample)

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

    response = await model.generate(messages)
    content = response.completion

    # Parse and validate the response
    # Handle markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        # Log the problematic content for debugging
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e

    structure = DSEMStructure.model_validate(data)

    return structure.model_dump()


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
