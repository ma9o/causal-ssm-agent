"""Worker agents using Inspect AI with OpenRouter."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from .prompts import WORKER_SYSTEM, WORKER_USER
from .schemas import WorkerOutput

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


@dataclass
class WorkerResult:
    """Result from a worker including both raw output and parsed dataframe."""

    output: WorkerOutput
    dataframe: pl.DataFrame


def _format_dimensions(schema: dict) -> str:
    """Format observable dimensions for the worker prompt.

    Only includes observed dimensions - latent variables are excluded
    since workers shouldn't try to measure them directly.
    """
    dimensions = schema.get("dimensions", [])
    lines = []
    for dim in dimensions:
        # Skip latent dimensions - workers only extract observed variables
        if dim.get("observability") == "latent":
            continue
        name = dim.get("name", "unknown")
        desc = dim.get("description", "")
        dtype = dim.get("base_dtype", "")
        lines.append(f"- {name}: {desc} ({dtype})")
    return "\n".join(lines)


def _get_observed_dimension_dtypes(schema: dict) -> dict[str, str]:
    """Get mapping of observed dimension names to their expected dtypes."""
    dimensions = schema.get("dimensions", [])
    return {
        dim.get("name"): dim.get("base_dtype")
        for dim in dimensions
        if dim.get("observability") == "observed"
    }


def _get_outcome_description(schema: dict) -> str:
    """Get the description of the outcome variable."""
    dimensions = schema.get("dimensions", [])
    for dim in dimensions:
        if dim.get("is_outcome"):
            return dim.get("description", dim.get("name", "outcome"))
    return "Not specified"


async def process_chunk_async(
    chunk: str,
    question: str,
    schema: dict,
) -> WorkerResult:
    """
    Process a single data chunk against the candidate schema.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        schema: The candidate schema from the orchestrator (DSEMStructure as dict)

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    model_name = get_config().stage2_workers.model
    model = get_model(model_name)

    # Format inputs for the prompt
    dimensions_text = _format_dimensions(schema)
    outcome_description = _get_outcome_description(schema)

    messages = [
        ChatMessageSystem(content=WORKER_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                dimensions=dimensions_text,
                chunk=chunk,
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
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse worker response as JSON: {e}") from e

    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()

    return WorkerResult(output=output, dataframe=dataframe)


def process_chunk(
    chunk: str,
    question: str,
    schema: dict,
) -> WorkerResult:
    """
    Synchronous wrapper for process_chunk_async.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        schema: The candidate schema from the orchestrator

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    return asyncio.run(process_chunk_async(chunk, question, schema))


async def process_chunks_async(
    chunks: list[str],
    question: str,
    schema: dict,
) -> list[WorkerResult]:
    """
    Process multiple chunks in parallel.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        schema: The candidate schema from the orchestrator

    Returns:
        List of WorkerResults
    """
    tasks = [
        process_chunk_async(chunk, question, schema)
        for chunk in chunks
    ]

    return await asyncio.gather(*tasks)


def process_chunks(
    chunks: list[str],
    question: str,
    schema: dict,
) -> list[WorkerResult]:
    """
    Synchronous wrapper for process_chunks_async.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        schema: The candidate schema from the orchestrator

    Returns:
        List of WorkerResults
    """
    return asyncio.run(process_chunks_async(chunks, question, schema))
