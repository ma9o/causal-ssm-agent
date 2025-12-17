"""Worker agents using Inspect AI with OpenRouter."""

import asyncio
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
from causal_agent.utils.llm import make_validate_worker_output_tool, multi_turn_generate, parse_date, parse_json_response
from .prompts import WORKER_SYSTEM, WORKER_USER
from .schemas import WorkerOutput, validate_worker_output

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

    Shows: name, how_to_measure, dtype, role, temporal_status, causal_granularity
    """
    dimensions = schema.get("dimensions", [])
    lines = []
    for dim in dimensions:
        # Skip latent dimensions - workers only extract observed variables
        if dim.get("observability") == "latent":
            continue
        name = dim.get("name", "unknown")
        how_to_measure = dim.get("how_to_measure", "")
        dtype = dim.get("measurement_dtype", "")
        role = dim.get("role", "")
        temporal = dim.get("temporal_status", "")
        granularity = dim.get("causal_granularity", "")

        # Build info string
        info_parts = [dtype, role, temporal]
        if granularity:
            info_parts.append(granularity)
        info = ", ".join(info_parts)

        lines.append(f"- {name}: {how_to_measure} ({info})")
    return "\n".join(lines)


def _get_observed_dimension_dtypes(schema: dict) -> dict[str, str]:
    """Get mapping of observed dimension names to their expected dtypes."""
    dimensions = schema.get("dimensions", [])
    return {
        dim.get("name"): dim.get("measurement_dtype")
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

    # Create validation tool bound to this schema
    validation_tool = make_validate_worker_output_tool(schema)

    # Generate with tools available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        tools=[validation_tool, parse_date()],
    )
    data = parse_json_response(completion)

    # Final validation (should pass if LLM used the tool correctly)
    output, errors = validate_worker_output(data, schema)
    if errors:
        # Fallback to Pydantic validation for error message
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
