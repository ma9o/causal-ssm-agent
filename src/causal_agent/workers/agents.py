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
from causal_agent.utils.llm import make_worker_tools, multi_turn_generate, parse_json_response
from .prompts import WORKER_WO_PROPOSALS_SYSTEM, WORKER_USER
from .schemas import WorkerOutput, validate_worker_output

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


@dataclass
class WorkerResult:
    """Result from a worker including both raw output and parsed dataframe."""

    output: WorkerOutput
    dataframe: pl.DataFrame


def _format_indicators(dsem_model: dict) -> str:
    """Format indicators for the worker prompt.

    Supports both new format (structural + measurement) and old format (dimensions).

    Shows: name, dtype, measurement_granularity, how_to_measure
    """
    # New format
    if "measurement" in dsem_model and "indicators" in dsem_model["measurement"]:
        indicators = dsem_model["measurement"]["indicators"]
        lines = []
        for ind in indicators:
            name = ind.get("name", "unknown")
            how_to_measure = ind.get("how_to_measure", "")
            dtype = ind.get("measurement_dtype", "")
            measurement_granularity = ind.get("measurement_granularity", "")

            # Build info string with dtype and measurement_granularity
            info_parts = [dtype]
            if measurement_granularity:
                info_parts.append(f"@{measurement_granularity}")
            info = ", ".join(info_parts)

            lines.append(f"- {name} ({info}): {how_to_measure}")
        return "\n".join(lines)

    # Old format (dimensions)
    if "dimensions" in dsem_model:
        dimensions = dsem_model["dimensions"]
        lines = []
        for dim in dimensions:
            # Skip latent dimensions - workers only extract observed variables
            if dim.get("observability") == "latent":
                continue
            name = dim.get("name", "unknown")
            how_to_measure = dim.get("how_to_measure", "")
            dtype = dim.get("measurement_dtype", "")
            measurement_granularity = dim.get("measurement_granularity", "")

            # Build info string with dtype and measurement_granularity
            info_parts = [dtype]
            if measurement_granularity:
                info_parts.append(f"@{measurement_granularity}")
            info = ", ".join(info_parts)

            lines.append(f"- {name} ({info}): {how_to_measure}")
        return "\n".join(lines)

    return ""


def _get_outcome_description(dsem_model: dict) -> str:
    """Get the description of the outcome variable.

    Supports both new format (structural + measurement) and old format (dimensions).
    """
    # New format
    if "structural" in dsem_model and "constructs" in dsem_model["structural"]:
        constructs = dsem_model["structural"]["constructs"]
        for c in constructs:
            if c.get("is_outcome"):
                return c.get("description", c.get("name", "outcome"))
        return "Not specified"

    # Old format
    if "dimensions" in dsem_model:
        dimensions = dsem_model["dimensions"]
        for dim in dimensions:
            if dim.get("is_outcome"):
                return dim.get("description", dim.get("name", "outcome"))
        return "Not specified"

    return "Not specified"


async def process_chunk_async(
    chunk: str,
    question: str,
    dsem_model: dict,
) -> WorkerResult:
    """
    Process a single data chunk against the DSEM model.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        dsem_model: The DSEMModel dict (new or old format)

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    model_name = get_config().stage2_workers.model
    model = get_model(model_name)

    # Format inputs for the prompt
    indicators_text = _format_indicators(dsem_model)
    outcome_description = _get_outcome_description(dsem_model)

    messages = [
        ChatMessageSystem(content=WORKER_WO_PROPOSALS_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                indicators=indicators_text,
                chunk=chunk,
            )
        ),
    ]

    # Generate with tools available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        tools=make_worker_tools(dsem_model),
    )
    data = parse_json_response(completion)

    # Final validation (should pass if LLM used the tool correctly)
    output, errors = validate_worker_output(data, dsem_model)
    if errors:
        # Fallback to Pydantic validation for error message
        output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()

    return WorkerResult(output=output, dataframe=dataframe)


def process_chunk(
    chunk: str,
    question: str,
    dsem_model: dict,
) -> WorkerResult:
    """
    Synchronous wrapper for process_chunk_async.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        dsem_model: The DSEMModel dict (new or old format)

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    return asyncio.run(process_chunk_async(chunk, question, dsem_model))


async def process_chunks_async(
    chunks: list[str],
    question: str,
    dsem_model: dict,
) -> list[WorkerResult]:
    """
    Process multiple chunks in parallel.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        dsem_model: The DSEMModel dict (new or old format)

    Returns:
        List of WorkerResults
    """
    tasks = [
        process_chunk_async(chunk, question, dsem_model)
        for chunk in chunks
    ]

    return await asyncio.gather(*tasks)


def process_chunks(
    chunks: list[str],
    question: str,
    dsem_model: dict,
) -> list[WorkerResult]:
    """
    Synchronous wrapper for process_chunks_async.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        dsem_model: The DSEMModel dict (new or old format)

    Returns:
        List of WorkerResults
    """
    return asyncio.run(process_chunks_async(chunks, question, dsem_model))
