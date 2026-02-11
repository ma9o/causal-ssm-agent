"""Worker agents using Inspect AI with OpenRouter."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import get_model

from dsem_agent.utils.config import get_config
from dsem_agent.utils.llm import make_worker_generate_fn

from .core import (
    WorkerResult,
    run_worker_extraction,
)

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


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
        dsem_model: The DSEMModel dict

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    model = get_model(get_config().stage2_workers.model)
    generate = make_worker_generate_fn(model)
    return await run_worker_extraction(
        chunk=chunk,
        question=question,
        dsem_model=dsem_model,
        generate=generate,
    )


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
        dsem_model: The DSEMModel dict

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
        dsem_model: The DSEMModel dict

    Returns:
        List of WorkerResults
    """
    model = get_model(get_config().stage2_workers.model)
    generate = make_worker_generate_fn(model)
    tasks = [run_worker_extraction(chunk, question, dsem_model, generate) for chunk in chunks]

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
        dsem_model: The DSEMModel dict

    Returns:
        List of WorkerResults
    """
    return asyncio.run(process_chunks_async(chunks, question, dsem_model))
