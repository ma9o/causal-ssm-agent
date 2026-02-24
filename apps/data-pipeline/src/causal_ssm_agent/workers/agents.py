"""Worker agents using Inspect AI with OpenRouter."""

import asyncio
import logging

from inspect_ai.model import get_model

from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.llm import make_worker_generate_fn

from .core import (
    WorkerResult,
    run_worker_extraction,
)

logger = logging.getLogger(__name__)


async def process_chunk_async(
    chunk: str,
    question: str,
    causal_spec: dict,
) -> WorkerResult:
    """
    Process a single data chunk against the causal model.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    model = get_model(get_config().stage2_workers.model)
    generate = make_worker_generate_fn(model)
    return await run_worker_extraction(
        chunk=chunk,
        question=question,
        causal_spec=causal_spec,
        generate=generate,
    )


def process_chunk(
    chunk: str,
    question: str,
    causal_spec: dict,
) -> WorkerResult:
    """
    Synchronous wrapper for process_chunk_async.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    return asyncio.run(process_chunk_async(chunk, question, causal_spec))


async def process_chunks_async(
    chunks: list[str],
    question: str,
    causal_spec: dict,
) -> list[WorkerResult]:
    """
    Process multiple chunks in parallel.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        List of WorkerResults
    """
    model = get_model(get_config().stage2_workers.model)
    generate = make_worker_generate_fn(model)
    tasks = [run_worker_extraction(chunk, question, causal_spec, generate) for chunk in chunks]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            logger.warning("Chunk %d failed: %s", i, r)
        else:
            results.append(r)
    return results


def process_chunks(
    chunks: list[str],
    question: str,
    causal_spec: dict,
) -> list[WorkerResult]:
    """
    Synchronous wrapper for process_chunks_async.

    Args:
        chunks: List of data chunks to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        List of WorkerResults
    """
    return asyncio.run(process_chunks_async(chunks, question, causal_spec))
