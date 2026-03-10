"""Worker agents using LiteLLM-backed runtime clients."""

import asyncio
import logging

import polars as pl

from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.llm import make_worker_generate_fn

from .core import (
    WorkerResult,
    run_worker_extraction,
)

logger = logging.getLogger(__name__)


async def process_chunk_async(
    chunk_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
) -> WorkerResult:
    """
    Process a single DataFrame chunk against the causal model.

    Args:
        chunk_df: A slice of the raw DataFrame to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    generate = make_worker_generate_fn(get_config().stage2_workers.model)
    return await run_worker_extraction(
        chunk_df=chunk_df,
        question=question,
        causal_spec=causal_spec,
        generate=generate,
    )


def process_chunk(
    chunk_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
) -> WorkerResult:
    """
    Synchronous wrapper for process_chunk_async.

    Args:
        chunk_df: A slice of the raw DataFrame to process
        question: The causal research question
        causal_spec: The CausalSpec dict

    Returns:
        WorkerResult with validated output and Polars dataframe
    """
    return asyncio.run(process_chunk_async(chunk_df, question, causal_spec))
