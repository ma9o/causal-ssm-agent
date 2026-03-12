"""Worker agents using LiteLLM-backed runtime clients."""

import asyncio

import polars as pl

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.llm import get_stage2_generate_config, make_generate_fn

from .core import (
    WorkerResult,
    run_worker_extraction,
)

logger = get_prefect_logger(__name__)


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
    config = get_config()
    generate = make_generate_fn(
        config.stage2_workers.model,
        config=get_stage2_generate_config(),
    )
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
