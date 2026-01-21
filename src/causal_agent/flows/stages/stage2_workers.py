"""Stage 2: Indicator Population (Workers).

Workers process chunks in parallel to extract indicator values.
Each worker returns a validated Polars dataframe of extractions.
"""

from pathlib import Path

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.utils.aggregations import aggregate_worker_measurements
from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    get_worker_chunk_size,
)
from causal_agent.workers.agents import process_chunk, WorkerResult


@task(cache_policy=INPUTS)
def load_worker_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for workers (stage 2)."""
    return load_text_chunks_util(input_path, chunk_size=get_worker_chunk_size())


@task(
    retries=2,
    retry_delay_seconds=10,
)
def populate_indicators(chunk: str, question: str, dsem_model: dict) -> WorkerResult:
    """Worker extracts indicator values from a chunk.

    Returns:
        WorkerResult containing:
        - output: Validated WorkerOutput with extractions
        - dataframe: Polars DataFrame with columns (indicator, value, timestamp)
    """
    return process_chunk(chunk, question, dsem_model)


@task
def aggregate_measurements(
    worker_results: list[WorkerResult],
    dsem_model: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate worker measurements into time-series DataFrames by granularity.

    Combines all worker extractions and aggregates to causal_granularity:
    1. Concatenates worker DataFrames (indicator, value, timestamp)
    2. Groups indicators by their construct's causal_granularity
    3. Parses timestamps and buckets to each granularity
    4. Applies indicator-specific aggregation (mean, sum, max, etc.)
    5. Returns one DataFrame per granularity

    Args:
        worker_results: List of WorkerResults from parallel workers
        dsem_model: DSEMModel dict (new or old format)

    Returns:
        Dict mapping granularity -> DataFrame. Each DataFrame has 'time_bucket'
        column and indicator columns. Time-invariant indicators are in
        'time_invariant' key as a single-row DataFrame.
    """
    dataframes = [wr.dataframe for wr in worker_results]
    return aggregate_worker_measurements(dataframes, dsem_model)
