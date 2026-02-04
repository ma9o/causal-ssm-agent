"""Stage 2: Indicator Extraction (Workers).

Workers process chunks in parallel to extract raw indicator values.
Each worker returns a Polars DataFrame with (indicator, value, timestamp) tuples.

This is the "E" (Extract) in ETL. Transformation (aggregation) happens in Stage 3.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.utils.data import (
    get_worker_chunk_size,
)
from dsem_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
)
from dsem_agent.workers.agents import WorkerResult, process_chunk


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
