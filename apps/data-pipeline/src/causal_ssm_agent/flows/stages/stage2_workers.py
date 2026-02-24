"""Stage 2: Indicator Extraction (Workers).

Workers process chunks in parallel to extract raw indicator values.
Each worker returns a Polars DataFrame with (indicator, value, timestamp) tuples.

This is the "E" (Extract) in ETL. Transformation (aggregation) happens in Stage 3.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.data import chunk_lines, get_worker_chunk_size
from causal_ssm_agent.workers.agents import WorkerResult, process_chunk_async


@task(cache_policy=INPUTS, result_serializer="json")
def load_worker_chunks(lines: list[str]) -> list[str]:
    """Group preprocessed lines into worker-sized chunks."""
    return chunk_lines(lines, chunk_size=get_worker_chunk_size())


@task(
    retries=2,
    retry_delay_seconds=10,
)
async def populate_indicators(chunk: str, question: str, causal_spec: dict) -> WorkerResult:
    """Worker extracts indicator values from a chunk.

    Returns:
        WorkerResult containing:
        - output: Validated WorkerOutput with extractions
        - dataframe: Polars DataFrame with columns (indicator, value, timestamp)
    """
    return await process_chunk_async(chunk, question, causal_spec)
