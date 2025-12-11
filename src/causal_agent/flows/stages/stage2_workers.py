"""Stage 2: Dimension Population (Workers).

Workers process chunks in parallel to populate dimensions and suggest graph edits.
The orchestrator then performs a 3-way merge of suggestions.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    get_worker_chunk_size,
)
from causal_agent.workers.agents import process_chunk


@task(cache_policy=INPUTS)
def load_worker_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for workers (stage 2)."""
    return load_text_chunks_util(input_path, chunk_size=get_worker_chunk_size())


@task(
    retries=2,
    retry_delay_seconds=10,
    task_run_name="populate-chunk-{chunk_id}",
)
def populate_dimensions(chunk: str, chunk_id: int, schema: dict) -> dict:
    """Workers populate candidate dimensions for each chunk."""
    return process_chunk(chunk, f"chunk_{chunk_id:04d}", schema)


@task(retries=1, cache_policy=INPUTS)
def merge_suggestions(base_schema: dict, worker_outputs: list[dict]) -> dict:
    """Orchestrator performs 3-way merge of worker suggestions.

    TODO: Implement merge logic:
    - Aggregate dimension extractions
    - Reconcile conflicting edge suggestions
    - Evaluate new dimension proposals
    """
    # For now, return base schema unchanged
    return base_schema
