"""Stage 1: Structure Proposal (Orchestrator).

The orchestrator proposes dimensions, autocorrelations, time granularities, and DAG
based on a sample of the data.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.orchestrator.agents import propose_structure as propose_structure_agent
from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    get_orchestrator_chunk_size,
)


@task(cache_policy=INPUTS)
def load_orchestrator_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for orchestrator (stage 1)."""
    return load_text_chunks_util(input_path, chunk_size=get_orchestrator_chunk_size())


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_structure(question: str, data_sample: list[str]) -> dict:
    """Orchestrator proposes dimensions, autocorrelations, time granularities, DAG."""
    return propose_structure_agent(question, data_sample)
