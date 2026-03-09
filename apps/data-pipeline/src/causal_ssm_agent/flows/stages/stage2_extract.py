"""Stage 2: Worker Extraction (DataFrame chunks).

Workers process DataFrame chunks in parallel to extract indicator values
according to the measurement model. Each worker receives a slice of rows
from the raw DataFrame + the causal spec, and uses an LLM to semantically
extract indicator measurements.

Uses task.map() for fan-out + Prefect concurrency() to cap parallel API calls.
"""

import logging

import polars as pl
from prefect import flow, task
from prefect.concurrency.asyncio import concurrency
from prefect.task_runners import ThreadPoolTaskRunner

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT_NAME = "stage2-api"


@task(
    retries=2,
    retry_delay_seconds=10,
    task_run_name="extract-chunk-{chunk_idx}",
)
async def extract_chunk_task(
    chunk_df: pl.DataFrame,
    chunk_idx: int,  # noqa: ARG001  used in task_run_name template
    question: str,
    causal_spec: dict,
) -> dict:
    """Extract indicator values from a single DataFrame chunk.

    Uses a global concurrency limit to avoid flooding the LLM API.

    Args:
        chunk_df: Slice of the raw DataFrame to process.
        chunk_idx: Index of this chunk (for logging/naming).
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.

    Returns:
        Dict with 'dataframe' (as list of dicts for serialization),
        'n_extractions', and 'status'.
    """
    from inspect_ai.model import get_model

    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import make_worker_generate_fn
    from causal_ssm_agent.workers.core import run_worker_extraction

    async with concurrency(CONCURRENCY_LIMIT_NAME, occupy=1, strict=True):
        config = get_config()
        model = get_model(config.stage2_workers.model)
        generate = make_worker_generate_fn(model)

        result = await run_worker_extraction(
            chunk_df=chunk_df,
            question=question,
            causal_spec=causal_spec,
            generate=generate,
        )

        return {
            "dataframe": result.dataframe.to_dicts(),
            "n_extractions": len(result.output.extractions),
            "status": "completed",
        }


@flow(
    name="stage2-worker-extraction",
    log_prints=True,
    persist_result=True,
    result_serializer="json",
    task_runner=ThreadPoolTaskRunner(max_workers=20),
)
async def stage2_extraction_flow(
    raw_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
    chunk_size: int | None = None,
) -> dict:
    """Stage 2: Extract indicator values from raw DataFrame via parallel workers.

    1. Chunks the raw DataFrame into slices of `chunk_size` rows
    2. Fans out extraction tasks via task.map()
    3. Concurrency limit caps parallel LLM calls
    4. Collects and concatenates results

    Args:
        raw_df: Raw wide-format DataFrame from Stage 0 ingestion.
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.
        chunk_size: Rows per chunk (default: from config).

    Returns:
        Dict with 'raw_data' (long-format DataFrame as list of dicts),
        'worker_statuses', and 'n_total_extractions'.
    """
    from prefect.client.orchestration import get_client
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.data import chunk_dataframe

    config = get_config()
    if chunk_size is None:
        chunk_size = config.stage2_workers.chunk_size
    max_concurrent = config.pipeline.max_concurrent_workers

    # Upsert the global concurrency limit (idempotent)
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            CONCURRENCY_LIMIT_NAME, limit=max_concurrent
        )
    logger.info("Concurrency limit '%s' set to %d", CONCURRENCY_LIMIT_NAME, max_concurrent)

    # Chunk the DataFrame
    chunks = chunk_dataframe(raw_df, chunk_size)
    logger.info("Stage 2: %d chunks of up to %d rows each", len(chunks), chunk_size)

    if not chunks:
        return {
            "raw_data": [],
            "worker_statuses": [],
            "n_total_extractions": 0,
        }

    # Fan out via task.map()
    chunk_indices = list(range(len(chunks)))
    results = extract_chunk_task.map(
        chunks,
        chunk_idx=chunk_indices,
        question=unmapped(question),
        causal_spec=unmapped(causal_spec),
    )

    # Collect results
    all_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    n_total = 0

    for i, future in enumerate(results):
        try:
            result = future.result()
        except Exception as e:
            logger.warning("Chunk %d failed: %s", i, e)
            worker_statuses.append(
                {
                    "worker_id": i,
                    "status": "failed",
                    "n_extractions": 0,
                    "chunk_size": len(chunks[i]),
                    "error": str(e),
                }
            )
            continue

        n_ext = result.get("n_extractions", 0)
        n_total += n_ext
        all_dicts.extend(result.get("dataframe", []))
        worker_statuses.append(
            {
                "worker_id": i,
                "status": result.get("status", "completed"),
                "n_extractions": n_ext,
                "chunk_size": len(chunks[i]),
            }
        )

    logger.info("Stage 2: %d total extractions from %d workers", n_total, len(chunks))

    return {
        "raw_data": all_dicts,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
