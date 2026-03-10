"""Stage 2: Worker Extraction (DataFrame chunks).

Workers process DataFrame chunks in parallel to extract indicator values
according to the measurement model. Each worker receives a slice of rows
from the raw DataFrame + the causal spec, and uses an LLM to semantically
extract indicator measurements.

Uses task.map() for fan-out with batched submission to avoid overwhelming Prefect.
"""

from pathlib import Path
from time import perf_counter

import polars as pl
from prefect import flow, get_run_logger, task

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


@task(
    retries=2,
    retry_delay_seconds=10,
    task_run_name="extract-chunk-{chunk_idx}",
)
async def extract_chunk_task(
    chunk_df: pl.DataFrame,
    chunk_idx: int,
    question: str,
    causal_spec: dict,
) -> dict:
    """Extract indicator values from a single DataFrame chunk.

    The Stage 2 wrapper flow bounds parallel execution via the configured
    task runner on the enclosing subflow invocation.

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

    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import make_worker_generate_fn
    from causal_ssm_agent.workers.core import run_worker_extraction

    run_logger = get_run_logger()
    config = get_config()
    model = get_model(config.stage2_workers.model)
    generate = make_worker_generate_fn(model)
    indicator_count = len(get_indicators(causal_spec))

    run_logger.info(
        "Starting chunk %d with %d rows, %d columns, %d indicators using model %s",
        chunk_idx,
        len(chunk_df),
        len(chunk_df.columns),
        indicator_count,
        config.stage2_workers.model,
    )

    started_at = perf_counter()
    try:
        result = await run_worker_extraction(
            chunk_df=chunk_df,
            question=question,
            causal_spec=causal_spec,
            generate=generate,
            logger=run_logger,
        )
    except Exception:
        run_logger.exception(
            "Chunk %d failed after %.1fs",
            chunk_idx,
            perf_counter() - started_at,
        )
        raise

    elapsed = perf_counter() - started_at
    run_logger.info(
        "Finished chunk %d in %.1fs with %d extractions and %d output rows",
        chunk_idx,
        elapsed,
        len(result.output.extractions),
        result.dataframe.height,
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
)
async def stage2_extraction_flow(
    raw_df_path: str,
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
        raw_df_path: Path to the raw wide-format DataFrame from Stage 0 ingestion.
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.
        chunk_size: Rows per chunk (default: from config).

    Returns:
        Dict with 'raw_data' (long-format DataFrame as list of dicts),
        'worker_statuses', and 'n_total_extractions'.
    """
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.data import chunk_dataframe

    config = get_config()
    if chunk_size is None:
        chunk_size = config.stage2_workers.chunk_size
    submission_batch_size = config.stage2_workers.submission_batch_size
    raw_df = pl.read_parquet(Path(raw_df_path))

    # Chunk the DataFrame
    chunks = chunk_dataframe(raw_df, chunk_size)
    logger.info(
        "Stage 2: %d chunks of up to %d rows each (max_concurrent_workers=%d, submission_batch_size=%d)",
        len(chunks),
        chunk_size,
        config.stage2_workers.max_concurrent_workers,
        submission_batch_size,
    )

    if not chunks:
        return {
            "raw_data": [],
            "worker_statuses": [],
            "n_total_extractions": 0,
        }

    # Collect results
    all_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    n_total = 0

    # Batch mapped task creation so Prefect is not asked to register thousands
    # of task runs in a single burst for large datasets.
    for batch_start in range(0, len(chunks), submission_batch_size):
        batch_chunks = chunks[batch_start : batch_start + submission_batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_chunks)))
        logger.info(
            "Stage 2: submitting chunk batch %d-%d (%d tasks)",
            batch_indices[0],
            batch_indices[-1],
            len(batch_chunks),
        )
        results = extract_chunk_task.map(
            batch_chunks,
            chunk_idx=batch_indices,
            question=unmapped(question),
            causal_spec=unmapped(causal_spec),
        )

        for worker_id, chunk_df, future in zip(batch_indices, batch_chunks, results, strict=True):
            try:
                result = future.result()
            except Exception as e:
                logger.warning("Chunk %d failed: %s", worker_id, e)
                worker_statuses.append(
                    {
                        "worker_id": worker_id,
                        "status": "failed",
                        "n_extractions": 0,
                        "chunk_size": len(chunk_df),
                        "error": str(e),
                    }
                )
                continue

            n_ext = result.get("n_extractions", 0)
            n_total += n_ext
            all_dicts.extend(result.get("dataframe", []))
            logger.info("Chunk %d completed with %d extractions", worker_id, n_ext)
            worker_statuses.append(
                {
                    "worker_id": worker_id,
                    "status": result.get("status", "completed"),
                    "n_extractions": n_ext,
                    "chunk_size": len(chunk_df),
                }
            )

    logger.info("Stage 2: %d total extractions from %d workers", n_total, len(chunks))

    return {
        "raw_data": all_dicts,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
