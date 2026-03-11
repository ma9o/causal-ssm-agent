"""Stage 2: Worker Extraction (DataFrame chunks).

Workers process DataFrame chunks in parallel to extract indicator values
according to the measurement model. Each worker receives a slice of rows
from the raw DataFrame + the causal spec, and uses an LLM to semantically
extract indicator measurements.

Uses task.map() for fan-out with batched submission to avoid overwhelming Prefect.
"""

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
from prefect import flow, get_run_logger, task
from prefect.futures import as_completed

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


def _project_to_source_columns(df: pl.DataFrame, indicators: list[dict]) -> pl.DataFrame:
    """Project DataFrame to only the columns referenced by indicators.

    Keeps the union of all indicator source_columns.  Time/date columns are
    NOT auto-detected — the stage 1b prompt instructs the LLM to include
    them in source_columns when they are needed for temporal context.

    Falls back to the full DataFrame when no source_columns are specified.
    """
    source_cols: set[str] = set()
    for ind in indicators:
        source_cols.update(ind.get("source_columns", []))

    if not source_cols:
        return df

    missing = source_cols - set(df.columns)
    if missing:
        logger.warning(
            "Stage 2: source_columns not found in DataFrame, skipping them: %s",
            sorted(missing),
        )
    keep = [c for c in df.columns if c in source_cols]
    if not keep:
        return df

    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.info(
            "Stage 2: projected %d→%d columns (dropped %d)",
            len(df.columns),
            len(keep),
            dropped,
        )
    return df.select(keep)


def _chunk_log_label(chunk_idx: int, chunk_df: pl.DataFrame) -> str:
    """Build a stable log label for a worker chunk."""
    return f"stage2 chunk={chunk_idx} rows={len(chunk_df)} cols={len(chunk_df.columns)}"


def _collect_batch_results(
    *,
    futures: Sequence[Any],
    batch_indices: Sequence[int],
    batch_chunks: Sequence[pl.DataFrame],
    logger: Any,
    completed_before: int,
    total_chunks: int,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Collect mapped worker futures in completion order while preserving output order."""
    batch_meta = {
        future: {
            "worker_id": worker_id,
            "chunk_size": len(chunk_df),
        }
        for worker_id, chunk_df, future in zip(batch_indices, batch_chunks, futures, strict=True)
    }
    batch_total = len(batch_meta)
    rows_by_worker: dict[int, list[dict]] = {}
    statuses_by_worker: dict[int, dict] = {}
    sampled_trace: dict | None = None
    n_total = 0

    logger.info(
        "Stage 2: waiting for %d worker results (already complete=%d/%d)",
        batch_total,
        completed_before,
        total_chunks,
    )

    for batch_completed, future in enumerate(as_completed(list(batch_meta)), start=1):
        meta = batch_meta[future]
        worker_id = meta["worker_id"]
        chunk_size = meta["chunk_size"]
        overall_completed = completed_before + batch_completed

        try:
            result = future.result()
        except Exception as exc:
            logger.warning(
                "Stage 2: worker %d failed (progress=%d/%d, batch=%d/%d, chunk_size=%d): %s",
                worker_id,
                overall_completed,
                total_chunks,
                batch_completed,
                batch_total,
                chunk_size,
                exc,
            )
            statuses_by_worker[worker_id] = {
                "worker_id": worker_id,
                "status": "failed",
                "n_extractions": 0,
                "chunk_size": chunk_size,
                "error": str(exc),
            }
            continue

        n_ext = result.get("n_extractions", 0)
        output_rows = result.get("dataframe", [])
        status = result.get("status", "completed")
        if sampled_trace is None and "llm_trace" in result:
            sampled_trace = result["llm_trace"]
        n_total += n_ext
        rows_by_worker[worker_id] = output_rows
        statuses_by_worker[worker_id] = {
            "worker_id": worker_id,
            "status": status,
            "n_extractions": n_ext,
            "chunk_size": chunk_size,
        }
        logger.info(
            "Stage 2: worker %d %s (progress=%d/%d, batch=%d/%d, chunk_size=%d, extractions=%d, output_rows=%d)",
            worker_id,
            status,
            overall_completed,
            total_chunks,
            batch_completed,
            batch_total,
            chunk_size,
            n_ext,
            len(output_rows),
        )

    ordered_rows = [row for worker_id in batch_indices for row in rows_by_worker.get(worker_id, [])]
    ordered_statuses = [
        statuses_by_worker[worker_id]
        for worker_id in batch_indices
        if worker_id in statuses_by_worker
    ]
    return ordered_rows, ordered_statuses, n_total, sampled_trace


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
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import StageContext, get_stage2_generate_config
    from causal_ssm_agent.workers.core import run_worker_extraction

    run_logger = get_run_logger()
    config = get_config()
    generate_config = get_stage2_generate_config()
    ctx = StageContext("stage-2", live_trace=False)
    generate = ctx.make_generate(config.stage2_workers.model, config=generate_config)
    indicator_count = len(get_indicators(causal_spec))
    chunk_label = _chunk_log_label(chunk_idx, chunk_df)

    run_logger.info(
        "[%s] Starting extraction with %d rows, %d indicators using model %s (max_tokens=%d, reasoning_effort=%s)",
        chunk_label,
        len(chunk_df),
        indicator_count,
        config.stage2_workers.model,
        generate_config.max_tokens,
        generate_config.reasoning_effort,
    )

    started_at = perf_counter()
    try:
        result = await run_worker_extraction(
            chunk_df=chunk_df,
            question=question,
            causal_spec=causal_spec,
            generate=generate,
            logger=run_logger,
            call_label=chunk_label,
        )
    except Exception:
        run_logger.exception(
            "[%s] Failed after %.1fs",
            chunk_label,
            perf_counter() - started_at,
        )
        raise

    elapsed = perf_counter() - started_at
    run_logger.info(
        "[%s] Finished in %.1fs with %d extractions and %d output rows",
        chunk_label,
        elapsed,
        len(result.output.extractions),
        result.dataframe.height,
    )

    result_dict: dict = {
        "dataframe": result.dataframe.to_dicts(),
        "n_extractions": len(result.output.extractions),
        "status": "completed",
    }
    return ctx.finalize(result_dict)


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

    from causal_ssm_agent.utils.causal_spec import get_indicators, make_extraction_context
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.data import chunk_dataframe

    config = get_config()
    # Pre-extract only what workers need (indicators + outcome) from the full
    # CausalSpec so each parallel task doesn't carry edges/all constructs.
    extraction_ctx = make_extraction_context(causal_spec)
    if chunk_size is None:
        chunk_size = config.stage2_workers.chunk_size
    submission_batch_size = config.stage2_workers.submission_batch_size
    raw_df = pl.read_parquet(Path(raw_df_path))

    # Project DataFrame to only the columns workers need (source_columns + time)
    raw_df = _project_to_source_columns(raw_df, get_indicators(extraction_ctx))

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
    sampled_llm_trace: dict | None = None
    n_total = 0
    n_finished = 0

    # Batch mapped task creation so Prefect is not asked to register thousands
    # of task runs in a single burst for large datasets.
    for batch_start in range(0, len(chunks), submission_batch_size):
        batch_chunks = chunks[batch_start : batch_start + submission_batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_chunks)))
        logger.info(
            "Stage 2: submitting chunk batch %d-%d (%d tasks, submitted=%d/%d)",
            batch_indices[0],
            batch_indices[-1],
            len(batch_chunks),
            batch_indices[-1] + 1,
            len(chunks),
        )
        results = extract_chunk_task.map(
            batch_chunks,
            chunk_idx=batch_indices,
            question=unmapped(question),
            causal_spec=unmapped(extraction_ctx),
        )
        batch_rows, batch_statuses, batch_total, batch_trace = _collect_batch_results(
            futures=results,
            batch_indices=batch_indices,
            batch_chunks=batch_chunks,
            logger=logger,
            completed_before=n_finished,
            total_chunks=len(chunks),
        )
        n_finished += len(batch_statuses)
        n_total += batch_total
        all_dicts.extend(batch_rows)
        worker_statuses.extend(batch_statuses)
        if sampled_llm_trace is None and batch_trace is not None:
            sampled_llm_trace = batch_trace
        batch_failed = sum(1 for status in batch_statuses if status["status"] == "failed")
        logger.info(
            "Stage 2: batch %d-%d finished (completed=%d, failed=%d, cumulative_extractions=%d)",
            batch_indices[0],
            batch_indices[-1],
            len(batch_statuses) - batch_failed,
            batch_failed,
            n_total,
        )

    logger.info("Stage 2: %d total extractions from %d workers", n_total, len(chunks))

    result = {
        "raw_data": all_dicts,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
    if sampled_llm_trace is not None:
        result["llm_trace"] = sampled_llm_trace
    return result
