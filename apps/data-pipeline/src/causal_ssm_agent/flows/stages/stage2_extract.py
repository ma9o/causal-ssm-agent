"""Stage 2: Hybrid Computed/Semantic Extraction.

Indicators with extraction_mode='computed' are aggregated directly via Polars
(~50ms). Indicators with extraction_mode='semantic' go through LLM workers
that process chunks of model-clock ticks in parallel.

Both paths produce the same (indicator, value, timestamp) schema as Utf8 strings.
Results are merged before returning to the caller.
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


def _chunk_log_label(chunk_idx: int, n_ticks: int, n_events: int) -> str:
    """Build a stable log label for a worker chunk."""
    return f"stage2 chunk={chunk_idx} ticks={n_ticks} events={n_events}"


def _collect_batch_results(
    *,
    futures: Sequence[Any],
    batch_indices: Sequence[int],
    batch_n_ticks: Sequence[int],
    logger: Any,
    completed_before: int,
    total_chunks: int,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Collect mapped worker futures in completion order while preserving output order."""
    batch_meta = {
        future: {
            "worker_id": worker_id,
            "n_ticks": n_ticks,
        }
        for worker_id, n_ticks, future in zip(batch_indices, batch_n_ticks, futures, strict=True)
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
        n_ticks = meta["n_ticks"]
        overall_completed = completed_before + batch_completed

        try:
            result = future.result()
        except Exception as exc:
            logger.warning(
                "Stage 2: worker %d failed (progress=%d/%d, batch=%d/%d, ticks=%d): %s",
                worker_id,
                overall_completed,
                total_chunks,
                batch_completed,
                batch_total,
                n_ticks,
                exc,
            )
            statuses_by_worker[worker_id] = {
                "worker_id": worker_id,
                "status": "failed",
                "n_extractions": 0,
                "n_ticks": n_ticks,
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
            "n_ticks": n_ticks,
        }
        logger.info(
            "Stage 2: worker %d %s (progress=%d/%d, batch=%d/%d, ticks=%d, extractions=%d, output_rows=%d)",
            worker_id,
            status,
            overall_completed,
            total_chunks,
            batch_completed,
            batch_total,
            n_ticks,
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
    task_run_name="extract-ticks-{chunk_idx}",
)
async def extract_tick_chunk_task(
    tick_text: str,
    tick_ids: list[str],
    chunk_idx: int,
    question: str,
    causal_spec: dict,
) -> dict:
    """Extract indicator values from a chunk of ticks.

    Args:
        tick_text: Pre-formatted text of tick events.
        tick_ids: Expected tick IDs in this chunk.
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
    n_events = tick_text.count("\n")
    chunk_label = _chunk_log_label(chunk_idx, len(tick_ids), n_events)

    run_logger.info(
        "[%s] Starting extraction with %d ticks, %d indicators using model %s (max_tokens=%d, reasoning_effort=%s)",
        chunk_label,
        len(tick_ids),
        indicator_count,
        config.stage2_workers.model,
        generate_config.max_tokens,
        generate_config.reasoning_effort,
    )

    started_at = perf_counter()
    try:
        result = await run_worker_extraction(
            tick_text=tick_text,
            tick_ids=tick_ids,
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
    chunk_size: int | None = None,  # noqa: ARG001 - kept for Prefect call-site compat
) -> dict:
    """Stage 2: Extract indicator values via hybrid computed/semantic paths.

    1. Splits indicators by extraction_mode (computed vs semantic)
    2. Computed path: direct Polars aggregation (~50ms)
    3. Semantic path: tick-based LLM workers (existing pipeline)
    4. Merges results into unified (indicator, value, timestamp) schema

    Args:
        raw_df_path: Path to the raw wide-format DataFrame from Stage 0 ingestion.
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.
        chunk_size: Deprecated, ignored. Use ticks_per_chunk in config.

    Returns:
        Dict with 'raw_data' (long-format DataFrame as list of dicts),
        'worker_statuses', and 'n_total_extractions'.
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators, make_extraction_context
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.data import detect_time_column

    config = get_config()
    ticks_per_chunk = config.stage2_workers.ticks_per_chunk
    max_events_per_tick = config.stage2_workers.max_events_per_tick
    submission_batch_size = config.stage2_workers.submission_batch_size

    # Load DataFrame and detect time column
    raw_df = pl.read_parquet(Path(raw_df_path))
    all_indicators = get_indicators(causal_spec)
    time_col = detect_time_column(raw_df)
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "1d")
    logger.info("Stage 2: detected time column '%s', model_clock='%s'", time_col, model_clock)

    # Split indicators by extraction mode
    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"]
    logger.info(
        "Stage 2: %d computed + %d semantic indicators",
        len(computed_inds),
        len(semantic_inds),
    )

    # ── Computed path (fast Polars aggregation) ──────────────────────────
    computed_dicts: list[dict] = []
    if computed_inds:
        from causal_ssm_agent.utils.aggregations import compute_indicators

        computed_df = compute_indicators(raw_df, computed_inds, model_clock, time_col)
        computed_dicts = computed_df.to_dicts()
        logger.info(
            "Stage 2: computed %d indicator(s) via Polars (%d rows)",
            len(computed_inds),
            len(computed_df),
        )

    # ── Semantic path (LLM workers) ─────────────────────────────────────
    semantic_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    sampled_llm_trace: dict | None = None
    n_semantic_total = 0

    if semantic_inds:
        from prefect.utilities.annotations import unmapped

        from causal_ssm_agent.utils.data import bucket_by_clock
        from causal_ssm_agent.workers.ticks import chunk_ticks, format_tick_chunk

        # Build extraction context with only semantic indicators
        semantic_spec = {
            **causal_spec,
            "measurement": {**causal_spec.get("measurement", {}), "indicators": semantic_inds},
        }
        extraction_ctx = make_extraction_context(semantic_spec)

        # Project to semantic-only source columns, ensuring time column is kept
        projected = _project_to_source_columns(raw_df, semantic_inds)
        if time_col not in projected.columns:
            projected = projected.with_columns(raw_df[time_col])

        # Bucket by model_clock
        ticks = bucket_by_clock(projected, model_clock, time_col)
        logger.info(
            "Stage 2: bucketed %d rows into %d ticks",
            len(raw_df),
            len(ticks),
        )

        if ticks:
            # Determine display columns (all except time column)
            display_cols = [c for c in projected.columns if c != time_col]

            # Chunk ticks and format text
            chunks = chunk_ticks(ticks, ticks_per_chunk)
            chunk_texts = []
            chunk_tick_ids = []
            for chunk in chunks:
                text = format_tick_chunk(chunk, time_col, display_cols, max_events_per_tick)
                ids = [tick_id for tick_id, _ in chunk]
                chunk_texts.append(text)
                chunk_tick_ids.append(ids)

            logger.info(
                "Stage 2: %d chunks of up to %d ticks each (max_concurrent_workers=%d, submission_batch_size=%d)",
                len(chunks),
                ticks_per_chunk,
                config.stage2_workers.max_concurrent_workers,
                submission_batch_size,
            )

            # Fan out and collect results
            n_finished = 0
            for batch_start in range(0, len(chunks), submission_batch_size):
                batch_end = min(batch_start + submission_batch_size, len(chunks))
                batch_texts = chunk_texts[batch_start:batch_end]
                batch_ids = chunk_tick_ids[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                batch_n_ticks = [len(ids) for ids in batch_ids]

                logger.info(
                    "Stage 2: submitting chunk batch %d-%d (%d tasks, submitted=%d/%d)",
                    batch_indices[0],
                    batch_indices[-1],
                    len(batch_texts),
                    batch_indices[-1] + 1,
                    len(chunks),
                )
                results = extract_tick_chunk_task.map(
                    batch_texts,
                    tick_ids=batch_ids,
                    chunk_idx=batch_indices,
                    question=unmapped(question),
                    causal_spec=unmapped(extraction_ctx),
                )
                batch_rows, batch_statuses, batch_total, batch_trace = _collect_batch_results(
                    futures=results,
                    batch_indices=batch_indices,
                    batch_n_ticks=batch_n_ticks,
                    logger=logger,
                    completed_before=n_finished,
                    total_chunks=len(chunks),
                )
                n_finished += len(batch_statuses)
                n_semantic_total += batch_total
                semantic_dicts.extend(batch_rows)
                worker_statuses.extend(batch_statuses)
                if sampled_llm_trace is None and batch_trace is not None:
                    sampled_llm_trace = batch_trace
                batch_failed = sum(1 for s in batch_statuses if s["status"] == "failed")
                logger.info(
                    "Stage 2: batch %d-%d finished (completed=%d, failed=%d, cumulative_extractions=%d)",
                    batch_indices[0],
                    batch_indices[-1],
                    len(batch_statuses) - batch_failed,
                    batch_failed,
                    n_semantic_total,
                )

    # ── Merge results ───────────────────────────────────────────────────
    all_dicts = computed_dicts + semantic_dicts
    n_total = len(computed_dicts) + n_semantic_total
    logger.info(
        "Stage 2: %d total extractions (%d computed, %d semantic from %d workers)",
        n_total,
        len(computed_dicts),
        n_semantic_total,
        len(worker_statuses),
    )

    result = {
        "raw_data": all_dicts,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
    if sampled_llm_trace is not None:
        result["llm_trace"] = sampled_llm_trace
    return result
