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
from prefect.events import emit_event
from prefect.futures import as_completed

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)

WORKER_EVENT_PREFIX = "causal-ssm.worker"
MAX_FREE_TICKS = 100


def _emit_worker_event(
    root_run_id: str,
    *,
    worker_id: int,
    status: str,
    total_workers: int,
    completed_count: int,
    n_ticks: int,
    n_extractions: int | None = None,
    n_llm_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Emit a worker progress event on the root flow run resource."""
    payload: dict[str, Any] = {
        "stage_id": "stage-2",
        "worker_id": worker_id,
        "status": status,
        "n_ticks": n_ticks,
        "total_workers": total_workers,
        "completed_count": completed_count,
    }
    if n_extractions is not None:
        payload["n_extractions"] = n_extractions
    if n_llm_calls is not None:
        payload["n_llm_calls"] = n_llm_calls
    if error is not None:
        payload["error"] = error
    emit_event(
        event=f"{WORKER_EVENT_PREFIX}.{status}",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{root_run_id}",
            "prefect.resource.name": root_run_id,
        },
        payload=payload,
    )


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
    root_run_id: str | None = None,
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
            if root_run_id:
                _emit_worker_event(
                    root_run_id,
                    worker_id=worker_id,
                    status="failed",
                    total_workers=total_chunks,
                    completed_count=overall_completed,
                    n_ticks=n_ticks,
                    error=str(exc),
                )
            continue

        n_ext = result.get("n_extractions", 0)
        output_rows = result.get("dataframe", [])
        status = result.get("status", "completed")
        llm_trace = result.get("llm_trace")
        if sampled_trace is None and llm_trace is not None:
            sampled_trace = llm_trace
        # Count LLM API calls from the trace (each assistant message = 1 call)
        worker_llm_calls = 0
        if llm_trace and isinstance(llm_trace, dict):
            worker_llm_calls = sum(
                1 for m in llm_trace.get("messages", []) if m.get("role") == "assistant"
            )
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
        if root_run_id:
            _emit_worker_event(
                root_run_id,
                worker_id=worker_id,
                status="completed",
                total_workers=total_chunks,
                completed_count=overall_completed,
                n_ticks=n_ticks,
                n_extractions=n_ext,
                n_llm_calls=worker_llm_calls or None,
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
    timeout_seconds=300,
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
    from causal_ssm_agent.utils.litellm_client import GenerateConfig
    from causal_ssm_agent.utils.llm import LLMStageContext, get_generate_config
    from causal_ssm_agent.workers.core import run_worker_extraction

    run_logger = get_run_logger()
    config = get_config()
    generate_config = get_generate_config()
    # Use shorter timeout for extraction workers (prevents hung LLM calls)
    worker_timeout = getattr(
        config.stage2_workers, "worker_timeout", getattr(generate_config, "timeout", None)
    )
    generate_config = GenerateConfig(
        max_tokens=generate_config.max_tokens,
        timeout=worker_timeout,
        reasoning_effort=generate_config.reasoning_effort,
        reasoning_history=generate_config.reasoning_history,
        max_tool_output=generate_config.max_tool_output,
    )
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

    async with LLMStageContext(f"stage-2/chunk-{chunk_idx}") as ctx:
        generate = ctx.make_generate(config.stage2_workers.model, config=generate_config)

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
    root_run_id: str | None = None,
    max_ticks: int | None = None,
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
    from causal_ssm_agent.utils.litellm_client import RpmLimiter, set_limiter

    config = get_config()
    ticks_per_chunk = config.stage2_workers.ticks_per_chunk
    max_events_per_tick = config.stage2_workers.max_events_per_tick

    # Load DataFrame and detect time column
    raw_df = pl.read_parquet(Path(raw_df_path))
    all_indicators = get_indicators(causal_spec)
    time_col = detect_time_column(raw_df)
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "1d")
    logger.info("Stage 2: detected time column '%s', model_clock='%s'", time_col, model_clock)

    # Split indicators by extraction mode
    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [
        i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"
    ]
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

        if max_ticks is not None and len(ticks) > max_ticks:
            logger.warning(
                "Stage 2: free-tier tick cap active — truncating %d ticks to most recent %d",
                len(ticks),
                max_ticks,
            )
            ticks = ticks[-max_ticks:]

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

            all_indices = list(range(len(chunks)))
            all_n_ticks = [len(ids) for ids in chunk_tick_ids]

            logger.info(
                "Stage 2: %d chunks of up to %d ticks each (max_concurrent_workers=%d, max_rpm=%d)",
                len(chunks),
                ticks_per_chunk,
                config.stage2_workers.max_concurrent_workers,
                config.stage2_workers.max_rpm,
            )

            # Activate RPM limiter for the duration of extraction
            if config.stage2_workers.max_rpm:
                set_limiter("llm", RpmLimiter(config.stage2_workers.max_rpm))

            try:
                # Submit ALL chunks at once — the thread pool controls concurrency,
                # the RPM limiter gates individual LLM calls to stay under the limit.
                # No more batch loop: one hung worker cannot block others.
                results = extract_tick_chunk_task.map(
                    chunk_texts,
                    tick_ids=chunk_tick_ids,
                    chunk_idx=all_indices,
                    question=unmapped(question),
                    causal_spec=unmapped(extraction_ctx),
                )
                if root_run_id:
                    for idx, n_t in zip(all_indices, all_n_ticks, strict=True):
                        _emit_worker_event(
                            root_run_id,
                            worker_id=idx,
                            status="submitted",
                            total_workers=len(chunks),
                            completed_count=0,
                            n_ticks=n_t,
                        )
                (
                    semantic_dicts,
                    worker_statuses,
                    n_semantic_total,
                    sampled_llm_trace,
                ) = _collect_batch_results(
                    futures=results,
                    batch_indices=all_indices,
                    batch_n_ticks=all_n_ticks,
                    logger=logger,
                    completed_before=0,
                    total_chunks=len(chunks),
                    root_run_id=root_run_id,
                )
            finally:
                set_limiter("llm", None)

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
