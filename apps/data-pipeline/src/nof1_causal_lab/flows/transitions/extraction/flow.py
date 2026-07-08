"""extraction: Hybrid computed/semantic extraction entrypoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

import polars as pl

from nof1_causal_lab.flows.runtime_events import (
    emit_extraction_plan_event,
    emit_extraction_worker_event,
)
from nof1_causal_lab.flows.transitions.extraction.execution import (
    collect_worker_results,
    run_with_retries,
)
from nof1_causal_lab.flows.transitions.extraction.planning import (
    chunk_log_label,
    prepare_semantic_chunks,
    project_to_source_columns,
)
from nof1_causal_lab.flows.transitions.extraction.progress import (
    clear_extraction_progress_tracker,
    emit_extraction_snapshot,
    get_extraction_progress_tracker,
    register_extraction_progress_tracker,
)

logger = logging.getLogger(__name__)

SemanticChunkRunner = Callable[..., "Awaitable[tuple[list[dict], list[dict], int, dict | None]]"]

__all__ = [
    "extract_window_chunk",
    "project_to_source_columns",
    "run_extraction",
    "run_extraction_core",
]


async def extract_window_chunk(
    window_text: str,
    window_starts: list[str],
    chunk_idx: int,
    question: str,
    measurement_structure: dict,
    workspace_id: str | None = None,
) -> dict:
    """Extract indicator values from a chunk of support windows.

    Args:
        window_text: Pre-formatted text of support-window events.
        window_starts: Expected support-window starts in this chunk.
        chunk_idx: Index of this chunk (for logging/naming).
        question: The causal research question.
        measurement_structure: MeasurementStructure dict for this chunk.

    Returns:
        Dict with 'dataframe' (as list of dicts for serialization),
        'n_extractions', and 'status'.
    """
    from dataclasses import replace

    from nof1_causal_lab.utils.agent_session import ScopedSessionFactory
    from nof1_causal_lab.utils.config import get_config
    from nof1_causal_lab.workers.core import run_worker_extraction

    config = get_config()
    # extraction workers override only the per-call timeout (prevents hung calls).
    # Everything else inherits the embedded LLM defaults.
    extraction_llm = replace(
        config.extraction_workers.llm,
        timeout=config.extraction_workers.worker_timeout,
    )
    indicator_count = len(measurement_structure.get("indicators", []))
    n_events = window_text.count("\n")
    chunk_label = chunk_log_label(chunk_idx, len(window_starts), n_events)

    logger.info(
        "[%s] Starting extraction with %d windows, %d indicators using model %s (timeout=%ds)",
        chunk_label,
        len(window_starts),
        indicator_count,
        extraction_llm.model,
        extraction_llm.timeout,
    )
    if workspace_id:
        emit_extraction_worker_event(
            workspace_id,
            worker_id=chunk_idx,
            state="running",
            n_windows=len(window_starts),
        )
        tracker = get_extraction_progress_tracker(workspace_id)
        if tracker is not None:
            emit_extraction_snapshot(workspace_id, tracker.mark_running(chunk_idx))

    factory = ScopedSessionFactory(
        extraction_llm,
        config.llm,
        context_id=f"extraction/chunk-{chunk_idx}",
        max_tool_turns=config.extraction_workers.max_tool_turns,
    )

    started_at = perf_counter()
    result = await run_worker_extraction(
        window_text=window_text,
        window_starts=window_starts,
        question=question,
        measurement_structure=measurement_structure,
        session_factory=factory,
        logger=logger,
        call_label=chunk_label,
    )

    elapsed = perf_counter() - started_at
    logger.info(
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
    if factory.accumulated_trace.messages:
        result_dict["llm_trace"] = factory.accumulated_trace.model_dump(mode="json")
    return result_dict


async def _run_semantic_chunks_asyncio(
    *,
    chunk_texts: list[str],
    chunk_window_starts: list[list[str]],
    chunk_contexts: list[dict],
    question: str,
    workspace_id: str | None,
    max_concurrent_workers: int,
    max_rpm: int,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Execute semantic chunks concurrently with a semaphore and retries."""
    from nof1_causal_lab.utils.openrouter_client import RpmLimiter, set_limiter

    all_indices = list(range(len(chunk_texts)))
    all_n_windows = [len(ids) for ids in chunk_window_starts]

    logger.info(
        "extraction: %d semantic chunks of up to %d windows each (max_concurrent_workers=%d, max_rpm=%d)",
        len(chunk_texts),
        max(all_n_windows) if all_n_windows else 0,
        max_concurrent_workers,
        max_rpm,
    )

    if max_rpm:
        set_limiter("llm", RpmLimiter(max_rpm))

    semaphore = asyncio.Semaphore(max_concurrent_workers)

    def _worker(chunk_idx: int) -> Awaitable[dict]:
        async def _attempt() -> dict:
            return await extract_window_chunk(
                chunk_texts[chunk_idx],
                chunk_window_starts[chunk_idx],
                chunk_idx,
                question,
                chunk_contexts[chunk_idx],
                workspace_id=workspace_id,
            )

        async def _bounded() -> dict:
            async with semaphore:
                return await run_with_retries(_attempt)

        return _bounded()

    try:
        if workspace_id:
            emit_extraction_plan_event(
                workspace_id,
                total_workers=len(chunk_texts),
                max_concurrent_workers=max_concurrent_workers,
                max_rpm=max_rpm,
            )
            tracker = register_extraction_progress_tracker(
                workspace_id,
                total_workers=len(chunk_texts),
            )
            emit_extraction_snapshot(workspace_id, tracker.snapshot())
        tasks = [asyncio.ensure_future(_worker(idx)) for idx in all_indices]
        return await collect_worker_results(
            tasks=tasks,
            batch_indices=all_indices,
            batch_n_windows=all_n_windows,
            logger=logger,
            total_chunks=len(chunk_texts),
            workspace_id=workspace_id,
            emit_worker_event=emit_extraction_worker_event,
            get_tracker=get_extraction_progress_tracker,
            emit_snapshot=emit_extraction_snapshot,
        )
    finally:
        if workspace_id:
            clear_extraction_progress_tracker(workspace_id)
        set_limiter("llm", None)


async def run_extraction_core(
    *,
    raw_df: pl.DataFrame,
    question: str,
    measurement_structure: dict,
    extraction_workers: Any,
    workspace_id: str | None = None,
    max_windows: int | None = None,
    semantic_chunk_runner: SemanticChunkRunner | None = None,
) -> dict:
    """Shared extraction extraction helper for the machine runner and evals.

    This is the deterministic orchestration core for:
    1. splitting computed vs semantic indicators
    2. computing direct indicators via Polars
    3. preparing semantic support-window chunks
    4. delegating semantic execution to an injected backend
    5. annotating canonical observation-row support metadata
    """
    from nof1_causal_lab.utils.data import ObservationRecord, annotate_observation_rows

    semantic_chunk_runner = semantic_chunk_runner or _run_semantic_chunks_asyncio

    all_indicators = list(measurement_structure.get("indicators", []))
    time_col = "timestamp"
    model_clock = measurement_structure.get("model_clock", "1d")
    logger.info("extraction: time_col='%s', model_clock='%s'", time_col, model_clock)

    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [
        i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"
    ]
    logger.info(
        "extraction: %d computed + %d semantic indicators",
        len(computed_inds),
        len(semantic_inds),
    )

    computed_dicts: list[dict] = []
    if computed_inds:
        from nof1_causal_lab.utils.aggregations import compute_indicators

        computed_df = compute_indicators(raw_df, computed_inds, model_clock, time_col)
        computed_dicts = computed_df.to_dicts()
        logger.info(
            "extraction: computed %d indicator(s) via Polars (%d rows)",
            len(computed_inds),
            len(computed_df),
        )

    semantic_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    sampled_llm_trace: dict | None = None
    n_semantic_total = 0

    if semantic_inds:
        chunk_texts, chunk_window_starts, chunk_contexts = prepare_semantic_chunks(
            raw_df=raw_df,
            semantic_inds=semantic_inds,
            measurement_structure=measurement_structure,
            model_clock=model_clock,
            time_col=time_col,
            windows_per_chunk=extraction_workers.windows_per_chunk,
            max_events_per_window=extraction_workers.max_events_per_window,
            max_windows=max_windows,
        )

        if chunk_texts:
            (
                semantic_dicts,
                worker_statuses,
                n_semantic_total,
                sampled_llm_trace,
            ) = await semantic_chunk_runner(
                chunk_texts=chunk_texts,
                chunk_window_starts=chunk_window_starts,
                chunk_contexts=chunk_contexts,
                question=question,
                workspace_id=workspace_id,
                max_concurrent_workers=extraction_workers.max_concurrent_workers,
                max_rpm=extraction_workers.max_rpm,
            )

    all_dicts = computed_dicts + semantic_dicts
    n_total = len(computed_dicts) + n_semantic_total
    logger.info(
        "extraction: %d total extractions (%d computed, %d semantic from %d workers)",
        n_total,
        len(computed_dicts),
        n_semantic_total,
        len(worker_statuses),
    )

    observation_rows = cast(
        "list[ObservationRecord]",
        annotate_observation_rows(pl.DataFrame(all_dicts), measurement_structure).to_dicts(),
    )

    result = {
        "observation_rows": observation_rows,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
    if sampled_llm_trace is not None:
        result["llm_trace"] = sampled_llm_trace
    return result


async def run_extraction(
    raw_df: pl.DataFrame,
    question: str,
    measurement_structure: dict,
    workspace_id: str | None = None,
    max_windows: int | None = None,
) -> dict:
    """extraction: Extract indicator values via hybrid computed/semantic paths.

    1. Splits indicators by extraction_mode (computed vs semantic)
    2. Computed path: direct Polars aggregation (~50ms)
    3. Semantic path: support-window LLM workers (asyncio fan-out)
    4. Merges results into canonical observation rows with support metadata
    """
    from nof1_causal_lab.utils.config import get_config

    config = get_config()
    return await run_extraction_core(
        raw_df=raw_df,
        question=question,
        measurement_structure=measurement_structure,
        extraction_workers=config.extraction_workers,
        workspace_id=workspace_id,
        max_windows=max_windows,
        semantic_chunk_runner=_run_semantic_chunks_asyncio,
    )
