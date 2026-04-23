"""Stage 2: Hybrid Computed/Semantic Extraction.

Indicators with extraction_mode='computed' are aggregated directly via Polars
(~50ms). Indicators with extraction_mode='semantic' go through LLM workers
that process chunks of support windows in parallel.

Both paths first emit one scalar per support window. Stage 2 then annotates the
merged output into canonical observation rows with ``anchor_time``,
``support_start``, and ``support_end``.
"""

import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import polars as pl
from prefect import flow, get_run_logger, task
from prefect.futures import as_completed
from pydantic import ValidationError

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.runtime_events import (
    emit_nested_stage_running_event,
    emit_stage2_plan_event,
    emit_stage2_snapshot_event,
    emit_stage2_worker_event,
)

logger = get_prefect_logger(__name__)

SemanticChunkRunner = Callable[..., Awaitable[tuple[list[dict], list[dict], int, dict | None]]]
_TERMINAL_STAGE2_WORKER_STATES = {"completed", "failed"}
_RECOVERABLE_STAGE2_WORKER_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValidationError,
    ValueError,
)


@dataclass
class _Stage2ProgressTracker:
    total_workers: int
    pending_workers: int
    running_workers: int = 0
    completed_workers: int = 0
    failed_workers: int = 0
    worker_states: dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, total_workers: int) -> "_Stage2ProgressTracker":
        return cls(
            total_workers=total_workers,
            pending_workers=total_workers,
            worker_states=dict.fromkeys(range(total_workers), "pending"),
        )

    def mark_running(self, worker_id: int) -> dict[str, int]:
        return self._transition(worker_id, "running")

    def mark_terminal(self, worker_id: int, state: str) -> dict[str, int]:
        return self._transition(worker_id, state)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, int]:
        return {
            "total_workers": self.total_workers,
            "pending_workers": self.pending_workers,
            "running_workers": self.running_workers,
            "completed_workers": self.completed_workers,
            "failed_workers": self.failed_workers,
        }

    def _transition(self, worker_id: int, next_state: str) -> dict[str, int]:
        with self._lock:
            current_state = self.worker_states.get(worker_id, "pending")
            if current_state == next_state:
                return self._snapshot_unlocked()
            if current_state in _TERMINAL_STAGE2_WORKER_STATES:
                return self._snapshot_unlocked()

            self._decrement_state_unlocked(current_state)
            self._increment_state_unlocked(next_state)
            self.worker_states[worker_id] = next_state
            return self._snapshot_unlocked()

    def _decrement_state_unlocked(self, state: str) -> None:
        if state == "pending":
            self.pending_workers -= 1
        elif state == "running":
            self.running_workers -= 1
        elif state == "completed":
            self.completed_workers -= 1
        elif state == "failed":
            self.failed_workers -= 1

    def _increment_state_unlocked(self, state: str) -> None:
        if state == "pending":
            self.pending_workers += 1
        elif state == "running":
            self.running_workers += 1
        elif state == "completed":
            self.completed_workers += 1
        elif state == "failed":
            self.failed_workers += 1


_stage2_progress_trackers: dict[str, _Stage2ProgressTracker] = {}
_stage2_progress_trackers_lock = threading.Lock()


def _register_stage2_progress_tracker(
    root_run_id: str,
    *,
    total_workers: int,
) -> _Stage2ProgressTracker:
    tracker = _Stage2ProgressTracker.create(total_workers)
    with _stage2_progress_trackers_lock:
        _stage2_progress_trackers[root_run_id] = tracker
    return tracker


def _get_stage2_progress_tracker(root_run_id: str) -> _Stage2ProgressTracker | None:
    with _stage2_progress_trackers_lock:
        return _stage2_progress_trackers.get(root_run_id)


def _clear_stage2_progress_tracker(root_run_id: str) -> None:
    with _stage2_progress_trackers_lock:
        _stage2_progress_trackers.pop(root_run_id, None)


def _emit_stage2_snapshot(root_run_id: str, snapshot: dict[str, int]) -> None:
    from causal_ssm_agent.utils.openrouter_client import get_limiter_request_count

    emit_stage2_snapshot_event(
        root_run_id,
        snapshot={
            **snapshot,
            "llm_requests_last_60s": get_limiter_request_count("llm"),
        },
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


def _group_indicators_by_window(
    indicators: list[dict],
    model_clock: str,
) -> list[tuple[str, list[dict]]]:
    """Group indicators by effective extraction/support window."""
    from causal_ssm_agent.utils.causal_spec import get_effective_observation_window

    grouped: dict[str, list[dict]] = {}
    for indicator in indicators:
        window = get_effective_observation_window(indicator, model_clock) or model_clock
        grouped.setdefault(window, []).append(indicator)
    return sorted(grouped.items(), key=lambda item: item[0])


def _chunk_log_label(chunk_idx: int, n_windows: int, n_events: int) -> str:
    """Build a stable log label for a worker chunk."""
    return f"stage2 chunk={chunk_idx} windows={n_windows} events={n_events}"


def _collect_batch_results(
    *,
    futures: Sequence[Any],
    batch_indices: Sequence[int],
    batch_n_windows: Sequence[int],
    logger: Any,
    completed_before: int,
    total_chunks: int,
    root_run_id: str | None = None,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Collect mapped worker futures in completion order while preserving output order."""
    batch_meta = {
        future: {
            "worker_id": worker_id,
            "n_windows": n_windows,
        }
        for worker_id, n_windows, future in zip(
            batch_indices, batch_n_windows, futures, strict=True
        )
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
        n_windows = meta["n_windows"]
        overall_completed = completed_before + batch_completed

        try:
            result = future.result()
        except _RECOVERABLE_STAGE2_WORKER_ERRORS as exc:
            logger.warning(
                "Stage 2: worker %d failed (progress=%d/%d, batch=%d/%d, windows=%d): %s",
                worker_id,
                overall_completed,
                total_chunks,
                batch_completed,
                batch_total,
                n_windows,
                exc,
            )
            statuses_by_worker[worker_id] = {
                "worker_id": worker_id,
                "status": "failed",
                "n_extractions": 0,
                "n_windows": n_windows,
                "error": str(exc),
            }
            if root_run_id:
                emit_stage2_worker_event(
                    root_run_id,
                    worker_id=worker_id,
                    state="failed",
                    n_windows=n_windows,
                    error=str(exc),
                )
                tracker = _get_stage2_progress_tracker(root_run_id)
                if tracker is not None:
                    _emit_stage2_snapshot(root_run_id, tracker.mark_terminal(worker_id, "failed"))
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
            "n_windows": n_windows,
        }
        logger.info(
            "Stage 2: worker %d %s (progress=%d/%d, batch=%d/%d, windows=%d, extractions=%d, output_rows=%d)",
            worker_id,
            status,
            overall_completed,
            total_chunks,
            batch_completed,
            batch_total,
            n_windows,
            n_ext,
            len(output_rows),
        )
        if root_run_id:
            emit_stage2_worker_event(
                root_run_id,
                worker_id=worker_id,
                state="completed",
                n_windows=n_windows,
                n_extractions=n_ext,
                n_llm_calls=worker_llm_calls or None,
            )
            tracker = _get_stage2_progress_tracker(root_run_id)
            if tracker is not None:
                _emit_stage2_snapshot(root_run_id, tracker.mark_terminal(worker_id, "completed"))

    ordered_rows = [row for worker_id in batch_indices for row in rows_by_worker.get(worker_id, [])]
    ordered_statuses = [
        statuses_by_worker[worker_id]
        for worker_id in batch_indices
        if worker_id in statuses_by_worker
    ]
    return ordered_rows, ordered_statuses, n_total, sampled_trace


def _prepare_semantic_chunks(
    *,
    raw_df: pl.DataFrame,
    semantic_inds: list[dict],
    causal_spec: dict,
    model_clock: str,
    time_col: str,
    windows_per_chunk: int,
    max_events_per_window: int,
    max_windows: int | None,
) -> tuple[list[str], list[list[str]], list[dict]]:
    """Prepare semantic extraction chunks without executing them.

    This isolates the deterministic Stage 2 windowing/chunking logic from the
    execution backend so both Prefect flows and eval harnesses can reuse it.
    """
    from causal_ssm_agent.utils.causal_spec import make_extraction_context
    from causal_ssm_agent.utils.data import bucket_by_clock
    from causal_ssm_agent.workers.windows import chunk_windows, format_window_chunk

    chunk_texts: list[str] = []
    chunk_window_starts: list[list[str]] = []
    chunk_contexts: list[dict] = []

    for observation_window, semantic_group in _group_indicators_by_window(
        semantic_inds, model_clock
    ):
        semantic_spec = {
            **causal_spec,
            "measurement": {**causal_spec.get("measurement", {}), "indicators": semantic_group},
        }
        extraction_ctx = make_extraction_context(semantic_spec)

        projected = _project_to_source_columns(raw_df, semantic_group)
        if time_col not in projected.columns:
            projected = projected.with_columns(raw_df[time_col])

        windows = bucket_by_clock(projected, observation_window, time_col)
        logger.info(
            "Stage 2: bucketed %d rows into %d support windows (window=%s, indicators=%d)",
            len(projected),
            len(windows),
            observation_window,
            len(semantic_group),
        )

        if max_windows is not None and len(windows) > max_windows:
            logger.warning(
                "Stage 2: free-tier window cap active for window=%s — truncating %d windows to most recent %d",
                observation_window,
                len(windows),
                max_windows,
            )
            windows = windows[-max_windows:]

        if not windows:
            continue

        display_cols = [c for c in projected.columns if c != time_col]
        chunks = chunk_windows(windows, windows_per_chunk)
        for chunk in chunks:
            chunk_texts.append(
                format_window_chunk(chunk, time_col, display_cols, max_events_per_window)
            )
            chunk_window_starts.append([window_start for window_start, _ in chunk])
            chunk_contexts.append(extraction_ctx)

    return chunk_texts, chunk_window_starts, chunk_contexts


async def _run_semantic_chunks_prefect(
    *,
    chunk_texts: list[str],
    chunk_window_starts: list[list[str]],
    chunk_contexts: list[dict],
    question: str,
    root_run_id: str | None,
    openrouter_api_key: str | None,
    max_concurrent_workers: int,
    max_rpm: int,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Execute semantic chunks through the existing Prefect worker path."""
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.utils.openrouter_client import RpmLimiter, set_limiter

    all_indices = list(range(len(chunk_texts)))
    all_n_windows = [len(ids) for ids in chunk_window_starts]

    logger.info(
        "Stage 2: %d semantic chunks of up to %d windows each (max_concurrent_workers=%d, max_rpm=%d)",
        len(chunk_texts),
        max(all_n_windows) if all_n_windows else 0,
        max_concurrent_workers,
        max_rpm,
    )

    if max_rpm:
        set_limiter("llm", RpmLimiter(max_rpm))

    try:
        if root_run_id:
            emit_stage2_plan_event(
                root_run_id,
                total_workers=len(chunk_texts),
                max_concurrent_workers=max_concurrent_workers,
                max_rpm=max_rpm,
            )
            tracker = _register_stage2_progress_tracker(
                root_run_id,
                total_workers=len(chunk_texts),
            )
            _emit_stage2_snapshot(root_run_id, tracker.snapshot())
        results = extract_window_chunk_task.map(
            chunk_texts,
            window_starts=chunk_window_starts,
            chunk_idx=all_indices,
            question=unmapped(question),
            causal_spec=chunk_contexts,
            root_run_id=unmapped(root_run_id),
            openrouter_api_key=unmapped(openrouter_api_key),
        )
        return _collect_batch_results(
            futures=results,
            batch_indices=all_indices,
            batch_n_windows=all_n_windows,
            logger=logger,
            completed_before=0,
            total_chunks=len(chunk_texts),
            root_run_id=root_run_id,
        )
    finally:
        if root_run_id:
            _clear_stage2_progress_tracker(root_run_id)
        set_limiter("llm", None)


async def run_stage2_extraction_core(
    *,
    raw_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
    stage2_workers: Any,
    root_run_id: str | None = None,
    max_windows: int | None = None,
    openrouter_api_key: str | None = None,
    semantic_chunk_runner: SemanticChunkRunner | None = None,
) -> dict:
    """Shared Stage 2 extraction helper for flows and evals.

    This is the deterministic orchestration core for:
    1. splitting computed vs semantic indicators
    2. computing direct indicators via Polars
    3. preparing semantic support-window chunks
    4. delegating semantic execution to an injected backend
    5. annotating canonical observation-row support metadata
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.data import ObservationRecord, annotate_observation_rows

    semantic_chunk_runner = semantic_chunk_runner or _run_semantic_chunks_prefect

    all_indicators = get_indicators(causal_spec)
    time_col = "timestamp"
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "1d")
    logger.info("Stage 2: time_col='%s', model_clock='%s'", time_col, model_clock)

    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [
        i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"
    ]
    logger.info(
        "Stage 2: %d computed + %d semantic indicators",
        len(computed_inds),
        len(semantic_inds),
    )

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

    semantic_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    sampled_llm_trace: dict | None = None
    n_semantic_total = 0

    if semantic_inds:
        chunk_texts, chunk_window_starts, chunk_contexts = _prepare_semantic_chunks(
            raw_df=raw_df,
            semantic_inds=semantic_inds,
            causal_spec=causal_spec,
            model_clock=model_clock,
            time_col=time_col,
            windows_per_chunk=stage2_workers.windows_per_chunk,
            max_events_per_window=stage2_workers.max_events_per_window,
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
                root_run_id=root_run_id,
                openrouter_api_key=openrouter_api_key,
                max_concurrent_workers=stage2_workers.max_concurrent_workers,
                max_rpm=stage2_workers.max_rpm,
            )

    all_dicts = computed_dicts + semantic_dicts
    n_total = len(computed_dicts) + n_semantic_total
    logger.info(
        "Stage 2: %d total extractions (%d computed, %d semantic from %d workers)",
        n_total,
        len(computed_dicts),
        n_semantic_total,
        len(worker_statuses),
    )

    observation_rows = cast(
        "list[ObservationRecord]",
        annotate_observation_rows(pl.DataFrame(all_dicts), causal_spec).to_dicts(),
    )

    result = {
        "observation_rows": observation_rows,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
    if sampled_llm_trace is not None:
        result["llm_trace"] = sampled_llm_trace
    return result


def materialize_stage2_outputs(stage2_result: dict, causal_spec: dict) -> dict[str, Any]:
    """Materialize the Stage 2 observation table from a serialized extraction result.

    Encodes non-continuous indicators in place (binary/ordinal/categorical → numeric),
    casts values to Float64, parses timestamps, drops rows with null anchor_time,
    and sorts by (indicator, anchor_time).
    """
    from causal_ssm_agent.utils.aggregations import _encode_non_continuous
    from causal_ssm_agent.utils.causal_spec import get_indicator_dtypes, get_indicators
    from causal_ssm_agent.utils.data import ObservationRecord, observation_row_schema

    observation_dicts = cast("list[ObservationRecord]", stage2_result.get("observation_rows", []))
    if observation_dicts:
        data_for_model = pl.DataFrame(observation_dicts)
    else:
        data_for_model = pl.DataFrame(schema=observation_row_schema())

    if len(data_for_model) > 0:
        dtype_lookup = get_indicator_dtypes(causal_spec)
        ordinal_levels_lookup: dict[str, list[str]] = {
            ind["name"]: ind["ordinal_levels"]
            for ind in get_indicators(causal_spec)
            if ind.get("ordinal_levels")
        }
        data_for_model = _encode_non_continuous(data_for_model, dtype_lookup, ordinal_levels_lookup)
        data_for_model = data_for_model.with_columns(
            pl.col("value").cast(pl.Float64, strict=False).alias("value"),
            pl.col("anchor_time")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("anchor_time"),
            pl.col("support_start")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_start"),
            pl.col("support_end")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_end"),
        ).drop_nulls(subset=["anchor_time"])
        data_for_model = data_for_model.sort("indicator", "anchor_time")

    return {
        "data_for_model": data_for_model,
        "worker_statuses": stage2_result.get("worker_statuses", []),
        "llm_trace": stage2_result.get("llm_trace"),
    }


@task(
    retries=2,
    retry_delay_seconds=10,
    timeout_seconds=300,
    task_run_name="extract-windows-{chunk_idx}",
)
async def extract_window_chunk_task(
    window_text: str,
    window_starts: list[str],
    chunk_idx: int,
    question: str,
    causal_spec: dict,
    root_run_id: str | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """Extract indicator values from a chunk of support windows.

    Args:
        window_text: Pre-formatted text of support-window events.
        window_starts: Expected support-window starts in this chunk.
        chunk_idx: Index of this chunk (for logging/naming).
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.

    Returns:
        Dict with 'dataframe' (as list of dicts for serialization),
        'n_extractions', and 'status'.
    """
    from dataclasses import replace

    from causal_ssm_agent.utils.agent_session import StageSessionFactory
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key
    from causal_ssm_agent.workers.core import run_worker_extraction

    run_logger = get_run_logger()
    config = get_config()
    # Stage 2 workers override only the per-call timeout (prevents hung calls).
    # Everything else inherits the embedded LLM defaults.
    stage2_llm = replace(
        config.stage2_workers.llm,
        timeout=config.stage2_workers.worker_timeout,
    )
    indicator_count = len(get_indicators(causal_spec))
    n_events = window_text.count("\n")
    chunk_label = _chunk_log_label(chunk_idx, len(window_starts), n_events)

    run_logger.info(
        "[%s] Starting extraction with %d windows, %d indicators using model %s (timeout=%ds)",
        chunk_label,
        len(window_starts),
        indicator_count,
        stage2_llm.model,
        stage2_llm.timeout,
    )
    if root_run_id:
        emit_stage2_worker_event(
            root_run_id,
            worker_id=chunk_idx,
            state="running",
            n_windows=len(window_starts),
        )
        tracker = _get_stage2_progress_tracker(root_run_id)
        if tracker is not None:
            _emit_stage2_snapshot(root_run_id, tracker.mark_running(chunk_idx))

    with use_openrouter_api_key(openrouter_api_key):
        factory = StageSessionFactory(
            stage2_llm,
            config.llm,
            stage_id=f"stage-2/chunk-{chunk_idx}",
            max_tool_turns=config.stage2_workers.max_tool_turns,
        )

        started_at = perf_counter()
        result = await run_worker_extraction(
            window_text=window_text,
            window_starts=window_starts,
            question=question,
            causal_spec=causal_spec,
            session_factory=factory,
            logger=run_logger,
            call_label=chunk_label,
        )

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
        if factory.accumulated_trace.messages:
            result_dict["llm_trace"] = factory.accumulated_trace.model_dump(mode="json")
        return result_dict


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
    root_run_id: str | None = None,
    max_windows: int | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """Stage 2: Extract indicator values via hybrid computed/semantic paths.

    1. Splits indicators by extraction_mode (computed vs semantic)
    2. Computed path: direct Polars aggregation (~50ms)
    3. Semantic path: support-window LLM workers
    4. Merges results into canonical observation rows with support metadata

    Args:
        raw_df_path: Path to the raw wide-format DataFrame from Stage 0 ingestion.
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.

    Returns:
        Dict with 'observation_rows' (long-format observation rows as list of dicts),
        'worker_statuses', and 'n_total_extractions'.
    """
    from causal_ssm_agent.utils.config import get_config

    if root_run_id:
        emit_nested_stage_running_event(root_run_id, "stage-2")

    config = get_config()
    raw_df = pl.read_parquet(Path(raw_df_path))
    return await run_stage2_extraction_core(
        raw_df=raw_df,
        question=question,
        causal_spec=causal_spec,
        stage2_workers=config.stage2_workers,
        root_run_id=root_run_id,
        max_windows=max_windows,
        openrouter_api_key=openrouter_api_key,
        semantic_chunk_runner=_run_semantic_chunks_prefect,
    )
