"""Stage 2 execution helpers for semantic worker batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def collect_batch_results(
    *,
    futures: Sequence[Any],
    batch_indices: Sequence[int],
    batch_n_windows: Sequence[int],
    logger: Any,
    completed_before: int,
    total_chunks: int,
    as_completed_fn: Any,
    root_run_id: str | None = None,
    emit_worker_event: Any | None = None,
    get_tracker: Any | None = None,
    emit_snapshot: Any | None = None,
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

    for batch_completed, future in enumerate(as_completed_fn(list(batch_meta)), start=1):
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
            if root_run_id and emit_worker_event is not None:
                emit_worker_event(
                    root_run_id,
                    worker_id=worker_id,
                    state="failed",
                    n_windows=n_windows,
                    error=str(exc),
                )
                tracker = get_tracker(root_run_id) if get_tracker is not None else None
                if tracker is not None and emit_snapshot is not None:
                    emit_snapshot(root_run_id, tracker.mark_terminal(worker_id, "failed"))
            continue

        n_ext = result.get("n_extractions", 0)
        output_rows = result.get("dataframe", [])
        status = result.get("status", "completed")
        llm_trace = result.get("llm_trace")
        if sampled_trace is None and llm_trace is not None:
            sampled_trace = llm_trace

        worker_llm_calls = 0
        if llm_trace and isinstance(llm_trace, dict):
            worker_llm_calls = sum(
                1 for message in llm_trace.get("messages", []) if message.get("role") == "assistant"
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
        if root_run_id and emit_worker_event is not None:
            emit_worker_event(
                root_run_id,
                worker_id=worker_id,
                state="completed",
                n_windows=n_windows,
                n_extractions=n_ext,
                n_llm_calls=worker_llm_calls or None,
            )
            tracker = get_tracker(root_run_id) if get_tracker is not None else None
            if tracker is not None and emit_snapshot is not None:
                emit_snapshot(root_run_id, tracker.mark_terminal(worker_id, "completed"))

    ordered_rows = [row for worker_id in batch_indices for row in rows_by_worker.get(worker_id, [])]
    missing = [worker_id for worker_id in batch_indices if worker_id not in statuses_by_worker]
    if missing:
        raise RuntimeError(
            f"Stage 2 missing status for worker(s) {missing}; "
            f"as_completed returned {len(statuses_by_worker)}/{len(batch_indices)} futures"
        )
    ordered_statuses = [statuses_by_worker[worker_id] for worker_id in batch_indices]
    return ordered_rows, ordered_statuses, n_total, sampled_trace
