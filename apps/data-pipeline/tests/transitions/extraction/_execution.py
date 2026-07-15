"""Test support for the retired asyncio semantic-worker batches."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

_RECOVERABLE_STAGE2_WORKER_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValidationError,
    ValueError,
)

WORKER_RETRIES = 2
WORKER_RETRY_DELAY_SECONDS = 10
WORKER_ATTEMPT_TIMEOUT_SECONDS = 300


async def run_with_retries(
    attempt: Callable[[], Awaitable[dict]],
    *,
    retries: int = WORKER_RETRIES,
    delay_seconds: float = WORKER_RETRY_DELAY_SECONDS,
    timeout_seconds: float = WORKER_ATTEMPT_TIMEOUT_SECONDS,
) -> dict:
    """Per-worker retry envelope (was Prefect task retries/timeout)."""
    for attempt_index in range(retries + 1):
        try:
            return await asyncio.wait_for(attempt(), timeout_seconds)
        except _RECOVERABLE_STAGE2_WORKER_ERRORS:
            if attempt_index == retries:
                raise
            await asyncio.sleep(delay_seconds)
    raise AssertionError("unreachable")


async def collect_worker_results(
    *,
    tasks: Sequence[asyncio.Task],
    batch_indices: Sequence[int],
    batch_n_windows: Sequence[int],
    logger: Any,
    total_chunks: int,
    workspace_id: str | None = None,
    emit_worker_event: Any | None = None,
    get_tracker: Any | None = None,
    emit_snapshot: Any | None = None,
) -> tuple[list[dict], list[dict], int, str | None]:
    """Await worker tasks in completion order while preserving output order.

    A worker that exhausts its retries is recorded as failed and the batch
    continues — partial extraction is a finding, not a stage failure.
    """
    meta_by_task = {
        task: {"worker_id": worker_id, "n_windows": n_windows}
        for worker_id, n_windows, task in zip(batch_indices, batch_n_windows, tasks, strict=True)
    }
    batch_total = len(meta_by_task)
    rows_by_worker: dict[int, list[dict]] = {}
    statuses_by_worker: dict[int, dict] = {}
    sampled_trace_ref: str | None = None
    n_total = 0
    batch_completed = 0

    logger.info("extraction: waiting for %d worker results", batch_total)

    pending: set[asyncio.Task] = set(meta_by_task)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            batch_completed += 1
            meta = meta_by_task[task]
            worker_id = meta["worker_id"]
            n_windows = meta["n_windows"]

            try:
                result = task.result()
            except _RECOVERABLE_STAGE2_WORKER_ERRORS as exc:
                logger.warning(
                    "extraction: worker %d failed (batch=%d/%d, windows=%d): %s",
                    worker_id,
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
                if workspace_id and emit_worker_event is not None:
                    emit_worker_event(
                        workspace_id,
                        worker_id=worker_id,
                        state="failed",
                        n_windows=n_windows,
                        error=str(exc),
                    )
                    tracker = get_tracker(workspace_id) if get_tracker is not None else None
                    if tracker is not None and emit_snapshot is not None:
                        emit_snapshot(workspace_id, tracker.mark_terminal(worker_id, "failed"))
                continue

            n_ext = result.get("n_extractions", 0)
            output_rows = result.get("dataframe", [])
            status = result.get("status", "completed")
            llm_trace_ref = result.get("llm_trace_ref")
            if sampled_trace_ref is None and isinstance(llm_trace_ref, str):
                sampled_trace_ref = llm_trace_ref

            worker_llm_calls = int(result.get("n_llm_calls") or 0)
            n_total += n_ext
            rows_by_worker[worker_id] = output_rows
            statuses_by_worker[worker_id] = {
                "worker_id": worker_id,
                "status": status,
                "n_extractions": n_ext,
                "n_windows": n_windows,
            }
            logger.info(
                "extraction: worker %d %s (batch=%d/%d, windows=%d, extractions=%d, output_rows=%d)",
                worker_id,
                status,
                batch_completed,
                batch_total,
                n_windows,
                n_ext,
                len(output_rows),
            )
            if workspace_id and emit_worker_event is not None:
                emit_worker_event(
                    workspace_id,
                    worker_id=worker_id,
                    state="completed",
                    n_windows=n_windows,
                    n_extractions=n_ext,
                    n_llm_calls=worker_llm_calls or None,
                )
                tracker = get_tracker(workspace_id) if get_tracker is not None else None
                if tracker is not None and emit_snapshot is not None:
                    emit_snapshot(workspace_id, tracker.mark_terminal(worker_id, "completed"))

    ordered_rows = [row for worker_id in batch_indices for row in rows_by_worker.get(worker_id, [])]
    missing = [worker_id for worker_id in batch_indices if worker_id not in statuses_by_worker]
    if missing:
        raise RuntimeError(
            f"extraction missing status for worker(s) {missing}; "
            f"collected {len(statuses_by_worker)}/{len(batch_indices)} tasks"
        )
    ordered_statuses = [statuses_by_worker[worker_id] for worker_id in batch_indices]
    logger.info("extraction: collected %d/%d workers", len(statuses_by_worker), total_chunks)
    return ordered_rows, ordered_statuses, n_total, sampled_trace_ref
