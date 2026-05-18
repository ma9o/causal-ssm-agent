"""Stage 2 progress tracking helpers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from nof1_causal_lab.flows.runtime_events import emit_stage2_snapshot_event

_TERMINAL_STAGE2_WORKER_STATES = {"completed", "failed"}


@dataclass
class Stage2ProgressTracker:
    total_workers: int
    pending_workers: int
    running_workers: int = 0
    completed_workers: int = 0
    failed_workers: int = 0
    worker_states: dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, total_workers: int) -> Stage2ProgressTracker:
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
            if current_state == next_state or current_state in _TERMINAL_STAGE2_WORKER_STATES:
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


_stage2_progress_trackers: dict[str, Stage2ProgressTracker] = {}
_stage2_progress_trackers_lock = threading.Lock()


def register_stage2_progress_tracker(
    root_run_id: str,
    *,
    total_workers: int,
) -> Stage2ProgressTracker:
    tracker = Stage2ProgressTracker.create(total_workers)
    with _stage2_progress_trackers_lock:
        _stage2_progress_trackers[root_run_id] = tracker
    return tracker


def get_stage2_progress_tracker(root_run_id: str) -> Stage2ProgressTracker | None:
    with _stage2_progress_trackers_lock:
        return _stage2_progress_trackers.get(root_run_id)


def clear_stage2_progress_tracker(root_run_id: str) -> None:
    with _stage2_progress_trackers_lock:
        _stage2_progress_trackers.pop(root_run_id, None)


def emit_stage2_snapshot(root_run_id: str, snapshot: dict[str, int]) -> None:
    from nof1_causal_lab.utils.openrouter_client import get_limiter_request_count

    emit_stage2_snapshot_event(
        root_run_id,
        snapshot={
            **snapshot,
            "llm_requests_last_60s": get_limiter_request_count("llm"),
        },
    )
