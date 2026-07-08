"""Transition and delegated-context telemetry events, persisted for UI polling.

Worker fan-out progress and model-spec admission streaming are live telemetry
the web UI renders while a transition runs. Events land as one JSON file each under
``data/{workspace_id}/episode/events/`` (neither local fs nor R2 supports
atomic append); consumers list the directory and sort by filename, which is
time-ordered by construction.

``workspace_id`` is the stream key — one episode per workspace.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

TRANSITION_EVENT_PREFIX = "nof1-causal-lab.transition"
EXTRACTION_EVENT_PREFIX = "nof1-causal-lab.extraction"
MODEL_SPEC_ADMISSION_EVENT_PREFIX = "nof1-causal-lab.model-spec.admission"


def events_dir(workspace_id: str) -> str:
    return storage.join(data_module.DATA_URI, workspace_id, "episode", "events")


def emit_event(workspace_id: str, event: str, payload: dict[str, Any]) -> None:
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps({"event": event, "payload": payload}),
    )


def read_events(workspace_id: str, *, after: str | None = None) -> list[dict[str, Any]]:
    """Events in emission order; ``after`` is the last seen filename cursor."""
    directory = events_dir(workspace_id)
    if not storage.exists(directory):
        return []
    entries = sorted(e for e in storage.listdir(directory) if e.endswith(".json"))
    events = []
    for entry in entries:
        cursor = entry.rsplit("/", 1)[-1]
        if after is not None and cursor <= after:
            continue
        record = storage.read_json(entry)
        record["cursor"] = cursor
        events.append(record)
    return events


def emit_transition_event(
    workspace_id: str,
    transition_id: str,
    status: str,
    *,
    error: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"transition_id": transition_id, "status": status}
    if error is not None:
        payload["error"] = error
    emit_event(workspace_id, f"{TRANSITION_EVENT_PREFIX}.{status}", payload)


def emit_extraction_plan_event(
    workspace_id: str,
    *,
    total_workers: int,
    max_concurrent_workers: int | None,
    max_rpm: int | None,
) -> None:
    """Emit the static extraction execution plan for replay/bootstrap."""
    emit_event(
        workspace_id,
        f"{EXTRACTION_EVENT_PREFIX}.plan",
        {
            "context_id": "measurement",
            "type": "plan",
            "total_workers": total_workers,
            "max_concurrent_workers": max_concurrent_workers,
            "max_rpm": max_rpm,
        },
    )


def emit_extraction_worker_event(
    workspace_id: str,
    *,
    worker_id: int,
    state: str,
    n_windows: int,
    n_extractions: int | None = None,
    n_llm_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Emit an extraction worker state transition."""
    payload: dict[str, Any] = {
        "context_id": "measurement",
        "type": "worker",
        "worker_id": worker_id,
        "state": state,
        "n_windows": n_windows,
    }
    if n_extractions is not None:
        payload["n_extractions"] = n_extractions
    if n_llm_calls is not None:
        payload["n_llm_calls"] = n_llm_calls
    if error is not None:
        payload["error"] = error
    emit_event(workspace_id, f"{EXTRACTION_EVENT_PREFIX}.worker", payload)


def emit_extraction_snapshot_event(workspace_id: str, *, snapshot: dict[str, Any]) -> None:
    """Emit an extraction runtime snapshot."""
    emit_event(
        workspace_id,
        f"{EXTRACTION_EVENT_PREFIX}.snapshot",
        {"context_id": "measurement", "type": "snapshot", **snapshot},
    )


def emit_model_spec_admission_event(workspace_id: str, event: str, payload: dict[str, Any]) -> None:
    """Emit one model-spec construct-admission telemetry event.

    ``event`` is the sub-name (``plan`` / ``construct_started`` / ``construct_checking`` /
    ``construct_report`` / ``done`` / ``failed``); the web UI reduces the stream into the live
    construct-admission view. Payloads are assembled by the admission flow, which owns the
    translation of ``AdmissionReport``/``ConstructContribution`` into the UI contract.
    """
    emit_event(
        workspace_id,
        f"{MODEL_SPEC_ADMISSION_EVENT_PREFIX}.{event}",
        {"context_id": "statistical-model-spec", **payload},
    )


__all__ = [
    "EXTRACTION_EVENT_PREFIX",
    "MODEL_SPEC_ADMISSION_EVENT_PREFIX",
    "TRANSITION_EVENT_PREFIX",
    "emit_event",
    "emit_extraction_plan_event",
    "emit_extraction_snapshot_event",
    "emit_extraction_worker_event",
    "emit_model_spec_admission_event",
    "emit_transition_event",
    "events_dir",
    "read_events",
]
