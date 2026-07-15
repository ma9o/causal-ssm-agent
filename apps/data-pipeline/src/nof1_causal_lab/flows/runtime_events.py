"""Transition and delegated-context telemetry events, persisted for UI polling.

Worker fan-out progress and model-spec admission streaming are live telemetry
the web UI renders while a transition runs. Events land as one JSON file each under
``data/{workspace_id}/scratch/events/`` (neither local fs nor R2 supports
atomic append); consumers list the directory and sort by filename, which is
time-ordered by construction. Events are transport, not a read model: nothing
may reconstruct state from them, and the sweep truncates the stream freely.

``workspace_id`` is the stream key — one episode per workspace.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

EXTRACTION_EVENT_PREFIX = "nof1-causal-lab.extraction"


class RuntimeEventModel(BaseModel):
    """Immutable JSON event record, with a cursor added only on reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cursor: str | None = None


class RuntimeEventError(BaseModel):
    """Serialized transition failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    message: str


class TransitionRuntimeEventPayload(BaseModel):
    """Payload for transition lifecycle telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: str
    status: Literal["running", "completed", "failed"]
    error: RuntimeEventError | None = None


class TransitionRuntimeEvent(RuntimeEventModel):
    """One transition lifecycle event."""

    event: Literal[
        "nof1-causal-lab.transition.running",
        "nof1-causal-lab.transition.completed",
        "nof1-causal-lab.transition.failed",
    ]
    payload: TransitionRuntimeEventPayload


class ExtractionPlanEventPayload(BaseModel):
    """Static extraction fan-out plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: Literal["measurement"] = "measurement"
    type: Literal["plan"] = "plan"
    total_workers: int = Field(ge=0)
    max_concurrent_workers: int | None = Field(default=None, gt=0)
    max_rpm: int | None = Field(default=None, gt=0)


class ExtractionPlanEvent(RuntimeEventModel):
    """Extraction plan telemetry event."""

    event: Literal["nof1-causal-lab.extraction.plan"]
    payload: ExtractionPlanEventPayload


class ExtractionWorkerEventPayload(BaseModel):
    """One extraction worker state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: Literal["measurement"] = "measurement"
    type: Literal["worker"] = "worker"
    worker_id: int = Field(ge=0)
    state: Literal["pending", "running", "completed", "failed"]
    n_windows: int = Field(ge=0)
    n_extractions: int | None = Field(default=None, ge=0)
    n_llm_calls: int | None = Field(default=None, ge=0)
    error: str | None = None


class ExtractionWorkerEvent(RuntimeEventModel):
    """Extraction worker telemetry event."""

    event: Literal["nof1-causal-lab.extraction.worker"]
    payload: ExtractionWorkerEventPayload


class ExtractionSnapshotEventPayload(BaseModel):
    """Aggregate extraction progress snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: Literal["measurement"] = "measurement"
    type: Literal["snapshot"] = "snapshot"
    total_workers: int = Field(ge=0)
    pending_workers: int = Field(ge=0)
    running_workers: int = Field(ge=0)
    completed_workers: int = Field(ge=0)
    failed_workers: int = Field(ge=0)
    llm_requests_last_60s: int = Field(ge=0)


class ExtractionSnapshotEvent(RuntimeEventModel):
    """Extraction snapshot telemetry event."""

    event: Literal["nof1-causal-lab.extraction.snapshot"]
    payload: ExtractionSnapshotEventPayload


type ModelSpecAdmissionEventName = Literal[
    "plan",
    "resumed",
    "construct_started",
    "construct_checking",
    "construct_report",
    "barrier_report",
    "done",
    "failed",
]

type ModelSpecAdmissionEventType = Literal[
    "nof1-causal-lab.model-spec.admission.plan",
    "nof1-causal-lab.model-spec.admission.resumed",
    "nof1-causal-lab.model-spec.admission.construct_started",
    "nof1-causal-lab.model-spec.admission.construct_checking",
    "nof1-causal-lab.model-spec.admission.construct_report",
    "nof1-causal-lab.model-spec.admission.barrier_report",
    "nof1-causal-lab.model-spec.admission.done",
    "nof1-causal-lab.model-spec.admission.failed",
]

_MODEL_SPEC_ADMISSION_EVENT_TYPES: dict[
    ModelSpecAdmissionEventName, ModelSpecAdmissionEventType
] = {
    "plan": "nof1-causal-lab.model-spec.admission.plan",
    "resumed": "nof1-causal-lab.model-spec.admission.resumed",
    "construct_started": "nof1-causal-lab.model-spec.admission.construct_started",
    "construct_checking": "nof1-causal-lab.model-spec.admission.construct_checking",
    "construct_report": "nof1-causal-lab.model-spec.admission.construct_report",
    "barrier_report": "nof1-causal-lab.model-spec.admission.barrier_report",
    "done": "nof1-causal-lab.model-spec.admission.done",
    "failed": "nof1-causal-lab.model-spec.admission.failed",
}


class ModelSpecAdmissionEvent(RuntimeEventModel):
    """Construct-admission event with JSON-safe event-specific fields."""

    event: Literal[
        "nof1-causal-lab.model-spec.admission.plan",
        "nof1-causal-lab.model-spec.admission.resumed",
        "nof1-causal-lab.model-spec.admission.construct_started",
        "nof1-causal-lab.model-spec.admission.construct_checking",
        "nof1-causal-lab.model-spec.admission.construct_report",
        "nof1-causal-lab.model-spec.admission.barrier_report",
        "nof1-causal-lab.model-spec.admission.done",
        "nof1-causal-lab.model-spec.admission.failed",
    ]
    payload: JsonObject

    @field_validator("payload")
    @classmethod
    def validate_context(cls, payload: JsonObject) -> JsonObject:
        if payload.get("context_id") != "statistical-model-spec":
            raise ValueError("model-spec admission events require context_id")
        return payload


type RuntimeEvent = Annotated[
    TransitionRuntimeEvent
    | ExtractionPlanEvent
    | ExtractionWorkerEvent
    | ExtractionSnapshotEvent
    | ModelSpecAdmissionEvent,
    Field(discriminator="event"),
]

_RUNTIME_EVENT_ADAPTER = TypeAdapter(RuntimeEvent)


def events_dir(workspace_id: str) -> str:
    return data_module.scratch_events_dir(workspace_id)


def emit_event(workspace_id: str, event: RuntimeEvent) -> None:
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps(event.model_dump(mode="json", exclude_none=True, exclude={"cursor"})),
    )


def read_events(workspace_id: str, *, after: str | None = None) -> list[RuntimeEvent]:
    """Events in emission order; ``after`` is the last seen filename cursor."""
    directory = events_dir(workspace_id)
    if not storage.exists(directory):
        return []
    entries = sorted(e for e in storage.listdir(directory) if e.endswith(".json"))
    events: list[RuntimeEvent] = []
    for entry in entries:
        cursor = entry.rsplit("/", 1)[-1]
        if after is not None and cursor <= after:
            continue
        record = storage.read_json(entry)
        events.append(_RUNTIME_EVENT_ADAPTER.validate_python({**record, "cursor": cursor}))
    return events


def emit_transition_event(
    workspace_id: str,
    transition_id: str,
    status: Literal["running", "completed", "failed"],
    *,
    error: JsonObject | None = None,
) -> None:
    event_type: Literal[
        "nof1-causal-lab.transition.running",
        "nof1-causal-lab.transition.completed",
        "nof1-causal-lab.transition.failed",
    ]
    if status == "running":
        event_type = "nof1-causal-lab.transition.running"
    elif status == "completed":
        event_type = "nof1-causal-lab.transition.completed"
    else:
        event_type = "nof1-causal-lab.transition.failed"
    emit_event(
        workspace_id,
        TransitionRuntimeEvent(
            event=event_type,
            payload=TransitionRuntimeEventPayload(
                transition_id=transition_id,
                status=status,
                error=RuntimeEventError.model_validate(error) if error is not None else None,
            ),
        ),
    )


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
        ExtractionPlanEvent(
            event=f"{EXTRACTION_EVENT_PREFIX}.plan",
            payload=ExtractionPlanEventPayload(
                total_workers=total_workers,
                max_concurrent_workers=max_concurrent_workers,
                max_rpm=max_rpm,
            ),
        ),
    )


def emit_extraction_worker_event(
    workspace_id: str,
    *,
    worker_id: int,
    state: Literal["pending", "running", "completed", "failed"],
    n_windows: int,
    n_extractions: int | None = None,
    n_llm_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Emit an extraction worker state transition."""
    emit_event(
        workspace_id,
        ExtractionWorkerEvent(
            event=f"{EXTRACTION_EVENT_PREFIX}.worker",
            payload=ExtractionWorkerEventPayload(
                worker_id=worker_id,
                state=state,
                n_windows=n_windows,
                n_extractions=n_extractions,
                n_llm_calls=n_llm_calls,
                error=error,
            ),
        ),
    )


def emit_extraction_snapshot_event(workspace_id: str, *, snapshot: dict[str, int]) -> None:
    """Emit an extraction runtime snapshot."""
    emit_event(
        workspace_id,
        ExtractionSnapshotEvent(
            event=f"{EXTRACTION_EVENT_PREFIX}.snapshot",
            payload=ExtractionSnapshotEventPayload.model_validate(snapshot),
        ),
    )


def emit_model_spec_admission_event(
    workspace_id: str,
    event: ModelSpecAdmissionEventName,
    payload: JsonObject,
) -> None:
    """Emit one model-spec construct-admission telemetry event.

    ``event`` is the sub-name (``plan`` / ``construct_started`` / ``construct_checking`` /
    ``construct_report`` / ``done`` / ``failed``); the web UI reduces the stream into the live
    construct-admission view. Payloads are assembled by the admission flow, which owns the
    translation of ``AdmissionReport``/``ConstructContribution`` into the UI contract.
    """
    emit_event(
        workspace_id,
        ModelSpecAdmissionEvent(
            event=_MODEL_SPEC_ADMISSION_EVENT_TYPES[event],
            payload={"context_id": "statistical-model-spec", **payload},
        ),
    )


__all__ = [
    "EXTRACTION_EVENT_PREFIX",
    "emit_event",
    "emit_extraction_plan_event",
    "emit_extraction_snapshot_event",
    "emit_extraction_worker_event",
    "emit_model_spec_admission_event",
    "emit_transition_event",
    "events_dir",
    "read_events",
]
