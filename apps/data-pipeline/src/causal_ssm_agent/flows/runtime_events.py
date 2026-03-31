"""Shared runtime event helpers for stage progress and nested stage metadata."""

from __future__ import annotations

from typing import Any

from prefect.context import get_run_context
from prefect.events import emit_event

STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage"


def _normalize_log_flow_run_ids(
    stage_subflow_run_id: str | None,
    log_flow_run_ids: list[str] | None,
) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()

    for candidate in [stage_subflow_run_id, *(log_flow_run_ids or [])]:
        if not candidate:
            continue
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_ids.append(normalized)

    return ordered_ids


def emit_stage_progress_event(
    resource_run_id: str,
    stage_id: str,
    status: str,
    *,
    outcome: str | None = None,
    error: dict[str, Any] | None = None,
    stage_subflow_run_id: str | None = None,
    log_flow_run_ids: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage_id": stage_id,
        "status": status,
    }
    if outcome is not None:
        payload["outcome"] = outcome
    if error is not None:
        payload["error"] = error

    normalized_subflow_run_id = (
        stage_subflow_run_id.strip()
        if stage_subflow_run_id and stage_subflow_run_id.strip()
        else None
    )
    normalized_log_flow_run_ids = _normalize_log_flow_run_ids(
        normalized_subflow_run_id,
        log_flow_run_ids,
    )

    if normalized_subflow_run_id is not None:
        payload["stage_subflow_run_id"] = normalized_subflow_run_id
    if normalized_log_flow_run_ids:
        payload["log_flow_run_ids"] = normalized_log_flow_run_ids

    related: list[dict[str, str]] | None = None
    if normalized_subflow_run_id is not None:
        related = [
            {
                "prefect.resource.id": f"prefect.flow-run.{normalized_subflow_run_id}",
                "prefect.resource.role": "stage-subflow",
            }
        ]

    emit_event(
        event=f"{STAGE_PROGRESS_EVENT_PREFIX}.{status}",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        related=related,
        payload=payload,
    )


def emit_nested_stage_running_event(
    root_run_id: str,
    stage_id: str,
    *,
    log_flow_run_ids: list[str] | None = None,
) -> str:
    stage_subflow_run_id = str(get_run_context().flow_run.id)
    emit_stage_progress_event(
        root_run_id,
        stage_id,
        "running",
        stage_subflow_run_id=stage_subflow_run_id,
        log_flow_run_ids=log_flow_run_ids,
    )
    return stage_subflow_run_id


STAGE4_EVENT_PREFIX = "causal-ssm.stage4"


def emit_stage4_graph_event(
    resource_run_id: str,
    *,
    graph: dict[str, Any],
) -> None:
    """Emit the static Stage 4 graph topology as a Prefect custom event."""
    emit_event(
        event=f"{STAGE4_EVENT_PREFIX}.graph",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload={"stage_id": "stage-4", "type": "graph", **graph},
    )


def emit_stage4_snapshot_event(
    resource_run_id: str,
    *,
    snapshot: dict[str, Any],
) -> None:
    """Emit a Stage 4 runtime state snapshot as a Prefect custom event."""
    emit_event(
        event=f"{STAGE4_EVENT_PREFIX}.snapshot",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload={"stage_id": "stage-4", "type": "snapshot", **snapshot},
    )


__all__ = [
    "STAGE4_EVENT_PREFIX",
    "STAGE_PROGRESS_EVENT_PREFIX",
    "emit_nested_stage_running_event",
    "emit_stage4_graph_event",
    "emit_stage4_snapshot_event",
    "emit_stage_progress_event",
]
