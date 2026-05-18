"""Shared runtime event helpers for stage progress and nested stage metadata."""

from __future__ import annotations

from typing import Any

from prefect.events import emit_event

from causal_ssm_agent.flows import get_current_flow_run_id

STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage"
STAGE2_EVENT_PREFIX = "causal-ssm.stage2"


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
    stage_subflow_run_id = get_current_flow_run_id()
    emit_stage_progress_event(
        root_run_id,
        stage_id,
        "running",
        stage_subflow_run_id=stage_subflow_run_id,
        log_flow_run_ids=log_flow_run_ids,
    )
    return stage_subflow_run_id


def emit_stage2_plan_event(
    resource_run_id: str,
    *,
    total_workers: int,
    max_concurrent_workers: int | None,
    max_rpm: int | None,
) -> None:
    """Emit the static Stage 2 execution plan for replay/bootstrap."""
    emit_event(
        event=f"{STAGE2_EVENT_PREFIX}.plan",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload={
            "stage_id": "stage-2",
            "type": "plan",
            "total_workers": total_workers,
            "max_concurrent_workers": max_concurrent_workers,
            "max_rpm": max_rpm,
        },
    )


def emit_stage2_worker_event(
    resource_run_id: str,
    *,
    worker_id: int,
    state: str,
    n_windows: int,
    n_extractions: int | None = None,
    n_llm_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Emit a Stage 2 worker state transition on the root flow run resource."""
    payload: dict[str, Any] = {
        "stage_id": "stage-2",
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

    emit_event(
        event=f"{STAGE2_EVENT_PREFIX}.worker",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload=payload,
    )


def emit_stage2_snapshot_event(
    resource_run_id: str,
    *,
    snapshot: dict[str, Any],
) -> None:
    """Emit a Stage 2 runtime snapshot as a Prefect custom event."""
    emit_event(
        event=f"{STAGE2_EVENT_PREFIX}.snapshot",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload={"stage_id": "stage-2", "type": "snapshot", **snapshot},
    )


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


def emit_stage4_block_transition_event(
    resource_run_id: str,
    *,
    transition: dict[str, Any],
) -> None:
    """Emit one Stage 4 dot-level transition event for replayable UI history."""
    emit_event(
        event=f"{STAGE4_EVENT_PREFIX}.block_transition",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{resource_run_id}",
            "prefect.resource.name": resource_run_id,
        },
        payload={"stage_id": "stage-4", "type": "block_transition", **transition},
    )


__all__ = [
    "STAGE2_EVENT_PREFIX",
    "STAGE4_EVENT_PREFIX",
    "STAGE_PROGRESS_EVENT_PREFIX",
    "emit_nested_stage_running_event",
    "emit_stage2_plan_event",
    "emit_stage2_snapshot_event",
    "emit_stage2_worker_event",
    "emit_stage4_block_transition_event",
    "emit_stage4_graph_event",
    "emit_stage4_snapshot_event",
    "emit_stage_progress_event",
]
