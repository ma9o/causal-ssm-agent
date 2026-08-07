"""Replay-safe support shared by transition workflows."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.temporal.messages import (
    TransitionRuntimeError,
    TransitionRuntimeEventInput,
    TransitionRuntimeStatus,
)

EVENT_TIMEOUT = timedelta(seconds=30)
EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)
type TemporalFailureDiagnostics = dict[str, Any]


async def emit_transition_runtime_event(
    workspace_id: str,
    transition_id: str,
    status: TransitionRuntimeStatus,
    error: TransitionRuntimeError | None = None,
) -> None:
    """Emit a transition lifecycle event using the common activity policy."""
    await workflow.execute_activity(
        "emit_transition_runtime_event_activity",
        TransitionRuntimeEventInput(
            workspace_id=workspace_id,
            transition_id=transition_id,
            status=status,
            error=error,
        ),
        start_to_close_timeout=EVENT_TIMEOUT,
        retry_policy=EVENT_RETRY,
    )


def temporal_failure_details(
    exc: BaseException,
) -> tuple[str, str, TemporalFailureDiagnostics]:
    """Unwrap a Temporal cause chain into a stable runtime error payload."""
    cause = exc
    while not isinstance(cause, ApplicationError):
        next_cause = getattr(cause, "cause", None)
        if not isinstance(next_cause, BaseException):
            break
        cause = next_cause
    if isinstance(cause, ApplicationError):
        diagnostics = (
            cause.details[0] if cause.details and isinstance(cause.details[0], dict) else {}
        )
        return cause.type or "ApplicationError", cause.message, dict(diagnostics)
    return type(cause).__name__, str(cause), {}
