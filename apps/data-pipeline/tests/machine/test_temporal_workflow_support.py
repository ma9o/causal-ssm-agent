"""Tests for replay-safe transition workflow helpers."""

import asyncio
from typing import Any

from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.temporal.messages import TransitionRuntimeError
from nof1_causal_lab.machine.temporal.workflow_support import (
    EVENT_RETRY,
    EVENT_TIMEOUT,
    emit_transition_runtime_event,
    temporal_failure_details,
)


class _WrappedError(Exception):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def test_temporal_failure_details_unwraps_application_error_and_copies_diagnostics() -> None:
    diagnostics = {"reason": "invalid"}
    error = _WrappedError(
        _WrappedError(
            ApplicationError(
                "model failed",
                diagnostics,
                type="ModelCompileError",
                non_retryable=True,
            )
        )
    )

    error_type, message, extracted = temporal_failure_details(error)
    extracted["local"] = True

    assert error_type == "ModelCompileError"
    assert message == "model failed"
    assert diagnostics == {"reason": "invalid"}


def test_temporal_failure_details_uses_deepest_untyped_cause() -> None:
    assert temporal_failure_details(_WrappedError(ValueError("bad input"))) == (
        "ValueError",
        "bad input",
        {},
    )


def test_emit_transition_runtime_event_uses_shared_activity_policy(monkeypatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def execute_activity(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    from nof1_causal_lab.machine.temporal import workflow_support

    monkeypatch.setattr(workflow_support.workflow, "execute_activity", execute_activity)
    runtime_error = TransitionRuntimeError(type="ValueError", message="bad input")

    asyncio.run(
        emit_transition_runtime_event(
            "workspace-1",
            "measurements",
            "failed",
            runtime_error,
        )
    )

    (activity_name, event), kwargs = calls[0]
    assert activity_name == "emit_transition_runtime_event_activity"
    assert event.workspace_id == "workspace-1"
    assert event.transition_id == "measurements"
    assert event.status == "failed"
    assert event.error == runtime_error
    assert kwargs == {
        "start_to_close_timeout": EVENT_TIMEOUT,
        "retry_policy": EVENT_RETRY,
    }
