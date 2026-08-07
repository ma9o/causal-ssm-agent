"""Shared conversion of transition failures into Temporal activity failures."""

from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.errors import TransitionExecutionError


def as_non_retryable_application_error(exc: Exception) -> ApplicationError:
    """Preserve transition diagnostics while preventing deterministic retries."""
    if isinstance(exc, TransitionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)
