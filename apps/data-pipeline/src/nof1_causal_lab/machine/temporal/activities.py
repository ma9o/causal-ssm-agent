"""Activities: the machine's I/O, executed outside the workflow sandbox.

Typed stage failures map to non-retryable ``ApplicationError``s whose
details carry the diagnostics dict — the workflow journals them and hands
them to the navigator as the move's outcome. Anything else (network,
OOM, Modal preemption) is transient infra: the retry policy re-runs it
and the navigator never sees it.
"""

from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.errors import ArtifactWriteRejected, StageExecutionError

# Runtime imports (not TYPE_CHECKING): temporalio resolves activity type
# hints at registration time to drive payload conversion.
from nof1_causal_lab.machine.moves import TransitionEffects  # noqa: TC001
from nof1_causal_lab.machine.runners import execute_stage
from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord, utc_now_iso
from nof1_causal_lab.machine.temporal.messages import (  # noqa: TC001
    JournalInput,
    RunStageInput,
    WriteArtifactInput,
)
from nof1_causal_lab.machine.writes import execute_write


@activity.defn
async def run_stage_activity(input: RunStageInput) -> TransitionEffects:
    try:
        return await execute_stage(input.workspace_id, input.stage_id, input.state, input.options)
    except StageExecutionError as exc:
        raise ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def write_artifact_activity(input: WriteArtifactInput) -> TransitionEffects:
    try:
        return execute_write(input.workspace_id, input.artifact_id, input.payload, input.provenance)
    except ArtifactWriteRejected as exc:
        raise ApplicationError(
            str(exc),
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def journal_activity(input: JournalInput) -> None:
    EpisodeJournal(input.workspace_id).append(
        TransitionRecord(
            seq=input.seq,
            ts=utc_now_iso(),
            move=input.move,
            status=input.status,  # type: ignore[arg-type]
            reason=input.reason,
            error_type=input.error_type,
            error_message=input.error_message,
            diagnostics=input.diagnostics,
            produced=input.produced,
            retracted=input.retracted,
            state_after=input.state_after,
        )
    )


ALL_ACTIVITIES = [run_stage_activity, write_artifact_activity, journal_activity]
