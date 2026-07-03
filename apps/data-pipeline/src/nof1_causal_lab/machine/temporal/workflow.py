"""The episode entity workflow: durable shell around the pure machine.

One workflow per workspace episode. Local state (current artifact
versions, move counter) survives crashes and redeploys by replay; the
``propose`` update is the single entry point for moves. It always
accepts and validates *inside* the handler — a Temporal update-validator
rejection would leave no trace in history, and rejected proposals are
exactly what the timeline scrubber wants to show. Every attempt
(applied, rejected, raised) is projected into the episode journal by an
activity before the outcome returns to the caller.

Handlers stay thin (validate → execute activity → apply → journal) so
workflow-code versioning churn stays small; all semantics live in the
pure functions, all I/O in activities (referenced by name so the
sandbox never imports storage/polars/jax).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.artifacts import (
        ArtifactId,
        ArtifactVersionInfo,
        EpisodeState,
    )
    from nof1_causal_lab.machine.graph import stage_spec
    from nof1_causal_lab.machine.moves import (
        Move,
        RunStage,
        TransitionEffects,
        WriteArtifact,
        apply_transition,
        freshness_report,
        legal_moves,
        run_retractions,
        validate_move,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        EpisodeInit,
        EpisodeStatus,
        JournalInput,
        MoveOutcome,
        MoveRequest,
        RunStageInput,
        WriteArtifactInput,
    )

_RUN_STAGE_TIMEOUT = timedelta(hours=4)  # stage-4/5b run up to 3h on Modal
_WRITE_TIMEOUT = timedelta(minutes=5)
_JOURNAL_TIMEOUT = timedelta(minutes=1)

_NON_RETRYABLE_ERRORS = [
    "StageExecutionError",
    "ModelCompileError",
    "ModelFitError",
    "ArtifactWriteRejected",
    "ValueError",
]

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
    non_retryable_error_types=_NON_RETRYABLE_ERRORS,
)
_JOURNAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=5,
)


@workflow.defn
class EpisodeWorkflow:
    @workflow.init
    def __init__(self, init: EpisodeInit) -> None:
        # workflow.init: updates can be dispatched before the run method's
        # first line executes, and every handler needs workspace_id.
        self._workspace_id = init.workspace_id
        self._state = EpisodeState()
        self._seq = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @workflow.run
    async def run(self, init: EpisodeInit) -> EpisodeState:
        del init  # consumed by @workflow.init
        await workflow.wait_condition(lambda: self._closed)
        return self._state

    # -- moves -----------------------------------------------------------

    @workflow.update
    async def propose(self, request: MoveRequest) -> MoveOutcome:
        # Serialize moves: one transition at a time per episode. Validation
        # happens inside the accepted update so rejections reach history
        # and the journal (scrubber requirement).
        async with self._lock:
            self._seq += 1
            seq = self._seq
            move = request.move

            reason = validate_move(self._state, move)
            if reason is None and isinstance(move, WriteArtifact) and request.payload is None:
                reason = "write moves require a payload"
            if reason is not None:
                await self._journal(seq, move, status="rejected", reason=reason)
                return self._outcome(seq, status="rejected", reason=reason)

            try:
                if isinstance(move, RunStage):
                    effects = await workflow.execute_activity(
                        "run_stage_activity",
                        RunStageInput(
                            workspace_id=self._workspace_id,
                            stage_id=move.stage_id,
                            state=self._state,
                            options=request.options,
                        ),
                        result_type=TransitionEffects,
                        start_to_close_timeout=_RUN_STAGE_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    produced = effects.produced
                    retracted = run_retractions(self._state, stage_spec(move.stage_id), produced)
                else:
                    effects = await workflow.execute_activity(
                        "write_artifact_activity",
                        WriteArtifactInput(
                            workspace_id=self._workspace_id,
                            artifact_id=move.artifact_id,
                            payload=request.payload or {},
                            provenance=move.provenance,
                        ),
                        result_type=TransitionEffects,
                        start_to_close_timeout=_WRITE_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    produced = effects.produced
                    retracted = [
                        artifact for artifact in effects.retracted if self._state.has(artifact)
                    ]
            except ActivityError as exc:
                error_type, error_message, diagnostics = _unwrap_activity_error(exc)
                await self._journal(
                    seq,
                    move,
                    status="raised",
                    error_type=error_type,
                    error_message=error_message,
                    diagnostics=diagnostics,
                )
                return self._outcome(
                    seq,
                    status="raised",
                    error_type=error_type,
                    error_message=error_message,
                    diagnostics=diagnostics,
                )

            self._state = apply_transition(self._state, produced, retracted)
            await self._journal(seq, move, status="applied", produced=produced, retracted=retracted)
            return self._outcome(seq, status="applied", produced=produced, retracted=retracted)

    @workflow.signal
    def close(self) -> None:
        self._closed = True

    # -- queries ---------------------------------------------------------

    @workflow.query
    def get_state(self) -> EpisodeState:
        return self._state

    @workflow.query
    def get_status(self) -> EpisodeStatus:
        return EpisodeStatus(
            workspace_id=self._workspace_id,
            seq=self._seq,
            state=self._state,
            artifacts=freshness_report(self._state),
            legal=legal_moves(self._state),
        )

    # -- internals -------------------------------------------------------

    def _outcome(self, seq: int, **kwargs: Any) -> MoveOutcome:
        return MoveOutcome(seq=seq, state=self._state, **kwargs)

    async def _journal(
        self,
        seq: int,
        move: Move,
        *,
        status: str,
        reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        produced: list[ArtifactVersionInfo] | None = None,
        retracted: list[ArtifactId] | None = None,
    ) -> None:
        await workflow.execute_activity(
            "journal_activity",
            JournalInput(
                workspace_id=self._workspace_id,
                seq=seq,
                move=move,
                status=status,
                reason=reason,
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics or {},
                produced=produced or [],
                retracted=retracted or [],
                state_after=self._state,
            ),
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )


def _unwrap_activity_error(exc: ActivityError) -> tuple[str, str, dict[str, Any]]:
    cause = exc.cause
    if isinstance(cause, ApplicationError):
        diagnostics: dict[str, Any] = {}
        if cause.details:
            first = cause.details[0]
            if isinstance(first, dict):
                diagnostics = first
        return cause.type or "ApplicationError", cause.message, diagnostics
    return type(cause).__name__ if cause else "ActivityError", str(cause or exc), {}
