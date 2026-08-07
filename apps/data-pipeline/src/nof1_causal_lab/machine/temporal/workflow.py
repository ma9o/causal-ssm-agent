"""The episode entity workflow: durable shell around the pure machine.

One workflow per workspace episode. Local state (current artifact
versions, move counter) survives crashes and redeploys by replay; the
``propose`` update is the single entry point for moves. It always
accepts and validates *inside* the handler — a Temporal update-validator
rejection would leave no trace in history, and rejected proposals are
exactly what the timeline scrubber wants to show. Every attempt
(applied, rejected, raised) is projected into the episode journal by an
activity before the outcome returns to the caller.

Handlers stay thin (validate → execute activity → journal → apply) so
workflow-code versioning churn stays small; all semantics live in the
pure functions, all I/O in activities (referenced by name so the
sandbox never imports storage/polars/jax).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, TypeGuard

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, ChildWorkflowError

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.artifacts import (
        ArtifactVersionInfo,
        EpisodeState,
    )
    from nof1_causal_lab.machine.moves import (
        Move,
        RetractedArtifact,
        RunArtifact,
        TransitionEffects,
        WriteArtifact,
        apply_transition,
        freshness_report,
        legal_moves,
        validate_move,
    )
    from nof1_causal_lab.machine.store import ResumeRef
    from nof1_causal_lab.machine.temporal.messages import (
        EpisodeInit,
        EpisodeStatus,
        JournalInput,
        JournalStatus,
        MeasurementsWorkflowInput,
        MoveOutcome,
        MoveRequest,
        RunArtifactInput,
        SingleLLMTransitionId,
        SingleLLMTransitionWorkflowInput,
        StatisticalModelSpecWorkflowInput,
        WriteArtifactInput,
    )

_RUN_TRANSITION_TIMEOUT = timedelta(hours=4)
_WRITE_TIMEOUT = timedelta(minutes=5)
_JOURNAL_TIMEOUT = timedelta(minutes=1)
_RUN_COLLECTION_TIMEOUT = timedelta(minutes=10)
_SINGLE_LLM_TRANSITIONS: frozenset[SingleLLMTransitionId] = frozenset(
    {
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "baseline_report",
    }
)


def _is_single_llm_transition(artifact_id: str) -> TypeGuard[SingleLLMTransitionId]:
    return artifact_id in _SINGLE_LLM_TRANSITIONS


_NON_RETRYABLE_ERRORS = [
    "TransitionExecutionError",
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
        # Seed from durable applied transition effects when resuming a lost
        # workflow; empty state / seq 0 for a new episode. The facade
        # reconstructs the seed (I/O can't happen here in the deterministic
        # workflow) and passes it as a start argument, so replay stays
        # deterministic.
        self._state = init.initial_state if init.initial_state is not None else EpisodeState()
        self._seq = init.initial_seq
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
                if isinstance(move, RunArtifact) and _is_single_llm_transition(move.artifact_id):
                    effects = await workflow.execute_child_workflow(
                        "SingleLLMTransitionWorkflow",
                        SingleLLMTransitionWorkflowInput(
                            workspace_id=self._workspace_id,
                            seq=seq,
                            transition_id=move.artifact_id,
                            state=self._state,
                            options=request.options,
                        ),
                        id=(f"{move.artifact_id.replace('_', '-')}-{self._workspace_id}-{seq:06d}"),
                        result_type=TransitionEffects,
                        execution_timeout=_RUN_TRANSITION_TIMEOUT,
                        static_summary=f"Run {move.artifact_id}",
                        static_details=(
                            f"workspace={self._workspace_id}; seq={seq}; "
                            f"artifact={move.artifact_id}; workflow=single_llm_transition"
                        ),
                        memo={
                            "workspace_id": self._workspace_id,
                            "seq": seq,
                            "artifact_id": move.artifact_id,
                            "workflow_kind": "single_llm_transition",
                        },
                    )
                elif isinstance(move, RunArtifact) and move.artifact_id == "measurements":
                    effects = await workflow.execute_child_workflow(
                        "MeasurementsWorkflow",
                        MeasurementsWorkflowInput(
                            workspace_id=self._workspace_id,
                            seq=seq,
                            state=self._state,
                            options=request.options,
                        ),
                        id=f"measurements-{self._workspace_id}-{seq:06d}",
                        result_type=TransitionEffects,
                        execution_timeout=_RUN_TRANSITION_TIMEOUT,
                        static_summary="Run measurements",
                        static_details=(
                            f"workspace={self._workspace_id}; seq={seq}; "
                            "artifact=measurements; workflow=batch_llm_transition"
                        ),
                        memo={
                            "workspace_id": self._workspace_id,
                            "seq": seq,
                            "artifact_id": "measurements",
                            "workflow_kind": "batch_llm_transition",
                        },
                    )
                elif isinstance(move, RunArtifact) and move.artifact_id == "statistical_model_spec":
                    effects = await workflow.execute_child_workflow(
                        "StatisticalModelSpecWorkflow",
                        StatisticalModelSpecWorkflowInput(
                            workspace_id=self._workspace_id,
                            seq=seq,
                            state=self._state,
                            options=request.options,
                        ),
                        id=f"statistical-model-spec-{self._workspace_id}-{seq:06d}",
                        result_type=TransitionEffects,
                        execution_timeout=_RUN_TRANSITION_TIMEOUT,
                        static_summary="Run statistical model spec",
                        static_details=(
                            f"workspace={self._workspace_id}; seq={seq}; "
                            "artifact=statistical_model_spec; workflow=construct_admission"
                        ),
                        memo={
                            "workspace_id": self._workspace_id,
                            "seq": seq,
                            "artifact_id": "statistical_model_spec",
                            "workflow_kind": "construct_admission",
                        },
                    )
                elif isinstance(move, RunArtifact):
                    effects = await workflow.execute_activity(
                        "run_transition_activity",
                        RunArtifactInput(
                            workspace_id=self._workspace_id,
                            artifact_id=move.artifact_id,
                            state=self._state,
                            options=request.options,
                        ),
                        result_type=TransitionEffects,
                        start_to_close_timeout=_RUN_TRANSITION_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                else:
                    effects = await workflow.execute_activity(
                        "write_artifact_activity",
                        WriteArtifactInput(
                            workspace_id=self._workspace_id,
                            artifact_id=move.artifact_id,
                            payload=request.payload or {},
                            provenance=move.provenance,
                            state=self._state,
                        ),
                        result_type=TransitionEffects,
                        start_to_close_timeout=_WRITE_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                produced = effects.produced
                retracted = effects.retracted
            except (ActivityError, ChildWorkflowError) as exc:
                error_type, error_message, diagnostics, resume = _unwrap_temporal_failure(exc)
                await self._journal(
                    seq,
                    move,
                    status="raised",
                    error_type=error_type,
                    error_message=error_message,
                    diagnostics=diagnostics,
                    resume=resume,
                )
                return self._outcome(
                    seq,
                    status="raised",
                    error_type=error_type,
                    error_message=error_message,
                    diagnostics=diagnostics,
                )

            await self._journal(
                seq,
                move,
                status="applied",
                produced=produced,
                retracted=retracted,
            )
            self._state = apply_transition(self._state, produced, retracted)
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
        status: JournalStatus,
        reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        diagnostics: UncheckedJsonObject | None = None,
        produced: list[ArtifactVersionInfo] | None = None,
        retracted: list[RetractedArtifact] | None = None,
        resume: ResumeRef | None = None,
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
                resume=resume,
            ),
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        # Run collection is lifecycle hygiene, never part of the move commit.
        try:
            await workflow.execute_activity(
                "collect_completed_runs_activity",
                self._workspace_id,
                start_to_close_timeout=_RUN_COLLECTION_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except ActivityError as exc:
            workflow.logger.warning("run scratch collection failed after seq %d: %s", seq, exc)


def _unwrap_temporal_failure(
    exc: ActivityError | ChildWorkflowError,
) -> tuple[str, str, UncheckedJsonObject, ResumeRef | None]:
    cause = exc.cause
    while isinstance(cause, (ActivityError, ChildWorkflowError)):
        cause = cause.cause
    if isinstance(cause, ApplicationError):
        diagnostics: UncheckedJsonObject = {}
        if cause.details:
            first = cause.details[0]
            if isinstance(first, dict):
                diagnostics = dict(first)
        resume_payload = diagnostics.pop("resume", None)
        resume = ResumeRef.model_validate(resume_payload) if resume_payload is not None else None
        return cause.type or "ApplicationError", cause.message, diagnostics, resume
    return type(cause).__name__ if cause else "ActivityError", str(cause or exc), {}, None
