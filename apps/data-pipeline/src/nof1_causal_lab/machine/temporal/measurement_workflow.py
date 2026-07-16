"""Temporal workflows for the measurements transition."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, ChildWorkflowError

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.moves import TransitionEffects
    from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
    from nof1_causal_lab.machine.temporal.messages import (
        ExtractionChunkFinalizeInput,
        ExtractionChunkResult,
        ExtractionChunkWorkflowInput,
        ExtractionProgressEventInput,
        ExtractionProgressSnapshot,
        LLMSubroutineInput,
        LLMSubroutineResult,
        MeasurementChunkRef,
        MeasurementsFinalizeInput,
        MeasurementsPlan,
        MeasurementsWorkflowInput,
        TransitionRuntimeError,
        TransitionRuntimeEventInput,
        TransitionRuntimeStatus,
    )

_EVENT_TIMEOUT = timedelta(seconds=30)
_PLAN_TIMEOUT = timedelta(minutes=30)
_FINALIZE_CHUNK_TIMEOUT = timedelta(minutes=5)
_FINALIZE_MEASUREMENTS_TIMEOUT = timedelta(minutes=30)

_EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)
_CHUNK_WORKFLOW_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=1.0,
    maximum_attempts=3,
)


async def _emit_transition_event(
    workspace_id: str,
    transition_id: str,
    status: TransitionRuntimeStatus,
    error: TransitionRuntimeError | None = None,
) -> None:
    await workflow.execute_activity(
        "emit_transition_runtime_event_activity",
        TransitionRuntimeEventInput(
            workspace_id=workspace_id,
            transition_id=transition_id,
            status=status,
            error=error,
        ),
        start_to_close_timeout=_EVENT_TIMEOUT,
        retry_policy=_EVENT_RETRY,
    )


async def _emit_extraction_event(input: ExtractionProgressEventInput) -> None:
    await workflow.execute_activity(
        "emit_extraction_progress_event_activity",
        input,
        start_to_close_timeout=_EVENT_TIMEOUT,
        retry_policy=_EVENT_RETRY,
    )


def _failure_details(exc: BaseException) -> tuple[str, str, dict]:
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


@workflow.defn
class ExtractionChunkWorkflow:
    @workflow.run
    async def run(self, input: ExtractionChunkWorkflowInput) -> ExtractionChunkResult:
        attempt = workflow.info().attempt
        subroutine_id = f"measurement-chunk-{input.worker_id:06d}-attempt-{attempt:03d}"
        subroutine = await workflow.execute_child_workflow(
            LLMSubroutineWorkflow.run,
            LLMSubroutineInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                subroutine_id=subroutine_id,
                context_kind="measurement_extraction",
                context_ref=input.spec_ref,
                llm=input.llm,
                max_tool_turns=input.max_tool_turns,
                require_result=True,
            ),
            id=(
                f"llm-measurement-{input.workspace_id}-{input.run_id}-"
                f"chunk-{input.worker_id:06d}-attempt-{attempt:03d}"
            ),
            task_queue=workflow.info().task_queue,
            result_type=LLMSubroutineResult,
            static_summary=f"LLM extraction subroutine chunk {input.worker_id}",
            static_details=(
                f"workspace={input.workspace_id}; run={input.run_id}; "
                f"chunk={input.worker_id}; attempt={attempt}; "
                "context=measurement_extraction"
            ),
            memo={
                "workspace_id": input.workspace_id,
                "run_id": input.run_id,
                "worker_id": input.worker_id,
                "attempt": attempt,
                "context_kind": "measurement_extraction",
                "subroutine_id": subroutine_id,
            },
        )
        if subroutine.result_ref is None:
            raise RuntimeError("measurement extraction subroutine completed without a result ref")
        return await workflow.execute_activity(
            "finalize_extraction_chunk_activity",
            ExtractionChunkFinalizeInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                worker_id=input.worker_id,
                attempt=attempt,
                n_windows=input.n_windows,
                result_ref=subroutine.result_ref,
                conversation_ref=subroutine.conversation_ref,
                n_llm_calls=subroutine.n_llm_calls,
            ),
            result_type=ExtractionChunkResult,
            start_to_close_timeout=_FINALIZE_CHUNK_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            summary=f"Finalize extraction chunk {input.worker_id}",
        )


@workflow.defn
class MeasurementsWorkflow:
    @workflow.run
    async def run(self, input: MeasurementsWorkflowInput) -> TransitionEffects:
        chunk_results: list[ExtractionChunkResult] = []
        await _emit_transition_event(input.workspace_id, "measurements", "running")
        try:
            plan = await workflow.execute_activity(
                "plan_measurements_activity",
                input,
                result_type=MeasurementsPlan,
                start_to_close_timeout=_PLAN_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Plan measurements extraction",
            )
            await _emit_extraction_event(
                ExtractionProgressEventInput(
                    workspace_id=input.workspace_id,
                    kind="plan",
                    total_workers=len(plan.chunks),
                    max_concurrent_workers=plan.max_concurrent_workers,
                    max_rpm=plan.max_rpm,
                )
            )

            total_workers = len(plan.chunks)
            pending_workers = total_workers
            running_workers = 0
            completed_workers = 0
            failed_workers = 0
            llm_call_times = []

            async def emit_snapshot() -> None:
                now = workflow.now()
                cutoff = now - timedelta(seconds=60)
                llm_call_times[:] = [ts for ts in llm_call_times if ts >= cutoff]
                await _emit_extraction_event(
                    ExtractionProgressEventInput(
                        workspace_id=input.workspace_id,
                        kind="snapshot",
                        snapshot=ExtractionProgressSnapshot(
                            total_workers=total_workers,
                            pending_workers=pending_workers,
                            running_workers=running_workers,
                            completed_workers=completed_workers,
                            failed_workers=failed_workers,
                            llm_requests_last_60s=len(llm_call_times),
                        ),
                    )
                )

            await emit_snapshot()

            semaphore = asyncio.Semaphore(max(1, plan.max_concurrent_workers))

            async def run_chunk(chunk: MeasurementChunkRef) -> ExtractionChunkResult:
                nonlocal pending_workers, running_workers, completed_workers, failed_workers
                async with semaphore:
                    pending_workers -= 1
                    running_workers += 1
                    await _emit_extraction_event(
                        ExtractionProgressEventInput(
                            workspace_id=input.workspace_id,
                            kind="worker",
                            worker_id=chunk.worker_id,
                            state="running",
                            n_windows=chunk.n_windows,
                        )
                    )
                    await emit_snapshot()

                    try:
                        result = await workflow.execute_child_workflow(
                            ExtractionChunkWorkflow.run,
                            ExtractionChunkWorkflowInput(
                                workspace_id=input.workspace_id,
                                run_id=plan.run_id,
                                worker_id=chunk.worker_id,
                                n_windows=chunk.n_windows,
                                spec_ref=chunk.spec_ref,
                                attempt=1,
                                llm=plan.llm,
                                max_tool_turns=plan.max_tool_turns,
                            ),
                            id=(
                                f"measurements-{input.workspace_id}-"
                                f"{input.seq:06d}-chunk-{chunk.worker_id:06d}"
                            ),
                            task_queue=workflow.info().task_queue,
                            result_type=ExtractionChunkResult,
                            retry_policy=_CHUNK_WORKFLOW_RETRY,
                            static_summary=f"Extract measurements chunk {chunk.worker_id}",
                            static_details=(
                                f"workspace={input.workspace_id}; run={plan.run_id}; "
                                f"chunk={chunk.worker_id}; windows={chunk.n_windows}"
                            ),
                            memo={
                                "workspace_id": input.workspace_id,
                                "run_id": plan.run_id,
                                "worker_id": chunk.worker_id,
                                "n_windows": chunk.n_windows,
                                "workflow_kind": "measurement_chunk",
                            },
                        )
                    except ChildWorkflowError as exc:
                        _, failure_message, _ = _failure_details(exc)
                        result = ExtractionChunkResult(
                            worker_id=chunk.worker_id,
                            status="failed",
                            n_extractions=0,
                            n_windows=chunk.n_windows,
                            error=failure_message,
                        )

                    running_workers -= 1
                    if result.status == "completed":
                        completed_workers += 1
                    else:
                        failed_workers += 1
                    now = workflow.now()
                    llm_call_times.extend([now] * result.n_llm_calls)
                    await _emit_extraction_event(
                        ExtractionProgressEventInput(
                            workspace_id=input.workspace_id,
                            kind="worker",
                            worker_id=result.worker_id,
                            state=result.status,
                            n_windows=result.n_windows,
                            n_extractions=result.n_extractions,
                            n_llm_calls=result.n_llm_calls or None,
                            error=result.error,
                        )
                    )
                    await emit_snapshot()
                    return result

            tasks = [asyncio.create_task(run_chunk(chunk)) for chunk in plan.chunks]
            chunk_results = [await task for task in workflow.as_completed(tasks)]
            chunk_results.sort(key=lambda result: result.worker_id)

            effects = await workflow.execute_activity(
                "finalize_measurements_activity",
                MeasurementsFinalizeInput(
                    workspace_id=input.workspace_id,
                    state=input.state,
                    run_id=plan.run_id,
                    plan_ref=plan.plan_ref,
                    pins=plan.pins,
                    chunk_results=chunk_results,
                ),
                result_type=TransitionEffects,
                start_to_close_timeout=_FINALIZE_MEASUREMENTS_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Finalize measurements artifacts",
            )
            await _emit_transition_event(input.workspace_id, "measurements", "completed")
            return effects
        except Exception as exc:
            failure_type, failure_message, _ = _failure_details(exc)
            await _emit_transition_event(
                input.workspace_id,
                "measurements",
                "failed",
                error=TransitionRuntimeError(type=failure_type, message=failure_message),
            )
            raise
