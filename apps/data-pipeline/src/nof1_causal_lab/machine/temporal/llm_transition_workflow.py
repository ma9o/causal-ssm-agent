"""Temporal workflow for single-subroutine LLM transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.moves import TransitionEffects
    from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
    from nof1_causal_lab.machine.temporal.messages import (
        LLMSubroutineContextKind,
        LLMSubroutineInput,
        LLMSubroutineResult,
        SingleLLMTransitionFinalizeInput,
        SingleLLMTransitionId,
        SingleLLMTransitionPlan,
        SingleLLMTransitionWorkflowInput,
        TransitionRuntimeEventInput,
    )

_EVENT_TIMEOUT = timedelta(seconds=30)
_FINALIZE_TIMEOUT = timedelta(minutes=5)

_EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)


@dataclass(frozen=True)
class SingleLLMTransitionSpec:
    plan_activity: str
    finalize_activity: str
    context_kind: LLMSubroutineContextKind
    subroutine_id: str
    require_result: bool
    plan_timeout: timedelta
    summary: str


_SINGLE_LLM_TRANSITION_SPECS: dict[SingleLLMTransitionId, SingleLLMTransitionSpec] = {
    "raw_data": SingleLLMTransitionSpec(
        plan_activity="plan_raw_data_activity",
        finalize_activity="finalize_raw_data_activity",
        context_kind="raw_data_ingestion",
        subroutine_id="raw-data",
        require_result=True,
        plan_timeout=timedelta(minutes=30),
        summary="raw-data ingestion",
    ),
    "latent_structure": SingleLLMTransitionSpec(
        plan_activity="plan_latent_structure_activity",
        finalize_activity="finalize_latent_structure_activity",
        context_kind="latent_structure",
        subroutine_id="latent-structure",
        require_result=True,
        plan_timeout=timedelta(minutes=5),
        summary="latent structure",
    ),
    "measurement_structure": SingleLLMTransitionSpec(
        plan_activity="plan_measurement_structure_activity",
        finalize_activity="finalize_measurement_structure_activity",
        context_kind="measurement_structure",
        subroutine_id="measurement-structure",
        require_result=True,
        plan_timeout=timedelta(minutes=5),
        summary="measurement structure",
    ),
    "baseline_report": SingleLLMTransitionSpec(
        plan_activity="plan_baseline_report_activity",
        finalize_activity="finalize_baseline_report_activity",
        context_kind="analysis_commentary",
        subroutine_id="baseline-report",
        require_result=False,
        plan_timeout=timedelta(minutes=30),
        summary="baseline report",
    ),
}


async def _emit_single_llm_transition_event(
    workspace_id: str,
    transition_id: str,
    status: str,
    error: dict | None = None,
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


def _single_llm_failure_message(exc: BaseException) -> str:
    cause = getattr(exc, "cause", None)
    if cause is not None:
        return str(cause)
    return str(exc)


@workflow.defn
class SingleLLMTransitionWorkflow:
    @workflow.run
    async def run(self, input: SingleLLMTransitionWorkflowInput) -> TransitionEffects:
        spec = _SINGLE_LLM_TRANSITION_SPECS[input.transition_id]
        summary = spec.summary
        await _emit_single_llm_transition_event(input.workspace_id, input.transition_id, "running")
        try:
            plan = await workflow.execute_activity(
                spec.plan_activity,
                input,
                result_type=SingleLLMTransitionPlan,
                start_to_close_timeout=spec.plan_timeout,
                retry_policy=_ACTIVITY_RETRY,
                summary=f"Plan {summary}",
            )
            subroutine = await workflow.execute_child_workflow(
                LLMSubroutineWorkflow.run,
                LLMSubroutineInput(
                    workspace_id=input.workspace_id,
                    run_id=plan.run_id,
                    subroutine_id=spec.subroutine_id,
                    context_kind=spec.context_kind,
                    context_ref=plan.context_ref,
                    llm=plan.llm,
                    max_tool_turns=plan.max_tool_turns,
                    require_result=spec.require_result,
                ),
                id=(
                    f"llm-{input.transition_id.replace('_', '-')}-"
                    f"{input.workspace_id}-{input.seq:06d}"
                ),
                task_queue=workflow.info().task_queue,
                result_type=LLMSubroutineResult,
                static_summary=f"LLM {summary} subroutine",
                static_details=(
                    f"workspace={input.workspace_id}; transition={input.transition_id}; "
                    f"subroutine={spec.subroutine_id}; context={spec.context_kind}"
                ),
                memo={
                    "workspace_id": input.workspace_id,
                    "transition_id": input.transition_id,
                    "subroutine_id": spec.subroutine_id,
                    "context_kind": spec.context_kind,
                    "run_id": plan.run_id,
                },
            )
            if spec.require_result and subroutine.result_ref is None:
                raise RuntimeError(f"{summary} subroutine completed without a result ref")
            if subroutine.trace_ref is None:
                raise RuntimeError(f"{summary} subroutine completed without a trace ref")

            effects = await workflow.execute_activity(
                spec.finalize_activity,
                SingleLLMTransitionFinalizeInput(
                    workspace_id=input.workspace_id,
                    transition_id=input.transition_id,
                    state=input.state,
                    pins=plan.pins,
                    context_ref=plan.context_ref,
                    result_ref=subroutine.result_ref,
                    trace_ref=subroutine.trace_ref,
                ),
                result_type=TransitionEffects,
                start_to_close_timeout=_FINALIZE_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary=f"Finalize {summary}",
            )
        except Exception as exc:
            await _emit_single_llm_transition_event(
                input.workspace_id,
                input.transition_id,
                "failed",
                error={"message": _single_llm_failure_message(exc)},
            )
            raise

        await _emit_single_llm_transition_event(
            input.workspace_id, input.transition_id, "completed"
        )
        return effects
