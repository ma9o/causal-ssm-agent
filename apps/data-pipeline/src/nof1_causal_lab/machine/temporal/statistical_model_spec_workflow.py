"""Temporal workflow for the statistical-model-spec transition."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.moves import TransitionEffects
    from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
    from nof1_causal_lab.machine.temporal.messages import (
        LLMSubroutineInput,
        LLMSubroutineResult,
        StatisticalModelSpecAttemptFinalizeInput,
        StatisticalModelSpecAttemptPlan,
        StatisticalModelSpecAttemptPlanInput,
        StatisticalModelSpecAttemptResult,
        StatisticalModelSpecFailedEventInput,
        StatisticalModelSpecFinalizeInput,
        StatisticalModelSpecPlan,
        StatisticalModelSpecWorkflowInput,
        TransitionRuntimeEventInput,
    )

_EVENT_TIMEOUT = timedelta(seconds=30)
_PLAN_TIMEOUT = timedelta(minutes=30)
_ATTEMPT_PLAN_TIMEOUT = timedelta(minutes=5)
_ATTEMPT_FINALIZE_TIMEOUT = timedelta(minutes=5)
_FINALIZE_TIMEOUT = timedelta(minutes=30)

_EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)


async def _emit_model_spec_transition_event(
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


def _model_spec_failure_message(exc: BaseException) -> str:
    cause = getattr(exc, "cause", None)
    if cause is not None:
        return str(cause)
    return str(exc)


@workflow.defn
class StatisticalModelSpecWorkflow:
    @workflow.run
    async def run(self, input: StatisticalModelSpecWorkflowInput) -> TransitionEffects:
        await _emit_model_spec_transition_event(
            input.workspace_id, "statistical_model_spec", "running"
        )
        trace_refs: list[str] = []
        current_construct: str | None = None
        try:
            plan = await workflow.execute_activity(
                "plan_statistical_model_spec_activity",
                input,
                result_type=StatisticalModelSpecPlan,
                start_to_close_timeout=_PLAN_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Plan statistical model spec",
            )

            for construct in plan.order:
                current_construct = construct
                admitted = False
                last_outcome = "no report"
                for attempt in range(1, plan.max_attempts_per_construct + 1):
                    attempt_plan = await workflow.execute_activity(
                        "plan_statistical_model_spec_attempt_activity",
                        StatisticalModelSpecAttemptPlanInput(
                            workspace_id=input.workspace_id,
                            run_id=plan.run_id,
                            state_ref=plan.state_ref,
                            context_ref=plan.context_ref,
                            attempt=attempt,
                        ),
                        result_type=StatisticalModelSpecAttemptPlan,
                        start_to_close_timeout=_ATTEMPT_PLAN_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                        summary=f"Plan model-spec {construct} attempt {attempt}",
                    )
                    subroutine = await workflow.execute_child_workflow(
                        LLMSubroutineWorkflow.run,
                        LLMSubroutineInput(
                            workspace_id=input.workspace_id,
                            run_id=plan.run_id,
                            subroutine_id=attempt_plan.subroutine_id,
                            context_kind="model_spec_construct",
                            context_ref=attempt_plan.context_ref,
                            llm=plan.llm,
                            max_tool_turns=plan.max_tool_turns,
                            require_result=False,
                        ),
                        id=(
                            f"llm-model-spec-{input.workspace_id}-{input.seq:06d}-"
                            f"{attempt_plan.subroutine_id}"
                        ),
                        task_queue=workflow.info().task_queue,
                        result_type=LLMSubroutineResult,
                        static_summary=(
                            f"LLM model-spec construct {attempt_plan.construct_name} attempt "
                            f"{attempt}"
                        ),
                        static_details=(
                            f"workspace={input.workspace_id}; run={plan.run_id}; "
                            f"construct={attempt_plan.construct_name}; attempt={attempt}; "
                            "context=model_spec_construct"
                        ),
                        memo={
                            "workspace_id": input.workspace_id,
                            "run_id": plan.run_id,
                            "construct_name": attempt_plan.construct_name,
                            "attempt": attempt,
                            "context_kind": "model_spec_construct",
                            "subroutine_id": attempt_plan.subroutine_id,
                        },
                    )
                    if subroutine.trace_ref is not None:
                        trace_refs.append(subroutine.trace_ref)
                    attempt_result = await workflow.execute_activity(
                        "finalize_statistical_model_spec_attempt_activity",
                        StatisticalModelSpecAttemptFinalizeInput(
                            state_ref=plan.state_ref,
                            construct_name=attempt_plan.construct_name,
                            attempt=attempt,
                        ),
                        result_type=StatisticalModelSpecAttemptResult,
                        start_to_close_timeout=_ATTEMPT_FINALIZE_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                        summary=f"Finalize model-spec {construct} attempt {attempt}",
                    )
                    last_outcome = attempt_result.outcome
                    if attempt_result.admitted:
                        admitted = True
                        break

                if not admitted:
                    raise RuntimeError(
                        f"model-spec construct `{construct}` was not admitted after "
                        f"{plan.max_attempts_per_construct} attempts "
                        f"(last outcome: {last_outcome})."
                    )
                current_construct = None

            effects = await workflow.execute_activity(
                "finalize_statistical_model_spec_activity",
                StatisticalModelSpecFinalizeInput(
                    workspace_id=input.workspace_id,
                    state=input.state,
                    pins=plan.pins,
                    state_ref=plan.state_ref,
                    context_ref=plan.context_ref,
                    trace_refs=trace_refs,
                ),
                result_type=TransitionEffects,
                start_to_close_timeout=_FINALIZE_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Finalize statistical model spec",
            )
        except Exception as exc:
            failure_message = _model_spec_failure_message(exc)
            if current_construct is not None:
                await workflow.execute_activity(
                    "emit_model_spec_failed_event_activity",
                    StatisticalModelSpecFailedEventInput(
                        workspace_id=input.workspace_id,
                        construct_name=current_construct,
                        message=failure_message,
                    ),
                    start_to_close_timeout=_EVENT_TIMEOUT,
                    retry_policy=_EVENT_RETRY,
                    summary="Emit model-spec admission failure",
                )
            await _emit_model_spec_transition_event(
                input.workspace_id,
                "statistical_model_spec",
                "failed",
                error={"message": failure_message},
            )
            raise

        await _emit_model_spec_transition_event(
            input.workspace_id, "statistical_model_spec", "completed"
        )
        return effects
