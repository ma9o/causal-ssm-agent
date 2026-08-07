"""Temporal workflow for single-subroutine LLM transitions."""

from __future__ import annotations

from datetime import timedelta
from typing import assert_never

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    # Temporal resolves workflow result annotations when registering the class.
    from nof1_causal_lab.machine.moves import TransitionEffects  # noqa: TC001
    from nof1_causal_lab.machine.temporal.baseline_report_activities import (
        finalize_baseline_report_activity,
        plan_baseline_report_activity,
    )
    from nof1_causal_lab.machine.temporal.latent_structure_activities import (
        finalize_latent_structure_activity,
        plan_latent_structure_activity,
    )
    from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
    from nof1_causal_lab.machine.temporal.measurement_structure_activities import (
        finalize_measurement_structure_activity,
        plan_measurement_structure_activity,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        LLMSubroutineContextKind,
        LLMSubroutineInput,
        SingleLLMTransitionFinalizeInput,
        SingleLLMTransitionWorkflowInput,
        TransitionRuntimeError,
    )
    from nof1_causal_lab.machine.temporal.raw_data_activities import (
        finalize_raw_data_activity,
        plan_raw_data_activity,
    )
    from nof1_causal_lab.machine.temporal.workflow_support import (
        emit_transition_runtime_event,
        temporal_failure_details,
    )

_FINALIZE_TIMEOUT = timedelta(minutes=5)

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)


async def _run_single_llm_transition(
    input: SingleLLMTransitionWorkflowInput,
) -> TransitionEffects:
    await emit_transition_runtime_event(input.workspace_id, input.transition_id, "running")
    try:
        context_kind: LLMSubroutineContextKind
        match input.transition_id:
            case "raw_data":
                context_kind = "raw_data_ingestion"
                subroutine_id = "raw-data"
                require_result = True
                summary = "raw-data ingestion"
                plan = await workflow.execute_activity(
                    plan_raw_data_activity,
                    input,
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Plan raw-data ingestion",
                )
            case "latent_structure":
                context_kind = "latent_structure"
                subroutine_id = "latent-structure"
                require_result = True
                summary = "latent structure"
                plan = await workflow.execute_activity(
                    plan_latent_structure_activity,
                    input,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Plan latent structure",
                )
            case "measurement_structure":
                context_kind = "measurement_structure"
                subroutine_id = "measurement-structure"
                require_result = True
                summary = "measurement structure"
                plan = await workflow.execute_activity(
                    plan_measurement_structure_activity,
                    input,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Plan measurement structure",
                )
            case "baseline_report":
                context_kind = "analysis_commentary"
                subroutine_id = "baseline-report"
                require_result = False
                summary = "baseline report"
                plan = await workflow.execute_activity(
                    plan_baseline_report_activity,
                    input,
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Plan baseline report",
                )
            case unsupported:
                assert_never(unsupported)

        subroutine = await workflow.execute_child_workflow(
            LLMSubroutineWorkflow.run,
            LLMSubroutineInput(
                workspace_id=input.workspace_id,
                run_id=plan.run_id,
                subroutine_id=subroutine_id,
                context_kind=context_kind,
                context_ref=plan.context_ref,
                llm=plan.llm,
                max_tool_turns=plan.max_tool_turns,
                require_result=require_result,
            ),
            id=(
                f"llm-{input.transition_id.replace('_', '-')}-{input.workspace_id}-{input.seq:06d}"
            ),
            task_queue=workflow.info().task_queue,
            static_summary=f"LLM {summary} subroutine",
            static_details=(
                f"workspace={input.workspace_id}; transition={input.transition_id}; "
                f"subroutine={subroutine_id}; context={context_kind}"
            ),
            memo={
                "workspace_id": input.workspace_id,
                "transition_id": input.transition_id,
                "subroutine_id": subroutine_id,
                "context_kind": context_kind,
                "run_id": plan.run_id,
            },
        )
        if require_result and subroutine.result_ref is None:
            raise RuntimeError(f"{summary} subroutine completed without a result ref")
        finalize_input = SingleLLMTransitionFinalizeInput(
            workspace_id=input.workspace_id,
            transition_id=input.transition_id,
            state=input.state,
            pins=plan.pins,
            context_ref=plan.context_ref,
            result_ref=subroutine.result_ref,
            trace_ref=subroutine.trace_ref,
        )
        match input.transition_id:
            case "raw_data":
                effects = await workflow.execute_activity(
                    finalize_raw_data_activity,
                    finalize_input,
                    start_to_close_timeout=_FINALIZE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Finalize raw-data ingestion",
                )
            case "latent_structure":
                effects = await workflow.execute_activity(
                    finalize_latent_structure_activity,
                    finalize_input,
                    start_to_close_timeout=_FINALIZE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Finalize latent structure",
                )
            case "measurement_structure":
                effects = await workflow.execute_activity(
                    finalize_measurement_structure_activity,
                    finalize_input,
                    start_to_close_timeout=_FINALIZE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Finalize measurement structure",
                )
            case "baseline_report":
                effects = await workflow.execute_activity(
                    finalize_baseline_report_activity,
                    finalize_input,
                    start_to_close_timeout=_FINALIZE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Finalize baseline report",
                )
            case unsupported:
                assert_never(unsupported)
    except Exception as exc:
        failure_type, failure_message, _ = temporal_failure_details(exc)
        await emit_transition_runtime_event(
            input.workspace_id,
            input.transition_id,
            "failed",
            error=TransitionRuntimeError(type=failure_type, message=failure_message),
        )
        raise

    await emit_transition_runtime_event(input.workspace_id, input.transition_id, "completed")
    return effects


@workflow.defn
class SingleLLMTransitionWorkflow:
    @workflow.run
    async def run(self, input: SingleLLMTransitionWorkflowInput) -> TransitionEffects:
        return await _run_single_llm_transition(input)
