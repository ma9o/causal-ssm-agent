"""Temporal workflow for the statistical-model-spec transition."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.moves import TransitionEffects
    from nof1_causal_lab.machine.temporal.client import MODEL_SPEC_SIMULATION_TASK_QUEUE
    from nof1_causal_lab.machine.temporal.messages import (
        LLMSubroutineInput,
        LLMSubroutineResult,
        StatisticalModelSpecAttemptFinalizeInput,
        StatisticalModelSpecAttemptPlan,
        StatisticalModelSpecAttemptPlanInput,
        StatisticalModelSpecAttemptResult,
        StatisticalModelSpecBarrierInput,
        StatisticalModelSpecBarrierResult,
        StatisticalModelSpecFailedEventInput,
        StatisticalModelSpecFinalizeInput,
        StatisticalModelSpecFrontierMergeInput,
        StatisticalModelSpecFrontierMergeResult,
        StatisticalModelSpecPlan,
        StatisticalModelSpecWorkflowInput,
        TransitionRuntimeError,
        TransitionRuntimeEventInput,
        TransitionRuntimeStatus,
    )

_EVENT_TIMEOUT = timedelta(seconds=30)
_ATTEMPT_PLAN_TIMEOUT = timedelta(minutes=5)
_ATTEMPT_FINALIZE_TIMEOUT = timedelta(minutes=5)
_FINALIZE_TIMEOUT = timedelta(minutes=30)
_SIMULATION_TIMEOUT = timedelta(hours=1)

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


def _model_spec_failure_details(
    exc: BaseException,
) -> tuple[str, str, UncheckedJsonObject]:
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


def _ready_constructs(
    plan: StatisticalModelSpecPlan,
    accepted_constructs: set[str],
) -> list[str]:
    """Return the deterministic ready frontier, serializing members within an SCC."""
    unit_by_id = {unit.unit_id: unit for unit in plan.units}
    ready: list[str] = []
    for unit in plan.units:
        remaining = [name for name in unit.constructs if name not in accepted_constructs]
        if not remaining:
            continue
        predecessors_admitted = all(
            all(name in accepted_constructs for name in unit_by_id[parent].constructs)
            for parent in unit.predecessors
        )
        if predecessors_admitted:
            ready.append(remaining[0])
    return ready


async def _run_construct(
    input: StatisticalModelSpecWorkflowInput,
    plan: StatisticalModelSpecPlan,
    checkpoint_ref: str,
    construct: str,
) -> tuple[str | None, str]:
    """Run one ready construct against an immutable frontier checkpoint."""
    last_outcome = "no report"
    for attempt in range(1, plan.max_attempts_per_construct + 1):
        attempt_plan = await workflow.execute_activity(
            "plan_statistical_model_spec_attempt_activity",
            StatisticalModelSpecAttemptPlanInput(
                workspace_id=input.workspace_id,
                run_id=plan.run_id,
                checkpoint_ref=checkpoint_ref,
                context_ref=plan.context_ref,
                construct_name=construct,
                attempt=attempt,
            ),
            result_type=StatisticalModelSpecAttemptPlan,
            start_to_close_timeout=_ATTEMPT_PLAN_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            summary=f"Plan model-spec {construct} attempt {attempt}",
        )
        await workflow.execute_child_workflow(
            "LLMSubroutineWorkflow",
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
                f"llm-model-spec-{input.workspace_id}-{input.seq:06d}-{attempt_plan.subroutine_id}"
            ),
            task_queue=workflow.info().task_queue,
            result_type=LLMSubroutineResult,
            static_summary=(
                f"LLM model-spec construct {attempt_plan.construct_name} attempt {attempt}"
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
        attempt_result = await workflow.execute_activity(
            "finalize_statistical_model_spec_attempt_activity",
            StatisticalModelSpecAttemptFinalizeInput(
                result_ref=attempt_plan.result_ref,
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
            if attempt_result.checkpoint_ref is None:
                raise RuntimeError(
                    f"admitted model-spec construct `{construct}` produced no checkpoint"
                )
            return attempt_result.checkpoint_ref, last_outcome
    return None, last_outcome


@workflow.defn
class StatisticalModelSpecWorkflow:
    @workflow.run
    async def run(self, input: StatisticalModelSpecWorkflowInput) -> TransitionEffects:
        await _emit_model_spec_transition_event(
            input.workspace_id, "statistical_model_spec", "running"
        )
        current_construct: str | None = None
        checkpoint_ref: str | None = None
        plan: StatisticalModelSpecPlan | None = None
        try:
            plan = await workflow.execute_activity(
                "plan_statistical_model_spec_activity",
                input,
                result_type=StatisticalModelSpecPlan,
                task_queue=MODEL_SPEC_SIMULATION_TASK_QUEUE,
                start_to_close_timeout=_SIMULATION_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Plan statistical model spec",
            )
            checkpoint_ref = plan.checkpoint_ref
            accepted_constructs = set(plan.accepted_constructs)
            construct_order = [construct for unit in plan.units for construct in unit.constructs]
            order_index = {construct: index for index, construct in enumerate(construct_order)}
            barrier_repairs = 0
            while True:
                tasks: dict[str, asyncio.Task[tuple[str | None, str]]] = {}
                failures: list[tuple[str, str]] = []
                while len(accepted_constructs) < len(construct_order):
                    if not failures:
                        for construct in _ready_constructs(plan, accepted_constructs):
                            if construct not in tasks:
                                tasks[construct] = asyncio.create_task(
                                    _run_construct(input, plan, checkpoint_ref, construct)
                                )
                    if not tasks:
                        if failures:
                            break
                        missing = [
                            name for name in construct_order if name not in accepted_constructs
                        ]
                        raise RuntimeError(
                            "model-spec topology has no ready construct while work remains: "
                            + ", ".join(missing)
                        )
                    done, _pending = await asyncio.wait(
                        tuple(tasks.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    finished = sorted(
                        [(construct, task) for construct, task in tasks.items() if task in done],
                        key=lambda item: order_index[item[0]],
                    )
                    branch_refs: list[str] = []
                    for construct, task in finished:
                        del tasks[construct]
                        task_error = task.exception()
                        if task_error is not None:
                            failures.append((construct, str(task_error)))
                            continue
                        task_result = task.result()
                        branch_ref, last_outcome = task_result
                        if branch_ref is None:
                            failures.append((construct, last_outcome))
                        else:
                            branch_refs.append(branch_ref)

                    if branch_refs:
                        merged = await workflow.execute_activity(
                            "merge_statistical_model_spec_frontier_activity",
                            StatisticalModelSpecFrontierMergeInput(
                                workspace_id=input.workspace_id,
                                checkpoint_ref=checkpoint_ref,
                                branch_checkpoint_refs=branch_refs,
                                construct_order=construct_order,
                            ),
                            result_type=StatisticalModelSpecFrontierMergeResult,
                            start_to_close_timeout=_ATTEMPT_FINALIZE_TIMEOUT,
                            retry_policy=_ACTIVITY_RETRY,
                            summary="Merge model-spec ready frontier",
                        )
                        checkpoint_ref = merged.checkpoint_ref
                        accepted_constructs = set(merged.accepted_constructs)

                if failures:
                    current_construct, last_outcome = min(
                        failures,
                        key=lambda item: order_index[item[0]],
                    )
                    raise RuntimeError(
                        f"model-spec construct `{current_construct}` was not admitted after "
                        f"{plan.max_attempts_per_construct} attempts "
                        f"(last outcome: {last_outcome})."
                    )
                current_construct = None

                barrier = await workflow.execute_activity(
                    "validate_statistical_model_spec_barrier_activity",
                    StatisticalModelSpecBarrierInput(
                        workspace_id=input.workspace_id,
                        checkpoint_ref=checkpoint_ref,
                        context_ref=plan.context_ref,
                        construct_order=construct_order,
                    ),
                    result_type=StatisticalModelSpecBarrierResult,
                    task_queue=MODEL_SPEC_SIMULATION_TASK_QUEUE,
                    start_to_close_timeout=_SIMULATION_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    summary="Validate exact full model-spec barrier",
                )
                checkpoint_ref = barrier.checkpoint_ref
                accepted_constructs = set(barrier.accepted_constructs)
                if barrier.passed:
                    break
                barrier_repairs += 1
                if barrier_repairs >= plan.max_attempts_per_construct:
                    current_construct = (
                        barrier.reopened_constructs[0] if barrier.reopened_constructs else None
                    )
                    raise RuntimeError(
                        "model-spec full-model barrier did not pass after "
                        f"{barrier_repairs} repair cycles"
                    )

            effects = await workflow.execute_activity(
                "finalize_statistical_model_spec_activity",
                StatisticalModelSpecFinalizeInput(
                    workspace_id=input.workspace_id,
                    run_id=plan.run_id,
                    state=input.state,
                    pins=plan.pins,
                    checkpoint_ref=checkpoint_ref,
                    context_ref=plan.context_ref,
                ),
                result_type=TransitionEffects,
                start_to_close_timeout=_FINALIZE_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Finalize statistical model spec",
            )
        except Exception as exc:
            failure_type, failure_message, diagnostics = _model_spec_failure_details(exc)
            if current_construct is not None:
                await workflow.execute_activity(
                    "emit_model_spec_failed_event_activity",
                    StatisticalModelSpecFailedEventInput(
                        workspace_id=input.workspace_id,
                        construct_name=current_construct,
                        message=failure_message,
                        checkpoint_ref=checkpoint_ref,
                    ),
                    start_to_close_timeout=_EVENT_TIMEOUT,
                    retry_policy=_EVENT_RETRY,
                    summary="Emit model-spec admission failure",
                )
            await _emit_model_spec_transition_event(
                input.workspace_id,
                "statistical_model_spec",
                "failed",
                error=TransitionRuntimeError(type=failure_type, message=failure_message),
            )
            diagnostics.update(
                {
                    "transition_id": "statistical_model_spec",
                    "construct": current_construct,
                }
            )
            if checkpoint_ref is not None:
                if plan is None:
                    raise RuntimeError("model-spec checkpoint exists without a run plan") from exc
                diagnostics["resume"] = {
                    "kind": "model_spec",
                    "run_id": plan.run_id,
                    "checkpoint_id": checkpoint_ref.rsplit("/", 1)[-1],
                }
            raise ApplicationError(
                failure_message,
                diagnostics,
                type=failure_type,
                non_retryable=True,
            ) from exc

        await _emit_model_spec_transition_event(
            input.workspace_id, "statistical_model_spec", "completed"
        )
        return effects
