"""Temporal workflow for one tool-validated LLM subroutine."""

from __future__ import annotations

from datetime import timedelta
from typing import NotRequired, TypedDict

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.machine.temporal.client import (
        HARNESS_CLAUDE_TASK_QUEUE,
        HARNESS_CODEX_TASK_QUEUE,
        HARNESS_PI_TASK_QUEUE,
        MODEL_SPEC_SIMULATION_TASK_QUEUE,
        OPENROUTER_TASK_QUEUE,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        AppendLLMRepairMessageInput,
        AppendLLMRepairMessageResult,
        AppendLLMUserMessageInput,
        AppendLLMUserMessageResult,
        HarnessToolExecutionResult,
        HarnessToolRequest,
        HarnessTurnInput,
        HarnessTurnResult,
        LLMBackendConfig,
        LLMSubroutineInput,
        LLMSubroutineResult,
        LLMSubroutineStart,
        LLMSubroutineStartInput,
        LLMSubroutineTraceInput,
        LLMSubroutineTraceResult,
        LLMToolExecutionInput,
        LLMToolExecutionResult,
        LLMToolSpec,
        OpenRouterCallInput,
        OpenRouterCallResult,
        OpenRouterLLMConfig,
    )

_LOCAL_TIMEOUT = timedelta(minutes=5)
_SIMULATION_TIMEOUT = timedelta(hours=1)
_TRACE_TIMEOUT = timedelta(minutes=5)
_LOCAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)
_OPENROUTER_CALL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=1.0,
    maximum_attempts=2,
)
_HARNESS_TURN_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=1.0,
    maximum_attempts=2,
)
_HARNESS_HEARTBEAT_TIMEOUT = timedelta(minutes=2)


class ActivityRoutingOptions(TypedDict):
    """Temporal activity placement and execution deadline."""

    start_to_close_timeout: timedelta
    task_queue: NotRequired[str]


def _openrouter_config(llm: LLMBackendConfig) -> OpenRouterLLMConfig:
    if llm.harness != "none":
        raise ValueError(f"OpenRouter config requested for backend {llm.harness!r}")
    return OpenRouterLLMConfig(
        model=llm.model,
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
    )


def _harness_task_queue(llm: LLMBackendConfig) -> str:
    if llm.harness == "claude-code":
        return HARNESS_CLAUDE_TASK_QUEUE
    if llm.harness == "codex":
        return HARNESS_CODEX_TASK_QUEUE
    if llm.harness == "pi":
        return HARNESS_PI_TASK_QUEUE
    raise ValueError(f"harness task queue requested for backend {llm.harness!r}")


def _executes_model_spec_simulation(tool: LLMToolSpec) -> bool:
    return tool.executor == "model_spec_submit_construct"


def _openrouter_turn_executes_model_spec_simulation(
    call: OpenRouterCallResult,
    tools: list[LLMToolSpec],
) -> bool:
    tool_by_name = {tool.name: tool for tool in tools}
    return any(
        (tool := tool_by_name.get(item.name)) is not None and _executes_model_spec_simulation(tool)
        for item in call.tool_calls
    )


def _provider_call_timeout(llm: LLMBackendConfig) -> timedelta:
    return timedelta(seconds=(llm.timeout or 300) + 30)


def _harness_turn_timeout(llm: LLMBackendConfig) -> timedelta:
    if llm.harness == "claude-code":
        return timedelta(seconds=(llm.timeout or 900) + 30)
    if llm.harness == "codex":
        return timedelta(seconds=(llm.timeout or 1800) + 30)
    if llm.harness == "pi":
        return timedelta(seconds=(llm.timeout or 1800) + 30)
    raise ValueError(f"harness timeout requested for backend {llm.harness!r}")


async def _append_user_message(
    input: LLMSubroutineInput,
    conversation_ref: str,
    user_message_index: int,
) -> str:
    appended = await workflow.execute_activity(
        "append_llm_user_message_activity",
        AppendLLMUserMessageInput(
            workspace_id=input.workspace_id,
            run_id=input.run_id,
            subroutine_id=input.subroutine_id,
            context_kind=input.context_kind,
            context_ref=input.context_ref,
            conversation_ref=conversation_ref,
            user_message_index=user_message_index,
        ),
        result_type=AppendLLMUserMessageResult,
        start_to_close_timeout=_LOCAL_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        summary=f"Append LLM user message {input.subroutine_id} #{user_message_index + 1}",
    )
    return appended.conversation_ref


async def _execute_harness_turn(
    input: LLMSubroutineInput,
    start: LLMSubroutineStart,
    user_message_index: int,
    user_label: str,
    pending_tool_requests: list[HarnessToolRequest],
) -> HarnessTurnResult:
    info = workflow.info()
    harness_input = HarnessTurnInput(
        workflow_id=info.workflow_id,
        workflow_run_id=info.run_id,
        workspace_id=input.workspace_id,
        run_id=input.run_id,
        subroutine_id=input.subroutine_id,
        context_kind=input.context_kind,
        context_ref=input.context_ref,
        harness_state_ref=start.harness_state_ref,
        harness_tool_ref_base=f"{start.harness_tool_ref_base}/{user_label}",
        result_ref=f"{start.result_ref_base}/{user_label}.json",
        llm=input.llm,
        tools=start.tools,
        user_message_index=user_message_index,
        log_label=f"{input.subroutine_id}/{user_label}",
    )

    if not start.tools:
        return await workflow.execute_activity(
            "run_harness_turn_activity",
            harness_input,
            result_type=HarnessTurnResult,
            task_queue=_harness_task_queue(input.llm),
            start_to_close_timeout=_harness_turn_timeout(input.llm),
            heartbeat_timeout=_HARNESS_HEARTBEAT_TIMEOUT,
            retry_policy=_HARNESS_TURN_RETRY,
            summary=f"Harness {input.llm.harness} {input.subroutine_id} {user_label}",
        )

    harness_handle = workflow.start_activity(
        "run_harness_turn_activity",
        harness_input,
        result_type=HarnessTurnResult,
        task_queue=_harness_task_queue(input.llm),
        start_to_close_timeout=_harness_turn_timeout(input.llm),
        heartbeat_timeout=_HARNESS_HEARTBEAT_TIMEOUT,
        retry_policy=_HARNESS_TURN_RETRY,
        summary=f"Harness {input.llm.harness} {input.subroutine_id} {user_label}",
    )
    while not harness_handle.done():
        await workflow.wait_condition(lambda: harness_handle.done() or bool(pending_tool_requests))
        while pending_tool_requests:
            request = pending_tool_requests.pop(0)
            activity_options: ActivityRoutingOptions = (
                {
                    "task_queue": MODEL_SPEC_SIMULATION_TASK_QUEUE,
                    "start_to_close_timeout": _SIMULATION_TIMEOUT,
                }
                if _executes_model_spec_simulation(request.tool)
                else {"start_to_close_timeout": _LOCAL_TIMEOUT}
            )
            await workflow.execute_activity(
                "execute_harness_tool_request_activity",
                request,
                result_type=HarnessToolExecutionResult,
                retry_policy=_LOCAL_RETRY,
                summary=f"Execute harness tool {request.tool_name}",
                **activity_options,
            )

    return await harness_handle


def _activity_error_text(exc: ActivityError) -> str:
    cause = getattr(exc, "cause", None)
    if cause is not None:
        return str(cause)
    return str(exc)


async def _execute_openrouter_call(
    input: LLMSubroutineInput,
    start: LLMSubroutineStart,
    conversation_ref: str,
    turn_label: str,
) -> tuple[OpenRouterCallResult, int]:
    def _call(label: str, source_conversation_ref: str) -> OpenRouterCallInput:
        return OpenRouterCallInput(
            conversation_ref=source_conversation_ref,
            next_conversation_ref=f"{start.conversation_ref_base}/{label}-assistant.json",
            call_ref=f"{start.call_ref_base}/{label}.json",
            assistant_ref=f"{start.assistant_ref_base}/{label}.json",
            llm=_openrouter_config(input.llm),
            tools=start.tools,
            log_label=f"{input.subroutine_id}/{label}",
        )

    try:
        call = await workflow.execute_activity(
            "call_openrouter_activity",
            _call(turn_label, conversation_ref),
            result_type=OpenRouterCallResult,
            task_queue=OPENROUTER_TASK_QUEUE,
            start_to_close_timeout=_provider_call_timeout(input.llm),
            retry_policy=_OPENROUTER_CALL_RETRY,
            summary=f"OpenRouter {input.subroutine_id} {turn_label}",
        )
        return call, 1
    except ActivityError as exc:
        if not start.tools:
            raise
        repair_label = f"{turn_label}-repair-001"
        repaired = await workflow.execute_activity(
            "append_llm_repair_message_activity",
            AppendLLMRepairMessageInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                subroutine_id=input.subroutine_id,
                conversation_ref=conversation_ref,
                next_conversation_ref=f"{start.conversation_ref_base}/{repair_label}.json",
                error_text=_activity_error_text(exc),
                tools=start.tools,
            ),
            result_type=AppendLLMRepairMessageResult,
            start_to_close_timeout=_LOCAL_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            summary=f"Append LLM repair message {input.subroutine_id} {turn_label}",
        )
        call = await workflow.execute_activity(
            "call_openrouter_activity",
            _call(repair_label, repaired.conversation_ref),
            result_type=OpenRouterCallResult,
            task_queue=OPENROUTER_TASK_QUEUE,
            start_to_close_timeout=_provider_call_timeout(input.llm),
            retry_policy=_OPENROUTER_CALL_RETRY,
            summary=f"OpenRouter {input.subroutine_id} {repair_label}",
        )
        return call, 2


@workflow.defn
class LLMSubroutineWorkflow:
    def __init__(self) -> None:
        self._pending_harness_tool_requests: list[HarnessToolRequest] = []

    @workflow.signal
    async def harness_tool_requested(self, request: HarnessToolRequest) -> None:
        self._pending_harness_tool_requests.append(request)

    @workflow.run
    async def run(self, input: LLMSubroutineInput) -> LLMSubroutineResult:
        start = await workflow.execute_activity(
            "start_llm_subroutine_activity",
            LLMSubroutineStartInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                subroutine_id=input.subroutine_id,
                context_kind=input.context_kind,
                context_ref=input.context_ref,
            ),
            result_type=LLMSubroutineStart,
            start_to_close_timeout=_LOCAL_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            summary=f"Start LLM subroutine {input.subroutine_id}",
        )

        conversation_ref = start.conversation_ref
        last_result_ref: str | None = None
        n_llm_calls = 0
        n_harness_turns = 0
        harness_trace_refs: list[str] = []
        terminal_error: str | None = None

        for user_message_index in range(start.user_message_count):
            conversation_ref = await _append_user_message(
                input, conversation_ref, user_message_index
            )
            user_label = f"user-{user_message_index + 1:03d}"

            if input.llm.harness == "none":
                for turn in range(1, input.max_tool_turns + 1):
                    turn_label = f"{user_label}-turn-{turn:03d}"
                    call, call_count = await _execute_openrouter_call(
                        input,
                        start,
                        conversation_ref,
                        turn_label,
                    )
                    n_llm_calls += call_count
                    conversation_ref = call.conversation_ref

                    if not start.tools:
                        break
                    if not call.tool_calls:
                        break

                    activity_options: ActivityRoutingOptions = (
                        {
                            "task_queue": MODEL_SPEC_SIMULATION_TASK_QUEUE,
                            "start_to_close_timeout": _SIMULATION_TIMEOUT,
                        }
                        if _openrouter_turn_executes_model_spec_simulation(call, start.tools)
                        else {"start_to_close_timeout": _LOCAL_TIMEOUT}
                    )
                    tool_execution = await workflow.execute_activity(
                        "execute_llm_tool_calls_activity",
                        LLMToolExecutionInput(
                            workspace_id=input.workspace_id,
                            run_id=input.run_id,
                            subroutine_id=input.subroutine_id,
                            context_kind=input.context_kind,
                            context_ref=input.context_ref,
                            conversation_ref=conversation_ref,
                            assistant_ref=call.assistant_ref,
                            execution_ref=f"{start.tool_execution_ref_base}/{turn_label}.json",
                            result_ref=f"{start.result_ref_base}/{turn_label}.json",
                            tools=start.tools,
                        ),
                        result_type=LLMToolExecutionResult,
                        retry_policy=_LOCAL_RETRY,
                        summary=f"Execute LLM tools {input.subroutine_id} {turn_label}",
                        **activity_options,
                    )
                    conversation_ref = tool_execution.conversation_ref
                    if tool_execution.terminal_success:
                        if tool_execution.result_ref is not None:
                            last_result_ref = tool_execution.result_ref
                        elif input.require_result:
                            terminal_error = (
                                f"LLM subroutine {input.subroutine_id} terminal tool "
                                "succeeded without a result ref"
                            )
                        break
                else:
                    terminal_error = (
                        f"LLM subroutine {input.subroutine_id} exceeded "
                        f"{input.max_tool_turns} turns without validation success."
                    )

                if terminal_error is not None:
                    break
                continue

            harness = await _execute_harness_turn(
                input,
                start,
                user_message_index,
                user_label,
                self._pending_harness_tool_requests,
            )
            n_harness_turns += 1
            harness_trace_refs.append(harness.trace_ref)
            if harness.result_ref is not None:
                last_result_ref = harness.result_ref

        trace = await workflow.execute_activity(
            "finalize_llm_subroutine_trace_activity",
            LLMSubroutineTraceInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                subroutine_id=input.subroutine_id,
                context_kind=input.context_kind,
                conversation_ref=conversation_ref,
                call_ref_base=start.call_ref_base,
                harness_trace_refs=harness_trace_refs,
            ),
            result_type=LLMSubroutineTraceResult,
            start_to_close_timeout=_TRACE_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            summary=f"Finalize LLM trace {input.subroutine_id}",
        )

        if terminal_error is not None:
            raise RuntimeError(terminal_error)
        if input.require_result and last_result_ref is None:
            raise RuntimeError(f"LLM subroutine {input.subroutine_id} produced no valid result")

        return LLMSubroutineResult(
            result_ref=last_result_ref,
            conversation_ref=conversation_ref,
            trace_ref=trace.trace_ref,
            n_llm_calls=n_llm_calls,
            n_harness_turns=n_harness_turns,
        )
