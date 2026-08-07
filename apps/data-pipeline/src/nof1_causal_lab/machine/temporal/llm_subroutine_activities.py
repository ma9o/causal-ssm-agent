"""Shared activities for workflow-driven LLM subroutines."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError
from temporalio import activity

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.temporal.llm_context_adapters import subroutine_context_messages
from nof1_causal_lab.machine.temporal.llm_subroutine_storage import (
    read_subroutine_json,
    subroutine_conversation_path,
    subroutine_root,
    write_subroutine_json,
)
from nof1_causal_lab.machine.temporal.llm_tool_adapters import execute_subroutine_tool
from nof1_causal_lab.machine.temporal.messages import (
    AppendLLMRepairMessageInput,
    AppendLLMRepairMessageResult,
    AppendLLMUserMessageInput,
    AppendLLMUserMessageResult,
    HarnessToolExecutionResult,
    HarnessToolRequest,
    HarnessTurnInput,
    HarnessTurnResult,
    LLMSubroutineStart,
    LLMSubroutineStartInput,
    LLMSubroutineTraceInput,
    LLMSubroutineTraceResult,
    LLMToolExecutionInput,
    LLMToolExecutionResult,
    LLMToolSpec,
)
from nof1_causal_lab.utils import storage

_RECOVERABLE_TOOL_EXECUTION_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValidationError,
    ValueError,
)


def _subroutine_tool_message(
    tool_call: UncheckedJsonObject, content: str, error: str | None = None
) -> UncheckedJsonObject:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": str(tool_call.get("id", "")),
        "name": str((tool_call.get("function") or {}).get("name") or tool_call.get("name", "")),
        "error": error,
    }


def _terminal_tool_succeeded(tool: LLMToolSpec, output: str, error: str | None) -> bool:
    if tool.kind != "terminal" or error is not None:
        return False
    result_text = output.strip()
    if tool.success_output is None:
        recoverable_prefixes = (
            "JSON parse error:",
            "VALIDATION ERRORS:",
            "Tool execution failed:",
            "Unknown tool:",
            "Unsupported tool executor:",
            "Error:",
        )
        return not result_text.startswith(recoverable_prefixes)
    return result_text == tool.success_output


def _tool_execution_failed(exc: BaseException) -> str:
    return f"Tool execution failed: {exc}"


@activity.defn
async def start_llm_subroutine_activity(input: LLMSubroutineStartInput) -> LLMSubroutineStart:
    system_prompt, user_messages, tools = subroutine_context_messages(
        input.context_kind, input.context_ref
    )
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    conversation_ref = subroutine_conversation_path(
        input.workspace_id,
        input.run_id,
        input.subroutine_id,
        "turn-000-system.json",
    )
    root = subroutine_root(input.workspace_id, input.run_id, input.subroutine_id)
    write_subroutine_json(
        conversation_ref,
        {
            "messages": messages,
            "context_kind": input.context_kind,
            "context_ref": input.context_ref,
            "user_messages": user_messages,
        },
    )

    return LLMSubroutineStart(
        conversation_ref=conversation_ref,
        conversation_ref_base=storage.join(root, "conversation"),
        user_message_count=len(user_messages),
        tools=tools,
        call_ref_base=storage.join(root, "calls"),
        assistant_ref_base=storage.join(root, "assistants"),
        tool_execution_ref_base=storage.join(root, "tool-executions"),
        harness_state_ref=storage.join(root, "harness-state.json"),
        harness_tool_ref_base=storage.join(root, "harness-tools"),
        result_ref_base=storage.join(root, "results"),
    )


@activity.defn
async def append_llm_user_message_activity(
    input: AppendLLMUserMessageInput,
) -> AppendLLMUserMessageResult:
    system_prompt, user_messages, _tool = subroutine_context_messages(
        input.context_kind, input.context_ref
    )
    del system_prompt
    if input.user_message_index >= len(user_messages):
        raise IndexError(f"user message index {input.user_message_index} out of range")

    conversation = read_subroutine_json(input.conversation_ref)
    messages = [
        *conversation["messages"],
        {"role": "user", "content": user_messages[input.user_message_index]},
    ]
    conversation_ref = subroutine_conversation_path(
        input.workspace_id,
        input.run_id,
        input.subroutine_id,
        f"user-{input.user_message_index + 1:03d}.json",
    )
    write_subroutine_json(conversation_ref, {"messages": messages})
    return AppendLLMUserMessageResult(conversation_ref=conversation_ref)


@activity.defn
async def append_llm_repair_message_activity(
    input: AppendLLMRepairMessageInput,
) -> AppendLLMRepairMessageResult:
    if storage.exists(input.next_conversation_ref):
        return AppendLLMRepairMessageResult(conversation_ref=input.next_conversation_ref)

    from nof1_causal_lab.utils.llm import _tool_retry_message

    conversation = read_subroutine_json(input.conversation_ref)
    repair_message = _tool_retry_message(input.error_text, input.tools)
    write_subroutine_json(
        input.next_conversation_ref,
        {"messages": [*conversation["messages"], repair_message]},
    )
    return AppendLLMRepairMessageResult(conversation_ref=input.next_conversation_ref)


@activity.defn
async def execute_llm_tool_calls_activity(input: LLMToolExecutionInput) -> LLMToolExecutionResult:
    if storage.exists(input.execution_ref):
        return LLMToolExecutionResult.model_validate(
            read_subroutine_json(input.execution_ref)["result"]
        )

    assistant_output = read_subroutine_json(input.assistant_ref)
    assistant_message = assistant_output["message"]
    conversation = read_subroutine_json(input.conversation_ref)
    messages = list(conversation["messages"])
    tool_messages: list[UncheckedJsonObject] = []
    tool_calls_fired: list[str] = []
    terminal_success = False
    captured_result_ref: str | None = None
    tool_by_name = {tool.name: tool for tool in input.tools}

    for tool_index, tool_call in enumerate(assistant_message.get("tool_calls") or []):
        fn = tool_call.get("function") or {}
        tool_name = str(fn.get("name") or tool_call.get("name", ""))
        tool_calls_fired.append(tool_name)
        tool = tool_by_name.get(tool_name)
        if tool is None:
            result_text = f"Unknown tool: {tool_name}"
            tool_messages.append(
                _subroutine_tool_message(tool_call, result_text, error=result_text)
            )
            continue
        try:
            raw_args = str(fn.get("arguments") or tool_call.get("arguments", "{}") or "{}")
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must decode to a JSON object")
            result_text, tool_result_ref = await execute_subroutine_tool(
                input=input,
                tool=tool,
                args=args,
                result_ref=input.result_ref,
                request_id=str(tool_call.get("id") or f"{input.execution_ref}:{tool_index}"),
            )
        except json.JSONDecodeError as exc:
            result_text = f"JSON parse error: {exc}"
            error_text = None
        except _RECOVERABLE_TOOL_EXECUTION_ERRORS as exc:
            result_text = _tool_execution_failed(exc)
            error_text = str(exc)
        else:
            error_text = None
            if tool_result_ref is not None:
                captured_result_ref = tool_result_ref
        tool_messages.append(_subroutine_tool_message(tool_call, result_text, error=error_text))
        if _terminal_tool_succeeded(tool, result_text, error_text):
            terminal_success = True

    next_conversation_ref = subroutine_conversation_path(
        input.workspace_id,
        input.run_id,
        input.subroutine_id,
        f"tool-execution-{Path(input.execution_ref).stem}.json",
    )
    next_messages = [*messages, *tool_messages]
    write_subroutine_json(next_conversation_ref, {"messages": next_messages})

    feedback_text = "\n".join(str(message.get("content", "")) for message in tool_messages)
    if input.max_tool_output is not None and len(feedback_text) > input.max_tool_output:
        feedback_text = feedback_text[: input.max_tool_output] + "\n...[truncated]"
    result = LLMToolExecutionResult(
        conversation_ref=next_conversation_ref,
        terminal_success=terminal_success,
        result_ref=captured_result_ref,
        feedback_preview=feedback_text[:240],
        tool_calls_fired=tool_calls_fired,
    )
    write_subroutine_json(input.execution_ref, {"result": result.model_dump(mode="json")})
    return result


def _harness_tool_request_path(base: str, folder: str, request_id: str) -> str:
    return storage.join(base, folder, f"{request_id}.json")


def _build_harness_bridge_tools(input: HarnessTurnInput):
    from nof1_causal_lab.utils.openrouter_client import Tool

    if not input.tools:
        raise ValueError("harness tool requested for a no-tool LLM subroutine")

    def _build_one(tool: LLMToolSpec):
        async def _execute(**kwargs: str) -> str:
            request_id = uuid4().hex
            response_ref = _harness_tool_request_path(
                input.harness_tool_ref_base,
                "responses",
                request_id,
            )
            request_ref = _harness_tool_request_path(
                input.harness_tool_ref_base,
                "requests",
                request_id,
            )
            request = HarnessToolRequest(
                request_id=request_id,
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                subroutine_id=input.subroutine_id,
                context_kind=input.context_kind,
                context_ref=input.context_ref,
                result_ref=input.result_ref,
                tool=tool,
                tool_name=tool.name,
                arguments=dict(kwargs),
                request_ref=request_ref,
                response_ref=response_ref,
            )
            write_subroutine_json(request_ref, request.model_dump(mode="json"))
            from nof1_causal_lab.machine.temporal.client import connect_client

            client = await connect_client()
            handle = client.get_workflow_handle(
                input.workflow_id,
                run_id=input.workflow_run_id,
            )
            await handle.signal("harness_tool_requested", request)
            while not storage.exists(response_ref):
                await asyncio.sleep(0.2)
            response = HarnessToolExecutionResult.model_validate(read_subroutine_json(response_ref))
            return response.output

        return Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            execute=_execute,
            stop_on_success=tool.kind == "terminal",
            success_output=tool.success_output,
        )

    return [_build_one(tool) for tool in input.tools]


async def _await_harness_turn(turn: Any, subroutine_id: str) -> Any:
    """Await a harness turn while heartbeating from the activity task.

    Harness MCP callbacks execute in the server's task context, where Temporal's
    activity context is unavailable. The owning activity task carries the
    heartbeat instead while the harness and any bridged tool request run.
    """
    task = asyncio.ensure_future(turn)
    while not task.done():
        activity.heartbeat({"waiting_for_harness_turn": subroutine_id})
        await asyncio.sleep(1)
    return await task


@activity.defn
async def execute_harness_tool_request_activity(
    input: HarnessToolRequest,
) -> HarnessToolExecutionResult:
    if storage.exists(input.response_ref):
        return HarnessToolExecutionResult.model_validate(read_subroutine_json(input.response_ref))

    output = ""
    captured_result_ref: str | None = None
    success = False
    if input.tool_name != input.tool.name:
        output = f"Unknown tool: {input.tool_name}"
    elif input.tool.executor != "context_json_validation":
        try:
            output, captured_result_ref = await execute_subroutine_tool(
                input=input,
                tool=input.tool,
                args=input.arguments,
                result_ref=input.result_ref,
                request_id=input.request_id,
            )
        except json.JSONDecodeError as exc:
            output = f"JSON parse error: {exc}"
        except _RECOVERABLE_TOOL_EXECUTION_ERRORS as exc:
            output = _tool_execution_failed(exc)
        else:
            success = _terminal_tool_succeeded(input.tool, output, None)
    else:
        try:
            output, captured_result_ref = await execute_subroutine_tool(
                input=input,
                tool=input.tool,
                args=input.arguments,
                result_ref=input.result_ref,
                request_id=input.request_id,
            )
        except json.JSONDecodeError as exc:
            output = f"JSON parse error: {exc}"
        except _RECOVERABLE_TOOL_EXECUTION_ERRORS as exc:
            output = _tool_execution_failed(exc)
        else:
            success = _terminal_tool_succeeded(input.tool, output, None)

    result = HarnessToolExecutionResult(
        request_id=input.request_id,
        tool_name=input.tool_name,
        output=output,
        result_ref=captured_result_ref,
        success=success,
    )
    write_subroutine_json(input.response_ref, result.model_dump(mode="json"))
    return result


@activity.defn
async def run_harness_turn_activity(input: HarnessTurnInput) -> HarnessTurnResult:
    from nof1_causal_lab.utils.harness.claude import open_claude_harness_session
    from nof1_causal_lab.utils.harness.codex import open_codex_harness_session
    from nof1_causal_lab.utils.harness.pi import open_pi_harness_session

    _system_prompt, user_messages, _tools = subroutine_context_messages(
        input.context_kind, input.context_ref
    )
    if input.user_message_index >= len(user_messages):
        raise IndexError(f"user message index {input.user_message_index} out of range")

    state = cast(
        "UncheckedJsonObject",
        read_subroutine_json(input.harness_state_ref)
        if storage.exists(input.harness_state_ref)
        else {"raw_events": [], "turn_index": 0, "session_id": None},
    )
    raw_events = cast("list[UncheckedJsonObject]", state.get("raw_events") or [])
    turn_index = cast("int", state.get("turn_index") or 0)
    session_id = cast("str | None", state.get("session_id"))
    tools = [] if not input.tools else _build_harness_bridge_tools(input)
    user_message = user_messages[input.user_message_index]

    if input.llm.harness == "claude-code":
        async with open_claude_harness_session(
            tools=tools,
            system_prompt=_system_prompt if turn_index == 0 else None,
            model=input.llm.model,
            bin=input.llm.bin or "claude",
            effort=input.llm.effort,
            max_turns=input.llm.max_turns,
            max_budget_usd=input.llm.max_budget_usd,
            fallback_model=input.llm.fallback_model,
            timeout_seconds=float(input.llm.timeout or 900),
            log_label=input.log_label,
            session_id=session_id,
            initial_events=raw_events,
            turn_index=turn_index,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                input.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": session.session_id,
            }

    elif input.llm.harness == "codex":
        async with open_codex_harness_session(
            tools=tools,
            system_prompt=_system_prompt if turn_index == 0 else None,
            model=input.llm.model,
            bin=input.llm.bin or "codex",
            reasoning_effort=input.llm.reasoning_effort,
            service_tier=input.llm.service_tier,
            timeout_seconds=float(input.llm.timeout or 1800),
            log_label=input.log_label,
            initial_events=raw_events,
            turn_index=turn_index,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                input.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": None,
            }
    elif input.llm.harness == "pi":
        async with open_pi_harness_session(
            tools=tools,
            system_prompt=_system_prompt,
            provider=input.llm.provider or "openai-codex",
            model=input.llm.model,
            thinking=input.llm.thinking or "high",
            bin=input.llm.bin or "pi",
            timeout_seconds=float(input.llm.timeout or 1800),
            log_label=input.log_label,
            initial_events=raw_events,
            initial_session_jsonl=cast("str | None", state.get("session_jsonl")),
            session_id=session_id,
        ) as session:
            turn = await _await_harness_turn(
                session.turn(user_message),
                input.subroutine_id,
            )
            result = session.result
            next_state = {
                "raw_events": session.raw_events,
                "turn_index": turn_index + 1,
                "session_id": session.session_id,
                "session_jsonl": session.session_jsonl,
            }
    else:
        raise ValueError(f"harness turn activity received backend {input.llm.harness!r}")

    write_subroutine_json(input.harness_state_ref, next_state)
    trace_ref = storage.join(
        subroutine_root(input.workspace_id, input.run_id, input.subroutine_id),
        "traces",
        f"harness-turn-{input.user_message_index + 1:03d}.json",
    )
    write_subroutine_json(trace_ref, result.trace.model_dump(mode="json"))
    return HarnessTurnResult(
        harness_state_ref=input.harness_state_ref,
        trace_ref=trace_ref,
        completion_preview=turn.completion[:240],
        result_ref=input.result_ref if storage.exists(input.result_ref) else None,
        terminal_tool_name=turn.terminal_tool_name,
        tool_calls_fired=turn.tool_calls_fired,
    )


@activity.defn
async def finalize_llm_subroutine_trace_activity(
    input: LLMSubroutineTraceInput,
) -> LLMSubroutineTraceResult:
    from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage, TraceUsage, _merge_trace

    root = subroutine_root(input.workspace_id, input.run_id, input.subroutine_id)
    trace_path = storage.join(root, "trace.json")

    if input.harness_trace_refs:
        trace = LLMTrace()
        for harness_trace_ref in input.harness_trace_refs:
            trace = _merge_trace(
                trace, LLMTrace.model_validate(read_subroutine_json(harness_trace_ref))
            )
        storage.write_text(trace_path, trace.model_dump_json())
        return LLMSubroutineTraceResult(trace_ref=trace_path)

    conversation = read_subroutine_json(input.conversation_ref)
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    has_reasoning_tokens = False
    total_time = 0.0
    model = ""
    for entry in sorted(storage.listdir(input.call_ref_base)):
        if not entry.endswith(".json"):
            continue
        call = read_subroutine_json(entry)["result"]
        model = call.get("model") or model
        total_time += float(call.get("time") or 0.0)
        usage = call.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        if usage.get("reasoning_tokens") is not None:
            has_reasoning_tokens = True
            reasoning_tokens += int(usage["reasoning_tokens"])

    trace = LLMTrace(
        messages=[
            TraceMessage(
                role=message["role"],
                content=str(message.get("content", "")),
                reasoning=message.get("reasoning"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
                tool_name=message.get("name"),
                tool_result=str(message.get("content", "")) if message["role"] == "tool" else None,
                tool_is_error=message.get("error") is not None,
            )
            for message in conversation["messages"]
        ],
        model=model,
        total_time_seconds=total_time,
        usage=TraceUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens if has_reasoning_tokens else None,
        ),
    )
    storage.write_text(trace_path, trace.model_dump_json())
    return LLMSubroutineTraceResult(trace_ref=trace_path)


LLM_SUBROUTINE_ACTIVITIES = [
    start_llm_subroutine_activity,
    append_llm_user_message_activity,
    append_llm_repair_message_activity,
    execute_llm_tool_calls_activity,
    execute_harness_tool_request_activity,
    run_harness_turn_activity,
    finalize_llm_subroutine_trace_activity,
]
