"""Parse Claude Code and Codex CLI event streams into our :class:`LLMTrace`.

Both CLIs emit newline-delimited JSON when run with their streaming
output flags:

* ``claude -p --output-format stream-json --verbose`` emits events with
  a top-level ``type`` field (``system``, ``user``, ``assistant``,
  ``result``, etc.). Assistant messages carry Anthropic-style content
  blocks (``text``, ``tool_use``); tool results come back as user
  messages containing ``tool_result`` blocks.
* ``codex exec --json`` emits events keyed by ``type`` with
  ``thread.started``, ``agent_message``, ``tool_call``, etc.

This module converts each stream into a :class:`LLMTrace` with the same
shape the embedded path produces, so downstream consumers (artifact
writers, the web viewer) don't need to care which backend ran.

Neither parser is complete today — the coverage is "enough to build a
useful trace for the primary cases". Unknown event types are recorded
but do not fail the parse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from causal_ssm_agent.utils.llm import LLMTrace, TraceMessage, TraceUsage


@dataclass
class ClaudeStreamState:
    """Accumulator for :func:`parse_claude_stream`."""

    session_id: str | None = None
    model: str = ""
    messages: list[TraceMessage] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)
    total_time_seconds: float = 0.0
    stop_reason: str | None = None
    final_text: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)


def _coerce_content_text(content: Any) -> str:
    """Flatten Anthropic ``content`` blocks into a plain text string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _claude_assistant_message(message: dict[str, Any]) -> TraceMessage:
    """Build a TraceMessage from a Claude assistant stream-json message."""
    content = message.get("content", [])
    text = _coerce_content_text(content)
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
    return TraceMessage(
        role="assistant",
        content=text,
        tool_calls=tool_calls or None,
    )


def _claude_tool_result_messages(message: dict[str, Any]) -> list[TraceMessage]:
    """Extract tool-result blocks from a Claude user stream-json message."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    out: list[TraceMessage] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        result_content = block.get("content")
        if isinstance(result_content, list):
            text = _coerce_content_text(result_content)
        elif isinstance(result_content, str):
            text = result_content
        else:
            text = json.dumps(result_content) if result_content is not None else ""
        out.append(
            TraceMessage(
                role="tool",
                content=text,
                tool_call_id=str(block.get("tool_use_id", "")),
                tool_result=text,
                tool_is_error=bool(block.get("is_error", False)),
            )
        )
    return out


def _claude_user_prompt_message(message: dict[str, Any]) -> TraceMessage | None:
    """Handle a pure user prompt (string content, not tool_result blocks)."""
    content = message.get("content")
    if isinstance(content, str):
        return TraceMessage(role="user", content=content)
    if isinstance(content, list):
        # If no tool_result blocks present, treat remaining text as a user prompt.
        has_tool_result = any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
        if not has_tool_result:
            return TraceMessage(role="user", content=_coerce_content_text(content))
    return None


def _extract_usage(usage_raw: Any) -> TraceUsage:
    if not isinstance(usage_raw, dict):
        return TraceUsage()
    input_tokens = int(usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0)
    output_tokens = int(usage_raw.get("output_tokens") or usage_raw.get("completion_tokens") or 0)
    reasoning_tokens_raw = usage_raw.get("reasoning_tokens") or (
        usage_raw.get("completion_tokens_details") or {}
    ).get("reasoning_tokens")
    reasoning_tokens = int(reasoning_tokens_raw) if reasoning_tokens_raw is not None else None
    return TraceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def apply_claude_event(state: ClaudeStreamState, event: dict[str, Any]) -> None:
    """Fold one Claude stream-json event into the accumulator."""
    state.raw_events.append(event)
    etype = event.get("type")

    if etype == "system" and event.get("subtype") == "init":
        if isinstance(event.get("session_id"), str):
            state.session_id = event["session_id"]
        if isinstance(event.get("model"), str):
            state.model = event["model"]
        return

    if etype == "user":
        message = event.get("message") or {}
        prompt = _claude_user_prompt_message(message)
        if prompt is not None:
            state.messages.append(prompt)
        state.messages.extend(_claude_tool_result_messages(message))
        return

    if etype == "assistant":
        message = event.get("message") or {}
        state.messages.append(_claude_assistant_message(message))
        usage = _extract_usage(message.get("usage"))
        state.usage = TraceUsage(
            input_tokens=state.usage.input_tokens + usage.input_tokens,
            output_tokens=state.usage.output_tokens + usage.output_tokens,
            reasoning_tokens=((state.usage.reasoning_tokens or 0) + (usage.reasoning_tokens or 0))
            or None,
        )
        stop_reason = message.get("stop_reason")
        if isinstance(stop_reason, str):
            state.stop_reason = stop_reason
        return

    if etype == "result":
        result_text = event.get("result")
        if isinstance(result_text, str):
            state.final_text = result_text
        duration_ms = event.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            state.total_time_seconds = float(duration_ms) / 1000.0
        usage = _extract_usage(event.get("usage"))
        if usage.input_tokens or usage.output_tokens:
            state.usage = usage
        subtype = event.get("subtype")
        if isinstance(subtype, str):
            state.stop_reason = state.stop_reason or subtype
        return


def finalize_trace(state: ClaudeStreamState) -> LLMTrace:
    """Materialize an :class:`LLMTrace` from an accumulator."""
    return LLMTrace(
        messages=list(state.messages),
        model=state.model,
        total_time_seconds=state.total_time_seconds,
        usage=state.usage,
    )


def parse_claude_stream(lines: list[str]) -> ClaudeStreamState:
    """Parse a list of Claude stream-json lines into a populated state.

    Non-JSON lines are silently skipped (harness CLIs occasionally print
    stderr-style notices between events).
    """
    state = ClaudeStreamState()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        apply_claude_event(state, event)
    return state


# ---------------------------------------------------------------------------
# Codex parser
# ---------------------------------------------------------------------------


@dataclass
class CodexStreamState:
    """Accumulator for :func:`parse_codex_stream`."""

    thread_id: str | None = None
    model: str = ""
    messages: list[TraceMessage] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)
    total_time_seconds: float = 0.0
    stop_reason: str | None = None
    final_text: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    _open_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)


def apply_codex_event(state: CodexStreamState, event: dict[str, Any]) -> None:
    """Fold one Codex ``--json`` event into the accumulator.

    Codex's event schema is less stable than Claude's; this handler
    extracts the commonly-seen shapes (thread start, agent messages,
    tool calls) and records everything in ``raw_events`` so downstream
    code can reach back to anything we don't yet model.
    """
    state.raw_events.append(event)
    etype = event.get("type")

    if etype == "thread.started":
        tid = event.get("thread_id")
        if isinstance(tid, str):
            state.thread_id = tid
        return

    # Codex 0.121 emits `item.completed` with a nested ``item`` carrying
    # the actual type (``agent_message``, ``reasoning``, ``mcp_tool_call``,
    # etc.). Earlier/prototype schemas used top-level ``agent_message``
    # events; both shapes are handled.
    if etype == "item.completed":
        item = event.get("item") or {}
        return apply_codex_event(state, {**item, "_from_item_completed": True})

    if etype in {"agent_message", "message"}:
        # The item may or may not carry an explicit role; codex's
        # agent_message items are always assistant output.
        message = event.get("message") or event
        role = message.get("role") or "assistant"
        text = message.get("text")
        if not isinstance(text, str):
            content = message.get("content") or ""
            text = _coerce_content_text(content) if isinstance(content, list) else str(content)
        if role == "assistant":
            state.messages.append(TraceMessage(role="assistant", content=text))
            if text:
                state.final_text = text
        elif role == "user":
            state.messages.append(TraceMessage(role="user", content=text))
        return

    if etype in {"tool_call", "mcp_tool_call"}:
        call_id = str(event.get("call_id") or event.get("id") or "")
        tool_name = str(event.get("name") or event.get("tool") or "")
        arguments = event.get("arguments") or event.get("input") or {}
        if isinstance(arguments, dict):
            arguments_json = json.dumps(arguments)
        else:
            arguments_json = str(arguments)
        tool_call_entry = {
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": arguments_json},
        }
        state.messages.append(
            TraceMessage(
                role="assistant",
                content="",
                tool_calls=[tool_call_entry],
            )
        )
        state._open_tool_calls[call_id] = tool_call_entry
        return

    if etype in {"tool_result", "mcp_tool_result"}:
        call_id = str(event.get("call_id") or event.get("id") or "")
        result = event.get("output") or event.get("result") or event.get("text") or ""
        if not isinstance(result, str):
            result = json.dumps(result)
        state.messages.append(
            TraceMessage(
                role="tool",
                content=result,
                tool_call_id=call_id,
                tool_result=result,
                tool_is_error=bool(event.get("is_error", False)),
            )
        )
        state._open_tool_calls.pop(call_id, None)
        return

    if etype in {"thread.completed", "turn.completed", "done", "result"}:
        usage = _extract_usage(event.get("usage"))
        if usage.input_tokens or usage.output_tokens:
            state.usage = usage
        duration_ms = event.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            state.total_time_seconds = float(duration_ms) / 1000.0
        reason = event.get("stop_reason") or event.get("subtype")
        if isinstance(reason, str):
            state.stop_reason = reason
        return

    if etype in {"turn.failed", "error"}:
        err = event.get("error") or event.get("message") or ""
        if isinstance(err, dict):
            err = json.dumps(err)
        state.stop_reason = "error"
        state.final_text = state.final_text or str(err)
        return


def finalize_codex_trace(state: CodexStreamState) -> LLMTrace:
    """Materialize an :class:`LLMTrace` from a Codex accumulator."""
    return LLMTrace(
        messages=list(state.messages),
        model=state.model,
        total_time_seconds=state.total_time_seconds,
        usage=state.usage,
    )


def parse_codex_stream(lines: list[str]) -> CodexStreamState:
    """Parse a list of ``codex exec --json`` lines into an accumulator."""
    state = CodexStreamState()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        apply_codex_event(state, event)
    return state
