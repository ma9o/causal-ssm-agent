"""Trace contracts and small shared LLM boundary helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.utils.openrouter_client import Tool

MAX_TOOL_REPAIR_ERROR_CHARS = 1200


class TraceMessage(BaseModel):
    """A single message in an LLM trace."""

    role: str
    content: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result: str | None = None
    tool_is_error: bool = False


class TraceUsage(BaseModel):
    """Token usage for an LLM trace."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None


class LLMTrace(BaseModel):
    """Full trace of an LLM conversation."""

    messages: list[TraceMessage] = Field(default_factory=list)
    model: str = ""
    total_time_seconds: float = 0.0
    usage: TraceUsage = Field(default_factory=TraceUsage)


def _merge_trace(existing: LLMTrace, new_trace: LLMTrace) -> LLMTrace:
    """Append a trace segment onto an existing stage-local trace."""
    return LLMTrace(
        messages=[*existing.messages, *new_trace.messages],
        model=new_trace.model or existing.model,
        total_time_seconds=existing.total_time_seconds + new_trace.total_time_seconds,
        usage=TraceUsage(
            input_tokens=existing.usage.input_tokens + new_trace.usage.input_tokens,
            output_tokens=existing.usage.output_tokens + new_trace.usage.output_tokens,
            reasoning_tokens=(
                (existing.usage.reasoning_tokens or 0) + (new_trace.usage.reasoning_tokens or 0)
            )
            or None,
        ),
    )


def _validate_json_and_format(
    json_str: str,
    validate_fn: Callable[[dict], tuple[Any, list[str]]],
    capture: dict | None = None,
    capture_key: str | None = None,
    capture_result: bool = False,
) -> str:
    """Parse JSON, run a validator, and format its feedback."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as error:
        return f"JSON parse error: {error}"

    result, errors = validate_fn(data)
    if not errors:
        if capture is not None and capture_key:
            capture[capture_key] = result if capture_result else data
        return "VALID"
    return "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)


def _truncate_tool_error(error_text: str, limit: int = MAX_TOOL_REPAIR_ERROR_CHARS) -> str:
    """Trim provider error payloads before echoing them back to the model."""
    if len(error_text) <= limit:
        return error_text
    return error_text[:limit] + "\n...[truncated]"


def _tool_retry_message(error_text: str, tools: list[Tool] | None) -> dict[str, str]:
    """Build a repair instruction after a malformed or failed tool response."""
    guidance = (
        "Retry the same step. If you need a tool, emit a valid tool call with a JSON object "
        f"for arguments and use only these tools: {', '.join(tool.name for tool in tools)}."
        if tools
        else "Retry the same step in plain text only. No tools are available on this turn, "
        "so do not emit any tool calls."
    )
    return {
        "role": "user",
        "content": (
            "Your previous response could not be processed.\n\n"
            "Error:\n"
            f"{_truncate_tool_error(error_text)}\n\n"
            f"{guidance}"
        ),
    }
