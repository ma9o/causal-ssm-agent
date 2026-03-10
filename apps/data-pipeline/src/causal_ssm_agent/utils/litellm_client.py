"""Minimal LiteLLM runtime helpers.

Runtime orchestration uses plain OpenAI-style message dicts and a small local
tool abstraction. LiteLLM handles transport; this module only fills the gaps
LiteLLM does not cover for us directly.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

from litellm import acompletion
from pydantic import Field, create_model


@dataclass(frozen=True)
class GenerateConfig:
    """Generation settings shared across model calls."""

    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    reasoning_history: str | None = None
    max_tool_output: int = 16_000


@dataclass
class Tool:
    """Callable tool with model-facing JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Any

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        return cast("str", await self.execute(*args, **kwargs))


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _parse_arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Extract argument descriptions from a Google-style docstring."""

    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    in_args = False
    current_name: str | None = None

    for raw_line in inspect.cleandoc(docstring).splitlines():
        stripped = raw_line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if not stripped:
            current_name = None
            continue
        if not raw_line.startswith(" "):
            break

        name, _, description = stripped.partition(":")
        if _ and name.replace("_", "").isalnum():
            descriptions[name.strip()] = description.strip()
            current_name = name.strip()
            continue
        if current_name is not None:
            descriptions[current_name] = f"{descriptions[current_name]} {stripped}".strip()

    return descriptions


def _parameter_schema(handler: Any) -> dict[str, Any]:
    """Build a JSON schema from a tool handler signature."""

    signature = inspect.signature(handler)
    descriptions = _parse_arg_descriptions(inspect.getdoc(handler))
    fields: dict[str, tuple[Any, Any]] = {}

    for name, param in signature.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Signature.empty else Any
        default = param.default if param.default is not inspect.Signature.empty else ...
        if default is ...:
            field = Field(..., description=descriptions.get(name))
        else:
            field = Field(default, description=descriptions.get(name))
        fields[name] = (annotation, field)

    if not fields:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    model = create_model(f"{handler.__name__.title()}ToolParams", **fields)
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def tool(factory: Any) -> Any:
    """Decorator that converts a tool factory into a Tool object factory."""

    def wrapper(*args: Any, **kwargs: Any) -> Tool:
        handler = factory(*args, **kwargs)
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"Tool factory {factory.__name__} must return an async function")
        return Tool(
            name=factory.__name__,
            description=inspect.getdoc(factory) or "",
            parameters=_parameter_schema(handler),
            execute=handler,
        )

    wrapper.__name__ = factory.__name__
    wrapper.__doc__ = factory.__doc__
    return wrapper


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize a message to the OpenAI chat/tool shape used at runtime."""

    normalized = {
        "role": message["role"],
        "content": message.get("content", ""),
    }
    for key in ("tool_calls", "tool_call_id", "name"):
        value = message.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


def _tool_schema(tool_obj: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters,
        },
    }


def _message_content_parts(content: Any) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return "", None

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content:
        part_type = _get_attr(part, "type")
        text = _get_attr(part, "text")
        reasoning = _get_attr(part, "reasoning")
        if part_type in {"text", "output_text"} and text:
            text_parts.append(str(text))
        elif part_type in {"reasoning", "thinking"} and reasoning:
            reasoning_parts.append(str(reasoning))
        elif text:
            text_parts.append(str(text))

    joined_reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return "\n".join(text_parts), joined_reasoning


def _assistant_message(message: Any) -> dict[str, Any]:
    content_text, content_reasoning = _message_content_parts(_get_attr(message, "content"))
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": content_text,
    }

    tool_calls_raw = _get_attr(message, "tool_calls") or []
    tool_calls = []
    for tool_call in tool_calls_raw:
        function = _get_attr(tool_call, "function")
        arguments = _get_attr(function, "arguments", "")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": str(_get_attr(tool_call, "id", "")),
                "name": str(_get_attr(function, "name", "")),
                "arguments": str(arguments or "{}"),
            }
        )
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    reasoning = _get_attr(message, "reasoning")
    if reasoning is None:
        reasoning = _get_attr(message, "reasoning_content")
    if reasoning is None:
        reasoning = content_reasoning
    if isinstance(reasoning, str) and reasoning:
        assistant_message["reasoning"] = reasoning

    reasoning_details = _get_attr(message, "reasoning_details")
    if reasoning_details is not None:
        assistant_message["reasoning_details"] = reasoning_details

    return assistant_message


def _usage_from_response(response: Any) -> dict[str, int | None] | None:
    usage = _get_attr(response, "usage")
    if usage is None:
        return None

    details = _get_attr(usage, "completion_tokens_details")
    reasoning_tokens = _get_attr(details, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _get_attr(usage, "reasoning_tokens")

    return {
        "input_tokens": int(_get_attr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(_get_attr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(reasoning_tokens) if reasoning_tokens is not None else None,
    }


async def call_model(
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """Call LiteLLM and normalize the first choice into a plain dict."""

    request = config or GenerateConfig()
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [normalize_message(message) for message in messages],
        "drop_params": True,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.timeout is not None:
        kwargs["timeout"] = request.timeout
    if request.reasoning_effort is not None:
        kwargs["reasoning_effort"] = request.reasoning_effort
    if tools:
        kwargs["tools"] = [_tool_schema(tool_obj) for tool_obj in tools]

    started_at = perf_counter()
    response = await acompletion(**kwargs)
    elapsed = perf_counter() - started_at

    choices = _get_attr(response, "choices") or []
    if not choices:
        raise ValueError("LiteLLM returned no choices")

    choice = choices[0]
    message = _assistant_message(_get_attr(choice, "message"))

    return {
        "message": message,
        "completion": message.get("content", ""),
        "usage": _usage_from_response(response),
        "model": str(_get_attr(response, "model", model_name)),
        "time": elapsed,
        "stop_reason": _get_attr(choice, "finish_reason"),
    }


async def execute_tools(
    assistant_message: dict[str, Any],
    tools: list[Tool],
    max_tool_output: int | None = None,
) -> list[dict[str, Any]]:
    """Execute tool calls from a normalized assistant message."""

    tool_calls = assistant_message.get("tool_calls") or []
    if not tool_calls:
        return []

    tool_map = {tool_obj.name: tool_obj for tool_obj in tools}
    tool_messages: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name", ""))
        result_text: str
        error_text: str | None = None
        tool_obj = tool_map.get(tool_name)

        if tool_obj is None:
            result_text = f"Unknown tool: {tool_name}"
            error_text = result_text
        else:
            try:
                args = json.loads(str(tool_call.get("arguments", "{}")) or "{}")
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must decode to a JSON object")
                result = await tool_obj(**args)
                result_text = str(result)
            except Exception as exc:
                result_text = f"Tool execution failed: {exc}"
                error_text = str(exc)

        if max_tool_output is not None and len(result_text) > max_tool_output:
            result_text = result_text[:max_tool_output] + "\n...[truncated]"

        tool_messages.append(
            {
                "role": "tool",
                "content": result_text,
                "tool_call_id": str(tool_call.get("id", "")),
                "name": tool_name,
                "error": error_text,
            }
        )

    return tool_messages
