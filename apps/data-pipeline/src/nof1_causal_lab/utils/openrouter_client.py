"""Minimal OpenRouter runtime helpers.

Runtime orchestration uses plain OpenAI-style message dicts and a small local
tool abstraction. This module owns the transport to OpenRouter directly.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import Field, create_model

from nof1_causal_lab.utils.config import get_secret

logger = logging.getLogger(__name__)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_PREFIX = "openrouter/"


# ---------------------------------------------------------------------------
# RPM (requests-per-minute) rate limiter
# ---------------------------------------------------------------------------


class RpmLimiter:
    """Thread-safe async sliding-window rate limiter.

    Tracks API calls over a rolling window and blocks when the configured
    maximum would be exceeded.  Uses ``threading.Lock`` for cross-thread
    safety (Prefect ThreadPoolTaskRunner) and ``asyncio.sleep`` to yield
    control while waiting.

    Args:
        max_requests: Maximum number of requests allowed within the window.
        window_seconds: Length of the sliding window (default 60 for RPM).
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._timestamps and self._timestamps[0] <= now - self._window:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        """Wait until a request slot is available within the window."""
        while True:
            with self._lock:
                now = monotonic()
                self._purge(now)
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return
                # Calculate how long until the oldest entry expires
                wait_for = self._timestamps[0] + self._window - now
            await asyncio.sleep(min(wait_for + 0.05, 1.0))


_limiters: dict[str, RpmLimiter] = {}
_openrouter_client: AsyncOpenAI | None = None
_openrouter_client_lock = threading.Lock()


async def acquire_limiter(name: str) -> None:
    """Acquire a slot from the named limiter (no-op if not registered)."""
    limiter = _limiters.get(name)
    if limiter is not None:
        await limiter.acquire()


def _get_openrouter_client() -> AsyncOpenAI:
    """The process-wide client, keyed by the ambient ``OPENROUTER_API_KEY``."""
    global _openrouter_client
    with _openrouter_client_lock:
        if _openrouter_client is None:
            _openrouter_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                # The SDK requires a string up front; missing credentials still
                # surface as a normal authentication error on the first request.
                api_key=get_secret("OPENROUTER_API_KEY") or "missing",
            )
    return _openrouter_client


def normalize_openrouter_model_name(model_name: str) -> str:
    """Translate repo-local model IDs to the upstream OpenRouter format."""

    normalized = model_name.strip()
    if normalized.startswith(OPENROUTER_MODEL_PREFIX):
        return normalized[len(OPENROUTER_MODEL_PREFIX) :]
    return normalized


@dataclass(frozen=True)
class GenerateConfig:
    """Generation settings shared across model calls."""

    max_tokens: int | None = None
    timeout: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_tool_output: int | None = None


@dataclass
class Tool:
    """Callable tool with model-facing JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Any
    stop_on_success: bool = False
    success_output: str | None = None

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

    model = create_model(
        f"{handler.__name__.title()}ToolParams",
        **cast("dict[str, Any]", fields),
    )
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
    for key in ("tool_calls", "tool_call_id", "name", "reasoning", "reasoning_details"):
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
                "type": "function",
                "function": {
                    "name": str(_get_attr(function, "name", "")),
                    "arguments": str(arguments or "{}"),
                },
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


def _log_response_details(
    *,
    log_label: str | None,
    message: dict[str, Any],
    completion_text: str,
) -> None:
    """Log raw assistant outputs (completion, tool calls, reasoning)."""
    prefix = f"[{log_label}] " if log_label else ""

    logger.info(
        "%scall_model completion:\n%s",
        prefix,
        completion_text or "<empty>",
    )

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        logger.info(
            "%scall_model tool_calls:\n%s",
            prefix,
            json.dumps(tool_calls, indent=2, sort_keys=True),
        )

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        logger.info(
            "%scall_model reasoning:\n%s",
            prefix,
            reasoning,
        )
    reasoning_details = message.get("reasoning_details")
    if reasoning_details is not None:
        logger.info(
            "%scall_model reasoning_details:\n%s",
            prefix,
            json.dumps(reasoning_details, indent=2, sort_keys=True, default=str),
        )


async def call_model(
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    log_label: str | None = None,
) -> dict[str, Any]:
    """Call OpenRouter and normalize the first choice into a plain dict."""

    request = config or GenerateConfig()
    normalized_model_name = normalize_openrouter_model_name(model_name)

    await acquire_limiter("llm")

    kwargs: dict[str, Any] = {
        "model": normalized_model_name,
        "messages": [normalize_message(message) for message in messages],
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.timeout is not None:
        kwargs["timeout"] = request.timeout
    extra_body: dict[str, Any] = {
        "provider": {
            "sort": "throughput",
        }
    }
    if request.reasoning_effort is not None:
        extra_body["reasoning"] = {
            "effort": request.reasoning_effort,
        }
    kwargs["extra_body"] = extra_body
    if tools:
        kwargs["tools"] = [_tool_schema(tool_obj) for tool_obj in tools]

    if log_label:
        logger.info(
            "[%s] call_model request: model=%s messages=%d tools=%d timeout=%s max_tokens=%s",
            log_label,
            normalized_model_name,
            len(messages),
            len(tools or []),
            request.timeout,
            request.max_tokens,
        )

    started_at = perf_counter()
    request_coro = _get_openrouter_client().chat.completions.create(**kwargs)
    try:
        if request.timeout is not None:
            response = await asyncio.wait_for(request_coro, timeout=request.timeout)
        else:
            response = await request_coro
    except TimeoutError as exc:
        elapsed = perf_counter() - started_at
        if log_label:
            logger.warning(
                "[%s] call_model timeout: model=%s time=%.1fs timeout=%ss",
                log_label,
                normalized_model_name,
                elapsed,
                request.timeout,
            )
        raise TimeoutError(f"call_model timed out after {request.timeout}s") from exc
    elapsed = perf_counter() - started_at

    choices = _get_attr(response, "choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")

    choice = choices[0]
    message = _assistant_message(_get_attr(choice, "message"))
    completion_text = str(message.get("content", ""))
    tool_call_count = len(message.get("tool_calls") or [])
    stop_reason = _get_attr(choice, "finish_reason")

    if log_label:
        logger.info(
            "[%s] call_model response: model=%s stop=%s time=%.1fs tool_calls=%d completion_chars=%d",
            log_label,
            str(_get_attr(response, "model", normalized_model_name)),
            stop_reason or "end_turn",
            elapsed,
            tool_call_count,
            len(completion_text),
        )
    _log_response_details(
        log_label=log_label,
        message=message,
        completion_text=completion_text,
    )

    return {
        "message": message,
        "completion": completion_text,
        "usage": _usage_from_response(response),
        "model": str(_get_attr(response, "model", normalized_model_name)),
        "time": elapsed,
        "stop_reason": stop_reason,
    }
