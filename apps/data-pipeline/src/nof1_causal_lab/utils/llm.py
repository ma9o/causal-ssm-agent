"""Shared LLM utilities for multi-turn generation."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from nof1_causal_lab.utils.openrouter_client import (
    GenerateConfig,
    Tool,
    call_model,
    execute_tools,
    normalize_message,
)

logger = logging.getLogger(__name__)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

DEFAULT_MAX_TOOL_LOOP_TURNS = 40
WARN_TOOL_LOOP_TURNS = 10
MAX_TOOL_REPAIR_RETRIES = 1
MAX_CALL_TIMEOUT_RETRIES = 1
MAX_TOOL_REPAIR_ERROR_CHARS = 1200


# ---------------------------------------------------------------------------
# Trace models
# ---------------------------------------------------------------------------


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
    """Full trace of an LLM multi-turn conversation."""

    messages: list[TraceMessage] = Field(default_factory=list)
    model: str = ""
    total_time_seconds: float = 0.0
    usage: TraceUsage = Field(default_factory=TraceUsage)


def _chat_message_to_trace(msg: dict[str, Any]) -> TraceMessage:
    """Convert a runtime chat message to a TraceMessage."""

    return TraceMessage(
        role=msg["role"],
        content=str(msg.get("content", "")),
        reasoning=msg.get("reasoning"),
        tool_calls=msg.get("tool_calls"),
        tool_call_id=msg.get("tool_call_id"),
        tool_name=msg.get("name"),
        tool_result=str(msg.get("content", "")) if msg["role"] == "tool" else None,
        tool_is_error=msg.get("error") is not None,
    )


def _build_trace(all_messages: list[dict[str, Any]], output: dict[str, Any]) -> LLMTrace:
    """Build an LLMTrace from a final message list and response summary."""
    messages = [_chat_message_to_trace(m) for m in all_messages]
    usage = TraceUsage()
    if output.get("usage"):
        output_usage = output["usage"]
        usage = TraceUsage(
            input_tokens=output_usage["input_tokens"],
            output_tokens=output_usage["output_tokens"],
            reasoning_tokens=output_usage["reasoning_tokens"],
        )
    return LLMTrace(
        messages=messages,
        model=output.get("model", ""),
        total_time_seconds=output.get("time") or 0.0,
        usage=usage,
    )


def _merge_trace(existing: LLMTrace, new_trace: LLMTrace) -> LLMTrace:
    """Append a new trace segment onto an existing stage-local trace."""
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


def _record_trace_segment(
    trace_capture: dict | None,
    trace_messages: list[dict[str, Any]],
    output: dict[str, Any],
) -> None:
    """Record one trace segment into stage-local trace capture."""
    if trace_capture is None:
        return
    new_trace = _build_trace(trace_messages, output)
    existing = trace_capture.get("trace")
    if isinstance(existing, LLMTrace):
        trace_capture["trace"] = _merge_trace(existing, new_trace)
    else:
        trace_capture["trace"] = new_trace


# ---------------------------------------------------------------------------
# Type aliases for generate functions (unified)
# ---------------------------------------------------------------------------

MessageRewriter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
ToolRewriter = Callable[[list[Tool]], list[Tool]]


def _combine_log_label(*parts: str | None) -> str | None:
    """Join non-empty label fragments into a stable log scope."""
    labels = [part for part in parts if part]
    if not labels:
        return None
    return " / ".join(labels)


def scoped_log(label: str | None, msg: str) -> str:
    """Prefix a log format string with ``[label]`` when a label is provided."""
    return f"[{label}] {msg}" if label else msg


def get_generate_config() -> GenerateConfig:
    """Get standard GenerateConfig for all model calls.

    Reads settings from config.yaml llm section.
    """
    from nof1_causal_lab.utils.config import get_config

    embedded = get_config().llm.embedded
    reasoning_effort: ReasoningEffort | None
    if embedded.reasoning_effort == "none":
        reasoning_effort = "none"
    elif embedded.reasoning_effort == "minimal":
        reasoning_effort = "minimal"
    elif embedded.reasoning_effort == "low":
        reasoning_effort = "low"
    elif embedded.reasoning_effort == "medium":
        reasoning_effort = "medium"
    elif embedded.reasoning_effort == "high":
        reasoning_effort = "high"
    elif embedded.reasoning_effort == "xhigh":
        reasoning_effort = "xhigh"
    else:
        reasoning_effort = None
    return GenerateConfig(
        max_tokens=embedded.max_tokens,
        timeout=embedded.timeout,
        reasoning_effort=reasoning_effort,
    )


def dict_messages_to_chat(messages: list[dict]) -> list[dict[str, Any]]:
    """Normalize dict messages for the OpenRouter/OpenAI chat format.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        Normalized runtime chat messages.
    """
    chat_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in {"system", "user", "assistant", "tool"}:
            chat_messages.append(normalize_message(msg))
    return chat_messages


def _rewrite_context_messages(
    messages: list[dict[str, Any]],
    rewrite_messages: MessageRewriter | None,
) -> list[dict[str, Any]]:
    """Apply an optional context-only message rewrite."""
    if rewrite_messages is None:
        return messages
    return rewrite_messages(list(messages))


def _rewrite_available_tools(
    tools: list[Tool],
    rewrite_tools: ToolRewriter | None,
) -> list[Tool]:
    """Apply an optional tool-list rewrite for the current turn."""
    if rewrite_tools is None:
        return tools
    return rewrite_tools(list(tools))


# ---------------------------------------------------------------------------
# Generate function factory (unified for orchestrator and worker)
# ---------------------------------------------------------------------------


def make_generate_fn(
    model_name: str,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
    max_tool_turns: int = DEFAULT_MAX_TOOL_LOOP_TURNS,
) -> Callable[..., Awaitable[str]]:
    """Create a generate function for LLM calls.

    The returned function has signature: (messages, tools=None, follow_ups=None) -> str
    Works for both orchestrator stages (with follow_ups) and worker stages (without).

    Args:
        model_name: OpenRouter model identifier
        config: Optional generation config (uses get_generate_config() if None)
        trace_capture: Optional dict for capturing the LLM trace
        max_tool_turns: Maximum number of tool-loop turns for each multi-turn call

    Returns:
        An async function that handles multi-turn generation with tools and follow-ups
    """
    if config is None:
        config = get_generate_config()

    async def generate(
        messages: list,
        tools: list | None = None,
        follow_ups: list[str] | None = None,
        label: str | None = None,
        rewrite_messages: MessageRewriter | None = None,
        rewrite_tools: ToolRewriter | None = None,
    ) -> str:
        chat_messages = dict_messages_to_chat(messages)
        trace_messages = list(chat_messages)

        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model_name=model_name,
                follow_ups=follow_ups,
                tools=tools or [],
                config=config,
                trace_capture=trace_capture,
                log_label=label,
                max_tool_turns=max_tool_turns,
                rewrite_messages=rewrite_messages,
                rewrite_tools=rewrite_tools,
            )
        chat_messages = _rewrite_context_messages(chat_messages, rewrite_messages)
        response = await call_model(model_name, chat_messages, config=config, log_label=label)
        trace_messages.append(response["message"])
        _record_trace_segment(trace_capture, trace_messages, response)
        return response["completion"]

    return generate


def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("JSON parsing error: %s (content length: %d)", e, len(content))
        logger.debug("Content preview: %s", content[:500])
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e


# ---------------------------------------------------------------------------
# Shared validation logic for all validation tools
# ---------------------------------------------------------------------------


def _validate_json_and_format(
    json_str: str,
    validate_fn: Callable[[dict], tuple[Any, list[str]]],
    capture: dict | None = None,
    capture_key: str | None = None,
    capture_result: bool = False,
) -> str:
    """Parse JSON, validate, and format errors.

    Args:
        json_str: Raw JSON string to parse
        validate_fn: (data_dict) -> (validated_result_or_None, error_list)
        capture: Optional dict to store successful results in
        capture_key: Key under which to store in capture dict
        capture_result: If True, store the validated result; if False, store raw data
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}"

    result, errors = validate_fn(data)

    if not errors:
        if capture is not None and capture_key:
            capture[capture_key] = result if capture_result else data
        return "VALID"

    return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)


# ---------------------------------------------------------------------------
# Validation tool factories
# ---------------------------------------------------------------------------


def make_validation_tool(
    name: str,
    description: str,
    param_name: str,
    param_description: str,
    validator: Callable[[dict], tuple[Any, list[str]]],
    capture_key: str,
    capture_result: bool = False,
) -> tuple[Tool, dict]:
    """Generic factory for JSON-validation tools.

    Thin adapter over :func:`make_context_tool` that bridges the
    ``(result, errors)`` validator interface to the ``(context_output, feedback)``
    grounding interface.
    """
    from nof1_causal_lab.flows.context_tool_factory import make_context_tool

    def _adapted(data: dict) -> tuple[dict | None, str]:
        result, errors = validator(data)
        if errors:
            return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)
        value = result if capture_result else data
        return {capture_key: value}, "VALID"

    return make_context_tool(name, description, param_name, param_description, _adapted)


# ---------------------------------------------------------------------------
# Per-turn logging helpers
# ---------------------------------------------------------------------------


def _summarize_output(output: dict[str, Any], elapsed: float) -> str:
    """One-line summary of a normalized response summary for logging."""
    parts = []
    usage = output.get("usage")
    if usage:
        parts.append(f"tokens(in={usage['input_tokens']},out={usage['output_tokens']})")
    parts.append(f"time={elapsed:.1f}s")
    tool_calls = output["message"].get("tool_calls") or []
    if tool_calls:
        names = [
            call.get("function", {}).get("name") or call.get("name", "?") for call in tool_calls
        ]
        parts.append(f"tool_calls={names}")
    else:
        parts.append(f"stop={output.get('stop_reason') or 'end_turn'}")
    text = output.get("completion", "")
    preview = text[:120].replace("\n", " ")
    if preview:
        parts.append(f'preview="{preview}..."' if len(text) > 120 else f'preview="{preview}"')
    return " | ".join(parts)


def _terminal_tool_success(
    tool_messages: list[dict[str, Any]],
    tools: list[Tool],
) -> tuple[str, str] | None:
    """Return the first successful terminal tool result, if any."""
    tool_map = {tool.name: tool for tool in tools}
    recoverable_error_prefixes = ("JSON parse error:", "VALIDATION ERRORS:")
    for tool_message in tool_messages:
        tool_name = str(tool_message.get("name", ""))
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None or not tool_obj.stop_on_success:
            continue
        if tool_message.get("error") is not None:
            continue
        result_text = str(tool_message.get("content", "")).strip()
        if tool_obj.success_output is None:
            if result_text.startswith(recoverable_error_prefixes):
                continue
            return tool_name, result_text
        if result_text == tool_obj.success_output:
            return tool_name, result_text
    return None


def _has_tool_context(messages: list[dict[str, Any]], tools: list[Tool] | None = None) -> bool:
    """Whether the current conversation is in a tool-using phase."""

    if tools:
        return True
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _truncate_tool_error(error_text: str, limit: int = MAX_TOOL_REPAIR_ERROR_CHARS) -> str:
    """Trim provider error payloads before echoing them back to the model."""

    if len(error_text) <= limit:
        return error_text
    return error_text[:limit] + "\n...[truncated]"


def _tool_retry_message(error_text: str, tools: list[Tool] | None) -> dict[str, str]:
    """Build a repair instruction after a malformed or failed tool-call response."""

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


def _is_call_timeout(exc: Exception) -> bool:
    """Whether a model-call exception should be treated as a request timeout."""

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    timeout_class_names = {cls.__name__ for cls in type(exc).__mro__}
    if any("Timeout" in name for name in timeout_class_names):
        return True
    error_text = (str(exc) or "").lower()
    return "timed out" in error_text or "timeout" in error_text


async def _call_model_with_tool_repair(
    messages: list[dict[str, Any]],
    model_name: str,
    tools: list[Tool] | None,
    config: GenerateConfig,
    log_label: str | None,
    trace_messages: list[dict[str, Any]] | None = None,
    max_retries: int = MAX_TOOL_REPAIR_RETRIES,
) -> dict[str, Any]:
    """Call the model and repair malformed tool-call turns by prompting a retry."""

    tool_context = _has_tool_context(messages, tools)
    repair_attempt = 0
    timeout_attempt = 0

    while True:
        attempt_suffixes: list[str] = []
        if timeout_attempt > 0:
            attempt_suffixes.append(f"timeout-retry-{timeout_attempt}")
        if repair_attempt > 0:
            attempt_suffixes.append(f"repair-{repair_attempt}")
        attempt_label = (
            log_label if not attempt_suffixes else _combine_log_label(log_label, *attempt_suffixes)
        )
        try:
            return await call_model(
                model_name,
                messages,
                tools=tools,
                config=config,
                log_label=attempt_label,
            )
        except Exception as exc:
            if _is_call_timeout(exc):
                if timeout_attempt >= MAX_CALL_TIMEOUT_RETRIES:
                    raise
                timeout_attempt += 1
                logger.warning(
                    scoped_log(
                        log_label,
                        "call_model timed out; retrying same request (attempt %d/%d): %s",
                    ),
                    timeout_attempt,
                    MAX_CALL_TIMEOUT_RETRIES,
                    _truncate_tool_error(str(exc) or exc.__class__.__name__, limit=240).replace(
                        "\n", " "
                    ),
                )
                continue
            if not tool_context or repair_attempt >= max_retries:
                raise
            repair_attempt += 1
            error_text = str(exc) or exc.__class__.__name__
            logger.warning(
                scoped_log(
                    log_label,
                    "call_model failed during tool-context turn; retrying with repair prompt "
                    "(attempt %d/%d): %s",
                ),
                repair_attempt,
                max_retries,
                _truncate_tool_error(error_text, limit=240).replace("\n", " "),
            )
            retry_message = _tool_retry_message(error_text, tools)
            messages.append(retry_message)
            if trace_messages is not None:
                trace_messages.append(retry_message)


async def _run_tool_loop(
    context_messages: list[dict[str, Any]],
    trace_messages: list[dict[str, Any]],
    model_name: str,
    tools: list[Tool],
    config: GenerateConfig | None,
    label: str = "tool",
    log_label: str | None = None,
    max_turns: int = DEFAULT_MAX_TOOL_LOOP_TURNS,
    warn_turns: int = WARN_TOOL_LOOP_TURNS,
    rewrite_messages: MessageRewriter | None = None,
    rewrite_tools: ToolRewriter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a tool loop with per-turn logging and an infinite-loop guard.

    Replaces model.generate_loop() with identical semantics but adds:
    - INFO log per turn (tokens, timing, tool calls, content preview)
    - WARNING when turn count hits warn_turns
    - RuntimeError when turn count exceeds max_turns
    """
    _config = config or GenerateConfig()
    t0 = time.monotonic()
    turn = 0
    scoped_label = _combine_log_label(log_label, label)
    warn_turns = min(warn_turns, max_turns)

    while True:
        turn += 1
        if turn > max_turns:
            elapsed = time.monotonic() - t0
            logger.error(
                scoped_log(scoped_label, "exceeded %d turns (elapsed=%.1fs). Terminating."),
                max_turns,
                elapsed,
            )
            raise RuntimeError(f"LLM {label} loop exceeded {max_turns} turns without converging.")
        if turn == warn_turns:
            elapsed = time.monotonic() - t0
            logger.warning(
                scoped_log(
                    scoped_label, "reached %d turns (elapsed=%.1fs). Possible infinite loop."
                ),
                warn_turns,
                elapsed,
            )

        t_turn = time.monotonic()
        current_tools = _rewrite_available_tools(tools, rewrite_tools)
        context_messages = _rewrite_context_messages(context_messages, rewrite_messages)
        output = await _call_model_with_tool_repair(
            context_messages,
            model_name,
            tools=current_tools or None,
            config=_config,
            log_label=_combine_log_label(scoped_label, f"turn-{turn}", "llm"),
            trace_messages=trace_messages,
        )
        context_messages.append(output["message"])
        trace_messages.append(output["message"])
        elapsed_turn = time.monotonic() - t_turn

        logger.info(
            scoped_log(scoped_label, "turn=%d | %s"), turn, _summarize_output(output, elapsed_turn)
        )

        tool_messages: list[dict[str, Any]] = []
        if output["message"].get("tool_calls"):
            tool_messages = await execute_tools(
                output["message"],
                current_tools,
                _config.max_tool_output,
                log_label=_combine_log_label(scoped_label, f"turn-{turn}", "tools"),
            )
            context_messages.extend(tool_messages)
            trace_messages.extend(tool_messages)

        terminal_tool = (
            _terminal_tool_success(tool_messages, current_tools) if tool_messages else None
        )
        if terminal_tool is not None:
            tool_name, result_text = terminal_tool
            elapsed_total = time.monotonic() - t0
            logger.info(
                scoped_log(
                    scoped_label, "terminal tool %s returned %r; stopping after %d turns in %.1fs"
                ),
                tool_name,
                result_text,
                turn,
                elapsed_total,
            )
            return context_messages, output

        if not output["message"].get("tool_calls"):
            elapsed_total = time.monotonic() - t0
            logger.info(
                scoped_log(scoped_label, "completed: %d turns in %.1fs"), turn, elapsed_total
            )
            return context_messages, output


# ---------------------------------------------------------------------------
# Multi-turn generation
# ---------------------------------------------------------------------------


async def multi_turn_generate(
    messages: list[dict[str, Any]],
    model_name: str,
    follow_ups: list[str] | None = None,
    tools: list[Tool] | None = None,
    follow_up_tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
    log_label: str | None = None,
    max_tool_turns: int = DEFAULT_MAX_TOOL_LOOP_TURNS,
    rewrite_messages: MessageRewriter | None = None,
    rewrite_tools: ToolRewriter | None = None,
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Uses a manual tool loop (via _run_tool_loop) instead of model.generate_loop()
    to provide per-turn logging, timing, and an infinite-loop safety guard.

    Args:
        messages: Initial messages (typically system + user prompt)
        model_name: OpenRouter model identifier
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        tools: Optional list of tools the model can use on the first turn
        follow_up_tools: Optional list of tools for follow-up (self-review) turns.
            Defaults to the same tools as the initial turn, so every LLM
            invocation within a stage is grounded by the same validation tool.
            Pass an explicit empty list to disable tools on follow-ups.
        config: Optional generation config
        trace_capture: Optional dict; when provided, the full LLMTrace is stored
            under ``trace_capture["trace"]`` before returning.
        max_tool_turns: Maximum number of tool-loop turns for each tool-using phase
        rewrite_messages: Optional hook that rewrites only the model-facing
            conversation context between turns. The full trace remains append-only.
        rewrite_tools: Optional hook that rewrites the available tool list
            between turns. The underlying trace remains append-only.

    Returns:
        The final completion string
    """
    t0 = time.monotonic()
    context_messages = list(messages)  # Don't mutate original
    trace_messages = list(messages)
    follow_ups = follow_ups or []
    _config = config or GenerateConfig()

    # Default: follow-up turns use the same tools as the initial turn so
    # every LLM output within a stage is grounded by the validation tool.
    # Pass follow_up_tools=[] explicitly to opt out.
    if follow_up_tools is None:
        follow_up_tools = tools or None

    logger.info(
        scoped_log(log_label, "multi_turn_generate starting (tools=%d, follow_ups=%d)"),
        len(tools or []),
        len(follow_ups),
    )

    # --- Initial turn ---
    if tools:
        context_messages, output = await _run_tool_loop(
            context_messages,
            trace_messages,
            model_name,
            tools,
            config,
            label="initial",
            log_label=log_label,
            max_turns=max_tool_turns,
            rewrite_messages=rewrite_messages,
            rewrite_tools=rewrite_tools,
        )
    else:
        t_gen = time.monotonic()
        context_messages = _rewrite_context_messages(context_messages, rewrite_messages)
        output = await _call_model_with_tool_repair(
            context_messages,
            model_name,
            tools=None,
            config=_config,
            log_label=_combine_log_label(log_label, "initial", "llm"),
            trace_messages=trace_messages,
        )
        context_messages.append(output["message"])
        trace_messages.append(output["message"])
        elapsed_gen = time.monotonic() - t_gen
        logger.info(
            scoped_log(log_label, "single-turn | %s"), _summarize_output(output, elapsed_gen)
        )

    last_nonempty = output["completion"]

    # --- Follow-up turns ---
    for i, prompt in enumerate(follow_ups):
        follow_up_label = _combine_log_label(log_label, f"follow-up-{i + 1}")
        logger.info(scoped_log(follow_up_label, "starting (%d/%d)"), i + 1, len(follow_ups))
        follow_up_message = {"role": "user", "content": prompt}
        context_messages.append(follow_up_message)
        trace_messages.append(follow_up_message)

        if follow_up_tools:
            context_messages, output = await _run_tool_loop(
                context_messages,
                trace_messages,
                model_name,
                follow_up_tools,
                config,
                label=f"follow-up-{i + 1}",
                log_label=log_label,
                max_turns=max_tool_turns,
                rewrite_messages=rewrite_messages,
                rewrite_tools=rewrite_tools,
            )
        else:
            t_fu = time.monotonic()
            context_messages = _rewrite_context_messages(context_messages, rewrite_messages)
            output = await _call_model_with_tool_repair(
                context_messages,
                model_name,
                tools=None,
                config=_config,
                log_label=_combine_log_label(follow_up_label, "llm"),
                trace_messages=trace_messages,
            )
            context_messages.append(output["message"])
            trace_messages.append(output["message"])
            elapsed_fu = time.monotonic() - t_fu
            logger.info(
                scoped_log(follow_up_label, "%d/%d | %s"),
                i + 1,
                len(follow_ups),
                _summarize_output(output, elapsed_fu),
            )

        if output["completion"] and output["completion"].strip():
            last_nonempty = output["completion"]

    # --- Finalize ---
    _record_trace_segment(trace_capture, trace_messages, output)

    elapsed_total = time.monotonic() - t0
    logger.info(scoped_log(log_label, "multi_turn_generate completed in %.1fs"), elapsed_total)
    return last_nonempty


# ---------------------------------------------------------------------------
# Legacy note
# ---------------------------------------------------------------------------
# The previous ``LLMStageContext`` + ``attach_trace`` helpers have been
# removed; stages now use :class:`nof1_causal_lab.utils.agent_session.ScopedSessionFactory`
# which subsumes both trace accumulation and lifecycle logging. The
# ``multi_turn_generate`` + ``make_generate_fn`` pair is kept for eval
# scripts outside the main pipeline; new production code should use
# :func:`nof1_causal_lab.utils.agent_session_factory.open_session`.
