"""Shared LLM utilities for multi-turn generation."""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.litellm_client import (
    GenerateConfig,
    Tool,
    call_model,
    execute_tools,
    normalize_message,
)

logger = get_prefect_logger(__name__)


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


# ---------------------------------------------------------------------------
# Type aliases for generate functions (unified)
# ---------------------------------------------------------------------------

GenerateFn = Callable[..., Awaitable[str]]


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
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
        reasoning_history="all",  # Preserve reasoning across tool calls (required by Gemini)
    )


def dict_messages_to_chat(messages: list[dict]) -> list[dict[str, Any]]:
    """Normalize dict messages for LiteLLM/OpenAI chat format.

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


# ---------------------------------------------------------------------------
# Generate function factory (unified for orchestrator and worker)
# ---------------------------------------------------------------------------


def make_generate_fn(
    model_name: str,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
) -> GenerateFn:
    """Create a generate function for LLM calls.

    The returned function has signature: (messages, tools=None, follow_ups=None) -> str
    Works for both orchestrator stages (with follow_ups) and worker stages (without).

    Args:
        model_name: LiteLLM model identifier
        config: Optional generation config (uses get_generate_config() if None)
        trace_capture: Optional dict for capturing the LLM trace

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
    ) -> str:
        chat_messages = dict_messages_to_chat(messages)

        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model_name=model_name,
                follow_ups=follow_ups,
                tools=tools or [],
                config=config,
                trace_capture=trace_capture,
                log_label=label,
            )
        response = await call_model(model_name, chat_messages, config=config, log_label=label)
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

    Thin adapter over :func:`make_stage_tool` that bridges the
    ``(result, errors)`` validator interface to the ``(stage_output, feedback)``
    grounding interface.
    """
    from causal_ssm_agent.flows.stages.stage_tools import make_stage_tool

    def _adapted(data: dict) -> tuple[dict | None, str]:
        result, errors = validator(data)
        if errors:
            return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)
        value = result if capture_result else data
        return {capture_key: value}, "VALID"

    return make_stage_tool(name, description, param_name, param_description, _adapted)


# ---------------------------------------------------------------------------
# Per-turn logging helpers
# ---------------------------------------------------------------------------

MAX_TOOL_LOOP_TURNS = 40
WARN_TOOL_LOOP_TURNS = 10
MAX_TOOL_REPAIR_RETRIES = 1
MAX_TOOL_REPAIR_ERROR_CHARS = 1200


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
    for tool_message in tool_messages:
        tool_name = str(tool_message.get("name", ""))
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None or not tool_obj.stop_on_success:
            continue
        if tool_message.get("error") is not None:
            continue
        result_text = str(tool_message.get("content", "")).strip()
        if tool_obj.success_output is None or result_text == tool_obj.success_output:
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


async def _call_model_with_tool_repair(
    messages: list[dict[str, Any]],
    model_name: str,
    tools: list[Tool] | None,
    config: GenerateConfig,
    log_label: str | None,
    max_retries: int = MAX_TOOL_REPAIR_RETRIES,
) -> dict[str, Any]:
    """Call the model and repair malformed tool-call turns by prompting a retry."""

    tool_context = _has_tool_context(messages, tools)
    attempt = 0

    while True:
        attempt_label = (
            log_label if attempt == 0 else _combine_log_label(log_label, f"repair-{attempt}")
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
            if not tool_context or attempt >= max_retries:
                raise
            attempt += 1
            error_text = str(exc) or exc.__class__.__name__
            logger.warning(
                scoped_log(
                    log_label,
                    "call_model failed during tool-context turn; retrying with repair prompt "
                    "(attempt %d/%d): %s",
                ),
                attempt,
                max_retries,
                _truncate_tool_error(error_text, limit=240).replace("\n", " "),
            )
            messages.append(_tool_retry_message(error_text, tools))


async def _run_tool_loop(
    messages: list[dict[str, Any]],
    model_name: str,
    tools: list[Tool],
    config: GenerateConfig | None,
    label: str = "tool",
    log_label: str | None = None,
    max_turns: int = MAX_TOOL_LOOP_TURNS,
    warn_turns: int = WARN_TOOL_LOOP_TURNS,
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
        output = await _call_model_with_tool_repair(
            messages,
            model_name,
            tools=tools,
            config=_config,
            log_label=_combine_log_label(scoped_label, f"turn-{turn}", "llm"),
        )
        messages.append(output["message"])
        elapsed_turn = time.monotonic() - t_turn

        logger.info(
            scoped_log(scoped_label, "turn=%d | %s"), turn, _summarize_output(output, elapsed_turn)
        )

        tool_messages: list[dict[str, Any]] = []
        if output["message"].get("tool_calls"):
            tool_messages = await execute_tools(
                output["message"],
                tools,
                _config.max_tool_output,
                log_label=_combine_log_label(scoped_label, f"turn-{turn}", "tools"),
            )
            messages.extend(tool_messages)

        terminal_tool = _terminal_tool_success(tool_messages, tools) if tool_messages else None
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
            return messages, output

        if not output["message"].get("tool_calls"):
            elapsed_total = time.monotonic() - t0
            logger.info(
                scoped_log(scoped_label, "completed: %d turns in %.1fs"), turn, elapsed_total
            )
            return messages, output


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
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Uses a manual tool loop (via _run_tool_loop) instead of model.generate_loop()
    to provide per-turn logging, timing, and an infinite-loop safety guard.

    Args:
        messages: Initial messages (typically system + user prompt)
        model_name: LiteLLM model identifier
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        tools: Optional list of tools the model can use on the first turn
        follow_up_tools: Optional list of tools for follow-up (self-review) turns.
            Defaults to the same tools as the initial turn, so every LLM
            invocation within a stage is grounded by the same validation tool.
            Pass an explicit empty list to disable tools on follow-ups.
        config: Optional generation config
        trace_capture: Optional dict; when provided, the full LLMTrace is stored
            under ``trace_capture["trace"]`` before returning.

    Returns:
        The final completion string
    """
    t0 = time.monotonic()
    messages = list(messages)  # Don't mutate original
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
        messages, output = await _run_tool_loop(
            messages,
            model_name,
            tools,
            config,
            label="initial",
            log_label=log_label,
        )
    else:
        t_gen = time.monotonic()
        output = await _call_model_with_tool_repair(
            messages,
            model_name,
            tools=None,
            config=_config,
            log_label=_combine_log_label(log_label, "initial", "llm"),
        )
        messages.append(output["message"])
        elapsed_gen = time.monotonic() - t_gen
        logger.info(
            scoped_log(log_label, "single-turn | %s"), _summarize_output(output, elapsed_gen)
        )

    last_nonempty = output["completion"]

    # --- Follow-up turns ---
    for i, prompt in enumerate(follow_ups):
        follow_up_label = _combine_log_label(log_label, f"follow-up-{i + 1}")
        logger.info(scoped_log(follow_up_label, "starting (%d/%d)"), i + 1, len(follow_ups))
        messages.append({"role": "user", "content": prompt})

        if follow_up_tools:
            messages, output = await _run_tool_loop(
                messages,
                model_name,
                follow_up_tools,
                config,
                label=f"follow-up-{i + 1}",
                log_label=log_label,
            )
        else:
            t_fu = time.monotonic()
            output = await _call_model_with_tool_repair(
                messages,
                model_name,
                tools=None,
                config=_config,
                log_label=_combine_log_label(follow_up_label, "llm"),
            )
            messages.append(output["message"])
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
    if trace_capture is not None:
        trace_capture["trace"] = _build_trace(messages, output)

    elapsed_total = time.monotonic() - t0
    logger.info(scoped_log(log_label, "multi_turn_generate completed in %.1fs"), elapsed_total)
    return last_nonempty


# ---------------------------------------------------------------------------
# LLMStageContext — eliminates per-stage LLM trace boilerplate
# ---------------------------------------------------------------------------


class LLMStageContext:
    """Encapsulates trace capture, generate function creation, and lifecycle logging.

    Replaces the repeated boilerplate of:
        trace_capture = {}
        generate = make_generate_fn(model, trace_capture=trace_capture)
        ...
        attach_trace(output, trace_capture)

    Usage::

        async with LLMStageContext("stage-1a") as ctx:
            generate = ctx.make_generate(model_name)
            # ... run stage logic ...
            return ctx.finalize({"latent_model": ...})
            # output now has llm_trace attached; lifecycle logged automatically

    Can also be used without ``async with`` — lifecycle logging still works
    via :meth:`make_generate` (start) and :meth:`finalize` (completion).
    """

    def __init__(self, stage_id: str) -> None:
        self.stage_id = stage_id
        self._t0 = time.monotonic()
        self._model_name: str | None = None
        self._trace_capture: dict = {}

    @property
    def trace_capture(self) -> dict:
        """Direct access to the trace capture dict (for advanced use)."""
        return self._trace_capture

    def make_generate(self, model_name: str, config: GenerateConfig | None = None) -> GenerateFn:
        """Create a generate function wired to this context's trace capture."""
        self._model_name = model_name
        logger.info("[%s] starting (model=%s)", self.stage_id, model_name)
        return make_generate_fn(
            model_name,
            config=config,
            trace_capture=self._trace_capture,
        )

    def finalize(self, output: dict) -> dict:
        """Attach the captured LLM trace to the output dict and return it."""
        elapsed = time.monotonic() - self._t0
        attach_trace(output, self._trace_capture)
        # Build a concise completion summary
        parts = [f"completed in {elapsed:.1f}s"]
        trace: LLMTrace | None = self._trace_capture.get("trace")
        if trace and trace.usage:
            u = trace.usage
            parts.append(f"tokens(in={u.input_tokens},out={u.output_tokens})")
        logger.info("[%s] %s", self.stage_id, " ".join(parts))
        return output

    async def __aenter__(self) -> "LLMStageContext":
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> bool:
        if exc_type is not None:
            elapsed = time.monotonic() - self._t0
            logger.error("[%s] failed after %.1fs: %s", self.stage_id, elapsed, exc_val)
        return False


# ---------------------------------------------------------------------------
# Trace capture helper
# ---------------------------------------------------------------------------


def attach_trace(output: dict, trace_capture: dict) -> None:
    """Attach LLM trace to output dict if available.

    Replaces the repeated boilerplate:
        trace = trace_capture.get("trace")
        if trace is not None:
            out["llm_trace"] = trace.model_dump(mode="json")
    """
    trace = trace_capture.get("trace")
    if trace is not None:
        output["llm_trace"] = trace.model_dump(mode="json")
