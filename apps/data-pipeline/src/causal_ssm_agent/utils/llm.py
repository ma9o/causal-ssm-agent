"""Shared LLM utilities for multi-turn generation."""

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.litellm_client import (
    GenerateConfig,
    Tool,
    call_model,
    execute_tools,
    normalize_message,
    tool,
)

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from causal_ssm_agent.orchestrator.schemas import LatentModel


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
# Live trace persistence (intermediate disk writes)
# ---------------------------------------------------------------------------

_RESULT_STORAGE = Path("results")


def make_live_trace_path(stage_id: str) -> Path:
    """Create a path for live trace persistence.

    Writes to the same ``results/{flow_run_id}/{stage_id}.json`` file that
    ``persist_web_result`` will eventually overwrite with the full stage output.
    This lets the frontend display intermediate LLM conversation state while
    a stage is still running.

    Uses the Prefect flow run ID when running inside a flow, otherwise
    falls back to a timestamp-based directory.

    Args:
        stage_id: Stage identifier (e.g. "stage-1a", "stage-4")

    Returns:
        Path like ``results/{run_id}/{stage_id}.json``
    """
    run_id = None
    try:
        from prefect.runtime import flow_run

        run_id = flow_run.id
    except Exception:
        logger.debug("Could not get Prefect flow run ID; using timestamp fallback")
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    return _RESULT_STORAGE / str(run_id) / f"{stage_id}.json"


def _persist_partial_trace(
    messages: list[dict[str, Any]],
    trace_path: Path,
    label: str,
    turn: int,
    elapsed: float,
) -> None:
    """Write accumulated messages to disk as a partial stage result.

    Builds a ``PartialStageResult`` (a subset of the full stage contract with
    only ``llm_trace`` + ``_live`` metadata) and serialises it to disk so the
    frontend can render intermediate conversation state.

    Overwrites the file each turn. Failures are logged but never bubble up.
    """
    from causal_ssm_agent.flows.stages.contracts import LiveMetadata, PartialStageResult

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        partial = PartialStageResult(  # ty: ignore[missing-argument]
            llm_trace=LLMTrace(
                messages=[_chat_message_to_trace(m) for m in messages],
                total_time_seconds=round(elapsed, 1),
            ),
            live=LiveMetadata(  # ty: ignore[unknown-argument]
                status="running",
                label=label,
                turn=turn,
                elapsed_seconds=round(elapsed, 1),
            ),
        )
        trace_path.write_text(partial.model_dump_json(indent=2, by_alias=True))
    except Exception:
        logger.debug("Failed to write partial trace to %s", trace_path, exc_info=True)


# ---------------------------------------------------------------------------
# Type aliases for generate functions (unified)
# ---------------------------------------------------------------------------

GenerateFn = Callable[..., Awaitable[str]]

# Backward-compatible aliases
OrchestratorGenerateFn = GenerateFn
WorkerGenerateFn = GenerateFn


def _combine_log_label(*parts: str | None) -> str | None:
    """Join non-empty label fragments into a stable log scope."""
    labels = [part for part in parts if part]
    if not labels:
        return None
    return " / ".join(labels)


def _scoped(label: str | None, msg: str) -> str:
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
        verbose_logging=llm.verbose_logging,
        log_reasoning=llm.log_reasoning,
        log_output_char_limit=llm.log_output_char_limit,
    )


def get_stage2_generate_config() -> GenerateConfig:
    """Get a Stage-2-tuned GenerateConfig for worker extraction."""
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
        reasoning_history="all",  # Preserve reasoning across tool retries when validation fails.
        verbose_logging=llm.verbose_logging,
        log_reasoning=llm.log_reasoning,
        log_output_char_limit=llm.log_output_char_limit,
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
    trace_path: Path | None = None,
) -> GenerateFn:
    """Create a generate function for LLM calls.

    The returned function has signature: (messages, tools=None, follow_ups=None) -> str
    Works for both orchestrator stages (with follow_ups) and worker stages (without).

    Args:
        model_name: LiteLLM model identifier
        config: Optional generation config (uses get_generate_config() if None)
        trace_capture: Optional dict for capturing the LLM trace
        trace_path: Optional path for live trace persistence (partial JSON written
            after each LLM turn so agents can inspect mid-run state)

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
                trace_path=trace_path,
                log_label=label,
            )
        response = await call_model(model_name, chat_messages, config=config, log_label=label)
        return response["completion"]

    return generate


# Backward-compatible aliases
make_orchestrator_generate_fn = make_generate_fn
make_worker_generate_fn = make_generate_fn


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


def _make_validation_tool(
    name: str,
    description: str,
    param_name: str,
    param_description: str,
    validator: Callable[[dict], tuple[Any, list[str]]],
    capture_key: str,
    capture_result: bool = False,
) -> tuple[Tool, dict]:
    """Generic factory for JSON-validation tools.

    Builds a Tool that parses JSON from the LLM, runs a validator, captures
    valid results, and returns "VALID" or a formatted error list. All four
    stage-specific validation tools share this core pattern.
    """
    capture: dict = {}

    async def _execute(**kwargs: str) -> str:
        return _validate_json_and_format(
            kwargs[param_name],
            validator,
            capture=capture,
            capture_key=capture_key,
            capture_result=capture_result,
        )

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                param_name: {"type": "string", "description": param_description}
            },
            "required": [param_name],
            "additionalProperties": False,
        },
        execute=_execute,
        stop_on_success=True,
        success_output="VALID",
    ), capture


def make_validate_latent_model_tool() -> tuple[Tool, dict]:
    """Create a validation tool for latent model JSON (Stage 1a)."""
    from causal_ssm_agent.orchestrator.schemas import validate_latent_model

    return _make_validation_tool(
        name="validate_latent_model_tool",
        description="Tool for validating latent model JSON (Stage 1a).",
        param_name="structure_json",
        param_description="The JSON string containing the latent model to validate.",
        validator=validate_latent_model,
        capture_key="latent",
    )


def make_validate_measurement_model_tool(latent_model: "LatentModel") -> tuple[Tool, dict]:
    """Create a validation tool for measurement model, bound to a latent model."""
    from causal_ssm_agent.models.ssm_compiler import (
        validate_measurement_model_for_compilation,
    )

    return _make_validation_tool(
        name="validate_measurement_model_tool",
        description="Tool for validating measurement model JSON plus compiler constraints.",
        param_name="measurement_json",
        param_description="The JSON string containing the measurement model to validate.",
        validator=lambda data: validate_measurement_model_for_compilation(data, latent_model),
        capture_key="measurement",
    )


def make_validate_model_spec_tool(
    causal_spec: dict,
    *,
    resolved_likelihoods: list[dict] | None = None,
    ambiguous_indicators: list[dict] | None = None,
    parameters: list[dict] | None = None,
    loading_params: list[dict] | None = None,  # noqa: ARG001
) -> tuple[Tool, dict]:
    """Create a validation tool for model spec, bound to a causal spec.

    When skeleton parts (resolved_likelihoods, parameters, etc.) are provided,
    validates ModelSpecDecisions and merges with the skeleton. Otherwise falls
    back to validating a full ModelSpec dict.
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators

    indicators = get_indicators(causal_spec)

    def _validator(data: dict) -> tuple[Any, list[str]]:
        if resolved_likelihoods is not None and parameters is not None:
            from causal_ssm_agent.orchestrator.schemas_model import (
                validate_model_spec_decisions_dict,
            )

            return validate_model_spec_decisions_dict(
                data,
                resolved_likelihoods=resolved_likelihoods,
                ambiguous_indicators=ambiguous_indicators or [],
                parameters=parameters,
            )
        from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict

        return validate_model_spec_dict(data, indicators=indicators or None)

    return _make_validation_tool(
        name="validate_model_spec_tool",
        description="Tool for validating model specification JSON (Stage 4).",
        param_name="model_spec_json",
        param_description="The JSON string containing the model spec to validate.",
        validator=_validator,
        capture_key="spec",
        capture_result=True,
    )


def make_worker_tools(schema: dict) -> tuple[list[Tool], dict]:
    """Create the standard toolset for worker agents."""
    tool_obj, capture = make_validate_worker_output_tool(schema)
    return [tool_obj], capture


def make_validate_worker_output_tool(schema: dict) -> tuple[Tool, dict]:
    """Create a validation tool for worker output, bound to a specific schema."""
    from causal_ssm_agent.workers.schemas import validate_worker_output

    return _make_validation_tool(
        name="validate_extractions",
        description="Tool for validating worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output to validate.",
        validator=lambda data: validate_worker_output(data, schema),
        capture_key="output",
    )


@tool
def calculate():
    """Tool for evaluating simple arithmetic calculations."""

    async def execute(expression: str) -> str:
        """
        Evaluate a simple arithmetic expression.

        Args:
            expression: A Python arithmetic expression (e.g., "2 + 3 * 4", "100 / 5", "(10 + 5) * 2", "10 % 3", "2 ** 8")

        Returns:
            The result of the calculation, or an error message if evaluation fails.
        """
        import ast
        import operator

        _OPERATORS: dict[type, object] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _safe_eval(node: ast.AST) -> float | int:
            if isinstance(node, ast.Expression):
                return _safe_eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
                op = _OPERATORS[type(node.op)]
                return op(_safe_eval(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
                op = _OPERATORS[type(node.op)]
                return op(_safe_eval(node.left), _safe_eval(node.right))
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return str(result)
        except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
            return f"Error evaluating expression: {e}"

    return execute


@tool
def parse_date():
    """Tool for parsing dates into a human-readable spelled out format."""

    async def execute(date_string: str) -> str:
        """
        Parse a date or timestamp into spelled out format.

        Args:
            date_string: A date or timestamp string (e.g., "2024-03-15", "2024-03-15T10:30:00")

        Returns:
            Spelled out date (e.g., "Friday, March 15, 2024") or an error message if parsing fails.
        """
        from datetime import datetime

        # Common formats to try
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_string.strip(), fmt)
                return dt.strftime("%A, %B %d, %Y")  # e.g., "Friday, March 15, 2024"
            except ValueError:
                continue

        return f"Could not parse date: {date_string}"

    return execute


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
        names = [call["name"] for call in tool_calls]
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
        attempt_label = log_label if attempt == 0 else _combine_log_label(log_label, f"repair-{attempt}")
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
                _scoped(
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
    trace_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a tool loop with per-turn logging and an infinite-loop guard.

    Replaces model.generate_loop() with identical semantics but adds:
    - INFO log per turn (tokens, timing, tool calls, content preview)
    - WARNING when turn count hits warn_turns
    - RuntimeError when turn count exceeds max_turns
    - Optional partial trace written to disk after each turn
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
                _scoped(scoped_label, "exceeded %d turns (elapsed=%.1fs). Terminating."),
                max_turns,
                elapsed,
            )
            raise RuntimeError(f"LLM {label} loop exceeded {max_turns} turns without converging.")
        if turn == warn_turns:
            elapsed = time.monotonic() - t0
            logger.warning(
                _scoped(scoped_label, "reached %d turns (elapsed=%.1fs). Possible infinite loop."),
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
            _scoped(scoped_label, "turn=%d | %s"), turn, _summarize_output(output, elapsed_turn)
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

        if trace_path is not None:
            _persist_partial_trace(messages, trace_path, label, turn, time.monotonic() - t0)

        terminal_tool = _terminal_tool_success(tool_messages, tools) if tool_messages else None
        if terminal_tool is not None:
            tool_name, result_text = terminal_tool
            elapsed_total = time.monotonic() - t0
            logger.info(
                _scoped(
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
            logger.info(_scoped(scoped_label, "completed: %d turns in %.1fs"), turn, elapsed_total)
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
    trace_path: Path | None = None,
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
            Defaults to None, meaning follow-up turns use no tools (plain generation).
            This prevents the LLM from re-invoking validation tools during self-review
            and potentially overwriting a previously captured valid model.
        config: Optional generation config
        trace_capture: Optional dict; when provided, the full LLMTrace is stored
            under ``trace_capture["trace"]`` before returning.
        trace_path: Optional path; when provided, a partial JSON trace is written
            to disk after each LLM turn for live observability.

    Returns:
        The final completion string
    """
    t0 = time.monotonic()
    messages = list(messages)  # Don't mutate original
    follow_ups = follow_ups or []
    _config = config or GenerateConfig()

    logger.info(
        _scoped(log_label, "multi_turn_generate starting (tools=%d, follow_ups=%d)"),
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
            trace_path=trace_path,
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
            _scoped(log_label, "single-turn | %s"), _summarize_output(output, elapsed_gen)
        )

    last_nonempty = output["completion"]

    # --- Follow-up turns ---
    for i, prompt in enumerate(follow_ups):
        follow_up_label = _combine_log_label(log_label, f"follow-up-{i + 1}")
        logger.info(_scoped(follow_up_label, "starting (%d/%d)"), i + 1, len(follow_ups))
        messages.append({"role": "user", "content": prompt})

        if follow_up_tools:
            messages, output = await _run_tool_loop(
                messages,
                model_name,
                follow_up_tools,
                config,
                label=f"follow-up-{i + 1}",
                log_label=log_label,
                trace_path=trace_path,
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
                _scoped(follow_up_label, "%d/%d | %s"),
                i + 1,
                len(follow_ups),
                _summarize_output(output, elapsed_fu),
            )
            if trace_path is not None:
                _persist_partial_trace(
                    messages, trace_path, f"follow-up-{i + 1}", 1, time.monotonic() - t0
                )

        if output["completion"] and output["completion"].strip():
            last_nonempty = output["completion"]

    # --- Finalize ---
    if trace_capture is not None:
        trace_capture["trace"] = _build_trace(messages, output)

    elapsed_total = time.monotonic() - t0
    logger.info(_scoped(log_label, "multi_turn_generate completed in %.1fs"), elapsed_total)
    return last_nonempty


# ---------------------------------------------------------------------------
# StageContext — eliminates per-stage trace boilerplate
# ---------------------------------------------------------------------------


class StageContext:
    """Encapsulates trace capture and generate function creation for a stage.

    Replaces the repeated boilerplate of:
        trace_capture = {}
        generate = make_generate_fn(model, trace_capture=trace_capture,
                                     trace_path=make_live_trace_path(stage_id))
        ...
        attach_trace(output, trace_capture)

    Usage::

        ctx = StageContext("stage-1a")
        generate = ctx.generate(model_name)
        # ... run stage logic ...
        output = ctx.finalize({"latent_model": ..., "treatments": ...})
        # output now has llm_trace attached
    """

    def __init__(self, stage_id: str, *, live_trace: bool = True) -> None:
        self.stage_id = stage_id
        self._trace_capture: dict = {}
        self._trace_path = make_live_trace_path(stage_id) if live_trace else None

    @property
    def trace_capture(self) -> dict:
        """Direct access to the trace capture dict (for advanced use)."""
        return self._trace_capture

    def make_generate(
        self, model_name: str, config: GenerateConfig | None = None
    ) -> GenerateFn:
        """Create a generate function wired to this context's trace capture."""
        return make_generate_fn(
            model_name,
            config=config,
            trace_capture=self._trace_capture,
            trace_path=self._trace_path,
        )

    def finalize(self, output: dict) -> dict:
        """Attach the captured LLM trace to the output dict and return it."""
        attach_trace(output, self._trace_capture)
        return output


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
