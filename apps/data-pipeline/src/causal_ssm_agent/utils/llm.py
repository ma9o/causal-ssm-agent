"""Shared LLM utilities for multi-turn generation."""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    execute_tools,
)
from inspect_ai.tool import Tool, tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

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


def _chat_message_to_trace(msg: "ChatMessage") -> TraceMessage:
    """Convert an inspect_ai ChatMessage to a TraceMessage."""
    from inspect_ai._util.content import ContentReasoning, ContentText

    role = msg.role
    content_text = ""
    reasoning_text = None
    tool_calls_list = None
    tool_name = None
    tool_result = None
    tool_is_error = False

    # Extract text and reasoning from content
    if isinstance(msg.content, str):
        content_text = msg.content
    elif isinstance(msg.content, list):
        text_parts = []
        reasoning_parts = []
        for part in msg.content:
            if isinstance(part, ContentText):
                text_parts.append(part.text)
            elif isinstance(part, ContentReasoning):
                reasoning_parts.append(part.reasoning)
        content_text = "\n".join(text_parts)
        if reasoning_parts:
            reasoning_text = "\n".join(reasoning_parts)

    # Extract tool calls from assistant messages
    if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
        tool_calls_list = [
            {"name": tc.function, "arguments": tc.arguments} for tc in msg.tool_calls
        ]

    # Extract tool results from tool messages
    if isinstance(msg, ChatMessageTool):
        tool_name = msg.function
        tool_result = content_text
        tool_is_error = msg.error is not None

    return TraceMessage(
        role=role,
        content=content_text,
        reasoning=reasoning_text,
        tool_calls=tool_calls_list,
        tool_name=tool_name,
        tool_result=tool_result,
        tool_is_error=tool_is_error,
    )


def _build_trace(all_messages: list["ChatMessage"], output: ModelOutput) -> LLMTrace:
    """Build an LLMTrace from a final message list and ModelOutput."""
    messages = [_chat_message_to_trace(m) for m in all_messages]
    usage = TraceUsage()
    if output.usage:
        usage = TraceUsage(
            input_tokens=output.usage.input_tokens,
            output_tokens=output.usage.output_tokens,
            reasoning_tokens=output.usage.reasoning_tokens,
        )
    return LLMTrace(
        messages=messages,
        model=output.model or "",
        total_time_seconds=output.time or 0.0,
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
    messages: list["ChatMessage"],
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


def get_generate_config() -> GenerateConfig:
    """Get standard GenerateConfig for all model calls.

    Reads settings from config.yaml llm section.
    """
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=cast(
            "Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] | None",
            llm.reasoning_effort,
        ),
        reasoning_history="all",  # Preserve reasoning across tool calls (required by Gemini)
    )


def dict_messages_to_chat(messages: list[dict]) -> list["ChatMessage"]:
    """Convert dict messages to ChatMessage objects.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        List of ChatMessage objects (ChatMessageSystem or ChatMessageUser)
    """
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            chat_messages.append(ChatMessageSystem(content=msg["content"]))
        elif msg["role"] == "user":
            chat_messages.append(ChatMessageUser(content=msg["content"]))
    return chat_messages


# ---------------------------------------------------------------------------
# Generate function factory (unified for orchestrator and worker)
# ---------------------------------------------------------------------------


def make_generate_fn(
    model: Model,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
    trace_path: Path | None = None,
) -> GenerateFn:
    """Create a generate function for LLM calls.

    The returned function has signature: (messages, tools=None, follow_ups=None) -> str
    Works for both orchestrator stages (with follow_ups) and worker stages (without).

    Args:
        model: The model to use for generation
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
    ) -> str:
        chat_messages = dict_messages_to_chat(messages)

        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model=model,
                follow_ups=follow_ups,
                tools=tools or [],
                config=config,
                trace_capture=trace_capture,
                trace_path=trace_path,
            )
        response = await model.generate(chat_messages, config=config)
        return response.completion

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


def make_validate_latent_model_tool() -> tuple[Tool, dict]:
    """Create a validation tool for latent model JSON that captures the last valid result.

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["latent"] for the last validated LatentModel dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_latent_model_tool():
        """Tool for validating latent model JSON (Stage 1a)."""

        async def execute(structure_json: str) -> str:
            """
            Validate a latent model and return all validation errors.

            Args:
                structure_json: The JSON string containing the latent model to validate.

            Returns:
                "VALID" if the structure passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas import validate_latent_model

            return _validate_json_and_format(
                structure_json,
                validate_latent_model,
                capture=capture,
                capture_key="latent",
            )

        return execute

    return validate_latent_model_tool(), capture


def make_validate_measurement_model_tool(latent_model: "LatentModel") -> tuple[Tool, dict]:
    """Create a validation tool for measurement model, bound to a latent model.

    Args:
        latent_model: The latent model to validate against

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["measurement"] for the last validated measurement dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_measurement_model_tool():
        """Tool for validating measurement model JSON (Stage 1b)."""

        async def execute(measurement_json: str) -> str:
            """
            Validate a measurement model and return all validation errors.

            Args:
                measurement_json: The JSON string containing the measurement model to validate.

            Returns:
                "VALID" if the model passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas import validate_measurement_model

            return _validate_json_and_format(
                measurement_json,
                lambda data: validate_measurement_model(data, latent_model),
                capture=capture,
                capture_key="measurement",
            )

        return execute

    return validate_measurement_model_tool(), capture


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
    back to validating a full ModelSpec dict (backward-compatible).

    Args:
        causal_spec: The full CausalSpec dict (to extract indicators for dtype checking)
        resolved_likelihoods: Pre-computed deterministic likelihoods
        ambiguous_indicators: Indicators needing LLM decisions
        parameters: Pre-computed parameters
        loading_params: Loading parameters needing constraint decisions

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["spec"] for the last validated ModelSpec (or None).
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators

    indicators = get_indicators(causal_spec)
    capture: dict = {}

    @tool
    def validate_model_spec_tool():
        """Tool for validating model specification JSON (Stage 4)."""

        async def execute(model_spec_json: str) -> str:
            """
            Validate a model specification and return all validation errors.

            Args:
                model_spec_json: The JSON string containing the model spec to validate.

            Returns:
                "VALID" if the spec passes validation, otherwise a list of all errors found.
            """
            if resolved_likelihoods is not None and parameters is not None:
                from causal_ssm_agent.orchestrator.schemas_model import (
                    validate_model_spec_decisions_dict,
                )

                return _validate_json_and_format(
                    model_spec_json,
                    lambda data: validate_model_spec_decisions_dict(
                        data,
                        resolved_likelihoods=resolved_likelihoods,
                        ambiguous_indicators=ambiguous_indicators or [],
                        parameters=parameters,
                    ),
                    capture=capture,
                    capture_key="spec",
                    capture_result=True,
                )
            from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict

            return _validate_json_and_format(
                model_spec_json,
                lambda data: validate_model_spec_dict(data, indicators=indicators or None),
                capture=capture,
                capture_key="spec",
                capture_result=True,
            )

        return execute

    return validate_model_spec_tool(), capture


def make_worker_tools(schema: dict) -> tuple[list[Tool], dict]:
    """Create the standard toolset for worker agents.

    This is the single source of truth for worker tools.
    Used by both production workers and evals.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        Tuple of (tools_list, capture_dict). After generate_loop, check
        capture["output"] for the last validated worker output dict (or None).
    """
    tool, capture = make_validate_worker_output_tool(schema)
    return [tool, parse_date(), calculate()], capture


def make_validate_worker_output_tool(schema: dict) -> tuple[Tool, dict]:
    """Create a validation tool for worker output, bound to a specific schema.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["output"] for the last validated worker output dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_extractions():
        """Tool for validating worker extraction output JSON."""

        async def execute(output_json: str) -> str:
            """
            Validate worker extractions and return all validation errors.

            Args:
                output_json: The JSON string containing the worker output to validate.

            Returns:
                "VALID" if the output passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.workers.schemas import validate_worker_output

            return _validate_json_and_format(
                output_json,
                lambda data: validate_worker_output(data, schema),
                capture=capture,
                capture_key="output",
            )

        return execute

    return validate_extractions(), capture


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
        # Whitelist of allowed characters for safe evaluation
        allowed_chars = set("0123456789+-*/%()._ ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and +-*/%()._ are allowed."

        try:
            result = eval(expression)
            return str(result)
        except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
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

MAX_TOOL_LOOP_TURNS = 25
WARN_TOOL_LOOP_TURNS = 10


def _summarize_output(output: ModelOutput, elapsed: float) -> str:
    """One-line summary of a ModelOutput for logging."""
    parts = []
    if output.usage:
        parts.append(f"tokens(in={output.usage.input_tokens},out={output.usage.output_tokens})")
    parts.append(f"time={elapsed:.1f}s")
    if output.message.tool_calls:
        names = [tc.function for tc in output.message.tool_calls]
        parts.append(f"tool_calls={names}")
    else:
        parts.append(f"stop={output.stop_reason or 'end_turn'}")
    text = output.completion or ""
    preview = text[:120].replace("\n", " ")
    if preview:
        parts.append(f'preview="{preview}..."' if len(text) > 120 else f'preview="{preview}"')
    return " | ".join(parts)


async def _run_tool_loop(
    messages: list["ChatMessage"],
    model: Model,
    tools: list[Tool],
    config: GenerateConfig | None,
    label: str = "tool",
    max_turns: int = MAX_TOOL_LOOP_TURNS,
    warn_turns: int = WARN_TOOL_LOOP_TURNS,
    trace_path: Path | None = None,
) -> tuple[list["ChatMessage"], ModelOutput]:
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

    while True:
        turn += 1
        if turn > max_turns:
            elapsed = time.monotonic() - t0
            logger.error(
                "[%s] exceeded %d turns (elapsed=%.1fs). Terminating.",
                label,
                max_turns,
                elapsed,
            )
            raise RuntimeError(f"LLM {label} loop exceeded {max_turns} turns without converging.")
        if turn == warn_turns:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[%s] reached %d turns (elapsed=%.1fs). Possible infinite loop.",
                label,
                warn_turns,
                elapsed,
            )

        t_turn = time.monotonic()
        output = await model.generate(input=messages, tools=tools, config=_config)
        messages.append(output.message)
        elapsed_turn = time.monotonic() - t_turn

        logger.info(
            "[%s] turn=%d | %s",
            label,
            turn,
            _summarize_output(output, elapsed_turn),
        )

        if output.message.tool_calls:
            tool_messages, tool_output = await execute_tools(
                messages,
                tools,
                _config.max_tool_output,
            )
            messages.extend(tool_messages)
            if tool_output is not None:
                output = tool_output

        if trace_path is not None:
            _persist_partial_trace(messages, trace_path, label, turn, time.monotonic() - t0)

        if not output.message.tool_calls:
            elapsed_total = time.monotonic() - t0
            logger.info("[%s] completed: %d turns in %.1fs", label, turn, elapsed_total)
            return messages, output


# ---------------------------------------------------------------------------
# Multi-turn generation
# ---------------------------------------------------------------------------


async def multi_turn_generate(
    messages: list["ChatMessage"],
    model: Model,
    follow_ups: list[str] | None = None,
    tools: list[Tool] | None = None,
    follow_up_tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
    trace_path: Path | None = None,
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Uses a manual tool loop (via _run_tool_loop) instead of model.generate_loop()
    to provide per-turn logging, timing, and an infinite-loop safety guard.

    Args:
        messages: Initial messages (typically system + user prompt)
        model: The model to use for generation
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
        "multi_turn_generate starting (tools=%d, follow_ups=%d)",
        len(tools or []),
        len(follow_ups),
    )

    if tools:
        # Tool-enabled generation with per-turn logging
        messages, output = await _run_tool_loop(
            messages,
            model,
            tools,
            config,
            label="initial",
            trace_path=trace_path,
        )
        last_nonempty = output.completion

        # Follow-up turns
        for i, prompt in enumerate(follow_ups):
            logger.info("Follow-up %d/%d starting", i + 1, len(follow_ups))
            messages.append(ChatMessageUser(content=prompt))

            if follow_up_tools:
                messages, output = await _run_tool_loop(
                    messages,
                    model,
                    follow_up_tools,
                    config,
                    label=f"follow-up-{i + 1}",
                    trace_path=trace_path,
                )
            else:
                t_fu = time.monotonic()
                response = await model.generate(messages, config=_config)
                messages.append(ChatMessageAssistant(content=response.completion))
                output = response
                elapsed_fu = time.monotonic() - t_fu
                logger.info(
                    "Follow-up %d/%d | %s",
                    i + 1,
                    len(follow_ups),
                    _summarize_output(output, elapsed_fu),
                )
                # Persist after plain follow-up turns too
                if trace_path is not None:
                    _persist_partial_trace(
                        messages, trace_path, f"follow-up-{i + 1}", 1, time.monotonic() - t0
                    )

            if output.completion and output.completion.strip():
                last_nonempty = output.completion

        if trace_capture is not None:
            trace_capture["trace"] = _build_trace(messages, output)

        elapsed_total = time.monotonic() - t0
        logger.info("multi_turn_generate completed in %.1fs", elapsed_total)
        # No finalization needed — persist_web_result overwrites with full stage output
        return last_nonempty
    # Simple generation without tools
    t_gen = time.monotonic()
    response = await model.generate(messages, config=_config)
    messages.append(ChatMessageAssistant(content=response.completion))
    elapsed_gen = time.monotonic() - t_gen
    logger.info("single-turn | %s", _summarize_output(response, elapsed_gen))
    last_nonempty = response.completion

    for i, prompt in enumerate(follow_ups):
        logger.info("Follow-up %d/%d starting", i + 1, len(follow_ups))
        messages.append(ChatMessageUser(content=prompt))
        t_fu = time.monotonic()
        response = await model.generate(messages, config=_config)
        messages.append(ChatMessageAssistant(content=response.completion))
        elapsed_fu = time.monotonic() - t_fu
        logger.info(
            "Follow-up %d/%d | %s",
            i + 1,
            len(follow_ups),
            _summarize_output(response, elapsed_fu),
        )
        if response.completion and response.completion.strip():
            last_nonempty = response.completion

    if trace_capture is not None:
        trace_capture["trace"] = _build_trace(messages, response)

    elapsed_total = time.monotonic() - t0
    logger.info("multi_turn_generate completed in %.1fs", elapsed_total)
    # No finalization needed — persist_web_result overwrites with full stage output
    return last_nonempty


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
