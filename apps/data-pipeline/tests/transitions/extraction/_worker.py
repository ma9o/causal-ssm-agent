"""Test-owned worker extraction logic for retired support-window entrypoints.

Core logic for worker data extraction, decoupled from Prefect and model-client
frameworks. Uses dependency injection for the LLM generate function.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import polars as pl

from nof1_causal_lab.workers.messages import WorkerMessages, _format_indicators
from nof1_causal_lab.workers.schemas import WorkerOutput


def scoped_log(label: str | None, message: str) -> str:
    return f"[{label}] {message}" if label else message


@dataclass
class WorkerResult:
    """Result of worker extraction for a single chunk."""

    output: WorkerOutput
    dataframe: pl.DataFrame
    raw_completion: str


def _make_validation_tool(
    name,
    description,
    param_name,
    param_description,
    validator,
    capture_key,
):
    from nof1_causal_lab.utils.openrouter_client import Tool

    capture = {}

    async def _execute(**kwargs):
        try:
            data = json.loads(kwargs[param_name])
        except json.JSONDecodeError as error:
            return f"JSON parse error: {error}"

        started_at = time.monotonic()
        _result, errors = validator(data)
        if errors:
            return "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        capture[capture_key] = data
        logging.getLogger(__name__).info(
            "[%s] validation passed (%.1fs)", name, time.monotonic() - started_at
        )
        return "VALID"

    return (
        Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {param_name: {"type": "string", "description": param_description}},
                "required": [param_name],
                "additionalProperties": False,
            },
            execute=_execute,
            stop_on_success=True,
            success_output="VALID",
        ),
        capture,
    )


async def run_worker_extraction(
    window_text: str,
    window_starts: list[str],
    question: str,
    measurement_structure: dict[str, Any],
    session_factory: Any,
    logger: logging.Logger | None = None,
    call_label: str | None = None,
) -> WorkerResult:
    """Run worker extraction for a chunk of support windows.

    Opens a single :class:`AgentSession` with the validation tool bound,
    sends the extraction prompt, and returns the tool-captured result.
    """
    active_logger = logger or logging.getLogger(__name__)
    msgs = WorkerMessages(
        question,
        measurement_structure,
        window_text,
        n_windows=len(window_starts),
    )

    extraction_msgs = msgs.extraction_messages()
    from nof1_causal_lab.workers.schemas import validate_worker_output

    tool, capture = _make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, measurement_structure, window_starts),
        capture_key="output",
    )
    tools = [tool]
    tool_names = [tool.name for tool in tools]

    # extraction_messages() returns [system, user] — split for the session API.
    system_prompt = None
    user_message = ""
    for msg in extraction_msgs:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_message = msg["content"]

    active_logger.info(
        scoped_log(
            call_label,
            "Prepared worker prompt with %d windows, %d indicators, %d text chars",
        ),
        len(window_starts),
        len(measurement_structure.get("indicators", [])),
        len(window_text),
    )
    active_logger.info(scoped_log(call_label, "Using worker tools: %s"), tool_names)
    active_logger.info(scoped_log(call_label, "Calling extraction model"))

    async with session_factory.open(
        system_prompt=system_prompt,
        tools=tools,
        log_label=call_label,
    ) as session:
        turn_result = await session.turn(user_message)
    completion = turn_result.completion
    active_logger.info(scoped_log(call_label, "Model call returned %d characters"), len(completion))

    data = capture.get("output")
    if data is None:
        raise RuntimeError(
            f"Worker extraction did not capture structured output via the "
            f"validate_extractions tool. terminal_tool={turn_result.terminal_tool_name!r}, "
            f"tool_calls_fired={turn_result.tool_calls_fired}, completion_len={len(completion)}"
        )
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()
    active_logger.info(
        scoped_log(call_label, "Validated %d extractions into %d output rows"),
        len(output.extractions),
        dataframe.height,
    )

    return WorkerResult(
        output=output,
        dataframe=dataframe,
        raw_completion=completion,
    )


__all__ = [
    "WorkerMessages",
    "WorkerResult",
    "_format_indicators",
    "run_worker_extraction",
]
