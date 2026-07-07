"""Worker extraction core logic for support-window extraction.

Core logic for worker data extraction, decoupled from Prefect and model-client
frameworks. Uses dependency injection for the LLM generate function.
"""

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from nof1_causal_lab.utils.agent_session import StageSessionFactory
from nof1_causal_lab.utils.llm import (
    make_validation_tool,
    scoped_log,
)
from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

from .prompts.extraction import SYSTEM, USER
from .schemas import WorkerOutput


@dataclass
class WorkerResult:
    """Result of worker extraction for a single chunk."""

    output: WorkerOutput
    dataframe: pl.DataFrame
    raw_completion: str


def _format_indicators(measurement_structure: dict) -> str:
    """Format indicators for the worker prompt.

    Shows: name, dtype, operator, support kind, window, how_to_measure.
    Omits anchor_policy (internal SSM plumbing the worker doesn't need).
    """
    lines = []
    model_clock = measurement_structure.get("model_clock", "")
    for ind in measurement_structure.get("indicators", []):
        name = ind.get("name", "unknown")
        how_to_measure = ind.get("how_to_measure", "")
        dtype = ind.get("measurement_dtype", "")
        sem = get_observation_semantics(ind)
        support_kind = ind.get("support_kind") or sem.support_kind.value
        summary_operator = ind.get("summary_operator") or sem.summary_operator.value
        window = ind.get("observation_window") or model_clock
        ordinal_levels = ind.get("ordinal_levels") or []

        details = [dtype, f"operator={summary_operator}", f"support={support_kind}"]
        if window:
            details.append(f"window={window}")
        if dtype == "ordinal" and ordinal_levels:
            codebook = ", ".join(f"{idx}={level}" for idx, level in enumerate(ordinal_levels))
            details.append(f"ordinal_codes={codebook}")

        lines.append(f"- {name} ({', '.join(details)}): {how_to_measure}")
    return "\n".join(lines)


@dataclass
class WorkerMessages:
    """Message builders for worker prompts."""

    question: str
    measurement_structure: dict
    window_text: str
    n_windows: int

    def extraction_messages(self) -> list[dict]:
        """Build messages for worker extraction."""
        indicators_text = _format_indicators(self.measurement_structure)

        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=self.question,
                    indicators=indicators_text,
                    n_windows=self.n_windows,
                    window_text=self.window_text,
                ),
            },
        ]


async def run_worker_extraction(
    window_text: str,
    window_starts: list[str],
    question: str,
    measurement_structure: dict,
    session_factory: StageSessionFactory,
    logger: Any | None = None,
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

    tool, capture = make_validation_tool(
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
    "run_worker_extraction",
]
