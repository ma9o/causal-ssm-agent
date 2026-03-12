"""Worker extraction core logic (tick-based).

Core logic for worker data extraction, decoupled from Prefect and model-client
frameworks. Uses dependency injection for the LLM generate function.
"""

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from causal_ssm_agent.utils.causal_spec import get_indicators, get_outcome_construct
from causal_ssm_agent.utils.llm import (
    WorkerGenerateFn,
    make_validation_tool,
    parse_json_response,
)

from .prompts.extraction import SYSTEM, USER
from .schemas import WorkerOutput


@dataclass
class WorkerResult:
    """Result of worker extraction for a single chunk."""

    output: WorkerOutput
    dataframe: pl.DataFrame
    raw_completion: str


def _pfx(label: str | None, message: str) -> str:
    """Prefix a log message with a chunk label when available."""
    return f"[{label}] {message}" if label else message


def _format_indicators(causal_spec: dict) -> str:
    """Format indicators for the worker prompt.

    Shows: name, dtype, aggregation, how_to_measure
    """
    lines = []
    for ind in get_indicators(causal_spec):
        name = ind.get("name", "unknown")
        how_to_measure = ind.get("how_to_measure", "")
        dtype = ind.get("measurement_dtype", "")
        agg = ind.get("aggregation", "mean")

        lines.append(f"- {name} ({dtype}, agg={agg}): {how_to_measure}")
    return "\n".join(lines)


def _get_outcome_description(causal_spec: dict) -> str:
    """Get the description of the outcome variable."""
    outcome = get_outcome_construct(causal_spec)
    if outcome:
        return outcome.get("description", outcome.get("name", "outcome"))
    return "Not specified"


@dataclass
class WorkerMessages:
    """Message builders for worker prompts."""

    question: str
    causal_spec: dict
    tick_text: str
    n_ticks: int

    def extraction_messages(self) -> list[dict]:
        """Build messages for worker extraction."""
        indicators_text = _format_indicators(self.causal_spec)
        outcome_description = _get_outcome_description(self.causal_spec)

        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=self.question,
                    outcome_description=outcome_description,
                    indicators=indicators_text,
                    n_ticks=self.n_ticks,
                    tick_text=self.tick_text,
                ),
            },
        ]


async def run_worker_extraction(
    tick_text: str,
    tick_ids: list[str],
    question: str,
    causal_spec: dict,
    generate: WorkerGenerateFn,
    logger: Any | None = None,
    call_label: str | None = None,
) -> WorkerResult:
    """
    Run worker extraction for a chunk of ticks.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        tick_text: Pre-formatted text of tick events for the LLM prompt.
        tick_ids: Expected tick IDs in this chunk (for validation).
        question: The causal research question.
        causal_spec: The CausalSpec dict with latent and measurement.
        generate: Async function (messages, tools) -> completion.
        logger: Optional logger instance.
        call_label: Optional label for log messages.

    Returns:
        WorkerResult with output, dataframe, and raw completion.
    """
    active_logger = logger or logging.getLogger(__name__)
    msgs = WorkerMessages(question, causal_spec, tick_text, n_ticks=len(tick_ids))

    # Build messages and tools
    extraction_msgs = msgs.extraction_messages()
    from causal_ssm_agent.workers.schemas import validate_worker_output

    tool, capture = make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, causal_spec, tick_ids),
        capture_key="output",
    )
    tools = [tool]
    tool_names = [tool.name for tool in tools]

    active_logger.info(
        _pfx(
            call_label,
            "Prepared worker prompt with %d ticks, %d indicators, %d text chars",
        ),
        len(tick_ids),
        len(get_indicators(causal_spec)),
        len(tick_text),
    )
    active_logger.info(_pfx(call_label, "Using worker tools: %s"), tool_names)

    # Generate extraction
    active_logger.info(_pfx(call_label, "Calling extraction model"))
    completion = await generate(extraction_msgs, tools=tools, label=call_label)
    active_logger.info(_pfx(call_label, "Model call returned %d characters"), len(completion))

    # Prefer the captured result from the validation tool
    data = capture.get("output")
    if data is None:
        # Fallback: try parsing the final completion directly
        active_logger.warning(
            _pfx(
                call_label,
                "Validation tool did not capture structured output; falling back to completion parsing",
            ),
        )
        data = parse_json_response(completion)
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()
    active_logger.info(
        _pfx(call_label, "Validated %d extractions into %d output rows"),
        len(output.extractions),
        dataframe.height,
    )

    return WorkerResult(
        output=output,
        dataframe=dataframe,
        raw_completion=completion,
    )


__all__ = [
    "WorkerResult",
    "WorkerMessages",
    "run_worker_extraction",
]
