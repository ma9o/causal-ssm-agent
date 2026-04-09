"""Worker extraction core logic for support-window extraction.

Core logic for worker data extraction, decoupled from Prefect and model-client
frameworks. Uses dependency injection for the LLM generate function.
"""

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from causal_ssm_agent.utils.causal_spec import (
    get_indicators,
    get_outcome_construct,
    get_summary_operator,
    get_support_kind,
)
from causal_ssm_agent.utils.llm import (
    GenerateFn,
    make_validation_tool,
    parse_json_response,
    scoped_log,
)

from .prompts.extraction import SYSTEM, USER
from .schemas import WorkerOutput


@dataclass
class WorkerResult:
    """Result of worker extraction for a single chunk."""

    output: WorkerOutput
    dataframe: pl.DataFrame
    raw_completion: str


def _format_indicators(causal_spec: dict) -> str:
    """Format indicators for the worker prompt.

    Shows: name, dtype, operator, support kind, window, how_to_measure.
    Omits anchor_policy (internal SSM plumbing the worker doesn't need).
    """
    lines = []
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "")
    for ind in get_indicators(causal_spec):
        name = ind.get("name", "unknown")
        how_to_measure = ind.get("how_to_measure", "")
        dtype = ind.get("measurement_dtype", "")
        support_kind = ind.get("support_kind") or get_support_kind(ind)
        summary_operator = ind.get("summary_operator") or get_summary_operator(ind)
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
    window_text: str
    n_windows: int

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
                    n_windows=self.n_windows,
                    window_text=self.window_text,
                ),
            },
        ]


async def run_worker_extraction(
    window_text: str,
    window_starts: list[str],
    question: str,
    causal_spec: dict,
    generate: GenerateFn,
    logger: Any | None = None,
    call_label: str | None = None,
) -> WorkerResult:
    """
    Run worker extraction for a chunk of support windows.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        window_text: Pre-formatted text of support-window events for the LLM prompt.
        window_starts: Expected support-window starts in this chunk (for validation).
        question: The causal research question.
        causal_spec: The CausalSpec dict with latent and measurement.
        generate: Async function (messages, tools) -> completion.
        logger: Optional logger instance.
        call_label: Optional label for log messages.

    Returns:
        WorkerResult with output, dataframe, and raw completion.
    """
    active_logger = logger or logging.getLogger(__name__)
    msgs = WorkerMessages(question, causal_spec, window_text, n_windows=len(window_starts))

    # Build messages and tools
    extraction_msgs = msgs.extraction_messages()
    from causal_ssm_agent.workers.schemas import validate_worker_output

    tool, capture = make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, causal_spec, window_starts),
        capture_key="output",
    )
    tools = [tool]
    tool_names = [tool.name for tool in tools]

    active_logger.info(
        scoped_log(
            call_label,
            "Prepared worker prompt with %d windows, %d indicators, %d text chars",
        ),
        len(window_starts),
        len(get_indicators(causal_spec)),
        len(window_text),
    )
    active_logger.info(scoped_log(call_label, "Using worker tools: %s"), tool_names)

    # Generate extraction
    active_logger.info(scoped_log(call_label, "Calling extraction model"))
    completion = await generate(extraction_msgs, tools=tools, label=call_label)
    active_logger.info(scoped_log(call_label, "Model call returned %d characters"), len(completion))

    # Prefer the captured result from the validation tool
    data = capture.get("output")
    if data is None:
        # Fallback: try parsing the final completion directly
        active_logger.warning(
            scoped_log(
                call_label,
                "Validation tool did not capture structured output; falling back to completion parsing",
            ),
        )
        data = parse_json_response(completion)
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
