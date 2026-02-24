"""Worker extraction core logic.

Core logic for worker data extraction, decoupled from Prefect/Inspect frameworks.
Uses dependency injection for the LLM generate function.
"""

from dataclasses import dataclass

import polars as pl

from causal_ssm_agent.utils.causal_spec import get_indicators, get_outcome_construct
from causal_ssm_agent.utils.llm import (
    WorkerGenerateFn,
    make_worker_tools,
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


def _format_indicators(causal_spec: dict) -> str:
    """Format indicators for the worker prompt.

    Shows: name, dtype, how_to_measure
    """
    lines = []
    for ind in get_indicators(causal_spec):
        name = ind.get("name", "unknown")
        how_to_measure = ind.get("how_to_measure", "")
        dtype = ind.get("measurement_dtype", "")

        lines.append(f"- {name} ({dtype}): {how_to_measure}")
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
    chunk: str

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
                    chunk=self.chunk,
                ),
            },
        ]


async def run_worker_extraction(
    chunk: str,
    question: str,
    causal_spec: dict,
    generate: WorkerGenerateFn,
) -> WorkerResult:
    """
    Run worker extraction for a single chunk.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        causal_spec: The CausalSpec dict with latent and measurement
        generate: Async function (messages, tools) -> completion

    Returns:
        WorkerResult with output, dataframe, and raw completion
    """
    msgs = WorkerMessages(question, causal_spec, chunk)

    # Build messages and tools
    # The validation tool captures the last valid output so we don't depend
    # on the final completion being valid JSON.
    extraction_msgs = msgs.extraction_messages()
    tools, capture = make_worker_tools(causal_spec)

    # Generate extraction
    completion = await generate(extraction_msgs, tools)

    # Prefer the captured result from the validation tool
    data = capture.get("output")
    if data is None:
        # Fallback: try parsing the final completion directly
        data = parse_json_response(completion)
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()

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
