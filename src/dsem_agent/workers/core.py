"""Worker extraction core logic.

Core logic for worker data extraction, decoupled from Prefect/Inspect frameworks.
Uses dependency injection for the LLM generate function.
"""

from dataclasses import dataclass

import polars as pl

from dsem_agent.utils.llm import (
    WorkerGenerateFn,
    make_worker_tools,
    parse_json_response,
)

from .prompts import WORKER_USER, WORKER_WO_PROPOSALS_SYSTEM
from .schemas import WorkerOutput


@dataclass
class WorkerExtractionResult:
    """Result of worker extraction for a single chunk."""

    output: WorkerOutput
    dataframe: pl.DataFrame
    raw_completion: str


def _format_indicators(dsem_model: dict) -> str:
    """Format indicators for the worker prompt.

    Shows: name, dtype, measurement_granularity, how_to_measure
    """
    indicators = dsem_model.get("measurement", {}).get("indicators", [])
    lines = []
    for ind in indicators:
        name = ind.get("name", "unknown")
        how_to_measure = ind.get("how_to_measure", "")
        dtype = ind.get("measurement_dtype", "")
        measurement_granularity = ind.get("measurement_granularity", "")

        # Build info string with dtype and measurement_granularity
        info_parts = [dtype]
        if measurement_granularity:
            info_parts.append(f"@{measurement_granularity}")
        info = ", ".join(info_parts)

        lines.append(f"- {name} ({info}): {how_to_measure}")
    return "\n".join(lines)


def _get_outcome_description(dsem_model: dict) -> str:
    """Get the description of the outcome variable."""
    constructs = dsem_model.get("latent", {}).get("constructs", [])
    for c in constructs:
        if c.get("is_outcome"):
            return c.get("description", c.get("name", "outcome"))
    return "Not specified"


@dataclass
class WorkerMessages:
    """Message builders for worker prompts."""

    question: str
    dsem_model: dict
    chunk: str

    def extraction_messages(self) -> list[dict]:
        """Build messages for worker extraction."""
        indicators_text = _format_indicators(self.dsem_model)
        outcome_description = _get_outcome_description(self.dsem_model)

        return [
            {"role": "system", "content": WORKER_WO_PROPOSALS_SYSTEM},
            {
                "role": "user",
                "content": WORKER_USER.format(
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
    dsem_model: dict,
    generate: WorkerGenerateFn,
) -> WorkerExtractionResult:
    """
    Run worker extraction for a single chunk.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        chunk: The data chunk to process
        question: The causal research question
        dsem_model: The DSEMModel dict with latent and measurement
        generate: Async function (messages, tools) -> completion

    Returns:
        WorkerExtractionResult with output, dataframe, and raw completion
    """
    msgs = WorkerMessages(question, dsem_model, chunk)

    # Build messages and tools
    extraction_msgs = msgs.extraction_messages()
    tools = make_worker_tools(dsem_model)

    # Generate extraction
    completion = await generate(extraction_msgs, tools)

    # Parse and validate
    data = parse_json_response(completion)
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()

    return WorkerExtractionResult(
        output=output,
        dataframe=dataframe,
        raw_completion=completion,
    )


# Re-export helper functions for backwards compatibility
__all__ = [
    "WorkerExtractionResult",
    "WorkerMessages",
    "_format_indicators",
    "_get_outcome_description",
    "run_worker_extraction",
]
