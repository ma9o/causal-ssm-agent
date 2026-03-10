"""Worker extraction core logic.

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


def _format_dataframe_schema(df: pl.DataFrame) -> str:
    """Format a DataFrame schema for the worker prompt."""
    lines = ["| Column | Type |", "|--------|------|"]
    for col in df.columns:
        lines.append(f"| {col} | {df.schema[col]} |")
    return "\n".join(lines)


def _format_dataframe_chunk(df: pl.DataFrame) -> str:
    """Format a DataFrame chunk as CSV for the worker prompt.

    Uses CSV rather than str(df) because Polars' default __str__
    truncates both rows and columns (replacing them with '…'),
    which hides most of the data from the LLM.
    """
    return df.write_csv()


@dataclass
class WorkerMessages:
    """Message builders for worker prompts."""

    question: str
    causal_spec: dict
    chunk_df: pl.DataFrame

    def extraction_messages(self) -> list[dict]:
        """Build messages for worker extraction."""
        indicators_text = _format_indicators(self.causal_spec)
        outcome_description = _get_outcome_description(self.causal_spec)
        schema_text = _format_dataframe_schema(self.chunk_df)
        chunk_text = _format_dataframe_chunk(self.chunk_df)

        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=self.question,
                    outcome_description=outcome_description,
                    indicators=indicators_text,
                    schema=schema_text,
                    n_rows=len(self.chunk_df),
                    chunk=chunk_text,
                ),
            },
        ]


async def run_worker_extraction(
    chunk_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
    generate: WorkerGenerateFn,
    logger: Any | None = None,
) -> WorkerResult:
    """
    Run worker extraction for a single DataFrame chunk.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        chunk_df: A slice of the raw DataFrame to process
        question: The causal research question
        causal_spec: The CausalSpec dict with latent and measurement
        generate: Async function (messages, tools) -> completion

    Returns:
        WorkerResult with output, dataframe, and raw completion
    """
    active_logger = logger or logging.getLogger(__name__)
    msgs = WorkerMessages(question, causal_spec, chunk_df)

    # Build messages and tools
    # The validation tool captures the last valid output so we don't depend
    # on the final completion being valid JSON.
    extraction_msgs = msgs.extraction_messages()
    tools, capture = make_worker_tools(causal_spec)
    chunk_csv = _format_dataframe_chunk(chunk_df)

    active_logger.info(
        "Prepared worker prompt with %d rows, %d columns, %d indicators, %d CSV chars",
        len(chunk_df),
        len(chunk_df.columns),
        len(get_indicators(causal_spec)),
        len(chunk_csv),
    )

    # Generate extraction
    active_logger.info("Calling extraction model")
    completion = await generate(extraction_msgs, tools)
    active_logger.info("Model call returned %d characters", len(completion))

    # Prefer the captured result from the validation tool
    data = capture.get("output")
    if data is None:
        # Fallback: try parsing the final completion directly
        active_logger.warning(
            "Validation tool did not capture structured output; falling back to completion parsing",
        )
        data = parse_json_response(completion)
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()
    active_logger.info(
        "Validated %d extractions into %d output rows",
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
