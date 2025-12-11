"""Inspect AI evaluation for worker data extraction.

Evaluates smaller LLMs on their ability to extract dimension values from
data chunks given a schema from the orchestrator.

Usage:
    inspect eval evals/worker_extraction.py --model openrouter/google/gemini-2.0-flash-001
    inspect eval evals/worker_extraction.py --model openrouter/anthropic/claude-sonnet-4
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, system_message

from causal_agent.workers.prompts import WORKER_SYSTEM, WORKER_USER
from causal_agent.workers.schemas import WorkerOutput
from causal_agent.workers.agents import (
    _format_dimensions,
    _get_observed_dimension_dtypes,
    _get_outcome_description,
)

from evals.common import (
    extract_json_from_response,
    get_sample_chunks_worker,
    load_example_dag,
    reasoning_generate,
)

# Worker models for parallel execution (via OpenRouter)
# Using reasoning-capable models with thinking budget
MODELS = {
    "openrouter/moonshotai/kimi-k2-thinking": "kimi-k2",
    "openrouter/deepseek/deepseek-v3.2-exp": "deepseek",
    "openrouter/google/gemini-2.5-flash": "gemini-flash",
    "openrouter/x-ai/grok-4.1-fast": "grok-fast",
    "openrouter/anthropic/claude-4.5-haiku-20251001": "haiku",
}

# Default question for worker eval (matches example_dag.json)
DEFAULT_QUESTION = "I want to feel like a competent engineer"


def create_eval_dataset(
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
    question: str = DEFAULT_QUESTION,
) -> MemoryDataset:
    """Create evaluation dataset with chunks and the example DAG schema.

    Args:
        n_chunks: Number of chunks to include (each becomes a sample)
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest
        question: The causal question to use

    Returns:
        MemoryDataset with one sample per chunk
    """
    # Load the example DAG schema
    schema = load_example_dag()
    dimensions_text = _format_dimensions(schema)
    outcome_description = _get_outcome_description(schema)
    dimension_dtypes = _get_observed_dimension_dtypes(schema)

    # Count observed dimensions (what workers actually see)
    n_observed = len(dimension_dtypes)

    # Get chunks (using worker chunk size from config)
    chunks = get_sample_chunks_worker(n_chunks, seed, input_file)

    samples = []
    for i, chunk in enumerate(chunks):
        user_prompt = WORKER_USER.format(
            question=question,
            outcome_description=outcome_description,
            dimensions=dimensions_text,
            chunk=chunk,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"chunk_{i:04d}",
                metadata={
                    "chunk_index": i,
                    "question": question,
                    "n_observed_dimensions": n_observed,
                    "dimension_dtypes": dimension_dtypes,
                },
            )
        )

    return MemoryDataset(samples)


# Base points for valid schema (even with no extractions)
VALID_SCHEMA_POINTS = 10


def _validate_dtype(value, expected_dtype: str) -> bool:
    """Check if a value matches the expected dtype.

    Args:
        value: The extracted value
        expected_dtype: One of 'continuous', 'binary', 'count', 'ordinal', 'categorical'

    Returns:
        True if value matches expected dtype
    """
    if value is None:
        return True  # None is always valid (missing data)

    if expected_dtype == "continuous":
        return isinstance(value, (int, float))
    elif expected_dtype == "binary":
        return isinstance(value, bool) or value in (0, 1, "0", "1", "true", "false")
    elif expected_dtype == "count":
        return isinstance(value, int) and value >= 0
    elif expected_dtype == "ordinal":
        return isinstance(value, (int, float, str))
    elif expected_dtype == "categorical":
        return isinstance(value, str)
    else:
        return True  # Unknown dtype, accept anything


@scorer(metrics=[mean(), stderr()])
def worker_extraction_scorer():
    """Score worker extractions.

    Returns:
        - 0 if output is invalid (JSON parse error, schema validation error)
        - 10 + number of valid extraction rows (dtype-checked)
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        dimension_dtypes = state.metadata.get("dimension_dtypes", {})

        # Extract JSON from response
        json_str = extract_json_from_response(completion)
        if json_str is None:
            return Score(
                value=0,
                answer="[No valid JSON found]",
                explanation=(
                    "ERROR: Could not extract JSON from model response.\n"
                    f"Response preview: {completion[:500]}..."
                ),
            )

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return Score(
                value=0,
                answer=json_str[:200] + "..." if len(json_str) > 200 else json_str,
                explanation=f"ERROR: JSON parse failed - {e}",
            )

        # Validate against schema
        try:
            output = WorkerOutput.model_validate(data)
        except Exception as e:
            return Score(
                value=0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Convert to dataframe and count rows
        try:
            df = output.to_dataframe()
            n_rows = len(df)
        except Exception as e:
            return Score(
                value=0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: DataFrame conversion failed - {e}",
            )

        # Validate dtypes - any error means score 0
        dtype_errors = []

        for extraction in output.extractions:
            dim_name = extraction.dimension
            expected_dtype = dimension_dtypes.get(dim_name)

            if expected_dtype is not None and not _validate_dtype(extraction.value, expected_dtype):
                dtype_errors.append(
                    f"{dim_name}: got {type(extraction.value).__name__}={extraction.value}, expected {expected_dtype}"
                )

        n_dtype_errors = len(dtype_errors)

        if n_dtype_errors > 0:
            error_summary = "; ".join(dtype_errors[:5])
            if n_dtype_errors > 5:
                error_summary += f"... and {n_dtype_errors - 5} more"
            return Score(
                value=0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: Dtype validation failed ({n_dtype_errors} errors): {error_summary}",
                metadata={
                    "n_extractions": n_rows,
                    "n_dtype_errors": n_dtype_errors,
                },
            )

        # Build explanation
        n_proposed = len(output.proposed_dimensions) if output.proposed_dimensions else 0
        unique_dims = df["dimension"].n_unique() if n_rows > 0 else 0
        total_score = VALID_SCHEMA_POINTS + n_rows

        explanation = (
            f"Valid schema (+{VALID_SCHEMA_POINTS}). "
            f"Extracted {n_rows} observations across {unique_dims} dimensions. "
            f"Proposed {n_proposed} new dimension(s)."
        )

        return Score(
            value=total_score,
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=explanation,
            metadata={
                "n_extractions": n_rows,
                "n_dtype_errors": 0,
                "n_unique_dimensions": unique_dims,
                "n_proposed_dimensions": n_proposed,
            },
        )

    return score


@task
def worker_eval(
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
    question: str = DEFAULT_QUESTION,
):
    """Evaluate LLM ability to extract dimension values from chunks.

    Args:
        n_chunks: Number of chunks to include in evaluation
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
        question: The causal question to use
    """
    return Task(
        dataset=create_eval_dataset(
            n_chunks=n_chunks,
            seed=seed,
            input_file=input_file,
            question=question,
        ),
        solver=[
            system_message(WORKER_SYSTEM),
            reasoning_generate(),
        ],
        scorer=worker_extraction_scorer(),
    )
