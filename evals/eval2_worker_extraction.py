"""Inspect AI evaluation for worker data extraction.

Evaluates smaller LLMs on their ability to extract indicator values from
data chunks given a DSEMModel schema from the orchestrator.

Usage:
    inspect eval evals/eval2_worker_extraction.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/eval2_worker_extraction.py --model openrouter/anthropic/claude-haiku-4.5
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

from causal_agent.workers.prompts import WORKER_WO_PROPOSALS_SYSTEM, WORKER_USER
from causal_agent.workers.schemas import WorkerOutput, _get_indicator_info
from causal_agent.workers.agents import _format_indicators, _get_outcome_description
from causal_agent.utils.llm import make_worker_tools

from evals.common import (
    extract_json_from_response,
    get_sample_chunks_worker,
    load_dsem_model_by_question_id,
    tool_assisted_generate,
)

# Worker models for parallel execution
# Using reasoning-capable models with thinking budget
# Note: Gemini 3 uses Vertex AI directly (not OpenRouter) for proper thought signature support
MODELS = {
    "openrouter/moonshotai/kimi-k2-thinking": "kimi",
    "openrouter/deepseek/deepseek-v3.2-exp": "deepseek",
    "google/vertex/gemini-3-flash-preview": "gemini",  # Vertex AI - requires GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
    "openrouter/x-ai/grok-4.1-fast": "grok",
    "openrouter/anthropic/claude-haiku-4.5": "haiku",
    "openrouter/minimax/minimax-m2": "minimax",
    "openrouter/openai/gpt-oss-120b": "gpt-oss",
}

# Default question ID for worker eval (uses question 4: "I want to sleep better")
DEFAULT_QUESTION_ID = 4


def _get_indicator_dtypes(dsem_model: dict) -> dict[str, str]:
    """Get mapping of indicator names to their expected dtypes."""
    indicator_info = _get_indicator_info(dsem_model)
    return {name: info["dtype"] for name, info in indicator_info.items()}


def create_eval_dataset(
    question_id: int = DEFAULT_QUESTION_ID,
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset with chunks and the DSEMModel schema.

    Args:
        question_id: The question ID (1-5) to load DSEMModel for
        n_chunks: Number of chunks to include (each becomes a sample)
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with one sample per chunk
    """
    # Load the DSEMModel schema
    dsem_model = load_dsem_model_by_question_id(question_id)
    indicators_text = _format_indicators(dsem_model)
    outcome_description = _get_outcome_description(dsem_model)
    indicator_dtypes = _get_indicator_dtypes(dsem_model)

    # Count indicators
    n_indicators = len(indicator_dtypes)

    # Get the question text from config
    from evals.common import get_eval_questions
    questions = get_eval_questions()
    question = next((q["question"] for q in questions if q["id"] == question_id), "")

    # Get chunks (using worker chunk size from config)
    chunks = get_sample_chunks_worker(n_chunks, seed, input_file)

    samples = []
    for i, chunk in enumerate(chunks):
        user_prompt = WORKER_USER.format(
            question=question,
            outcome_description=outcome_description,
            indicators=indicators_text,
            chunk=chunk,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"chunk_{i:04d}",
                metadata={
                    "chunk_index": i,
                    "question_id": question_id,
                    "question": question,
                    "n_indicators": n_indicators,
                    "indicator_dtypes": indicator_dtypes,
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
        indicator_dtypes = state.metadata.get("indicator_dtypes", {})

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
            ind_name = extraction.indicator
            expected_dtype = indicator_dtypes.get(ind_name)

            if expected_dtype is not None and not _validate_dtype(extraction.value, expected_dtype):
                dtype_errors.append(
                    f"{ind_name}: got {type(extraction.value).__name__}={extraction.value}, expected {expected_dtype}"
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
        n_proposed = len(output.proposed_indicators) if output.proposed_indicators else 0
        unique_inds = df["indicator"].n_unique() if n_rows > 0 else 0
        total_score = VALID_SCHEMA_POINTS + n_rows

        explanation = (
            f"Valid schema (+{VALID_SCHEMA_POINTS}). "
            f"Extracted {n_rows} observations across {unique_inds} indicators. "
            f"Proposed {n_proposed} new indicator(s)."
        )

        return Score(
            value=total_score,
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=explanation,
            metadata={
                "n_extractions": n_rows,
                "n_dtype_errors": 0,
                "n_unique_indicators": unique_inds,
                "n_proposed_indicators": n_proposed,
            },
        )

    return score


@task
def worker_eval(
    question_id: int = DEFAULT_QUESTION_ID,
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to extract indicator values from chunks.

    Args:
        question_id: The question ID (1-5) to load DSEMModel for
        n_chunks: Number of chunks to include in evaluation
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    # Load DSEMModel for tools (same schema for all samples)
    dsem_model = load_dsem_model_by_question_id(question_id)

    return Task(
        dataset=create_eval_dataset(
            question_id=question_id,
            n_chunks=n_chunks,
            seed=seed,
            input_file=input_file,
        ),
        solver=[
            system_message(WORKER_WO_PROPOSALS_SYSTEM),
            tool_assisted_generate(tools=make_worker_tools(dsem_model)),
        ],
        scorer=worker_extraction_scorer(),
    )
