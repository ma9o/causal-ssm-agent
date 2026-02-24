"""Inspect AI evaluation for worker data extraction.

Evaluates smaller LLMs on their ability to extract indicator values from
data chunks given a CausalSpec schema from the orchestrator.

Uses the same core logic as production (via run_worker_extraction), just with
a different model configuration.

Usage:
    inspect eval evals/single_model/eval2_worker_extraction.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/single_model/eval2_worker_extraction.py --model openrouter/anthropic/claude-haiku-4.5
    inspect eval evals/single_model/eval2_worker_extraction.py -T question=4
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

from evals.common import (
    get_questions_with_causal_spec,
    get_sample_chunks_worker,
    load_eval_config,
    select_question,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from causal_ssm_agent.utils.llm import make_worker_generate_fn
from causal_ssm_agent.workers.core import WorkerResult, run_worker_extraction
from causal_ssm_agent.workers.prompts.extraction import SYSTEM
from causal_ssm_agent.workers.schemas import _check_dtype_match, _get_indicator_info

_CONFIG = load_eval_config()
MODELS = {m["id"]: m["alias"] for m in _CONFIG["worker_models"]}


def _get_indicator_dtypes(causal_spec: dict) -> dict[str, str]:
    """Get mapping of indicator names to their expected dtypes."""
    indicator_info = _get_indicator_info(causal_spec)
    return {name: info["dtype"] for name, info in indicator_info.items()}


def create_eval_dataset(
    question: str | None = None,
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset with chunks and the CausalSpec schema.

    Args:
        question: Question selector (prefix ID or full slug). Defaults to first available.
        n_chunks: Number of chunks to include (each becomes a sample)
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with one sample per chunk
    """
    available = get_questions_with_causal_spec()
    if question:
        q = select_question(available, question)
    else:
        q = available[0]

    # Load the CausalSpec schema
    causal_spec = q.load_causal_spec()
    indicator_dtypes = _get_indicator_dtypes(causal_spec)
    n_indicators = len(indicator_dtypes)

    # Get chunks (using worker chunk size from config)
    chunks = get_sample_chunks_worker(n_chunks, seed, input_file)

    samples = []
    for i, chunk in enumerate(chunks):
        samples.append(
            Sample(
                input=chunk,  # Just the chunk, core logic builds the full prompt
                id=f"chunk_{i:04d}",
                metadata={
                    "chunk_index": i,
                    "chunk": chunk,
                    "question_slug": q.slug,
                    "question": q.question,
                    "n_indicators": n_indicators,
                    "indicator_dtypes": indicator_dtypes,
                },
            )
        )

    return MemoryDataset(samples)


# Base points for valid schema (even with no extractions)
VALID_SCHEMA_POINTS = 10


def _score_worker_result(
    result: WorkerResult,
    indicator_dtypes: dict[str, str],
) -> dict:
    """Score a worker extraction result.

    Returns:
        - 0 if output is invalid (dtype validation error)
        - 10 + number of valid extraction rows otherwise
    """
    output = result.output
    df = result.dataframe

    n_rows = len(df)

    # Validate dtypes
    dtype_errors = []
    for extraction in output.extractions:
        ind_name = extraction.indicator
        expected_dtype = indicator_dtypes.get(ind_name)

        if expected_dtype is not None and not _check_dtype_match(extraction.value, expected_dtype):
            dtype_errors.append(
                f"{ind_name}: got {type(extraction.value).__name__}={extraction.value}, expected {expected_dtype}"
            )

    n_dtype_errors = len(dtype_errors)

    if n_dtype_errors > 0:
        error_summary = "; ".join(dtype_errors[:5])
        if n_dtype_errors > 5:
            error_summary += f"... and {n_dtype_errors - 5} more"
        return {
            "total": 0,
            "error": True,
            "explanation": f"Dtype validation failed ({n_dtype_errors} errors): {error_summary}",
            "n_extractions": n_rows,
            "n_dtype_errors": n_dtype_errors,
        }

    # Build explanation
    unique_inds = df["indicator"].n_unique() if n_rows > 0 else 0
    total_score = VALID_SCHEMA_POINTS + n_rows

    explanation = (
        f"Valid schema (+{VALID_SCHEMA_POINTS}). "
        f"Extracted {n_rows} observations across {unique_inds} indicators."
    )

    return {
        "total": total_score,
        "error": False,
        "explanation": explanation,
        "n_extractions": n_rows,
        "n_dtype_errors": 0,
        "n_unique_indicators": unique_inds,
    }


@scorer(metrics=[mean(), stderr()])
def worker_extraction_scorer():
    """Score worker extractions.

    Returns:
        - 0 if output is invalid (JSON parse error, schema validation error, dtype error)
        - 10 + number of valid extraction rows (dtype-checked)
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        # Get the WorkerResult from metadata (set by solver)
        result: WorkerResult | None = state.metadata.get("worker_result")
        indicator_dtypes = state.metadata.get("indicator_dtypes", {})

        if result is None:
            return Score(
                value=0,
                answer="[No result]",
                explanation="Worker extraction did not produce a result",
            )

        scoring = _score_worker_result(result, indicator_dtypes)

        if scoring.get("error"):
            return Score(
                value=0,
                answer=state.output.completion[:500],
                explanation=f"ERROR: {scoring['explanation']}",
                metadata={
                    "n_extractions": scoring.get("n_extractions", 0),
                    "n_dtype_errors": scoring.get("n_dtype_errors", 0),
                },
            )

        return Score(
            value=scoring["total"],
            answer=state.output.completion[:500],
            explanation=scoring["explanation"],
            metadata={
                "n_extractions": scoring["n_extractions"],
                "n_dtype_errors": 0,
                "n_unique_indicators": scoring.get("n_unique_indicators", 0),
            },
        )

    return score


def worker_extraction_solver(question: str | None = None):
    """Solver that runs the full worker extraction flow using core logic."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            generate_fn = make_worker_generate_fn(model)

            # Get metadata
            question_text = state.metadata.get("question", "")
            chunk = state.metadata.get("chunk", "")
            question_slug = state.metadata.get("question_slug", question or "")

            available = get_questions_with_causal_spec()
            q = select_question(available, question_slug)
            causal_spec = q.load_causal_spec()

            # Run the SAME core logic as production
            try:
                result = await run_worker_extraction(
                    chunk=chunk,
                    question=question_text,
                    causal_spec=causal_spec,
                    generate=generate_fn,
                )

                # Store result in metadata for scorer
                state.metadata["worker_result"] = result
                state.output.completion = json.dumps(result.output.model_dump(), indent=2)

            except Exception as e:
                # Store error for scorer
                state.metadata["worker_result"] = None
                state.output.completion = f"[ERROR: {e}]"

            return state

        return solve

    return _solver()


@task
def worker_eval(
    question: str | None = None,
    n_chunks: int = 10,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to extract indicator values from chunks.

    Args:
        question: Question selector (prefix ID or full slug). Defaults to first with causal_spec.
        n_chunks: Number of chunks to include in evaluation
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    return Task(
        dataset=create_eval_dataset(
            question=question,
            n_chunks=n_chunks,
            seed=seed,
            input_file=input_file,
        ),
        solver=[
            system_message(SYSTEM),
            worker_extraction_solver(question=question),
        ],
        scorer=worker_extraction_scorer(),
    )
