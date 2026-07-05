"""Inspect AI evaluation for worker data extraction.

Evaluates smaller LLMs on their ability to extract indicator values from
data chunks given a CausalSpec schema from the orchestrator.

Uses the same core logic as production (via run_worker_extraction), just with
a different model configuration.

Usage:
    inspect eval evals/single_model/eval2_worker_extraction.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/single_model/eval2_worker_extraction.py --model openrouter/anthropic/claude-haiku-4.5
    inspect eval evals/single_model/eval2_worker_extraction.py -T workspace_id=DEMO
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json

from evaluation.inspect_evals.common import (
    get_stage2_eval_chunks,
    load_eval_config,
    make_generate_fn,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from nof1_causal_lab.workers.core import WorkerResult, run_worker_extraction
from nof1_causal_lab.workers.prompts.extraction import SYSTEM
from nof1_causal_lab.workers.schemas import _check_dtype_match, _get_indicator_info

_CONFIG = load_eval_config()


def _get_indicator_dtypes(causal_spec: dict) -> dict[str, str]:
    """Get mapping of indicator names to their expected dtypes."""
    indicator_info = _get_indicator_info(causal_spec)
    return {name: info["dtype"] for name, info in indicator_info.items()}


def create_eval_dataset(
    n_chunks: int = 10,
    seed: int = 42,
    workspace_id: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset with chunks and the CausalSpec schema.

    Args:
        n_chunks: Number of chunks to include (each becomes a sample)
        seed: Random seed for reproducible chunk sampling
        workspace_id: Workspace to load persisted Stage 2 inputs from.

    Returns:
        MemoryDataset with one sample per chunk
    """
    stage2_inputs = get_stage2_eval_chunks(n_chunks, seed, workspace_id)
    causal_spec = stage2_inputs["causal_spec"]
    indicator_dtypes = _get_indicator_dtypes(causal_spec)
    n_indicators = len(indicator_dtypes)
    chunks = stage2_inputs["sampled_chunk_texts"]

    samples = []
    for i, chunk in enumerate(chunks):
        samples.append(
            Sample(
                input=chunk,  # Just the chunk, core logic builds the full prompt
                id=f"chunk_{i:04d}",
                metadata={
                    "chunk_index": i,
                    "chunk": chunk,
                    "workspace_id": stage2_inputs["workspace_id"],
                    "question": stage2_inputs["question"],
                    "causal_spec": causal_spec,
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


def worker_extraction_solver():
    """Solver that runs the full worker extraction flow using core logic."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            generate_fn = make_generate_fn(model)

            # Get metadata
            question_text = state.metadata.get("question", "")
            chunk = state.metadata.get("chunk", "")
            causal_spec = state.metadata.get("causal_spec", {})

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

            except Exception as e:  # noqa: BLE001
                # Store error for scorer
                state.metadata["worker_result"] = None
                state.output.completion = f"[ERROR: {e}]"

            return state

        return solve

    return _solver()


@task
def worker_eval(
    n_chunks: int = 10,
    seed: int = 42,
    workspace_id: str | None = None,
):
    """Evaluate LLM ability to extract indicator values from chunks.

    Args:
        n_chunks: Number of chunks to include in evaluation
        seed: Random seed for chunk sampling (reproducibility)
        workspace_id: Workspace to load persisted Stage 2 inputs from.
    """
    return Task(
        dataset=create_eval_dataset(
            n_chunks=n_chunks,
            seed=seed,
            workspace_id=workspace_id,
        ),
        solver=[
            system_message(SYSTEM),
            worker_extraction_solver(),
        ],
        scorer=worker_extraction_scorer(),
    )
