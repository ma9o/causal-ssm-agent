"""Inspect AI evaluation for aggregation function robustness.

Tests the aggregate_worker_measurements function on N different sets of M chunks
processed by workers. Scores 1 if aggregation completes successfully, 0 if it
raises an exception.

This eval verifies that the aggregation pipeline can handle the diverse outputs
produced by worker LLMs without breaking. Sets cycle through all questions that
have a causal_spec, testing each CausalSpec with its corresponding question.

Uses the same core logic as production (via run_worker_extraction) for generating
worker outputs, just with different model configurations.

Usage:
    inspect eval evals/eval4_aggregation_robustness.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/eval4_aggregation_robustness.py -T n_sets=10 -T chunks_per_set=5
    inspect eval evals/eval4_aggregation_robustness.py -T questions=1,4
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import traceback

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_ssm_agent.utils.aggregations import aggregate_worker_measurements
from causal_ssm_agent.utils.llm import make_worker_generate_fn
from causal_ssm_agent.workers.core import (
    _format_indicators,
    _get_outcome_description,
    run_worker_extraction,
)
from causal_ssm_agent.workers.prompts.extraction import USER
from evals.common import (
    get_questions_with_causal_spec,
    get_sample_chunks_worker,
    select_question,
    select_questions,
)


def create_eval_dataset(
    n_sets: int = 10,
    chunks_per_set: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    questions: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset with chunk sets for aggregation testing.

    Each sample is a set of chunks that will be processed by workers and then
    aggregated together. The eval tests whether aggregation succeeds or fails.

    Sets cycle through all available questions with causal_spec, ensuring each
    CausalSpec is tested with its corresponding question.

    Args:
        n_sets: Number of chunk sets (each becomes one aggregation test)
        chunks_per_set: Number of chunks per set (M workers processing M chunks)
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest
        questions: Optional comma-separated question selectors to filter

    Returns:
        MemoryDataset with one sample per chunk set
    """
    available = get_questions_with_causal_spec()
    if questions:
        available = select_questions(available, questions)
    n_questions = len(available)

    # Get all chunks needed
    total_chunks = n_sets * chunks_per_set
    all_chunks = get_sample_chunks_worker(total_chunks, seed, input_file)

    samples = []
    for set_idx in range(n_sets):
        # Cycle through questions
        q = available[set_idx % n_questions]
        causal_spec = q.load_causal_spec()

        indicators_text = _format_indicators(causal_spec)
        outcome_description = _get_outcome_description(causal_spec)

        # Get chunks for this set
        start_idx = set_idx * chunks_per_set
        end_idx = start_idx + chunks_per_set
        chunk_set = all_chunks[start_idx:end_idx]

        # Store metadata for each chunk's prompt
        chunk_prompts = []
        for chunk in chunk_set:
            user_prompt = USER.format(
                question=q.question,
                outcome_description=outcome_description,
                indicators=indicators_text,
                chunk=chunk,
            )
            chunk_prompts.append(user_prompt)

        samples.append(
            Sample(
                input=f"Aggregation test set {set_idx + 1} (Q{q.id_prefix}: {q.question[:30]}...) with {len(chunk_set)} chunks",
                id=f"set_{set_idx:04d}_q_{q.slug}",
                metadata={
                    "set_index": set_idx,
                    "question_slug": q.slug,
                    "chunk_prompts": chunk_prompts,
                    "chunks": chunk_set,
                    "question": q.question,
                },
            )
        )

    return MemoryDataset(samples)


async def generate_worker_output(
    model_id: str,
    chunk: str,
    question: str,
    causal_spec: dict,
):
    """Generate worker output for a single chunk using core logic.

    Returns the dataframe from the extraction result.
    """
    model = get_model(model_id)
    generate = make_worker_generate_fn(model)

    result = await run_worker_extraction(
        chunk=chunk,
        question=question,
        causal_spec=causal_spec,
        generate=generate,
    )

    return result.dataframe


def aggregation_solver(worker_timeout: float = 300):
    """Solver that generates worker outputs for all chunks and runs aggregation.

    Args:
        worker_timeout: Timeout in seconds for each worker.
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            question_slug = state.metadata["question_slug"]
            available = get_questions_with_causal_spec()
            q = select_question(available, question_slug)
            causal_spec = q.load_causal_spec()

            question_text = state.metadata["question"]
            chunks = state.metadata["chunks"]

            # Get the model being evaluated (passed via --model flag)
            model = get_model()
            model_id = str(model)

            # Generate outputs from workers in parallel
            async def safe_generate(chunk: str, chunk_idx: int) -> tuple[int, any, str | None]:
                """Generate with error handling, returns (chunk_idx, result, error)."""
                try:
                    result = await asyncio.wait_for(
                        generate_worker_output(model_id, chunk, question_text, causal_spec),
                        timeout=worker_timeout,
                    )
                    return chunk_idx, result, None
                except TimeoutError:
                    return chunk_idx, None, f"TIMEOUT after {worker_timeout}s"
                except Exception as e:
                    return chunk_idx, None, str(e)

            results = await asyncio.gather(*[
                safe_generate(chunk, i) for i, chunk in enumerate(chunks)
            ])

            # Collect worker outputs
            worker_outputs = []
            worker_errors = []
            for chunk_idx, result, error in results:
                if error:
                    worker_errors.append(f"chunk_{chunk_idx}: {error}")
                elif result is not None:
                    worker_outputs.append(result)

            # Store worker info in metadata
            state.metadata["n_successful_workers"] = len(worker_outputs)
            state.metadata["n_failed_workers"] = len(worker_errors)
            state.metadata["worker_errors"] = worker_errors

            # Now try aggregation
            agg_result = None
            agg_error = None

            if worker_outputs:
                try:
                    agg_result = aggregate_worker_measurements(worker_outputs, causal_spec)
                    state.metadata["agg_keys"] = list(agg_result.keys())
                    state.metadata["agg_shapes"] = {
                        k: (df.height, df.width) for k, df in agg_result.items()
                    }
                    # Store dataframes as markdown tables for Inspect UI viewing
                    state.metadata["aggregates"] = {
                        k: df.to_pandas().to_markdown() for k, df in agg_result.items()
                    }
                except Exception as e:
                    agg_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            # Build completion message summarizing results
            completion_parts = [
                f"Workers: {len(worker_outputs)} successful, {len(worker_errors)} failed",
            ]

            if worker_errors:
                completion_parts.append(f"Worker errors: {'; '.join(worker_errors[:3])}")

            if agg_result is not None:
                completion_parts.append(f"Aggregation: SUCCESS - keys={list(agg_result.keys())}")
            elif agg_error:
                completion_parts.append(f"Aggregation: FAILED - {agg_error}")
            else:
                completion_parts.append("Aggregation: SKIPPED (no worker outputs)")

            state.output.completion = "\n".join(completion_parts)
            state.metadata["agg_error"] = agg_error
            state.metadata["agg_success"] = agg_result is not None

            return state

        return solve

    return _solver()


@scorer(metrics=[accuracy(), stderr()])
def aggregation_scorer():
    """Score aggregation robustness.

    Returns:
        - "C" (correct) if aggregation completed successfully
        - "I" (incorrect) if aggregation raised an exception
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        agg_success = state.metadata.get("agg_success", False)
        agg_error = state.metadata.get("agg_error")
        n_successful = state.metadata.get("n_successful_workers", 0)
        n_failed = state.metadata.get("n_failed_workers", 0)
        agg_keys = state.metadata.get("agg_keys", [])
        agg_shapes = state.metadata.get("agg_shapes", {})

        if agg_success:
            # Build success explanation
            shape_info = ", ".join(f"{k}={s}" for k, s in agg_shapes.items())
            explanation = (
                f"Aggregation succeeded with {n_successful} worker outputs. "
                f"Result keys: {agg_keys}. Shapes: {shape_info}"
            )
            return Score(
                value="C",
                answer="SUCCESS",
                explanation=explanation,
                metadata={
                    "n_successful_workers": n_successful,
                    "n_failed_workers": n_failed,
                    "agg_keys": agg_keys,
                    "agg_shapes": agg_shapes,
                },
            )
        else:
            # Build failure explanation
            explanation = f"Aggregation failed with {n_successful} worker outputs."
            if agg_error:
                explanation += f"\nError: {agg_error}"
            if n_successful == 0:
                explanation += "\nNote: No successful worker outputs to aggregate."

            return Score(
                value="I",
                answer="FAILED",
                explanation=explanation,
                metadata={
                    "n_successful_workers": n_successful,
                    "n_failed_workers": n_failed,
                    "error": agg_error,
                },
            )

    return score


@task
def aggregation_robustness_eval(
    n_sets: int = 10,
    chunks_per_set: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    worker_timeout: int = 300,
    questions: str | None = None,
):
    """Evaluate aggregation function robustness on worker outputs.

    Generates N sets of M chunks, processes each set through workers, and tests
    whether aggregate_worker_measurements handles the outputs without crashing.

    Sets cycle through all question-CausalSpec pairs with available causal_spec
    artifacts.

    Args:
        n_sets: Number of aggregation tests (N)
        chunks_per_set: Chunks per test (M workers per test)
        seed: Random seed for chunk sampling
        input_file: Specific preprocessed file name, or None for latest
        worker_timeout: Timeout in seconds for each worker (default: 300s)
        questions: Optional comma-separated question selectors (e.g. "1,4")
    """
    return Task(
        dataset=create_eval_dataset(
            n_sets=n_sets,
            chunks_per_set=chunks_per_set,
            seed=seed,
            input_file=input_file,
            questions=questions,
        ),
        solver=[
            aggregation_solver(worker_timeout=worker_timeout),
        ],
        scorer=aggregation_scorer(),
    )
