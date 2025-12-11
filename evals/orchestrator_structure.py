"""Inspect AI evaluation for the orchestrator DSEM structure proposals.

Evaluates top-tier LLMs with max thinking budget on their ability to propose
valid DSEM structures given a causal question and sample data chunks.

Usage:
    inspect eval evals/orchestrator_structure.py --model openrouter/anthropic/claude-opus-4.5
    inspect eval evals/orchestrator_structure.py --model openrouter/google/gemini-3-pro-preview-20251117
    inspect eval evals/orchestrator_structure.py --model openrouter/openai/gpt-5.1
    inspect eval evals/orchestrator_structure.py --model openrouter/deepseek/deepseek-v3.2
    inspect eval evals/orchestrator_structure.py --model openrouter/moonshotai/kimi-k2
"""

import json
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, system_message

from causal_agent.orchestrator.prompts import (
    STRUCTURE_PROPOSER_SYSTEM,
    STRUCTURE_PROPOSER_USER,
)
from causal_agent.orchestrator.scoring import _count_rule_points_detailed
from causal_agent.orchestrator.schemas import DSEMStructure
from causal_agent.utils.data import PROCESSED_DIR

from .common import (
    extract_json_from_response,
    format_chunks,
    get_sample_chunks_orchestrator,
    reasoning_generate,
)

# Top-tier models for orchestrator eval (via OpenRouter)
# Model ID -> short alias for CLI convenience
MODELS = {
    "openrouter/anthropic/claude-opus-4.5": "claude",
    "openrouter/google/gemini-3-pro-preview-20251117": "gemini",
    "openrouter/openai/gpt-5.1": "gpt",
    "openrouter/deepseek/deepseek-v3.2": "deepseek",
    "openrouter/moonshotai/kimi-k2": "kimi",
}


@dataclass
class EvalQuestion:
    """An evaluation question with metadata."""

    id: int
    question: str
    difficulty: float
    domain: str
    primary_challenge: str


def load_eval_questions() -> list[EvalQuestion]:
    """Load evaluation questions from the JSON file."""
    eval_file = PROCESSED_DIR.parent / "eval" / "orchestrator_eval_questions.json"

    with open(eval_file) as f:
        data = json.load(f)

    return [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            difficulty=q["total_difficulty"],
            domain=q["domain"],
            primary_challenge=q["primary_challenge"],
        )
        for q in data["evaluation_questions"]
    ]


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset by combining questions with sampled chunks.

    Args:
        n_chunks: Number of chunks to sample per question
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples for each question
    """
    questions = load_eval_questions()

    # Sample chunks (same for all questions for fair comparison)
    chunks = get_sample_chunks_orchestrator(n_chunks, seed, input_file)
    formatted_chunks = format_chunks(chunks)

    samples = []
    for q in questions:
        # Build the user prompt
        user_prompt = STRUCTURE_PROPOSER_USER.format(
            question=q.question,
            dataset_summary="Personal activity data export",
            chunks=formatted_chunks,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                    "difficulty": q.difficulty,
                    "domain": q.domain,
                    "primary_challenge": q.primary_challenge,
                    "n_chunks": n_chunks,
                    "seed": seed,
                },
            )
        )

    return MemoryDataset(samples)


@scorer(metrics=[mean(), stderr()])
def dsem_structure_scorer():
    """Score DSEM structure proposals using cumulative points.

    Returns numeric score:
        - 0.0 if structure is invalid (with detailed error explanation)
        - Cumulative points from _count_rule_points() if valid
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion

        # Extract JSON from response
        json_str = extract_json_from_response(completion)
        if json_str is None:
            return Score(
                value=0.0,
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
                value=0.0,
                answer=json_str[:200] + "..." if len(json_str) > 200 else json_str,
                explanation=f"ERROR: JSON parse failed - {e}",
            )

        # Validate against schema
        try:
            structure = DSEMStructure(**data)
        except Exception as e:
            return Score(
                value=0.0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Count points with detailed breakdown
        scoring = _count_rule_points_detailed(structure)

        return Score(
            value=scoring["total"],
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=scoring["breakdown"],
            metadata={
                "dimensions": scoring["dimensions"],
                "edges": scoring["edges"],
                "n_dimensions": len(structure.dimensions),
                "n_edges": len(structure.edges),
            },
        )

    return score


@task
def orchestrator_eval(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to propose DSEM structures.

    Args:
        n_chunks: Number of data chunks to include in each sample
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            system_message(STRUCTURE_PROPOSER_SYSTEM),
            reasoning_generate(),
        ],
        scorer=dsem_structure_scorer(),
    )
