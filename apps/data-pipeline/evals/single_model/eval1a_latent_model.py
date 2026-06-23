"""Inspect AI evaluation for Stage 1a: Latent Model Proposal.

Tests the orchestrator's ability to propose valid theoretical causal structures
from a research question alone, WITHOUT seeing any data.

This evaluates domain knowledge and causal reasoning, not data operationalization.

Uses the same core logic as production (``run_stage1a``), driven through a real
``StageSessionFactory`` for the model under test (the ``-T model=`` task arg,
defaulting to the configured Stage 1 model).

Usage:
    inspect eval evals/single_model/eval1a_latent_model.py -T model=openrouter/anthropic/claude-sonnet-4
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

from evals.common import (
    discover_questions,
    make_eval_session_factory,
    select_questions,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.artifacts.latent_model import LatentModel
from nof1_causal_lab.flows.stages.stage1a.run import Stage1aResult, run_stage1a
from nof1_causal_lab.orchestrator.scoring import _count_rule_points


def create_eval_dataset(questions: str | None = None) -> MemoryDataset:
    """Create evaluation dataset from questions.

    Note: Stage 1a uses questions ONLY - no data samples.

    Args:
        questions: Optional comma-separated selectors to filter questions.

    Returns:
        MemoryDataset with samples for each question
    """
    all_questions = discover_questions()
    if questions:
        all_questions = select_questions(all_questions, questions)

    samples = []
    for q in all_questions:
        samples.append(
            Sample(
                input=q.question,  # Just the question, stage1a builds the full prompt
                id=f"q_{q.slug}",
                metadata={
                    "question": q.question,
                },
            )
        )

    return MemoryDataset(samples)


@scorer(metrics=[mean(), stderr()])
def latent_model_scorer():
    """Score latent model proposals using cumulative points.

    Returns numeric score:
        - 0.0 if structure is invalid (with detailed error explanation)
        - Cumulative points from scoring rules if valid:
          - Points per construct (role, temporal_status, granularity)
          - Points per edge (valid endpoints, not exogenous effect, timescale)
          - Bonus for cross-timescale edges (complexity)
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        # Get the Stage1aResult from metadata (set by solver)
        result: Stage1aResult | None = state.metadata.get("stage1a_result")

        if result is None:
            return Score(
                value=0.0,
                answer="[No result]",
                explanation="Stage 1a did not produce a result",
            )

        # Validate against schema
        try:
            structure = LatentModel(**result.latent_model)
        except Exception as e:  # noqa: BLE001
            return Score(
                value=0.0,
                answer=json.dumps(result.latent_model)[:500],
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Count points
        total = _count_rule_points(structure)

        return Score(
            value=total,
            answer=json.dumps(result.latent_model, indent=2)[:500],
            explanation=f"Score: {total} points",
            metadata={
                "n_constructs": len(structure.constructs),
                "n_edges": len(structure.edges),
            },
        )

    return score


def latent_model_solver(model: str | None = None):
    """Solver that runs the production Stage 1a flow via a real session factory."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            question = state.metadata.get("question", "")

            # Run the SAME core logic as production, on the model under test.
            async with make_eval_session_factory("stage-1a", model) as factory:
                result = await run_stage1a(question=question, session_factory=factory)

            # Store result in metadata for scorer
            state.metadata["stage1a_result"] = result
            state.output.completion = json.dumps(result.latent_model, indent=2)

            return state

        return solve

    return _solver()


@task
def latent_model_eval(questions: str | None = None, model: str | None = None):
    """Evaluate LLM ability to propose theoretical causal structures (latent models).

    Stage 1a evaluation:
    - Input: Research question only (NO data)
    - Output: LatentModel (constructs + causal edges)

    Args:
        questions: Optional comma-separated question selectors (e.g. "1,3,5")
        model: Model under test as an ``openrouter/...`` slug; defaults to the
            configured Stage 1 model.
    """
    return Task(
        dataset=create_eval_dataset(questions=questions),
        solver=[latent_model_solver(model=model)],
        scorer=latent_model_scorer(),
    )
