"""Inspect AI evaluation for Target 1a: Latent Structure Proposal.

Tests the orchestrator's ability to propose valid theoretical causal structures
from a research question alone, WITHOUT seeing any data.

This evaluates domain knowledge and causal reasoning, not data operationalization.

Uses the same core logic as production (``run_latent_structure``), driven through a real
``ScopedSessionFactory`` for the model under test (the ``-T model=`` task arg,
defaulting to the configured Target 1 model).

Usage:
    inspect eval evals/single_model/eval1a_latent_structure.py -T model=openrouter/anthropic/claude-sonnet-4
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json

from evaluation.inspect_evals.common import (
    discover_questions,
    make_eval_session_factory,
    select_questions,
)
from evaluation.scorers.constructs import count_rule_points
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.flows.transitions.latent_structure.run import (
    LatentStructureResult,
    run_latent_structure,
)


def create_eval_dataset(questions: str | None = None) -> MemoryDataset:
    """Create evaluation dataset from questions.

    Note: Target 1a uses questions ONLY - no data samples.

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
                input=q.question,  # Just the question, latent_structure builds the full prompt
                id=f"q_{q.slug}",
                metadata={
                    "question": q.question,
                },
            )
        )

    return MemoryDataset(samples)


@scorer(metrics=[mean(), stderr()])
def latent_structure_scorer():
    """Score latent structure proposals using cumulative points.

    Returns numeric score:
        - 0.0 if structure is invalid (with detailed error explanation)
        - Cumulative points from scoring rules if valid:
          - Points per construct (role, temporal_status, granularity)
          - Points per edge (valid endpoints, not exogenous effect, timescale)
          - Bonus for cross-timescale edges (complexity)
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        # Get the LatentStructureResult from metadata (set by solver)
        result: LatentStructureResult | None = state.metadata.get("latent_result")

        if result is None:
            return Score(
                value=0.0,
                answer="[No result]",
                explanation="Target 1a did not produce a result",
            )

        # Validate against schema
        try:
            structure = LatentStructure(**result.latent_structure)
        except Exception as e:  # noqa: BLE001
            return Score(
                value=0.0,
                answer=json.dumps(result.latent_structure)[:500],
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Count points
        total = count_rule_points(structure)

        return Score(
            value=total,
            answer=json.dumps(result.latent_structure, indent=2)[:500],
            explanation=f"Score: {total} points",
            metadata={
                "n_constructs": len(structure.constructs),
                "n_edges": len(structure.edges),
            },
        )

    return score


def latent_structure_solver(model: str | None = None):
    """Solver that runs the production Target 1a flow via a real session factory."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            question = state.metadata.get("question", "")

            # Run the SAME core logic as production, on the model under test.
            async with make_eval_session_factory("target-1a", model) as factory:
                result = await run_latent_structure(question=question, session_factory=factory)

            # Store result in metadata for scorer
            state.metadata["latent_result"] = result
            state.output.completion = json.dumps(result.latent_structure, indent=2)

            return state

        return solve

    return _solver()


@task
def latent_structure_eval(questions: str | None = None, model: str | None = None):
    """Evaluate LLM ability to propose theoretical causal structures (latent structures).

    Target 1a evaluation:
    - Input: Research question only (NO data)
    - Output: LatentStructure (constructs + causal edges)

    Args:
        questions: Optional comma-separated question selectors (e.g. "1,3,5")
        model: Model under test as an ``openrouter/...`` slug; defaults to the
            configured Target 1 model.
    """
    return Task(
        dataset=create_eval_dataset(questions=questions),
        solver=[latent_structure_solver(model=model)],
        scorer=latent_structure_scorer(),
    )
