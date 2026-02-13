"""Inspect AI evaluation for Stage 1a: Latent Model Proposal.

Tests the orchestrator's ability to propose valid theoretical causal structures
from a research question alone, WITHOUT seeing any data.

This evaluates domain knowledge and causal reasoning, not data operationalization.

Uses the same core logic as production (via run_stage1a), just with
a different model configuration.

Usage:
    inspect eval evals/eval1a_latent_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval1a_latent_model.py --model openrouter/google/gemini-2.5-pro-preview
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from causal_ssm_agent.orchestrator.prompts import latent_model
from causal_ssm_agent.orchestrator.schemas import LatentModel
from causal_ssm_agent.orchestrator.scoring import _count_rule_points
from causal_ssm_agent.orchestrator.stage1a import Stage1aResult, run_stage1a
from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn
from evals.common import (
    discover_questions,
    load_eval_config,
    select_questions,
)

# Load config for models
_CONFIG = load_eval_config()

# Top-tier models for orchestrator eval
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


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
        except Exception as e:
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


def latent_model_solver():
    """Solver that runs the full Stage 1a flow using core logic."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            generate_fn = make_orchestrator_generate_fn(model)

            # Get metadata
            question = state.metadata.get("question", "")

            # Run the SAME core logic as production
            result = await run_stage1a(
                question=question,
                generate=generate_fn,
            )

            # Store result in metadata for scorer
            state.metadata["stage1a_result"] = result
            state.output.completion = json.dumps(result.latent_model, indent=2)

            return state

        return solve

    return _solver()


@task
def latent_model_eval(questions: str | None = None):
    """Evaluate LLM ability to propose theoretical causal structures (latent models).

    Stage 1a evaluation:
    - Input: Research question only (NO data)
    - Output: LatentModel (constructs + causal edges)

    Uses the production two-stage pipeline:
    1. Initial proposal from question
    2. Self-review focusing on theoretical coherence

    Args:
        questions: Optional comma-separated question selectors (e.g. "1,3,5")
    """
    return Task(
        dataset=create_eval_dataset(questions=questions),
        solver=[
            system_message(latent_model.SYSTEM),
            latent_model_solver(),
        ],
        scorer=latent_model_scorer(),
    )
