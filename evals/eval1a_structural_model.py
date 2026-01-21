"""Inspect AI evaluation for Stage 1a: Structural Model Proposal.

Tests the orchestrator's ability to propose valid theoretical causal structures
from a research question alone, WITHOUT seeing any data.

This evaluates domain knowledge and causal reasoning, not data operationalization.

Usage:
    inspect eval evals/eval1a_structural_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval1a_structural_model.py --model openrouter/google/gemini-2.5-pro-preview
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, system_message

from causal_agent.orchestrator.prompts import (
    STRUCTURAL_MODEL_SYSTEM,
    STRUCTURAL_MODEL_USER,
    STRUCTURAL_MODEL_REVIEW,
)
from causal_agent.orchestrator.scoring import _count_rule_points_detailed
from causal_agent.orchestrator.schemas import StructuralModel
from causal_agent.utils.llm import validate_structural_model_tool

from evals.common import (
    extract_json_from_response,
    get_eval_questions,
    load_eval_config,
    tool_assisted_generate,
)

# Load config for models
_CONFIG = load_eval_config()

# Top-tier models for orchestrator eval
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


@dataclass
class EvalQuestion:
    """An evaluation question with metadata."""

    id: int
    question: str


def load_questions() -> list[EvalQuestion]:
    """Load evaluation questions from config."""
    return [
        EvalQuestion(id=q["id"], question=q["question"])
        for q in get_eval_questions()
    ]


def create_eval_dataset() -> MemoryDataset:
    """Create evaluation dataset from questions.

    Note: Stage 1a uses questions ONLY - no data samples.

    Returns:
        MemoryDataset with samples for each question
    """
    questions = load_questions()

    samples = []
    for q in questions:
        # Build the user prompt - question only, no data
        user_prompt = STRUCTURAL_MODEL_USER.format(question=q.question)

        samples.append(
            Sample(
                input=user_prompt,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                },
            )
        )

    return MemoryDataset(samples)


@scorer(metrics=[mean(), stderr()])
def structural_model_scorer():
    """Score structural model proposals using cumulative points.

    Returns numeric score:
        - 0.0 if structure is invalid (with detailed error explanation)
        - Cumulative points from scoring rules if valid:
          - Points per construct (role, temporal_status, granularity)
          - Points per edge (valid endpoints, not exogenous effect, timescale)
          - Bonus for cross-timescale edges (complexity)
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
            structure = StructuralModel(**data)
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
                "constructs": scoring["constructs"],
                "edges": scoring["edges"],
                "n_constructs": len(structure.constructs),
                "n_edges": len(structure.edges),
            },
        )

    return score


@task
def structural_model_eval():
    """Evaluate LLM ability to propose theoretical causal structures.

    Stage 1a evaluation:
    - Input: Research question only (NO data)
    - Output: StructuralModel (constructs + causal edges)

    Uses the production two-stage pipeline:
    1. Initial proposal from question
    2. Self-review focusing on theoretical coherence
    """
    return Task(
        dataset=create_eval_dataset(),
        solver=[
            system_message(STRUCTURAL_MODEL_SYSTEM),
            tool_assisted_generate(
                tools=[validate_structural_model_tool()],
                follow_ups=[STRUCTURAL_MODEL_REVIEW],
            ),
        ],
        scorer=structural_model_scorer(),
    )
