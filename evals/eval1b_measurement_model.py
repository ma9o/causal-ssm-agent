"""Inspect AI evaluation for Stage 1b: Measurement Model Proposal.

Tests the orchestrator's ability to operationalize theoretical constructs
into measurable indicators given sample data.

This evaluates data understanding and operationalization skills.
Requires reference structural models from Stage 1a.

Usage:
    inspect eval evals/eval1b_measurement_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval1b_measurement_model.py --model openrouter/google/gemini-2.5-pro-preview
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
    MEASUREMENT_MODEL_SYSTEM,
    MEASUREMENT_MODEL_USER,
    MEASUREMENT_MODEL_REVIEW,
)
from causal_agent.orchestrator.schemas import MeasurementModel, StructuralModel
from causal_agent.utils.llm import make_validate_measurement_model_tool

from evals.common import (
    extract_json_from_response,
    format_chunks,
    get_eval_questions,
    get_sample_chunks_orchestrator,
    load_eval_config,
    load_structural_model_by_question_id,
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


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset by combining questions with structural models and data.

    Args:
        n_chunks: Number of chunks to sample per question
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples for each question
    """
    questions = load_questions()

    # Sample chunks (same for all questions for fair comparison)
    chunks = get_sample_chunks_orchestrator(n_chunks, seed, input_file)
    formatted_chunks = format_chunks(chunks)

    samples = []
    for q in questions:
        # Load reference structural model for this question
        structural_model = load_structural_model_by_question_id(q.id)
        structural_json = json.dumps(structural_model, indent=2)

        # Build the user prompt
        user_prompt = MEASUREMENT_MODEL_USER.format(
            question=q.question,
            structural_model_json=structural_json,
            dataset_summary="Personal activity data export",
            chunks=formatted_chunks,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                    "structural_model": structural_model,
                    "n_chunks": n_chunks,
                    "seed": seed,
                },
            )
        )

    return MemoryDataset(samples)


def _score_measurement_model(
    measurement: MeasurementModel,
    structural: StructuralModel,
) -> dict:
    """Score a measurement model against its structural model.

    Scoring rules:
    - +2 per valid indicator (references known construct)
    - +1 for valid dtype
    - +1 for valid aggregation
    - +1 for specific how_to_measure (>50 chars)
    - +2 bonus for multiple indicators per construct (reliability)

    Returns dict with 'total', 'indicators', and 'breakdown'.
    """
    breakdown = []
    indicator_points = {}
    total = 0.0

    construct_names = {c.name for c in structural.constructs}
    indicators_per_construct: dict[str, int] = {}

    for indicator in measurement.indicators:
        pts = 0
        details = []

        # Valid construct reference
        if indicator.construct_name in construct_names:
            pts += 2
            details.append(f"+2 references valid construct '{indicator.construct_name}'")
            indicators_per_construct[indicator.construct_name] = (
                indicators_per_construct.get(indicator.construct_name, 0) + 1
            )
        else:
            details.append(f"+0 unknown construct '{indicator.construct_name}'")

        # Valid dtype
        valid_dtypes = {"continuous", "binary", "count", "ordinal", "categorical"}
        if indicator.measurement_dtype in valid_dtypes:
            pts += 1
            details.append(f"+1 valid dtype '{indicator.measurement_dtype}'")

        # Valid aggregation (already validated by schema, but count it)
        pts += 1
        details.append(f"+1 valid aggregation '{indicator.aggregation}'")

        # Specific how_to_measure
        if len(indicator.how_to_measure) > 50:
            pts += 1
            details.append("+1 specific how_to_measure (>50 chars)")
        else:
            details.append("+0 vague how_to_measure (<50 chars)")

        indicator_points[indicator.name] = {"points": pts, "details": details}
        total += pts

    # Bonus for multiple indicators per construct (reliability)
    multi_indicator_bonus = 0
    for construct, count in indicators_per_construct.items():
        if count > 1:
            bonus = (count - 1) * 2
            multi_indicator_bonus += bonus
            breakdown.append(f"+{bonus} multi-indicator bonus for '{construct}' ({count} indicators)")

    total += multi_indicator_bonus

    # Build breakdown summary
    breakdown.insert(0, f"INDICATORS ({len(measurement.indicators)}):")
    for name, info in indicator_points.items():
        breakdown.append(f"  {name}: {info['points']} pts")
        for d in info["details"]:
            breakdown.append(f"    {d}")

    breakdown.append(f"\nTOTAL: {total} points")

    return {
        "total": total,
        "indicators": indicator_points,
        "breakdown": "\n".join(breakdown),
        "indicators_per_construct": indicators_per_construct,
    }


@scorer(metrics=[mean(), stderr()])
def measurement_model_scorer():
    """Score measurement model proposals.

    Returns numeric score:
        - 0.0 if model is invalid (with detailed error explanation)
        - Cumulative points if valid:
          - Points per indicator (construct ref, dtype, aggregation, specificity)
          - Bonus for multiple indicators per construct
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        structural_data = state.metadata.get("structural_model", {})

        # Parse structural model from metadata
        try:
            structural = StructuralModel(**structural_data)
        except Exception as e:
            return Score(
                value=0.0,
                answer="[Invalid structural model in metadata]",
                explanation=f"ERROR: Could not parse structural model - {e}",
            )

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
            measurement = MeasurementModel(**data)
        except Exception as e:
            return Score(
                value=0.0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Score the measurement model
        scoring = _score_measurement_model(measurement, structural)

        return Score(
            value=scoring["total"],
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=scoring["breakdown"],
            metadata={
                "indicators": scoring["indicators"],
                "n_indicators": len(measurement.indicators),
                "indicators_per_construct": scoring["indicators_per_construct"],
            },
        )

    return score


@task
def measurement_model_eval(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to operationalize constructs into indicators.

    Stage 1b evaluation:
    - Input: Question + reference structural model + data sample
    - Output: MeasurementModel (indicators)

    Uses the production two-stage pipeline:
    1. Initial proposal from structural model + data
    2. Self-review focusing on operationalization coherence

    Args:
        n_chunks: Number of data chunks to include in each sample
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    # Load a structural model to create the validation tool
    # (We use the first question's structural model as representative)
    questions = get_eval_questions()
    structural_data = load_structural_model_by_question_id(questions[0]["id"])
    structural = StructuralModel(**structural_data)

    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            system_message(MEASUREMENT_MODEL_SYSTEM),
            tool_assisted_generate(
                tools=[make_validate_measurement_model_tool(structural)],
                follow_ups=[MEASUREMENT_MODEL_REVIEW],
            ),
        ],
        scorer=measurement_model_scorer(),
    )
