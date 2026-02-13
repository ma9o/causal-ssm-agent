"""Inspect AI evaluation for Stage 1b: Measurement Model with Identifiability.

Tests the orchestrator's ability to:
1. Operationalize theoretical constructs into measurable indicators
2. Check identifiability of target causal effects
3. Request proxies for blocking confounders when needed

Uses the same core logic as production (via run_stage1b), just with
a different model configuration.

Usage:
    inspect eval evals/eval1b_measurement_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval1b_measurement_model.py --model openrouter/google/gemini-2.5-pro-preview-06-05
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

from causal_ssm_agent.orchestrator.prompts import measurement_model
from causal_ssm_agent.orchestrator.schemas import LatentModel, MeasurementModel
from causal_ssm_agent.orchestrator.stage1b import Stage1bResult, run_stage1b
from causal_ssm_agent.utils.effects import get_all_treatments, get_outcome_from_latent_model
from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn
from evals.common import (
    get_questions_with_latent_model,
    get_sample_chunks_orchestrator,
    load_eval_config,
    select_questions,
)

# Load config for models
_CONFIG = load_eval_config()
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    questions: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Args:
        n_chunks: Number of data chunks to include.
        seed: Random seed for sampling.
        input_file: Specific input file name, or None for latest.
        questions: Optional comma-separated selectors to filter questions.
    """
    all_questions = get_questions_with_latent_model()
    if questions:
        all_questions = select_questions(all_questions, questions)

    chunks = get_sample_chunks_orchestrator(n_chunks, seed, input_file)

    samples = []
    for q in all_questions:
        latent_model = q.load_latent_model()
        outcome = get_outcome_from_latent_model(latent_model)
        treatments = get_all_treatments(latent_model)

        samples.append(
            Sample(
                input=q.question,  # Just the question, stage1b builds the full prompt
                id=f"q_{q.slug}",
                metadata={
                    "question": q.question,
                    "latent_model": latent_model,
                    "outcome": outcome,
                    "treatments": treatments,
                    "chunks": chunks,
                    "n_chunks": n_chunks,
                    "seed": seed,
                },
            )
        )

    return MemoryDataset(samples)


def _score_stage1b_result(
    result: Stage1bResult,
    latent: LatentModel,
) -> dict:
    """Score a Stage 1b result.

    Scoring rules:
    - +2 per valid indicator (references known construct)
    - +1 for valid dtype
    - +1 for valid aggregation
    - +1 for specific how_to_measure (>50 chars)
    - +2 bonus for multiple indicators per construct
    - Identifiability bonuses:
      - +10 if ALL treatments identifiable from start
      - +15 if ALL treatments identifiable after proxy fix
      - +5 if partial improvement from proxy request
    """
    breakdown = []
    indicator_points = {}
    total = 0.0

    # Parse measurement model
    try:
        measurement = MeasurementModel.model_validate(result.measurement_model)
    except Exception as e:
        return {
            "total": 0.0,
            "breakdown": f"Invalid measurement model: {e}",
            "error": True,
        }

    construct_names = {c.name for c in latent.constructs}
    indicators_per_construct: dict[str, int] = {}

    for indicator in measurement.indicators:
        pts = 0
        details = []

        if indicator.construct_name in construct_names:
            pts += 2
            details.append(f"+2 valid construct '{indicator.construct_name}'")
            indicators_per_construct[indicator.construct_name] = (
                indicators_per_construct.get(indicator.construct_name, 0) + 1
            )
        else:
            details.append(f"+0 unknown construct '{indicator.construct_name}'")

        valid_dtypes = {"continuous", "binary", "count", "ordinal", "categorical"}
        if indicator.measurement_dtype in valid_dtypes:
            pts += 1
            details.append("+1 valid dtype")

        pts += 1  # Valid aggregation (schema-validated)
        details.append("+1 valid aggregation")

        if len(indicator.how_to_measure) > 50:
            pts += 1
            details.append("+1 specific how_to_measure")

        indicator_points[indicator.name] = {"points": pts, "details": details}
        total += pts

    # Multi-indicator bonus
    for construct, count in indicators_per_construct.items():
        if count > 1:
            bonus = (count - 1) * 2
            total += bonus
            breakdown.append(f"+{bonus} multi-indicator for '{construct}' ({count})")

    # Identifiability bonuses
    initial_non_id = len(result.initial_identifiability["non_identifiable_treatments"])
    final_non_id = len(result.final_identifiability["non_identifiable_treatments"])

    if final_non_id == 0:
        if initial_non_id > 0:
            breakdown.append("+15 ALL identifiable after proxy fix!")
            total += 15
        else:
            breakdown.append("+10 ALL identifiable from start!")
            total += 10
    elif initial_non_id > final_non_id:
        improved = initial_non_id - final_non_id
        breakdown.append(f"+5 Fixed {improved} treatments via proxies")
        total += 5
    else:
        breakdown.append(f"+0 {final_non_id} treatments still not identifiable")

    # Build breakdown summary
    breakdown.insert(0, f"INDICATORS ({len(measurement.indicators)}):")
    for name, info in indicator_points.items():
        breakdown.append(f"  {name}: {info['points']} pts")

    breakdown.append(f"\nTOTAL: {total} points")

    return {
        "total": total,
        "indicators": indicator_points,
        "breakdown": "\n".join(breakdown),
        "indicators_per_construct": indicators_per_construct,
    }


@scorer(metrics=[mean(), stderr()])
def measurement_model_scorer():
    """Score Stage 1b results."""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        # Get the Stage1bResult from metadata (set by solver)
        result: Stage1bResult | None = state.metadata.get("stage1b_result")

        if result is None:
            return Score(
                value=0.0,
                answer="[No result]",
                explanation="Stage 1b did not produce a result",
            )

        latent_data = state.metadata.get("latent_model", {})
        try:
            latent = LatentModel(**latent_data)
        except Exception as e:
            return Score(
                value=0.0,
                answer="[Invalid latent]",
                explanation=f"Could not parse latent model: {e}",
            )

        scoring = _score_stage1b_result(result, latent)

        if scoring.get("error"):
            return Score(
                value=0.0,
                answer="[Invalid measurement]",
                explanation=scoring["breakdown"],
            )

        return Score(
            value=scoring["total"],
            answer=json.dumps(result.measurement_model, indent=2)[:500],
            explanation=scoring["breakdown"],
            metadata={
                "n_indicators": len(result.measurement_model.get("indicators", [])),
                "proxy_requested": result.proxy_requested,
                "initial_non_identifiable": len(result.initial_identifiability["non_identifiable_treatments"]),
                "final_non_identifiable": len(result.final_identifiability["non_identifiable_treatments"]),
            },
        )

    return score


def measurement_model_solver():
    """Solver that runs the full Stage 1b flow using core logic."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            generate_fn = make_orchestrator_generate_fn(model)

            # Get metadata
            latent_model = state.metadata.get("latent_model", {})
            question = state.metadata.get("question", "")
            chunks = state.metadata.get("chunks", [])

            # Run the SAME core logic as production
            result = await run_stage1b(
                question=question,
                latent_model=latent_model,
                chunks=chunks,
                generate=generate_fn,
            )

            # Store result in metadata for scorer
            state.metadata["stage1b_result"] = result
            state.output.completion = json.dumps(result.measurement_model, indent=2)

            return state

        return solve

    return _solver()


@task
def measurement_model_eval(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    questions: str | None = None,
):
    """Evaluate Stage 1b using the production logic.

    The eval uses the exact same run_stage1b() function as production,
    just with a different model. This ensures the eval tests what actually runs.

    Scoring:
    - Points per indicator (construct ref, dtype, aggregation, specificity)
    - +10: All identifiable from start
    - +15: All identifiable after proxy fix
    - +5: Partial improvement from proxies

    Args:
        n_chunks: Number of data chunks to include.
        seed: Random seed for sampling.
        input_file: Specific preprocessed file name, or None for latest.
        questions: Optional comma-separated question selectors (e.g. "1,3")
    """
    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file, questions=questions),
        solver=[
            system_message(measurement_model.SYSTEM),
            measurement_model_solver(),
        ],
        scorer=measurement_model_scorer(),
    )
