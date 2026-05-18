"""Inspect AI evaluation for Stage 1b: Measurement Model with Identifiability.

Tests the orchestrator's ability to:
1. Operationalize theoretical constructs into measurable indicators
2. Check identifiability of target causal effects
3. Request proxies for blocking confounders when needed

Uses the same core logic as production (via run_stage1b), just with
a different model configuration.

Usage:
    inspect eval evals/single_model/eval1b_measurement_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/single_model/eval1b_measurement_model.py \
        --model openrouter/google/gemini-2.5-pro-preview-06-05 \
        -T workspace_id=SMALLGOLDEN
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

from evals.common import (
    load_eval_config,
    load_workspace_stage1b_inputs,
    make_generate_fn,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from nof1_causal_lab.orchestrator.prompts import measurement_model
from nof1_causal_lab.orchestrator.schemas import LatentModel, MeasurementModel
from nof1_causal_lab.orchestrator.stage1b import Stage1bResult, run_stage1b
from nof1_causal_lab.utils.causal_spec import get_all_treatments, get_outcome_name

# Load config for models
_CONFIG = load_eval_config()
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


def create_eval_dataset(
    workspace_id: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Args:
        workspace_id: Workspace to load persisted Stage 0 and Stage 1a inputs from.
    """
    inputs = load_workspace_stage1b_inputs(workspace_id)
    latent_model = inputs["latent_model"]
    outcome = get_outcome_name(latent_model)
    treatments = get_all_treatments(latent_model)

    return MemoryDataset(
        [
            Sample(
                input=inputs["question"],
                id=f"workspace_{inputs['workspace_id']}",
                metadata={
                    "workspace_id": inputs["workspace_id"],
                    "question": inputs["question"],
                    "latent_model": latent_model,
                    "outcome": outcome,
                    "treatments": treatments,
                    "chunks": inputs["chunks"],
                    "dataset_summary": inputs["dataset_summary"],
                },
            )
        ]
    )


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
    non_id = len(result.identifiability_status.get("non_identifiable_treatments", {}))

    if non_id == 0:
        breakdown.append("+10 ALL identifiable!")
        total += 10
    else:
        breakdown.append(f"+0 {non_id} treatments not identifiable")

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
                "non_identifiable": len(
                    result.identifiability_status.get("non_identifiable_treatments", {})
                ),
            },
        )

    return score


def measurement_model_solver():
    """Solver that runs the full Stage 1b flow using core logic."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            generate_fn = make_generate_fn(model)

            # Get metadata
            latent_model = state.metadata.get("latent_model", {})
            question = state.metadata.get("question", "")
            chunks = state.metadata.get("chunks", [])
            dataset_summary = state.metadata.get("dataset_summary", "")

            # Run the SAME core logic as production
            result = await run_stage1b(
                question=question,
                latent_model=latent_model,
                chunks=chunks,
                generate=generate_fn,
                dataset_summary=dataset_summary,
            )

            # Store result in metadata for scorer
            state.metadata["stage1b_result"] = result
            state.output.completion = json.dumps(result.measurement_model, indent=2)

            return state

        return solve

    return _solver()


@task
def measurement_model_eval(
    workspace_id: str | None = None,
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
        workspace_id: Workspace to load persisted Stage 0 and Stage 1a inputs from.
    """
    return Task(
        dataset=create_eval_dataset(workspace_id=workspace_id),
        solver=[
            system_message(measurement_model.SYSTEM),
            measurement_model_solver(),
        ],
        scorer=measurement_model_scorer(),
    )
