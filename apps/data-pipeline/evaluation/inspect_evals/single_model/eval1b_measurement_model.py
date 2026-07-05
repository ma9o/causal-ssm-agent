"""Inspect AI evaluation for Stage 1b: Measurement Model with Identifiability.

Tests the orchestrator's ability to:
1. Operationalize theoretical constructs into measurable indicators
2. Check identifiability of target causal effects
3. Request proxies for blocking confounders when needed

Uses the same core logic as production (``run_stage1b``), driven through a real
``StageSessionFactory`` for the model under test.

Usage:
    inspect eval evals/single_model/eval1b_measurement_model.py \
        -T model=openrouter/anthropic/claude-sonnet-4 -T workspace_id=DEMO
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json

from evaluation.inspect_evals.common import (
    load_workspace_stage1b_inputs,
    make_eval_session_factory,
)
from evaluation.scorers.measurement import score_measurement_model
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.artifacts.latent_model import LatentModel
from nof1_causal_lab.flows.stages.stage1b.run import Stage1bResult, run_stage1b
from nof1_causal_lab.utils.causal_spec import get_all_treatments, get_outcome_name


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
        except Exception as e:  # noqa: BLE001
            return Score(
                value=0.0,
                answer="[Invalid latent]",
                explanation=f"Could not parse latent model: {e}",
            )

        scoring = score_measurement_model(result, latent)

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


def measurement_model_solver(model: str | None = None):
    """Solver that runs the production Stage 1b flow via a real session factory."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            latent_model = state.metadata.get("latent_model", {})
            question = state.metadata.get("question", "")
            chunks = state.metadata.get("chunks", [])
            dataset_summary = state.metadata.get("dataset_summary", "")

            # Run the SAME core logic as production, on the model under test.
            async with make_eval_session_factory("stage-1b", model) as factory:
                result = await run_stage1b(
                    question=question,
                    latent_model=latent_model,
                    chunks=chunks,
                    session_factory=factory,
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
    model: str | None = None,
):
    """Evaluate Stage 1b using the production logic.

    The eval uses the exact same run_stage1b() function as production, just with
    a chosen model. This ensures the eval tests what actually runs.

    Scoring:
    - Points per indicator (construct ref, dtype, aggregation, specificity)
    - +10: All identifiable

    Args:
        workspace_id: Workspace to load persisted Stage 0 and Stage 1a inputs from.
        model: Model under test as an ``openrouter/...`` slug; defaults to the
            configured Stage 1 model.
    """
    return Task(
        dataset=create_eval_dataset(workspace_id=workspace_id),
        solver=[measurement_model_solver(model=model)],
        scorer=measurement_model_scorer(),
    )
