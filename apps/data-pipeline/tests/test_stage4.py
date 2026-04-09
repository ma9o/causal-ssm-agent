"""Tests for Stage 4: Model Specification & Prior Elicitation.

Unit tests for prior validation helpers, default priors, and aggregation live
in their dedicated files:
- test_prior_predictive.py (NaN/constraint/extreme checks, format functions)
- test_prior_aggregation.py (simple/GMM aggregation)
- test_get_default_prior.py (constraint→distribution mapping)
Grounding helpers live in ``test_stage4_grounding.py``.
This file tests Stage 4 prompt assembly, orchestration, prior predictive
validation, failed parameter identification with causal_spec context, SSM prior
conversion, and trial compilation.
"""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from causal_ssm_agent.flows import stage_registry
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop import run_stage4
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_cards import build_prior_cards
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
    Stage4GroundingResult,
    make_stage4_grounding_result,
    make_stage4_validation_packet,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_navigation import (
    _activate_prior_phase,
    _set_block_cursor,
    _set_done_cursor,
    _set_model_spec_lock_cursor,
    _set_repair_barrier_cursor,
    get_active_plan_block,
    get_stage4_phase,
    make_stage4_runtime,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    build_stage4_plan,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_prompt_context import (
    Stage4Messages,
    format_stage4_plan_status,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer import (
    build_model_spec_from_decisions,
    compute_stage4_validate_step_with_transitions,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import (
    ResolvedRepairScope,
    classify_prior_failure_blocks,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_runtime_projections import (
    project_stage4_graph,
    project_stage4_initial_state,
    project_stage4_snapshot,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_session import Stage4Session
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    Stage4Skeleton,
    derive_deterministic_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import (
    Stage4AcceptedState,
    Stage4RepairCampaignState,
    Stage4Runtime,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_submission import get_stage4_block_handler
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_types import Stage4Deps
from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
from causal_ssm_agent.flows.stages.stage4.flow import (
    _stage4_generate_config,
)
from causal_ssm_agent.models.predictive_simulation import (
    PredictiveObservationMeanOverflow,
)
from causal_ssm_agent.models.prior_predictive import (
    get_failed_parameters,
    validate_prior_predictive,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_priors as compile_ssm_priors,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_ssm_inputs_from_model_spec,
)
from causal_ssm_agent.utils.llm import LLMTrace, make_generate_fn
from causal_ssm_agent.utils.openrouter_client import GenerateConfig
from causal_ssm_agent.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorRepairScope,
    PriorValidationResult,
)
from tests.helpers import make_stage4_plan as _make_plan
from tests.ssm_test_utils import make_ssm_spec


def compute_stage4_validate_step(data, *, plan, runtime, deps):
    """Test helper: advance the reducer by one step, discarding transitions."""
    stage_output, feedback, _transitions = compute_stage4_validate_step_with_transitions(
        data,
        plan=plan,
        runtime=runtime,
        deps=deps,
    )
    return stage_output, feedback


_ORDINAL_LEVELS = ("low", "high")

# --- Fixtures ---


@pytest.fixture
def simple_model_spec() -> dict:
    """A minimal model spec for testing."""
    return {
        "likelihoods": [
            {
                "variable": "mood_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            }
        ],
        "parameters": [
            {
                "name": "rho_mood",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) coefficient for mood",
            },
            {
                "name": "sigma_mood",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for mood",
            },
        ],
    }


@pytest.fixture
def simple_priors() -> dict:
    """Simple priors matching the model spec."""
    return {
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "reasoning": "Weakly informative for AR coefficient",
        },
        "sigma_mood": {
            "parameter": "sigma_mood",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative for residual SD",
        },
    }


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Simple test data with lagged columns."""
    n = 50
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mood_score": rng.normal(5, 1.5, n),
            "mood_score_lag1": rng.normal(5, 1.5, n),
            "subject_id": np.repeat(np.arange(5), 10),
        }
    )


def _make_runtime(
    plan: Stage4Plan,
    *,
    phase: str | None = None,
    active_block_id: str | None = None,
    accepted: Stage4AcceptedState | None = None,
    last_validation_packet: Any = None,
) -> Stage4Runtime:
    """Build a Stage 4 runtime for focused unit tests."""
    runtime = make_stage4_runtime(plan)
    if active_block_id is not None:
        _set_runtime_block(plan, runtime, active_block_id)
    elif phase == "done":
        _set_done_cursor(runtime)
    elif phase == "model_decisions":
        _set_model_spec_lock_cursor(runtime, reason="test override")
    elif phase is not None:
        _set_repair_barrier_cursor(
            runtime,
            reason="test override",
            scope_block_ids=(),
        )
    if accepted is not None:
        runtime.accepted = accepted
    runtime.last_validation_packet = last_validation_packet
    return runtime


def _set_runtime_block(plan: Stage4Plan, runtime: Stage4Runtime, block_id: str) -> None:
    """Move a test runtime onto one promptable Stage 4 block."""
    block = plan.get_block(block_id)
    assert block is not None
    _set_block_cursor(runtime, block)


def _with_positive_indicator_polarity(spec: dict[str, Any]) -> dict[str, Any]:
    """Backfill positive polarity on test indicators unless a test overrides it."""
    spec = deepcopy(spec)
    measurement = spec.get("measurement") or {}
    indicators = measurement.get("indicators") or []
    for indicator in indicators:
        if isinstance(indicator, dict):
            indicator.setdefault("construct_polarity", "positive")
    return spec


def _make_stub_grounding_result(stage_output: dict | None, feedback: str):
    """Wrap test grounding payloads in the typed Stage 4 grounding result."""
    validation = stage_output.get("validation") if isinstance(stage_output, dict) else None
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        status = "compile_error"
    elif (
        validation is not None
        and getattr(validation, "pp_checked", False)
        and getattr(validation, "pp_valid", True) is False
    ):
        status = "prior_predictive_failure"
    elif "missing priors" in feedback.lower():
        status = "accepted_pending_priors"
    else:
        status = "accepted"
    return make_stage4_grounding_result(
        stage_output=stage_output,
        status=status,
        feedback=feedback,
        validation=validation,
        retain_for_next_prompt=feedback != "VALID",
        capture_stage_output=stage_output is not None
        and status in {"accepted", "accepted_pending_priors"},
    )


def test_project_stage4_graph_includes_repair_barrier_and_prior_review_route():
    plan = _make_plan(
        model_blocks=(
            Stage4FrontierBlock(
                id="indicator:sleep_quality",
                kind="indicator_decision",
                label="Sleep Quality",
                variable_names=("sleep_quality",),
            ),
        ),
        review_block=Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Review model spec",
        ),
        prior_blocks=(
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Sleep effects",
                parameter_names=("beta_stress_sleep",),
            ),
        ),
        prior_review_block=Stage4FrontierBlock(
            id="review:prior_system",
            kind="global_prior_review",
            label="Review prior system",
        ),
    )

    graph = project_stage4_graph(plan)
    node_ids = {node["id"] for node in graph["nodes"]}
    edge_pairs = {(edge["from"], edge["to"], edge["kind"]) for edge in graph["edges"]}

    assert "__repair_barrier__" in node_ids
    assert "review:prior_system" in node_ids
    assert ("effects:sleep", "review:prior_system", "repair_transition") in edge_pairs
    assert ("__repair_barrier__", "review:prior_system", "repair_transition") in edge_pairs
    assert ("review:prior_system", "__done__", "phase_advance") in edge_pairs


def test_project_stage4_snapshot_serializes_repair_barrier_cursor_scope():
    plan = _make_plan(
        prior_blocks=(
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Sleep effects",
                parameter_names=("beta_stress_sleep",),
            ),
            Stage4FrontierBlock(
                id="effects:stress",
                kind="effect_prior",
                label="Stress effects",
                parameter_names=("beta_sleep_stress",),
            ),
        ),
    )
    runtime = make_stage4_runtime(plan)
    _set_repair_barrier_cursor(
        runtime,
        reason="repair barrier pending",
        scope_block_ids=("effects:sleep", "effects:stress"),
    )

    snapshot = project_stage4_snapshot(plan, runtime)

    assert snapshot["cursor"] == {
        "kind": "repair_barrier",
        "scope_block_ids": ["effects:sleep", "effects:stress"],
    }


def test_project_stage4_initial_state_matches_initial_plan_runtime():
    causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
    del skeleton, _data_for_model

    graph, snapshot = project_stage4_initial_state(causal_spec)

    assert graph == project_stage4_graph(plan)
    assert snapshot == project_stage4_snapshot(plan, runtime)


def test_scale_mismatch_for_single_indicator_construct_routes_to_dynamics_block():
    causal_spec = _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": [
                    {
                        "name": "chronotype",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "name": "sleep_quality",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "observation_window": "1mo",
                    },
                    {
                        "name": "sleep_quality_search_count",
                        "construct_name": "sleep_quality",
                        "measurement_dtype": "count",
                        "aggregation": "sum",
                    },
                ],
            },
            "estimation": {
                "state_order": ["chronotype", "sleep_quality"],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
                "induced_dependencies": [],
            },
        }
    )
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    active_block = _require_plan_block(plan, "dynamics:sleep_quality")

    repair_plan = classify_prior_failure_blocks(
        plan,
        active_block,
        AssemblyValidation(
            normalized_model_spec={"likelihoods": [], "parameters": []},
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="scale_monthly_eveningness_activity_timing",
                    is_valid=False,
                    code="scale_mismatch",
                    origin="prior_predictive",
                    issue="Scale mismatch for monthly_eveningness_activity_timing",
                    suggested_adjustment="Adjust diffusion/drift priors to match data scale",
                )
            ],
        ),
        runtime,
    )

    assert repair_plan.scope.scope_kind == "direct_writer_blocks"
    assert repair_plan.block_ids == ("dynamics:chronotype",)
    assert (
        repair_plan.scope.reason
        == "Scale mismatch for monthly_eveningness_activity_timing Suggested fix: "
        "Adjust diffusion/drift priors to match data scale"
    )


def test_prior_failure_classification_raises_without_concrete_reason():
    causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
    del causal_spec, skeleton
    active_block = _require_plan_block(plan, "dynamics:sleep")

    with pytest.raises(
        ValueError,
        match="requires a concrete reason",
    ):
        classify_prior_failure_blocks(
            plan,
            active_block,
            AssemblyValidation(
                normalized_model_spec={"likelihoods": [], "parameters": []},
                compile_ok=True,
                pp_checked=True,
                pp_valid=False,
                diagnostics=[
                    PriorValidationResult(
                        parameter="rho_sleep",
                        is_valid=False,
                        code="unspecified",
                        origin="prior_predictive",
                        issue=None,
                        suggested_adjustment=None,
                        related_parameters=["rho_sleep"],
                    )
                ],
            ),
            runtime,
        )


# --- Prompt assembly tests ---


class TestStage4Messages:
    def test_messages_for_scope_include_compact_model_context(self):
        block = Stage4FrontierBlock(
            id="indicator:pss_score",
            kind="indicator_decision",
            label="Choose likelihood for pss_score",
            construct_names=("stress",),
            variable_names=("pss_score",),
            payload={
                "variable": "pss_score",
                "fixed_distribution": "gaussian",
                "valid_links": ["identity"],
            },
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Stress reduces subsequent sleep quality.",
                    }
                ],
            },
            distribution_cards=[
                {
                    "variable": "pss_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the pss column directly",
                    "options": [
                        {"distribution": "gaussian", "links": ["identity"]},
                    ],
                    "profile": {
                        "n_obs": 40,
                        "mean": 12.0,
                        "std": 3.5,
                        "min": 3.0,
                        "max": 21.0,
                    },
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the pss column directly",
                            "is_reference": True,
                            "has_distribution_decision_card": True,
                            "profile": {
                                "n_obs": 40,
                                "mean": 12.0,
                                "std": 3.5,
                                "min": 3.0,
                                "max": 21.0,
                            },
                        }
                    ],
                }
            ],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
        )
        plan = _make_plan(model_blocks=(block,))
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "## Model Topology" in user_content
        assert "Stress reduces subsequent sleep quality." not in user_content
        assert "Use the pss column directly" in user_content
        assert "model_interval_days" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "see distribution decision card" in user_content
        assert "### Parameter Prior Cards" not in user_content
        assert "### Loading Constraints" not in user_content

    def test_messages_for_scope_render_frontier_contract(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "pss_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the pss column directly",
                    "options": [{"distribution": "gaussian", "links": ["identity"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[
                {
                    "name": "lambda_pss_score_stress",
                    "construct": "stress",
                    "indicator": "pss_score",
                }
            ],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
            enable_literature=True,
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "accepted prior",
                    }
                },
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "## Fixed Model Context" in user_content
        assert "## Frontier Status" in user_content
        assert "## Effect-Block Stability Discipline" in user_content
        assert "`id`: `effects:sleep`" in user_content
        assert "Use `submit_prior_block` with exactly this argument object:" in user_content
        assert (
            "This block owns one target construct's full incoming lagged-effect row."
            in user_content
        )
        assert "Treat the remaining headroom as advisory stability telemetry" in user_content
        assert "Normal(0, 0.1-0.2)" in user_content
        assert "Feedback Loop" in user_content
        assert "### Distribution Decision Cards" not in user_content
        assert "### Loading Constraints" not in user_content
        system_content = messages[0]["content"]
        assert "## Effect Row Budget Discipline" in system_content
        assert "batch those `search_literature` calls in the same turn" in system_content
        assert "advisory stability guidance" in system_content
        assert "stop searching and call `submit_prior_block`" in system_content

    def test_messages_for_scope_omit_literature_prompt_parts_when_disabled(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
            enable_literature=False,
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={},
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )

        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        assert "## Literature Evidence" not in system_content
        assert "search_literature" not in system_content
        assert "If you include non-empty `sources`" not in user_content

    def test_messages_for_scope_include_parameter_prior_cards(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress",),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": True,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "accepted prior",
                    }
                },
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### Parameter Prior Cards" in user_content
        assert "#### Current Accepted Priors" in user_content
        assert "Normal(mu=0.0, sigma=0.2)" in user_content
        assert "#### Fixed Effects" in user_content
        assert "| beta_stress_sleep | stress | sleep | lagged | 1.0 | yes | none |" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "## Scope Snapshot" in user_content
        assert "Use `submit_prior_block` with exactly this argument object:" in user_content

    def test_messages_for_scope_include_accepted_coupled_priors_outside_local_scope(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
                {
                    "parameter": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "stress"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={
                    "parameters": [
                        {"name": "beta_stress_sleep"},
                        {"name": "rho_stress"},
                    ]
                },
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "local accepted prior",
                    },
                    "rho_stress": {
                        "parameter": "rho_stress",
                        "distribution": "Beta",
                        "params": {"alpha": 3.0, "beta": 2.0},
                        "sources": [],
                        "reasoning": "coupled accepted prior",
                    },
                },
            ),
            last_validation_packet=make_stage4_validation_packet(
                status="prior_predictive_failure",
                feedback="PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED",
                validation=AssemblyValidation(
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="beta_stress_sleep",
                            is_valid=False,
                            code="scale_mismatch",
                            origin="prior_predictive",
                            issue="Effect prior mismatches the stress dynamics scale.",
                            suggested_adjustment="Reconcile the effect prior with stress persistence.",
                            related_parameters=["rho_stress"],
                        )
                    ],
                ),
                active_scope_id=block.id,
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "Accepted Coupled Priors Outside This Edit Scope" in user_content
        assert "Beta(alpha=3.0, beta=2.0)" in user_content
        assert "`rho_stress`" in user_content

    def test_messages_for_dynamics_scope_include_budget_discipline(self):
        block = Stage4FrontierBlock(
            id="dynamics:sleep",
            kind="dynamics_prior",
            label="Dynamics prior",
            construct_names=("sleep",),
            parameter_names=("rho_sleep", "sigma_sleep"),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                    "reference_indicator": "sleep_quality",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "rho_sleep",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "sleep"},
                },
                {
                    "parameter": "sigma_sleep",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "sleep"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={
                    "parameters": [
                        {"name": "rho_sleep"},
                        {"name": "sigma_sleep"},
                    ]
                }
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]
        system_content = messages[0]["content"]

        assert "clear damping headroom for later incoming effects" in user_content
        assert "## Dynamics Budget Discipline" in system_content
        assert "conservative decay" in system_content

    def test_messages_for_effect_scope_include_neighboring_topology_context(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Primary effect under review",
                    },
                    {
                        "cause": "mood",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Competing parent of sleep",
                    },
                ],
            },
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": False,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| stress | sleep | yes | Primary effect under review |" in user_content
        assert "| mood | sleep | yes | Competing parent of sleep |" in user_content

    def test_messages_for_narrowed_effect_repair_scope_do_not_reexpand_neighboring_topology(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior (internal SCC parameters only)",
            construct_names=("activity", "sleep"),
            parameter_names=("beta_activity_sleep",),
            expand_neighbor_topology=False,
        )
        msgs = Stage4Messages(
            question="Does activity affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Outside-SCC parent that should stay hidden",
                    },
                    {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Internal SCC edge under repair",
                    },
                    {
                        "cause": "sleep",
                        "effect": "activity",
                        "lagged": True,
                        "description": "Reciprocal SCC edge under repair",
                    },
                ],
            },
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_activity_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": True,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={"parameters": [{"name": "beta_activity_sleep"}]}
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| activity | sleep | yes | Internal SCC edge under repair |" in user_content
        assert "| sleep | activity | yes | Reciprocal SCC edge under repair |" in user_content
        assert "Outside-SCC parent that should stay hidden" not in user_content
        assert "| stress | sleep | yes |" not in user_content

    def test_messages_for_scope_respects_visible_sections_even_when_data_matches(self):
        block = Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Model review",
            construct_names=("stress",),
            parameter_names=("lambda_worry_stress",),
            payload={"reopenable_block_ids": ("indicator:worry_score",)},
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the worry column directly",
                    "options": [{"distribution": "gaussian", "links": ["identity"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[
                {
                    "name": "lambda_worry_stress",
                    "construct": "stress",
                    "indicator": "worry_score",
                    "constraint": "positive",
                }
            ],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "lambda_worry_stress",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "stress",
                        "indicator": "worry_score",
                        "reference_indicator": "pss_score",
                    },
                }
            ],
        )
        plan = _make_plan(review_block=block)
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### Loading Orientation" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "### Distribution Decision Cards" not in user_content
        assert "### Parameter Prior Cards" not in user_content

    def test_messages_for_scope_render_extended_profile_and_support_window_metadata(self):
        block = Stage4FrontierBlock(
            id="indicator:worry_score",
            kind="indicator_decision",
            label="Indicator decision",
            construct_names=("stress",),
            variable_names=("worry_score",),
            payload={
                "variable": "worry_score",
                "fixed_distribution": "bernoulli",
                "valid_links": ["logit", "probit"],
            },
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1w",
                    "how_to_measure": "Weekly worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {
                        "n_obs": 40,
                        "q50": 1.0,
                        "time_coverage_ratio": 0.75,
                        "max_gap_ratio": 2.5,
                        "duplicate_pct": 0.05,
                        "n_unparseable_timestamps": 1,
                    },
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "worry_score",
                            "measurement_dtype": "binary",
                            "aggregation": "mean",
                            "effective_window": "1w",
                            "is_reference": False,
                            "has_distribution_decision_card": True,
                            "profile": {
                                "n_obs": 40,
                                "q50": 1.0,
                                "time_coverage_ratio": 0.75,
                            },
                            "how_to_measure": "Weekly worry indicator",
                        }
                    ],
                }
            ],
            prior_cards=[],
        )
        plan = _make_plan(model_blocks=(block,))
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| worry_score | stress | binary | mean | 1w |" in user_content
        assert "q50=1" in user_content
        assert "coverage=75%" in user_content
        assert "max_gap=2.5x" in user_content
        assert "dups=5.0%" in user_content
        assert "bad_ts=1" in user_content

    def test_messages_for_scope_render_stateful_accepted_decisions(self):
        block = Stage4FrontierBlock(
            id="measurement:stress",
            kind="measurement_prior",
            label="Measurement prior",
            construct_names=("stress",),
            variable_names=("pss_score", "worry_score"),
            parameter_names=("lambda_worry_score_stress",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "effective_window": "1d",
                            "is_reference": True,
                            "has_distribution_decision_card": False,
                            "profile": {"n_obs": 40},
                            "how_to_measure": "Daily PSS score",
                        },
                        {
                            "indicator": "worry_score",
                            "measurement_dtype": "binary",
                            "aggregation": "mean",
                            "effective_window": "1d",
                            "is_reference": False,
                            "has_distribution_decision_card": True,
                            "profile": {"n_obs": 40},
                            "how_to_measure": "Daily worry indicator",
                        },
                    ],
                }
            ],
            prior_cards=[
                {
                    "parameter": "lambda_worry_score_stress",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "stress",
                        "indicator": "worry_score",
                        "reference_indicator": "pss_score",
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedState(
                model_spec={
                    "likelihoods": [
                        {
                            "variable": "pss_score",
                            "distribution": "gaussian",
                            "link": "identity",
                        },
                        {
                            "variable": "worry_score",
                            "distribution": "bernoulli",
                            "link": "probit",
                        },
                    ],
                    "parameters": [
                        {
                            "name": "lambda_worry_score_stress",
                            "role": "loading",
                            "constraint": "positive",
                        }
                    ],
                }
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "`bernoulli` / `probit`" in user_content
        assert (
            "| lambda_worry_score_stress | stress | worry_score | pss_score | positive |"
            in user_content
        )

    def test_stage4_turn_exposes_tools_by_block_kind(self):
        model_block = Stage4FrontierBlock(
            id="indicator:steps",
            kind="indicator_decision",
            label="Indicator decision",
            variable_names=("steps",),
            payload={
                "variable": "steps",
                "dtype": "count",
                "valid_distributions": ["negative_binomial", "poisson"],
                "link_options": {
                    "negative_binomial": ["log"],
                    "poisson": ["log"],
                },
            },
        )
        measurement_block = Stage4FrontierBlock(
            id="measurement:activity",
            kind="measurement_prior",
            label="Measurement prior",
            parameter_names=("lambda_steps_activity",),
            construct_names=("activity",),
            variable_names=("steps",),
        )
        prior_block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            parameter_names=("beta_activity_sleep",),
        )
        plan = _make_plan(
            model_blocks=(model_block,),
            prior_blocks=(measurement_block, prior_block),
        )
        runtime = _make_runtime(plan)
        session = _make_stage4_session(
            question="Does activity improve sleep?",
            plan=plan,
            runtime=runtime,
            skeleton=Stage4Skeleton(),
            causal_spec={},
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail("grounding should not run"),
            prior_cards=[
                {
                    "parameter": "lambda_steps_activity",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "activity",
                        "indicator": "steps",
                        "reference_indicator": "steps_ref",
                    },
                },
                {
                    "parameter": "beta_activity_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
            ],
            enable_literature=True,
            enable_paraphrasing=True,
        )
        tools = [
            SimpleNamespace(name="submit_indicator_choice"),
            SimpleNamespace(name="submit_prior_block"),
            SimpleNamespace(name="search_literature"),
            SimpleNamespace(name="elicit_prior_gmm"),
        ]
        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_indicator_choice"
        ]

        runtime.accepted.model_spec = {
            "parameters": [
                {"name": "lambda_steps_activity", "role": "loading"},
                {"name": "beta_activity_sleep"},
            ]
        }
        _set_runtime_block(plan, runtime, "measurement:activity")

        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_prior_block",
            "elicit_prior_gmm",
        ]

        _set_runtime_block(plan, runtime, "effects:sleep")

        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_prior_block",
            "search_literature",
            "elicit_prior_gmm",
        ]


def test_stage4_generate_config_sets_stage4_timeout(monkeypatch):
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.flow.get_generate_config",
        lambda: GenerateConfig(
            max_tokens=65536,
            timeout=321,
            reasoning_effort="high",
            max_tool_output=1234,
        ),
    )

    config = _stage4_generate_config()

    assert config.max_tokens is None
    assert config.timeout == 180
    assert config.reasoning_effort == "high"
    assert config.max_tool_output is None


# --- SSMModelBuilder Tests ---


class TestSSMModelBuilder:
    """Test SSM model building."""

    def test_builder_builds_model(self, simple_model_spec, simple_priors, simple_data):
        """Builder creates an SSMModel with correct dimensions."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=simple_model_spec,
            priors=simple_priors,
        )
        model = builder.build_model(simple_data)
        assert model.spec.n_manifest == 1  # mood_score only
        assert model.spec.n_latent >= 1
        assert model.likelihood is not None
        # Lambda should map latent to manifest
        assert model.spec.lambda_mat.shape == (model.spec.n_manifest, model.spec.n_latent)


# --- Prior Predictive Validation Tests ---


def _make_polars_data() -> pl.DataFrame:
    """Create polars long-format data for validation tests."""
    rng = np.random.default_rng(42)
    n = 30
    anchor_times = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%dT00:00:00Z")
    return pl.DataFrame(
        {
            "indicator": ["mood_score"] * n,
            "value": (rng.standard_normal(n) * 1.5 + 5).tolist(),
            "anchor_time": anchor_times,
            "support_start": anchor_times,
            "support_end": anchor_times,
            "support_kind": ["point"] * n,
            "summary_operator": ["last"] * n,
            "anchor_policy": ["support_end"] * n,
            "observation_window": [None] * n,
        }
    )


def _make_stage4_mechanics_spec() -> dict:
    """Stage 4 spec with an ambiguous indicator, loading block, and effect prior."""
    constructs = [
        {
            "name": "activity",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    edges = [{"cause": "activity", "effect": "sleep"}]
    indicators = [
        {
            "name": "steps",
            "construct_name": "activity",
            "measurement_dtype": "count",
            "how_to_measure": "Daily step count",
            "aggregation": "sum",
        },
        {
            "name": "activity_vas",
            "construct_name": "activity",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Activity visual analog scale",
            "aggregation": "mean",
        },
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {"constructs": constructs, "edges": edges},
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": [construct["name"] for construct in constructs],
                "edges": edges,
                "induced_dependencies": [],
            },
        }
    )


def _make_stage4_global_repair_spec() -> dict:
    """Stage 4 spec with a correlation block and separate dynamics blocks."""
    constructs = [
        {
            "name": "activity",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    indicators = [
        {
            "name": "activity_vas",
            "construct_name": "activity",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Activity visual analog scale",
            "aggregation": "mean",
        },
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": constructs,
                "edges": [{"cause": "activity", "effect": "sleep"}],
            },
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["activity", "sleep"],
                "edges": [{"cause": "activity", "effect": "sleep"}],
                "induced_dependencies": [
                    {
                        "between": ["activity", "sleep"],
                        "kind": "initial_state_correlation",
                        "source_confounders": ["U"],
                    }
                ],
            },
        }
    )


def _make_stage4_two_effect_spec() -> dict:
    """Stage 4 spec with two sequential effect blocks."""
    constructs = [
        {
            "name": "activity",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "mood",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    edges = [
        {"cause": "activity", "effect": "sleep"},
        {"cause": "activity", "effect": "mood"},
    ]
    indicators = [
        {
            "name": "steps",
            "construct_name": "activity",
            "measurement_dtype": "count",
            "how_to_measure": "Daily step count",
            "aggregation": "sum",
        },
        {
            "name": "activity_vas",
            "construct_name": "activity",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Activity visual analog scale",
            "aggregation": "mean",
        },
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
        {
            "name": "mood_rating",
            "construct_name": "mood",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Mood rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {"constructs": constructs, "edges": edges},
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["activity", "sleep", "mood"],
                "edges": edges,
                "induced_dependencies": [],
            },
        }
    )


def _make_stage4_no_model_block_spec() -> dict:
    """Stage 4 spec whose reducer starts directly in the prior phase."""
    constructs = [
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    indicators = [
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {"constructs": constructs, "edges": []},
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["sleep"],
                "edges": [],
                "induced_dependencies": [],
            },
        }
    )


def _make_stage4_mechanics_context() -> tuple[
    dict[str, Any], Stage4Skeleton, Stage4Plan, Stage4Runtime, pl.DataFrame
]:
    """Build the standard deterministic Stage 4 mechanics fixture."""
    causal_spec = _make_stage4_mechanics_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    return causal_spec, skeleton, plan, runtime, pl.DataFrame()


def _make_stage4_deps(
    *,
    causal_spec: dict[str, Any],
    skeleton: Stage4Skeleton,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    stage4_grounding_fn,
) -> Stage4Deps:
    """Build a Stage 4 reducer environment for tests."""

    def _wrap_grounding(*args, **kwargs):
        result = stage4_grounding_fn(*args, **kwargs)
        if isinstance(result, Stage4GroundingResult):
            return result
        stage_output, feedback = result
        validation = stage_output.get("validation") if isinstance(stage_output, dict) else None
        status = (
            "compile_error"
            if validation is not None and getattr(validation, "compile_ok", True) is False
            else "prior_predictive_failure"
            if validation is not None
            and getattr(validation, "pp_checked", False)
            and getattr(validation, "pp_valid", True) is False
            else "accepted"
        )
        return make_stage4_grounding_result(
            stage_output=stage_output,
            status=status,
            feedback=feedback,
            validation=validation,
            retain_for_next_prompt=feedback != "VALID",
            capture_stage_output=stage_output is not None and status == "accepted",
        )

    return Stage4Deps(
        skeleton=skeleton,
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        indicator_audits=indicator_audits,
        grounding_fn=_wrap_grounding,
    )


def _make_stage4_session(
    *,
    question: str,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    skeleton: Stage4Skeleton,
    causal_spec: dict[str, Any],
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    stage4_grounding_fn,
    model_topology: dict[str, Any] | None = None,
    distribution_cards: list[dict[str, Any]] | None = None,
    loading_params: list[dict[str, Any]] | None = None,
    construct_scale_cards: list[dict[str, Any]] | None = None,
    prior_cards: list[dict[str, Any]] | None = None,
    enable_literature: bool = False,
    enable_paraphrasing: bool = False,
) -> Stage4Session:
    """Build a Stage 4 session for tests."""
    return Stage4Session(
        plan=plan,
        prompt_context=Stage4Messages(
            question=question,
            model_topology=model_topology or {},
            distribution_cards=distribution_cards or [],
            loading_params=loading_params or [],
            construct_scale_cards=construct_scale_cards or [],
            prior_cards=prior_cards or [],
            enable_literature=enable_literature,
            enable_paraphrasing=enable_paraphrasing,
        ),
        deps=_make_stage4_deps(
            causal_spec=causal_spec,
            skeleton=skeleton,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            stage4_grounding_fn=stage4_grounding_fn,
        ),
        runtime=runtime,
    )


def _apply_stage4_step_and_capture(
    payload: dict[str, Any],
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    *,
    skeleton: Stage4Skeleton,
    causal_spec: dict[str, Any],
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    stage4_grounding_fn,
) -> tuple[dict | None, str]:
    """Run one reducer step."""
    return compute_stage4_validate_step(
        _stage4_test_payload(payload),
        plan=plan,
        runtime=runtime,
        deps=_make_stage4_deps(
            causal_spec=causal_spec,
            skeleton=skeleton,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            stage4_grounding_fn=stage4_grounding_fn,
        ),
    )


def _require_plan_block(plan: Stage4Plan, block_id: str) -> Stage4FrontierBlock:
    """Fetch a Stage 4 block and assert the test fixture contains it."""
    block = plan.get_block(block_id)
    assert block is not None
    return block


def _require_active_plan_block(plan: Stage4Plan, runtime: Stage4Runtime) -> Stage4FrontierBlock:
    """Fetch the active block and assert the runtime is currently promptable."""
    block = get_active_plan_block(plan, runtime)
    assert block is not None
    return block


def _current_stage4_state(current: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional grounding state payloads for test stubs."""
    return {} if current is None else current


def _current_model_spec(current: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the current model spec when the grounding state carries one."""
    model_spec = _current_stage4_state(current).get("model_spec")
    return model_spec if isinstance(model_spec, dict) else None


def _merge_current_authored_priors(
    current: dict[str, Any] | None,
    priors: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Merge newly proposed priors onto the current accepted prior state."""
    current_state = _current_stage4_state(current)
    authored_priors = dict(current_state.get("authored_priors") or {})
    authored_priors.update(priors)
    return authored_priors, _current_model_spec(current_state)


def _require_text(value: str | None) -> str:
    """Assert an optional diagnostic field is present before string matching."""
    assert value is not None
    return value


async def _await_string(awaitable) -> str:
    """Bridge `Awaitable[str]` helpers into `asyncio.run()` in tests."""
    return await awaitable


def _require_trace(trace_capture: dict[str, object]) -> LLMTrace:
    """Extract the accumulated trace from a generate capture."""
    trace = trace_capture["trace"]
    assert isinstance(trace, LLMTrace)
    return trace


def _stage4_submit_tool_name(block_kind: str) -> str:
    """Return the primary submit-tool name for one Stage 4 block kind."""
    if block_kind == "indicator_decision":
        return "submit_indicator_choice"
    if block_kind == "global_review":
        return "submit_model_review"
    return "submit_prior_block"


def _stage4_submit_tool_args(submission: dict[str, object]) -> dict[str, object]:
    """Extract the direct tool arguments from a legacy block-keyed submission fixture."""
    proposal = submission.get("proposal")
    assert isinstance(proposal, dict)
    normalized: dict[str, object] = {}
    for key, value in proposal.items():
        assert isinstance(key, str)
        normalized[key] = value
    return normalized


def _stage4_test_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy block-envelope fixtures into reducer payloads."""
    proposal = payload.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    return payload


def _make_scripted_stage4_generate(
    submissions: list[dict[str, object]],
    *,
    visited_blocks: list[str],
    visible_tools: list[list[str]],
):
    """Drive ``run_stage4()`` with scripted block-local submit-tool calls only."""
    turn_index = 0

    async def _generate(messages, tools, rewrite_messages=None, rewrite_tools=None, label=None):
        nonlocal turn_index
        del messages, rewrite_messages, rewrite_tools
        block_id = None
        if isinstance(label, str) and label.startswith("stage-4:"):
            block_id = label.removeprefix("stage-4:")
        if turn_index >= len(submissions):
            raise AssertionError(
                f"Unexpected extra Stage 4 turn at {block_id!r}; visited={visited_blocks}"
            )
        submission = submissions[turn_index]
        turn_index += 1
        visited_blocks.append(block_id or str(submission["block_id"]))
        visible_tools.append([tool.name for tool in tools])
        submit_tool_name = _stage4_submit_tool_name(str(submission["block_kind"]))
        submit_tool = next(tool for tool in tools if tool.name == submit_tool_name)
        feedback = await submit_tool(**_stage4_submit_tool_args(submission))
        assert isinstance(feedback, str)
        if feedback.startswith("VALIDATION ERRORS:"):
            raise AssertionError(feedback)
        return ""

    return _generate


def _make_scripted_stage4_generate_by_block(
    submissions_by_block: dict[str, dict[str, object]],
    *,
    visited_blocks: list[str],
    visible_tools: list[list[str]],
):
    """Drive ``run_stage4()`` with block-keyed scripted submissions."""

    async def _generate(messages, tools, rewrite_messages=None, rewrite_tools=None, label=None):
        del messages, rewrite_messages, rewrite_tools
        assert label is not None
        assert label.startswith("stage-4:")
        block_id = label.removeprefix("stage-4:")
        submission = submissions_by_block[block_id]
        visited_blocks.append(block_id)
        visible_tools.append([tool.name for tool in tools])
        submit_tool_name = _stage4_submit_tool_name(str(submission["block_kind"]))
        submit_tool = next(tool for tool in tools if tool.name == submit_tool_name)
        feedback = await submit_tool(**_stage4_submit_tool_args(submission))
        assert isinstance(feedback, str)
        if feedback.startswith("VALIDATION ERRORS:"):
            raise AssertionError(f"{block_id}: {feedback}")
        return ""

    return _generate


class TestPriorPredictiveValidation:
    """Test prior predictive validation end-to-end."""

    def test_valid_priors_pass(self, simple_model_spec, simple_priors):
        """Simple spec + priors + polars data -> is_valid=True with all checks passing."""
        data_for_model = _make_polars_data()
        is_valid, results, _samples = validate_prior_predictive(
            simple_model_spec, simple_priors, data_for_model, n_samples=10
        )
        assert is_valid is True
        assert len(results) > 0
        assert all(r.is_valid for r in results), (
            f"Expected all checks to pass but got failures: "
            f"{[(r.parameter, r.issue) for r in results if not r.is_valid]}"
        )

    def test_model_build_failure(self):
        """Broken spec -> is_valid=False, error in results."""
        broken_spec = {
            "likelihoods": [
                {
                    "variable": "nonexistent_col",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coeff",
                }
            ],
        }
        broken_priors = {
            "rho_x": {
                "parameter": "rho_x",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            }
        }
        # This should still build (builder is tolerant), but let's test
        # with a truly broken spec by patching build_model to raise
        with patch(
            "causal_ssm_agent.models.ssm_builder.SSMModelBuilder.build_model",
            side_effect=ValueError("deliberate test failure"),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                broken_spec, broken_priors, None, n_samples=10
            )
            assert is_valid is False
            assert any("model_build" in r.parameter for r in results)
            assert any("deliberate test failure" in (r.issue or "") for r in results)

    def test_no_data_uses_support_compatible_dummy_build_data(self):
        """Support-restricted likelihoods should still validate without raw data."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "screen_gap",
                    "distribution": "gamma",
                    "link": "log",
                    "reasoning": "Positive continuous gap",
                }
            ],
            "parameters": [
                {
                    "name": "rho_screen_gap",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                }
            ],
        }
        priors = {
            "rho_screen_gap": {
                "parameter": "rho_screen_gap",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "Weakly informative",
            }
        }

        with patch(
            "causal_ssm_agent.models.ssm_builder.SSMModelBuilder.sample_prior_predictive",
            return_value={"drift_diag_free": np.ones((2, 1))},
        ):
            is_valid, results, _samples = validate_prior_predictive(
                model_spec, priors, None, n_samples=2
            )

        assert is_valid is True
        assert not any(r.parameter == "model_build" for r in results)

    def test_build_validation_payload_from_assembly(self, simple_model_spec, simple_priors):
        """Shared Stage 4 assembly helpers return the expected payload shape."""
        from causal_ssm_agent.flows.stages.stage4.assembly import (
            build_validation_payload,
            validate_assembly,
        )

        data_for_model = _make_polars_data()
        validation = validate_assembly(simple_model_spec, simple_priors, data_for_model, None, None)
        result = build_validation_payload(validation, simple_model_spec)
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["results"], list)
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)
        # Issues must be strings, each describing a validation failure
        for issue in result["issues"]:
            assert isinstance(issue, str)
        for warning in result["warnings"]:
            assert isinstance(warning, str)

    def test_materialize_stage4_result_persists_validation_warnings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Final Stage 4 artifacts should carry non-fatal validation warnings."""
        from causal_ssm_agent.flows.stages.stage4.assembly import (
            AssemblyValidation,
            materialize_stage4_result,
        )

        validation = AssemblyValidation(
            normalized_model_spec=simple_model_spec,
            compile_ok=True,
            diagnostics=[
                PriorValidationResult(
                    parameter="beta_stress_sleep",
                    is_valid=True,
                    code="interval_reference_missing",
                    origin="compile",
                    severity="warning",
                    issue="Weekly evidence is being interpreted on the daily model interval.",
                    suggested_adjustment=(
                        "Set `reference_interval_days` if that weekly interval is intended."
                    ),
                )
            ],
            compiled_ssm={"compiled_prior_semantics": {}, "parameter_bindings": []},
            pp_checked=True,
            pp_valid=True,
            pp_raw_samples={},
        )

        with (
            patch(
                "causal_ssm_agent.flows.stages.stage4.assembly.compile_model_artifact",
                return_value={
                    "model_built": True,
                    "model_type": "test",
                    "version": "0",
                    "compiled_ssm": {"compiled_prior_semantics": {}, "parameter_bindings": []},
                },
            ),
            patch(
                "causal_ssm_agent.models.ssm_compiler.resolve_prior_proposals",
                return_value=[],
            ),
        ):
            result = materialize_stage4_result(
                model_spec=simple_model_spec,
                authored_priors=simple_priors,
                data_for_model=_make_polars_data(),
                indicator_audits=None,
                causal_spec=None,
                validation=validation,
            )

        assert result["validation_warnings"] == [
            "Weekly evidence is being interpreted on the daily model interval."
        ]

    def test_validate_assembly_reuses_compiled_artifact_for_prior_checks(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Stage 4 should compile once per validation attempt and pass that artifact through."""
        from causal_ssm_agent.flows.stages.stage4.assembly import validate_assembly

        compiled_artifact = {"schema_version": 1}
        seen_compiled: list[dict[str, Any] | None] = []

        def stub_validate_prior_predictive(*args, compiled_ssm=None, **kwargs):
            seen_compiled.append(compiled_ssm)
            return True, [], {}

        with (
            patch(
                "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
                return_value=compiled_artifact,
            ) as compile_mock,
            patch(
                "causal_ssm_agent.models.prior_predictive.validate_prior_predictive",
                side_effect=stub_validate_prior_predictive,
            ),
        ):
            validation = validate_assembly(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                None,
                None,
            )

        assert compile_mock.call_count == 1
        assert seen_compiled == [compiled_artifact]
        assert validation.compiled_ssm == compiled_artifact

    def test_validate_assembly_keeps_lagged_prior_mismatches_as_warnings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Lagged DT/CT heuristics should surface as warnings, not compile errors."""
        from causal_ssm_agent.flows.stages.stage4.assembly import validate_assembly

        compiled_artifact = {
            "schema_version": 1,
            "spec": {},
            "compiled_prior_semantics": {},
        }

        with (
            patch(
                "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
                return_value=compiled_artifact,
            ),
            patch(
                "causal_ssm_agent.flows.stages.stage4.assembly._collect_compile_diagnostics",
                return_value=[
                    PriorValidationResult(
                        parameter="beta_stress_sleep",
                        is_valid=True,
                        code="lagged_response_weak",
                        origin="compile",
                        severity="warning",
                        issue="Median one-lag response is much slower than the nominal lag.",
                        suggested_adjustment="Confirm that this slow response is intended.",
                    )
                ],
            ),
            patch(
                "causal_ssm_agent.models.prior_predictive.validate_prior_predictive",
                return_value=(True, [], {}),
            ) as pp_mock,
        ):
            validation = validate_assembly(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                None,
                {"measurement": {"model_clock": "1d"}},
            )

        assert validation.compile_ok is True
        assert validation.pp_valid is True
        assert [
            warning.model_dump()
            for warning in validation.compile_diagnostics
            if warning.severity == "warning"
        ] == [
            PriorValidationResult(
                parameter="beta_stress_sleep",
                is_valid=True,
                code="lagged_response_weak",
                origin="compile",
                severity="warning",
                issue="Median one-lag response is much slower than the nominal lag.",
                suggested_adjustment="Confirm that this slow response is intended.",
            ).model_dump()
        ]
        pp_mock.assert_called_once()

    def test_validate_prior_predictive_skips_recompile_when_artifact_provided(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Explicit compiled_ssm should bypass compile_ssm_artifact entirely."""

        class _DummyBuilder:
            def sample_prior_predictive(self, samples: int = 500):
                return {
                    "drift_diag_free": np.ones((samples, 1)),
                    "observations": np.random.default_rng(0).normal(
                        loc=5.0,
                        scale=1.5,
                        size=(samples, 4, 1),
                    ),
                }

        runtime = SimpleNamespace(builder=_DummyBuilder())

        with (
            patch(
                "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
                side_effect=AssertionError("compile should not be called"),
            ),
            patch(
                "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
                return_value=runtime,
            ),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                n_samples=3,
                compiled_ssm={"schema_version": 1},
            )

        assert is_valid is True
        assert results

    def test_validate_prior_predictive_reports_log_link_mean_overflow(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Prior predictive should surface predictive-mean overflow as a typed diagnostic."""

        class _OverflowBuilder:
            def sample_prior_predictive(self, samples: int = 500):
                del samples
                raise PredictiveObservationMeanOverflow(
                    bad_manifest_names=("monthly_eveningness_activity_timing",),
                    manifest_indices=(0,),
                    failing_draw_indices=(0, 2),
                    first_bad_time_index=73,
                    max_linear_predictor=111.52,
                    overflow_threshold=88.72,
                )

        runtime = SimpleNamespace(builder=_OverflowBuilder())

        with patch(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            return_value=runtime,
        ):
            is_valid, results, _samples = validate_prior_predictive(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                n_samples=3,
                compiled_ssm={"schema_version": 1, "compile_diagnostics": []},
            )

        assert is_valid is False
        assert [result.code for result in results if not result.is_valid] == [
            "prior_predictive_observation_mean_overflow"
        ]
        result = next(result for result in results if not result.is_valid)
        assert result.failure_stage == "observation_mean"
        assert result.bad_manifest_names == ["monthly_eveningness_activity_timing"]
        assert result.failing_draw_indices == [0, 2]
        assert result.first_bad_time_index == 73

    def test_resolve_prior_proposals_reads_compiled_semantics_per_state(self):
        """Implicit initial-state priors should come from compiled semantics."""
        from causal_ssm_agent.models.ssm_compiler import resolve_prior_proposals

        compiled_ssm = {
            "spec": {"latent_names": ["stress", "sleep"]},
            "compiled_prior_semantics": {
                "schema_version": 4,
                "site_registry": [
                    {
                        "name": "t0_means_free",
                        "shape": [2],
                        "support": "real",
                        "assembly_group": "t0",
                        "site_kind": "t0_means",
                        "transform_kind": "identity",
                        "deterministic_name": "t0_means",
                        "fixed_spec_field": "t0_means",
                        "priors_field": "t0_means",
                        "runtime_prior_key": "t0_means_free",
                        "is_runtime_prior_controlled": True,
                    },
                    {
                        "name": "t0_var_diag_free",
                        "shape": [2],
                        "support": "positive",
                        "assembly_group": "t0",
                        "site_kind": "t0_var_diag",
                        "transform_kind": "exp",
                        "deterministic_name": "t0_cov",
                        "fixed_spec_field": "t0_var",
                        "priors_field": "t0_var_diag",
                        "runtime_prior_key": "t0_var_diag_free",
                        "is_runtime_prior_controlled": True,
                    },
                ],
                "prior_state": {
                    "t0_means_free": {"family": 0, "loc": [0.0, 1.0], "scale": [2.0, 3.0]},
                    "t0_var_diag_free": {
                        "family": 0,
                        "scale": [4.0, 5.0],
                        "concentration": [1.0, 1.0],
                        "rate": [1.0, 1.0],
                    },
                },
            },
            "parameter_bindings": [],
        }

        assert resolve_prior_proposals(compiled_ssm, authored_priors={}) == [
            {
                "parameter": "t0_mean_stress",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 2.0},
                "sources": [],
                "reasoning": "Default weakly informative prior for the initial state mean of stress.",
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_mean_sleep",
                "distribution": "Normal",
                "params": {"mu": 1.0, "sigma": 3.0},
                "sources": [],
                "reasoning": "Default weakly informative prior for the initial state mean of sleep.",
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_sd_stress",
                "distribution": "HalfNormal",
                "params": {"sigma": 4.0},
                "sources": [],
                "reasoning": (
                    "Default weakly informative prior for the initial state standard deviation "
                    "of stress."
                ),
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_sd_sleep",
                "distribution": "HalfNormal",
                "params": {"sigma": 5.0},
                "sources": [],
                "reasoning": (
                    "Default weakly informative prior for the initial state standard deviation "
                    "of sleep."
                ),
                "reference_interval_days": None,
                "density_points": None,
            },
        ]

    def test_resolve_prior_proposals_preserves_authored_metadata_for_lossy_bindings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Resolved public priors should retain authored semantics when compilation is lossy."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            resolve_prior_proposals,
        )

        compiled_ssm = compile_ssm_artifact(simple_model_spec, simple_priors)
        resolved = {
            prior["parameter"]: prior
            for prior in resolve_prior_proposals(compiled_ssm, authored_priors=simple_priors)
        }

        assert resolved["rho_mood"]["distribution"] == "Beta"
        assert resolved["rho_mood"]["params"] == {"alpha": 2.0, "beta": 2.0}
        assert resolved["rho_mood"]["reasoning"] == "Weakly informative for AR coefficient"
        assert resolved["sigma_mood"]["distribution"] == "HalfNormal"
        assert resolved["sigma_mood"]["params"] == {"sigma": 1.0}

    def test_resolve_prior_proposals_roundtrips_new_supported_prior_families(self):
        """Compiled semantics should surface LogNormal and bounded real priors."""
        from causal_ssm_agent.models.ssm_compiler import resolve_prior_proposals

        compiled_ssm = {
            "compiled_prior_semantics": {
                "schema_version": 4,
                "site_registry": [
                    {
                        "name": "diffusion_diag_free",
                        "shape": [1],
                        "support": "positive",
                        "assembly_group": "diffusion",
                        "site_kind": "diffusion_diag",
                        "transform_kind": "exp",
                        "deterministic_name": "diffusion",
                        "fixed_spec_field": "diffusion",
                        "priors_field": "diffusion_diag",
                        "runtime_prior_key": "diffusion_diag_free",
                        "is_runtime_prior_controlled": True,
                    },
                    {
                        "name": "drift_offdiag_free",
                        "shape": [1],
                        "support": "real",
                        "assembly_group": "drift",
                        "site_kind": "drift_offdiag",
                        "transform_kind": "identity",
                        "deterministic_name": "drift",
                        "fixed_spec_field": "drift",
                        "priors_field": "drift_offdiag",
                        "runtime_prior_key": "drift_offdiag_free",
                        "is_runtime_prior_controlled": True,
                    },
                ],
                "prior_state": {
                    "diffusion_diag_free": {
                        "family": [2],
                        "loc": [0.2],
                        "scale": [0.7],
                        "concentration": [1.0],
                        "rate": [1.0],
                    },
                    "drift_offdiag_free": {
                        "family": 2,
                        "loc": [0.0],
                        "scale": [0.3],
                        "low": [-1.0],
                        "high": [1.0],
                    },
                },
            },
            "parameter_bindings": [
                {"parameter": "sigma_mood", "site_name": "diffusion_diag_free", "flat_index": 0},
                {
                    "parameter": "cor_stress_sleep",
                    "site_name": "drift_offdiag_free",
                    "flat_index": 0,
                },
            ],
        }

        resolved = {
            prior["parameter"]: prior
            for prior in resolve_prior_proposals(compiled_ssm, authored_priors={})
        }
        assert resolved["sigma_mood"]["distribution"] == "LogNormal"
        assert resolved["sigma_mood"]["params"]["mu"] == pytest.approx(0.2)
        assert resolved["sigma_mood"]["params"]["sigma"] == pytest.approx(0.7)
        assert resolved["cor_stress_sleep"]["distribution"] == "Uniform"
        assert resolved["cor_stress_sleep"]["params"]["lower"] == pytest.approx(-1.0)
        assert resolved["cor_stress_sleep"]["params"]["upper"] == pytest.approx(1.0)

    def test_resolve_prior_proposals_roundtrips_correlation_support_sites(self):
        """Compiled correlation-support sites should reconstruct bounded real priors."""
        from causal_ssm_agent.models.ssm_compiler import resolve_prior_proposals

        compiled_ssm = {
            "compiled_prior_semantics": {
                "schema_version": 4,
                "site_registry": [
                    {
                        "name": "t0_var_lower_free",
                        "shape": [1],
                        "support": "correlation",
                        "assembly_group": "t0",
                        "site_kind": "t0_var_lower",
                        "transform_kind": "identity",
                        "deterministic_name": "t0_cov",
                        "fixed_spec_field": "t0_var",
                        "priors_field": "t0_var_offdiag",
                        "runtime_prior_key": "t0_var_lower_free",
                        "is_runtime_prior_controlled": True,
                    }
                ],
                "prior_state": {
                    "t0_var_lower_free": {
                        "family": [2],
                        "loc": [0.0],
                        "scale": [0.25],
                        "low": [-1.0],
                        "high": [1.0],
                    }
                },
            },
            "parameter_bindings": [
                {
                    "parameter": "cor0_sleep_stress",
                    "site_name": "t0_var_lower_free",
                    "flat_index": 0,
                }
            ],
        }

        resolved = resolve_prior_proposals(compiled_ssm, authored_priors={})

        assert resolved == [
            {
                "parameter": "cor0_sleep_stress",
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 1.0},
                "sources": [],
                "reasoning": "Compiler-resolved prior for cor0_sleep_stress.",
                "reference_interval_days": None,
                "density_points": None,
            }
        ]


class TestFailedParameters:
    """Test failed parameter identification."""

    def test_scale_mismatch_with_causal_spec_targets_construct(self):
        """Scale mismatch with causal_spec targets only the affected construct."""
        results = [
            PriorValidationResult(
                parameter="scale_mood_score",
                is_valid=False,
                issue="Scale mismatch for mood_score",
                suggested_adjustment=None,
            ),
        ]
        causal_spec = _with_positive_indicator_polarity(
            {
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {"name": "mood_score", "construct_name": "mood"},
                        {"name": "stress_score", "construct_name": "stress"},
                    ],
                },
            }
        )
        all_params = ["rho_mood", "sigma_mood", "rho_stress", "sigma_stress", "beta_stress_mood"]
        failed = get_failed_parameters(results, all_params, causal_spec=causal_spec)
        # Only mood-related params should be re-elicited
        assert "rho_mood" in failed
        assert "sigma_mood" in failed
        assert "beta_stress_mood" in failed  # contains "mood"
        assert "rho_stress" not in failed
        assert "sigma_stress" not in failed


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_prior_converts_to_mu_sigma(self, simple_model_spec):
        """Beta(2,2) AR prior converts via AR-to-drift transform."""
        import math

        priors = {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_spec = make_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])
        ssm_priors, _idx, _diagnostics = compile_ssm_priors(
            priors,
            simple_model_spec,
            ssm_spec=ssm_spec,
        )

        # Beta(2,2): E[X] = 0.5 → drift mu = -ln(0.5)/1.0 ≈ 0.693
        # Per-element with 1 entry: mu is a list [0.693]
        expected_mu = -math.log(0.5) / 1.0
        mu = ssm_priors.drift_diag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.01
        sigma = ssm_priors.drift_diag["sigma"]
        sigma_val = sigma[0] if isinstance(sigma, list) else sigma
        assert sigma_val > 0.4  # delta method sigma

    def test_structured_prior_requires_structural_binding_for_residual_sd(self, simple_model_spec):
        """Structured priors should fail without a translated SSM binding."""
        priors = {
            "sigma_mood": {
                "parameter": "sigma_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(priors, simple_model_spec, ssm_spec=None)

    def test_compile_ssm_inputs_validates_dict_once(self, simple_model_spec, simple_priors):
        """Compilation should validate a dict spec once, then pass the parsed object through."""
        from causal_ssm_agent.artifacts import ModelSpec

        with patch.object(ModelSpec, "model_validate", wraps=ModelSpec.model_validate) as validate:
            compile_ssm_inputs_from_model_spec(simple_model_spec, simple_priors)

        assert validate.call_count == 1

    def test_structured_prior_requires_structural_binding_for_loading(self, simple_model_spec):
        """Loading priors should fail without a translated SSM binding."""
        spec = dict(simple_model_spec)
        spec["parameters"] = [
            {
                "name": "lambda_mood",
                "role": "loading",
                "constraint": "positive",
                "description": "Factor loading",
            },
        ]
        priors = {
            "lambda_mood": {
                "parameter": "lambda_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.8},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(priors, spec, ssm_spec=None)

    def test_unbound_prior_name_fails_without_model_spec(self):
        """Semantic prior compilation should fail fast when model_spec is missing."""
        priors = {
            "rho_x": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.5},
            },
        }
        with pytest.raises(ValueError, match="requires model_spec"):
            compile_ssm_priors(priors, {}, ssm_spec=None)

    def test_compile_priors_aggregates_independent_prior_errors(self):
        """Independent prior compile failures should be reported together."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                }
            ],
        }
        priors = {
            "rho_mood": {
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 2.0},
            },
            "bogus_param": {
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 1.0},
            },
        }
        ssm_spec = make_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match="Prior compilation failed") as exc_info:
            compile_ssm_priors(priors, model_spec, ssm_spec=ssm_spec)

        message = str(exc_info.value)
        assert "Prior compilation failed" in message
        assert "lower bound is -1" in message
        assert "upper bound is 2" in message
        assert "bogus_param" in message

    def test_compile_ssm_artifact_aggregates_strict_binding_errors(self):
        """Strict causal-spec binding errors should be reported together."""
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "mood",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                        {
                            "name": "stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "stress", "effect": "mood"}],
                },
                "estimation": {
                    "state_order": ["mood", "stress"],
                    "edges": [{"cause": "stress", "effect": "mood"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "mood_score",
                            "construct_name": "mood",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use mood_score directly",
                        },
                        {
                            "name": "stress_score",
                            "construct_name": "stress",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use stress_score directly",
                        },
                    ],
                },
            }
        )
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
            ],
            "parameters": [
                {
                    "name": "rho_affect",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Invalid AR name",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Valid AR name",
                },
                {
                    "name": "beta_mood_stress",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "Wrong causal direction",
                },
            ],
        }
        priors = {
            "rho_affect": {
                "parameter": "rho_affect",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "rho_stress": {
                "parameter": "rho_stress",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "beta_mood_stress": {
                "parameter": "beta_mood_stress",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }

        with pytest.raises(ValueError, match="Prior index binding failed") as exc_info:
            compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)

        message = str(exc_info.value)
        assert "Prior index binding failed" in message
        assert "rho_affect" in message
        assert "beta_mood_stress" in message

    def test_multiple_ar_params_produce_per_element_drift_diag(self):
        """Multiple AR params map to separate drift_diag array entries."""
        import math

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 5.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 5.0}},
        }
        ssm_spec = make_ssm_spec(n_latent=2, n_manifest=2, latent_names=["mood", "stress"])
        ssm_priors, _idx, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
        )

        # Both should produce per-element arrays (lists), not scalars
        assert isinstance(ssm_priors.drift_diag["mu"], list)
        assert len(ssm_priors.drift_diag["mu"]) == 2

        # Beta(5,2) → E=5/7≈0.714, Beta(2,5) → E=2/7≈0.286
        mu_ar_mood = 5.0 / 7.0
        mu_ar_stress = 2.0 / 7.0
        expected_mood = -math.log(mu_ar_mood) / 1.0
        expected_stress = -math.log(mu_ar_stress) / 1.0
        assert abs(ssm_priors.drift_diag["mu"][0] - expected_mood) < 0.01
        assert abs(ssm_priors.drift_diag["mu"][1] - expected_stress) < 0.01

    def test_ar_transform_respects_granularity(self):
        """Hourly construct → dt=1/24, producing larger drift magnitude."""
        import math

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [],
                },
                "measurement": {"model_clock": "1h", "indicators": []},
            }
        )
        ssm_spec = make_ssm_spec(n_latent=1, n_manifest=1, latent_names=["heart_rate"])
        ssm_priors, _idx, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Beta(2,2) → E=0.5; hourly dt = 1/24
        # drift mu = -ln(0.5) / (1/24) = 0.693 * 24 ≈ 16.64
        dt_hourly = 1.0 / 24.0
        expected_mu = -math.log(0.5) / dt_hourly
        mu = ssm_priors.drift_diag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.1

    def test_beta_prior_dt_to_ct_transform(self):
        """FIXED_EFFECT beta priors are converted via element-wise beta/dt scaling."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        # drift_mask enables off-diagonal at [mood, stress] position
        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_mask=drift_mask,
        )
        ssm_priors, _idx, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(0, 1): 1.0},
        )

        # Resolved 1d lag metadata: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu = ssm_priors.drift_offdiag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - 0.3) < 0.01

    def test_lagged_beta_diagnostics_explain_default_authored_interval(self):
        """Lagged-edge diagnostics should mention the default authored interval semantics."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.05},
                "sources": [
                    {
                        "title": "Weekly study",
                        "snippet": "Observed at weekly intervals.",
                        "study_interval_days": 7.0,
                    }
                ],
            },
        }
        ssm_spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            drift_mask=np.array([[True, False], [True, True]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warnings = diagnostics
        assert len(warnings) == 1
        assert "`reference_interval_days` is omitted" in _require_text(warnings[0].issue)
        assert "default model interval (1.0d)" in _require_text(warnings[0].issue)
        assert "`reference_interval_days`" in _require_text(warnings[0].suggested_adjustment)

    def test_lagged_beta_diagnostics_preserve_reference_interval_language(self):
        """Lagged-edge diagnostics should talk about the authored reference interval."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
                "sources": [
                    {
                        "title": "Daily study",
                        "snippet": "Observed at daily intervals.",
                        "study_interval_days": 1.0,
                    }
                ],
            },
        }
        ssm_spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            drift_mask=np.array([[True, False], [True, True]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warnings = diagnostics
        assert len(warnings) == 1
        assert "`reference_interval_days`" in _require_text(warnings[0].issue)
        assert "7.0d" in _require_text(warnings[0].issue)

    def test_beta_prior_dt_to_ct_respects_granularity(self):
        """FIXED_EFFECT beta transform uses effect construct's granularity."""

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
                {
                    "variable": "act",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "activity",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                },
                "measurement": {"model_clock": "1h", "indicators": []},
            }
        )
        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["heart_rate", "activity"],
            drift_mask=drift_mask,
        )
        ssm_priors, _idx, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Hourly dt = 1/24 → beta_CT = 0.3 / (1/24) = 7.2
        dt_hourly = 1.0 / 24.0
        expected_mu = 0.3 / dt_hourly  # 7.2
        mu = ssm_priors.drift_offdiag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.5

    def test_compile_ssm_inputs_attaches_direct_writer_to_dt_ct_warning(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "hr",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "act",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "activity",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                },
                "estimation": {
                    "state_order": ["heart_rate", "activity"],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1h",
                    "indicators": [
                        {
                            "name": "hr",
                            "construct_name": "heart_rate",
                            "measurement_dtype": "continuous",
                        },
                        {
                            "name": "act",
                            "construct_name": "activity",
                            "measurement_dtype": "continuous",
                        },
                    ],
                },
            }
        )

        _ssm_spec, _ssm_priors, _bindings, diagnostics, _edge_lag_days = (
            compile_ssm_inputs_from_model_spec(
                model_spec,
                priors,
                causal_spec=causal_spec,
            )
        )

        dt_ct_warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "dt_ct_approximation_warning"
        )
        assert dt_ct_warning.parameter == "drift_offdiag"
        assert dt_ct_warning.related_parameters == ["beta_activity_heart_rate"]


# --- Trial Compile Tests ---


class TestTrialCompile:
    """Test trial_compile_model_spec catches structural errors early."""

    def test_valid_spec_returns_none(self, simple_model_spec):
        """A well-formed spec compiles successfully with default priors."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        result = trial_compile_model_spec(simple_model_spec)
        assert result is None

    def test_compile_failure_returns_error(self):
        """When compilation raises, trial_compile returns the error string."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        with patch(
            "causal_ssm_agent.models.ssm_compiler._compile_validated_ssm_artifact",
            side_effect=ValueError("dimension mismatch in drift matrix"),
        ):
            result = trial_compile_model_spec(spec)
        assert result is not None
        assert "dimension mismatch" in result

    def test_role_constraint_mismatch_returns_error(self):
        """Compiler should reject parameter-role constraint mismatches."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "none",
                    "description": "test",
                },
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "constraint 'none' unexpected for role 'residual_sd'" in result

    def test_missing_ar_parameters_returns_error(self):
        """Compiler should reject ModelSpecs with no latent dimensionality signal."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "test",
                }
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "No AR_COEFFICIENT parameters found" in result

    def test_rank_deficient_structure_returns_error(self):
        """Compiler should reject model specs with fewer manifests than latents."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "outcome_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "Treatment",
                            "role": "exogenous",
                            "description": "Treatment",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "Outcome",
                            "role": "endogenous",
                            "description": "Outcome",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [],
                },
                "estimation": {
                    "state_order": ["Treatment", "Outcome"],
                    "edges": [],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "outcome_score",
                            "construct_name": "Outcome",
                            "how_to_measure": "Use the outcome column directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
            }
        )

        result = trial_compile_model_spec(spec, causal_spec)

        assert result is not None
        assert "Loading matrix is rank-deficient" in result

    def test_trial_compile_aggregates_initial_state_translation_errors(self):
        """Translation should report multiple initial-state correlation errors together."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "X",
                            "role": "exogenous",
                            "description": "X",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "Y",
                            "role": "endogenous",
                            "description": "Y",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [{"cause": "X", "effect": "Y"}],
                },
                "estimation": {
                    "state_order": ["X", "Y"],
                    "edges": [{"cause": "X", "effect": "Y"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "x_score",
                            "construct_name": "X",
                            "how_to_measure": "Use x_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "y_score",
                            "construct_name": "Y",
                            "how_to_measure": "Use y_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
            }
        )
        spec = {
            "likelihoods": [
                {
                    "variable": "x_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
                {
                    "variable": "y_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
            ],
            "parameters": [
                {
                    "name": "rho_X",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "rho_Y",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "cor0_X_X",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid self correlation",
                },
                {
                    "name": "cor0_unknown_pair",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid parse",
                },
            ],
        }

        result = trial_compile_model_spec(spec, causal_spec)

        assert result is not None
        assert "Initial-state correlation resolution failed" in result
        assert "cor0_X_X" in result
        assert "cor0_unknown_pair" in result


def test_run_stage4_returns_captured_validation(monkeypatch):
    """The last successful validation should be carried into materialization."""
    skeleton = SimpleNamespace(
        all_params=[],
        loading_params=[],
        resolved_likelihoods=[],
        ambiguous_indicators=[],
    )
    validation = AssemblyValidation(pp_checked=True, pp_valid=True)
    capture = {
        "model_spec": {"likelihoods": [{"variable": "mood_score"}]},
        "authored_priors": {"rho_mood": {"distribution": "Beta"}},
        "validation": validation,
    }

    def stub_derive_deterministic_spec(causal_spec):
        del causal_spec
        return skeleton

    def stub_build_model_topology(causal_spec):
        del causal_spec
        return {}

    def stub_build_distribution_cards(causal_spec, indicator_audits, skeleton):
        del causal_spec, indicator_audits, skeleton
        return []

    def stub_build_construct_scale_cards(causal_spec, indicator_audits, skeleton):
        del causal_spec, indicator_audits, skeleton
        return []

    def stub_build_prior_cards(causal_spec, skeleton):
        del causal_spec, skeleton
        return []

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.derive_deterministic_spec",
        stub_derive_deterministic_spec,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_model_topology",
        stub_build_model_topology,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_distribution_cards",
        stub_build_distribution_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_construct_scale_cards",
        stub_build_construct_scale_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_prior_cards",
        stub_build_prior_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_stage4_plan",
        lambda _causal_spec, _skeleton: _make_plan(),
    )

    def stub_stage4_grounding(*_args, **_kwargs):
        return _make_stub_grounding_result(capture, "VALID")

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
        stub_stage4_grounding,
    )

    async def fake_generate(messages, tools, label=None):
        del messages, tools, label
        pytest.fail("generate should not run when Stage 4 auto-completes before prompting")

    result = asyncio.run(
        run_stage4(
            causal_spec={},
            question="How can I be more productive?",
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            generate=fake_generate,
            enable_literature=False,
        )
    )

    assert result.validation is validation


def test_finalize_stage4_marks_missing_compiled_ssm_as_failure():
    extras = stage_registry._finalize_stage4_extras({}, "workspace")

    assert extras == {
        "outcome": "fail",
        "fail_reason": "model_compile_failed",
    }


class TestStage4Mechanics:
    def test_format_plan_status_exposes_effect_row_budget(self):
        causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
        runtime.decisions.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.decisions,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        runtime.accepted = Stage4AcceptedState(
            model_spec=model_spec,
            authored_priors={
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "obs_ordered_base": {
                    "parameter": "obs_ordered_base",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "ordered-threshold location prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
                "rho_sleep": {
                    "parameter": "rho_sleep",
                    "distribution": "Beta",
                    "params": {"alpha": 4.0, "beta": 3.0},
                    "sources": [],
                    "reasoning": "sleep persistence",
                },
                "sigma_sleep": {
                    "parameter": "sigma_sleep",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "sleep residual scale",
                },
            },
        )
        _activate_prior_phase(plan, runtime)
        block = _require_plan_block(plan, "effects:sleep")

        status = format_stage4_plan_status(
            plan,
            runtime,
            block,
            causal_spec=causal_spec,
        )

        assert "stability budget source" in status
        assert "target row budget" in status
        assert "remaining headroom" in status

    @pytest.mark.parametrize(
        ("payload", "expected_feedback"),
        [
            (
                {
                    "variable": "sleep",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "wrong indicator variable",
                },
                "proposal variable must be `steps`",
            ),
            (
                {
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "missing variable",
                },
                "VALIDATION ERRORS:",
            ),
        ],
    )
    def test_compute_stage4_validate_step_rejects_invalid_indicator_payloads(
        self,
        payload,
        expected_feedback,
    ):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()

        stage_output, feedback = compute_stage4_validate_step(
            payload,
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                indicator_audits={},
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for invalid submissions"
                ),
            ),
        )

        assert stage_output is None
        assert expected_feedback in feedback
        assert runtime.last_validation_packet is not None
        assert runtime.last_validation_packet.model_feedback == feedback
        assert runtime.decisions.distribution_choices == {}
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"

    def test_compute_stage4_validate_step_reopens_model_block_when_model_lock_fails(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()

        indicator_payload = {
            "block_id": "indicator:steps",
            "block_kind": "indicator_decision",
            "proposal": {
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            assert current == {}
            assert "model_spec" in data
            model_spec = data["model_spec"]
            return {
                "validation": AssemblyValidation(
                    normalized_model_spec=model_spec,
                    compile_ok=False,
                    compile_error="steps support mismatch",
                )
            }, "COMPILE ERROR:\nsteps support mismatch"

        stage_output, feedback = _apply_stage4_step_and_capture(
            indicator_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "COMPILE ERROR:\nsteps support mismatch"
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.block_status["indicator:steps"] == "reopened"
        assert runtime.last_validation_packet is not None
        assert runtime.last_validation_packet.model_feedback == feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.accepted.as_current() == {}

    def test_compute_stage4_validate_step_emits_indicator_last_state_transitions(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()

        indicator_payload = {
            "block_id": "indicator:steps",
            "block_kind": "indicator_decision",
            "proposal": {
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            assert current == {}
            assert "model_spec" in data
            model_spec = data["model_spec"]
            return {
                "validation": AssemblyValidation(
                    normalized_model_spec=model_spec,
                    compile_ok=False,
                    compile_error="steps support mismatch",
                )
            }, "COMPILE ERROR:\nsteps support mismatch"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(indicator_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback == "COMPILE ERROR:\nsteps support mismatch"
        assert transitions == (
            {
                "block_id": "indicator:steps",
                "status": "accepted",
                "detail_kind": "indicator_choice",
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
            {
                "block_id": "indicator:steps",
                "status": "reopened",
                "detail_kind": "revision",
                "reason": "steps support mismatch",
                "scope_kind": "compile_local",
            },
        )

    def test_compute_stage4_validate_step_keeps_effect_block_when_only_budget_is_tight(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        runtime.decisions.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.decisions,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec=model_spec,
            authored_priors={
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "obs_ordered_base": {
                    "parameter": "obs_ordered_base",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "ordered-threshold location prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
                "rho_sleep": {
                    "parameter": "rho_sleep",
                    "distribution": "Beta",
                    "params": {"alpha": 20.0, "beta": 1.0},
                    "sources": [],
                    "reasoning": "very persistent sleep state",
                },
                "sigma_sleep": {
                    "parameter": "sigma_sleep",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "sleep residual scale",
                },
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.25, "sigma": 0.05},
                        "sources": [],
                        "reasoning": "strong incoming sleep effect",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "BLOCK ACCEPTED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "BLOCK ACCEPTED"
        assert runtime.repair_campaign is None
        assert runtime.block_status["effects:sleep"] == "accepted"
        assert "beta_activity_sleep" in runtime.accepted.authored_priors

    def test_compute_stage4_validate_step_reopens_dynamics_block_on_partial_drift_guard(
        self, monkeypatch
    ):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        runtime.decisions.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.decisions,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        _set_runtime_block(plan, runtime, "dynamics:sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec=model_spec,
            authored_priors={
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
            },
        )
        dynamics_payload = {
            "block_id": "dynamics:sleep",
            "block_kind": "dynamics_prior",
            "proposal": {
                "priors": {
                    "rho_sleep": {
                        "parameter": "rho_sleep",
                        "distribution": "Beta",
                        "params": {"alpha": 20.0, "beta": 1.0},
                        "sources": [],
                        "reasoning": "very persistent sleep state",
                    },
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.5},
                        "sources": [],
                        "reasoning": "sleep residual scale",
                    },
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "BLOCK ACCEPTED"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_partial_drift.validate_dynamics_block_partial_drift",
            lambda **_kwargs: (
                PriorValidationResult(
                    parameter="rho_sleep",
                    is_valid=False,
                    code="partial_dynamics_budget_exhausted",
                    origin="prior_predictive",
                    issue="The sleep row has no conservative damping headroom yet.",
                    suggested_adjustment="Tighten rho_sleep toward faster decay.",
                    related_parameters=["sigma_sleep"],
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.1,
                    ),
                ),
                "\n".join(
                    [
                        "PARTIAL DRIFT CHECK FAILED:",
                        "- target row `sleep` has no conservative damping headroom yet",
                        "- revise this dynamics block before eliciting downstream effect priors",
                    ]
                ),
            ),
        )

        stage_output, feedback = _apply_stage4_step_and_capture(
            dynamics_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is None
        assert feedback.startswith("PARTIAL DRIFT CHECK FAILED:")
        assert runtime.repair_campaign is not None
        assert runtime.repair_campaign.scope_block_ids == ("dynamics:sleep",)
        assert runtime.block_status["dynamics:sleep"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"
        assert "rho_sleep" not in runtime.accepted.authored_priors
        assert "sigma_sleep" not in runtime.accepted.authored_priors
        assert runtime.last_validation_packet is not None
        assert runtime.last_validation_packet.status == "partial_drift_failure"
        assert runtime.last_validation_packet.failing_parameters == ("rho_sleep",)
        assert runtime.last_validation_packet.coupled_parameters == ("sigma_sleep",)
        assert runtime.last_validation_packet.diagnostic_codes == (
            "partial_dynamics_budget_exhausted",
        )

    def test_global_review_can_reopen_model_block_set(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        _set_runtime_block(plan, runtime, "review:model_spec")
        runtime.accepted = Stage4AcceptedState(model_spec={"parameters": [{"name": "locked"}]})
        runtime.block_status["indicator:steps"] = "accepted"
        runtime.block_status["review:model_spec"] = "pending"

        stage_output, feedback = compute_stage4_validate_step(
            {
                "decision": "reopen",
                "reopen_block_ids": ["indicator:steps"],
                "reasoning": "The count likelihood should be reconsidered.",
            },
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                indicator_audits={},
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for review-only reopen decisions"
                ),
            ),
        )

        assert stage_output is None
        assert "MODEL REVIEW REOPENED" in feedback
        assert "`indicator:steps`" in feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert get_stage4_phase(runtime, plan=plan) == "model_decisions"
        assert runtime.block_status["indicator:steps"] == "reopened"

    def test_global_review_allows_reopening_more_than_three_model_blocks(self):
        model_blocks = (
            Stage4FrontierBlock(
                id="indicator:a",
                kind="indicator_decision",
                label="Indicator a",
                variable_names=("a",),
            ),
            Stage4FrontierBlock(
                id="indicator:b",
                kind="indicator_decision",
                label="Indicator b",
                variable_names=("b",),
            ),
            Stage4FrontierBlock(
                id="indicator:c",
                kind="indicator_decision",
                label="Indicator c",
                variable_names=("c",),
            ),
            Stage4FrontierBlock(
                id="indicator:d",
                kind="indicator_decision",
                label="Indicator d",
                variable_names=("d",),
            ),
        )
        review_block = Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Review",
            payload={"reopenable_block_ids": tuple(block.id for block in model_blocks)},
        )
        plan = _make_plan(model_blocks=model_blocks, review_block=review_block)
        runtime = _make_runtime(
            plan,
            phase="global_review",
            active_block_id="review:model_spec",
            accepted=Stage4AcceptedState(model_spec={"parameters": [{"name": "locked"}]}),
        )
        for block in model_blocks:
            runtime.block_status[block.id] = "accepted"
        runtime.block_status["review:model_spec"] = "pending"

        stage_output, feedback = compute_stage4_validate_step(
            {
                "decision": "reopen",
                "reopen_block_ids": [block.id for block in model_blocks],
                "reasoning": "These measurement decisions need to be reconsidered together.",
            },
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec={},
                skeleton=Stage4Skeleton(),
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for review-only reopen decisions"
                ),
            ),
        )

        assert stage_output is None
        assert "MODEL REVIEW REOPENED" in feedback
        assert "`indicator:a`, `indicator:b`, `indicator:c`, `indicator:d`" in feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:a"
        assert get_stage4_phase(runtime, plan=plan) == "model_decisions"
        for block in model_blocks:
            assert runtime.block_status[block.id] == "reopened"

    def test_compute_stage4_validate_step_reopens_indicator_on_support_mismatch(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {"variable": "steps", "distribution": "poisson", "link": "log"},
                ],
                "parameters": [
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "lambda_activity_vas_activity"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        _set_runtime_block(plan, runtime, "effects:sleep")
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "effect prior with support issue",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="model_build",
                            is_valid=False,
                            code="model_build",
                            origin="prior_predictive",
                            issue=(
                                "Observation support check failed:\n"
                                "- 'steps' uses gamma emission but observations are outside support"
                            ),
                            suggested_adjustment="Fix the emission support",
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE CHECKS FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "PRIOR PREDICTIVE CHECKS FAILED"
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.block_status["indicator:steps"] == "reopened"
        assert "beta_activity_sleep" in runtime.accepted.authored_priors
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"

    def test_compute_stage4_validate_step_accepts_correlation_and_reopens_dynamics_scope(self):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "correlation:cor0_activity_sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        correlation_payload = {
            "block_id": "correlation:cor0_activity_sleep",
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "cor0_activity_sleep": {
                        "parameter": "cor0_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2, "lower": -1.0, "upper": 1.0},
                        "sources": [],
                        "reasoning": "correlation prior",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        ),
                        PriorValidationResult(
                            parameter="dynamics_stability",
                            is_valid=False,
                            code="dynamics_stability",
                            origin="prior_predictive",
                            issue="Unstable dynamics: 32/50 prior draws have unstable drift",
                            suggested_adjustment="Tighten drift_diag prior toward more negative values",
                            repair_scope=PriorRepairScope(
                                kind="dynamics_scc",
                                construct_names=["sleep"],
                            ),
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            correlation_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"
        assert runtime.block_status["correlation:cor0_activity_sleep"] == "accepted"
        assert runtime.block_status["dynamics:sleep"] == "reopened"
        assert runtime.block_status["effects:sleep"] == "pending"
        assert runtime.repair_campaign is not None
        assert runtime.repair_campaign.scope_key == "validator_scope:sleep"
        assert runtime.repair_campaign.scope_block_ids == ("dynamics:sleep",)
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"
        assert "cor0_activity_sleep" in runtime.accepted.authored_priors
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"

    def test_compute_stage4_validate_step_emits_prior_and_revision_last_state_transitions(self):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "correlation:cor0_activity_sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        correlation_payload = {
            "block_id": "correlation:cor0_activity_sleep",
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "cor0_activity_sleep": {
                        "parameter": "cor0_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2, "lower": -1.0, "upper": 1.0},
                        "sources": [],
                        "reasoning": "correlation prior",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        ),
                        PriorValidationResult(
                            parameter="dynamics_stability",
                            is_valid=False,
                            code="dynamics_stability",
                            origin="prior_predictive",
                            issue="Unstable dynamics: 32/50 prior draws have unstable drift",
                            suggested_adjustment="Tighten drift_diag prior toward more negative values",
                            repair_scope=PriorRepairScope(
                                kind="dynamics_scc",
                                construct_names=["sleep"],
                            ),
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(correlation_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback == "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"
        assert transitions[0] == {
            "block_id": "correlation:cor0_activity_sleep",
            "status": "accepted",
            "detail_kind": "prior_bundle",
            "parameter_names": ["cor0_activity_sleep"],
            "priors": [
                {
                    "parameter": "cor0_activity_sleep",
                    "distribution": "Normal",
                    "params": {
                        "mu": 0.0,
                        "sigma": 0.2,
                        "lower": -1.0,
                        "upper": 1.0,
                    },
                    "reasoning": "correlation prior",
                }
            ],
        }
        assert transitions[1]["block_id"] == "dynamics:sleep"
        assert transitions[1]["status"] == "reopened"
        assert transitions[1]["detail_kind"] == "revision"
        assert transitions[1]["scope_kind"] == "validator_scope"
        assert (
            transitions[1]["reason"]
            == "Unstable dynamics: 32/50 prior draws have unstable drift Suggested "
            "fix: Tighten drift_diag prior toward more negative values"
        )

    def test_compute_stage4_validate_step_emits_accepted_transition_for_barrier_campaign_repair(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        repair_block = _require_plan_block(plan, "correlation:cor0_activity_sleep")
        next_block = _require_plan_block(plan, "effects:sleep")
        _set_runtime_block(plan, runtime, repair_block.id)
        runtime.block_status[repair_block.id] = "reopened"
        runtime.block_status[next_block.id] = "reopened"
        runtime.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_nonfinite_samples",), (), ("activity", "sleep")),
            scope_kind="validator_scope",
            scope_key="validator_scope:sleep",
            scope_rank=0,
            scope_block_ids=(repair_block.id, next_block.id),
            prompt_blocks_by_id={
                repair_block.id: repair_block,
                next_block.id: next_block,
            },
            completed_block_ids=frozenset(),
            requires_barrier_validation=True,
            attempts_at_scope=1,
            best_certificate=None,
        )
        repair_payload = {
            "block_id": repair_block.id,
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "cor0_activity_sleep": {
                        "parameter": "cor0_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15, "lower": -1.0, "upper": 1.0},
                        "sources": [],
                        "reasoning": "repair tightened the correlation prior",
                    }
                }
            },
        }

        seen_skip_ppc: list[bool] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            seen_skip_ppc.append(_kwargs["skip_ppc"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                ),
            }, "VALID"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(repair_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback.startswith("REPAIR CAMPAIGN PROGRESS:\n")
        assert seen_skip_ppc == [True]
        assert transitions == (
            {
                "block_id": "correlation:cor0_activity_sleep",
                "status": "accepted",
                "detail_kind": "prior_bundle",
                "parameter_names": ["cor0_activity_sleep"],
                "priors": [
                    {
                        "parameter": "cor0_activity_sleep",
                        "distribution": "Normal",
                        "params": {
                            "mu": 0.0,
                            "sigma": 0.15,
                            "lower": -1.0,
                            "upper": 1.0,
                        },
                        "reasoning": "repair tightened the correlation prior",
                    }
                ],
            },
        )
        assert runtime.block_status[repair_block.id] == "accepted"
        assert runtime.repair_campaign is not None
        assert runtime.repair_campaign.completed_block_ids == frozenset((repair_block.id,))
        assert _require_active_plan_block(plan, runtime).id == next_block.id

    def test_compute_stage4_validate_step_reruns_ppc_after_final_barrier_repair(self, monkeypatch):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        repair_block = _require_plan_block(plan, "correlation:cor0_activity_sleep")
        final_block = _require_plan_block(plan, "effects:sleep")

        model_spec = {
            "likelihoods": [
                {
                    "variable": "activity_vas",
                    "distribution": "ordered_logistic",
                    "link": "logit",
                },
                {
                    "variable": "sleep_quality",
                    "distribution": "ordered_logistic",
                    "link": "logit",
                },
            ],
            "parameters": [
                {"name": "lambda_activity_vas_activity"},
                {"name": "lambda_sleep_quality_sleep"},
                {"name": "rho_activity"},
                {"name": "sigma_activity"},
                {"name": "rho_sleep"},
                {"name": "sigma_sleep"},
                {"name": "beta_activity_sleep"},
                {"name": "cor0_activity_sleep"},
            ],
        }
        runtime.accepted = Stage4AcceptedState(
            model_spec=model_spec,
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "cor0_activity_sleep": {"distribution": "Normal"},
            },
        )
        _set_runtime_block(plan, runtime, final_block.id)
        runtime.block_status[repair_block.id] = "accepted"
        runtime.block_status[final_block.id] = "reopened"
        runtime.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_nonfinite_samples",), (), ("activity", "sleep")),
            scope_kind="validator_scope",
            scope_key="validator_scope:sleep",
            scope_rank=0,
            scope_block_ids=(repair_block.id, final_block.id),
            prompt_blocks_by_id={
                repair_block.id: repair_block,
                final_block.id: final_block,
            },
            completed_block_ids=frozenset((repair_block.id,)),
            requires_barrier_validation=True,
            attempts_at_scope=1,
            best_certificate=None,
        )
        repair_payload = {
            "block_id": final_block.id,
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "repair tightened the effect prior",
                    }
                }
            },
        }

        seen_skip_ppc: list[bool] = []
        barrier_validations: list[dict[str, Any]] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            seen_skip_ppc.append(kwargs["skip_ppc"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "VALID"

        def stub_validate_assembly(
            model_spec_arg,
            authored_priors_arg,
            _data_for_model,
            _indicator_audits,
            _causal_spec,
            *,
            skip_ppc=False,
        ):
            barrier_validations.append(
                {
                    "model_spec": model_spec_arg,
                    "authored_priors": dict(authored_priors_arg or {}),
                    "skip_ppc": skip_ppc,
                }
            )
            return AssemblyValidation(
                normalized_model_spec=model_spec_arg,
                compile_ok=True,
                pp_checked=True,
                pp_valid=False,
                diagnostics=[
                    PriorValidationResult(
                        parameter="beta_activity_sleep",
                        is_valid=False,
                        code="effect_still_too_wide",
                        origin="prior_predictive",
                        issue="Effect prior still too wide",
                        suggested_adjustment="Tighten beta_activity_sleep",
                    )
                ],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_partial_drift.validate_effect_block_partial_drift",
            lambda **_kwargs: None,
        )

        stage_output, feedback, _transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(repair_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\n")
        assert seen_skip_ppc == [True]
        assert len(barrier_validations) == 1
        assert barrier_validations[0]["skip_ppc"] is False
        assert "beta_activity_sleep" in barrier_validations[0]["authored_priors"]
        assert runtime.accepted.validation is not None
        assert runtime.accepted.validation.pp_checked is True
        assert runtime.accepted.validation.pp_valid is False
        assert runtime.last_validation_packet is not None
        assert runtime.last_validation_packet.status == "prior_predictive_failure"

    def test_compute_stage4_validate_step_synthesizes_bounded_repair_bundle_from_supporting_diagnostics(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "global repair trigger",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="drift_offdiag",
                            is_valid=True,
                            code="dt_ct_approximation_warning",
                            origin="compile",
                            severity="warning",
                            issue="off-diagonal drift is large relative to damping",
                            related_parameters=["beta_activity_sleep"],
                        ),
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                            related_parameters=["drift_offdiag"],
                            supporting_codes=["dt_ct_approximation_warning"],
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("REPAIR CAMPAIGN ACTIVE:\n")
        assert "scope: `local_drift_motif:activity|sleep|beta_activity_sleep`" in feedback
        assert runtime.block_status["effects:sleep"] == "accepted"
        assert runtime.block_status["dynamics:activity"] == "reopened"
        assert runtime.block_status["dynamics:sleep"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "dynamics:activity"
        assert get_stage4_phase(runtime, plan=plan) == "prior_blocks"
        assert "beta_activity_sleep" in runtime.accepted.authored_priors

    def test_prior_failure_classification_allows_one_same_scope_retry_on_certificate_improvement(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(
                ("prior_predictive_nonfinite_samples",),
                ("dt_ct_approximation_warning",),
                ("activity", "sleep"),
            ),
            scope_kind="local_drift_motif",
            scope_key="local_drift_motif:activity|sleep|beta_activity_sleep",
            scope_rank=0,
            scope_block_ids=("dynamics:activity", "dynamics:sleep", "effects:sleep"),
            completed_block_ids=frozenset(("dynamics:activity", "dynamics:sleep", "effects:sleep")),
            attempts_at_scope=1,
            best_certificate=PriorPathologyCertificate(
                kind="nonfinite_samples",
                primary_score=1.0,
            ),
        )

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="drift_offdiag",
                    is_valid=True,
                    code="dt_ct_approximation_warning",
                    origin="compile",
                    severity="warning",
                    issue="off-diagonal drift is large relative to damping",
                    related_parameters=["beta_activity_sleep"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dt_ct_approximation",
                        primary_score=0.30,
                    ),
                ),
                PriorValidationResult(
                    parameter="prior_predictive",
                    is_valid=False,
                    code="prior_predictive_nonfinite_samples",
                    origin="prior_predictive",
                    issue="NaN/Inf detected in sample sites: observations",
                    suggested_adjustment="Check for degenerate priors",
                    related_parameters=["drift_offdiag"],
                    supporting_codes=["dt_ct_approximation_warning"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="nonfinite_samples",
                        primary_score=0.75,
                    ),
                ),
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "effects:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.scope.scope_key == "local_drift_motif:activity|sleep|beta_activity_sleep"

    def test_prior_failure_classification_prefers_lowest_rank_scope_over_validator_scope(self):
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "sleep",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "activity_vas",
                            "construct_name": "activity",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Activity rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "sleep_quality",
                            "construct_name": "sleep",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Sleep quality rating",
                            "aggregation": "mean",
                        },
                    ],
                },
                "estimation": {
                    "state_order": ["activity", "sleep"],
                    "edges": [
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                    "induced_dependencies": [],
                },
            }
        )
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="drift_offdiag",
                    is_valid=True,
                    code="dt_ct_approximation_warning",
                    origin="compile",
                    severity="warning",
                    related_parameters=["beta_activity_sleep"],
                ),
                PriorValidationResult(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="dynamics_stability",
                    origin="prior_predictive",
                    issue="Unstable reciprocal drift subsystem",
                    related_parameters=["beta_activity_sleep"],
                    supporting_codes=["dt_ct_approximation_warning"],
                    repair_scope=PriorRepairScope(
                        kind="dynamics_scc",
                        construct_names=["activity", "sleep"],
                    ),
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.6,
                    ),
                ),
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "effects:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.block_ids == ("dynamics:activity+sleep", "effects:sleep")

    def test_scc_repair_plan_narrows_effect_prompt_to_internal_scc_parameters(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import build_repair_plan

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {"name": "stress", "role": "exogenous", "temporal_status": "time_varying"},
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "sleep",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [
                        {"cause": "stress", "effect": "sleep"},
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "stress_vas",
                            "construct_name": "stress",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Stress rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "activity_vas",
                            "construct_name": "activity",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Activity rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "sleep_quality",
                            "construct_name": "sleep",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Sleep quality rating",
                            "aggregation": "mean",
                        },
                    ],
                },
                "estimation": {
                    "state_order": ["stress", "activity", "sleep"],
                    "edges": [
                        {"cause": "stress", "effect": "sleep"},
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                    "induced_dependencies": [],
                },
            }
        )
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)

        repair_plan = build_repair_plan(
            plan,
            ResolvedRepairScope(
                scope_kind="scc_drift_subsystem",
                scope_rank=2,
                scope_key="scc_drift_subsystem:activity|sleep",
                reason="repair internal SCC drift subsystem",
                failure_family=("dynamics_stability",),
                construct_names=("activity", "sleep"),
            ),
        )

        sleep_prompt = next(
            block for block in repair_plan.prompt_blocks if block.id == "effects:sleep"
        )
        assert sleep_prompt.parameter_names == ("beta_activity_sleep",)
        assert sleep_prompt.construct_names == ("activity", "sleep")
        assert "stress_vas" not in sleep_prompt.variable_names
        assert sleep_prompt.expand_neighbor_topology is False

    def test_prior_failure_classification_treats_partial_dynamics_codes_as_drift_related(self):
        _causal_spec, _skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
        _set_runtime_block(plan, runtime, "dynamics:sleep")

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="partial_dynamics_budget_exhausted",
                    origin="prior_predictive",
                    issue="The sleep row has no conservative damping headroom yet.",
                    related_parameters=["rho_sleep", "sigma_sleep"],
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.2,
                    ),
                )
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "dynamics:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.block_ids == ("dynamics:sleep",)

    def test_validate_dynamics_block_partial_drift_treats_budget_overrun_as_advisory(
        self, monkeypatch
    ):
        from causal_ssm_agent.flows.stages.stage4.agentic import (
            stage4_partial_drift as partial_drift_module,
        )

        state = partial_drift_module._PartialDriftState(
            latent_names=("activity", "sleep"),
            diag_mu=np.array([0.4, 0.6]),
            diag_sigma=np.array([0.1, 0.2]),
            diag_present=np.array([True, True]),
            diag_parameter_by_index={0: "rho_activity", 1: "rho_sleep"},
            offdiag_positions=((1, 0),),
            offdiag_mu=np.array([0.5]),
            offdiag_sigma=np.array([0.3]),
            offdiag_present=np.array([True]),
            offdiag_parameter_by_index={0: "beta_activity_sleep"},
        )

        monkeypatch.setattr(
            partial_drift_module,
            "_build_partial_drift_state",
            lambda **_kwargs: state,
        )

        result = partial_drift_module.validate_dynamics_block_partial_drift(
            model_spec={"parameters": [{"name": "rho_sleep"}]},
            authored_priors={
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
            causal_spec={},
            active_construct_names=("sleep",),
            active_parameter_names=("rho_sleep", "sigma_sleep"),
        )

        assert result is None

    def test_validate_effect_block_partial_drift_treats_budget_overrun_as_advisory(
        self, monkeypatch
    ):
        from causal_ssm_agent.flows.stages.stage4.agentic import (
            stage4_partial_drift as partial_drift_module,
        )

        state = partial_drift_module._PartialDriftState(
            latent_names=("activity", "sleep"),
            diag_mu=np.array([0.4, 0.6]),
            diag_sigma=np.array([0.1, 0.2]),
            diag_present=np.array([True, True]),
            diag_parameter_by_index={0: "rho_activity", 1: "rho_sleep"},
            offdiag_positions=((1, 0),),
            offdiag_mu=np.array([0.5]),
            offdiag_sigma=np.array([0.3]),
            offdiag_present=np.array([True]),
            offdiag_parameter_by_index={0: "beta_activity_sleep"},
        )

        monkeypatch.setattr(
            partial_drift_module,
            "_build_partial_drift_state",
            lambda **_kwargs: state,
        )

        result = partial_drift_module.validate_effect_block_partial_drift(
            model_spec={"parameters": [{"name": "beta_activity_sleep"}]},
            authored_priors={
                "rho_activity": {"distribution": "Beta"},
                "rho_sleep": {"distribution": "Beta"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
            causal_spec={},
            target_construct="sleep",
            active_parameter_names=("beta_activity_sleep",),
        )

        assert result is None

    def test_compile_failure_route_prefers_true_indicator_owner_for_exact_match(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import (
            classify_compile_failure_route,
        )

        causal_spec, skeleton, plan, _runtime, _data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton

        repair_plan = classify_compile_failure_route(
            plan,
            _require_plan_block(plan, "measurement:activity"),
            "Compile error: steps has incompatible support metadata.",
        )

        assert repair_plan.scope.scope_kind == "compile_local"
        assert repair_plan.block_ids == ("indicator:steps",)

    def test_compute_stage4_validate_step_escalates_unattributed_global_failure_to_prior_review(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "global repair trigger",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"
        assert runtime.block_status["effects:sleep"] == "accepted"
        assert runtime.block_status["review:prior_system"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "review:prior_system"
        assert get_stage4_phase(runtime, plan=plan) == "global_prior_review"
        assert "beta_activity_sleep" in runtime.accepted.authored_priors

    def test_compute_stage4_validate_step_raises_on_repeated_unattributed_global_prior_review_failure(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.block_status["review:prior_system"] = "reopened"
        runtime.accepted = Stage4AcceptedState(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
                "cor0_activity_sleep": {"distribution": "Normal"},
            },
        )
        review_payload = {
            "block_id": "review:prior_system",
            "block_kind": "global_prior_review",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "global repair attempt",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                            related_parameters=["drift_offdiag"],
                            supporting_codes=["dt_ct_approximation_warning"],
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        _apply_stage4_step_and_capture(
            review_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )
        assert runtime.repair_campaign is not None
        assert runtime.repair_campaign.scope_key == "global_prior_review:prior_system"
        assert runtime.repair_campaign.attempts_at_scope == 1
        assert _require_active_plan_block(plan, runtime).id == "review:prior_system"
        assert get_stage4_phase(runtime, plan=plan) == "global_prior_review"

        _apply_stage4_step_and_capture(
            review_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
        )
        assert runtime.repair_campaign is not None
        assert runtime.repair_campaign.attempts_at_scope == 2

        with pytest.raises(ValueError, match="exhausted the deterministic repair-scope ladder"):
            _apply_stage4_step_and_capture(
                review_payload,
                plan,
                runtime,
                skeleton=skeleton,
                causal_spec=causal_spec,
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            )

    def test_compute_stage4_validate_step_rejects_calls_after_completion(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        runtime.accepted = Stage4AcceptedState(
            model_spec={"parameters": [{"name": "done"}]},
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        _set_done_cursor(runtime)

        stage_output, feedback = compute_stage4_validate_step(
            {},
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                indicator_audits={},
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run after completion"
                ),
            ),
        )

        assert stage_output is None
        assert feedback == "VALIDATION ERRORS:\n- no active Stage 4 frontier block remains"
        assert runtime.last_validation_packet is None

    def test_compute_stage4_validate_step_tracks_frontier_path_without_llm(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()

        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.4},
                            "sources": [],
                            "reasoning": "initial measurement prior",
                        }
                    }
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "stable activity residual scale",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "bad sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "paired with bad dynamics prior",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior that exposes measurement mismatch",
                        }
                    }
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.25},
                            "sources": [],
                            "reasoning": "corrected measurement prior",
                        }
                    }
                },
            },
        ]

        expected_blocks = [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:sleep",
            "effects:sleep",
            "measurement:activity",
        ]
        expected_reopen_ids = [
            None,
            None,
            None,
            None,
            None,
            "dynamics:sleep",
            None,
            "measurement:activity",
            None,
        ]

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = current or {}
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            priors = data["priors"]
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(priors)
            model_spec = current.get("model_spec")

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "initial measurement prior"
            ):
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "obs_ordered_base" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "sigma_activity" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if priors.get("rho_sleep", {}).get("reasoning") == "bad sleep dynamics prior":
                return {
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=False,
                        compile_error="rho_sleep interval instability",
                    ),
                }, "COMPILE ERROR:\nrho_sleep interval instability"

            if "rho_sleep" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "beta_activity_sleep" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=False,
                        diagnostics=[
                            PriorValidationResult(
                                parameter="scale_activity_vas",
                                is_valid=False,
                                code="scale_mismatch",
                                origin="prior_predictive",
                                issue="Scale mismatch for activity_vas",
                                suggested_adjustment="Tighten the measurement prior",
                            )
                        ],
                    ),
                }, "PRIOR PREDICTIVE CHECKS FAILED"

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "corrected measurement prior"
            ):
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                }, "VALID"

            raise AssertionError(f"Unexpected Stage 4 grounding payload: {data}")

        visited_blocks: list[str] = []
        reopen_ids: list[str | None] = []
        for expected_block, _expected_reopen_id, payload in zip(
            expected_blocks, expected_reopen_ids, submissions, strict=True
        ):
            active_block = get_active_plan_block(plan, runtime)
            assert active_block is not None
            visited_blocks.append(active_block.id)
            assert active_block.id == expected_block

            _apply_stage4_step_and_capture(
                payload,
                plan,
                runtime,
                skeleton=skeleton,
                causal_spec=causal_spec,
                data_for_model=data_for_model,
                indicator_audits={},
                stage4_grounding_fn=stub_stage4_grounding,
            )
            reopened_block = get_active_plan_block(plan, runtime)
            reopen_ids.append(
                reopened_block.id
                if reopened_block is not None
                and runtime.block_status.get(reopened_block.id) == "reopened"
                else None
            )

        assert visited_blocks == expected_blocks
        assert reopen_ids == expected_reopen_ids
        assert get_active_plan_block(plan, runtime) is None
        assert get_stage4_phase(runtime, plan=plan) == "done"
        assert sorted(runtime.accepted.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "obs_ordered_base",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]

    def test_run_stage4_can_follow_scripted_submit_tool_path(self, monkeypatch):
        causal_spec = _make_stage4_mechanics_spec()

        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.25},
                            "sources": [],
                            "reasoning": "measurement prior",
                        }
                    }
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "activity residual scale",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior",
                        }
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = current or {}
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return _make_stub_grounding_result(
                    {
                        "model_spec": model_spec,
                        "validation": AssemblyValidation(
                            normalized_model_spec=model_spec,
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=len(authored_priors) == 6,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="Does activity improve sleep?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=_make_scripted_stage4_generate(
                    submissions,
                    visited_blocks=visited_blocks,
                    visible_tools=visible_tools,
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert visited_blocks == [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "effects:sleep",
        ]
        assert visible_tools == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert any(
            likelihood["variable"] == "steps" and likelihood["distribution"] == "poisson"
            for likelihood in result.model_spec["likelihoods"]
        )
        assert sorted(result.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "obs_ordered_base",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]

    def test_run_stage4_effect_blocks_run_sequentially_to_final_validation(self, monkeypatch):
        causal_spec = _make_stage4_two_effect_spec()

        submissions_by_block = {
            "indicator:steps": {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            "review:model_spec": {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            "measurement:activity": {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.25},
                            "sources": [],
                            "reasoning": "measurement prior",
                        }
                    }
                },
            },
            "observation:obs_ordered_base": {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            "dynamics:activity": {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.4},
                            "sources": [],
                            "reasoning": "activity residual scale",
                        },
                    }
                },
            },
            "dynamics:sleep": {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep persistence",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            "dynamics:mood": {
                "block_id": "dynamics:mood",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_mood": {
                            "parameter": "rho_mood",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "mood persistence",
                        },
                        "sigma_mood": {
                            "parameter": "sigma_mood",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "mood residual scale",
                        },
                    }
                },
            },
            "effects:sleep": {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "sleep effect prior",
                        }
                    }
                },
            },
            "effects:mood": {
                "block_id": "effects:mood",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_mood": {
                            "parameter": "beta_activity_mood",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "mood effect prior",
                        }
                    }
                },
            },
        }
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []
        final_prior_count = 9

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = current or {}
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return _make_stub_grounding_result(
                    {
                        "model_spec": model_spec,
                        "validation": AssemblyValidation(
                            normalized_model_spec=model_spec,
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            pp_checked = len(authored_priors) == final_prior_count
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=pp_checked,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        def stub_validate_assembly(model_spec, authored_priors, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                pp_checked=True,
                pp_valid=True,
                diagnostics=[],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="Does activity affect sleep and mood?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=_make_scripted_stage4_generate_by_block(
                    submissions_by_block,
                    visited_blocks=visited_blocks,
                    visible_tools=visible_tools,
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert visited_blocks == [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:mood",
            "effects:sleep",
            "effects:mood",
        ]
        assert visible_tools == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == [
            "beta_activity_mood",
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "obs_ordered_base",
            "rho_mood",
            "rho_sleep",
            "sigma_activity",
            "sigma_mood",
            "sigma_sleep",
        ]
        assert result.validation is not None
        assert result.validation.pp_checked is True
        assert result.validation.pp_valid is True

    def test_run_stage4_auto_locks_initial_model_spec_when_no_model_blocks(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        submissions = [
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []
        model_spec_calls: list[dict] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                model_spec_calls.append(data["model_spec"])
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=_make_scripted_stage4_generate(
                    submissions,
                    visited_blocks=visited_blocks,
                    visible_tools=visible_tools,
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert len(model_spec_calls) == 1
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
        ]
        assert visible_tools == [
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_run_stage4_tracks_submission_when_feedback_repeats(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        submissions = [
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []
        prior_attempts = 0

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            nonlocal prior_attempts
            current = current or {}
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            if "obs_ordered_base" in data["priors"]:
                return _make_stub_grounding_result(
                    {
                        "authored_priors": authored_priors,
                        "validation": AssemblyValidation(
                            normalized_model_spec=current.get("model_spec"),
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            prior_attempts += 1
            if prior_attempts < 3:
                return _make_stub_grounding_result(
                    {
                        "authored_priors": authored_priors,
                        "validation": AssemblyValidation(
                            normalized_model_spec=current.get("model_spec"),
                            compile_ok=True,
                            pp_checked=True,
                            pp_valid=False,
                            diagnostics=[
                                PriorValidationResult(
                                    parameter="rho_sleep",
                                    is_valid=False,
                                    code=f"local_prior_adjustment_{prior_attempts}",
                                    origin="prior_predictive",
                                    issue="Sleep persistence prior still needs adjustment",
                                    suggested_adjustment="Tighten the active dynamics prior",
                                )
                            ],
                        ),
                    },
                    "PRIOR PREDICTIVE FEEDBACK:\n- still failing",
                )

            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=_make_scripted_stage4_generate(
                    submissions,
                    visited_blocks=visited_blocks,
                    visible_tools=visible_tools,
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert prior_attempts == 3
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
            "dynamics:sleep",
            "dynamics:sleep",
        ]
        assert visible_tools == [
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_run_stage4_resumes_from_runtime_checkpoint(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        review_submission = {
            "block_id": "review:model_spec",
            "block_kind": "global_review",
            "proposal": {
                "decision": "approve",
                "reasoning": "The deterministic model form is coherent.",
            },
        }
        observation_submission = {
            "block_id": "observation:obs_ordered_base",
            "block_kind": "observation_prior",
            "proposal": {
                "priors": {
                    "obs_ordered_base": {
                        "parameter": "obs_ordered_base",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "ordered-threshold location prior",
                    }
                }
            },
        }
        dynamics_submission = {
            "block_id": "dynamics:sleep",
            "block_kind": "dynamics_prior",
            "proposal": {
                "priors": {
                    "rho_sleep": {
                        "parameter": "rho_sleep",
                        "distribution": "Beta",
                        "params": {"alpha": 3.0, "beta": 2.0},
                        "sources": [],
                        "reasoning": "sleep persistence",
                    },
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.35},
                        "sources": [],
                        "reasoning": "sleep residual scale",
                    },
                }
            },
        }
        saved_runtime: Stage4Runtime | None = None
        first_run_blocks: list[str] = []
        second_run_blocks: list[str] = []
        clear_calls = 0

        def save_checkpoint(runtime: Stage4Runtime) -> None:
            nonlocal saved_runtime
            saved_runtime = deepcopy(runtime)

        def clear_checkpoint() -> None:
            nonlocal clear_calls, saved_runtime
            clear_calls += 1
            saved_runtime = None

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        async def fail_after_review(
            messages,
            tools,
            rewrite_messages=None,
            rewrite_tools=None,
            label=None,
        ):
            del messages, rewrite_messages, rewrite_tools
            assert label is not None
            assert label.startswith("stage-4:")
            block_id = label.removeprefix("stage-4:")
            first_run_blocks.append(block_id)
            if block_id == "review:model_spec":
                submit_tool = next(tool for tool in tools if tool.name == "submit_model_review")
                feedback = await submit_tool(**_stage4_submit_tool_args(review_submission))
                assert isinstance(feedback, str)
                assert not feedback.startswith("VALIDATION ERRORS:")
                return ""
            raise RuntimeError("OpenRouter returned no choices")

        with pytest.raises(RuntimeError, match="OpenRouter returned no choices"):
            asyncio.run(
                run_stage4(
                    causal_spec=causal_spec,
                    question="How persistent is sleep quality?",
                    data_for_model=pl.DataFrame(),
                    indicator_audits={},
                    generate=fail_after_review,
                    enable_literature=False,
                    enable_paraphrasing=False,
                    save_checkpoint=save_checkpoint,
                )
            )

        assert saved_runtime is not None
        assert first_run_blocks == ["review:model_spec", "observation:obs_ordered_base"]

        async def resume_from_dynamics(
            messages,
            tools,
            rewrite_messages=None,
            rewrite_tools=None,
            label=None,
        ):
            del messages, rewrite_messages, rewrite_tools
            assert label is not None
            assert label.startswith("stage-4:")
            block_id = label.removeprefix("stage-4:")
            second_run_blocks.append(block_id)
            submission = (
                observation_submission
                if block_id == "observation:obs_ordered_base"
                else dynamics_submission
            )
            submit_tool = next(
                tool
                for tool in tools
                if tool.name == _stage4_submit_tool_name(submission["block_kind"])
            )
            feedback = await submit_tool(**_stage4_submit_tool_args(submission))
            assert isinstance(feedback, str)
            assert not feedback.startswith("VALIDATION ERRORS:")
            return ""

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=resume_from_dynamics,
                enable_literature=False,
                enable_paraphrasing=False,
                load_checkpoint=lambda: saved_runtime,
                save_checkpoint=save_checkpoint,
                clear_checkpoint=clear_checkpoint,
            )
        )

        assert second_run_blocks == ["observation:obs_ordered_base", "dynamics:sleep"]
        assert clear_calls == 1
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_run_stage4_discards_invalid_runtime_checkpoint(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        visited_blocks: list[str] = []
        cleared: list[bool] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        submissions_by_block = {
            "review:model_spec": {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            "observation:obs_ordered_base": {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            "dynamics:sleep": {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep persistence",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
        }

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                generate=_make_scripted_stage4_generate_by_block(
                    submissions_by_block,
                    visited_blocks=visited_blocks,
                    visible_tools=[],
                ),
                enable_literature=False,
                enable_paraphrasing=False,
                load_checkpoint=lambda: {"not": "a runtime"},
                clear_checkpoint=lambda: cleared.append(True),
            )
        )

        assert cleared == [True, True]
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_stage4_tool_loop_compacts_context_while_trace_grows(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage4.tools import (
            make_submit_indicator_choice_tool,
            make_submit_model_review_tool,
            make_submit_prior_block_tool,
        )

        causal_spec = _make_stage4_mechanics_spec()
        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.4},
                            "sources": [],
                            "reasoning": "initial measurement prior",
                        }
                    }
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "stable activity residual scale",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "bad sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "paired with bad dynamics prior",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior that exposes measurement mismatch",
                        }
                    }
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": {
                        "lambda_activity_vas_activity": {
                            "parameter": "lambda_activity_vas_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.25},
                            "sources": [],
                            "reasoning": "corrected measurement prior",
                        }
                    }
                },
            },
        ]
        expected_blocks = [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:sleep",
            "effects:sleep",
            "measurement:activity",
        ]
        seen_block_ids: list[str] = []
        seen_feedbacks: list[str] = []
        seen_message_counts: list[int] = []
        seen_message_roles: list[list[str]] = []
        seen_tool_names: list[list[str]] = []
        trace_capture: dict[str, object] = {}
        call_index = 0
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = current or {}
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            priors = data["priors"]
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(priors)
            model_spec = current.get("model_spec")

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "initial measurement prior"
            ):
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "obs_ordered_base" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "sigma_activity" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if priors.get("rho_sleep", {}).get("reasoning") == "bad sleep dynamics prior":
                return {
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=False,
                        compile_error="rho_sleep interval instability",
                    ),
                }, "COMPILE ERROR:\nrho_sleep interval instability"

            if "rho_sleep" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "beta_activity_sleep" in priors:
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=False,
                        diagnostics=[
                            PriorValidationResult(
                                parameter="scale_activity_vas",
                                is_valid=False,
                                code="scale_mismatch",
                                origin="prior_predictive",
                                issue="Scale mismatch for activity_vas",
                                suggested_adjustment="Tighten the measurement prior",
                            )
                        ],
                    ),
                }, "PRIOR PREDICTIVE CHECKS FAILED"

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "corrected measurement prior"
            ):
                return {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                }, "VALID"

            raise AssertionError(f"Unexpected Stage 4 grounding payload: {data}")

        session = _make_stage4_session(
            question="Does activity improve sleep?",
            plan=plan,
            runtime=runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            stage4_grounding_fn=stub_stage4_grounding,
            model_topology={},
            loading_params=skeleton.loading_params,
            prior_cards=build_prior_cards(causal_spec, skeleton),
        )

        async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
            nonlocal call_index
            assert model_name == "test-model"
            assert tools is not None
            turn = session.current_turn()
            assert turn is not None
            seen_message_counts.append(len(messages))
            seen_message_roles.append([message["role"] for message in messages])
            seen_tool_names.append([tool.name for tool in tools])
            seen_block_ids.append(turn.block.id)
            seen_feedbacks.append(turn.latest_feedback)
            payload = submissions[call_index]
            call_index += 1
            tool_name = _stage4_submit_tool_name(str(payload["block_kind"]))
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{call_index}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(_stage4_submit_tool_args(payload)),
                            },
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        tool_map = {
            "submit_indicator_choice": make_submit_indicator_choice_tool(session),
            "submit_model_review": make_submit_model_review_tool(session),
            "submit_prior_block": make_submit_prior_block_tool(session),
        }
        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        generate = make_generate_fn(
            "test-model",
            config=GenerateConfig(),
            trace_capture=trace_capture,
        )
        completion = ""
        while not session.is_done():
            turn = session.current_turn()
            assert turn is not None
            completion = asyncio.run(
                _await_string(
                    generate(
                        turn.messages,
                        [tool_map[turn.required_submission_tool_name]],
                    )
                )
            )

        assert completion == ""
        assert seen_block_ids == expected_blocks
        assert call_index == len(submissions)
        assert seen_message_counts == [2] * len(submissions)
        assert seen_message_roles == [["system", "user"]] * len(submissions)
        assert seen_tool_names == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert seen_feedbacks == [
            "No validator feedback yet. Submit the active block only.",
            "MODEL STATE SAVED:\n- missing priors",
            "BLOCK ACCEPTED:\n- saved `review:model_spec`\n- next block: `measurement:activity` (measurement_prior)",
            "MODEL STATE SAVED:\n- missing priors",
            "MODEL STATE SAVED:\n- missing priors",
            "MODEL STATE SAVED:\n- missing priors",
            "COMPILE ERROR:\nrho_sleep interval instability",
            "MODEL STATE SAVED:\n- missing priors",
            "PRIOR PREDICTIVE CHECKS FAILED",
        ]

        trace = _require_trace(trace_capture)
        assert len(trace.messages) == 4 * len(submissions)
        assert [message.role for message in trace.messages[:2]] == ["system", "user"]
        assert sum(message.role == "assistant" for message in trace.messages) == len(submissions)
        assert sum(message.role == "tool" for message in trace.messages) == len(submissions)
        assert len(trace.messages) > max(seen_message_counts)
        assert any(
            message.tool_result == "COMPILE ERROR:\nrho_sleep interval instability"
            for message in trace.messages
        )
        assert any(
            message.tool_result == "PRIOR PREDICTIVE CHECKS FAILED" for message in trace.messages
        )
        assert runtime.accepted.model_spec is not None
        assert sorted(runtime.accepted.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "obs_ordered_base",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]
