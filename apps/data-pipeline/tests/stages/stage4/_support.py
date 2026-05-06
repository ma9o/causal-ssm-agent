"""Shared fixtures for Stage 4 tests used by multiple test modules.

Helpers needed by only one test file (e.g. ``test_reducer_flow.py``)
live next to that file (``_reducer_flow_support.py``); single-use
utilities are inlined in their consumer.
"""

# ruff: noqa: F401

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

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop import (
    _load_resumable_stage4_runtime,
    _validate_stage4_runtime_checkpoint,
    run_stage4,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_cards import build_prior_cards
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
    Stage4GroundingResult,
    make_stage4_grounding_result,
    make_stage4_validation_packet,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_navigation import (
    _set_block_cursor,
    _set_done_cursor,
    get_active_plan_block,
    get_stage4_phase,
    make_stage4_runtime,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    Stage4RepairTopology,
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
    classify_validation_outcome,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_runtime_projections import (
    project_stage4_graph,
    project_stage4_initial_state,
    project_stage4_snapshot,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_session import (
    Stage4FatalSubmissionError,
    Stage4Session,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    Stage4Skeleton,
    derive_deterministic_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import (
    Stage4AcceptedArtifacts,
    Stage4DomainState,
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
from causal_ssm_agent.utils.openrouter_client import GenerateConfig, Tool, execute_tools
from causal_ssm_agent.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorRepairScope,
    PriorValidationResult,
)
from tests.ssm_test_utils import make_ssm_spec


def _make_plan(
    *,
    model_blocks: tuple[Stage4FrontierBlock, ...] = (),
    review_block: Stage4FrontierBlock | None = None,
    prior_blocks: tuple[Stage4FrontierBlock, ...] = (),
    prior_review_block: Stage4FrontierBlock | None = None,
) -> Stage4Plan:
    """Build a minimal Stage 4 plan for focused unit tests."""
    all_blocks = (
        *model_blocks,
        *((review_block,) if review_block is not None else ()),
        *prior_blocks,
        *((prior_review_block,) if prior_review_block is not None else ()),
    )
    blocks_by_id = {block.id: block for block in all_blocks}
    parameter_to_block_id: dict[str, str] = {}
    indicator_to_decision_block_id: dict[str, str] = {}
    indicator_to_measurement_block_id: dict[str, str] = {}

    for block in prior_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "measurement_prior":
            for indicator_name in block.variable_names:
                indicator_to_measurement_block_id[indicator_name] = block.id

    for block in model_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "indicator_decision":
            for indicator_name in block.variable_names:
                indicator_to_decision_block_id[indicator_name] = block.id

    return Stage4Plan(
        model_blocks=model_blocks,
        review_block=review_block,
        prior_blocks=prior_blocks,
        prior_review_block=prior_review_block,
        blocks_by_id=blocks_by_id,
        repair_topology=Stage4RepairTopology(
            parameter_to_block_id=parameter_to_block_id,
            indicator_to_decision_block_id=indicator_to_decision_block_id,
            indicator_to_measurement_block_id=indicator_to_measurement_block_id,
        ),
    )


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


def make_causal_spec_dict(
    constructs: list[dict],
    edges: list[dict],
    indicators: list[dict],
    *,
    model_clock: str | None = "1d",
) -> dict:
    """Build a CausalSpec dict (latent + measurement + estimation) for tests.

    Defaults indicator polarity to ``positive`` when not set; estimation block
    is derived from ``constructs`` (state_order = construct names) and ``edges``.
    Pass ``model_clock=None`` to omit the field entirely.
    """
    indicators = [
        {"construct_polarity": "positive", **indicator}
        if "construct_polarity" not in indicator
        else dict(indicator)
        for indicator in indicators
    ]
    measurement: dict = {"indicators": indicators}
    if model_clock is not None:
        measurement["model_clock"] = model_clock
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": measurement,
        "estimation": {
            "state_order": [c["name"] for c in constructs],
            "edges": edges,
            "induced_dependencies": [],
        },
    }


def _make_runtime(
    plan: Stage4Plan,
    *,
    phase: str | None = None,
    active_block_id: str | None = None,
    accepted: Stage4AcceptedArtifacts | None = None,
    last_validation_packet: Any = None,
) -> Stage4Runtime:
    """Build a Stage 4 runtime for focused unit tests."""
    runtime = make_stage4_runtime(plan)
    if active_block_id is not None:
        _set_runtime_block(plan, runtime, active_block_id)
    elif phase == "done":
        _set_done_cursor(runtime)
    elif phase == "global_review" and plan.review_block is not None:
        _set_block_cursor(runtime, plan.review_block)
    elif phase == "global_prior_review" and plan.prior_review_block is not None:
        _set_block_cursor(runtime, plan.prior_review_block)
    elif phase == "prior_blocks" and plan.prior_blocks:
        _set_block_cursor(runtime, plan.prior_blocks[0])
    elif phase == "model_decisions" and plan.model_blocks:
        _set_block_cursor(runtime, plan.model_blocks[0])
    if accepted is not None:
        runtime.domain.accepted = accepted
    runtime.interaction.last_validation_packet = last_validation_packet
    return runtime


def _set_runtime_block(plan: Stage4Plan, runtime: Stage4Runtime, block_id: str) -> None:
    """Move a test runtime onto one promptable Stage 4 block."""
    block = plan.get_block(block_id)
    assert block is not None
    _set_block_cursor(runtime, block)


def _with_positive_indicator_polarity(spec: dict[str, Any]) -> dict[str, Any]:
    """Backfill valid default indicator semantics for Stage 4 test fixtures."""
    spec = deepcopy(spec)
    measurement = spec.get("measurement") or {}
    indicators = measurement.get("indicators") or []
    for indicator in indicators:
        if isinstance(indicator, dict):
            indicator.setdefault("construct_polarity", "positive")
            dtype = indicator.get("measurement_dtype")
            aggregation = indicator.get("aggregation")
            if not isinstance(aggregation, str):
                if dtype == "continuous":
                    indicator["aggregation"] = "mean"
                elif dtype == "count":
                    indicator["aggregation"] = "sum"
                else:
                    indicator["aggregation"] = "last"
                aggregation = indicator["aggregation"]
            if dtype in {"binary", "ordinal", "categorical"} and aggregation not in {
                "first",
                "last",
            }:
                indicator["aggregation"] = "last"
    return spec


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


def _accept_default_model_configuration(
    *,
    causal_spec: dict[str, Any],
    skeleton: Stage4Skeleton,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    data_for_model: pl.DataFrame | None = None,
) -> None:
    """Advance one test runtime past the mandatory model-configuration block."""
    if data_for_model is None:
        data_for_model = pl.DataFrame()
    active_block = get_active_plan_block(plan, runtime)
    if active_block is None or active_block.id != "model:configuration":
        return

    _stage_output, feedback = _apply_stage4_step_and_capture(
        {
            "block_id": "model:configuration",
            "block_kind": "model_configuration",
            "proposal": {
                "initialization_policy": "stationary",
                "observation_intercept_policy": "free",
                "equilibrium_forcing": False,
                "reasoning": "Default Stage 4 test configuration.",
            },
        },
        plan,
        runtime,
        skeleton=skeleton,
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        indicator_audits={},
        stage4_grounding_fn=lambda data, *_args, **_kwargs: (
            (
                {
                    "model_spec": data["model_spec"],
                    "validation": AssemblyValidation(
                        normalized_model_spec=data["model_spec"],
                        compile_ok=True,
                    ),
                },
                "MODEL STATE SAVED:\n- missing priors",
            )
            if isinstance(data, dict) and "model_spec" in data
            else pytest.fail("unexpected non-model-spec grounding during model configuration")
        ),
    )

    assert not feedback.startswith("VALIDATION ERRORS:")
    runtime.interaction.last_validation_packet = None


def _make_stage4_mechanics_context(
    *,
    accept_default_configuration: bool = False,
) -> tuple[dict[str, Any], Stage4Skeleton, Stage4Plan, Stage4Runtime, pl.DataFrame]:
    """Build the standard deterministic Stage 4 mechanics fixture."""
    causal_spec = _make_stage4_mechanics_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    data_for_model = pl.DataFrame()
    if accept_default_configuration:
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
            data_for_model=data_for_model,
        )
    return causal_spec, skeleton, plan, runtime, data_for_model


def _make_stage4_deps(
    *,
    causal_spec: dict[str, Any],
    skeleton: Stage4Skeleton,
    stage4_grounding_fn,
    data_for_model: pl.DataFrame | None = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
) -> Stage4Deps:
    """Build a Stage 4 reducer environment for tests."""
    if data_for_model is None:
        data_for_model = pl.DataFrame()
    if indicator_audits is None:
        indicator_audits = {}

    def _wrap_grounding(*args, **kwargs):
        result = stage4_grounding_fn(*args, **kwargs)
        if isinstance(result, Stage4GroundingResult):
            return result
        stage_output, feedback = result
        validation = stage_output.get("validation") if isinstance(stage_output, dict) else None
        status = (
            "compile_error"
            if validation is not None and getattr(validation, "compile_ok", True) is False
            else "sensitivity_failure"
            if validation is not None and getattr(validation, "has_sensitivity_failure", False)
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
    stage4_grounding_fn,
    data_for_model: pl.DataFrame | None = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
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
    stage4_grounding_fn,
    data_for_model: pl.DataFrame | None = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
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


def _stage4_test_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy block-envelope fixtures into reducer payloads."""
    proposal = payload.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    return payload


def _require_plan_block(plan: Stage4Plan, block_id: str) -> Stage4FrontierBlock:
    """Fetch a Stage 4 block and assert the test fixture contains it."""
    block = plan.get_block(block_id)
    assert block is not None
    return block


def _make_polars_data() -> pl.DataFrame:
    """Long-format polars data for Stage 4 SSM-validation tests."""
    import pandas as pd

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


@pytest.fixture
def simple_model_spec() -> dict:
    """Minimal Stage 4 model spec used by SSM-validation tests."""
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
    """Priors matching ``simple_model_spec``."""
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
    """Tabular fixture aligned with ``simple_model_spec``."""
    n = 50
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mood_score": rng.normal(5, 1.5, n),
            "mood_score_lag1": rng.normal(5, 1.5, n),
            "subject_id": np.repeat(np.arange(5), 10),
        }
    )
