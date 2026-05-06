"""Shared fixtures and scripted helpers for Stage 4 tests."""

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


def derive_priors_from_model_spec(model_spec: dict) -> dict[str, dict]:
    """Generate weakly-informative priors for each parameter in a model spec.

    Picks the family by parameter name pattern: ``rho_*`` -> Beta(2,2),
    ``sigma_*`` -> HalfNormal(1), otherwise Normal(0, 0.5).
    """
    priors: dict[str, dict] = {}
    for p in model_spec["parameters"]:
        name = p["name"]
        if "rho" in name:
            priors[name] = {
                "parameter": name,
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "Weakly informative AR prior",
            }
        elif "sigma" in name:
            priors[name] = {
                "parameter": name,
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
                "sources": [],
                "reasoning": "Weakly informative SD prior",
            }
        else:
            priors[name] = {
                "parameter": name,
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "sources": [],
                "reasoning": "Weakly informative effect prior",
            }
    return priors


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


def _make_stub_grounding_result(stage_output: dict | None, feedback: str):
    """Wrap test grounding payloads in the typed Stage 4 grounding result."""
    validation = stage_output.get("validation") if isinstance(stage_output, dict) else None
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        status = "compile_error"
    elif validation is not None and getattr(validation, "has_sensitivity_failure", False):
        status = "sensitivity_failure"
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




# --- Shared Stage 4 mechanics helpers ---

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


def _activity_measurement_prior_bundle(
    *,
    lambda_sigma: float,
    lambda_reasoning: str,
    obs_sd_activity_vas_sigma: float = 0.5,
    obs_sd_steps_sigma: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Return a complete measurement block bundle for the mechanics fixtures."""
    return {
        "obs_sd_activity_vas": {
            "parameter": "obs_sd_activity_vas",
            "distribution": "HalfNormal",
            "params": {"sigma": obs_sd_activity_vas_sigma},
            "sources": [],
            "reasoning": "activity VAS measurement noise",
        },
        "obs_sd_steps": {
            "parameter": "obs_sd_steps",
            "distribution": "HalfNormal",
            "params": {"sigma": obs_sd_steps_sigma},
            "sources": [],
            "reasoning": "steps measurement noise",
        },
        "lambda_activity_vas_activity": {
            "parameter": "lambda_activity_vas_activity",
            "distribution": "HalfNormal",
            "params": {"sigma": lambda_sigma},
            "sources": [],
            "reasoning": lambda_reasoning,
        },
    }


def _activity_loading_prior_bundle(
    *,
    lambda_sigma: float,
    lambda_reasoning: str,
) -> dict[str, dict[str, Any]]:
    """Return only the activity loading prior for local repair prompts."""
    return {
        "lambda_activity_vas_activity": {
            "parameter": "lambda_activity_vas_activity",
            "distribution": "HalfNormal",
            "params": {"sigma": lambda_sigma},
            "sources": [],
            "reasoning": lambda_reasoning,
        },
    }


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


class _ScriptedStage4AgentSession:
    """Test adapter that drives the current StageSessionFactory API."""

    def __init__(self, generate, *, system_prompt: str | None, tools: list[Tool], log_label: str):
        self._generate = generate
        self._messages: list[dict[str, str]] = []
        if system_prompt is not None:
            self._messages.append({"role": "system", "content": system_prompt})
        self._tools = tools
        self._log_label = log_label
        self._completion = ""
        self._trace = LLMTrace()

    async def turn(self, user_message: str):
        self._messages.append({"role": "user", "content": user_message})
        self._completion = await self._generate(
            self._messages,
            self._tools,
            label=self._log_label,
        )
        return SimpleNamespace(
            completion=self._completion,
            terminal_tool_name=None,
            terminal_tool_output=None,
            tool_calls_fired=[],
        )

    @property
    def result(self):
        return SimpleNamespace(
            completion=self._completion,
            trace=self._trace,
            terminal_tool_name=None,
            terminal_tool_output=None,
        )


class _ScriptedStage4OpenContext:
    def __init__(self, session: _ScriptedStage4AgentSession):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ScriptedStage4SessionFactory:
    def __init__(self, generate):
        self._generate = generate
        self.accumulated_trace = LLMTrace()

    def open(
        self,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        log_label: str | None = None,
    ):
        return _ScriptedStage4OpenContext(
            _ScriptedStage4AgentSession(
                self._generate,
                system_prompt=system_prompt,
                tools=tools or [],
                log_label=log_label or "stage-4",
            )
        )


def _make_stage4_session_factory(generate) -> _ScriptedStage4SessionFactory:
    return _ScriptedStage4SessionFactory(generate)


def _stub_stage4_repair_barrier_success(_plan, runtime, _deps) -> tuple[dict[str, str], ...]:
    runtime.domain.accepted.validation = AssemblyValidation(
        normalized_model_spec=runtime.domain.accepted.model_spec,
        compile_ok=True,
        pp_checked=True,
        pp_valid=True,
        diagnostics=[],
    )
    runtime.domain.repair_campaign = None
    runtime.domain.active_block_id = None
    runtime.domain.done = True
    return ({"block_id": "repair:barrier", "status": "accepted"},)


def _stage4_submit_tool_name(block_kind: str) -> str:
    """Return the primary submit-tool name for one Stage 4 block kind."""
    if block_kind == "model_configuration":
        return "submit_model_configuration"
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
        if block_id == "model:configuration":
            submit_tool = next(tool for tool in tools if tool.name == "submit_model_configuration")
            feedback = await submit_tool(
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
                reasoning="Default Stage 4 test configuration.",
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(feedback)
            return ""
        if isinstance(block_id, str) and block_id.startswith("observation:manifest_mean_"):
            parameter = block_id.removeprefix("observation:")
            submit_tool = next(tool for tool in tools if tool.name == "submit_prior_block")
            feedback = await submit_tool(
                priors={
                    parameter: {
                        "parameter": parameter,
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "Default observation-intercept prior for scripted Stage 4 tests.",
                    }
                }
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(feedback)
            return ""
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
        if block_id == "model:configuration":
            submit_tool = next(tool for tool in tools if tool.name == "submit_model_configuration")
            feedback = await submit_tool(
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
                reasoning="Default Stage 4 test configuration.",
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(f"{block_id}: {feedback}")
            return ""
        if block_id.startswith("observation:manifest_mean_"):
            parameter = block_id.removeprefix("observation:")
            submit_tool = next(tool for tool in tools if tool.name == "submit_prior_block")
            feedback = await submit_tool(
                priors={
                    parameter: {
                        "parameter": parameter,
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "Default observation-intercept prior for scripted Stage 4 tests.",
                    }
                }
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(f"{block_id}: {feedback}")
            return ""
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


