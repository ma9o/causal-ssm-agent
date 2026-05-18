"""Synthetic local-vs-global tests for the shared Stage 4 tool surface."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.flows.stages.stage4.agentic.stage4_feedback import (
    Stage4GroundingResult,
    make_stage4_grounding_result,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_megaprompt import (
    Stage4MegapromptSessionAdapter,
    Stage4MegapromptState,
    _make_megaprompt_tools,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_navigation import (
    _set_block_cursor,
    make_stage4_runtime,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    build_stage4_plan,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_prompt_context import Stage4Messages
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_reducer import (
    build_model_spec_from_decisions,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_session import Stage4Session
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_skeleton import (
    Stage4Skeleton,
    derive_deterministic_spec,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_state import (
    Stage4DraftModel,
    Stage4Runtime,
)
from nof1_causal_lab.flows.stages.stage4.assembly import (
    AssemblyValidation,
    format_validation_feedback,
)
from nof1_causal_lab.flows.stages.stage4.tool_registry import (
    allowed_stage4_tool_names,
    build_stage4_session_tool_map,
)
from nof1_causal_lab.flows.stages.stage4.tools import (
    make_search_tool,
    make_submit_indicator_choice_tool,
    make_submit_model_configuration_tool,
    make_submit_model_review_tool,
    make_submit_prior_block_tool,
)
from nof1_causal_lab.workers.schemas_prior import PriorValidationResult
from tests.stages.stage4._support import _make_stage4_deps, make_causal_spec_dict

if TYPE_CHECKING:
    import pytest

    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_types import Stage4Deps


def _make_shared_tool_spec() -> dict[str, Any]:
    """Return a minimal Stage 4 spec with two ambiguous indicators."""
    return make_causal_spec_dict(
        constructs=[
            {"name": "activity", "role": "exogenous", "temporal_status": "time_varying"},
            {
                "name": "sleep",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
        ],
        edges=[{"cause": "activity", "effect": "sleep"}],
        indicators=[
            {
                "name": "steps",
                "construct_name": "activity",
                "measurement_dtype": "count",
                "how_to_measure": "Daily step count",
                "aggregation": "sum",
            },
            {
                "name": "awakenings",
                "construct_name": "sleep",
                "measurement_dtype": "count",
                "how_to_measure": "Night awakenings count",
                "aggregation": "sum",
            },
        ],
    )


def _parameter_inventory_from_skeleton(skeleton: Stage4Skeleton) -> tuple[str, ...]:
    """Return the deterministic parameter inventory from the skeleton."""
    return tuple(
        str(parameter["name"])
        for parameter in skeleton.all_params
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    )


def _valid_indicator_choice(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Return one valid distribution/link payload for an indicator block."""
    payload = block.payload
    variable = block.variable_names[0]
    if "fixed_distribution" in payload:
        distribution = str(payload["fixed_distribution"])
        links = payload.get("valid_links") or []
    else:
        valid_distributions = payload.get("valid_distributions") or []
        distribution = str(valid_distributions[0])
        links = payload.get("link_options", {}).get(distribution) or []
    return {
        "variable": variable,
        "distribution": distribution,
        "link": str(links[0]),
        "reasoning": f"Valid synthetic choice for {variable}.",
    }


def _make_grounding_result(
    *,
    stage_output: dict[str, Any] | None,
    feedback: str,
    validation: AssemblyValidation | None = None,
    status: str = "accepted",
) -> Stage4GroundingResult:
    """Wrap a synthetic grounding payload in the typed result shape."""
    return make_stage4_grounding_result(
        stage_output=stage_output,
        status=status,
        feedback=feedback,
        validation=validation,
        retain_for_next_prompt=feedback != "VALID",
        capture_stage_output=stage_output is not None,
    )


def _make_deps(
    *,
    causal_spec: dict[str, Any],
    skeleton: Stage4Skeleton,
    grounding_fn,
) -> Stage4Deps:
    """Build synthetic Stage 4 deps for shared-tool tests."""
    return _make_stage4_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        stage4_grounding_fn=grounding_fn,
    )


def _make_local_session(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> Stage4Session:
    """Build a local state-machine session for direct tool execution."""
    return Stage4Session(
        plan=plan,
        prompt_context=Stage4Messages(question="Synthetic Stage 4 tool test."),
        deps=deps,
        runtime=runtime,
    )


def _make_global_adapter(
    *,
    state: Stage4MegapromptState,
    deps: Stage4Deps,
    plan: Stage4Plan,
    skeleton: Stage4Skeleton,
) -> Stage4MegapromptSessionAdapter:
    """Build the megaprompt adapter used by the shared tool factories."""
    plan_block_by_variable = {
        block.variable_names[0]: block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    }
    ambiguous_variables = tuple(plan_block_by_variable)
    return Stage4MegapromptSessionAdapter(
        state,
        deps=deps,
        plan_block_by_variable=plan_block_by_variable,
        ambiguous_variables=ambiguous_variables,
        parameter_inventory=_parameter_inventory_from_skeleton(skeleton),
    )


def _locked_model_spec(
    *,
    plan: Stage4Plan,
    skeleton: Stage4Skeleton,
) -> dict[str, Any]:
    """Return a fully locked synthetic ModelSpec for prior-tool tests."""
    draft_model = Stage4DraftModel(
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
    )
    for block in plan.model_blocks:
        if block.kind != "indicator_decision":
            continue
        choice = _valid_indicator_choice(block)
        draft_model.distribution_choices[choice["variable"]] = choice
    model_spec, errors = build_model_spec_from_decisions(draft_model, skeleton)
    assert model_spec is not None, errors
    return model_spec


def _distinct_prior_blocks(plan: Stage4Plan) -> tuple[Stage4FrontierBlock, Stage4FrontierBlock]:
    """Return two distinct prior blocks with parameter surfaces."""
    blocks = [block for block in plan.prior_blocks if block.parameter_names]
    for index, first in enumerate(blocks):
        for second in blocks[index + 1 :]:
            if set(first.parameter_names) != set(second.parameter_names):
                return first, second
    raise AssertionError("synthetic spec did not produce two distinct prior blocks")


def _set_active_block(plan: Stage4Plan, runtime: Stage4Runtime, block_id: str) -> None:
    """Move a synthetic runtime cursor onto one promptable block."""
    block = plan.get_block(block_id)
    assert block is not None
    _set_block_cursor(runtime, block)


def _call_tool(tool, **kwargs: Any) -> str:
    """Execute one Stage 4 tool in tests."""
    return asyncio.run(tool(**kwargs))


def _unexpected_grounding(*_args, **_kwargs):
    """Fail fast when a tool unexpectedly reaches the grounding layer."""
    raise AssertionError("grounding should not run in this synthetic tool test")


def _synthetic_prior(parameter_name: str) -> dict[str, Any]:
    """Return one simple Normal prior payload."""
    return {
        "parameter": parameter_name,
        "distribution": "Normal",
        "params": {"mu": 0.0, "sigma": 0.2},
        "sources": [],
        "reasoning": f"Synthetic prior for {parameter_name}.",
    }


def test_model_configuration_tool_differs_between_local_and_global_modes() -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    deps = _make_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        grounding_fn=_unexpected_grounding,
    )

    local_runtime = make_stage4_runtime(plan)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = make_submit_model_configuration_tool(local_session)

    global_state = Stage4MegapromptState()
    global_tool = make_submit_model_configuration_tool(
        _make_global_adapter(state=global_state, deps=deps, plan=plan, skeleton=skeleton),
        stop_on_success=False,
    )

    args = {
        "initialization_policy": "stationary",
        "observation_intercept_policy": "free",
        "equilibrium_forcing": False,
        "reasoning": "Synthetic configuration.",
    }
    local_feedback = _call_tool(local_tool, **args)
    global_feedback = _call_tool(global_tool, **args)

    assert local_tool.stop_on_success is True
    assert global_tool.stop_on_success is False
    assert local_feedback.startswith("BLOCK ACCEPTED:")
    assert global_feedback.startswith("ACCEPTED model configuration:")
    assert local_runtime.domain.draft_model.initialization_policy == "stationary"
    assert global_state.draft_model.initialization_policy == "stationary"


def test_indicator_choice_tool_is_block_scoped_locally_but_variable_routed_globally() -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    deps = _make_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        grounding_fn=_unexpected_grounding,
    )

    local_runtime = make_stage4_runtime(plan)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    _call_tool(
        make_submit_model_configuration_tool(local_session),
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        reasoning="Advance to indicator phase.",
    )

    indicator_blocks = [
        block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    ]
    assert len(indicator_blocks) >= 2
    first_block, second_block = indicator_blocks[:2]
    assert local_session.current_block() is not None
    assert local_session.current_block().id == first_block.id

    local_tool = make_submit_indicator_choice_tool(local_session)
    local_feedback = _call_tool(local_tool, **_valid_indicator_choice(second_block))

    global_state = Stage4MegapromptState()
    global_tool = make_submit_indicator_choice_tool(
        _make_global_adapter(state=global_state, deps=deps, plan=plan, skeleton=skeleton),
        stop_on_success=False,
    )
    global_feedback = _call_tool(global_tool, **_valid_indicator_choice(second_block))

    assert "proposal variable must be" in local_feedback
    assert first_block.variable_names[0] not in global_state.draft_model.distribution_choices
    assert second_block.variable_names[0] in global_state.draft_model.distribution_choices
    assert global_feedback.startswith("ACCEPTED indicator choice:")


def test_prior_tool_rejects_foreign_block_parameters_locally_but_accepts_global_inventory() -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    model_spec = _locked_model_spec(plan=plan, skeleton=skeleton)

    def grounding_fn(
        data: dict[str, Any],
        _causal_spec: dict[str, Any],
        *,
        current: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Stage4GroundingResult:
        current_state = {} if current is None else deepcopy(current)
        authored_priors = dict(current_state.get("authored_priors") or {})
        authored_priors.update(data["priors"])
        return _make_grounding_result(
            stage_output={
                "model_spec": current_state.get("model_spec"),
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current_state.get("model_spec"),
                    compile_ok=True,
                ),
            },
            feedback="MODEL STATE SAVED: more priors needed",
            validation=AssemblyValidation(
                normalized_model_spec=current_state.get("model_spec"),
                compile_ok=True,
            ),
            status="accepted_pending_priors",
        )

    deps = _make_deps(causal_spec=causal_spec, skeleton=skeleton, grounding_fn=grounding_fn)
    local_block, foreign_block = _distinct_prior_blocks(plan)
    foreign_parameter = foreign_block.parameter_names[0]

    local_runtime = make_stage4_runtime(plan)
    local_runtime.domain.accepted.model_spec = model_spec
    _set_active_block(plan, local_runtime, local_block.id)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = make_submit_prior_block_tool(local_session)
    local_feedback = _call_tool(
        local_tool,
        priors={foreign_parameter: _synthetic_prior(foreign_parameter)},
    )

    global_state = Stage4MegapromptState()
    global_state.accepted.model_spec = model_spec
    global_tool = make_submit_prior_block_tool(
        _make_global_adapter(state=global_state, deps=deps, plan=plan, skeleton=skeleton),
        stop_on_success=False,
    )
    global_feedback = _call_tool(
        global_tool,
        priors={foreign_parameter: _synthetic_prior(foreign_parameter)},
    )

    assert "priors outside the active block" in local_feedback
    assert foreign_parameter in local_feedback
    assert global_feedback == "MODEL STATE SAVED: more priors needed"
    assert foreign_parameter in global_state.accepted.authored_priors


def test_prior_tool_scopes_local_feedback_but_keeps_global_feedback_full() -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    model_spec = _locked_model_spec(plan=plan, skeleton=skeleton)
    local_block, _foreign_block = _distinct_prior_blocks(plan)
    local_parameters = tuple(local_block.required_parameter_names or local_block.parameter_names)
    local_parameter = local_parameters[0]
    unknown_foreign_parameter = "beta_unknown_unknown"

    def grounding_fn(
        data: dict[str, Any],
        _causal_spec: dict[str, Any],
        *,
        current: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Stage4GroundingResult:
        current_state = {} if current is None else deepcopy(current)
        authored_priors = dict(current_state.get("authored_priors") or {})
        authored_priors.update(data["priors"])
        validation = AssemblyValidation(
            normalized_model_spec=current_state.get("model_spec"),
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter=local_parameter,
                    is_valid=False,
                    code="local_scale_failure",
                    origin="prior_predictive",
                    issue=f"{local_parameter} local prior is too wide.",
                    suggested_adjustment="Tighten the local parameter.",
                ),
                PriorValidationResult(
                    parameter=unknown_foreign_parameter,
                    is_valid=False,
                    code="foreign_scale_failure",
                    origin="prior_predictive",
                    issue=f"{unknown_foreign_parameter} foreign prior is too wide.",
                    suggested_adjustment="Tighten the foreign parameter.",
                ),
            ],
        )
        feedback = format_validation_feedback(validation, authored_priors)
        return _make_grounding_result(
            stage_output={
                "model_spec": current_state.get("model_spec"),
                "authored_priors": authored_priors,
                "validation": validation,
            },
            feedback=feedback,
            validation=validation,
            status="prior_predictive_failure",
        )

    deps = _make_deps(causal_spec=causal_spec, skeleton=skeleton, grounding_fn=grounding_fn)
    local_priors = {name: _synthetic_prior(name) for name in local_parameters}

    local_runtime = make_stage4_runtime(plan)
    local_runtime.domain.accepted.model_spec = model_spec
    _set_active_block(plan, local_runtime, local_block.id)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = make_submit_prior_block_tool(local_session)
    local_feedback = _call_tool(local_tool, priors=local_priors)

    global_state = Stage4MegapromptState()
    global_state.accepted.model_spec = model_spec
    global_tool = make_submit_prior_block_tool(
        _make_global_adapter(state=global_state, deps=deps, plan=plan, skeleton=skeleton),
        stop_on_success=False,
    )
    global_feedback = _call_tool(global_tool, priors=local_priors)

    assert local_parameter in local_feedback
    assert unknown_foreign_parameter not in local_feedback
    assert local_runtime.interaction.last_validation_packet is not None
    assert local_runtime.interaction.last_validation_packet.failing_parameters == (local_parameter,)
    assert (
        unknown_foreign_parameter
        in local_runtime.interaction.last_validation_packet.coupled_parameters
    )

    assert local_parameter in global_feedback
    assert unknown_foreign_parameter in global_feedback


def test_search_tool_records_queries_and_caches_separately_in_local_and_global_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    deps = _make_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        grounding_fn=_unexpected_grounding,
    )

    calls: list[str] = []

    async def fake_search(query: str) -> str:
        calls.append(query)
        return f"evidence::{query}"

    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage4.tools.search_literature", fake_search)

    local_runtime = make_stage4_runtime(plan)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = make_search_tool(local_session)

    global_state = Stage4MegapromptState()
    global_tool = make_search_tool(
        _make_global_adapter(state=global_state, deps=deps, plan=plan, skeleton=skeleton)
    )

    query = "daily activity sleep longitudinal effect sizes"
    local_result_1 = _call_tool(local_tool, query=query, parameter_name="beta_activity_sleep")
    local_result_2 = _call_tool(local_tool, query=query, parameter_name="beta_activity_sleep")
    global_result_1 = _call_tool(global_tool, query=query, parameter_name="beta_activity_sleep")
    global_result_2 = _call_tool(global_tool, query=query, parameter_name="beta_activity_sleep")

    assert local_result_1 == "evidence::" + query
    assert local_result_2 == local_result_1
    assert global_result_1 == local_result_1
    assert global_result_2 == global_result_1
    assert calls == [query, query]
    assert local_session.search_queries == {"beta_activity_sleep": query}
    assert global_state.search_queries == {"beta_activity_sleep": query}


def test_model_review_tool_is_local_only_and_absent_from_megaprompt_surface() -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    deps = _make_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        grounding_fn=_unexpected_grounding,
    )

    assert plan.review_block is not None
    assert allowed_stage4_tool_names(plan.review_block.kind) == ("submit_model_review",)

    local_runtime = make_stage4_runtime(plan)
    _set_active_block(plan, local_runtime, plan.review_block.id)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = make_submit_model_review_tool(local_session)
    local_feedback = _call_tool(
        local_tool,
        decision="approve",
        reasoning="Synthetic global review approval.",
    )

    global_tools = _make_megaprompt_tools(
        Stage4MegapromptState(),
        deps=deps,
        plan_block_by_variable={},
        ambiguous_variables=(),
        parameter_inventory=_parameter_inventory_from_skeleton(skeleton),
        enable_literature=True,
        enable_paraphrasing=False,
        question="Synthetic Stage 4 tool test.",
        paraphrase_session_factory=object(),
        n_paraphrases=3,
    )

    assert local_tool.stop_on_success is True
    assert local_feedback.startswith("BLOCK ACCEPTED:")
    assert "submit_model_review" not in {tool.name for tool in global_tools}


def test_elicit_prior_gmm_tool_matches_between_local_and_global_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    causal_spec = _make_shared_tool_spec()
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    deps = _make_deps(
        causal_spec=causal_spec,
        skeleton=skeleton,
        grounding_fn=_unexpected_grounding,
    )
    prior_block = next(block for block in plan.prior_blocks if block.kind != "global_prior_review")
    paraphrase_factory = object()
    calls: list[dict[str, Any]] = []

    async def fake_elicitation(**kwargs: Any) -> str:
        calls.append(kwargs)
        return f"gmm::{kwargs['parameter_name']}::{kwargs['n_paraphrases']}"

    monkeypatch.setattr(
        "nof1_causal_lab.workers.prior_research.run_gmm_elicitation",
        fake_elicitation,
    )

    assert "elicit_prior_gmm" in allowed_stage4_tool_names(prior_block.kind)

    local_runtime = make_stage4_runtime(plan)
    _set_active_block(plan, local_runtime, prior_block.id)
    local_session = _make_local_session(plan=plan, runtime=local_runtime, deps=deps)
    local_tool = build_stage4_session_tool_map(
        local_session,
        question="Synthetic Stage 4 tool test.",
        enable_literature=False,
        enable_paraphrasing=True,
        n_paraphrases=3,
        paraphrase_session_factory=paraphrase_factory,
        max_tool_turns=4,
    )["elicit_prior_gmm"]

    global_tool = next(
        tool
        for tool in _make_megaprompt_tools(
            Stage4MegapromptState(),
            deps=deps,
            plan_block_by_variable={},
            ambiguous_variables=(),
            parameter_inventory=_parameter_inventory_from_skeleton(skeleton),
            enable_literature=False,
            enable_paraphrasing=True,
            question="Synthetic Stage 4 tool test.",
            paraphrase_session_factory=paraphrase_factory,
            n_paraphrases=3,
        )
        if tool.name == "elicit_prior_gmm"
    )

    args = {
        "parameter_name": "beta_activity_sleep",
        "parameter_role": "fixed_effect",
        "parameter_constraint": "none",
        "context": "Effect of activity on sleep in an intensive longitudinal setting.",
    }
    local_feedback = _call_tool(local_tool, **args)
    global_feedback = _call_tool(global_tool, **args)

    assert local_feedback == "gmm::beta_activity_sleep::3"
    assert global_feedback == local_feedback
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["question"] == "Synthetic Stage 4 tool test."
    assert calls[0]["session_factory"] is paraphrase_factory
