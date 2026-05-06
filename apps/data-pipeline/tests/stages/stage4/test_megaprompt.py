"""Tests for the Stage 4 megaprompt (state-machine-disabled) mode.

The megaprompt mode exposes the same submit tools as the state-machine
path and routes every submission through the same grounding pipeline.
These tests drive the tool handlers directly so we can verify the
validation contract without spinning up a real LLM session.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
    make_stage4_grounding_result,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_megaprompt import (
    Stage4MegapromptState,
    _apply_indicator_choice,
    _apply_model_configuration,
    _apply_prior_block,
    _optional_prior_names_from_spec,
    _parameter_inventory_from_skeleton,
    _required_prior_names_from_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
    build_stage4_plan,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    derive_deterministic_spec,
)
from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
from tests.stages.stage4._support import _make_stage4_deps, make_causal_spec_dict

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_types import Stage4Deps


def _make_megaprompt_spec() -> dict:
    """Simple two-construct spec with one ambiguous indicator."""
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
                "name": "sleep_quality",
                "construct_name": "sleep",
                "measurement_dtype": "ordinal",
                "ordinal_levels": ["low", "high"],
                "how_to_measure": "Sleep quality rating",
                "aggregation": "last",
            },
        ],
    )


def _make_megaprompt_deps(causal_spec: dict, grounding_fn: Any) -> Stage4Deps:
    """Build a Stage4Deps wired to a stub grounding function for tests."""
    return _make_stage4_deps(
        causal_spec=causal_spec,
        skeleton=derive_deterministic_spec(causal_spec),
        stage4_grounding_fn=grounding_fn,
    )


class _GroundingRecorder:
    """Stub grounding function that records calls and replays scripted results."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        data: dict,
        causal_spec: dict,
        *,
        current: dict | None = None,
        data_for_model: Any = None,
        indicator_audits: dict | None = None,
        skip_ppc: bool = False,
    ) -> Any:
        del causal_spec, data_for_model, indicator_audits, skip_ppc
        call_record = {"data": deepcopy(data), "current": deepcopy(current or {})}
        self.calls.append(call_record)

        if "model_spec" in data:
            model_spec = data["model_spec"]
            return make_stage4_grounding_result(
                stage_output={
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                },
                status="accepted_pending_priors",
                feedback="MODEL STATE SAVED: missing priors",
                retain_for_next_prompt=True,
                capture_stage_output=True,
            )

        current = current or {}
        authored = dict(current.get("authored_priors") or {})
        authored.update(data["priors"])
        model_spec = current.get("model_spec")
        required = _required_prior_names_from_spec(model_spec)
        complete = required and all(name in authored for name in required)
        validation = AssemblyValidation(
            normalized_model_spec=model_spec,
            compile_ok=True,
            pp_checked=complete,
            pp_valid=True,
        )
        return make_stage4_grounding_result(
            stage_output={
                "model_spec": model_spec,
                "authored_priors": authored,
                "validation": validation,
            },
            status="accepted" if complete else "accepted_pending_priors",
            feedback="VALID" if complete else "MODEL STATE SAVED: more priors needed",
            validation=validation,
            retain_for_next_prompt=not complete,
            capture_stage_output=True,
        )


def test_apply_model_configuration_rejects_invalid_policy() -> None:
    spec = _make_megaprompt_spec()
    grounding = _GroundingRecorder()
    deps = _make_megaprompt_deps(spec, grounding)
    state = Stage4MegapromptState()

    feedback = _apply_model_configuration(
        state,
        deps=deps,
        ambiguous_variables=("sleep_quality",),
        initialization_policy="bogus",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        reasoning="x",
    )
    assert feedback.startswith("VALIDATION ERRORS")
    assert state.draft_model.initialization_policy is None
    assert not grounding.calls


def test_apply_indicator_choice_rejects_unknown_variable() -> None:
    spec = _make_megaprompt_spec()
    grounding = _GroundingRecorder()
    deps = _make_megaprompt_deps(spec, grounding)
    plan = build_stage4_plan(spec, deps.skeleton)
    plan_block_by_variable = {
        block.variable_names[0]: block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    }
    state = Stage4MegapromptState()

    feedback = _apply_indicator_choice(
        state,
        deps=deps,
        plan_block_by_variable=plan_block_by_variable,
        ambiguous_variables=tuple(plan_block_by_variable),
        variable="not_an_indicator",
        distribution="Normal",
        link="identity",
        reasoning="x",
    )
    assert feedback.startswith("VALIDATION ERRORS")
    assert not grounding.calls


def test_apply_prior_block_rejects_unknown_parameter() -> None:
    spec = _make_megaprompt_spec()
    grounding = _GroundingRecorder()
    deps = _make_megaprompt_deps(spec, grounding)
    state = Stage4MegapromptState()

    feedback = _apply_prior_block(
        state,
        deps=deps,
        parameter_inventory=_parameter_inventory_from_skeleton(deps.skeleton),
        priors={
            "beta_unknown_unknown": {
                "parameter": "beta_unknown_unknown",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 1.0},
                "sources": [],
                "reasoning": "x",
            }
        },
    )
    assert "outside the parameter inventory" in feedback
    assert not grounding.calls


def test_megaprompt_flow_locks_spec_and_authors_priors() -> None:
    spec = _make_megaprompt_spec()
    grounding = _GroundingRecorder()
    deps = _make_megaprompt_deps(spec, grounding)
    plan = build_stage4_plan(spec, deps.skeleton)
    plan_block_by_variable = {
        block.variable_names[0]: block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    }
    ambiguous_variables = tuple(item["variable"] for item in deps.skeleton.ambiguous_indicators)
    parameter_inventory = _parameter_inventory_from_skeleton(deps.skeleton)

    state = Stage4MegapromptState()
    configuration_feedback = _apply_model_configuration(
        state,
        deps=deps,
        ambiguous_variables=ambiguous_variables,
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        reasoning="Default.",
    )
    assert configuration_feedback.startswith("ACCEPTED model configuration")
    # The model spec cannot lock until every ambiguous indicator is chosen too.
    assert state.accepted.model_spec is None

    for variable in ambiguous_variables:
        block = plan_block_by_variable[variable]
        payload = block.payload
        if "fixed_distribution" in payload:
            distribution = str(payload["fixed_distribution"])
        else:
            distribution = str(payload["valid_distributions"][0])
        if payload.get("valid_links"):
            link = str(payload["valid_links"][0])
        else:
            link = str(payload["link_options"][distribution][0])
        feedback = _apply_indicator_choice(
            state,
            deps=deps,
            plan_block_by_variable=plan_block_by_variable,
            ambiguous_variables=ambiguous_variables,
            variable=variable,
            distribution=distribution,
            link=link,
            reasoning="scripted test pick",
        )
        assert feedback.startswith("ACCEPTED indicator choice")
    assert state.accepted.model_spec is not None, "model spec should lock after all decisions"

    required = _required_prior_names_from_spec(state.accepted.model_spec)
    optional = _optional_prior_names_from_spec(state.accepted.model_spec)
    assert required, "expected at least one required prior"
    assert set(optional).isdisjoint(required)

    priors = {
        name: {
            "parameter": name,
            "distribution": "Normal",
            "params": {"mu": 0.0, "sigma": 1.0},
            "sources": [],
            "reasoning": "test prior",
        }
        for name in required
    }
    final_feedback = _apply_prior_block(
        state,
        deps=deps,
        parameter_inventory=parameter_inventory,
        priors=priors,
    )
    assert final_feedback == "VALID"
    assert state.is_done()
    assert set(state.accepted.authored_priors) == set(required)


def test_is_done_requires_full_required_prior_coverage() -> None:
    """Partial prior coverage must not satisfy the done predicate.

    :func:`stage4_grounding` passes ``None`` into :func:`validate_assembly`
    when the authored priors do not cover every required parameter. That
    skips the prior-predictive pass and leaves ``validation.is_valid``
    trivially ``True``. The state machine avoids this because it only
    reaches the terminal cursor after every prior block is accepted; the
    megaprompt has to enforce the same invariant explicitly on the
    accepted state.
    """
    spec = _make_megaprompt_spec()
    grounding = _GroundingRecorder()
    deps = _make_megaprompt_deps(spec, grounding)
    plan = build_stage4_plan(spec, deps.skeleton)
    plan_block_by_variable = {
        block.variable_names[0]: block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    }
    ambiguous_variables = tuple(item["variable"] for item in deps.skeleton.ambiguous_indicators)
    parameter_inventory = _parameter_inventory_from_skeleton(deps.skeleton)

    state = Stage4MegapromptState()
    _apply_model_configuration(
        state,
        deps=deps,
        ambiguous_variables=ambiguous_variables,
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        reasoning="partial-coverage regression test",
    )
    for variable in ambiguous_variables:
        block = plan_block_by_variable[variable]
        payload = block.payload
        if "fixed_distribution" in payload:
            distribution = str(payload["fixed_distribution"])
        else:
            distribution = str(payload["valid_distributions"][0])
        if payload.get("valid_links"):
            link = str(payload["valid_links"][0])
        else:
            link = str(payload["link_options"][distribution][0])
        _apply_indicator_choice(
            state,
            deps=deps,
            plan_block_by_variable=plan_block_by_variable,
            ambiguous_variables=ambiguous_variables,
            variable=variable,
            distribution=distribution,
            link=link,
            reasoning="partial-coverage regression test",
        )
    assert state.accepted.model_spec is not None

    required = _required_prior_names_from_spec(state.accepted.model_spec)
    assert required, "regression test needs at least one required prior"

    partial_priors = {
        required[0]: {
            "parameter": required[0],
            "distribution": "Normal",
            "params": {"mu": 0.0, "sigma": 1.0},
            "sources": [],
            "reasoning": "single partial prior",
        }
    }
    feedback = _apply_prior_block(
        state,
        deps=deps,
        parameter_inventory=parameter_inventory,
        priors=partial_priors,
    )
    # Grounding reports a compile-ok validation without running PPC because
    # priors are incomplete; ``validation.is_valid`` would be True in
    # isolation. ``is_done`` must still return False.
    assert "VALID" not in feedback
    validation = state.accepted.validation
    assert validation is not None
    assert validation.is_valid
    assert not state.is_done(), (
        "is_done() must require every required prior to be authored, not just "
        "a compile-ok validation"
    )


def test_checkpoint_serde_roundtrip() -> None:
    """``serialize_*`` + ``deserialize_*`` must be a round-trip for the fields
    that actually matter for resume: draft_model, accepted.model_spec,
    authored_priors, search_cache/queries, last_feedback, tool_call_count.
    Ephemeral validation fields (compiled_ssm, pp_raw_samples) are
    intentionally dropped.
    """
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_megaprompt import (
        deserialize_stage4_megaprompt_state,
        serialize_stage4_megaprompt_state,
    )

    state = Stage4MegapromptState()
    state.draft_model.initialization_policy = "stationary"
    state.draft_model.observation_intercept_policy = "free"
    state.draft_model.equilibrium_forcing = False
    state.draft_model.distribution_choices["sleep_quality"] = {
        "variable": "sleep_quality",
        "distribution": "ordered_logit",
        "link": "logit",
        "reasoning": "ordinal",
    }
    state.accepted.model_spec = {"parameters": [{"name": "rho_x"}]}
    state.accepted.authored_priors = {
        "rho_x": {
            "parameter": "rho_x",
            "distribution": "Beta",
            "params": {"alpha": 5, "beta": 5},
            "sources": [],
            "reasoning": "checkpoint test",
        }
    }
    state.search_cache = {"q1": "r1"}
    state.search_queries = {"rho_x": "q1"}
    state.last_feedback = "VALID"
    state.tool_call_count = 17

    payload = serialize_stage4_megaprompt_state(state)
    restored = deserialize_stage4_megaprompt_state(payload)

    assert restored.draft_model.initialization_policy == "stationary"
    assert restored.draft_model.observation_intercept_policy == "free"
    assert restored.draft_model.equilibrium_forcing is False
    assert restored.draft_model.distribution_choices == state.draft_model.distribution_choices
    assert restored.accepted.model_spec == state.accepted.model_spec
    assert restored.accepted.authored_priors == state.accepted.authored_priors
    assert restored.search_cache == state.search_cache
    assert restored.search_queries == state.search_queries
    assert restored.last_feedback == "VALID"
    assert restored.tool_call_count == 17


def test_parameter_inventory_matches_skeleton_names() -> None:
    spec = _make_megaprompt_spec()
    skeleton = derive_deterministic_spec(spec)
    inventory = _parameter_inventory_from_skeleton(skeleton)
    assert all(isinstance(name, str) for name in inventory)
    # The skeleton's deterministic parameter list is the source of truth for the
    # locked ModelSpec; inventory must be a subset thereof.
    assert set(inventory) == {str(p["name"]) for p in skeleton.all_params if isinstance(p, dict)}
