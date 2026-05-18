"""Regression tests for shared Stage 4 prompt fragments."""

from __future__ import annotations

from nof1_causal_lab.flows.stages.stage4.agentic.prompts.accepted_state import (
    build_accepted_state_sections,
)
from nof1_causal_lab.flows.stages.stage4.agentic.prompts.megaprompt import (
    build_stage4_megaprompt_system_prompt,
    build_stage4_megaprompt_user_prompt,
)
from nof1_causal_lab.flows.stages.stage4.agentic.prompts.model_proposal import (
    build_stage4_system_prompt,
    build_stage4_user_prompt,
)
from nof1_causal_lab.flows.stages.stage4.agentic.prompts.shared_fragments import (
    CONTINUOUS_TIME_DYNAMICS_SECTION,
    INITIAL_STATE_SCALE_DISCIPLINE_SECTION,
    LAGGED_EFFECT_INTERVAL_GUIDANCE_SECTION,
    LINK_FUNCTION_RULES_SECTION,
    OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
    PRIOR_DISTRIBUTION_TYPES_SECTION,
    PRIOR_SOURCE_GUIDANCE,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_feedback import (
    Stage4ScopeSnapshot,
    default_stage4_validation_packet,
)


def test_shared_system_guidance_fragments_are_used_verbatim_in_both_modes() -> None:
    reducer_prompt = build_stage4_system_prompt(
        system_task="Author priors for the active scope.",
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
            "prior_distribution_types",
            "continuous_time_dynamics",
            "latent_initial_state_guidance",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=("beta",),
        submission_tool_name="submit_prior_block",
        enabled_tool_names=("submit_prior_block",),
    )
    megaprompt = build_stage4_megaprompt_system_prompt(
        enable_literature=False,
        enable_paraphrasing=False,
    )

    for fragment in (
        OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
        LINK_FUNCTION_RULES_SECTION,
        PRIOR_DISTRIBUTION_TYPES_SECTION,
        CONTINUOUS_TIME_DYNAMICS_SECTION,
        LAGGED_EFFECT_INTERVAL_GUIDANCE_SECTION,
        INITIAL_STATE_SCALE_DISCIPLINE_SECTION,
    ):
        assert reducer_prompt.count(fragment) == 1
        assert megaprompt.count(fragment) == 1


def test_prior_source_guidance_is_shared_verbatim_in_both_user_prompts() -> None:
    snapshot = Stage4ScopeSnapshot(
        block_id="prior:measurement",
        block_kind="measurement_prior",
        block_label="Measurement priors",
        block_instructions="Propose priors only for the active block.",
        frontier_status="ACTIVE FRONTIER (machine-generated)",
        model_topology={},
        distribution_cards=[],
        loading_params=[],
        construct_scale_cards=[],
        prior_cards=[],
        coupled_prior_cards=[],
        accepted_model_spec=None,
        accepted_authored_priors={},
        centerable_construct_names=(),
        baseline_factor_names=(),
        submission_example="Use `submit_prior_block` with this argument object.",
        include_prior_source_guidance=True,
        latest_validation=default_stage4_validation_packet(),
    )
    reducer_prompt = build_stage4_user_prompt(
        question="Does activity affect sleep?",
        snapshot=snapshot,
    )
    megaprompt = build_stage4_megaprompt_user_prompt(
        question="Does activity affect sleep?",
        model_topology={},
        distribution_cards=[],
        loading_params=[],
        construct_scale_cards=[],
        prior_cards=[],
        ambiguous_indicators=[],
        distribution_choices={},
        initialization_policy=None,
        observation_intercept_policy=None,
        equilibrium_forcing=None,
        centerable_construct_names=(),
        baseline_factor_names=(),
        required_prior_names=(),
        optional_prior_names=(),
        authored_priors={},
        accepted_model_spec=None,
        model_spec_locked=False,
        latest_feedback="",
        resumed_from_checkpoint=False,
        include_accepted_state_sections=True,
        include_prior_source_guidance=True,
    )

    source_guidance = PRIOR_SOURCE_GUIDANCE.replace("{{", "{").replace("}}", "}")
    assert reducer_prompt.count(source_guidance) == 1
    assert megaprompt.count(source_guidance) == 1


def test_accepted_state_sections_are_shared_verbatim_in_both_user_prompts() -> None:
    accepted_model_spec = {
        "initialization_policy": "stationary",
        "observation_intercept_policy": "free",
        "equilibrium_forcing": False,
        "likelihoods": [
            {
                "indicator": "sleep_quality",
                "distribution": "ordered_logit",
                "link": "logit",
            }
        ],
        "parameters": [
            {
                "name": "beta_activity_sleep",
                "role": "effect",
                "constraint": "real",
                "description": "activity to sleep",
            }
        ],
    }
    authored_priors = {
        "beta_activity_sleep": {
            "distribution": "Normal",
            "params": {"mu": 0.0, "sigma": 0.5},
            "reasoning": "Weakly informative around zero.",
            "sources": [
                {
                    "title": "Sleep study",
                    "snippet": "Small positive same-day association.",
                    "url": "https://example.org/study",
                }
            ],
        }
    }
    accepted_sections = build_accepted_state_sections(
        accepted_model_spec=accepted_model_spec,
        authored_priors=authored_priors,
        centerable_construct_names=("sleep",),
        baseline_factor_names=("u_sleep",),
    )
    snapshot = Stage4ScopeSnapshot(
        block_id="prior:effect",
        block_kind="effect_prior",
        block_label="Effect priors",
        block_instructions="Propose priors only for the active block.",
        frontier_status="ACTIVE FRONTIER (machine-generated)",
        model_topology={},
        distribution_cards=[],
        loading_params=[],
        construct_scale_cards=[],
        prior_cards=[],
        coupled_prior_cards=[],
        accepted_model_spec=accepted_model_spec,
        accepted_authored_priors=authored_priors,
        centerable_construct_names=("sleep",),
        baseline_factor_names=("u_sleep",),
        submission_example="Use `submit_prior_block` with this argument object.",
        include_prior_source_guidance=False,
        latest_validation=default_stage4_validation_packet(),
    )
    reducer_prompt = build_stage4_user_prompt(
        question="Does activity affect sleep?",
        snapshot=snapshot,
    )
    megaprompt = build_stage4_megaprompt_user_prompt(
        question="Does activity affect sleep?",
        model_topology={},
        distribution_cards=[],
        loading_params=[],
        construct_scale_cards=[],
        prior_cards=[],
        ambiguous_indicators=[],
        distribution_choices={},
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        centerable_construct_names=("sleep",),
        baseline_factor_names=("u_sleep",),
        required_prior_names=("beta_activity_sleep",),
        optional_prior_names=(),
        authored_priors=authored_priors,
        accepted_model_spec=accepted_model_spec,
        model_spec_locked=True,
        latest_feedback="VALIDATION ERRORS:\n- Example",
        resumed_from_checkpoint=False,
        include_accepted_state_sections=True,
        include_prior_source_guidance=False,
    )

    for section in accepted_sections:
        assert reducer_prompt.count(section) == 1
        assert megaprompt.count(section) == 1


def test_megaprompt_omits_accepted_state_after_seed_turn() -> None:
    prompt = build_stage4_megaprompt_user_prompt(
        question="Does activity affect sleep?",
        model_topology={},
        distribution_cards=[],
        loading_params=[],
        construct_scale_cards=[],
        prior_cards=[],
        ambiguous_indicators=[],
        distribution_choices={},
        initialization_policy="stationary",
        observation_intercept_policy="free",
        equilibrium_forcing=False,
        centerable_construct_names=("sleep",),
        baseline_factor_names=("u_sleep",),
        required_prior_names=("beta_activity_sleep",),
        optional_prior_names=(),
        authored_priors={
            "beta_activity_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "reasoning": "Weakly informative around zero.",
                "sources": [],
            }
        },
        accepted_model_spec={
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
            "equilibrium_forcing": False,
            "likelihoods": [],
            "parameters": [],
        },
        model_spec_locked=True,
        latest_feedback="VALIDATION ERRORS:\n- Example",
        resumed_from_checkpoint=False,
        include_accepted_state_sections=False,
        include_prior_source_guidance=False,
    )

    assert "## Accepted Locked Model Spec" not in prompt
    assert "## Accepted Authored Priors" not in prompt


