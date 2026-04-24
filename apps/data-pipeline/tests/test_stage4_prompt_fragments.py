"""Regression tests for shared Stage 4 prompt fragments."""

from __future__ import annotations

from causal_ssm_agent.flows.stages.stage4.agentic.prompts.megaprompt import (
    build_stage4_megaprompt_system_prompt,
    build_stage4_megaprompt_user_prompt,
)
from causal_ssm_agent.flows.stages.stage4.agentic.prompts.model_proposal import (
    build_stage4_system_prompt,
    build_stage4_user_prompt,
)
from causal_ssm_agent.flows.stages.stage4.agentic.prompts.shared_fragments import (
    CONTINUOUS_TIME_DYNAMICS_SECTION,
    INITIAL_STATE_SCALE_DISCIPLINE_SECTION,
    LAGGED_EFFECT_INTERVAL_GUIDANCE_SECTION,
    LINK_FUNCTION_RULES_SECTION,
    OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION,
    PRIOR_DISTRIBUTION_TYPES_SECTION,
    PRIOR_SOURCE_GUIDANCE,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
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
        model_spec_locked=False,
        latest_feedback="",
        include_prior_source_guidance=True,
    )

    source_guidance = PRIOR_SOURCE_GUIDANCE.replace("{{", "{").replace("}}", "}")
    assert reducer_prompt.count(source_guidance) == 1
    assert megaprompt.count(source_guidance) == 1


def test_moved_verbatim_fragments_preserve_existing_text() -> None:
    assert INITIAL_STATE_SCALE_DISCIPLINE_SECTION == (
        "## Initial-State Scale Discipline\n\n"
        "- `t0_mean_*` and `t0_sd_*` live on the latent state scale.\n"
        "- Do not set `t0_mean_*` to the raw reference-indicator mean or "
        "`log(mean(indicator))` just because the indicator uses an identity or log link.\n"
        "- Default to weakly informative latent-scale priors such as `Normal(0, 1)` "
        "and `HalfNormal(1)` unless the construct is explicitly identified on an "
        "observed scale."
    )
    assert PRIOR_SOURCE_GUIDANCE == """If you include non-empty `sources`, each entry must be an object with this shape:
```json
{{
  "title": "Source title",
  "snippet": "Relevant excerpt supporting the prior",
  "url": "https://example.org/paper",
  "effect_size": "β=0.21",
  "study_interval_days": 7.0
}}
```

Only `title` and `snippet` are required. Do not use raw strings or ad hoc keys such as `citation`, `finding`, `study_type`, or `notes`. If you are unsure, use `"sources": []`. `study_interval_days` belongs inside each source entry; `reference_interval_days` belongs on the prior itself."""
