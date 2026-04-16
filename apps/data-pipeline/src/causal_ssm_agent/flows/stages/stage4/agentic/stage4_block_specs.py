"""Single-source metadata for Stage 4 block kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from causal_ssm_agent.flows.stages.stage4.tool_registry import (
    allowed_stage4_tool_names,
)

Stage4BlockKind = Literal[
    "model_configuration",
    "indicator_decision",
    "measurement_prior",
    "observation_prior",
    "dynamics_prior",
    "effect_prior",
    "correlation_prior",
    "global_review",
    "global_prior_review",
]
Stage4BlockPhase = Literal[
    "model_decisions",
    "global_review",
    "prior_blocks",
    "global_prior_review",
]
Stage4SubmissionPayloadKind = Literal[
    "model_configuration_choice",
    "indicator_choice",
    "global_review_decision",
    "prior_bundle",
]
Stage4AcceptedTransitionKind = Literal[
    "model_configuration",
    "indicator_choice",
    "review_approval",
    "prior_bundle",
]


@dataclass(frozen=True)
class Stage4PromptScopePolicy:
    """Single source of truth for prompt behavior of a scope kind."""

    system_task: str
    user_task: str
    visible_sections: tuple[str, ...]
    guidance_section_keys: tuple[str, ...] = ()
    parameter_guidance_prefixes: tuple[str, ...] = ()
    allowed_tool_names: tuple[str, ...] = ("submit_prior_block",)


@dataclass(frozen=True)
class Stage4BlockKindSpec:
    """Static metadata for one authored Stage 4 block kind."""

    kind: Stage4BlockKind
    phase: Stage4BlockPhase
    prompt_policy: Stage4PromptScopePolicy
    submission_payload_kind: Stage4SubmissionPayloadKind
    accepted_transition_kind: Stage4AcceptedTransitionKind
    include_prior_source_guidance: bool = False

    @property
    def is_prior_bundle(self) -> bool:
        """Whether this block accepts prior bundles."""
        return self.submission_payload_kind == "prior_bundle"


_COMMON_PRIOR_VISIBLE_SECTIONS = ("construct_scale_cards", "prior_cards")
_COMMON_PRIOR_GUIDANCE_SECTION_KEYS = (
    "prior_distribution_types",
    "parameter_guidance",
)


def _make_block_kind_spec(
    kind: Stage4BlockKind,
    *,
    phase: Stage4BlockPhase,
    submission_payload_kind: Stage4SubmissionPayloadKind,
    accepted_transition_kind: Stage4AcceptedTransitionKind,
    system_task: str,
    user_task: str,
    visible_sections: tuple[str, ...],
    guidance_section_keys: tuple[str, ...] = (),
    parameter_guidance_prefixes: tuple[str, ...] = (),
    allowed_tool_names: tuple[str, ...] = ("submit_prior_block",),
    include_prior_source_guidance: bool = False,
) -> Stage4BlockKindSpec:
    """Build static metadata for one Stage 4 block kind."""
    return Stage4BlockKindSpec(
        kind=kind,
        phase=phase,
        submission_payload_kind=submission_payload_kind,
        accepted_transition_kind=accepted_transition_kind,
        include_prior_source_guidance=include_prior_source_guidance,
        prompt_policy=Stage4PromptScopePolicy(
            system_task=system_task,
            user_task=user_task,
            visible_sections=visible_sections,
            guidance_section_keys=guidance_section_keys,
            parameter_guidance_prefixes=parameter_guidance_prefixes,
            allowed_tool_names=allowed_tool_names,
        ),
    )


def _make_prior_block_kind_spec(
    kind: Stage4BlockKind,
    *,
    system_task: str,
    user_task: str,
    parameter_guidance_prefixes: tuple[str, ...],
    extra_guidance_section_keys: tuple[str, ...] = (),
    visible_sections: tuple[str, ...] = _COMMON_PRIOR_VISIBLE_SECTIONS,
    allowed_tool_names: tuple[str, ...] | None = None,
    phase: Stage4BlockPhase = "prior_blocks",
) -> Stage4BlockKindSpec:
    """Build a prior-authoring Stage 4 block kind with shared defaults."""
    resolved_allowed_tool_names = (
        allowed_stage4_tool_names(kind) if allowed_tool_names is None else allowed_tool_names
    )
    return _make_block_kind_spec(
        kind,
        phase=phase,
        submission_payload_kind="prior_bundle",
        accepted_transition_kind="prior_bundle",
        system_task=system_task,
        user_task=user_task,
        visible_sections=visible_sections,
        guidance_section_keys=(
            *_COMMON_PRIOR_GUIDANCE_SECTION_KEYS,
            *extra_guidance_section_keys,
        ),
        parameter_guidance_prefixes=parameter_guidance_prefixes,
        allowed_tool_names=resolved_allowed_tool_names,
        include_prior_source_guidance=True,
    )


_STAGE4_BLOCK_KIND_SPECS: dict[Stage4BlockKind, Stage4BlockKindSpec] = {
    "model_configuration": _make_block_kind_spec(
        "model_configuration",
        phase="model_decisions",
        submission_payload_kind="model_configuration_choice",
        accepted_transition_kind="model_configuration",
        system_task=(
            "Choose the global initialization policy and whether equilibrium forcing is enabled. "
            "These are model-level semantics, not prior choices."
        ),
        user_task=(
            "Choose the global initialization policy and whether equilibrium forcing is enabled. "
            "Do not submit indicator likelihood decisions or priors in this block."
        ),
        visible_sections=("construct_scale_cards",),
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
        ),
        allowed_tool_names=allowed_stage4_tool_names("model_configuration"),
    ),
    "indicator_decision": _make_block_kind_spec(
        "indicator_decision",
        phase="model_decisions",
        submission_payload_kind="indicator_choice",
        accepted_transition_kind="indicator_choice",
        system_task=(
            "Choose exactly one observation distribution/link pair for the active "
            "indicator. Do not propose loading metadata or priors."
        ),
        user_task=(
            "Choose exactly one distribution/link pair for the active indicator. "
            "Do not send loading metadata or priors in this block."
        ),
        visible_sections=("distribution_cards", "construct_scale_cards"),
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
        ),
        allowed_tool_names=allowed_stage4_tool_names("indicator_decision"),
    ),
    "measurement_prior": _make_prior_block_kind_spec(
        "measurement_prior",
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        extra_guidance_section_keys=("measurement_prior_guidance",),
        parameter_guidance_prefixes=("lambda", "obs_sd"),
    ),
    "observation_prior": _make_prior_block_kind_spec(
        "observation_prior",
        system_task=(
            "Propose full prior specifications only for the active observation-family "
            "hyperparameters. These priors control tails, dispersion, concentration, or "
            "threshold geometry for the locked likelihood family already chosen upstream."
        ),
        user_task=(
            "Propose priors only for this active observation-family block. Do not change the "
            "locked likelihood family or submit priors for other blocks."
        ),
        extra_guidance_section_keys=("measurement_prior_guidance",),
        parameter_guidance_prefixes=("obs_", "manifest_mean"),
        visible_sections=("distribution_cards", "construct_scale_cards", "prior_cards"),
    ),
    "dynamics_prior": _make_prior_block_kind_spec(
        "dynamics_prior",
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "These dynamics priors set the continuous-time damping budget that downstream effect "
            "priors must fit inside, so avoid near-unit-root or overly diffuse persistence unless "
            "strong evidence requires it."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. Choose dynamics "
            "priors that leave clear damping headroom for later incoming effects, and do not send "
            "model decisions or priors for other blocks."
        ),
        extra_guidance_section_keys=(
            "continuous_time_dynamics",
            "latent_initial_state_guidance",
            "dynamics_budget_discipline",
        ),
        parameter_guidance_prefixes=("rho", "sigma", "cint", "t0_mean", "t0_sd"),
    ),
    "effect_prior": _make_prior_block_kind_spec(
        "effect_prior",
        system_task=(
            "Author priors only for the active target construct's incoming lagged-effect row. "
            "Use the row-level stability budget reported in the user message as advisory "
            "headroom guidance, not as a mechanical acceptance rule. In dense feedback-coupled "
            "rows, default to tightly zero-centered priors with modest uncertainty unless strong "
            "longitudinal evidence supports larger effects."
        ),
        user_task=(
            "This block owns one target construct's full incoming lagged-effect row. Use the "
            "stability budget in Frontier Status as advisory headroom guidance for this row: "
            "prefer tightly zero-centered, small-scale priors for SCC-internal or "
            "`Feedback Loop = yes` edges, leave slack for uncertainty, and only grow effects "
            "when strong longitudinal evidence justifies it. Submit priors only for this block's "
            "parameters."
        ),
        extra_guidance_section_keys=(
            "continuous_time_dynamics",
            "effect_row_budget_discipline",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=("beta",),
    ),
    "correlation_prior": _make_prior_block_kind_spec(
        "correlation_prior",
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        parameter_guidance_prefixes=("cor", "tau"),
    ),
    "global_review": _make_block_kind_spec(
        "global_review",
        phase="global_review",
        submission_payload_kind="global_review_decision",
        accepted_transition_kind="review_approval",
        system_task=(
            "Review the fully locked Stage 4 model form before prior elicitation. "
            "Do not propose priors. Either approve the locked model or reopen the relevant "
            "model-decision blocks if something materially needs revision, including "
            "`model:configuration` and any affected `indicator:*` blocks."
        ),
        user_task=(
            "Review the locked model form shown below. If it is coherent, approve it. If not, "
            "reopen the relevant model-decision blocks and explain why, including `model:configuration` and any affected `indicator:*` blocks. "
            "Do not submit priors in this block."
        ),
        visible_sections=("distribution_cards", "loading_params", "construct_scale_cards"),
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
        ),
        allowed_tool_names=allowed_stage4_tool_names("global_review"),
    ),
    "global_prior_review": _make_prior_block_kind_spec(
        "global_prior_review",
        phase="global_prior_review",
        system_task=(
            "Repair the full Stage 4 prior system after a global validation failure. "
            "You may revise any prior parameters needed to resolve the failure, but do not "
            "change locked likelihood choices or loading orientations."
        ),
        user_task=(
            "Review the full accepted prior system shown below and revise any priors needed "
            "to resolve the latest global validation failure. You may submit priors for any "
            "Stage 4 parameter, but do not change model-form decisions."
        ),
        extra_guidance_section_keys=(
            "continuous_time_dynamics",
            "latent_initial_state_guidance",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=(
            "lambda",
            "obs_",
            "manifest_mean",
            "rho",
            "sigma",
            "cint",
            "beta",
            "cor",
            "tau",
            "t0_mean",
            "t0_sd",
        ),
    ),
}


def get_stage4_block_kind_spec(kind: str) -> Stage4BlockKindSpec:
    """Return the metadata for one Stage 4 block kind."""
    spec = _STAGE4_BLOCK_KIND_SPECS.get(kind)
    if spec is None:
        raise ValueError(f"Unsupported Stage 4 block kind {kind!r}")
    return spec


def get_stage4_prompt_scope_policy(kind: str) -> Stage4PromptScopePolicy:
    """Return the prompt policy for a Stage 4 block kind."""
    return get_stage4_block_kind_spec(kind).prompt_policy


def get_stage4_block_phase(kind: str) -> Stage4BlockPhase:
    """Project one authored block kind onto the public Stage 4 phase labels."""
    return get_stage4_block_kind_spec(kind).phase


def get_stage4_block_kinds() -> tuple[Stage4BlockKind, ...]:
    """Return all registered Stage 4 block kinds in deterministic order."""
    return tuple(_STAGE4_BLOCK_KIND_SPECS)
