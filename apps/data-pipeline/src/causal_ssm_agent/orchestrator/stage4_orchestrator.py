"""Stage 4 deterministic model skeleton and prompt context helpers.

Pre-computes everything that follows directly from the causal spec without
LLM judgment: parameter enumeration, deterministic likelihood choices,
and compact prompt-local context cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causal_ssm_agent.distributions import VALID_LIKELIHOODS_FOR_DTYPE
from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.orchestrator.schemas_model import (
    VALID_LINKS_FOR_DISTRIBUTION,
)
from causal_ssm_agent.utils.causal_spec import (
    get_estimation_edges,
    get_estimation_state_order,
    get_indicators,
    get_induced_dependencies,
    get_latent_constructs,
    get_outcome_name,
)


@dataclass(frozen=True)
class Stage4Skeleton:
    """Deterministic Stage 4 decision surface derived from the causal spec."""

    resolved_likelihoods: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_indicators: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    loading_params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_params(self) -> list[dict[str, Any]]:
        """Return the full final parameter inventory, including loadings."""
        return [*self.parameters, *self.loading_params]

    @property
    def final_parameter_names(self) -> list[str]:
        """Return the final parameter names in deterministic order."""
        return [param["name"] for param in self.all_params]


@dataclass(frozen=True)
class Stage4FrontierBlock:
    """A single reducer-owned Stage 4 decision block."""

    id: str
    kind: str
    label: str
    construct_names: tuple[str, ...] = ()
    variable_names: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage4Plan:
    """Immutable Stage 4 execution plan derived from the deterministic skeleton."""

    model_blocks: tuple[Stage4FrontierBlock, ...] = ()
    review_block: Stage4FrontierBlock | None = None
    prior_blocks: tuple[Stage4FrontierBlock, ...] = ()
    prior_review_block: Stage4FrontierBlock | None = None
    blocks_by_id: dict[str, Stage4FrontierBlock] = field(default_factory=dict)
    parameter_to_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_decision_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_measurement_block_id: dict[str, str] = field(default_factory=dict)
    dynamics_block_id_by_construct: dict[str, str] = field(default_factory=dict)

    @property
    def all_blocks(self) -> tuple[Stage4FrontierBlock, ...]:
        """Return all blocks in deterministic execution order."""
        review = (self.review_block,) if self.review_block is not None else ()
        prior_review = (self.prior_review_block,) if self.prior_review_block is not None else ()
        return (*self.model_blocks, *review, *self.prior_blocks, *prior_review)

    def get_block(self, block_id: str) -> Stage4FrontierBlock | None:
        """Return a block by id, if present."""
        return self.blocks_by_id.get(block_id)


@dataclass(frozen=True)
class Stage4PromptScopePolicy:
    """Single source of truth for prompt behavior of a scope kind."""

    system_task: str
    user_task: str
    visible_sections: tuple[str, ...]
    guidance_section_keys: tuple[str, ...] = ()
    parameter_guidance_prefixes: tuple[str, ...] = ()
    allowed_tool_names: tuple[str, ...] = ("validate_model",)


_PROMPT_SCOPE_CONFIG: dict[str, Stage4PromptScopePolicy] = {
    "indicator_decision": Stage4PromptScopePolicy(
        system_task=(
            "Choose exactly one observation distribution/link pair for the active "
            "indicator. Do not propose loading constraints or priors."
        ),
        user_task=(
            "Choose exactly one distribution/link pair for the active indicator. "
            "Do not send loading constraints or priors in this block."
        ),
        visible_sections=("distribution_cards", "construct_scale_cards"),
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
        ),
        allowed_tool_names=("validate_model",),
    ),
    "loading_decision": Stage4PromptScopePolicy(
        system_task=(
            "Choose loading/sign-identification constraints only for the active loading block. "
            "Use `positive` for sign identification unless negative loadings are substantively "
            "plausible. Do not propose likelihood changes or priors."
        ),
        user_task=(
            "Choose loading constraints only for the loading parameters shown below. "
            "You may submit one or more constraints from this block, but do not send priors."
        ),
        visible_sections=("loading_params", "construct_scale_cards"),
        allowed_tool_names=("validate_model",),
    ),
    "measurement_prior": Stage4PromptScopePolicy(
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "measurement_prior_guidance",
        ),
        parameter_guidance_prefixes=("lambda",),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "dynamics_prior": Stage4PromptScopePolicy(
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
        ),
        parameter_guidance_prefixes=("rho", "sigma"),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "effect_prior": Stage4PromptScopePolicy(
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=("beta",),
        allowed_tool_names=("validate_model", "search_literature", "elicit_prior_gmm"),
    ),
    "correlation_prior": Stage4PromptScopePolicy(
        system_task=(
            "Propose full prior specifications only for the active block's parameters. "
            "Choose the prior family, hyperparameters, and reasoning for the active scope only."
        ),
        user_task=(
            "Propose full prior specifications only for this block's parameters. "
            "Do not send model decisions or priors for other blocks."
        ),
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
        ),
        parameter_guidance_prefixes=("cor",),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "global_review": Stage4PromptScopePolicy(
        system_task=(
            "Review the fully locked Stage 4 model form before prior elicitation. "
            "Do not propose priors. Either approve the locked model or reopen the relevant "
            "model-decision blocks if something materially needs revision."
        ),
        user_task=(
            "Review the locked model form shown below. If it is coherent, approve it. If not, "
            "reopen the relevant model-decision blocks and explain why. "
            "Do not submit priors in this block."
        ),
        visible_sections=("distribution_cards", "loading_params", "construct_scale_cards"),
        guidance_section_keys=(
            "observation_distribution_guidance",
            "link_function_rules",
        ),
        allowed_tool_names=("validate_model",),
    ),
    "global_prior_review": Stage4PromptScopePolicy(
        system_task=(
            "Repair the full Stage 4 prior system after a global validation failure. "
            "You may revise any prior parameters needed to resolve the failure, but do not "
            "change likelihood or loading decisions."
        ),
        user_task=(
            "Review the full accepted prior system shown below and revise any priors needed "
            "to resolve the latest global validation failure. You may submit priors for any "
            "Stage 4 parameter, but do not change model-form decisions."
        ),
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=("lambda", "rho", "sigma", "beta", "cor"),
        allowed_tool_names=("validate_model",),
    ),
}


def get_stage4_prompt_scope_policy(kind: str) -> Stage4PromptScopePolicy:
    """Return the prompt policy for a Stage 4 block kind."""
    config = _PROMPT_SCOPE_CONFIG.get(kind)
    if config is None:
        raise ValueError(f"Unsupported Stage 4 prompt scope for block kind {kind!r}")
    return config


def derive_deterministic_spec(causal_spec: dict) -> Stage4Skeleton:
    """Pre-compute all deterministic parts of the stage-4 model skeleton."""
    retained_state_order = get_estimation_state_order(causal_spec)
    retained_edges = get_estimation_edges(causal_spec)
    indicators = get_indicators(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec)
    }
    retained_constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]

    indicators_per_construct = _indicators_per_construct(indicators)
    reference_indicator_lookup = {
        construct: indicator_names[0]
        for construct, indicator_names in indicators_per_construct.items()
        if indicator_names
    }

    resolved_likelihoods: list[dict[str, Any]] = []
    ambiguous_indicators: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    loading_params: list[dict[str, Any]] = []

    # --- Likelihoods ---
    for indicator in indicators:
        name = indicator["name"]
        dtype = indicator.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, ())

        if len(valid_dists) == 1:
            dist = next(iter(valid_dists))
            valid_links = VALID_LINKS_FOR_DISTRIBUTION[dist]
            if len(valid_links) == 1:
                link = next(iter(valid_links))
                resolved_likelihoods.append(
                    {
                        "variable": name,
                        "distribution": dist.value,
                        "link": link.value,
                        "reasoning": f"{dtype} dtype -> {dist.value} / {link.value}",
                    }
                )
            else:
                ambiguous_indicators.append(
                    {
                        "variable": name,
                        "dtype": dtype,
                        "fixed_distribution": dist.value,
                        "valid_links": sorted(link_fn.value for link_fn in valid_links),
                    }
                )
        else:
            link_options: dict[str, list[str]] = {}
            for distribution in sorted(valid_dists, key=lambda item: item.value):
                links = VALID_LINKS_FOR_DISTRIBUTION[distribution]
                link_options[distribution.value] = sorted(link_fn.value for link_fn in links)
            ambiguous_indicators.append(
                {
                    "variable": name,
                    "dtype": dtype,
                    "valid_distributions": sorted(dist.value for dist in valid_dists),
                    "link_options": link_options,
                }
            )

    # --- Autoregressive parameters ---
    for construct in retained_constructs:
        if (
            construct.get("temporal_status") == "time_varying"
            and construct.get("role") == "endogenous"
        ):
            construct_name = construct["name"]
            parameters.append(
                {
                    "name": f"rho_{construct_name}",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": f"AR(1) discrete-time persistence for {construct_name}",
                    "construct": construct_name,
                }
            )

    # --- Fixed effects ---
    for edge in retained_edges:
        cause = edge["cause"]
        effect = edge["effect"]
        parameters.append(
            {
                "name": f"beta_{cause}_{effect}",
                "role": "fixed_effect",
                "constraint": "none",
                "description": f"Effect of {cause} on {effect}",
                "cause": cause,
                "effect": effect,
                "lagged": edge.get("lagged", True),
            }
        )

    # --- Residual SDs ---
    for construct in retained_constructs:
        construct_name = construct["name"]
        parameters.append(
            {
                "name": f"sigma_{construct_name}",
                "role": "residual_sd",
                "constraint": "positive",
                "description": f"Residual/innovation SD for {construct_name}",
                "construct": construct_name,
            }
        )

    # --- Loadings ---
    reference_set: set[str] = set()
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if (
            not construct_name
            or construct_name not in indicators_per_construct
            or len(indicators_per_construct[construct_name]) <= 1
        ):
            continue

        if construct_name not in reference_set:
            reference_set.add(construct_name)
            continue

        reference_indicator = reference_indicator_lookup.get(construct_name)
        loading_params.append(
            {
                "name": f"lambda_{indicator['name']}_{construct_name}",
                "role": "loading",
                "constraint": "positive",
                "description": f"Factor loading for {indicator['name']} on {construct_name}",
                "indicator": indicator["name"],
                "construct": construct_name,
                "reference_indicator": reference_indicator,
            }
        )

    # --- Correlations from marginalized confounders ---
    for dependency in get_induced_dependencies(causal_spec):
        construct_1, construct_2 = dependency["between"]
        dependency_kind = dependency["kind"]
        parameter_name = (
            f"cor_{construct_1}_{construct_2}"
            if dependency_kind == "innovation_correlation"
            else f"cor0_{construct_1}_{construct_2}"
        )
        role = (
            "correlation"
            if dependency_kind == "innovation_correlation"
            else "initial_state_correlation"
        )
        parameters.append(
            {
                "name": parameter_name,
                "role": role,
                "constraint": "correlation",
                "description": (
                    f"{dependency_kind.replace('_', ' ')} between {construct_1} and {construct_2} "
                    f"(source confounders: {', '.join(dependency['source_confounders'])})"
                ),
                "construct_1": construct_1,
                "construct_2": construct_2,
                "dependency_kind": dependency_kind,
                "source_confounders": dependency["source_confounders"],
            }
        )

    return Stage4Skeleton(
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        parameters=parameters,
        loading_params=loading_params,
    )


def build_stage4_plan(causal_spec: dict, skeleton: Stage4Skeleton) -> Stage4Plan:
    """Build the immutable Stage 4 block plan from the deterministic skeleton."""
    construct_order = get_estimation_state_order(causal_spec)
    indicators = get_indicators(causal_spec)
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    indicators_per_construct = _indicators_per_construct(indicators)
    param_order = {parameter["name"]: idx for idx, parameter in enumerate(skeleton.all_params)}

    model_blocks: list[Stage4FrontierBlock] = []
    for item in skeleton.ambiguous_indicators:
        variable = item["variable"]
        construct_name = (indicator_lookup.get(variable) or {}).get("construct_name")
        model_blocks.append(
            Stage4FrontierBlock(
                id=f"indicator:{variable}",
                kind="indicator_decision",
                label=f"Choose likelihood for {variable}",
                construct_names=(construct_name,) if isinstance(construct_name, str) else (),
                variable_names=(variable,),
                payload=dict(item),
            )
        )

    loading_names_by_construct: dict[str, list[str]] = {}
    for parameter in skeleton.loading_params:
        construct_name = parameter.get("construct")
        if isinstance(construct_name, str):
            loading_names_by_construct.setdefault(construct_name, []).append(parameter["name"])
    for construct_name in construct_order:
        names = loading_names_by_construct.get(construct_name) or []
        if not names:
            continue
        model_blocks.append(
            Stage4FrontierBlock(
                id=f"loading:{construct_name}",
                kind="loading_decision",
                label=f"Choose loading constraints for {construct_name}",
                construct_names=(construct_name,),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    review_block = Stage4FrontierBlock(
        id="review:model_spec",
        kind="global_review",
        label="Review locked model specification before prior elicitation",
        construct_names=tuple(construct_order),
        variable_names=tuple(item["variable"] for item in skeleton.ambiguous_indicators),
        parameter_names=tuple(
            sorted(
                (parameter["name"] for parameter in skeleton.loading_params),
                key=param_order.__getitem__,
            )
        ),
        payload={"reopenable_block_ids": tuple(block.id for block in model_blocks)},
    )

    prior_blocks: list[Stage4FrontierBlock] = []
    for construct_name in construct_order:
        names = loading_names_by_construct.get(construct_name) or []
        if not names:
            continue
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"measurement:{construct_name}",
                kind="measurement_prior",
                label=f"Elicit measurement priors for {construct_name}",
                construct_names=(construct_name,),
                variable_names=tuple(indicators_per_construct.get(construct_name) or ()),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(construct_order)
    for edge in get_estimation_edges(causal_spec):
        graph.add_edge(edge["cause"], edge["effect"])
    order_lookup = {name: idx for idx, name in enumerate(construct_order)}
    dynamics_roles = {"ar_coefficient", "residual_sd"}
    for component in sorted(
        nx.strongly_connected_components(graph),
        key=lambda members: min(order_lookup[name] for name in members),
    ):
        ordered_members = tuple(name for name in construct_order if name in component)
        names = [
            parameter["name"]
            for parameter in skeleton.parameters
            if parameter["role"] in dynamics_roles and parameter.get("construct") in component
        ]
        if not names:
            continue
        label_suffix = ", ".join(ordered_members)
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"dynamics:{'+'.join(ordered_members)}",
                kind="dynamics_prior",
                label=f"Elicit dynamics priors for {label_suffix}",
                construct_names=ordered_members,
                variable_names=tuple(
                    indicator_name
                    for construct_name in ordered_members
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    effect_parameters_by_target: dict[str, list[dict[str, Any]]] = {}
    for parameter in skeleton.parameters:
        if parameter["role"] != "fixed_effect":
            continue
        effect_parameters_by_target.setdefault(parameter["effect"], []).append(parameter)
    for effect_name in construct_order:
        parameters = effect_parameters_by_target.get(effect_name) or []
        if not parameters:
            continue
        ordered_parameters = sorted(parameters, key=lambda parameter: param_order[parameter["name"]])
        cause_names = tuple(dict.fromkeys(parameter["cause"] for parameter in ordered_parameters))
        construct_names = (*cause_names, effect_name)
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"effects:{effect_name}",
                kind="effect_prior",
                label=f"Elicit effect priors for incoming effects on {effect_name}",
                construct_names=construct_names,
                variable_names=tuple(
                    indicator_name
                    for construct_name in construct_names
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ),
                parameter_names=tuple(parameter["name"] for parameter in ordered_parameters),
                payload={
                    "target_construct": effect_name,
                    "cause_names": cause_names,
                },
            )
        )

    correlation_roles = {"correlation", "initial_state_correlation"}
    for parameter in skeleton.parameters:
        if parameter["role"] not in correlation_roles:
            continue
        construct_names = tuple(
            name
            for name in (parameter.get("construct_1"), parameter.get("construct_2"))
            if isinstance(name, str)
        )
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"correlation:{parameter['name']}",
                kind="correlation_prior",
                label=f"Elicit correlation prior for {parameter['name']}",
                construct_names=construct_names,
                variable_names=tuple(
                    indicator_name
                    for construct_name in construct_names
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ),
                parameter_names=(parameter["name"],),
            )
        )

    prior_review_block = Stage4FrontierBlock(
        id="review:prior_system",
        kind="global_prior_review",
        label="Repair the full prior system after global validation failures",
        construct_names=tuple(construct_order),
        variable_names=tuple(indicator["name"] for indicator in indicators),
        parameter_names=tuple(
            sorted(
                (parameter["name"] for parameter in skeleton.all_params),
                key=param_order.__getitem__,
            )
        ),
    )

    blocks_by_id = {
        block.id: block
        for block in [*model_blocks, review_block, *prior_blocks, prior_review_block]
    }
    parameter_to_block_id: dict[str, str] = {}
    indicator_to_decision_block_id: dict[str, str] = {}
    indicator_to_measurement_block_id: dict[str, str] = {}
    dynamics_block_id_by_construct: dict[str, str] = {}

    for block in prior_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "measurement_prior":
            for indicator_name in block.variable_names:
                indicator_to_measurement_block_id[indicator_name] = block.id
        if block.kind == "dynamics_prior":
            for construct_name in block.construct_names:
                dynamics_block_id_by_construct[construct_name] = block.id

    for block in model_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "indicator_decision":
            for indicator_name in block.variable_names:
                indicator_to_decision_block_id[indicator_name] = block.id

    return Stage4Plan(
        model_blocks=tuple(model_blocks),
        review_block=review_block,
        prior_blocks=tuple(prior_blocks),
        prior_review_block=prior_review_block,
        blocks_by_id=blocks_by_id,
        parameter_to_block_id=parameter_to_block_id,
        indicator_to_decision_block_id=indicator_to_decision_block_id,
        indicator_to_measurement_block_id=indicator_to_measurement_block_id,
        dynamics_block_id_by_construct=dynamics_block_id_by_construct,
    )


def build_model_topology(causal_spec: dict) -> dict[str, Any]:
    """Build compact fixed model context for the Stage 4 prompt."""
    model_dt_days = get_construct_dt_days(causal_spec)
    return {
        "model_clock": causal_spec.get("measurement", {}).get("model_clock"),
        "model_interval_days": model_dt_days,
        "outcome": get_outcome_name(causal_spec),
        "latent_edges": [
            {
                "cause": edge["cause"],
                "effect": edge["effect"],
                "lagged": bool(edge.get("lagged", True)),
                "description": edge.get("description"),
            }
            for edge in get_estimation_edges(causal_spec)
        ],
    }


def build_distribution_cards(
    causal_spec: dict,
    indicator_audits: dict[str, dict[str, Any]] | None,
    skeleton: Stage4Skeleton,
) -> list[dict[str, Any]]:
    """Build compact cards for indicators whose likelihoods need judgment."""
    indicator_lookup = {indicator["name"]: indicator for indicator in get_indicators(causal_spec)}

    cards: list[dict[str, Any]] = []
    for item in skeleton.ambiguous_indicators:
        variable = item["variable"]
        indicator = indicator_lookup.get(variable, {})
        audit = (indicator_audits or {}).get(variable) or {}
        validation = audit.get("validation") or {}

        option_rows: list[dict[str, Any]] = []
        if "fixed_distribution" in item:
            option_rows.append(
                {
                    "distribution": item["fixed_distribution"],
                    "links": item.get("valid_links", []),
                    "distribution_fixed": True,
                }
            )
        else:
            for distribution in item.get("valid_distributions", []):
                option_rows.append(
                    {
                        "distribution": distribution,
                        "links": item.get("link_options", {}).get(distribution, []),
                        "distribution_fixed": False,
                    }
                )

        cards.append(
            {
                "variable": variable,
                "construct": indicator.get("construct_name"),
                "measurement_dtype": indicator.get("measurement_dtype"),
                "aggregation": indicator.get("aggregation"),
                "observation_window": indicator.get("observation_window"),
                "effective_window": indicator.get("observation_window")
                or causal_spec.get("measurement", {}).get("model_clock"),
                "how_to_measure": indicator.get("how_to_measure"),
                "options": option_rows,
                "profile": _compact_profile(audit.get("profile") or {}),
                "validation_issues": [
                    f"{issue['severity']} {issue['issue_type']}"
                    for issue in validation.get("issues") or []
                ],
            }
        )
    return cards


def build_construct_scale_cards(
    causal_spec: dict,
    indicator_audits: dict[str, dict[str, Any]] | None,
    skeleton: Stage4Skeleton | None = None,
) -> list[dict[str, Any]]:
    """Build one construct-local scale card per construct."""
    model_clock = causal_spec.get("measurement", {}).get("model_clock")
    retained_state_order = get_estimation_state_order(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec)
    }
    constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]
    indicators = get_indicators(causal_spec)
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    indicators_per_construct = _indicators_per_construct(indicators)
    reference_indicator_lookup = {
        construct: indicator_names[0]
        for construct, indicator_names in indicators_per_construct.items()
        if indicator_names
    }
    ambiguous_indicator_names = {
        item["variable"] for item in (skeleton.ambiguous_indicators if skeleton else [])
    }

    cards: list[dict[str, Any]] = []
    for construct in constructs:
        construct_name = construct["name"]
        cards.append(
            {
                "construct": construct_name,
                "description": construct.get("description"),
                "role": construct.get("role"),
                "temporal_status": construct.get("temporal_status"),
                "is_outcome": bool(construct.get("is_outcome", False)),
                "reference_indicator": reference_indicator_lookup.get(construct_name),
                "indicators": [
                    _build_indicator_anchor(
                        indicator_name,
                        indicator_lookup,
                        indicator_audits,
                        model_clock=model_clock,
                        is_reference=indicator_name
                        == reference_indicator_lookup.get(construct_name),
                        has_distribution_decision_card=indicator_name in ambiguous_indicator_names,
                    )
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ],
            }
        )
    return cards


def build_prior_cards(causal_spec: dict, skeleton: Stage4Skeleton) -> list[dict[str, Any]]:
    """Build compact prompt-local prior cards for every deterministic parameter."""
    model_interval_days = get_construct_dt_days(causal_spec)
    lagged_edges = {
        (edge["cause"], edge["effect"])
        for edge in get_estimation_edges(causal_spec)
        if edge.get("lagged", True)
    }
    cards: list[dict[str, Any]] = []
    for parameter in skeleton.all_params:
        role = parameter["role"]
        card: dict[str, Any] = {
            "parameter": parameter["name"],
            "role": role,
            "constraint": parameter["constraint"],
        }
        if role in {"ar_coefficient", "residual_sd"}:
            construct_name = parameter["construct"]
            card["structural_context"] = {"construct": construct_name}
        elif role == "fixed_effect":
            cause = parameter["cause"]
            effect = parameter["effect"]
            lagged = parameter.get("lagged", True)
            card["structural_context"] = {
                "cause": cause,
                "effect": effect,
                "lagged": lagged,
                "expected_lag_days": model_interval_days if lagged else 0.0,
                "feedback_loop": lagged and (effect, cause) in lagged_edges,
            }
        elif role == "loading":
            construct_name = parameter["construct"]
            indicator_name = parameter["indicator"]
            reference_indicator = parameter.get("reference_indicator")
            card["structural_context"] = {
                "construct": construct_name,
                "indicator": indicator_name,
                "reference_indicator": reference_indicator,
            }
        elif role in {"correlation", "initial_state_correlation"}:
            construct_1 = parameter["construct_1"]
            construct_2 = parameter["construct_2"]
            card["structural_context"] = {
                "construct_1": construct_1,
                "construct_2": construct_2,
                "dependency_kind": parameter["dependency_kind"],
                "source_confounders": parameter["source_confounders"],
            }
        else:
            card["structural_context"] = {}

        cards.append(card)

    return cards


def _indicators_per_construct(indicators: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if construct_name:
            grouped.setdefault(construct_name, []).append(indicator["name"])
    return grouped


def _build_indicator_anchor(
    indicator_name: str | None,
    indicator_lookup: dict[str, dict[str, Any]],
    indicator_audits: dict[str, dict[str, Any]] | None,
    *,
    model_clock: str | None,
    is_reference: bool,
    has_distribution_decision_card: bool,
) -> dict[str, Any] | None:
    if not indicator_name:
        return None

    indicator = indicator_lookup.get(indicator_name, {})
    profile = ((indicator_audits or {}).get(indicator_name) or {}).get("profile") or {}
    return {
        "indicator": indicator_name,
        "construct": indicator.get("construct_name"),
        "measurement_dtype": indicator.get("measurement_dtype"),
        "how_to_measure": indicator.get("how_to_measure"),
        "aggregation": indicator.get("aggregation"),
        "observation_window": indicator.get("observation_window"),
        "effective_window": indicator.get("observation_window") or model_clock,
        "is_reference": is_reference,
        "has_distribution_decision_card": has_distribution_decision_card,
        "profile": _compact_profile(profile),
    }


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    if not profile:
        return None

    compact: dict[str, Any] = {}
    for key in (
        "n_obs",
        "mean",
        "std",
        "q25",
        "q50",
        "q75",
        "min",
        "max",
        "time_coverage_ratio",
        "max_gap_ratio",
        "duplicate_pct",
        "n_unparseable_timestamps",
        "zero_fraction",
        "variance_to_mean_ratio",
    ):
        value = profile.get(key)
        if value is not None:
            compact[key] = value

    support_flags = [
        flag_name
        for flag_name in ("is_nonnegative", "is_unit_interval", "looks_integer_valued")
        if profile.get(flag_name) is not None
    ]
    for flag_name in support_flags:
        compact[flag_name] = profile[flag_name]
    return compact or None
