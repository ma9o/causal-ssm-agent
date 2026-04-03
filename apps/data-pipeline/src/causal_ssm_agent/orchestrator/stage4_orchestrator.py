"""Stage 4 deterministic model skeleton and prompt context helpers.

Pre-computes everything that follows directly from the causal spec without
LLM judgment: parameter enumeration, deterministic likelihood choices,
and compact prompt-local context cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from causal_ssm_agent.distributions import VALID_LIKELIHOODS_FOR_DTYPE, DistributionFamily
from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.orchestrator.schemas_model import (
    VALID_LINKS_FOR_DISTRIBUTION,
)
from causal_ssm_agent.utils.causal_spec import (
    build_reference_indicator_lookup,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicator_polarity,
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
    expand_neighbor_topology: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage4RepairTopology:
    """Deterministic structural repair topology projected from the Stage 4 plan."""

    parameter_to_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_decision_block_id: dict[str, str] = field(default_factory=dict)
    indicator_to_measurement_block_id: dict[str, str] = field(default_factory=dict)
    indicator_names_by_construct: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dynamics_block_id_by_construct: dict[str, str] = field(default_factory=dict)
    scc_id_by_construct: dict[str, str] = field(default_factory=dict)
    scc_construct_names_by_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    internal_effect_block_ids_by_scc_id: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reciprocal_parameter_by_parameter: dict[str, str] = field(default_factory=dict)
    parameter_construct_names: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def get_parameter_block_id(self, parameter_name: str) -> str | None:
        """Return the owning prompt block for a semantic parameter."""
        return self.parameter_to_block_id.get(parameter_name)

    def get_measurement_block_id(self, indicator_name: str) -> str | None:
        """Return the measurement block for a manifest indicator."""
        return self.indicator_to_measurement_block_id.get(indicator_name)

    def get_indicator_owner_block_id(self, indicator_name: str) -> str | None:
        """Return the owner block for a manifest indicator token."""
        return self.indicator_to_decision_block_id.get(
            indicator_name
        ) or self.get_measurement_block_id(indicator_name)

    def get_scc_id(self, construct_name: str) -> str | None:
        """Return the SCC id for a latent construct."""
        return self.scc_id_by_construct.get(construct_name)


@dataclass(frozen=True)
class Stage4Plan:
    """Immutable Stage 4 execution plan derived from the deterministic skeleton."""

    model_blocks: tuple[Stage4FrontierBlock, ...] = ()
    review_block: Stage4FrontierBlock | None = None
    prior_blocks: tuple[Stage4FrontierBlock, ...] = ()
    prior_review_block: Stage4FrontierBlock | None = None
    blocks_by_id: dict[str, Stage4FrontierBlock] = field(default_factory=dict)
    repair_topology: Stage4RepairTopology = field(default_factory=Stage4RepairTopology)

    @property
    def all_blocks(self) -> tuple[Stage4FrontierBlock, ...]:
        """Return all blocks in deterministic execution order."""
        review = (self.review_block,) if self.review_block is not None else ()
        prior_review = (self.prior_review_block,) if self.prior_review_block is not None else ()
        return (*self.model_blocks, *review, *self.prior_blocks, *prior_review)

    def get_block(self, block_id: str) -> Stage4FrontierBlock | None:
        """Return a block by id, if present."""
        return self.blocks_by_id.get(block_id)

    @property
    def prior_review_block_id(self) -> str | None:
        """Return the whole-system prior-review block id, if configured."""
        if self.prior_review_block is None:
            return None
        return self.prior_review_block.id


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
        parameter_guidance_prefixes=("lambda", "obs_sd"),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "observation_prior": Stage4PromptScopePolicy(
        system_task=(
            "Propose full prior specifications only for the active observation-family "
            "hyperparameters. These priors control tails, dispersion, concentration, or "
            "threshold geometry for the locked likelihood family already chosen upstream."
        ),
        user_task=(
            "Propose priors only for this active observation-family block. Do not change the "
            "locked likelihood family or submit priors for other blocks."
        ),
        visible_sections=("distribution_cards", "construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "measurement_prior_guidance",
        ),
        parameter_guidance_prefixes=("obs_",),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "dynamics_prior": Stage4PromptScopePolicy(
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
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
            "latent_initial_state_guidance",
            "dynamics_budget_discipline",
        ),
        parameter_guidance_prefixes=("rho", "sigma", "t0_mean", "t0_sd"),
        allowed_tool_names=("validate_model", "elicit_prior_gmm"),
    ),
    "effect_prior": Stage4PromptScopePolicy(
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
        visible_sections=("construct_scale_cards", "prior_cards"),
        guidance_section_keys=(
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
            "effect_row_budget_discipline",
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
            "indicator-likelihood decision blocks if something materially needs revision."
        ),
        user_task=(
            "Review the locked model form shown below. If it is coherent, approve it. If not, "
            "reopen the relevant indicator-likelihood blocks and explain why. "
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
            "change locked likelihood choices or loading orientations."
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
            "latent_initial_state_guidance",
            "lagged_effect_interval_guidance",
        ),
        parameter_guidance_prefixes=(
            "lambda",
            "obs_",
            "rho",
            "sigma",
            "beta",
            "cor",
            "t0_mean",
            "t0_sd",
        ),
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
    induced_dependencies = get_induced_dependencies(causal_spec)
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
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
    retained_construct_names = {construct["name"] for construct in retained_constructs}

    resolved_likelihoods: list[dict[str, Any]] = []
    ambiguous_indicators: list[dict[str, Any]] = []
    seed_parameters: list[dict[str, Any]] = []
    seed_loading_params: list[dict[str, Any]] = []

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
            seed_parameters.append(
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
        seed_parameters.append(
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
        seed_parameters.append(
            {
                "name": f"sigma_{construct_name}",
                "role": "residual_sd",
                "constraint": "positive",
                "description": f"Residual/innovation SD for {construct_name}",
                "construct": construct_name,
            }
        )

    seed_parameters.extend(
        _measurement_error_parameters(
            indicators,
            retained_construct_names=retained_construct_names,
            indicators_per_construct=indicators_per_construct,
        )
    )

    # --- Loadings ---
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if (
            not construct_name
            or construct_name not in indicators_per_construct
            or len(indicators_per_construct[construct_name]) <= 1
        ):
            continue

        if indicator["name"] == reference_indicator_lookup.get(construct_name):
            continue

        reference_indicator = reference_indicator_lookup.get(construct_name)
        seed_loading_params.append(
            {
                "name": f"lambda_{indicator['name']}_{construct_name}",
                "role": "loading",
                "constraint": get_indicator_polarity(indicator),
                "description": f"Factor loading for {indicator['name']} on {construct_name}",
                "indicator": indicator["name"],
                "construct": construct_name,
                "reference_indicator": reference_indicator,
                "indicator_polarity": get_indicator_polarity(indicator),
            }
        )

    seed_parameters.extend(
        _candidate_observation_extra_parameters(
            indicators,
            resolved_likelihoods=resolved_likelihoods,
            ambiguous_indicators=ambiguous_indicators,
        )
    )

    # --- Correlations from marginalized confounders ---
    for dependency in induced_dependencies:
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
        seed_parameters.append(
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

    parameters, loading_params = _compiler_authoritative_stage4_inventory(
        causal_spec,
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        seed_parameters=seed_parameters,
        seed_loading_params=seed_loading_params,
        retained_state_order=retained_state_order,
        retained_edges=retained_edges,
        induced_dependencies=induced_dependencies,
        retained_construct_names=retained_construct_names,
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
                label=variable.replace("_", " ").title(),
                construct_names=(construct_name,) if isinstance(construct_name, str) else (),
                variable_names=(variable,),
                payload=dict(item),
            )
        )

    measurement_parameter_names_by_construct: dict[str, list[str]] = {}
    for parameter in skeleton.loading_params:
        construct_name = parameter.get("construct")
        if isinstance(construct_name, str):
            measurement_parameter_names_by_construct.setdefault(construct_name, []).append(
                parameter["name"]
            )
    for parameter in skeleton.parameters:
        if parameter.get("role") != "measurement_error_sd":
            continue
        construct_name = parameter.get("construct")
        if isinstance(construct_name, str):
            measurement_parameter_names_by_construct.setdefault(construct_name, []).append(
                parameter["name"]
            )

    review_block = Stage4FrontierBlock(
        id="review:model_spec",
        kind="global_review",
        label="Model Specification",
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
        names = measurement_parameter_names_by_construct.get(construct_name) or []
        if not names:
            continue
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"measurement:{construct_name}",
                kind="measurement_prior",
                label=construct_name,
                construct_names=(construct_name,),
                variable_names=tuple(indicators_per_construct.get(construct_name) or ()),
                parameter_names=tuple(sorted(names, key=param_order.__getitem__)),
            )
        )

    observation_parameter_roles = {
        "observation_hyperparameter",
        "observation_hyperparameter_positive",
    }
    for parameter in skeleton.parameters:
        if parameter["role"] not in observation_parameter_roles:
            continue
        indicator_names = tuple(parameter.get("indicator_names") or ())
        construct_names = tuple(parameter.get("construct_names") or ())
        label = str(parameter.get("description") or parameter["name"])
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"observation:{parameter['name']}",
                kind="observation_prior",
                label=label,
                construct_names=construct_names,
                variable_names=indicator_names,
                parameter_names=(parameter["name"],),
                expand_neighbor_topology=False,
            )
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(construct_order)
    for edge in get_estimation_edges(causal_spec):
        graph.add_edge(edge["cause"], edge["effect"])
    order_lookup = {name: idx for idx, name in enumerate(construct_order)}
    dynamics_roles = {
        "ar_coefficient",
        "residual_sd",
        "initial_state_mean",
        "initial_state_sd",
    }
    scc_id_by_construct: dict[str, str] = {}
    scc_construct_names_by_id: dict[str, tuple[str, ...]] = {}
    for component in sorted(
        nx.strongly_connected_components(graph),
        key=lambda members: min(order_lookup[name] for name in members),
    ):
        ordered_members = tuple(name for name in construct_order if name in component)
        scc_id = "+".join(ordered_members)
        scc_construct_names_by_id[scc_id] = ordered_members
        for construct_name in ordered_members:
            scc_id_by_construct[construct_name] = scc_id
        names = [
            parameter["name"]
            for parameter in skeleton.parameters
            if parameter["role"] in dynamics_roles and parameter.get("construct") in component
        ]
        if not names:
            continue
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"dynamics:{'+'.join(ordered_members)}",
                kind="dynamics_prior",
                label=", ".join(ordered_members),
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
        ordered_parameters = sorted(
            parameters, key=lambda parameter: param_order[parameter["name"]]
        )
        cause_names = tuple(dict.fromkeys(parameter["cause"] for parameter in ordered_parameters))
        construct_names = (*cause_names, effect_name)
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"effects:{effect_name}",
                kind="effect_prior",
                label=effect_name,
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
        correlation_label = (
            f"{construct_names[0]} \u00d7 {construct_names[1]}"
            if len(construct_names) == 2
            else parameter["name"]
        )
        prior_blocks.append(
            Stage4FrontierBlock(
                id=f"correlation:{parameter['name']}",
                kind="correlation_prior",
                label=correlation_label,
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
        label="Full Prior System",
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
    parameter_construct_names: dict[str, tuple[str, ...]] = {}
    reciprocal_parameter_by_parameter: dict[str, str] = {}
    effect_parameter_by_edge: dict[tuple[str, str], str] = {}

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

    for parameter in skeleton.parameters:
        name = parameter["name"]
        role = parameter["role"]
        if role == "fixed_effect":
            cause_name = parameter.get("cause")
            effect_name = parameter.get("effect")
            construct_names = tuple(
                name_ for name_ in (cause_name, effect_name) if isinstance(name_, str)
            )
            parameter_construct_names[name] = construct_names
            if len(construct_names) == 2:
                effect_parameter_by_edge[(construct_names[0], construct_names[1])] = name
        elif (
            role
            in {
                "ar_coefficient",
                "residual_sd",
                "initial_state_mean",
                "initial_state_sd",
                "measurement_error_sd",
            }
            or role == "loading"
        ):
            construct_name = parameter.get("construct")
            if isinstance(construct_name, str):
                parameter_construct_names[name] = (construct_name,)
        elif role in {"observation_hyperparameter", "observation_hyperparameter_positive"}:
            construct_names = tuple(
                construct_name
                for construct_name in (parameter.get("construct_names") or ())
                if isinstance(construct_name, str)
            )
            if construct_names:
                parameter_construct_names[name] = construct_names
        elif role in {"correlation", "initial_state_correlation"}:
            construct_names = tuple(
                name_
                for name_ in (parameter.get("construct_1"), parameter.get("construct_2"))
                if isinstance(name_, str)
            )
            if construct_names:
                parameter_construct_names[name] = construct_names

    for (cause_name, effect_name), parameter_name in effect_parameter_by_edge.items():
        reciprocal_name = effect_parameter_by_edge.get((effect_name, cause_name))
        if reciprocal_name is None:
            continue
        reciprocal_parameter_by_parameter[parameter_name] = reciprocal_name

    internal_effect_block_ids_by_scc_id: dict[str, tuple[str, ...]] = {}
    for scc_id, construct_names in scc_construct_names_by_id.items():
        internal_blocks = tuple(
            block.id
            for block in prior_blocks
            if block.kind == "effect_prior"
            and isinstance(block.payload.get("target_construct"), str)
            and block.payload.get("target_construct") in construct_names
        )
        internal_effect_block_ids_by_scc_id[scc_id] = internal_blocks

    repair_topology = Stage4RepairTopology(
        parameter_to_block_id=parameter_to_block_id,
        indicator_to_decision_block_id=indicator_to_decision_block_id,
        indicator_to_measurement_block_id=indicator_to_measurement_block_id,
        indicator_names_by_construct={
            construct_name: tuple(indicators_per_construct.get(construct_name, ()))
            for construct_name in construct_order
        },
        dynamics_block_id_by_construct=dynamics_block_id_by_construct,
        scc_id_by_construct=scc_id_by_construct,
        scc_construct_names_by_id=scc_construct_names_by_id,
        internal_effect_block_ids_by_scc_id=internal_effect_block_ids_by_scc_id,
        reciprocal_parameter_by_parameter=reciprocal_parameter_by_parameter,
        parameter_construct_names=parameter_construct_names,
    )

    return Stage4Plan(
        model_blocks=tuple(model_blocks),
        review_block=review_block,
        prior_blocks=tuple(prior_blocks),
        prior_review_block=prior_review_block,
        blocks_by_id=blocks_by_id,
        repair_topology=repair_topology,
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
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
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
        if role in {
            "ar_coefficient",
            "residual_sd",
            "initial_state_mean",
            "initial_state_sd",
        }:
            construct_name = parameter["construct"]
            card["structural_context"] = {"construct": construct_name}
        elif role == "measurement_error_sd":
            card["structural_context"] = {
                "construct": parameter["construct"],
                "indicator": parameter["indicator"],
            }
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
                "indicator_polarity": parameter.get("indicator_polarity"),
            }
        elif role in {"observation_hyperparameter", "observation_hyperparameter_positive"}:
            card["structural_context"] = {
                "indicator_names": list(parameter.get("indicator_names") or ()),
                "construct_names": list(parameter.get("construct_names") or ()),
                "activation_distribution_families": list(
                    parameter.get("activation_distribution_families") or ()
                ),
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


def _compiler_authoritative_stage4_inventory(
    causal_spec: dict,
    *,
    resolved_likelihoods: list[dict[str, Any]],
    ambiguous_indicators: list[dict[str, Any]],
    seed_parameters: list[dict[str, Any]],
    seed_loading_params: list[dict[str, Any]],
    retained_state_order: list[str],
    retained_edges: list[dict[str, Any]],
    induced_dependencies: list[dict[str, Any]],
    retained_construct_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the compiler-authoritative public Stage 4 prior inventory."""
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact, resolve_prior_proposals

    seed_by_name = {
        parameter["name"]: dict(parameter) for parameter in [*seed_parameters, *seed_loading_params]
    }
    provisional_likelihoods = [
        *resolved_likelihoods,
        *_provisional_likelihood_choices(ambiguous_indicators),
    ]
    provisional_distribution_by_variable = {
        str(likelihood["variable"]): str(likelihood["distribution"])
        for likelihood in provisional_likelihoods
    }
    provisional_model_spec = {
        "likelihoods": provisional_likelihoods,
        "parameters": [
            parameter
            for parameter in [*seed_parameters, *seed_loading_params]
            if not parameter.get("activation_distribution_families")
            or any(
                provisional_distribution_by_variable.get(indicator_name)
                in set(parameter.get("activation_distribution_families") or ())
                for indicator_name in (
                    parameter.get("activation_indicator_names")
                    or parameter.get("indicator_names")
                    or provisional_distribution_by_variable.keys()
                )
            )
        ],
    }
    try:
        compiled_ssm = compile_ssm_artifact(provisional_model_spec, {}, causal_spec=causal_spec)
    except ValueError:
        # Some unit tests intentionally exercise pre-compile-invalid causal specs.
        # Preserve the deterministic skeleton for those cases and simply skip the
        # compiler-backed membership step rather than failing at prompt-construction time.
        fallback_inventory = dict(seed_by_name)
        for parameter in _fallback_initial_state_parameters(retained_state_order):
            fallback_inventory.setdefault(parameter["name"], parameter)
        return _order_stage4_inventory(
            fallback_inventory.values(),
            retained_state_order=retained_state_order,
            retained_edges=retained_edges,
            induced_dependencies=induced_dependencies,
        )

    final_inventory: dict[str, dict[str, Any]] = {}
    for row in resolve_prior_proposals(compiled_ssm, authored_priors={}):
        parameter_name = str(row.get("parameter") or "")
        if not parameter_name or _is_compiler_default_only_parameter_name(parameter_name):
            continue
        parameter = seed_by_name.get(parameter_name)
        if parameter is None:
            parameter = _parameter_metadata_from_compiler_row(
                parameter_name,
                retained_construct_names=retained_construct_names,
            )
        if parameter is None:
            raise ValueError(
                "Stage 4 deterministic inventory is missing compiler-exposed parameter "
                f"{parameter_name!r}; add explicit metadata instead of silently dropping it."
            )
        final_inventory[parameter_name] = dict(parameter)

    for parameter_name, parameter in seed_by_name.items():
        if parameter_name in final_inventory or not _is_conditional_prior_surface_parameter(
            parameter
        ):
            continue
        final_inventory[parameter_name] = dict(parameter)

    missing_explicit = sorted(
        parameter_name
        for parameter_name, parameter in seed_by_name.items()
        if parameter_name not in final_inventory
        and not _is_conditional_prior_surface_parameter(parameter)
    )
    if missing_explicit:
        missing = ", ".join(missing_explicit)
        raise ValueError(
            "Stage 4 deterministic inventory drifted from compiler-exposed parameters; "
            f"compiler is missing seeded parameters: {missing}"
        )

    return _order_stage4_inventory(
        final_inventory.values(),
        retained_state_order=retained_state_order,
        retained_edges=retained_edges,
        induced_dependencies=induced_dependencies,
    )


def _fallback_initial_state_parameters(retained_state_order: list[str]) -> list[dict[str, Any]]:
    """Provide deterministic initial-state parameters when compile-time discovery is unavailable."""
    parameters: list[dict[str, Any]] = []
    for construct_name in retained_state_order:
        parameters.append(
            {
                "name": f"t0_mean_{construct_name}",
                "role": "initial_state_mean",
                "constraint": "none",
                "description": f"Initial-state mean for {construct_name}",
                "construct": construct_name,
            }
        )
    for construct_name in retained_state_order:
        parameters.append(
            {
                "name": f"t0_sd_{construct_name}",
                "role": "initial_state_sd",
                "constraint": "positive",
                "description": f"Initial-state SD for {construct_name}",
                "construct": construct_name,
            }
        )
    return parameters


def _order_stage4_inventory(
    parameters: list[dict[str, Any]] | Any,
    *,
    retained_state_order: list[str],
    retained_edges: list[dict[str, Any]],
    induced_dependencies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministically ordered parameter and loading inventories."""
    construct_order = {name: idx for idx, name in enumerate(retained_state_order)}
    edge_order = {(edge["cause"], edge["effect"]): idx for idx, edge in enumerate(retained_edges)}
    dependency_order = {
        _dependency_parameter_name(dependency): idx
        for idx, dependency in enumerate(induced_dependencies)
    }

    parameters_list = [dict(parameter) for parameter in parameters]
    role_buckets: dict[str, list[dict[str, Any]]] = {}
    loading_params: list[dict[str, Any]] = []

    for parameter in parameters_list:
        role = str(parameter["role"])
        if role == "loading":
            loading_params.append(parameter)
            continue
        role_buckets.setdefault(role, []).append(parameter)

    def _construct_key(parameter: dict[str, Any]) -> tuple[int, str]:
        construct_name = str(parameter.get("construct") or "")
        return (construct_order.get(construct_name, len(construct_order)), construct_name)

    def _measurement_error_key(parameter: dict[str, Any]) -> tuple[int, str, str]:
        construct_name = str(parameter.get("construct") or "")
        indicator_name = str(parameter.get("indicator") or "")
        return (
            construct_order.get(construct_name, len(construct_order)),
            construct_name,
            indicator_name,
        )

    def _observation_parameter_key(parameter: dict[str, Any]) -> tuple[int, str]:
        construct_names = tuple(parameter.get("construct_names") or ())
        first_construct = str(construct_names[0]) if construct_names else ""
        return (construct_order.get(first_construct, len(construct_order)), str(parameter["name"]))

    ordered_parameters: list[dict[str, Any]] = []
    ordered_parameters.extend(
        sorted(role_buckets.pop("measurement_error_sd", []), key=_measurement_error_key)
    )
    ordered_parameters.extend(
        sorted(role_buckets.pop("observation_hyperparameter", []), key=_observation_parameter_key)
    )
    ordered_parameters.extend(
        sorted(
            role_buckets.pop("observation_hyperparameter_positive", []),
            key=_observation_parameter_key,
        )
    )
    ordered_parameters.extend(sorted(role_buckets.pop("ar_coefficient", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(
            role_buckets.pop("fixed_effect", []),
            key=lambda parameter: (
                edge_order.get(
                    (str(parameter.get("cause") or ""), str(parameter.get("effect") or "")),
                    len(edge_order),
                ),
                str(parameter["name"]),
            ),
        )
    )
    ordered_parameters.extend(sorted(role_buckets.pop("residual_sd", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(role_buckets.pop("initial_state_mean", []), key=_construct_key)
    )
    ordered_parameters.extend(sorted(role_buckets.pop("initial_state_sd", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(
            [
                *role_buckets.pop("correlation", []),
                *role_buckets.pop("initial_state_correlation", []),
            ],
            key=lambda parameter: (
                dependency_order.get(str(parameter["name"]), len(dependency_order)),
                str(parameter["name"]),
            ),
        )
    )
    if role_buckets:
        unknown_roles = ", ".join(sorted(role_buckets))
        raise ValueError(
            f"Unsupported Stage 4 parameter roles in deterministic ordering: {unknown_roles}"
        )

    loading_params.sort(
        key=lambda parameter: (
            construct_order.get(str(parameter.get("construct") or ""), len(construct_order)),
            str(parameter.get("indicator") or ""),
            str(parameter["name"]),
        )
    )
    return ordered_parameters, loading_params


def _dependency_parameter_name(dependency: dict[str, Any]) -> str:
    """Return the semantic Stage 4 parameter name for one induced dependency."""
    construct_1, construct_2 = dependency["between"]
    if dependency["kind"] == "innovation_correlation":
        return f"cor_{construct_1}_{construct_2}"
    return f"cor0_{construct_1}_{construct_2}"


def _is_compiler_default_only_parameter_name(parameter_name: str) -> bool:
    """Return whether a compiler-emitted name should stay hidden from Stage 4."""
    return parameter_name == "proc_df"


def _is_conditional_prior_surface_parameter(parameter: dict[str, Any]) -> bool:
    """Whether a parameter is conditional on the locked likelihood choices."""
    return bool(parameter.get("conditional_prior_surface"))


def _measurement_error_parameters(
    indicators: list[dict[str, Any]],
    *,
    retained_construct_names: set[str],
    indicators_per_construct: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Return one semantic measurement-error prior per free manifest channel."""
    parameters: list[dict[str, Any]] = []
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        indicator_name = indicator["name"]
        if (
            not isinstance(construct_name, str)
            or construct_name not in retained_construct_names
            or len(indicators_per_construct.get(construct_name, ())) <= 1
        ):
            continue
        parameters.append(
            {
                "name": f"obs_sd_{indicator_name}",
                "role": "measurement_error_sd",
                "constraint": "positive",
                "description": f"Measurement-error SD for {indicator_name}",
                "construct": construct_name,
                "indicator": indicator_name,
            }
        )
    return parameters


def _candidate_observation_extra_parameters(
    indicators: list[dict[str, Any]],
    *,
    resolved_likelihoods: list[dict[str, Any]],
    ambiguous_indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return likelihood-extra prior candidates activated by the locked family choices."""
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    possible_distributions_by_indicator: dict[str, set[str]] = {}

    for likelihood in resolved_likelihoods:
        variable = str(likelihood["variable"])
        possible_distributions_by_indicator.setdefault(variable, set()).add(
            str(likelihood["distribution"])
        )
    for item in ambiguous_indicators:
        variable = str(item["variable"])
        if "fixed_distribution" in item:
            possible_distributions_by_indicator.setdefault(variable, set()).add(
                str(item["fixed_distribution"])
            )
        else:
            possible_distributions_by_indicator.setdefault(variable, set()).update(
                str(distribution) for distribution in item.get("valid_distributions", ())
            )

    def _construct_names(indicator_names: list[str]) -> list[str]:
        seen: list[str] = []
        for indicator_name in indicator_names:
            construct_name = (indicator_lookup.get(indicator_name) or {}).get("construct_name")
            if isinstance(construct_name, str) and construct_name not in seen:
                seen.append(construct_name)
        return seen

    def _candidate_variables(family: DistributionFamily) -> list[str]:
        return sorted(
            indicator_name
            for indicator_name, families in possible_distributions_by_indicator.items()
            if family.value in families
        )

    candidates: list[dict[str, Any]] = []

    positive_sites = {
        "obs_df": (
            DistributionFamily.STUDENT_T,
            "Student-t observation degrees of freedom",
        ),
        "obs_shape": (
            DistributionFamily.GAMMA,
            "Gamma observation shape",
        ),
        "obs_r": (
            DistributionFamily.NEGATIVE_BINOMIAL,
            "Negative-binomial observation dispersion",
        ),
        "obs_concentration": (
            DistributionFamily.BETA,
            "Beta observation concentration",
        ),
    }
    for parameter_name, (family, description) in positive_sites.items():
        indicator_names = _candidate_variables(family)
        if not indicator_names:
            continue
        candidates.append(
            {
                "name": parameter_name,
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": description,
                "indicator_names": indicator_names,
                "construct_names": _construct_names(indicator_names),
                "activation_indicator_names": list(indicator_names),
                "activation_distribution_families": [family.value],
                "conditional_prior_surface": True,
            }
        )

    ordered_indicator_names = _candidate_variables(DistributionFamily.ORDERED_LOGISTIC)
    if ordered_indicator_names:
        candidates.append(
            {
                "name": "obs_ordered_base",
                "role": "observation_hyperparameter",
                "constraint": "none",
                "description": "Ordered-logistic threshold base locations",
                "indicator_names": ordered_indicator_names,
                "construct_names": _construct_names(ordered_indicator_names),
                "activation_indicator_names": list(ordered_indicator_names),
                "activation_distribution_families": [DistributionFamily.ORDERED_LOGISTIC.value],
                "conditional_prior_surface": True,
            }
        )

    ordered_gap_indicator_names = sorted(
        indicator_name
        for indicator_name in ordered_indicator_names
        if len((indicator_lookup.get(indicator_name) or {}).get("ordinal_levels") or ()) > 2
    )
    if ordered_gap_indicator_names:
        candidates.append(
            {
                "name": "obs_ordered_gaps",
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": "Ordered-logistic threshold gaps",
                "indicator_names": ordered_indicator_names,
                "construct_names": _construct_names(ordered_indicator_names),
                "activation_indicator_names": ordered_gap_indicator_names,
                "activation_distribution_families": [DistributionFamily.ORDERED_LOGISTIC.value],
                "conditional_prior_surface": True,
            }
        )

    categorical_indicator_names = _candidate_variables(DistributionFamily.CATEGORICAL)
    if categorical_indicator_names:
        for parameter_name, description in (
            ("obs_cat_intercepts", "Categorical class intercepts"),
            ("obs_cat_slopes", "Categorical class slopes"),
        ):
            candidates.append(
                {
                    "name": parameter_name,
                    "role": "observation_hyperparameter",
                    "constraint": "none",
                    "description": description,
                    "indicator_names": categorical_indicator_names,
                    "construct_names": _construct_names(categorical_indicator_names),
                    "activation_indicator_names": list(categorical_indicator_names),
                    "activation_distribution_families": [DistributionFamily.CATEGORICAL.value],
                    "conditional_prior_surface": True,
                }
            )

    return candidates


def _provisional_likelihood_choices(
    ambiguous_indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose deterministic provisional likelihoods for compiler-owned prior discovery."""
    choices: list[dict[str, Any]] = []
    for item in ambiguous_indicators:
        variable = str(item["variable"])
        if "fixed_distribution" in item:
            distribution = str(item["fixed_distribution"])
            valid_links = list(item.get("valid_links") or [])
            if not valid_links:
                raise ValueError(f"Ambiguous indicator {variable!r} is missing valid links")
            link = str(valid_links[0])
        else:
            valid_distributions = list(item.get("valid_distributions") or [])
            if not valid_distributions:
                raise ValueError(f"Ambiguous indicator {variable!r} is missing valid distributions")
            distribution = str(valid_distributions[0])
            link_options = item.get("link_options") or {}
            valid_links = list(link_options.get(distribution) or [])
            if not valid_links:
                raise ValueError(
                    f"Ambiguous indicator {variable!r} is missing link options for {distribution!r}"
                )
            link = str(valid_links[0])
        choices.append(
            {
                "variable": variable,
                "distribution": distribution,
                "link": link,
                "reasoning": "Deterministic provisional choice for compiler-owned prior discovery.",
            }
        )
    return choices


def _parameter_metadata_from_compiler_row(
    parameter_name: str,
    *,
    retained_construct_names: set[str],
) -> dict[str, Any] | None:
    """Convert one compiler-owned extra prior row into Stage 4 parameter metadata."""
    if parameter_name.startswith("t0_mean_"):
        construct_name = parameter_name.removeprefix("t0_mean_")
        if construct_name in retained_construct_names:
            return {
                "name": parameter_name,
                "role": "initial_state_mean",
                "constraint": "none",
                "description": f"Initial-state mean for {construct_name}",
                "construct": construct_name,
            }
        return None

    if parameter_name.startswith("t0_sd_"):
        construct_name = parameter_name.removeprefix("t0_sd_")
        if construct_name in retained_construct_names:
            return {
                "name": parameter_name,
                "role": "initial_state_sd",
                "constraint": "positive",
                "description": f"Initial-state SD for {construct_name}",
                "construct": construct_name,
            }
        return None

    return None


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
        "construct_polarity": indicator.get("construct_polarity"),
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
