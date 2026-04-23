"""Stage 4 block-local submission policies and validators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
from pydantic import ValidationError

from causal_ssm_agent.flows.stages.stage4.model_spec_decisions import (
    DistributionChoice,
    ModelConfigurationChoice,
)

from .stage4_block_specs import (
    Stage4BlockKindSpec,
    Stage4PromptScopePolicy,
    get_stage4_block_kind_spec,
    get_stage4_block_kinds,
)
from .stage4_navigation import (
    active_block_parameter_names,
    current_equilibrium_forcing,
    current_initialization_policy,
    current_observation_intercept_policy,
    required_block_parameter_names,
)
from .stage4_text import summarize_stage4_names

if TYPE_CHECKING:
    from collections.abc import Callable

    from .stage4_orchestrator import Stage4FrontierBlock
    from .stage4_state import Stage4Runtime


def _project_default_model_topology(
    model_topology: dict[str, Any],
    block: Stage4FrontierBlock,
) -> dict[str, Any]:
    """Restrict model topology to the constructs directly visible in this block."""
    if not model_topology:
        return {}

    filtered_edges = model_topology.get("latent_edges") or []
    construct_names = set(block.construct_names)
    if construct_names:
        filtered_edges = [
            edge
            for edge in filtered_edges
            if edge.get("cause") in construct_names and edge.get("effect") in construct_names
        ]

    return {
        "model_clock": model_topology.get("model_clock"),
        "model_interval_days": model_topology.get("model_interval_days"),
        "outcome": model_topology.get("outcome"),
        "latent_edges": filtered_edges,
    }


def _project_effect_prior_model_topology(
    model_topology: dict[str, Any],
    block: Stage4FrontierBlock,
) -> dict[str, Any]:
    """Expand effect-prior topology to immediate neighbors when requested."""
    if not model_topology:
        return {}

    filtered_edges = model_topology.get("latent_edges") or []
    construct_names = set(block.construct_names)
    if construct_names and block.expand_neighbor_topology:
        expanded_construct_names = set(construct_names)
        for edge in filtered_edges:
            cause = edge.get("cause")
            effect = edge.get("effect")
            if cause in construct_names or effect in construct_names:
                if isinstance(cause, str):
                    expanded_construct_names.add(cause)
                if isinstance(effect, str):
                    expanded_construct_names.add(effect)
        construct_names = expanded_construct_names

    if construct_names:
        filtered_edges = [
            edge
            for edge in filtered_edges
            if edge.get("cause") in construct_names and edge.get("effect") in construct_names
        ]

    ordered_focus_constructs = tuple(
        dict.fromkeys(
            name
            for name in (block.payload.get("target_construct"), *block.construct_names)
            if isinstance(name, str) and name
        )
    )
    scc_memberships = _effect_prior_scc_memberships(
        model_topology.get("latent_edges") or [],
        ordered_focus_constructs,
    )

    return {
        "model_clock": model_topology.get("model_clock"),
        "model_interval_days": model_topology.get("model_interval_days"),
        "outcome": model_topology.get("outcome"),
        "latent_edges": filtered_edges,
        "scc_memberships": scc_memberships,
    }


def _effect_prior_scc_memberships(
    latent_edges: list[dict[str, Any]],
    focus_constructs: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return compact SCC summaries for the constructs visible in an effect block."""
    if not latent_edges or not focus_constructs:
        return []

    graph = nx.DiGraph()
    node_order: dict[str, int] = {}

    def _remember_node(name: str) -> None:
        if name not in node_order:
            node_order[name] = len(node_order)

    for edge in latent_edges:
        cause = edge.get("cause")
        effect = edge.get("effect")
        if not isinstance(cause, str) or not isinstance(effect, str):
            continue
        _remember_node(cause)
        _remember_node(effect)
        graph.add_edge(cause, effect)
    for name in focus_constructs:
        _remember_node(name)
        graph.add_node(name)

    component_by_node = {
        node: component
        for component in nx.strongly_connected_components(graph)
        for node in component
    }

    seen_components: set[tuple[str, ...]] = set()
    summaries: list[dict[str, Any]] = []
    for focus_name in focus_constructs:
        component = component_by_node.get(focus_name)
        if component is None:
            continue
        ordered_members = tuple(sorted(component, key=node_order.__getitem__))
        if ordered_members in seen_components:
            continue
        seen_components.add(ordered_members)
        summaries.append(
            {
                "focus_constructs": [name for name in focus_constructs if name in component],
                "members": list(ordered_members),
                "feedback_coupled": len(component) > 1
                or any(graph.has_edge(name, name) for name in component),
            }
        )
    return summaries


def _default_frontier_status_lines(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
    *,
    causal_spec: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Return any block-family-specific frontier status lines."""
    del causal_spec
    if not block.parameter_names:
        return ()

    if block.coverage_policy == "exact_single_choice":
        return ()

    active_parameter_names = active_block_parameter_names(block, runtime)
    if not active_parameter_names:
        return ()

    if block.coverage_policy == "subset_allowed":
        return (
            "- submission coverage: `subset repairs allowed`",
            "- revise only the parameters needed to resolve the current failure",
        )

    required_parameter_names = required_block_parameter_names(block, runtime)
    optional_parameter_names = tuple(
        name for name in active_parameter_names if name not in set(required_parameter_names)
    )
    lines = [
        "- submission coverage: `submit all required active parameters in this block`",
        (f"- required parameters: {summarize_stage4_names(list(required_parameter_names))}"),
    ]
    if optional_parameter_names:
        lines.append(
            "- optional parameters that may be omitted: "
            f"{summarize_stage4_names(list(optional_parameter_names))}"
        )
    return tuple(lines)


def _model_configuration_frontier_status_lines(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
    *,
    causal_spec: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Return allowed model-configuration options and current draft state."""
    del causal_spec
    payload = block.payload
    initialization_policy = current_initialization_policy(runtime) or "unset"
    observation_intercept_policy = current_observation_intercept_policy(runtime) or "unset"
    equilibrium_forcing = current_equilibrium_forcing(runtime)
    equilibrium_text = "unset" if equilibrium_forcing is None else str(equilibrium_forcing).lower()
    return (
        "- allowed initialization_policy values: `stationary`, `free`",
        "- allowed observation_intercept_policy values: `fixed`, `free`",
        "- allowed equilibrium_forcing values: `true`, `false`",
        f"- current draft initialization_policy: `{initialization_policy}`",
        f"- current draft observation_intercept_policy: `{observation_intercept_policy}`",
        f"- current draft equilibrium_forcing: `{equilibrium_text}`",
        (
            "- centered-indicator constructs that can identify a latent baseline if forcing is enabled: "
            f"{summarize_stage4_names(list(payload.get('centerable_construct_names') or []))}"
        ),
        (
            "- compiled baseline-factor scales from marginalized time-invariant confounders: "
            f"{summarize_stage4_names(list(payload.get('baseline_factor_names') or []))}"
        ),
    )


def _effect_prior_frontier_status_lines(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
    *,
    causal_spec: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Return effect-row budget guidance for effect-prior blocks."""
    from .stage4_partial_drift import build_effect_row_budget

    target_construct = block.payload.get("target_construct")
    if not isinstance(target_construct, str):
        return ()

    budget = build_effect_row_budget(
        model_spec=runtime.domain.accepted.model_spec,
        authored_priors=runtime.domain.accepted.authored_priors,
        causal_spec=causal_spec,
        target_construct=target_construct,
    )
    if budget is None:
        return ()

    return (
        "- stability budget source: `compiled CT drift row` (advisory headroom guidance)",
        (
            f"- target row budget guidance: `{budget.diagonal_magnitude:.3f}` "
            f"(conservative lower bound `{budget.diagonal_lower_bound:.3f}`)"
        ),
        (
            f"- incoming effect mass currently used: `{budget.used_abs_mean:.3f}` "
            f"(conservative `{budget.used_abs_upper:.3f}`) across "
            f"`{budget.specified_incoming_edges}/{budget.total_incoming_edges}` edges"
        ),
        (
            f"- remaining headroom guidance: `{budget.remaining_abs_mean:.3f}` "
            f"(conservative `{budget.remaining_abs_upper:.3f}`)"
        ),
    )


def _serialize_stage4_transition_priors(
    block: Stage4FrontierBlock,
    priors: dict[str, Any],
) -> list[dict[str, Any]]:
    """Serialize one block's accepted priors for transition events."""
    serialized: list[dict[str, Any]] = []
    for parameter_name in block.parameter_names:
        prior = priors.get(parameter_name)
        if not isinstance(prior, dict):
            continue
        item: dict[str, Any] = {"parameter": parameter_name}
        for key in ("distribution", "params", "reasoning"):
            value = prior.get(key)
            if value is not None:
                item[key] = value
        serialized.append(item)
    return serialized


def _enabled_block_tool_names(
    policy: Stage4PromptScopePolicy,
    *,
    enable_literature: bool,
    enable_paraphrasing: bool,
) -> tuple[str, ...]:
    """Return the tool names that are both scope-allowed and runtime-enabled."""
    enabled: list[str] = []
    for tool_name in policy.allowed_tool_names:
        if tool_name == "search_literature" and not enable_literature:
            continue
        if tool_name == "elicit_prior_gmm" and not enable_paraphrasing:
            continue
        enabled.append(tool_name)
    return tuple(enabled)


def _build_indicator_choice_transition(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the accepted transition payload for an indicator decision."""
    choice = normalized.get("distribution_choice")
    if not isinstance(choice, dict):
        return None
    return {
        "block_id": block.id,
        "status": "accepted",
        "detail_kind": "indicator_choice",
        "variable": choice.get("variable"),
        "distribution": choice.get("distribution"),
        "link": choice.get("link"),
        "reasoning": choice.get("reasoning"),
    }


def _build_model_configuration_transition(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the accepted transition payload for a model-configuration decision."""
    choice = normalized.get("model_configuration")
    if not isinstance(choice, dict):
        return None
    return {
        "block_id": block.id,
        "status": "accepted",
        "detail_kind": "model_configuration",
        "initialization_policy": choice.get("initialization_policy"),
        "observation_intercept_policy": choice.get("observation_intercept_policy"),
        "equilibrium_forcing": choice.get("equilibrium_forcing"),
        "reasoning": choice.get("reasoning"),
    }


def _build_review_approval_transition(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the accepted transition payload for a global review approval."""
    if normalized.get("decision") != "approve":
        return None
    return {
        "block_id": block.id,
        "status": "accepted",
        "detail_kind": "review_approval",
        "reasoning": normalized.get("reasoning"),
    }


def _build_prior_bundle_transition(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the accepted transition payload for one prior bundle."""
    priors = normalized.get("priors")
    if not isinstance(priors, dict):
        return None
    return {
        "block_id": block.id,
        "status": "accepted",
        "detail_kind": "prior_bundle",
        "parameter_names": list(block.parameter_names),
        "priors": _serialize_stage4_transition_priors(block, priors),
    }


def _indicator_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for indicator-decision blocks."""
    del prior_cards
    payload = block.payload
    variable = block.variable_names[0]
    distribution = payload.get("fixed_distribution")
    if not isinstance(distribution, str):
        valid_distributions = payload.get("valid_distributions")
        if not isinstance(valid_distributions, list) or not valid_distributions:
            raise ValueError(f"Indicator block {block.id!r} is missing valid distributions")
        distribution = str(valid_distributions[0])

    valid_links = payload.get("valid_links")
    if isinstance(valid_links, list) and valid_links:
        link = str(valid_links[0])
    else:
        link_options = payload.get("link_options")
        if not isinstance(link_options, dict):
            raise ValueError(f"Indicator block {block.id!r} is missing link options")
        candidate_links = link_options.get(distribution)
        if not isinstance(candidate_links, list) or not candidate_links:
            raise ValueError(
                f"Indicator block {block.id!r} is missing links for distribution {distribution!r}"
            )
        link = str(candidate_links[0])

    return {
        "variable": variable,
        "distribution": distribution,
        "link": link,
        "reasoning": "Example only: choose one allowed distribution/link pair for the active indicator.",
    }


def _model_configuration_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for the model-configuration block."""
    del block, prior_cards
    return {
        "initialization_policy": "stationary",
        "observation_intercept_policy": "free",
        "equilibrium_forcing": False,
        "reasoning": (
            "Choose stationary initialization when the dynamic block should start on its "
            "equilibrium residual distribution, keep observation intercepts free when eligible "
            "channels need their own measurement baseline, and enable forcing only when centered "
            "additive indicators identify a latent baseline shift."
        ),
    }


def _example_prior_payload(prior_card: dict[str, Any]) -> dict[str, Any]:
    """Return one valid example prior payload for a concrete prompt-local prior card."""
    parameter = str(prior_card["parameter"])
    role = str(prior_card.get("role") or "")
    constraint = str(prior_card.get("constraint") or "")

    if role == "ar_coefficient" or constraint == "unit_interval":
        dist, params, reason = (
            "Beta",
            {"alpha": 2.0, "beta": 2.0},
            "unit-interval persistence prior for the active AR parameter.",
        )
    elif role == "fixed_effect":
        dist, params, reason = (
            "Normal",
            {"mu": 0.0, "sigma": 0.2},
            "conservative zero-centered lagged-effect prior for the active edge.",
        )
    elif role == "initial_state_mean":
        dist, params, reason = (
            "Normal",
            {"mu": 0.0, "sigma": 1.0},
            (
                "weakly informative latent-scale initial-state mean; do not copy "
                "raw indicator means or log-means unless the construct is explicitly identified "
                "on that observed scale."
            ),
        )
    elif role in {"residual_sd", "initial_state_sd", "static_state_sd", "measurement_error_sd"}:
        dist, params, reason = (
            "HalfNormal",
            {"sigma": 1.0},
            "positive scale prior for the active variance or measurement-noise parameter.",
        )
    elif role == "observation_hyperparameter_positive":
        dist, params, reason = (
            "Gamma",
            {"concentration": 5.0, "rate": 1.0},
            "positive observation-family hyperparameter prior.",
        )
    elif role == "observation_hyperparameter":
        dist, params, reason = (
            "Normal",
            {"mu": 0.0, "sigma": 1.0},
            "real-valued observation-family hyperparameter prior.",
        )
    elif role == "loading" and constraint == "negative":
        dist, params, reason = (
            "TruncatedNormal",
            {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0},
            "negative loading prior consistent with the locked indicator polarity.",
        )
    elif role == "loading":
        dist, params, reason = (
            "HalfNormal",
            {"sigma": 1.0},
            "positive loading prior consistent with the locked indicator polarity.",
        )
    elif role in {"correlation", "initial_state_correlation"} or constraint == "correlation":
        dist, params, reason = (
            "TruncatedNormal",
            {"mu": 0.0, "sigma": 0.3, "lower": -1.0, "upper": 1.0},
            "bounded correlation prior centered at zero.",
        )
    elif constraint == "positive":
        dist, params, reason = (
            "HalfNormal",
            {"sigma": 1.0},
            "positive scale prior for the active parameter.",
        )
    elif constraint == "negative":
        dist, params, reason = (
            "TruncatedNormal",
            {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0},
            "negative prior consistent with the active parameter constraint.",
        )
    else:
        dist, params, reason = (
            "Normal",
            {"mu": 0.0, "sigma": 1.0},
            "weakly informative unconstrained prior for the active parameter.",
        )

    return {
        "parameter": parameter,
        "distribution": dist,
        "params": params,
        "sources": [],
        "reasoning": f"Example only: {reason}",
    }


def _prior_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for prior-authoring blocks."""
    if not prior_cards:
        raise ValueError(f"Prior block {block.id!r} is missing prompt-local prior cards")
    if block.coverage_policy == "subset_allowed":
        prior_payload = _example_prior_payload(prior_cards[0])
        parameter = str(prior_payload["parameter"])
        return {"priors": {parameter: prior_payload}}

    required_parameter_names = set(block.required_parameter_names or block.parameter_names)
    example_priors: dict[str, Any] = {}
    for card in prior_cards:
        parameter_name = card.get("parameter")
        if not isinstance(parameter_name, str) or parameter_name not in required_parameter_names:
            continue
        example_priors[parameter_name] = _example_prior_payload(card)
    if not example_priors:
        prior_payload = _example_prior_payload(prior_cards[0])
        parameter = str(prior_payload["parameter"])
        example_priors[parameter] = prior_payload
    return {"priors": example_priors}


def _global_review_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for compact global-review blocks."""
    del block, prior_cards
    return {
        "decision": "approve",
        "reasoning": "The locked likelihoods and loading orientations are coherent for prior elicitation.",
    }


def _render_submission_example(
    submission_tool_name: str,
    example: dict[str, Any],
) -> str:
    """Render one block-local submit-tool example payload."""
    return (
        f"Use `{submission_tool_name}` with exactly this argument object:\n\n"
        "```json\n" + json.dumps(example, indent=2) + "\n```"
    )


_ACCEPTED_TRANSITION_BUILDERS = {
    "model_configuration": _build_model_configuration_transition,
    "indicator_choice": _build_indicator_choice_transition,
    "review_approval": _build_review_approval_transition,
    "prior_bundle": _build_prior_bundle_transition,
}
_SUBMISSION_EXAMPLE_BUILDERS = {
    "model_configuration_choice": _model_configuration_submission_example,
    "indicator_choice": _indicator_submission_example,
    "global_review_decision": _global_review_submission_example,
    "prior_bundle": _prior_submission_example,
}


@dataclass(frozen=True)
class Stage4BlockHandler:
    """Per-kind Stage 4 prompt and submission behavior."""

    spec: Stage4BlockKindSpec
    normalize_submission: Callable[
        [Stage4FrontierBlock, dict[str, Any]],
        tuple[dict[str, Any] | None, str | None],
    ]
    project_model_topology: Callable[[dict[str, Any], Stage4FrontierBlock], dict[str, Any]]
    build_frontier_status_lines: Callable[..., tuple[str, ...]]

    @property
    def kind(self) -> str:
        """Return the block kind handled by this object."""
        return self.spec.kind

    @property
    def prompt_policy(self) -> Stage4PromptScopePolicy:
        """Return the prompt policy for this handler."""
        return self.spec.prompt_policy

    @property
    def submission_payload_kind(self) -> str:
        """Return the normalized submission payload kind for this block kind."""
        return self.spec.submission_payload_kind

    @property
    def submission_tool_name(self) -> str:
        """Return the primary submit tool required for this block kind."""
        return self.spec.prompt_policy.allowed_tool_names[0]

    def allowed_tool_names(
        self,
        *,
        enable_literature: bool,
        enable_paraphrasing: bool,
    ) -> tuple[str, ...]:
        """Return runtime-enabled tools for this block kind."""
        return _enabled_block_tool_names(
            self.prompt_policy,
            enable_literature=enable_literature,
            enable_paraphrasing=enable_paraphrasing,
        )

    def include_prior_source_guidance_for_prompt(
        self,
        *,
        enable_literature: bool,
    ) -> bool:
        """Whether the prompt should mention authored literature-source payloads."""
        return self.spec.include_prior_source_guidance and enable_literature

    def build_accepted_transition(
        self,
        block: Stage4FrontierBlock,
        normalized: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build the accepted transition payload for one normalized submission."""
        return _ACCEPTED_TRANSITION_BUILDERS[self.spec.accepted_transition_kind](block, normalized)

    def render_submission_example(
        self,
        block: Stage4FrontierBlock,
        *,
        prior_cards: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render the block-local submit-tool example payload."""
        example = _SUBMISSION_EXAMPLE_BUILDERS[self.submission_payload_kind](
            block,
            prior_cards=prior_cards or [],
        )
        return _render_submission_example(self.submission_tool_name, example)

    def render_frontier_status_lines(
        self,
        block: Stage4FrontierBlock,
        runtime: Stage4Runtime,
        *,
        causal_spec: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        """Render any block-family-specific frontier status lines."""
        return self.build_frontier_status_lines(
            block,
            runtime,
            causal_spec=causal_spec,
        )


def validate_stage4_submission_payload(
    data: dict[str, Any],
) -> str | None:
    """Validate that a Stage 4 tool submission payload is a JSON object."""
    if not isinstance(data, dict):
        return "VALIDATION ERRORS:\n- submission must be a JSON object"
    return None


def validate_stage4_block_coverage(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
    normalized: dict[str, Any],
) -> str | None:
    """Validate the runtime-aware coverage contract for one normalized submission."""
    if block.coverage_policy != "all_required_parameters":
        return None
    priors = normalized.get("priors")
    if not isinstance(priors, dict):
        return None
    required_parameter_names = required_block_parameter_names(block, runtime)
    missing = tuple(
        parameter_name
        for parameter_name in required_parameter_names
        if parameter_name not in priors
    )
    if not missing:
        return None
    return (
        "VALIDATION ERRORS:\n- missing required priors for this block: "
        f"{summarize_stage4_names(list(missing))}"
    )


def _normalize_indicator_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an indicator-decision proposal."""
    try:
        choice = DistributionChoice.model_validate(proposal).model_dump(mode="json")
    except ValidationError as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"

    variable = block.variable_names[0]
    if choice["variable"] != variable:
        return None, f"VALIDATION ERRORS:\n- proposal variable must be `{variable}`"

    item = block.payload
    allowed_distributions = (
        [item["fixed_distribution"]]
        if "fixed_distribution" in item
        else item.get("valid_distributions", [])
    )
    if choice["distribution"] not in allowed_distributions:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- distribution `{choice['distribution']}` is invalid for `{variable}`",
        )

    allowed_links = (
        item.get("valid_links", [])
        if "fixed_distribution" in item
        else item.get("link_options", {}).get(choice["distribution"], [])
    )
    if choice["link"] not in allowed_links:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- link `{choice['link']}` is invalid for `{variable}` with `{choice['distribution']}`",
        )
    return {"distribution_choice": choice}, None


def _normalize_model_configuration_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a model-configuration proposal."""
    del block
    try:
        choice = ModelConfigurationChoice.model_validate(proposal).model_dump(mode="json")
    except ValidationError as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"
    return {"model_configuration": choice}, None


def _normalize_prior_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a prior-block proposal."""
    raw_priors = proposal.get("priors")
    if not isinstance(raw_priors, dict) or not raw_priors:
        return None, "VALIDATION ERRORS:\n- `proposal.priors` must be a non-empty object"

    allowed = set(block.parameter_names)
    invalid = sorted(name for name in raw_priors if name not in allowed)
    if invalid:
        return (
            None,
            "VALIDATION ERRORS:\n- priors outside the active block: "
            f"{summarize_stage4_names(invalid)}",
        )

    return {"priors": raw_priors}, None


def _normalize_global_review_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a compact global-review submission."""
    decision = proposal.get("decision")
    reasoning = proposal.get("reasoning")
    if decision not in {"approve", "reopen"}:
        return None, "VALIDATION ERRORS:\n- `proposal.decision` must be `approve` or `reopen`"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "VALIDATION ERRORS:\n- `proposal.reasoning` must be a non-empty string"
    reopen_block_ids = proposal.get("reopen_block_ids")
    if decision == "approve":
        if reopen_block_ids is not None:
            return (
                None,
                "VALIDATION ERRORS:\n- `reopen_block_ids` is only valid when decision=`reopen`",
            )
        return {"decision": decision, "reasoning": reasoning.strip()}, None

    if not isinstance(reopen_block_ids, list) or not reopen_block_ids:
        return (
            None,
            "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must be a non-empty list of model block ids",
        )
    if any(not isinstance(block_id, str) for block_id in reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must contain only strings"
    if len(set(reopen_block_ids)) != len(reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must not contain duplicates"

    allowed_ids_in_order = tuple(block.payload.get("reopenable_block_ids") or ())
    allowed_ids = set(allowed_ids_in_order)
    invalid_ids = [block_id for block_id in reopen_block_ids if block_id not in allowed_ids]
    if invalid_ids:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- `reopen_block_ids` must be drawn from {summarize_stage4_names(sorted(allowed_ids))}",
        )
    return {
        "decision": decision,
        "reasoning": reasoning.strip(),
        "reopen_block_ids": tuple(
            block_id for block_id in allowed_ids_in_order if block_id in set(reopen_block_ids)
        ),
    }, None


_NORMALIZE_SUBMISSION_BY_PAYLOAD_KIND = {
    "model_configuration_choice": _normalize_model_configuration_submission,
    "indicator_choice": _normalize_indicator_submission,
    "global_review_decision": _normalize_global_review_submission,
    "prior_bundle": _normalize_prior_submission,
}
_BLOCK_HANDLERS: dict[str, Stage4BlockHandler] = {
    kind: Stage4BlockHandler(
        spec=spec,
        normalize_submission=_NORMALIZE_SUBMISSION_BY_PAYLOAD_KIND[spec.submission_payload_kind],
        project_model_topology=(
            _project_effect_prior_model_topology
            if kind == "effect_prior"
            else _project_default_model_topology
        ),
        build_frontier_status_lines=(
            _effect_prior_frontier_status_lines
            if kind == "effect_prior"
            else _model_configuration_frontier_status_lines
            if kind == "model_configuration"
            else _default_frontier_status_lines
        ),
    )
    for kind in get_stage4_block_kinds()
    for spec in (get_stage4_block_kind_spec(kind),)
}


def get_stage4_block_handler(kind: str) -> Stage4BlockHandler:
    """Return the registered handler for a block kind."""
    handler = _BLOCK_HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"Unsupported Stage 4 block kind {kind!r}")
    return handler
