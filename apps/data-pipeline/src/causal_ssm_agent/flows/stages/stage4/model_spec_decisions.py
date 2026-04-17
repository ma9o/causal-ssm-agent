"""Stage 4-only decision schemas for locking a model spec from a deterministic skeleton."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from causal_ssm_agent.artifacts.model_spec import (
    DistributionFamily,
    InitializationPolicy,
    LinkFunction,
    ModelSpec,
    ObservationInterceptPolicy,
    validate_model_spec_dict,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_parameter_surfaces import (
    parameter_is_active_for_model_spec,
)
from causal_ssm_agent.models.model_semantics import should_auto_center_indicator


class DistributionChoice(BaseModel):
    """LLM's distribution/link choice for an indicator with ambiguous dtype."""

    variable: str = Field(description="Name of the indicator variable")
    distribution: DistributionFamily = Field(description="Chosen distribution")
    link: LinkFunction = Field(description="Chosen link function")
    reasoning: str = Field(description="Why this distribution/link")


class ModelConfigurationChoice(BaseModel):
    """LLM's global configuration for initialization, manifest intercepts, and forcing."""

    initialization_policy: InitializationPolicy = Field(
        description="Whether dynamic-state initial conditions are stationary-derived or free"
    )
    observation_intercept_policy: ObservationInterceptPolicy = Field(
        description="Whether eligible manifest intercepts remain free or are fixed"
    )
    equilibrium_forcing: bool = Field(
        description="Whether eligible dynamic states may have a continuous-time intercept"
    )
    reasoning: str = Field(description="Why this model-level configuration is coherent")


class ModelSpecDecisions(BaseModel):
    """LLM decisions for the non-deterministic parts of the model specification."""

    initialization_policy: InitializationPolicy = Field(
        description="Global initial-state policy for retained dynamic states"
    )
    observation_intercept_policy: ObservationInterceptPolicy = Field(
        description="Global policy for whether eligible manifest intercepts remain free"
    )
    equilibrium_forcing: bool = Field(
        description="Whether eligible dynamic states may have a continuous-time intercept"
    )
    distribution_choices: list[DistributionChoice] = Field(
        description="Distribution/link choices for indicators with ambiguous dtypes"
    )


def _resolve_centered_likelihoods(likelihoods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach deterministic centering tags to selected likelihood rows."""
    resolved: list[dict[str, Any]] = []
    for likelihood in likelihoods:
        support_kind = likelihood.get("support_kind")
        summary_operator = likelihood.get("summary_operator")
        distribution = likelihood["distribution"]
        link = likelihood["link"]
        resolved.append(
            {
                "variable": likelihood["variable"],
                "construct_name": likelihood.get("construct_name"),
                "distribution": distribution,
                "link": link,
                "support_kind": support_kind,
                "summary_operator": summary_operator,
                "centered": should_auto_center_indicator(
                    distribution,
                    link,
                    support_kind,
                    summary_operator,
                ),
                "reasoning": likelihood["reasoning"],
            }
        )
    return resolved


def merge_decisions_to_spec(
    resolved_likelihoods: list[dict],
    ambiguous_indicators: list[dict],
    parameters: list[dict],
    decisions: ModelSpecDecisions,
) -> tuple[ModelSpec | None, list[str]]:
    """Merge pre-computed skeleton with LLM decisions into a full ModelSpec."""
    indicator_semantics_lookup = {
        str(likelihood["variable"]): {
            "construct_name": likelihood.get("construct_name"),
            "support_kind": likelihood.get("support_kind"),
            "summary_operator": likelihood.get("summary_operator"),
        }
        for likelihood in resolved_likelihoods
        if isinstance(likelihood.get("variable"), str)
    }
    indicator_semantics_lookup.update(
        {
            str(item["variable"]): {
                "construct_name": item.get("construct_name"),
                "support_kind": item.get("support_kind"),
                "summary_operator": item.get("summary_operator"),
            }
            for item in ambiguous_indicators
            if isinstance(item.get("variable"), str)
        }
    )
    likelihood_lookup = {
        str(likelihood["variable"]): dict(likelihood)
        for likelihood in resolved_likelihoods
        if isinstance(likelihood.get("variable"), str)
    }
    for decision in decisions.distribution_choices:
        selected = {
            "variable": decision.variable,
            "distribution": decision.distribution.value,
            "link": decision.link.value,
            "reasoning": decision.reasoning,
        }
        selected.update(
            {
                "construct_name": indicator_semantics_lookup.get(decision.variable, {}).get(
                    "construct_name"
                ),
                "support_kind": indicator_semantics_lookup.get(decision.variable, {}).get(
                    "support_kind"
                ),
                "summary_operator": indicator_semantics_lookup.get(decision.variable, {}).get(
                    "summary_operator"
                ),
            }
        )
        likelihood_lookup[decision.variable] = selected

    likelihoods = _resolve_centered_likelihoods(list(likelihood_lookup.values()))
    chosen_likelihood_by_variable = {
        str(likelihood["variable"]): dict(likelihood)
        for likelihood in likelihoods
        if isinstance(likelihood.get("variable"), str)
    }
    active_parameters = [
        parameter
        for parameter in parameters
        if parameter_is_active_for_model_spec(
            parameter,
            chosen_likelihood_by_variable,
            initialization_policy=decisions.initialization_policy.value,
            observation_intercept_policy=decisions.observation_intercept_policy.value,
            equilibrium_forcing=decisions.equilibrium_forcing,
        )
    ]

    spec_dict = {
        "likelihoods": likelihoods,
        "parameters": active_parameters,
        "initialization_policy": decisions.initialization_policy.value,
        "observation_intercept_policy": decisions.observation_intercept_policy.value,
        "equilibrium_forcing": decisions.equilibrium_forcing,
    }
    return validate_model_spec_dict(spec_dict)


def validate_model_spec_decisions_dict(
    data: dict,
    resolved_likelihoods: list[dict],
    ambiguous_indicators: list[dict],
    parameters: list[dict],
) -> tuple[ModelSpec | None, list[str]]:
    """Validate a ModelSpecDecisions dict and merge with the deterministic skeleton."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    distribution_choices = data.get("distribution_choices", [])
    if not isinstance(distribution_choices, list):
        errors.append("'distribution_choices' must be a list")
        distribution_choices = []

    initialization_policy = data.get("initialization_policy")
    if initialization_policy is None:
        errors.append("'initialization_policy' is required")
    elif initialization_policy not in {entry.value for entry in InitializationPolicy}:
        errors.append(
            "'initialization_policy' invalid; must be one of "
            f"{sorted(entry.value for entry in InitializationPolicy)}"
        )

    observation_intercept_policy = data.get("observation_intercept_policy")
    if observation_intercept_policy is None:
        errors.append("'observation_intercept_policy' is required")
    elif observation_intercept_policy not in {
        entry.value for entry in ObservationInterceptPolicy
    }:
        errors.append(
            "'observation_intercept_policy' invalid; must be one of "
            f"{sorted(entry.value for entry in ObservationInterceptPolicy)}"
        )

    if "equilibrium_forcing" not in data:
        errors.append("'equilibrium_forcing' is required")
    elif not isinstance(data.get("equilibrium_forcing"), bool):
        errors.append("'equilibrium_forcing' must be a boolean")

    decided_variables = {
        choice.get("variable", "") for choice in distribution_choices if isinstance(choice, dict)
    }
    ambiguous_variables = {indicator["variable"] for indicator in ambiguous_indicators}
    missing = ambiguous_variables - decided_variables
    for variable in sorted(missing):
        errors.append(f"missing distribution_choice for ambiguous indicator '{variable}'")

    valid_distributions = {entry.value for entry in DistributionFamily}
    valid_links = {entry.value for entry in LinkFunction}
    for index, choice in enumerate(distribution_choices):
        if not isinstance(choice, dict):
            errors.append(f"distribution_choices[{index}]: must be a dictionary")
            continue
        distribution = choice.get("distribution", "")
        link = choice.get("link", "")
        if distribution and distribution not in valid_distributions:
            errors.append(
                f"distribution_choices[{index}]: distribution '{distribution}' invalid; "
                f"must be one of {sorted(valid_distributions)}"
            )
        if link and link not in valid_links:
            errors.append(
                f"distribution_choices[{index}]: link '{link}' invalid; "
                f"must be one of {sorted(valid_links)}"
            )

    if errors:
        return None, errors

    try:
        decisions = ModelSpecDecisions.model_validate(data)
    except ValidationError as exc:
        return None, [f"Schema validation error: {exc}"]

    return merge_decisions_to_spec(
        resolved_likelihoods,
        ambiguous_indicators,
        parameters,
        decisions,
    )


__all__ = [
    "DistributionChoice",
    "ModelConfigurationChoice",
    "ModelSpecDecisions",
    "merge_decisions_to_spec",
    "validate_model_spec_decisions_dict",
]
