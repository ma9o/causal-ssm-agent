"""Stage 4-only decision schemas for locking a model spec from a deterministic skeleton."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from causal_ssm_agent.artifacts.model_spec import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    validate_model_spec_dict,
)


class DistributionChoice(BaseModel):
    """LLM's distribution/link choice for an indicator with ambiguous dtype."""

    variable: str = Field(description="Name of the indicator variable")
    distribution: DistributionFamily = Field(description="Chosen distribution")
    link: LinkFunction = Field(description="Chosen link function")
    reasoning: str = Field(description="Why this distribution/link")


class ModelSpecDecisions(BaseModel):
    """LLM decisions for the non-deterministic parts of the model specification."""

    distribution_choices: list[DistributionChoice] = Field(
        description="Distribution/link choices for indicators with ambiguous dtypes"
    )


def merge_decisions_to_spec(
    resolved_likelihoods: list[dict],
    parameters: list[dict],
    decisions: ModelSpecDecisions,
) -> tuple[ModelSpec | None, list[str]]:
    """Merge pre-computed skeleton with LLM decisions into a full ModelSpec."""
    likelihoods: list[dict] = []
    for likelihood in resolved_likelihoods:
        likelihoods.append(
            {
                "variable": likelihood["variable"],
                "distribution": likelihood["distribution"],
                "link": likelihood["link"],
                "reasoning": (
                    f"Deterministic: {likelihood.get('reasoning', 'dtype has single valid option')}"
                ),
            }
        )
    for decision in decisions.distribution_choices:
        likelihoods.append(
            {
                "variable": decision.variable,
                "distribution": decision.distribution.value,
                "link": decision.link.value,
                "reasoning": decision.reasoning,
            }
        )

    chosen_distribution_by_variable = {
        str(likelihood["variable"]): str(likelihood["distribution"])
        for likelihood in likelihoods
        if isinstance(likelihood.get("variable"), str)
        and isinstance(likelihood.get("distribution"), str)
    }
    active_parameters = [
        parameter
        for parameter in parameters
        if _parameter_is_active_for_likelihoods(parameter, chosen_distribution_by_variable)
    ]

    spec_dict = {"likelihoods": likelihoods, "parameters": active_parameters}
    return validate_model_spec_dict(spec_dict)


def _parameter_is_active_for_likelihoods(
    parameter: dict,
    chosen_distribution_by_variable: dict[str, str],
) -> bool:
    """Return whether a candidate Stage 4 parameter is active for the locked likelihoods."""
    activation_families = parameter.get("activation_distribution_families")
    if not isinstance(activation_families, list) or not activation_families:
        return True

    relevant_variables = parameter.get("activation_indicator_names")
    if not isinstance(relevant_variables, list) or not relevant_variables:
        relevant_variables = parameter.get("indicator_names")
    if not isinstance(relevant_variables, list) or not relevant_variables:
        relevant_variables = list(chosen_distribution_by_variable)

    allowed_families = {str(family) for family in activation_families}
    return any(
        chosen_distribution_by_variable.get(str(variable)) in allowed_families
        for variable in relevant_variables
    )


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

    return merge_decisions_to_spec(resolved_likelihoods, parameters, decisions)


__all__ = [
    "DistributionChoice",
    "ModelSpecDecisions",
    "merge_decisions_to_spec",
    "validate_model_spec_decisions_dict",
]
