"""Stage 4 parameter activation against the locked model decisions."""

from __future__ import annotations

from typing import Any

from nof1_causal_lab.artifacts.statistical_model_spec import (
    InitializationPolicy,
    ObservationInterceptPolicy,
)
from nof1_causal_lab.models.model_semantics import indicator_requires_observation_intercept


def parameter_is_active_for_statistical_model_spec(
    parameter: dict[str, Any],
    chosen_likelihood_by_variable: dict[str, dict[str, Any]],
    *,
    initialization_policy: str,
    observation_intercept_policy: str,
    equilibrium_forcing: bool,
) -> bool:
    """Return whether a Stage 4 parameter survives the locked model decisions."""
    activation_families = parameter.get("activation_distribution_families")
    if not isinstance(activation_families, list) or not activation_families:
        family_active = True
    else:
        relevant_variables = parameter.get("activation_indicator_names")
        if not isinstance(relevant_variables, list) or not relevant_variables:
            relevant_variables = parameter.get("indicator_names")
        if not isinstance(relevant_variables, list) or not relevant_variables:
            relevant_variables = list(chosen_likelihood_by_variable)

        allowed_families = {str(family) for family in activation_families}
        family_active = any(
            str(chosen_likelihood_by_variable.get(str(variable), {}).get("distribution"))
            in allowed_families
            for variable in relevant_variables
        )
    if not family_active:
        return False

    role = str(parameter["role"])
    if role == "observation_intercept":
        if observation_intercept_policy == ObservationInterceptPolicy.FIXED.value:
            return False
        indicator_name = str(parameter.get("indicator") or "")
        likelihood = chosen_likelihood_by_variable.get(indicator_name) or {}
        distribution = likelihood.get("distribution")
        link = likelihood.get("link")
        if distribution is None or link is None:
            return False
        return indicator_requires_observation_intercept(
            distribution,
            link,
            likelihood.get("support_kind"),
            likelihood.get("summary_operator"),
            centered=bool(likelihood.get("centered")),
        )

    if role == "state_intercept":
        if not equilibrium_forcing:
            return False
        construct_name = str(parameter.get("construct") or "")
        return any(
            bool(likelihood.get("centered"))
            and str(likelihood.get("construct_name") or "") == construct_name
            for likelihood in chosen_likelihood_by_variable.values()
        )

    if role in {"initial_state_mean", "initial_state_sd"}:
        if initialization_policy == InitializationPolicy.FREE.value:
            return True
        return str(parameter.get("temporal_status") or "") == "time_invariant"

    return True
