"""Executable model-spec artifact models and validation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError

from nof1_causal_lab.distributions import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    PARAMETER_ROLE_SPECS,
    VALID_LIKELIHOODS_FOR_DTYPE,
    DistributionFamily,
)


class LinkFunction(StrEnum):
    """Link functions mapping linear predictor to distribution mean."""

    IDENTITY = "identity"
    LOG = "log"
    INVERSE = "inverse"
    LOGIT = "logit"
    PROBIT = "probit"
    CUMULATIVE_LOGIT = "cumulative_logit"
    SOFTMAX = "softmax"


class InitializationPolicy(StrEnum):
    """Global initial-state policy for retained dynamic states."""

    STATIONARY = "stationary"
    FREE = "free"


class ObservationInterceptPolicy(StrEnum):
    """Global policy for whether eligible manifest intercepts remain free."""

    FIXED = "fixed"
    FREE = "free"


class ParameterRole(StrEnum):
    """Role of a parameter in the model."""

    FIXED_EFFECT = "fixed_effect"
    AR_COEFFICIENT = "ar_coefficient"
    RESIDUAL_SD = "residual_sd"
    STATE_INTERCEPT = "state_intercept"
    OBSERVATION_INTERCEPT = "observation_intercept"
    INITIAL_STATE_MEAN = "initial_state_mean"
    INITIAL_STATE_SD = "initial_state_sd"
    STATIC_STATE_SD = "static_state_sd"
    CORRELATION = "correlation"
    INITIAL_STATE_CORRELATION = "initial_state_correlation"
    LOADING = "loading"
    MEASUREMENT_ERROR_SD = "measurement_error_sd"
    OBSERVATION_HYPERPARAMETER = "observation_hyperparameter"
    OBSERVATION_HYPERPARAMETER_POSITIVE = "observation_hyperparameter_positive"


class ParameterConstraint(StrEnum):
    """Constraints on parameter values."""

    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNIT_INTERVAL = "unit_interval"
    CORRELATION = "correlation"


VALID_LINKS_FOR_DISTRIBUTION: dict[DistributionFamily, set[LinkFunction]] = {
    family: {LinkFunction(link) for link in links}
    for family, links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.items()
}

EXPECTED_CONSTRAINT_FOR_ROLE: dict[ParameterRole, ParameterConstraint] = {
    ParameterRole(spec.role): ParameterConstraint(spec.constraint) for spec in PARAMETER_ROLE_SPECS
}
EXPECTED_CONSTRAINT_FOR_ROLE[ParameterRole.INITIAL_STATE_CORRELATION] = (
    ParameterConstraint.CORRELATION
)


class LikelihoodSource(BaseModel):
    """A source of evidence for a likelihood distribution choice."""

    title: str = Field(description="Title of the source (paper, textbook, etc.)")
    url: str | None = Field(default=None, description="URL of the source if available")
    snippet: str = Field(description="Relevant excerpt from the source")


class LikelihoodSpec(BaseModel):
    """Specification for a likelihood (observed variable distribution)."""

    variable: str = Field(description="Name of the observed indicator variable")
    distribution: DistributionFamily = Field(description="Distribution family for this variable")
    link: LinkFunction = Field(description="Link function mapping linear predictor to mean")
    centered: bool = Field(
        default=False,
        description="Whether deterministic additive centering is applied before fitting",
    )
    reasoning: str = Field(description="Why this distribution/link was chosen for this variable")
    sources: list[LikelihoodSource] = Field(
        default_factory=list,
        description="Literature sources supporting this likelihood choice",
    )


class ParameterSpec(BaseModel):
    """Specification for a parameter requiring a prior."""

    name: str = Field(description="Parameter name")
    role: ParameterRole = Field(description="Role of this parameter in the model")
    constraint: ParameterConstraint = Field(description="Constraint on parameter values")
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )


class ModelSpec(BaseModel):
    """Complete statistical model specification."""

    likelihoods: list[LikelihoodSpec] = Field(
        description="Likelihood specifications for each observed indicator"
    )
    parameters: list[ParameterSpec] = Field(description="All parameters requiring priors")
    initialization_policy: InitializationPolicy = Field(
        default=InitializationPolicy.STATIONARY,
        description="Whether dynamic-state initial conditions are stationary-derived or free",
    )
    observation_intercept_policy: ObservationInterceptPolicy = Field(
        default=ObservationInterceptPolicy.FREE,
        description="Whether eligible manifest intercepts remain free or are fixed",
    )
    equilibrium_forcing: bool = Field(
        default=False,
        description="Whether eligible dynamic states may carry a continuous-time intercept term",
    )


def validate_model_spec_dict(
    data: dict,
    indicators: list[dict] | None = None,
) -> tuple[ModelSpec | None, list[str]]:
    """Validate a model spec dict, collecting all errors in one pass."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    valid_roles = {entry.value for entry in ParameterRole}
    valid_constraints = {entry.value for entry in ParameterConstraint}
    valid_distributions = {entry.value for entry in DistributionFamily}
    valid_links = {entry.value for entry in LinkFunction}
    valid_initialization_policies = {entry.value for entry in InitializationPolicy}
    valid_observation_intercept_policies = {entry.value for entry in ObservationInterceptPolicy}

    likelihoods = data.get("likelihoods", [])
    if not isinstance(likelihoods, list):
        errors.append("'likelihoods' must be a list")
        likelihoods = []

    likelihood_variables = [
        item.get("variable", "") for item in likelihoods if isinstance(item, dict)
    ]
    seen_likelihood_variables: set[str] = set()
    for variable in likelihood_variables:
        if variable and variable in seen_likelihood_variables:
            errors.append(f"duplicate likelihood for variable '{variable}'")
        if variable:
            seen_likelihood_variables.add(variable)

    parameters_raw = data.get("parameters", [])
    if isinstance(parameters_raw, list):
        parameter_names = [
            item.get("name", "") for item in parameters_raw if isinstance(item, dict)
        ]
        seen_parameter_names: set[str] = set()
        for name in parameter_names:
            if name and name in seen_parameter_names:
                errors.append(f"duplicate parameter name '{name}'")
            if name:
                seen_parameter_names.add(name)

    initialization_policy = data.get(
        "initialization_policy",
        InitializationPolicy.STATIONARY.value,
    )
    if initialization_policy not in valid_initialization_policies:
        errors.append(
            "'initialization_policy' invalid; must be one of "
            f"{sorted(valid_initialization_policies)}"
        )

    observation_intercept_policy = data.get(
        "observation_intercept_policy",
        ObservationInterceptPolicy.FREE.value,
    )
    if observation_intercept_policy not in valid_observation_intercept_policies:
        errors.append(
            "'observation_intercept_policy' invalid; must be one of "
            f"{sorted(valid_observation_intercept_policies)}"
        )

    equilibrium_forcing = data.get("equilibrium_forcing", False)
    if not isinstance(equilibrium_forcing, bool):
        errors.append("'equilibrium_forcing' must be a boolean")

    indicator_dtype: dict[str, str] = {}
    if indicators:
        indicator_dtype = {
            indicator["name"]: indicator.get("measurement_dtype", "continuous")
            for indicator in indicators
        }
        missing = set(indicator_dtype) - seen_likelihood_variables
        for variable in sorted(missing):
            errors.append(f"indicator '{variable}' has no likelihood specification")

    for index, likelihood in enumerate(likelihoods):
        if not isinstance(likelihood, dict):
            errors.append(f"likelihoods[{index}]: must be a dictionary")
            continue

        variable = likelihood.get("variable", "")
        distribution = likelihood.get("distribution", "")
        link = likelihood.get("link", "")

        if distribution and distribution not in valid_distributions:
            errors.append(
                f"likelihoods[{index}] '{variable}': distribution '{distribution}' invalid; "
                f"must be one of {sorted(valid_distributions)}"
            )
        if link and link not in valid_links:
            errors.append(
                f"likelihoods[{index}] '{variable}': link '{link}' invalid; "
                f"must be one of {sorted(valid_links)}"
            )

        if distribution in valid_distributions and link in valid_links:
            distribution_enum = DistributionFamily(distribution)
            link_enum = LinkFunction(link)
            allowed_links = VALID_LINKS_FOR_DISTRIBUTION.get(distribution_enum)
            if allowed_links is not None and link_enum not in allowed_links:
                errors.append(
                    f"likelihoods[{index}] '{variable}': link '{link}' invalid for {distribution}; "
                    f"expected one of {{{', '.join(sorted(item.value for item in allowed_links))}}}"
                )

        if distribution in valid_distributions and variable in indicator_dtype:
            dtype = indicator_dtype[variable]
            allowed_distributions = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype)
            if (
                allowed_distributions is not None
                and DistributionFamily(distribution) not in allowed_distributions
            ):
                errors.append(
                    f"likelihoods[{index}] '{variable}': distribution '{distribution}' invalid for dtype '{dtype}'; "
                    f"expected one of {{{', '.join(sorted(item.value for item in allowed_distributions))}}}"
                )

    parameters = data.get("parameters", [])
    if not isinstance(parameters, list):
        errors.append("'parameters' must be a list")
        parameters = []

    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            errors.append(f"parameters[{index}]: must be a dictionary")
            continue

        name = parameter.get("name", f"[{index}]")
        role = parameter.get("role", "")
        constraint = parameter.get("constraint", "")

        if role and role not in valid_roles:
            errors.append(
                f"parameters[{index}] '{name}': role '{role}' invalid; "
                f"must be one of {sorted(valid_roles)}"
            )
        if constraint and constraint not in valid_constraints:
            errors.append(
                f"parameters[{index}] '{name}': constraint '{constraint}' invalid; "
                f"must be one of {sorted(valid_constraints)}"
            )

        if role in valid_roles and constraint in valid_constraints:
            role_enum = ParameterRole(role)
            constraint_enum = ParameterConstraint(constraint)
            expected = EXPECTED_CONSTRAINT_FOR_ROLE.get(role_enum)
            if role_enum == ParameterRole.LOADING:
                if constraint_enum not in {
                    ParameterConstraint.POSITIVE,
                    ParameterConstraint.NEGATIVE,
                }:
                    errors.append(
                        f"parameters[{index}] '{name}': constraint '{constraint}' unexpected "
                        "for role 'loading'; expected 'positive' or 'negative'"
                    )
            elif expected is not None and constraint_enum != expected:
                errors.append(
                    f"parameters[{index}] '{name}': constraint '{constraint}' unexpected "
                    f"for role '{role}'; expected '{expected.value}'"
                )
            if role_enum == ParameterRole.INITIAL_STATE_CORRELATION and not name.startswith(
                "cor0_"
            ):
                errors.append(
                    f"parameters[{index}] '{name}': initial_state_correlation parameters "
                    "must use canonical names starting with 'cor0_'"
                )

    if not errors:
        try:
            spec = ModelSpec.model_validate(data)
            return spec, []
        except ValidationError as exc:
            return None, [f"Unexpected validation error: {exc}"]

    return None, errors


__all__ = [
    "EXPECTED_CONSTRAINT_FOR_ROLE",
    "LinkFunction",
    "LikelihoodSource",
    "LikelihoodSpec",
    "ModelSpec",
    "ParameterConstraint",
    "ParameterRole",
    "ParameterSpec",
    "VALID_LINKS_FOR_DISTRIBUTION",
    "validate_model_spec_dict",
]
