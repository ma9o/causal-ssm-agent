"""Model specification schemas for Stage 4 orchestrator.

These schemas define the structure proposed by the orchestrator LLM
for the statistical model specification.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError

from causal_ssm_agent.distributions import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    PARAMETER_ROLE_SPECS,
    VALID_LIKELIHOODS_FOR_DTYPE,
    DistributionFamily,
)


class LinkFunction(StrEnum):
    """Link functions mapping linear predictor to distribution mean."""

    IDENTITY = "identity"  # Gaussian
    LOG = "log"  # Poisson, Gamma, NegativeBinomial
    INVERSE = "inverse"  # Gamma (canonical)
    LOGIT = "logit"  # Bernoulli, Beta
    PROBIT = "probit"  # Bernoulli
    CUMULATIVE_LOGIT = "cumulative_logit"  # OrderedLogistic
    SOFTMAX = "softmax"  # Categorical


class ParameterRole(StrEnum):
    """Role of a parameter in the model."""

    FIXED_EFFECT = "fixed_effect"  # Beta coefficients for causal effects
    AR_COEFFICIENT = "ar_coefficient"  # DT persistence rho for autoregressive terms
    RESIDUAL_SD = "residual_sd"  # Sigma for residual variance
    INITIAL_STATE_MEAN = "initial_state_mean"  # Mean of the latent initial state
    INITIAL_STATE_SD = "initial_state_sd"  # SD of the latent initial state
    STATIC_STATE_SD = "static_state_sd"  # Scale for quasi-constant latent states
    CORRELATION = "correlation"  # Correlation between latent innovations
    INITIAL_STATE_CORRELATION = (
        "initial_state_correlation"  # Correlation between initial latent states
    )
    LOADING = "loading"  # Factor loading for multi-indicator constructs
    MEASUREMENT_ERROR_SD = "measurement_error_sd"  # Manifest observation noise SD
    OBSERVATION_HYPERPARAMETER = "observation_hyperparameter"  # Real-valued obs extras
    OBSERVATION_HYPERPARAMETER_POSITIVE = (
        "observation_hyperparameter_positive"  # Positive obs extras
    )


class ParameterConstraint(StrEnum):
    """Constraints on parameter values."""

    NONE = "none"  # Unconstrained (can be any real number)
    POSITIVE = "positive"  # Must be > 0 (variances, SDs)
    NEGATIVE = "negative"  # Must be < 0 (inverse-coded loadings)
    UNIT_INTERVAL = "unit_interval"  # Must be in [0, 1] (probabilities)
    CORRELATION = "correlation"  # Must be in [-1, 1]


VALID_LINKS_FOR_DISTRIBUTION: dict[DistributionFamily, set[LinkFunction]] = {
    family: {LinkFunction(link) for link in links}
    for family, links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.items()
}

# Derived: role -> constraint lookup used by validation code.
# Authoritative source is PARAMETER_ROLE_SPECS in distributions.py.
EXPECTED_CONSTRAINT_FOR_ROLE: dict[ParameterRole, ParameterConstraint] = {
    ParameterRole(spec.role): ParameterConstraint(spec.constraint) for spec in PARAMETER_ROLE_SPECS
}
# initial_state_correlation shares the correlation constraint but is not
# a user-facing doc row, so add it manually.
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
    reasoning: str = Field(description="Why this distribution/link was chosen for this variable")
    sources: list[LikelihoodSource] = Field(
        default_factory=list,
        description="Literature sources supporting this likelihood choice",
    )


class ParameterSpec(BaseModel):
    """Specification for a parameter requiring a prior."""

    name: str = Field(description="Parameter name (e.g., 'beta_stress_anxiety', 'rho_mood')")
    role: ParameterRole = Field(description="Role of this parameter in the model")
    constraint: ParameterConstraint = Field(description="Constraint on parameter values")
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )


class ModelSpec(BaseModel):
    """Complete model specification from orchestrator.

    This is what the orchestrator proposes based on the CausalSpec structure.
    It enumerates all parameters needing priors and specifies the statistical model.
    """

    likelihoods: list[LikelihoodSpec] = Field(
        description="Likelihood specifications for each observed indicator"
    )
    parameters: list[ParameterSpec] = Field(description="All parameters requiring priors")


def validate_model_spec_dict(
    data: dict,
    indicators: list[dict] | None = None,
) -> tuple[ModelSpec | None, list[str]]:
    """Validate a model spec dict, collecting ALL errors in one pass.

    Matches the pattern of validate_latent_model() and validate_measurement_model()
    from schemas.py: works on raw dicts, surfaces both schema errors (bad enum values)
    and domain errors (wrong constraints, incompatible links) together.

    Args:
        data: Dictionary to validate as ModelSpec
        indicators: Optional list of indicator dicts for dtype checking

    Returns:
        Tuple of (validated ModelSpec or None, list of error messages)
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    valid_roles = {e.value for e in ParameterRole}
    valid_constraints = {e.value for e in ParameterConstraint}
    valid_distributions = {e.value for e in DistributionFamily}
    valid_links = {e.value for e in LinkFunction}

    # --- Uniqueness and coverage checks ---
    likelihoods = data.get("likelihoods", [])
    if not isinstance(likelihoods, list):
        errors.append("'likelihoods' must be a list")
        likelihoods = []

    # Duplicate likelihood variables
    lik_vars = [lik.get("variable", "") for lik in likelihoods if isinstance(lik, dict)]
    seen_lik_vars: set[str] = set()
    for var in lik_vars:
        if var and var in seen_lik_vars:
            errors.append(f"duplicate likelihood for variable '{var}'")
        if var:
            seen_lik_vars.add(var)

    # Duplicate parameter names
    parameters_raw = data.get("parameters", [])
    if isinstance(parameters_raw, list):
        param_names = [p.get("name", "") for p in parameters_raw if isinstance(p, dict)]
        seen_param_names: set[str] = set()
        for name in param_names:
            if name and name in seen_param_names:
                errors.append(f"duplicate parameter name '{name}'")
            if name:
                seen_param_names.add(name)

    # Coverage: one likelihood per indicator
    indicator_dtype = {}
    if indicators:
        indicator_dtype = {
            ind["name"]: ind.get("measurement_dtype", "continuous") for ind in indicators
        }
        missing = set(indicator_dtype.keys()) - seen_lik_vars
        for var in sorted(missing):
            errors.append(f"indicator '{var}' has no likelihood specification")

    for i, lik in enumerate(likelihoods):
        if not isinstance(lik, dict):
            errors.append(f"likelihoods[{i}]: must be a dictionary")
            continue

        var = lik.get("variable", "")
        dist = lik.get("distribution", "")
        link = lik.get("link", "")

        # Enum validation
        if dist and dist not in valid_distributions:
            errors.append(
                f"likelihoods[{i}] '{var}': distribution '{dist}' invalid; "
                f"must be one of {sorted(valid_distributions)}"
            )
        if link and link not in valid_links:
            errors.append(
                f"likelihoods[{i}] '{var}': link '{link}' invalid; "
                f"must be one of {sorted(valid_links)}"
            )

        # Domain: distribution <-> link compatibility
        if dist in valid_distributions and link in valid_links:
            dist_enum = DistributionFamily(dist)
            link_enum = LinkFunction(link)
            ok_links = VALID_LINKS_FOR_DISTRIBUTION.get(dist_enum)
            if ok_links is not None and link_enum not in ok_links:
                errors.append(
                    f"likelihoods[{i}] '{var}': link '{link}' invalid for {dist}; "
                    f"expected one of {{{', '.join(sorted(lf.value for lf in ok_links))}}}"
                )

        # Domain: dtype <-> distribution compatibility
        if dist in valid_distributions and var in indicator_dtype:
            dtype = indicator_dtype[var]
            ok_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype)
            if ok_dists is not None and DistributionFamily(dist) not in ok_dists:
                errors.append(
                    f"likelihoods[{i}] '{var}': distribution '{dist}' invalid for dtype '{dtype}'; "
                    f"expected one of {{{', '.join(sorted(d.value for d in ok_dists))}}}"
                )

    # --- Validate parameters ---
    parameters = data.get("parameters", [])
    if not isinstance(parameters, list):
        errors.append("'parameters' must be a list")
        parameters = []

    for i, param in enumerate(parameters):
        if not isinstance(param, dict):
            errors.append(f"parameters[{i}]: must be a dictionary")
            continue

        name = param.get("name", f"[{i}]")
        role = param.get("role", "")
        constraint = param.get("constraint", "")

        # Enum validation
        if role and role not in valid_roles:
            errors.append(
                f"parameters[{i}] '{name}': role '{role}' invalid; "
                f"must be one of {sorted(valid_roles)}"
            )
        if constraint and constraint not in valid_constraints:
            errors.append(
                f"parameters[{i}] '{name}': constraint '{constraint}' invalid; "
                f"must be one of {sorted(valid_constraints)}"
            )

        # Domain: role <-> constraint compatibility
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
                        f"parameters[{i}] '{name}': constraint '{constraint}' unexpected "
                        "for role 'loading'; expected 'positive' or 'negative'"
                    )
            elif expected is not None and constraint_enum != expected:
                errors.append(
                    f"parameters[{i}] '{name}': constraint '{constraint}' unexpected "
                    f"for role '{role}'; expected '{expected.value}'"
                )
            if role_enum == ParameterRole.INITIAL_STATE_CORRELATION and not name.startswith(
                "cor0_"
            ):
                errors.append(
                    f"parameters[{i}] '{name}': initial_state_correlation parameters "
                    "must use canonical names starting with 'cor0_'"
                )

    if not errors:
        # All checks passed — build the Pydantic model
        try:
            spec = ModelSpec.model_validate(data)
            return spec, []
        except ValidationError as e:
            return None, [f"Unexpected validation error: {e}"]

    return None, errors


# --- Decisions-only schema (LLM outputs only non-deterministic parts) ---


class DistributionChoice(BaseModel):
    """LLM's distribution/link choice for an indicator with ambiguous dtype."""

    variable: str = Field(description="Name of the indicator variable")
    distribution: DistributionFamily = Field(description="Chosen distribution")
    link: LinkFunction = Field(description="Chosen link function")
    reasoning: str = Field(description="Why this distribution/link")


class ModelSpecDecisions(BaseModel):
    """LLM decisions for the non-deterministic parts of the model specification.

    The deterministic parts (parameter enumeration, deterministic distributions/links,
    and loading polarities) are pre-computed from the CausalSpec. The LLM only provides
    the genuine distribution/link decisions that require statistical judgment.
    """

    distribution_choices: list[DistributionChoice] = Field(
        description="Distribution/link choices for indicators with ambiguous dtypes"
    )


def merge_decisions_to_spec(
    resolved_likelihoods: list[dict],
    parameters: list[dict],
    decisions: ModelSpecDecisions,
) -> tuple[ModelSpec | None, list[str]]:
    """Merge pre-computed skeleton with LLM decisions into a full ModelSpec.

    Args:
        resolved_likelihoods: Pre-computed [{variable, distribution, link}]
        parameters: Pre-computed [{name, role, constraint, description}]
        decisions: LLM's decisions

    Returns:
        (ModelSpec or None, list of error messages)
    """
    # Build likelihoods: resolved + LLM choices
    likelihoods = []
    for rl in resolved_likelihoods:
        likelihoods.append(
            {
                "variable": rl["variable"],
                "distribution": rl["distribution"],
                "link": rl["link"],
                "reasoning": f"Deterministic: {rl.get('reasoning', 'dtype has single valid option')}",
            }
        )
    for dc in decisions.distribution_choices:
        likelihoods.append(
            {
                "variable": dc.variable,
                "distribution": dc.distribution.value,
                "link": dc.link.value,
                "reasoning": dc.reasoning,
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
    """Validate a ModelSpecDecisions dict and merge with skeleton.

    Args:
        data: Raw dict to validate as ModelSpecDecisions
        resolved_likelihoods: Pre-computed deterministic likelihoods
        ambiguous_indicators: Indicators that need LLM decisions
        parameters: Pre-computed parameters with deterministic constraints

    Returns:
        (merged ModelSpec or None, list of error messages)
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    # Parse distribution_choices
    dist_choices = data.get("distribution_choices", [])
    if not isinstance(dist_choices, list):
        errors.append("'distribution_choices' must be a list")
        dist_choices = []

    # Check coverage: every ambiguous indicator should have a decision
    decided_vars = {dc.get("variable", "") for dc in dist_choices if isinstance(dc, dict)}
    ambiguous_vars = {ai["variable"] for ai in ambiguous_indicators}
    missing = ambiguous_vars - decided_vars
    for var in sorted(missing):
        errors.append(f"missing distribution_choice for ambiguous indicator '{var}'")

    # Validate distribution_choices entries
    valid_distributions = {e.value for e in DistributionFamily}
    valid_links = {e.value for e in LinkFunction}
    for i, dc in enumerate(dist_choices):
        if not isinstance(dc, dict):
            errors.append(f"distribution_choices[{i}]: must be a dictionary")
            continue
        dist = dc.get("distribution", "")
        link = dc.get("link", "")
        if dist and dist not in valid_distributions:
            errors.append(
                f"distribution_choices[{i}]: distribution '{dist}' invalid; "
                f"must be one of {sorted(valid_distributions)}"
            )
        if link and link not in valid_links:
            errors.append(
                f"distribution_choices[{i}]: link '{link}' invalid; "
                f"must be one of {sorted(valid_links)}"
            )

    if errors:
        return None, errors

    # Parse as ModelSpecDecisions and merge
    try:
        decisions = ModelSpecDecisions.model_validate(data)
    except ValidationError as e:
        return None, [f"Schema validation error: {e}"]

    return merge_decisions_to_spec(resolved_likelihoods, parameters, decisions)
