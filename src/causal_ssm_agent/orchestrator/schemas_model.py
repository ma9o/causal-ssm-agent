"""Model specification schemas for Stage 4 orchestrator.

These schemas define the structure proposed by the orchestrator LLM
for the statistical model specification.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class DistributionFamily(StrEnum):
    """Distribution families for observation and process noise.

    Used throughout the SSM pipeline: from LLM-proposed likelihoods to
    emission function dispatch.  Values are lowercase so they can be
    passed directly as strings to likelihood backends.
    """

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    POISSON = "poisson"
    GAMMA = "gamma"
    BERNOULLI = "bernoulli"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BETA = "beta"
    ORDERED_LOGISTIC = "ordered_logistic"
    CATEGORICAL = "categorical"

    @classmethod
    def _missing_(cls, value: str) -> "DistributionFamily | None":
        """Allow case-insensitive and legacy PascalCase construction.

        The LLM may propose PascalCase names (e.g. "Normal", "NegativeBinomial")
        and data files may still use them.  This maps them to the canonical
        lowercase members.
        """
        _ALIASES: dict[str, str] = {
            "normal": "gaussian",
            "negativebinomial": "negative_binomial",
            "orderedlogistic": "ordered_logistic",
        }
        normalized = value.lower().replace(" ", "_")
        normalized = _ALIASES.get(normalized, normalized)
        for member in cls:
            if member.value == normalized:
                return member
        return None


class LinkFunction(StrEnum):
    """Link functions mapping linear predictor to distribution mean."""

    IDENTITY = "identity"  # Gaussian
    LOG = "log"  # Poisson, Gamma, NegativeBinomial
    LOGIT = "logit"  # Bernoulli, Beta
    PROBIT = "probit"  # Bernoulli
    CUMULATIVE_LOGIT = "cumulative_logit"  # OrderedLogistic
    SOFTMAX = "softmax"  # Categorical


class ParameterRole(StrEnum):
    """Role of a parameter in the model."""

    FIXED_EFFECT = "fixed_effect"  # Beta coefficients for causal effects
    AR_COEFFICIENT = "ar_coefficient"  # Rho for autoregressive terms
    RESIDUAL_SD = "residual_sd"  # Sigma for residual variance
    CORRELATION = "correlation"  # Correlation between constructs
    LOADING = "loading"  # Factor loading for multi-indicator constructs


class ParameterConstraint(StrEnum):
    """Constraints on parameter values."""

    NONE = "none"  # Unconstrained (can be any real number)
    POSITIVE = "positive"  # Must be > 0 (variances, SDs)
    UNIT_INTERVAL = "unit_interval"  # Must be in [0, 1] (probabilities, AR coefficients)
    CORRELATION = "correlation"  # Must be in [-1, 1]


VALID_LIKELIHOODS_FOR_DTYPE: dict[str, set[DistributionFamily]] = {
    "binary": {DistributionFamily.BERNOULLI},
    "count": {DistributionFamily.POISSON, DistributionFamily.NEGATIVE_BINOMIAL},
    "continuous": {
        DistributionFamily.GAUSSIAN,
        DistributionFamily.STUDENT_T,
        DistributionFamily.GAMMA,
        DistributionFamily.BETA,
    },
    "ordinal": {DistributionFamily.ORDERED_LOGISTIC},
    "categorical": {DistributionFamily.CATEGORICAL, DistributionFamily.ORDERED_LOGISTIC},
}

VALID_LINKS_FOR_DISTRIBUTION: dict[DistributionFamily, set[LinkFunction]] = {
    DistributionFamily.BERNOULLI: {LinkFunction.LOGIT, LinkFunction.PROBIT},
    DistributionFamily.POISSON: {LinkFunction.LOG},
    DistributionFamily.NEGATIVE_BINOMIAL: {LinkFunction.LOG},
    DistributionFamily.GAUSSIAN: {LinkFunction.IDENTITY},
    DistributionFamily.STUDENT_T: {LinkFunction.IDENTITY},
    DistributionFamily.GAMMA: {LinkFunction.LOG},
    DistributionFamily.BETA: {LinkFunction.LOGIT},
    DistributionFamily.ORDERED_LOGISTIC: {LinkFunction.CUMULATIVE_LOGIT},
    DistributionFamily.CATEGORICAL: {LinkFunction.SOFTMAX},
}

EXPECTED_CONSTRAINT_FOR_ROLE: dict[ParameterRole, ParameterConstraint] = {
    ParameterRole.AR_COEFFICIENT: ParameterConstraint.UNIT_INTERVAL,
    ParameterRole.RESIDUAL_SD: ParameterConstraint.POSITIVE,
    ParameterRole.FIXED_EFFECT: ParameterConstraint.NONE,
    ParameterRole.LOADING: ParameterConstraint.POSITIVE,
    ParameterRole.CORRELATION: ParameterConstraint.CORRELATION,
}


class LikelihoodSpec(BaseModel):
    """Specification for a likelihood (observed variable distribution)."""

    variable: str = Field(description="Name of the observed indicator variable")
    distribution: DistributionFamily = Field(description="Distribution family for this variable")
    link: LinkFunction = Field(description="Link function mapping linear predictor to mean")
    reasoning: str = Field(description="Why this distribution/link was chosen for this variable")


class ParameterSpec(BaseModel):
    """Specification for a parameter requiring a prior."""

    name: str = Field(description="Parameter name (e.g., 'beta_stress_anxiety', 'rho_mood')")
    role: ParameterRole = Field(description="Role of this parameter in the model")
    constraint: ParameterConstraint = Field(description="Constraint on parameter values")
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )
    search_context: str = Field(
        description="Context for Exa literature search to find relevant effect sizes"
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
    reasoning: str = Field(description="Overall reasoning for the model specification choices")


def validate_model_spec(
    model_spec: ModelSpec,
    indicators: list[dict] | None = None,
) -> list[dict]:
    """Validate domain rules on a ModelSpec.

    Returns list of issues (empty = valid). Each issue:
        {"field": str, "name": str, "issue": str, "severity": "error"|"warning"}

    Checks:
    1. distribution<->link compatibility (always)
    2. role<->constraint compatibility (always)
    3. dtype<->distribution compatibility (when indicators provided)
    """
    issues: list[dict] = []

    # 0a. Duplicate likelihood variables
    lik_vars = [lik.variable for lik in model_spec.likelihoods]
    seen_vars: set[str] = set()
    for var in lik_vars:
        if var in seen_vars:
            issues.append(
                {
                    "field": "likelihoods",
                    "name": var,
                    "issue": f"duplicate likelihood for variable '{var}'",
                    "severity": "error",
                }
            )
        seen_vars.add(var)

    # 0b. Unique parameter names
    param_names = [p.name for p in model_spec.parameters]
    seen_params: set[str] = set()
    for name in param_names:
        if name in seen_params:
            issues.append(
                {
                    "field": "parameters",
                    "name": name,
                    "issue": f"duplicate parameter name '{name}'",
                    "severity": "error",
                }
            )
        seen_params.add(name)

    # 0c. One likelihood per indicator (coverage check, when indicators provided)
    if indicators is not None:
        indicator_names = {ind["name"] for ind in indicators}
        covered = set(lik_vars)
        missing = indicator_names - covered
        for var in sorted(missing):
            issues.append(
                {
                    "field": "likelihoods",
                    "name": var,
                    "issue": f"indicator '{var}' has no likelihood specification",
                    "severity": "warning",
                }
            )

    # 1. distribution <-> link compatibility
    for lik in model_spec.likelihoods:
        valid_links = VALID_LINKS_FOR_DISTRIBUTION.get(lik.distribution)
        if valid_links is not None and lik.link not in valid_links:
            issues.append(
                {
                    "field": "likelihoods",
                    "name": lik.variable,
                    "issue": (
                        f"link '{lik.link}' invalid for {lik.distribution}; "
                        f"expected one of {{{', '.join(sorted(lf.value for lf in valid_links))}}}"
                    ),
                    "severity": "error",
                }
            )

    # 2. role <-> constraint compatibility
    for param in model_spec.parameters:
        expected = EXPECTED_CONSTRAINT_FOR_ROLE.get(param.role)
        if expected is not None and param.constraint != expected:
            issues.append(
                {
                    "field": "parameters",
                    "name": param.name,
                    "issue": (
                        f"constraint '{param.constraint}' unexpected for role '{param.role}'; "
                        f"expected '{expected}'"
                    ),
                    "severity": "warning",
                }
            )

    # 3. dtype <-> distribution compatibility (only when indicators provided)
    if indicators is not None:
        indicator_dtype = {
            ind["name"]: ind.get("measurement_dtype", "continuous") for ind in indicators
        }
        for lik in model_spec.likelihoods:
            dtype = indicator_dtype.get(lik.variable)
            if dtype is not None:
                valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype)
                if valid_dists is not None and lik.distribution not in valid_dists:
                    issues.append(
                        {
                            "field": "likelihoods",
                            "name": lik.variable,
                            "issue": (
                                f"distribution '{lik.distribution}' invalid for dtype '{dtype}'; "
                                f"expected one of {{{', '.join(sorted(d.value for d in valid_dists))}}}"
                            ),
                            "severity": "error",
                        }
                    )

    return issues


def validate_model_spec_dict(
    data: dict,
    indicators: list[dict] | None = None,
) -> tuple["ModelSpec | None", list[str]]:
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
            if expected is not None and constraint_enum != expected:
                errors.append(
                    f"parameters[{i}] '{name}': constraint '{constraint}' unexpected "
                    f"for role '{role}'; expected '{expected.value}'"
                )

    if not errors:
        # All checks passed — build the Pydantic model
        try:
            spec = ModelSpec.model_validate(data)
            return spec, []
        except Exception as e:
            return None, [f"Unexpected validation error: {e}"]

    return None, errors


# Result schemas for the orchestrator stage


class Stage4OrchestratorResult(BaseModel):
    """Result of Stage 4 orchestrator: proposed model specification."""

    model_spec: ModelSpec = Field(description="The proposed model specification")
    raw_response: str = Field(description="Raw LLM response for debugging")
