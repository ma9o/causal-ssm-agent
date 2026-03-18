"""Model specification schemas for Stage 4 orchestrator.

These schemas define the structure proposed by the orchestrator LLM
for the statistical model specification.
"""

from __future__ import annotations

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

    @property
    def is_discrete(self) -> bool:
        """Whether this family has discrete (integer) support."""
        return self in {
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }

    @property
    def support_interior_point(self) -> float:
        """A scalar strictly inside this family's support (for dummy observations).

        For discrete families this returns 0.0; callers should use alternating
        0/1 values instead (discrete models need at least two distinct levels
        during tracing).
        """
        if self == DistributionFamily.GAMMA:
            return 1.0
        if self == DistributionFamily.BETA:
            return 0.5
        return 0.0

    @classmethod
    def _missing_(cls, value: object) -> DistributionFamily | None:
        """Allow case-insensitive and legacy PascalCase construction.

        The LLM may propose PascalCase names (e.g. "Normal", "NegativeBinomial")
        and data files may still use them.  This maps them to the canonical
        lowercase members.
        """
        if not isinstance(value, str):
            return None
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
    RANDOM_INTERCEPT_SD = "random_intercept_sd"  # Between-person SD for random intercepts
    CORRELATION = "correlation"  # Correlation between constructs
    LOADING = "loading"  # Factor loading for multi-indicator constructs


class ParameterConstraint(StrEnum):
    """Constraints on parameter values."""

    NONE = "none"  # Unconstrained (can be any real number)
    POSITIVE = "positive"  # Must be > 0 (variances, SDs)
    UNIT_INTERVAL = "unit_interval"  # Must be in [0, 1] (probabilities)
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
    DistributionFamily.GAMMA: {LinkFunction.LOG, LinkFunction.INVERSE},
    DistributionFamily.BETA: {LinkFunction.LOGIT, LinkFunction.PROBIT},
    DistributionFamily.ORDERED_LOGISTIC: {LinkFunction.CUMULATIVE_LOGIT},
    DistributionFamily.CATEGORICAL: {LinkFunction.SOFTMAX},
}

EXPECTED_CONSTRAINT_FOR_ROLE: dict[ParameterRole, ParameterConstraint] = {
    ParameterRole.AR_COEFFICIENT: ParameterConstraint.UNIT_INTERVAL,
    ParameterRole.RESIDUAL_SD: ParameterConstraint.POSITIVE,
    ParameterRole.FIXED_EFFECT: ParameterConstraint.NONE,
    ParameterRole.CORRELATION: ParameterConstraint.CORRELATION,
    ParameterRole.RANDOM_INTERCEPT_SD: ParameterConstraint.POSITIVE,
}


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


# --- Decisions-only schema (LLM outputs only non-deterministic parts) ---


class DistributionChoice(BaseModel):
    """LLM's distribution/link choice for an indicator with ambiguous dtype."""

    variable: str = Field(description="Name of the indicator variable")
    distribution: DistributionFamily = Field(description="Chosen distribution")
    link: LinkFunction = Field(description="Chosen link function")
    reasoning: str = Field(description="Why this distribution/link")


class LoadingConstraintChoice(BaseModel):
    """LLM's decision on whether a loading should be positive or unconstrained."""

    parameter: str = Field(description="Loading parameter name")
    constraint: ParameterConstraint = Field(description="positive or none")
    reasoning: str = Field(description="Why this constraint")


class ModelSpecDecisions(BaseModel):
    """LLM decisions for the non-deterministic parts of the model specification.

    The deterministic parts (parameter enumeration, deterministic distributions/links,
    role-based constraints) are pre-computed from the CausalSpec. The LLM only provides
    the genuine decisions that require statistical judgment.
    """

    distribution_choices: list[DistributionChoice] = Field(
        description="Distribution/link choices for indicators with ambiguous dtypes"
    )
    loading_constraints: list[LoadingConstraintChoice] = Field(
        default_factory=list,
        description="Constraint decisions for loading parameters",
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
    errors: list[str] = []

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

    # Apply loading constraint decisions
    loading_overrides = {lc.parameter: lc.constraint.value for lc in decisions.loading_constraints}
    final_params = []
    for p in parameters:
        param = dict(p)
        if param["role"] == "loading" and param["name"] in loading_overrides:
            param["constraint"] = loading_overrides[param["name"]]
        final_params.append(param)

    if errors:
        return None, errors

    spec_dict = {
        "likelihoods": likelihoods,
        "parameters": final_params,
    }
    return validate_model_spec_dict(spec_dict)


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
        parameters: Pre-computed parameters (with loading constraints TBD)

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

    # Validate loading_constraints entries
    loading_constraints = data.get("loading_constraints", [])
    if not isinstance(loading_constraints, list):
        errors.append("'loading_constraints' must be a list")
        loading_constraints = []

    valid_constraints = {e.value for e in ParameterConstraint}
    for i, lc in enumerate(loading_constraints):
        if not isinstance(lc, dict):
            errors.append(f"loading_constraints[{i}]: must be a dictionary")
            continue
        c = lc.get("constraint", "")
        if c and c not in valid_constraints:
            errors.append(
                f"loading_constraints[{i}]: constraint '{c}' invalid; "
                f"must be one of {sorted(valid_constraints)}"
            )

    if errors:
        return None, errors

    # Parse as ModelSpecDecisions and merge
    try:
        decisions = ModelSpecDecisions.model_validate(data)
    except Exception as e:
        return None, [f"Schema validation error: {e}"]

    return merge_decisions_to_spec(resolved_likelihoods, parameters, decisions)
