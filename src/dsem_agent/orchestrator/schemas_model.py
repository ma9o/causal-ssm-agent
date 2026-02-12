"""Model specification schemas for Stage 4 orchestrator.

These schemas define the structure proposed by the orchestrator LLM
for the statistical model specification.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class DistributionFamily(StrEnum):
    """Likelihood distribution families for observed variables."""

    NORMAL = "Normal"
    GAMMA = "Gamma"
    BERNOULLI = "Bernoulli"
    POISSON = "Poisson"
    NEGATIVE_BINOMIAL = "NegativeBinomial"
    BETA = "Beta"
    ORDERED_LOGISTIC = "OrderedLogistic"
    CATEGORICAL = "Categorical"


class LinkFunction(StrEnum):
    """Link functions mapping linear predictor to distribution mean."""

    IDENTITY = "identity"  # Normal
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
    RANDOM_INTERCEPT_SD = "random_intercept_sd"  # SD of random intercepts
    RANDOM_SLOPE_SD = "random_slope_sd"  # SD of random slopes
    CORRELATION = "correlation"  # Correlation between random effects
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
    "continuous": {DistributionFamily.NORMAL, DistributionFamily.GAMMA, DistributionFamily.BETA},
    "ordinal": {DistributionFamily.ORDERED_LOGISTIC},
    "categorical": {DistributionFamily.CATEGORICAL, DistributionFamily.ORDERED_LOGISTIC},
}

VALID_LINKS_FOR_DISTRIBUTION: dict[DistributionFamily, set[LinkFunction]] = {
    DistributionFamily.BERNOULLI: {LinkFunction.LOGIT, LinkFunction.PROBIT},
    DistributionFamily.POISSON: {LinkFunction.LOG},
    DistributionFamily.NEGATIVE_BINOMIAL: {LinkFunction.LOG},
    DistributionFamily.NORMAL: {LinkFunction.IDENTITY},
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
    ParameterRole.RANDOM_INTERCEPT_SD: ParameterConstraint.POSITIVE,
    ParameterRole.RANDOM_SLOPE_SD: ParameterConstraint.POSITIVE,
    ParameterRole.CORRELATION: ParameterConstraint.CORRELATION,
}


class LikelihoodSpec(BaseModel):
    """Specification for a likelihood (observed variable distribution)."""

    variable: str = Field(description="Name of the observed indicator variable")
    distribution: DistributionFamily = Field(description="Distribution family for this variable")
    link: LinkFunction = Field(description="Link function mapping linear predictor to mean")
    reasoning: str = Field(description="Why this distribution/link was chosen for this variable")


class RandomEffectSpec(BaseModel):
    """Specification for a random effect (hierarchical structure)."""

    grouping: str = Field(description="Grouping variable (e.g., 'subject', 'item', 'day')")
    effect_type: str = Field(description="Type of effect: 'intercept' or 'slope'")
    applies_to: list[str] = Field(
        description="Which constructs/coefficients have this random effect"
    )
    reasoning: str = Field(description="Why this random effect structure is appropriate")


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
    random_effects: list[RandomEffectSpec] = Field(
        default_factory=list, description="Random effect specifications for hierarchical structure"
    )
    parameters: list[ParameterSpec] = Field(description="All parameters requiring priors")
    model_clock: str = Field(
        description="Temporal granularity at which the model operates (e.g., 'daily')"
    )
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


# Result schemas for the orchestrator stage


class Stage4OrchestratorResult(BaseModel):
    """Result of Stage 4 orchestrator: proposed model specification."""

    model_spec: ModelSpec = Field(description="The proposed model specification")
    raw_response: str = Field(description="Raw LLM response for debugging")
