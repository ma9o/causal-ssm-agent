"""GLMM specification schemas for Stage 4 orchestrator.

These schemas define the structure proposed by the orchestrator LLM
for the Generalized Linear Mixed Model specification.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DistributionFamily(str, Enum):
    """Likelihood distribution families for observed variables."""

    NORMAL = "Normal"
    GAMMA = "Gamma"
    BERNOULLI = "Bernoulli"
    POISSON = "Poisson"
    NEGATIVE_BINOMIAL = "NegativeBinomial"
    BETA = "Beta"
    ORDERED_LOGISTIC = "OrderedLogistic"
    CATEGORICAL = "Categorical"


class LinkFunction(str, Enum):
    """Link functions mapping linear predictor to distribution mean."""

    IDENTITY = "identity"  # Normal
    LOG = "log"  # Poisson, Gamma, NegativeBinomial
    LOGIT = "logit"  # Bernoulli, Beta
    PROBIT = "probit"  # Bernoulli
    CUMULATIVE_LOGIT = "cumulative_logit"  # OrderedLogistic
    SOFTMAX = "softmax"  # Categorical


class ParameterRole(str, Enum):
    """Role of a parameter in the GLMM."""

    FIXED_EFFECT = "fixed_effect"  # Beta coefficients for causal effects
    AR_COEFFICIENT = "ar_coefficient"  # Rho for autoregressive terms
    RESIDUAL_SD = "residual_sd"  # Sigma for residual variance
    RANDOM_INTERCEPT_SD = "random_intercept_sd"  # SD of random intercepts
    RANDOM_SLOPE_SD = "random_slope_sd"  # SD of random slopes
    CORRELATION = "correlation"  # Correlation between random effects
    LOADING = "loading"  # Factor loading for multi-indicator constructs


class ParameterConstraint(str, Enum):
    """Constraints on parameter values."""

    NONE = "none"  # Unconstrained (can be any real number)
    POSITIVE = "positive"  # Must be > 0 (variances, SDs)
    UNIT_INTERVAL = "unit_interval"  # Must be in [0, 1] (probabilities, AR coefficients)
    CORRELATION = "correlation"  # Must be in [-1, 1]


class LikelihoodSpec(BaseModel):
    """Specification for a likelihood (observed variable distribution)."""

    variable: str = Field(
        description="Name of the observed indicator variable"
    )
    distribution: DistributionFamily = Field(
        description="Distribution family for this variable"
    )
    link: LinkFunction = Field(
        description="Link function mapping linear predictor to mean"
    )
    reasoning: str = Field(
        description="Why this distribution/link was chosen for this variable"
    )


class RandomEffectSpec(BaseModel):
    """Specification for a random effect (hierarchical structure)."""

    grouping: str = Field(
        description="Grouping variable (e.g., 'subject', 'item', 'day')"
    )
    effect_type: str = Field(
        description="Type of effect: 'intercept' or 'slope'"
    )
    applies_to: list[str] = Field(
        description="Which constructs/coefficients have this random effect"
    )
    reasoning: str = Field(
        description="Why this random effect structure is appropriate"
    )


class ParameterSpec(BaseModel):
    """Specification for a parameter requiring a prior."""

    name: str = Field(
        description="Parameter name (e.g., 'beta_stress_anxiety', 'rho_mood')"
    )
    role: ParameterRole = Field(
        description="Role of this parameter in the model"
    )
    constraint: ParameterConstraint = Field(
        description="Constraint on parameter values"
    )
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )
    search_context: str = Field(
        description="Context for Exa literature search to find relevant effect sizes"
    )


class GLMMSpec(BaseModel):
    """Complete GLMM specification from orchestrator.

    This is what the orchestrator proposes based on the DSEMModel structure.
    It enumerates all parameters needing priors and specifies the statistical model.
    """

    likelihoods: list[LikelihoodSpec] = Field(
        description="Likelihood specifications for each observed indicator"
    )
    random_effects: list[RandomEffectSpec] = Field(
        default_factory=list,
        description="Random effect specifications for hierarchical structure"
    )
    parameters: list[ParameterSpec] = Field(
        description="All parameters requiring priors"
    )
    model_clock: str = Field(
        description="Temporal granularity at which the model operates (e.g., 'daily')"
    )
    reasoning: str = Field(
        description="Overall reasoning for the GLMM specification choices"
    )


# Result schemas for the orchestrator stage


class Stage4OrchestratorResult(BaseModel):
    """Result of Stage 4 orchestrator: proposed GLMM specification."""

    glmm_spec: GLMMSpec = Field(
        description="The proposed GLMM specification"
    )
    raw_response: str = Field(
        description="Raw LLM response for debugging"
    )
