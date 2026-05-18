"""Prior research schemas for Stage 4 workers.

These schemas define the structure for per-parameter prior research
conducted by worker LLMs with Exa literature search.
"""

from typing import Literal

from pydantic import BaseModel, Field

from causal_ssm_agent.distributions import (
    PriorDistributionFamily,
    format_prior_distribution_name_list,
)

_QUOTE = "'"
PRIOR_DISTRIBUTION_DESCRIPTION = (
    f"Distribution family ({format_prior_distribution_name_list(quote=_QUOTE)})"
)


class PriorSource(BaseModel):
    """A source of evidence for a prior distribution."""

    title: str = Field(description="Title of the source (paper, meta-analysis, etc.)")
    url: str | None = Field(default=None, description="URL of the source if available")
    snippet: str = Field(description="Relevant excerpt from the source")
    effect_size: str | None = Field(
        default=None, description="Reported effect size if available (e.g., 'r=0.3', 'β=0.2')"
    )
    study_interval_days: float | None = Field(
        default=None,
        description="Observation/measurement interval of this study in days (daily=1, weekly=7, monthly=30)",
    )


class PriorProposal(BaseModel):
    """A proposed prior distribution for a parameter."""

    parameter: str = Field(description="Name of the parameter this prior is for")
    distribution: PriorDistributionFamily = Field(description=PRIOR_DISTRIBUTION_DESCRIPTION)
    params: dict[str, float] = Field(
        description="Distribution parameters (e.g., {'mu': 0.3, 'sigma': 0.1})"
    )
    sources: list[PriorSource] = Field(
        default_factory=list, description="Literature sources supporting this prior"
    )
    reasoning: str = Field(
        description="Justification for the chosen prior distribution and parameters"
    )
    reference_interval_days: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Observation interval (in days) that the DT prior is expressed in. "
            "Sourced from the study's measurement schedule (e.g., 7 for a weekly study). "
            "Used for DT→CT conversion of dynamic priors "
            "(e.g. beta/dt for cross-lags, -log(rho)/dt for baseline persistence)."
        ),
    )
    density_points: list[dict[str, float]] | None = Field(
        default=None,
        description=(
            "Pre-computed density curve points [{x, y}, ...] for frontend visualization. "
            "Computed by the pipeline before persistence so the frontend doesn't need "
            "to approximate the PDF client-side."
        ),
    )


class PriorRepairScope(BaseModel):
    """Deterministic repair scope for nonlocal prior-validation failures."""

    kind: Literal["dynamics_scc"] = Field(
        description="Repair-scope family for a nonlocal validation failure"
    )
    construct_names: list[str] = Field(
        default_factory=list,
        description="Ordered latent constructs included in the minimal repair scope",
    )


class PriorPathologyCertificate(BaseModel):
    """Comparable summary of one validation pathology.

    Lower scores are better. Certificates are used only to decide whether
    retrying the same repair scope is justified.
    """

    kind: Literal["nonfinite_samples", "dynamics_stability", "dt_ct_approximation"] = Field(
        description="Stable certificate family for same-scope retry gating"
    )
    primary_score: float = Field(
        ge=0,
        description="Primary severity score. Lower means the pathology improved.",
    )
    secondary_score: float | None = Field(
        default=None,
        ge=0,
        description="Optional tie-break severity score. Lower means the pathology improved.",
    )


class PriorValidationResult(BaseModel):
    """Typed Stage 4 validation diagnostic."""

    parameter: str = Field(description="Name of the parameter that was validated")
    is_valid: bool = Field(description="Whether the prior passed validation")
    code: str = Field(
        default="unspecified",
        description="Stable machine-readable diagnostic code",
    )
    origin: Literal["compile", "prior_predictive"] = Field(
        default="prior_predictive",
        description="Validation subsystem that emitted this diagnostic",
    )
    severity: Literal["error", "warning"] = Field(
        default="error",
        description="Validation severity. Warnings are non-fatal and do not invalidate the model.",
    )
    issue: str | None = Field(
        default=None, description="Description of the issue if validation failed"
    )
    suggested_adjustment: str | None = Field(
        default=None, description="Suggested fix if validation failed"
    )
    related_parameters: list[str] = Field(
        default_factory=list,
        description="Ordered parameter names most directly implicated by this diagnostic",
    )
    compiled_site_name: str | None = Field(
        default=None,
        description="Compiled runtime sample/prior site implicated by this diagnostic when known",
    )
    compiled_flat_index: int | None = Field(
        default=None,
        description="Flat index inside the compiled site implicated by this diagnostic when known",
    )
    supporting_codes: list[str] = Field(
        default_factory=list,
        description="Codes for supporting diagnostics that help explain this diagnostic",
    )
    repair_scope: PriorRepairScope | None = Field(
        default=None,
        description="Deterministic minimal repair scope for nonlocal failures",
    )
    failure_stage: (
        Literal[
            "compiled_parameters",
            "latent_dynamics",
            "observation_mean",
            "observation_sample",
            "support_violation",
            "model_build",
            "prior_sampling",
            "unknown",
        ]
        | None
    ) = Field(
        default=None,
        description="Earliest validation stage that can currently be localized for this failure",
    )
    bad_sample_sites: list[str] = Field(
        default_factory=list,
        description="Non-finite prior-predictive sample sites implicated by this diagnostic",
    )
    bad_manifest_names: list[str] = Field(
        default_factory=list,
        description="Manifest channels implicated by this diagnostic when they can be localized",
    )
    failing_draw_indices: list[int] = Field(
        default_factory=list,
        description="Prior-predictive draw indices implicated by this diagnostic when known",
    )
    first_bad_time_index: int | None = Field(
        default=None,
        description="Earliest failing time index within the localized prior-predictive draw",
    )
    pathology_certificate: PriorPathologyCertificate | None = Field(
        default=None,
        description="Comparable pathology summary for same-scope retry gating",
    )


class RawPriorSample(BaseModel):
    """A single prior elicitation from one paraphrased prompt."""

    paraphrase_id: int = Field(description="Index of the paraphrase template used (0-indexed)")
    mu: float = Field(description="Elicited mean/location parameter")
    sigma: float = Field(description="Elicited standard deviation/scale parameter")
    reasoning: str = Field(description="Justification for this elicitation")


class AggregatedPrior(BaseModel):
    """Aggregated prior from multiple paraphrased elicitations."""

    method: str = Field(description="Aggregation method used ('simple' or 'gmm')")
    mu: float = Field(description="Aggregated mean/location parameter")
    sigma: float = Field(description="Aggregated standard deviation/scale parameter")
    # GMM-specific fields (only populated when method='gmm')
    mixture_weights: list[float] | None = Field(
        default=None, description="Mixture weights for GMM components"
    )
    mixture_means: list[float] | None = Field(default=None, description="Means of GMM components")
    mixture_stds: list[float] | None = Field(
        default=None, description="Standard deviations of GMM components"
    )
    n_samples: int = Field(description="Number of paraphrase samples aggregated")
