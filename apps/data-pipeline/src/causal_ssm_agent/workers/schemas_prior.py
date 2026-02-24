"""Prior research schemas for Stage 4 workers.

These schemas define the structure for per-parameter prior research
conducted by worker LLMs with Exa literature search.
"""

from pydantic import BaseModel, Field


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
    distribution: str = Field(
        description="Distribution name (e.g., 'Normal', 'HalfNormal', 'Beta', 'Uniform')"
    )
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
            "Used for DT→CT conversion: drift = beta / reference_interval_days."
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


class PriorValidationResult(BaseModel):
    """Result of validating a prior via prior predictive check."""

    parameter: str = Field(description="Name of the parameter that was validated")
    is_valid: bool = Field(description="Whether the prior passed validation")
    issue: str | None = Field(
        default=None, description="Description of the issue if validation failed"
    )
    suggested_adjustment: str | None = Field(
        default=None, description="Suggested fix if validation failed"
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


class PriorResearchResult(BaseModel):
    """Result of researching a single parameter's prior."""

    parameter: str = Field(description="Name of the parameter")
    proposal: PriorProposal = Field(description="The proposed prior distribution")
    literature_found: bool = Field(description="Whether relevant literature was found")
    raw_response: str = Field(description="Raw LLM response for debugging")
    # AutoElicit-style aggregation fields (optional)
    aggregation: AggregatedPrior | None = Field(
        default=None, description="Aggregated prior from paraphrased elicitations (if enabled)"
    )
