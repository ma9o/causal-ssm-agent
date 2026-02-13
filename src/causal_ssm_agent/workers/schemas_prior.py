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
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this prior (0-1), based on evidence quality"
    )
    reasoning: str = Field(
        description="Justification for the chosen prior distribution and parameters"
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
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this elicitation (0-1)")
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
