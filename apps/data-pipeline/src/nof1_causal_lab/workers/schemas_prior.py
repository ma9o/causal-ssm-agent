"""Prior research schemas for model-spec workers.

These schemas define the structure for per-parameter prior research
conducted by worker LLMs with Exa literature search.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts import prior as prior_contracts
from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    format_prior_distribution_name_list,
)

_QUOTE = "'"
PRIOR_DISTRIBUTION_DESCRIPTION = (
    f"Distribution family ({format_prior_distribution_name_list(quote=_QUOTE)})"
)


class DensityPoint(BaseModel):
    """One evaluated point on a prior density curve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float = Field(ge=0)


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
    params: prior_contracts.PriorParams = Field(
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
    density_points: list[DensityPoint] | None = Field(
        default=None,
        description=(
            "Pre-computed density curve points [{x, y}, ...] for frontend visualization. "
            "Computed by the pipeline before persistence so the frontend doesn't need "
            "to approximate the PDF client-side."
        ),
    )

    @model_validator(mode="after")
    def validate_params_match_distribution(self) -> PriorProposal:
        expected_type = prior_contracts.prior_params_type(self.distribution)

        if not isinstance(self.params, expected_type):
            raise ValueError(
                f"{self.distribution.value} prior parameters must match {expected_type.__name__}"
            )
        return self
