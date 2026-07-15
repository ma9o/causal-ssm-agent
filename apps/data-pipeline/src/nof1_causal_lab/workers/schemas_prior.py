"""Prior research schemas for model-spec workers.

These schemas define the structure for per-parameter prior research
conducted by worker LLMs with Exa literature search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    format_prior_distribution_name_list,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_QUOTE = "'"
PRIOR_DISTRIBUTION_DESCRIPTION = (
    f"Distribution family ({format_prior_distribution_name_list(quote=_QUOTE)})"
)


class PriorParamsModel(BaseModel):
    """Strict immutable base for family-specific prior parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LocationScalePriorParams(PriorParamsModel):
    """Location and positive scale parameters for Normal or LogNormal priors."""

    mu: float
    sigma: float = Field(gt=0)


class ScalePriorParams(PriorParamsModel):
    """Positive scale parameter for HalfNormal priors."""

    sigma: float = Field(gt=0)


class BoundsPriorParams(PriorParamsModel):
    """Finite ordered bounds for Uniform priors."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_ordered_bounds(self) -> BoundsPriorParams:
        if self.lower >= self.upper:
            raise ValueError("Uniform prior requires lower < upper")
        return self


class TruncatedNormalPriorParams(LocationScalePriorParams):
    """Location, scale, and finite ordered bounds for TruncatedNormal priors."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_ordered_bounds(self) -> TruncatedNormalPriorParams:
        if self.lower >= self.upper:
            raise ValueError("TruncatedNormal prior requires lower < upper")
        return self


class GammaPriorParams(PriorParamsModel):
    """Positive shape and rate for Gamma priors."""

    concentration: float = Field(gt=0)
    rate: float = Field(gt=0)


class RatePriorParams(PriorParamsModel):
    """Positive rate for Exponential priors."""

    rate: float = Field(gt=0)


class BetaPriorParams(PriorParamsModel):
    """Positive shape parameters for Beta priors."""

    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class ValuePriorParams(PriorParamsModel):
    """Point value for Delta priors."""

    value: float


type PriorParams = (
    LocationScalePriorParams
    | ScalePriorParams
    | BoundsPriorParams
    | TruncatedNormalPriorParams
    | GammaPriorParams
    | RatePriorParams
    | BetaPriorParams
    | ValuePriorParams
)


def _prior_params_type(distribution: PriorDistributionFamily) -> type[PriorParamsModel]:
    if distribution in {
        PriorDistributionFamily.NORMAL,
        PriorDistributionFamily.LOG_NORMAL,
    }:
        return LocationScalePriorParams
    if distribution == PriorDistributionFamily.HALF_NORMAL:
        return ScalePriorParams
    if distribution == PriorDistributionFamily.UNIFORM:
        return BoundsPriorParams
    if distribution == PriorDistributionFamily.TRUNCATED_NORMAL:
        return TruncatedNormalPriorParams
    if distribution == PriorDistributionFamily.GAMMA:
        return GammaPriorParams
    if distribution == PriorDistributionFamily.EXPONENTIAL:
        return RatePriorParams
    if distribution == PriorDistributionFamily.BETA:
        return BetaPriorParams
    return ValuePriorParams


def prior_params_model(
    distribution: PriorDistributionFamily,
    params: Mapping[str, int | float],
) -> PriorParams:
    """Validate a parameter mapping against its declared prior family."""
    return cast("PriorParams", _prior_params_type(distribution).model_validate(params))


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
    params: PriorParams = Field(
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
        expected_type = _prior_params_type(self.distribution)

        if not isinstance(self.params, expected_type):
            raise ValueError(
                f"{self.distribution.value} prior parameters must match {expected_type.__name__}"
            )
        return self


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
    """Typed model-spec validation diagnostic."""

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
