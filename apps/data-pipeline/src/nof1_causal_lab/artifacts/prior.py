"""Typed executable prior plans and compiler diagnostics."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


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


def prior_params_type(distribution: PriorDistributionFamily) -> type[PriorParamsModel]:
    """Return the strict parameter model owned by a prior family."""
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
    params: dict[str, int | float],
) -> PriorParams:
    """Validate a parameter mapping against its declared prior family."""
    return cast("PriorParams", prior_params_type(distribution).model_validate(params))


class ExecutablePrior(BaseModel):
    """One authoring-scale prior consumed by the statistical-model compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter: str
    distribution: PriorDistributionFamily
    params: PriorParams
    reference_interval_days: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_params_match_distribution(self) -> ExecutablePrior:
        expected_type = prior_params_type(self.distribution)
        if not isinstance(self.params, expected_type):
            raise ValueError(
                f"{self.distribution.value} prior parameters must match {expected_type.__name__}"
            )
        return self


class PriorPlan(BaseModel):
    """Complete typed executable priors for a StatisticalModelSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    priors: dict[str, ExecutablePrior]

    @model_validator(mode="after")
    def validate_parameter_keys(self) -> PriorPlan:
        mismatches = sorted(key for key, prior in self.priors.items() if key != prior.parameter)
        if mismatches:
            raise ValueError(
                f"PriorPlan keys must equal their executable prior parameter names: {mismatches}"
            )
        return self

    def compiler_payloads(self) -> dict[str, UncheckedJsonObject]:
        """Return compiler-owned fields without agent evidence or presentation metadata."""
        return {
            parameter: prior.model_dump(mode="json") for parameter, prior in self.priors.items()
        }


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
    """Comparable summary of one validation pathology."""

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
    code: str = Field(default="unspecified")
    origin: Literal["compile", "prior_predictive"] = "prior_predictive"
    severity: Literal["error", "warning"] = "error"
    issue: str | None = None
    suggested_adjustment: str | None = None
    related_parameters: list[str] = Field(default_factory=list)
    compiled_site_name: str | None = None
    compiled_flat_index: int | None = None
    supporting_codes: list[str] = Field(default_factory=list)
    repair_scope: PriorRepairScope | None = None
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
    ) = None
    bad_sample_sites: list[str] = Field(default_factory=list)
    bad_manifest_names: list[str] = Field(default_factory=list)
    failing_draw_indices: list[int] = Field(default_factory=list)
    first_bad_time_index: int | None = None
    pathology_certificate: PriorPathologyCertificate | None = None


__all__ = [
    "BetaPriorParams",
    "BoundsPriorParams",
    "ExecutablePrior",
    "GammaPriorParams",
    "LocationScalePriorParams",
    "PriorParams",
    "PriorPathologyCertificate",
    "PriorPlan",
    "PriorRepairScope",
    "PriorValidationResult",
    "RatePriorParams",
    "ScalePriorParams",
    "TruncatedNormalPriorParams",
    "ValuePriorParams",
    "prior_params_model",
    "prior_params_type",
]
