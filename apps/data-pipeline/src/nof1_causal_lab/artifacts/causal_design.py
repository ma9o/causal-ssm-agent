"""Composite causal-design artifact models and validation."""

from __future__ import annotations

from typing import Literal, override

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .latent_structure import LatentStructure  # noqa: TC001
from .measurement_structure import MeasurementStructure  # noqa: TC001


class IdentifiedTreatmentStatus(BaseModel):
    """Details on how a treatment effect is identified."""

    method: str = Field(
        description="Identification strategy (e.g., do_calculus, instrumental_variable)"
    )
    estimand: str = Field(description="Closed-form estimand or IV placeholder")
    marginalized_confounders: list[str] = Field(
        default_factory=list,
        description="Unobserved confounders the estimand integrates out",
    )
    instruments: list[str] = Field(
        default_factory=list,
        description="Instrumental variables used (if method=instrumental_variable)",
    )


class NonIdentifiableTreatmentStatus(BaseModel):
    """Context on why a treatment effect is not identifiable."""

    confounders: list[str] = Field(
        default_factory=list,
        description="Unobserved constructs blocking identification",
    )
    notes: str | None = Field(
        default=None,
        description="Optional explanation if confounders cannot be enumerated",
    )


class IdentifiabilityStatus(BaseModel):
    """Status of causal effect identifiability."""

    identifiable_treatments: dict[str, IdentifiedTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatments with identifiable effects and how to estimate them",
    )
    non_identifiable_treatments: dict[str, NonIdentifiableTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatments whose effects are currently not identifiable",
    )


class KnownInput(BaseModel):
    """Authored declaration of an observed transition driver."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    construct_name: str = Field(
        alias="construct",
        description="Construct removed from the latent state vector",
    )
    source_indicator: str = Field(description="Measurement indicator column supplying u(t)")
    scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Positive divisor applied to the source indicator before inference",
    )
    missing_policy: Literal["zero", "forward_fill"] = Field(
        default="zero",
        description="How to fill missing input values on the model time grid",
    )

    @property
    @override
    def construct(self) -> str:
        """Artifact-facing construct name."""
        return self.construct_name


class ScientificOnlyConstruct(BaseModel):
    """Authored exclusion from the executable SSM projection."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    construct_name: str = Field(
        alias="construct",
        description="Measured construct retained for scientific context but not executable state",
    )
    reason: str = Field(
        min_length=1,
        description="Why the construct is excluded from the executable N-of-1 model",
    )

    @property
    @override
    def construct(self) -> str:
        """Artifact-facing construct name."""
        return self.construct_name


class CausalDesign(BaseModel):
    """Scientific causal design before executable structural compilation."""

    latent: LatentStructure = Field(description="Theoretical causal structure (topological)")
    measurement: MeasurementStructure = Field(description="Operationalization into indicators")
    identifiability: IdentifiabilityStatus | None = Field(
        default=None, description="Identifiability status of target causal effects"
    )
    known_inputs: list[KnownInput] = Field(
        default_factory=list,
        description="Authored observed-input declarations compiled by StructuralPlan",
    )
    scientific_only_constructs: list[ScientificOnlyConstruct] = Field(
        default_factory=list,
        description="Measured constructs explicitly excluded from the executable SSM",
    )

    @model_validator(mode="after")
    def validate_causal_design(self) -> CausalDesign:
        """Validate authored measurement and executable-disposition declarations."""
        construct_names = {construct.name for construct in self.latent.constructs}

        for indicator in self.measurement.indicators:
            if indicator.construct_name not in construct_names:
                raise ValueError(
                    f"Indicator '{indicator.name}' references unknown construct '{indicator.construct_name}'"
                )

        indicator_lookup = {indicator.name: indicator for indicator in self.measurement.indicators}
        known_input_names = {known_input.construct for known_input in self.known_inputs}
        if len(known_input_names) != len(self.known_inputs):
            raise ValueError("CausalDesign known_inputs contains duplicate constructs")
        scientific_only_names = {item.construct for item in self.scientific_only_constructs}
        if len(scientific_only_names) != len(self.scientific_only_constructs):
            raise ValueError(
                "CausalDesign scientific_only_constructs contains duplicate constructs"
            )
        overlap = known_input_names & scientific_only_names
        if overlap:
            raise ValueError(
                "CausalDesign constructs cannot be both known inputs and scientific-only: "
                f"{sorted(overlap)}"
            )
        unknown_scientific_only = scientific_only_names - construct_names
        if unknown_scientific_only:
            raise ValueError(
                "CausalDesign scientific_only_constructs reference unknown constructs: "
                f"{sorted(unknown_scientific_only)}"
            )
        observed_construct_names = {
            indicator.construct_name for indicator in self.measurement.indicators
        }
        unmeasured_scientific_only = scientific_only_names - observed_construct_names
        if unmeasured_scientific_only:
            raise ValueError(
                "CausalDesign scientific_only_constructs must have measurement evidence: "
                f"{sorted(unmeasured_scientific_only)}"
            )
        for known_input in self.known_inputs:
            if known_input.construct not in construct_names:
                raise ValueError(
                    "CausalDesign known_input references unknown construct: "
                    f"{known_input.construct!r}"
                )
            source_indicator = indicator_lookup.get(known_input.source_indicator)
            if source_indicator is None:
                raise ValueError(
                    "CausalDesign known_input references unknown source_indicator: "
                    f"{known_input.source_indicator!r}"
                )
            if source_indicator.construct_name != known_input.construct:
                raise ValueError(
                    "CausalDesign known_input source_indicator must measure the same "
                    f"construct: {known_input.source_indicator!r} measures "
                    f"{source_indicator.construct_name!r}, expected "
                    f"{known_input.construct!r}"
                )

        return self


__all__ = [
    "CausalDesign",
    "IdentifiabilityStatus",
    "IdentifiedTreatmentStatus",
    "KnownInput",
    "NonIdentifiableTreatmentStatus",
    "ScientificOnlyConstruct",
]
