"""Composite causal-spec artifact models and validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .latent_model import CausalEdge, LatentModel  # noqa: TC001
from .measurement_model import MeasurementModel, validate_measurement_model


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


class InducedDependency(BaseModel):
    """Dependence induced among retained states after marginalizing latent roots."""

    between: tuple[str, str] = Field(
        description="Pair of retained states whose joint dependence is induced"
    )
    kind: Literal["innovation_correlation", "initial_state_correlation"] = Field(
        description="Which covariance block the induced dependence belongs to"
    )
    source_confounders: list[str] = Field(
        default_factory=list,
        description="Marginalized source constructs that induce this dependence",
    )


class EstimationSpec(BaseModel):
    """Deterministic estimation-time projection of the user-facing latent DAG."""

    state_order: list[str] = Field(
        description="Retained latent states in canonical array order for compilation"
    )
    edges: list[CausalEdge] = Field(
        default_factory=list,
        description="Directed estimation graph over retained states",
    )
    induced_dependencies: list[InducedDependency] = Field(
        default_factory=list,
        description="Dependencies induced after marginalizing latent root confounders",
    )


class CausalSpec(BaseModel):
    """Complete causal specification combining latent and measurement models."""

    latent: LatentModel = Field(description="Theoretical causal structure (topological)")
    measurement: MeasurementModel = Field(description="Operationalization into indicators")
    identifiability: IdentifiabilityStatus | None = Field(
        default=None, description="Identifiability status of target causal effects"
    )
    estimation: EstimationSpec | None = Field(
        default=None,
        description="Deterministic estimation-time projection consumed by downstream fitting",
    )

    @model_validator(mode="after")
    def validate_causal_spec(self) -> CausalSpec:
        """Validate measurement model coverage and estimation projection integrity."""
        construct_names = {construct.name for construct in self.latent.constructs}

        for indicator in self.measurement.indicators:
            if indicator.construct_name not in construct_names:
                raise ValueError(
                    f"Indicator '{indicator.name}' references unknown construct '{indicator.construct_name}'"
                )

        estimation = self.estimation
        if estimation is not None:
            if len(estimation.state_order) != len(set(estimation.state_order)):
                raise ValueError("Estimation state_order contains duplicate construct names")

            state_names = set(estimation.state_order)
            unknown_states = state_names - construct_names
            if unknown_states:
                raise ValueError(
                    "Estimation state_order references unknown constructs: "
                    f"{sorted(unknown_states)}"
                )

            for edge in estimation.edges:
                if edge.cause not in state_names or edge.effect not in state_names:
                    raise ValueError(
                        "Estimation edge must reference retained states: "
                        f"{edge.cause!r} -> {edge.effect!r}"
                    )

            for dependency in estimation.induced_dependencies:
                state_1, state_2 = dependency.between
                if state_1 not in state_names or state_2 not in state_names:
                    raise ValueError(
                        f"Induced dependency must reference retained states: {dependency.between!r}"
                    )
                unknown_sources = set(dependency.source_confounders) - construct_names
                if unknown_sources:
                    raise ValueError(
                        "Induced dependency references unknown source confounders: "
                        f"{sorted(unknown_sources)}"
                    )

        return self

    def get_edge_lag_hours(self, edge: CausalEdge) -> float:
        """Compute lag in hours for a causal edge."""
        return self.measurement.model_clock_hours if edge.lagged else 0


def validate_causal_spec(
    latent_data: dict,
    measurement_data: dict,
) -> tuple[CausalSpec | None, list[str]]:
    """Validate both latent and measurement models together."""
    from causal_ssm_agent.utils.estimation_projection import build_estimation_projection

    from .latent_model import validate_latent_model

    latent, latent_errors = validate_latent_model(latent_data)
    if latent is None:
        return None, ["Latent model errors:", *latent_errors]

    measurement, measurement_errors = validate_measurement_model(measurement_data, latent)
    if measurement is None:
        return None, ["Measurement model errors:", *measurement_errors]

    try:
        latent_payload = latent.model_dump(mode="json")
        measurement_payload = measurement.model_dump(mode="json")
        model = CausalSpec(
            latent=latent,
            measurement=measurement,
            estimation=EstimationSpec.model_validate(
                build_estimation_projection(
                    latent_payload,
                    measurement_payload,
                    identifiability_result=None,
                )
            ),
        )
        return model, []
    except (ValidationError, ValueError, TypeError) as exc:
        return None, [f"CausalSpec validation failed: {exc}"]


__all__ = [
    "CausalSpec",
    "EstimationSpec",
    "IdentifiabilityStatus",
    "IdentifiedTreatmentStatus",
    "InducedDependency",
    "NonIdentifiableTreatmentStatus",
    "validate_causal_spec",
]
