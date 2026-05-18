"""Composite causal-spec artifact models and validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class KnownInput(BaseModel):
    """Observed input trajectory used as a deterministic transition driver."""

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
    def construct(self) -> str:
        """Artifact-facing construct name."""
        return self.construct_name


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
    known_inputs: list[KnownInput] = Field(
        default_factory=list,
        description="Observed construct trajectories compiled as B u(t) transition inputs",
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
            known_input_names = {known_input.construct for known_input in estimation.known_inputs}
            if len(known_input_names) != len(estimation.known_inputs):
                raise ValueError("Estimation known_inputs contains duplicate constructs")
            overlapping_inputs = sorted(state_names & known_input_names)
            if overlapping_inputs:
                raise ValueError(
                    "Known inputs cannot also be retained latent states: "
                    f"{overlapping_inputs}"
                )

            unknown_states = state_names - construct_names
            if unknown_states:
                raise ValueError(
                    "Estimation state_order references unknown constructs: "
                    f"{sorted(unknown_states)}"
                )

            indicator_lookup = {
                indicator.name: indicator for indicator in self.measurement.indicators
            }
            for known_input in estimation.known_inputs:
                if known_input.construct not in construct_names:
                    raise ValueError(
                        "Estimation known_input references unknown construct: "
                        f"{known_input.construct!r}"
                    )
                source_indicator = indicator_lookup.get(known_input.source_indicator)
                if source_indicator is None:
                    raise ValueError(
                        "Estimation known_input references unknown source_indicator: "
                        f"{known_input.source_indicator!r}"
                    )
                if source_indicator.construct_name != known_input.construct:
                    raise ValueError(
                        "Estimation known_input source_indicator must measure the same "
                        f"construct: {known_input.source_indicator!r} measures "
                        f"{source_indicator.construct_name!r}, expected "
                        f"{known_input.construct!r}"
                    )

            for edge in estimation.edges:
                if edge.effect not in state_names or edge.cause not in (
                    state_names | known_input_names
                ):
                    raise ValueError(
                        "Estimation edge must point into a retained state and originate "
                        "from either a retained state or known input: "
                        f"{edge.cause!r} -> {edge.effect!r}"
                    )

            kind_by_confounder: dict[str, str] = {}
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
                for confounder in dependency.source_confounders:
                    prior_kind = kind_by_confounder.setdefault(confounder, dependency.kind)
                    if prior_kind != dependency.kind:
                        raise ValueError(
                            f"Confounder {confounder!r} induces dependencies with "
                            f"inconsistent kinds ({prior_kind!r} and "
                            f"{dependency.kind!r}); a marginalized confounder must "
                            "project to exactly one covariance block."
                        )

        return self

    def get_edge_lag_hours(self, edge: CausalEdge) -> float:
        """Compute lag in hours for a causal edge."""
        return self.measurement.model_clock_hours if edge.lagged else 0


def validate_causal_spec(
    latent_data: dict,
    measurement_data: dict,
    known_inputs: list[dict] | None = None,
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
                    known_inputs=known_inputs,
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
    "KnownInput",
    "NonIdentifiableTreatmentStatus",
    "validate_causal_spec",
]
