"""measurement-structure causal-design assembly."""

from nof1_causal_lab.artifacts.causal_design import (
    CausalDesign,
    IdentifiabilityStatus,
    KnownInput,
    ScientificOnlyConstruct,
)
from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
from nof1_causal_lab.json_types import UncheckedJsonObject


def build_causal_design(
    latent_structure: UncheckedJsonObject,
    measurement_structure: UncheckedJsonObject,
    identifiability_status: UncheckedJsonObject | None = None,
    *,
    known_inputs: list[UncheckedJsonObject],
    scientific_only_constructs: list[UncheckedJsonObject],
) -> CausalDesign:
    """Combine scientific and measurement semantics into a CausalDesign."""
    return CausalDesign(
        latent=LatentStructure.model_validate(latent_structure),
        measurement=MeasurementStructure.model_validate(measurement_structure),
        identifiability=(
            IdentifiabilityStatus.model_validate(identifiability_status)
            if identifiability_status
            else None
        ),
        known_inputs=[KnownInput.model_validate(item) for item in known_inputs],
        scientific_only_constructs=[
            ScientificOnlyConstruct.model_validate(item) for item in scientific_only_constructs
        ],
    )
