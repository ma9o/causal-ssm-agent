"""Stage 1b causal-design assembly."""

from nof1_causal_lab.artifacts.causal_design import (
    CausalDesign,
    EstimationSpec,
    IdentifiabilityStatus,
)
from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure


def build_causal_design(
    latent_structure: dict,
    measurement_structure: dict,
    identifiability_status: dict | None = None,
    known_inputs: list[dict] | None = None,
) -> dict:
    """Combine latent and measurement structures into a full CausalDesign with identifiability."""
    from nof1_causal_lab.utils.estimation_projection import build_estimation_projection

    estimation = build_estimation_projection(
        latent_structure,
        measurement_structure,
        identifiability_status,
        known_inputs=known_inputs,
    )

    causal_design = CausalDesign(
        latent=LatentStructure.model_validate(latent_structure),
        measurement=MeasurementStructure.model_validate(measurement_structure),
        identifiability=(
            IdentifiabilityStatus.model_validate(identifiability_status)
            if identifiability_status
            else None
        ),
        estimation=EstimationSpec.model_validate(estimation) if estimation is not None else None,
    )
    return causal_design.model_dump()
