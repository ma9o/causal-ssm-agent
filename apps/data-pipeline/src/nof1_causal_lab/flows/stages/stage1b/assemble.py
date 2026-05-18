"""Stage 1b causal-spec assembly."""

from nof1_causal_lab.artifacts.causal_spec import CausalSpec, EstimationSpec, IdentifiabilityStatus
from nof1_causal_lab.artifacts.latent_model import LatentModel
from nof1_causal_lab.artifacts.measurement_model import MeasurementModel


def build_causal_spec(
    latent_model: dict,
    measurement_model: dict,
    identifiability_status: dict | None = None,
    known_inputs: list[dict] | None = None,
) -> dict:
    """Combine latent and measurement models into a full CausalSpec with identifiability."""
    from nof1_causal_lab.utils.estimation_projection import build_estimation_projection

    estimation = build_estimation_projection(
        latent_model,
        measurement_model,
        identifiability_status,
        known_inputs=known_inputs,
    )

    causal_spec = CausalSpec(
        latent=LatentModel.model_validate(latent_model),
        measurement=MeasurementModel.model_validate(measurement_model),
        identifiability=(
            IdentifiabilityStatus.model_validate(identifiability_status)
            if identifiability_status
            else None
        ),
        estimation=EstimationSpec.model_validate(estimation) if estimation is not None else None,
    )
    return causal_spec.model_dump()
