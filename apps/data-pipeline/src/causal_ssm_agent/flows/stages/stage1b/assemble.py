"""Stage 1b causal-spec assembly."""

from causal_ssm_agent.artifacts.causal_spec import CausalSpec, EstimationSpec, IdentifiabilityStatus
from causal_ssm_agent.artifacts.latent_model import LatentModel
from causal_ssm_agent.artifacts.measurement_model import MeasurementModel


def build_causal_spec(
    latent_model: dict,
    measurement_model: dict,
    identifiability_status: dict | None = None,
) -> dict:
    """Combine latent and measurement models into a full CausalSpec with identifiability."""
    from causal_ssm_agent.utils.estimation_projection import build_estimation_projection

    estimation = build_estimation_projection(
        latent_model,
        measurement_model,
        identifiability_status,
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
