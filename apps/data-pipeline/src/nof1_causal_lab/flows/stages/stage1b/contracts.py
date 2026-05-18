"""Stage 1b contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from causal_ssm_agent.artifacts.causal_spec import CausalSpec  # noqa: TC001
from causal_ssm_agent.flows.contracts_base import LLMStageContract, ToolContract

STAGE_ID = "stage-1b"
IS_INTERACTIVE_STAGE = True


class ValidateMeasurementModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_json: str = Field(
        description="The JSON string containing the measurement model to validate."
    )


STAGE1B_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_measurement_model",
        description=(
            "Validate measurement model JSON, check compiler constraints, "
            "and verify causal identifiability."
        ),
        input_schema=ValidateMeasurementModelInput,
    ),
]


class Stage1bContract(LLMStageContract):
    causal_spec: CausalSpec

    def summary_message(self) -> str:
        non_id = (
            self.causal_spec.identifiability.non_identifiable_treatments
            if self.causal_spec.identifiability
            else {}
        ) or {}
        return (
            f"Stage 1b summary: constructs={len(self.causal_spec.latent.constructs)} "
            f"indicators={len(self.causal_spec.measurement.indicators)} "
            f"filtered_treatments={len(non_id)} outcome={self.outcome}"
        )
