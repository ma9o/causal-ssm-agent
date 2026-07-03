"""Stage 1b contracts and tool metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.causal_spec import CausalSpec  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMStageContract, ToolContract

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
            f"filtered_treatments={len(non_id)}"
        )


class IdentificationReportContract(BaseModel):
    """The identification finding — always produced, positive or negative.

    A negative finding (``estimable_treatments`` empty) is a scientifically
    meaningful result the navigator reads to decide whether to revise the
    DAG or the question; it is never an execution failure.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_name: str
    estimable_treatments: list[str]
    non_identifiable_treatments: dict[str, Any] = Field(default_factory=dict)


class EstimandsContract(BaseModel):
    """The enabling artifact for fitting and interventions.

    Only produced when at least one treatment effect is identifiable —
    the epistemic gate is this artifact's existence, not a status flag.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    treatments: list[str]
