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


class IdentificationReportContract(BaseModel):
    """The positive identification finding.

    Only produced when at least one treatment effect is explicitly
    identifiable. Negative findings remain in ``causal_spec.identifiability``.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_name: str
    estimable_treatments: list[str] = Field(min_length=1)
    non_identifiable_treatments: dict[str, Any] = Field(default_factory=dict)
