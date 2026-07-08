"""measurement-structure contracts and tool metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMArtifactContract, ToolContract

IS_INTERACTIVE_CONTEXT = True


class ValidateMeasurementStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_json: str = Field(
        description="The JSON string containing the measurement structure to validate."
    )


MEASUREMENT_STRUCTURE_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_measurement_structure",
        description=("Validate measurement structure JSON and compiler constraints."),
        input_schema=ValidateMeasurementStructureInput,
    ),
]


class MeasurementStructureContract(LLMArtifactContract):
    measurement_structure: MeasurementStructure


class IdentificationReportContract(BaseModel):
    """The positive identification finding.

    Only produced when at least one treatment effect is explicitly
    identifiable. Negative findings remain in ``causal_design.identifiability``.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_name: str
    estimable_treatments: list[str] = Field(min_length=1)
    non_identifiable_treatments: dict[str, Any] = Field(default_factory=dict)
