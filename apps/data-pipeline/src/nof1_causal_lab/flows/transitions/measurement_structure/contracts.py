"""measurement-structure contracts and tool metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.causal_design import KnownInput  # noqa: TC001
from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import BaseArtifactContract, ToolContract


class ValidateMeasurementStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_json: str = Field(
        description=(
            "The JSON string containing the measurement structure and known-input "
            "declarations to validate."
        )
    )


MEASUREMENT_STRUCTURE_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_measurement_structure",
        description=(
            "Validate measurement structure, known-input declarations, and compiler constraints."
        ),
        input_schema=ValidateMeasurementStructureInput,
    ),
]


class MeasurementStructureContract(BaseArtifactContract):
    measurement_structure: MeasurementStructure
    known_inputs: list[KnownInput] = Field(
        description=(
            "Authored declarations of observed construct trajectories compiled as "
            "transition inputs rather than latent states"
        )
    )


class IdentificationReportContract(BaseModel):
    """The positive identification finding.

    Only produced when at least one treatment effect is explicitly
    identifiable. Negative findings remain in ``causal_design.identifiability``.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_name: str
    estimable_treatments: list[str] = Field(min_length=1)
    non_identifiable_treatments: dict[str, Any] = Field(default_factory=dict)
