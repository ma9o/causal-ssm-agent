"""latent-structure contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.latent_structure import LatentStructure  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMArtifactContract, ToolContract

IS_INTERACTIVE_CONTEXT = True


class ValidateLatentStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_json: str = Field(
        description="The JSON string containing the latent structure to validate."
    )


LATENT_STRUCTURE_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_latent_structure",
        description="Tool for validating latent structure JSON (latent-structure).",
        input_schema=ValidateLatentStructureInput,
    ),
]


class LatentStructureContract(LLMArtifactContract):
    latent_structure: LatentStructure
