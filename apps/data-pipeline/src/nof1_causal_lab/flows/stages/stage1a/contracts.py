"""Stage 1a contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.latent_structure import LatentStructure  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMStageContract, ToolContract

IS_INTERACTIVE_STAGE = True


class ValidateLatentStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_json: str = Field(
        description="The JSON string containing the latent structure to validate."
    )


STAGE1A_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_latent_structure",
        description="Tool for validating latent structure JSON (Stage 1a).",
        input_schema=ValidateLatentStructureInput,
    ),
]


class Stage1aContract(LLMStageContract):
    latent_structure: LatentStructure
