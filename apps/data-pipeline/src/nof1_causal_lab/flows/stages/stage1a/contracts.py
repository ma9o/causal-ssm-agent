"""Stage 1a contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.latent_model import LatentModel  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMStageContract, ToolContract

STAGE_ID = "stage-1a"
IS_INTERACTIVE_STAGE = True


class ValidateLatentModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_json: str = Field(
        description="The JSON string containing the latent model to validate."
    )


STAGE1A_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_latent_model",
        description="Tool for validating latent model JSON (Stage 1a).",
        input_schema=ValidateLatentModelInput,
    ),
]


class Stage1aContract(LLMStageContract):
    latent_model: LatentModel

    def summary_message(self) -> str:
        return (
            f"Stage 1a summary: constructs={len(self.latent_model.constructs)} "
            f"edges={len(self.latent_model.edges)}"
        )
