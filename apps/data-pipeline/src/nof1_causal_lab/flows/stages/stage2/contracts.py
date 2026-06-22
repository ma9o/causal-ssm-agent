"""Stage 2 contracts and tool metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import LLMStageContract, ToolContract

IS_INTERACTIVE_STAGE = False


class ValidateExtractionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_json: str = Field(
        description="The JSON string containing the worker output to validate."
    )


STAGE2_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="validate_extractions",
        description="Tool for validating worker extraction output JSON.",
        input_schema=ValidateExtractionsInput,
    ),
]


class WorkerStatusContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    status: Literal["pending", "running", "completed", "failed"]
    n_extractions: int
    n_windows: int
    error: str | None = None


class Stage2Contract(LLMStageContract):
    workers: list[WorkerStatusContract]

    def summary_message(self) -> str:
        completed = sum(1 for worker in self.workers if worker.status == "completed")
        failed = sum(1 for worker in self.workers if worker.status == "failed")
        return (
            f"Stage 2 summary: workers={len(self.workers)} completed={completed} "
            f"failed={failed} outcome={self.outcome}"
        )
