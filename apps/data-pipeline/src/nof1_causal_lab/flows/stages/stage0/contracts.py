"""Stage 0 contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import LLMStageContract, ToolContract

IS_INTERACTIVE_STAGE = False


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Relative path within the input directory.")


class ReadFileSampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Relative path to the file within the input directory.")
    n_lines: int = Field(default=50, description="Number of lines to read.")


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Python code to execute.")


class SubmitTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_descriptions_json: str = Field(
        description="JSON object mapping column names to descriptions."
    )


STAGE0_TOOL_CONTRACTS: list[ToolContract] = [
    ToolContract(
        name="list_files",
        description="List files in the prepared input directory.",
        input_schema=ListFilesInput,
    ),
    ToolContract(
        name="read_file_sample",
        description="Read a sample of lines from a file to understand its format.",
        input_schema=ReadFileSampleInput,
    ),
    ToolContract(
        name="execute_python",
        description="Execute Python code in a Modal sandbox to parse files into a Polars DataFrame.",
        input_schema=ExecutePythonInput,
    ),
    ToolContract(
        name="submit_table",
        description="Validate and finalize the ingested DataFrame with column descriptions.",
        input_schema=SubmitTableInput,
    ),
]


class Stage0ColumnDescriptionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class Stage0Contract(LLMStageContract):
    column_descriptions: list[Stage0ColumnDescriptionContract]

    def summary_message(self) -> str:
        return f"Stage 0 summary: described_columns={len(self.column_descriptions)}"
