"""Schemas for worker LLM outputs."""

import polars as pl
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """A single extracted observation for a dimension."""

    dimension: str = Field(description="Name of the dimension")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp if identifiable",
    )


class ProposedDimension(BaseModel):
    """A suggested new dimension found in local data."""

    name: str = Field(description="Variable name")
    description: str = Field(description="What this variable represents")
    evidence: str = Field(description="What was seen in this chunk")
    relevant_because: str = Field(description="How it connects to the causal question")
    not_already_in_dimensions_because: str = Field(
        description="Why it needs to be added and why existing dimensions don't capture it"
    )


class WorkerOutput(BaseModel):
    """Complete output from a worker processing a single chunk."""

    extractions: list[Extraction] = Field(
        default_factory=list,
        description="Extracted observations for dimensions",
    )
    proposed_dimensions: list[ProposedDimension] | None = Field(
        default=None,
        description="Suggested new dimensions if something important is missing",
    )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert extractions to a Polars DataFrame.

        Returns:
            DataFrame with columns: dimension, value, timestamp
            Value column uses pl.Object to preserve mixed types.
        """
        if not self.extractions:
            return pl.DataFrame(
                schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8}
            )

        return pl.DataFrame(
            [
                {
                    "dimension": e.dimension,
                    "value": e.value,
                    "timestamp": e.timestamp,
                }
                for e in self.extractions
            ],
            schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )
