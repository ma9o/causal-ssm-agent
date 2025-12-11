"""Schemas for worker LLM outputs."""

from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """A single extracted observation for a dimension."""

    dimension: str = Field(description="Name of the dimension")
    value: str = Field(description="Extracted value/observation")
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
