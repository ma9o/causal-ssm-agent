"""Shared inference-facing contract models used by multiple stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class InferenceMetadataContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    n_samples: int
    duration_seconds: float
