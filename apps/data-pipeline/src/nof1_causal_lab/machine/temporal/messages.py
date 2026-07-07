"""Message types crossing the workflow/activity/facade boundaries.

Kept free of heavy imports (only pydantic + the pure machine modules) so
the workflow sandbox can import this module without dragging in storage,
polars, or jax.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (pydantic field annotations)
    ArtifactId,
    ArtifactVersionInfo,
    EpisodeState,
    Provenance,
)
from nof1_causal_lab.machine.moves import (
    ArtifactStatus,
    ExecOptions,
    Move,
    RetractedArtifact,
)


class EpisodeInit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Rehydration seed: reconstructed from the on-disk episode journal so a
    # workflow (re)started after Temporal lost its in-memory history resumes
    # with the artifacts already produced, instead of re-running from stage-0.
    # Empty/0 for a genuinely new episode; ignored when attaching to a live
    # workflow (USE_EXISTING).
    initial_state: EpisodeState | None = None
    initial_seq: int = 0


class MoveRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    move: Move
    payload: dict[str, Any] | None = None  # write moves
    options: ExecOptions = Field(default_factory=ExecOptions)  # run moves


class MoveOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    status: str  # applied | rejected | raised
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state: EpisodeState


class EpisodeStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    artifacts: list[ArtifactStatus]
    legal: list[Move]


class RunArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    state: EpisodeState
    options: ExecOptions = Field(default_factory=ExecOptions)


class WriteArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    payload: dict[str, Any]
    provenance: Provenance
    state: EpisodeState


class JournalInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    move: Move
    status: str
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state_after: EpisodeState
