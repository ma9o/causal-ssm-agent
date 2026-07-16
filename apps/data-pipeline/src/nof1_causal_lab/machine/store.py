"""Append-only versioned artifact store and transition log.

Ledger layout under ``data/{workspace_id}/``::

    store/{artifact_id}/v{N}/          one immutable version
        meta.json                      ArtifactVersionInfo dump
        <payload files>                artifact-specific (json/parquet/pkl)
    episode/journal/{seq:06d}.json     one transition record per file
    episode/traces/{seq:06d}/          LLM traces promoted at commit time
        {subroutine_id}.json

Versions and transition entries are never overwritten. Current artifact state
is derived by replaying the produced and retracted versions of applied
transitions, so there is no written latest-state manifest. One JSON file per
transition entry because the storage backends (local fs, R2) have no atomic
append; sequence numbers are assigned by the workflow, which serializes moves
per episode.

Artifact provenance and trace references are closed within the ledger. LLM
traces are copied out of the run's scratch dir into ``episode/traces/`` by the
journal activity before the record file is written. Records carry only the
subroutine IDs discovered in that run — the trace locations are derived.
A raised transition may carry one typed ``resume`` pointer into scratch. It is
control state, not artifact provenance, and the collector keeps its run
reachable until a later transition supersedes it.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.machine.artifacts import (
    ArtifactId,
    ArtifactVersionInfo,
    EpisodeState,
    Provenance,
)
from nof1_causal_lab.machine.moves import Move, RetractedArtifact, apply_transition
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    import polars as pl


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Versioned payload storage for one workspace."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        # Attribute access keeps the storage tier explicit and lets tests
        # replace the data-module root once for every consumer.
        self._root = data_module.store_dir(workspace_id)

    # -- paths ---------------------------------------------------------------

    def artifact_dir(self, artifact_id: ArtifactId) -> str:
        return storage.join(self._root, artifact_id)

    def version_dir(self, artifact_id: ArtifactId, version: int) -> str:
        return storage.join(self.artifact_dir(artifact_id), f"v{version}")

    def file_path(self, artifact_id: ArtifactId, version: int, name: str) -> str:
        return storage.join(self.version_dir(artifact_id, version), name)

    # -- version listing -----------------------------------------------------

    def list_versions(self, artifact_id: ArtifactId) -> list[int]:
        directory = self.artifact_dir(artifact_id)
        if not storage.exists(directory):
            return []
        versions = []
        for entry in storage.listdir(directory):
            leaf = entry.rstrip("/").rsplit("/", 1)[-1]
            if leaf.startswith("v") and leaf[1:].isdigit():
                versions.append(int(leaf[1:]))
        return sorted(versions)

    def next_version(self, artifact_id: ArtifactId) -> int:
        versions = self.list_versions(artifact_id)
        return (versions[-1] + 1) if versions else 1

    # -- write ---------------------------------------------------------------

    def write_version(
        self,
        artifact_id: ArtifactId,
        *,
        provenance: Provenance,
        derived_from: dict[ArtifactId, int],
        produced_by: str | None,
        json_files: dict[str, Any] | None = None,
        parquet_files: dict[str, pl.DataFrame] | None = None,
        pickle_files: dict[str, Any] | None = None,
    ) -> ArtifactVersionInfo:
        """Persist one immutable artifact version and return its stamp."""
        version = self.next_version(artifact_id)
        directory = self.version_dir(artifact_id, version)
        storage.makedirs(directory)

        for name, value in (json_files or {}).items():
            storage.write_text(storage.join(directory, name), json.dumps(value))
        for name, df in (parquet_files or {}).items():
            path = storage.join(directory, name)
            if storage.is_remote():
                with storage.get_fs().open(path, "wb") as f:
                    df.write_parquet(f)
            else:
                df.write_parquet(path)
        for name, value in (pickle_files or {}).items():
            with storage.open_file(storage.join(directory, name), "wb") as f:
                cloudpickle.dump(value, f)

        info = ArtifactVersionInfo(
            artifact_id=artifact_id,
            version=version,
            provenance=provenance,
            derived_from=derived_from,
            produced_by=produced_by,
            created_at=utc_now_iso(),
        )
        storage.write_text(storage.join(directory, "meta.json"), info.model_dump_json())
        return info

    def delete_version(self, artifact_id: ArtifactId, version: int) -> None:
        """Remove one artifact version directory written by a failed move."""
        storage.rm_tree(self.version_dir(artifact_id, version))

    # -- read ----------------------------------------------------------------

    def read_meta(self, artifact_id: ArtifactId, version: int) -> ArtifactVersionInfo:
        path = self.file_path(artifact_id, version, "meta.json")
        return ArtifactVersionInfo.model_validate(storage.read_json(path))

    def read_json_file(self, artifact_id: ArtifactId, version: int, name: str) -> Any:
        return storage.read_json(self.file_path(artifact_id, version, name))

    def read_parquet_file(self, artifact_id: ArtifactId, version: int, name: str) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(
            self.file_path(artifact_id, version, name),
            storage_options=storage.polars_storage_options(),
        )

    def read_pickle_file(self, artifact_id: ArtifactId, version: int, name: str) -> Any:
        with storage.open_file(self.file_path(artifact_id, version, name), "rb") as f:
            return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Transition log and derived state
# ---------------------------------------------------------------------------


TransitionStatus = Literal["applied", "rejected", "raised"]
_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ResumeRef(BaseModel):
    """Stage-owned checkpoint selection retained by a raised transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["model_spec"]
    run_id: str
    checkpoint_id: str


class TransitionRecord(BaseModel):
    """One journaled transition attempt — applied, rejected, or raised.

    Rejections are recorded deliberately (a Temporal validator rejection
    leaves no trace in event history). Current state is reconstructed by
    replaying applied effects, not serialized into transition records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    ts: str
    move: Move
    status: TransitionStatus
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    trace_ids: list[str]
    resume: ResumeRef | None


class EpisodeJournal:
    """Per-workspace transition log: one JSON file per move attempt."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self._root = data_module.episode_dir(workspace_id)
        self._journal_dir = storage.join(self._root, "journal")

    def append(self, record: TransitionRecord) -> None:
        storage.makedirs(self._journal_dir)
        path = storage.join(self._journal_dir, f"{record.seq:06d}.json")
        if storage.exists(path):
            existing = TransitionRecord.model_validate(storage.read_json(path))
            if existing != record:
                raise FileExistsError(
                    f"Journal seq {record.seq} already exists with different content "
                    f"for {self.workspace_id}"
                )
            return
        storage.write_text(path, record.model_dump_json())

    def read(self, seq: int) -> TransitionRecord | None:
        path = storage.join(self._journal_dir, f"{seq:06d}.json")
        if not storage.exists(path):
            return None
        return TransitionRecord.model_validate(storage.read_json(path))

    def read_all(self) -> list[TransitionRecord]:
        if not storage.exists(self._journal_dir):
            return []
        entries = sorted(
            entry for entry in storage.listdir(self._journal_dir) if entry.endswith(".json")
        )
        return [TransitionRecord.model_validate(storage.read_json(entry)) for entry in entries]

    def latest_seq(self) -> int:
        """Highest transition sequence on disk, or 0 if the journal is empty.

        Journal filenames are zero-padded sequence numbers, so the max leaf is
        the last assigned seq — read from the directory listing without opening
        any record."""
        if not storage.exists(self._journal_dir):
            return 0
        seqs = [
            int(entry.rsplit("/", 1)[-1].removesuffix(".json"))
            for entry in storage.listdir(self._journal_dir)
            if entry.endswith(".json")
        ]
        return max(seqs, default=0)


def episode_trace_path(workspace_id: str, seq: int, subroutine_id: str) -> str:
    """Ledger location of one promoted transition trace."""
    if _TRACE_ID.fullmatch(subroutine_id) is None:
        raise ValueError(f"Invalid transition trace subroutine id: {subroutine_id!r}")
    return storage.join(
        data_module.episode_traces_dir(workspace_id),
        f"{seq:06d}",
        f"{subroutine_id}.json",
    )


def promote_run_traces(workspace_id: str, seq: int) -> list[str]:
    """Promote every finalized trace owned by this sequence's scratch run."""
    llm_root = storage.join(
        data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"),
        "llm",
    )
    trace_ids: list[str] = []
    for subroutine_root in sorted(storage.listdir(llm_root)):
        subroutine_id = subroutine_root.rstrip("/").rsplit("/", 1)[-1]
        source = storage.join(subroutine_root, "trace.json")
        if not storage.exists(source):
            continue
        destination = episode_trace_path(workspace_id, seq, subroutine_id)
        content = storage.read_text(source)
        if storage.exists(destination):
            if storage.read_text(destination) != content:
                raise ValueError(f"Transition trace collision at {destination}")
        else:
            storage.write_text(destination, content)
        trace_ids.append(subroutine_id)
    return trace_ids


def read_episode_trace(workspace_id: str, seq: int, subroutine_id: str) -> Any:
    return storage.read_json(episode_trace_path(workspace_id, seq, subroutine_id))


def derive_current_state(workspace_id: str) -> EpisodeState:
    """Replay committed transition effects into the current artifact state.

    Artifact activities persist immutable versions before returning their
    effects to the workflow. Only an ``applied`` transition record establishes
    that those versions became current, which also makes retractions exact and
    prevents partial or failed moves from leaking onto the read surface.
    """
    state = EpisodeState()
    for record in EpisodeJournal(workspace_id).read_all():
        if record.status == "applied":
            state = apply_transition(state, record.produced, record.retracted)
    return state
