"""Append-only versioned artifact store and episode journal.

Layout under ``data/{workspace_id}/``::

    store/{artifact_id}/v{N}/          one immutable version
        meta.json                      ArtifactVersionInfo dump
        <payload files>                artifact-specific (json/parquet/pkl)
    episode/journal/{seq:06d}.json     one transition record per file
    episode/state.json                 latest projected EpisodeState

Versions are never overwritten — the version log is what makes the state
machine's history reconstructible (timeline scrubber: state-at-t is a
filter over this log, plus ``state_after`` snapshots in the journal).
One JSON file per journal entry because the storage backends (local fs,
R2) have no atomic append; sequence numbers are assigned by the workflow,
which serializes moves per episode.
"""

from __future__ import annotations

import json
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
from nof1_causal_lab.machine.moves import Move  # noqa: TC001 (pydantic field annotations)
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
        # Attribute access (not a from-import) so tests can monkeypatch
        # nof1_causal_lab.utils.data.DATA_URI once for every consumer.
        self._root = storage.join(data_module.DATA_URI, workspace_id, "store")

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


def current_artifact_file(workspace_id: str, artifact_id: ArtifactId, filename: str) -> str:
    """Path to a file of the episode's CURRENT version of an artifact.

    For query-plane consumers that only need "the current X" without
    threading explicit pins. Raises FileNotFoundError when the artifact
    does not exist in the episode state.
    """
    state = EpisodeJournal(workspace_id).latest_state()
    info = state.get(artifact_id)
    if info is None:
        raise FileNotFoundError(f"No current '{artifact_id}' artifact for workspace {workspace_id}")
    return ArtifactStore(workspace_id).file_path(artifact_id, info.version, filename)


# ---------------------------------------------------------------------------
# Episode journal (transition log projection / read model)
# ---------------------------------------------------------------------------

TransitionStatus = Literal["applied", "rejected", "raised"]


class TransitionRecord(BaseModel):
    """One journaled transition attempt — applied, rejected, or raised.

    Rejections are recorded deliberately (a Temporal validator rejection
    leaves no trace in event history); ``state_after`` snapshots make the
    timeline scrubber a filter rather than a reconstruction.
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
    retracted: list[ArtifactId] = Field(default_factory=list)
    state_after: EpisodeState


class EpisodeJournal:
    """Per-workspace transition log: one JSON file per entry, plus a
    ``state.json`` read model holding the latest projected state."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self._root = storage.join(data_module.DATA_URI, workspace_id, "episode")
        self._journal_dir = storage.join(self._root, "journal")

    def append(self, record: TransitionRecord) -> None:
        storage.makedirs(self._journal_dir)
        path = storage.join(self._journal_dir, f"{record.seq:06d}.json")
        if storage.exists(path):
            raise FileExistsError(
                f"Journal seq {record.seq} already exists for {self.workspace_id}"
            )
        storage.write_text(path, record.model_dump_json())
        storage.write_text(
            storage.join(self._root, "state.json"),
            record.state_after.model_dump_json(),
        )

    def read_all(self) -> list[TransitionRecord]:
        if not storage.exists(self._journal_dir):
            return []
        entries = sorted(
            entry for entry in storage.listdir(self._journal_dir) if entry.endswith(".json")
        )
        return [TransitionRecord.model_validate(storage.read_json(entry)) for entry in entries]

    def latest_state(self) -> EpisodeState:
        path = storage.join(self._root, "state.json")
        if not storage.exists(path):
            return EpisodeState()
        return EpisodeState.model_validate(storage.read_json(path))
