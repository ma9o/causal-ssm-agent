"""Reachability-based collection for workspace scratch state and caches."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

DEFAULT_EVENT_RETENTION_SECONDS = 24 * 60 * 60
DEFAULT_CACHE_RETENTION_SECONDS = 30 * 24 * 60 * 60
DEFAULT_CACHE_MAX_BYTES = 5 * 1024**3


class SweepResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    removed_runs: int = 0
    removed_events: int = 0
    removed_cache_files: int = 0
    removed_cache_bytes: int = 0
    protected_runs: list[str]


def _latest_resume_run(records: list[TransitionRecord]) -> str | None:
    """Resume root selected by the latest model-spec transition attempt."""
    for record in reversed(records):
        if not (
            isinstance(record.move, RunArtifact)
            and record.move.artifact_id == "statistical_model_spec"
        ):
            continue
        if record.status == "raised" and record.resume is not None:
            return record.resume.run_id
        return None
    return None


def collect_completed_runs(workspace_id: str) -> tuple[int, list[str]]:
    """Delete run scratch while the episode move lock is held or the episode is offline."""
    resume_run = _latest_resume_run(EpisodeJournal(workspace_id).read_all())
    protected = {resume_run} if resume_run is not None else set()
    root = data_module.scratch_runs_dir(workspace_id)
    removed = 0
    for entry in storage.listdir(root):
        run_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if run_id in protected:
            continue
        storage.rm_tree(entry)
        removed += 1
    return removed, sorted(protected)


def _sweep_events(workspace_id: str, *, cutoff_seconds: float) -> int:
    removed = 0
    for entry in storage.listdir(data_module.scratch_events_dir(workspace_id)):
        cursor = entry.rsplit("/", 1)[-1]
        emitted_ns = int(cursor.split("-", 1)[0])
        if emitted_ns / 1_000_000_000 >= cutoff_seconds:
            continue
        storage.rm_file(entry)
        removed += 1
    return removed


def _modified_seconds(path: str) -> float:
    info = storage.file_info(path)
    if storage.is_remote():
        modified = info["LastModified"]
        if not isinstance(modified, datetime):
            raise TypeError(f"Remote object has non-datetime LastModified: {path}")
        return modified.timestamp()
    return float(info["mtime"])


def _sweep_cache(
    workspace_id: str,
    *,
    cutoff_seconds: float,
    max_bytes: int,
) -> tuple[int, int]:
    entries = [
        (path, _modified_seconds(path), int(storage.file_info(path)["size"]))
        for path in storage.walk_files(data_module.cache_dir(workspace_id))
    ]
    removed_files = 0
    removed_bytes = 0
    retained: list[tuple[str, float, int]] = []
    for path, modified, size in entries:
        if modified < cutoff_seconds:
            storage.rm_file(path)
            removed_files += 1
            removed_bytes += size
        else:
            retained.append((path, modified, size))

    total = sum(size for _, _, size in retained)
    for path, _, size in sorted(retained, key=lambda item: (item[1], item[0])):
        if total <= max_bytes:
            break
        storage.rm_file(path)
        total -= size
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


def sweep_workspace(
    workspace_id: str,
    *,
    now_seconds: float | None = None,
    event_retention_seconds: int = DEFAULT_EVENT_RETENTION_SECONDS,
    cache_retention_seconds: int = DEFAULT_CACHE_RETENTION_SECONDS,
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
    collect_runs: bool = False,
) -> SweepResult:
    """Expire telemetry/caches and optionally collect runs while the episode is offline."""
    now = time.time() if now_seconds is None else now_seconds
    if collect_runs:
        removed_runs, protected_runs = collect_completed_runs(workspace_id)
    else:
        removed_runs, protected_runs = 0, []
    removed_events = _sweep_events(
        workspace_id,
        cutoff_seconds=now - event_retention_seconds,
    )
    removed_cache_files, removed_cache_bytes = _sweep_cache(
        workspace_id,
        cutoff_seconds=now - cache_retention_seconds,
        max_bytes=cache_max_bytes,
    )
    return SweepResult(
        removed_runs=removed_runs,
        removed_events=removed_events,
        removed_cache_files=removed_cache_files,
        removed_cache_bytes=removed_cache_bytes,
        protected_runs=protected_runs,
    )


def sweep_cli() -> None:
    parser = argparse.ArgumentParser(description="Collect workspace scratch state and caches.")
    parser.add_argument("workspace_id")
    parser.add_argument("--event-retention-hours", type=int, default=24)
    parser.add_argument("--cache-retention-days", type=int, default=30)
    parser.add_argument("--cache-max-gib", type=float, default=5.0)
    parser.add_argument(
        "--collect-runs",
        action="store_true",
        help="collect run scratch; use only while the episode is offline",
    )
    args = parser.parse_args()
    result = sweep_workspace(
        args.workspace_id,
        event_retention_seconds=args.event_retention_hours * 60 * 60,
        cache_retention_seconds=args.cache_retention_days * 24 * 60 * 60,
        cache_max_bytes=int(args.cache_max_gib * 1024**3),
        collect_runs=args.collect_runs,
    )
    print(result.model_dump_json())


if __name__ == "__main__":
    sweep_cli()
