"""Lifecycle collection for scratch runs, telemetry, and caches."""

import os
from pathlib import Path

from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, ResumeRef, TransitionRecord
from nof1_causal_lab.machine.sweep import sweep_workspace
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _workspace(monkeypatch, tmp_path) -> str:
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "sweep-test"


def _run_file(workspace_id: str, run_id: str) -> str:
    path = storage.join(data_module.scratch_run_dir(workspace_id, run_id), "context.json")
    storage.write_text(path, "{}")
    return path


def test_offline_run_collection_preserves_only_latest_resume_run(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    EpisodeJournal(workspace_id).append(
        TransitionRecord(
            seq=3,
            ts="2026-07-15T00:00:00Z",
            move=RunArtifact(artifact_id="statistical_model_spec"),
            status="raised",
            trace_ids=[],
            resume=ResumeRef(
                kind="model_spec",
                run_id="seq-000003",
                checkpoint_id="checkpoint-000001.json",
            ),
        )
    )
    old_run = _run_file(workspace_id, "seq-000001")
    resume_run = _run_file(workspace_id, "seq-000003")
    unjournaled_run = _run_file(workspace_id, "seq-000004")

    result = sweep_workspace(workspace_id, now_seconds=1_000, collect_runs=True)

    assert result.removed_runs == 2
    assert result.protected_runs == ["seq-000003"]
    assert not storage.exists(old_run)
    assert storage.exists(resume_run)
    assert not storage.exists(unjournaled_run)


def test_default_sweep_does_not_infer_run_liveness(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    run = _run_file(workspace_id, "seq-000001")

    result = sweep_workspace(workspace_id, now_seconds=1_000)

    assert result.removed_runs == 0
    assert storage.exists(run)


def test_sweep_expires_events_and_cache_and_bounds_cache_size(monkeypatch, tmp_path):
    workspace_id = _workspace(monkeypatch, tmp_path)
    events = data_module.scratch_events_dir(workspace_id)
    old_event = storage.join(events, f"{100 * 1_000_000_000:020d}-old.json")
    new_event = storage.join(events, f"{900 * 1_000_000_000:020d}-new.json")
    storage.write_text(old_event, "{}")
    storage.write_text(new_event, "{}")

    old_cache = storage.join(data_module.cache_dir(workspace_id), "old.bin")
    first_cache = storage.join(data_module.cache_dir(workspace_id), "first.bin")
    second_cache = storage.join(data_module.cache_dir(workspace_id), "second.bin")
    for path in (old_cache, first_cache, second_cache):
        storage.write_text(path, "12345")
    os.utime(Path(old_cache), (100, 100))
    os.utime(Path(first_cache), (800, 800))
    os.utime(Path(second_cache), (900, 900))

    result = sweep_workspace(
        workspace_id,
        now_seconds=1_000,
        event_retention_seconds=500,
        cache_retention_seconds=500,
        cache_max_bytes=5,
    )

    assert result.removed_events == 1
    assert not storage.exists(old_event)
    assert storage.exists(new_event)
    assert result.removed_cache_files == 2
    assert result.removed_cache_bytes == 10
    assert not storage.exists(old_cache)
    assert not storage.exists(first_cache)
    assert storage.exists(second_cache)
