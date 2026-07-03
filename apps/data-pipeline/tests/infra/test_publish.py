"""Publish: idempotent store copy with mutable-read-model overwrite and excludes."""

import json

import fsspec
import pytest

from nof1_causal_lab import publish
from nof1_causal_lab.utils import data as data_module


def _seed_workspace(root):
    files = {
        "store/question/v1/meta.json": {"artifact_id": "question", "version": 1},
        "store/question/v1/question.json": {"text": "does X cause Y?"},
        "store/raw_data/v1/meta.json": {"artifact_id": "raw_data", "version": 1},
        "episode/journal/000001.json": {"seq": 1},
        "episode/state.json": {"current": {}},
        "run/stage-0.json": {"ok": True},
        "input/MyActivity.json": {"secret": "personal"},
    }
    for rel, payload in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))


@pytest.fixture
def memory_dest(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    monkeypatch.setattr(publish, "_dest_fs", lambda: fs)
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setenv("R2_BUCKET", "pub")
    monkeypatch.setenv("R2_PREFIX", "data")
    return fs


def test_publish_excludes_and_is_idempotent(monkeypatch, tmp_path, memory_dest):
    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    ws_root = tmp_path / "data" / "WS-PUB"
    _seed_workspace(ws_root)

    counts = publish.publish_workspace("WS-PUB", ["raw_data", "input"])
    assert counts == {"uploaded": 5, "skipped": 0, "excluded": 2}
    assert memory_dest.exists("pub/data/WS-PUB/store/question/v1/question.json")
    assert not memory_dest.exists("pub/data/WS-PUB/store/raw_data/v1/meta.json")
    assert not memory_dest.exists("pub/data/WS-PUB/input/MyActivity.json")

    # Second run: immutable keys skip, mutable read models re-upload.
    counts = publish.publish_workspace("WS-PUB", ["raw_data", "input"])
    assert counts == {"uploaded": 2, "skipped": 3, "excluded": 2}

    # A new journal entry (live tail) uploads without touching existing keys.
    (ws_root / "episode/journal/000002.json").write_text(json.dumps({"seq": 2}))
    counts = publish.publish_workspace("WS-PUB", ["raw_data", "input"])
    assert counts == {"uploaded": 3, "skipped": 3, "excluded": 2}


def test_publish_refuses_remote_source(monkeypatch, memory_dest):
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    with pytest.raises(RuntimeError, match="local store"):
        publish.publish_workspace("WS-PUB", [])
