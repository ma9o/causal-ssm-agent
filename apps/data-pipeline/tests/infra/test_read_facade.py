"""Read-only facade: same read endpoints as the full facade, move plane 403s."""

from fastapi.testclient import TestClient

from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.utils import data as data_module


def test_read_facade_serves_reads_and_rejects_moves(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    client = TestClient(create_read_facade_app())

    assert client.get("/api/capabilities").json() == {"moves_enabled": False}

    status = client.get("/api/episodes/WS-READONLY")
    assert status.status_code == 200
    body = status.json()
    assert body["seq"] == 0
    assert body["auto_running"] is False

    move = client.post(
        "/api/episodes/WS-READONLY/moves",
        json={"move": {"kind": "run", "artifact_id": "raw_data"}},
    )
    assert move.status_code == 403
    start = client.post("/api/episodes", json={"workspace_id": "WS-READONLY"})
    assert start.status_code == 403
    auto = client.post("/api/episodes/WS-READONLY/auto", json={})
    assert auto.status_code == 403


def test_artifact_endpoint_serves_pinned_versions(monkeypatch, tmp_path):
    from nof1_causal_lab.machine.store import ArtifactStore

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    store = ArtifactStore("WS-ART")
    store.write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "does X cause Y?"}},
    )
    client = TestClient(create_read_facade_app())

    pinned = client.get("/api/episodes/WS-ART/artifacts/question", params={"version": 1})
    assert pinned.status_code == 200
    body = pinned.json()
    assert body["payload"]["question.json"] == {"text": "does X cause Y?"}
    assert body["meta"]["provenance"] == "human"
    assert body["binary_files"] == []

    # No journal projection yet: there is no *current* version to default to.
    assert client.get("/api/episodes/WS-ART/artifacts/question").status_code == 404
    missing = client.get("/api/episodes/WS-ART/artifacts/question", params={"version": 7})
    assert missing.status_code == 404


def test_full_facade_advertises_moves(monkeypatch):
    monkeypatch.delenv("EPISODE_FACADE_READ_ONLY", raising=False)
    from nof1_causal_lab import tool_server

    client = TestClient(tool_server.app)
    assert client.get("/api/capabilities").json() == {"moves_enabled": True}
