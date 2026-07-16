"""Read-only facade: same read endpoints as the full facade, move plane 403s."""

from fastapi.testclient import TestClient

from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.utils import data as data_module


def test_read_facade_serves_reads_and_rejects_moves(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
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
    upload = client.post(
        "/api/upload",
        data={"workspaceId": "WS-READONLY"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )
    assert upload.status_code == 403


def test_artifact_endpoint_serves_pinned_versions(monkeypatch, tmp_path):
    from nof1_causal_lab.machine.moves import WriteArtifact
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    store = ArtifactStore("WS-ART")
    question = store.write_version(
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

    # Explicit version reads can inspect an uncommitted artifact, but it does
    # not become the current version until an applied move records the effect.
    assert client.get("/api/episodes/WS-ART/artifacts/question").status_code == 404
    EpisodeJournal("WS-ART").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            move=WriteArtifact(artifact_id="question"),
            status="applied",
            produced=[question],
            trace_ids=[],
            resume=None,
        )
    )
    current = client.get("/api/episodes/WS-ART/artifacts/question")
    assert current.status_code == 200
    assert current.json()["payload"]["question.json"] == {"text": "does X cause Y?"}
    missing = client.get("/api/episodes/WS-ART/artifacts/question", params={"version": 7})
    assert missing.status_code == 404


def test_trace_endpoints_join_artifact_version_to_promoted_trace(monkeypatch, tmp_path):
    from nof1_causal_lab.machine.moves import RunArtifact
    from nof1_causal_lab.machine.store import (
        ArtifactStore,
        EpisodeJournal,
        TransitionRecord,
        promote_run_traces,
    )
    from nof1_causal_lab.utils import storage
    from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    raw_data = ArtifactStore("WS-TRACE").write_version(
        "raw_data",
        provenance="llm",
        derived_from={},
        produced_by="run:raw_data",
        json_files={"profile.json": {"column_descriptions": []}},
    )
    source = str(tmp_path / "data/WS-TRACE/scratch/runs/seq-000001/llm/raw-data/trace.json")
    storage.write_text(
        source,
        LLMTrace(
            messages=[TraceMessage(role="assistant", content="profiled")],
            model="test-model",
        ).model_dump_json(),
    )
    trace_ids = promote_run_traces("WS-TRACE", 1)
    EpisodeJournal("WS-TRACE").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            move=RunArtifact(artifact_id="raw_data"),
            status="applied",
            produced=[raw_data],
            trace_ids=trace_ids,
            resume=None,
        )
    )
    client = TestClient(create_read_facade_app())

    trace_list = client.get("/api/episodes/WS-TRACE/artifacts/raw_data/traces")
    assert trace_list.status_code == 200
    assert trace_list.json()["trace_ids"] == ["raw-data"]
    trace = client.get("/api/episodes/WS-TRACE/traces/1/raw-data")
    assert trace.status_code == 200
    assert trace.json()["messages"][0]["content"] == "profiled"


def test_timeline_exposes_typed_resume_reference(monkeypatch, tmp_path):
    from nof1_causal_lab.machine.moves import RunArtifact
    from nof1_causal_lab.machine.store import EpisodeJournal, ResumeRef, TransitionRecord

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    EpisodeJournal("WS-RESUME").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            move=RunArtifact(artifact_id="statistical_model_spec"),
            status="raised",
            trace_ids=[],
            resume=ResumeRef(
                kind="model_spec",
                run_id="seq-000001",
                checkpoint_id="accepted-a.json",
            ),
        )
    )

    response = TestClient(create_read_facade_app()).get("/api/episodes/WS-RESUME/timeline")

    assert response.status_code == 200
    assert response.json()["transitions"][0]["resume"] == {
        "kind": "model_spec",
        "run_id": "seq-000001",
        "checkpoint_id": "accepted-a.json",
    }


def test_workspaces_endpoint_lists_episode_questions(monkeypatch, tmp_path):
    from nof1_causal_lab.machine.moves import WriteArtifact
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")

    store = ArtifactStore("WS-LIST")
    question = store.write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "does X cause Y?"}},
    )
    EpisodeJournal("WS-LIST").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            move=WriteArtifact(artifact_id="question"),
            status="applied",
            produced=[question],
            trace_ids=[],
            resume=None,
        )
    )

    client = TestClient(create_read_facade_app())
    response = client.get("/api/workspaces")

    assert response.status_code == 200
    assert response.json() == {
        "workspaces": [
            {
                "href": "/analysis/WS-LIST",
                "question": "does X cause Y?",
                "workspaceId": "WS-LIST",
            }
        ]
    }


def test_upload_endpoint_stages_input_file(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import storage

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.delenv("EPISODE_FACADE_READ_ONLY", raising=False)
    client = TestClient(create_read_facade_app())

    response = client.post(
        "/api/upload",
        data={"workspaceId": "WS-UPLOAD"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {"path": "WS-UPLOAD/input/data.csv"}
    assert storage.read_text(str(tmp_path / "data" / "WS-UPLOAD" / "input" / "data.csv")) == (
        "x,y\n1,2\n"
    )


def test_full_facade_advertises_moves(monkeypatch):
    monkeypatch.delenv("EPISODE_FACADE_READ_ONLY", raising=False)
    from nof1_causal_lab import tool_server

    client = TestClient(tool_server.app)
    assert client.get("/api/capabilities").json() == {"moves_enabled": True}
