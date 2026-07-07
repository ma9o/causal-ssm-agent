"""Versioned artifact store + episode journal."""

import polars as pl
import pytest

from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import RunArtifact, WriteArtifact
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


class TestArtifactStore:
    def test_versions_are_append_only_and_monotonic(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "does exercise improve sleep?"}},
        )
        second = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "does caffeine harm sleep?"}},
        )
        assert (first.version, second.version) == (1, 2)
        assert store.list_versions("question") == [1, 2]
        # Old version stays readable — nothing is overwritten.
        assert store.read_json_file("question", 1, "question.json")["text"].startswith(
            "does exercise"
        )
        assert store.read_json_file("question", 2, "question.json")["text"].startswith(
            "does caffeine"
        )

    def test_meta_roundtrip(self, workspace):
        store = ArtifactStore(workspace)
        info = store.write_version(
            "causal_design",
            provenance="computed",
            derived_from={"question": 1, "raw_data": 2, "latent_structure": 1},
            produced_by="stage-1b",
            json_files={"causal_design.json": {"latent": {}}},
        )
        loaded = store.read_meta("causal_design", info.version)
        assert loaded == info
        assert loaded.derived_from["raw_data"] == 2
        assert loaded.created_at

    def test_parquet_and_pickle_payloads(self, workspace):
        store = ArtifactStore(workspace)
        df = pl.DataFrame({"indicator": ["mood"], "value": [3.5]})
        info = store.write_version(
            "panel",
            provenance="computed",
            derived_from={},
            produced_by="stage-2",
            parquet_files={"panel.parquet": df},
            pickle_files={"aux.pkl": {"answer": 42}},
        )
        loaded_df = store.read_parquet_file("panel", info.version, "panel.parquet")
        assert loaded_df.equals(df)
        assert store.read_pickle_file("panel", info.version, "aux.pkl") == {"answer": 42}

    def test_empty_artifact_has_no_versions(self, workspace):
        store = ArtifactStore(workspace)
        assert store.list_versions("posterior") == []
        assert store.next_version("posterior") == 1


class TestEpisodeJournal:
    def _record(self, seq, move, status="applied", state=None, **kwargs):
        return TransitionRecord(
            seq=seq,
            ts="2026-07-03T00:00:00+00:00",
            move=move,
            status=status,
            state_after=state or EpisodeState(),
            **kwargs,
        )

    def test_append_and_read_back_in_order(self, workspace):
        journal = EpisodeJournal(workspace)
        journal.append(self._record(1, WriteArtifact(artifact_id="question")))
        journal.append(
            self._record(
                2,
                RunArtifact(artifact_id="measurement_structure"),
                status="rejected",
                reason=(
                    "measurement_structure requires artifacts that do not exist: "
                    "raw_data, latent_structure"
                ),
            )
        )
        journal.append(
            self._record(
                3,
                RunArtifact(artifact_id="posterior"),
                status="raised",
                error_type="ModelFitError",
                error_message="sampler diverged",
                diagnostics={"rhat_max": 2.4},
            )
        )
        records = journal.read_all()
        assert [r.seq for r in records] == [1, 2, 3]
        assert records[1].status == "rejected"
        assert "raw_data" in records[1].reason
        assert records[2].error_type == "ModelFitError"
        assert records[2].diagnostics["rhat_max"] == 2.4
        # Move discriminated union round-trips.
        assert records[0].move.kind == "write"
        assert records[2].move.artifact_id == "posterior"

    def test_duplicate_seq_refused(self, workspace):
        journal = EpisodeJournal(workspace)
        journal.append(self._record(1, WriteArtifact(artifact_id="question")))
        with pytest.raises(FileExistsError):
            journal.append(self._record(1, WriteArtifact(artifact_id="question")))

    def test_latest_state_is_last_applied_snapshot(self, workspace):
        from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo

        journal = EpisodeJournal(workspace)
        assert journal.latest_state() == EpisodeState()
        state = EpisodeState().with_versions(
            [ArtifactVersionInfo(artifact_id="question", version=1, provenance="human")]
        )
        journal.append(self._record(1, WriteArtifact(artifact_id="question"), state=state))
        assert journal.latest_state().has("question")
