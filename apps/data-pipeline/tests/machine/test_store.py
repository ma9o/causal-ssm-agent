"""Versioned artifact store + transition log."""

import polars as pl
import pytest

from nof1_causal_lab.machine.moves import RetractedArtifact, RunArtifact, WriteArtifact
from nof1_causal_lab.machine.store import (
    ArtifactStore,
    EpisodeJournal,
    TransitionRecord,
    derive_current_state,
)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
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
            produced_by="run:measurement_structure",
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
            produced_by="run:measurements",
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
    def _record(self, seq, move, status="applied", **kwargs):
        kwargs.setdefault("trace_ids", [])
        kwargs.setdefault("resume", None)
        return TransitionRecord(
            seq=seq,
            ts="2026-07-03T00:00:00+00:00",
            move=move,
            status=status,
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
        assert records[1].reason is not None
        assert "raw_data" in records[1].reason
        assert records[2].error_type == "ModelFitError"
        assert records[2].diagnostics["rhat_max"] == 2.4
        # Move discriminated union round-trips.
        assert records[0].move.kind == "write"
        assert records[2].move.artifact_id == "posterior"

    def test_identical_duplicate_seq_is_idempotent(self, workspace):
        journal = EpisodeJournal(workspace)
        record = self._record(1, WriteArtifact(artifact_id="question"))
        journal.append(record)
        journal.append(record)
        assert journal.read_all() == [record]

    def test_different_duplicate_seq_refused(self, workspace):
        journal = EpisodeJournal(workspace)
        journal.append(self._record(1, WriteArtifact(artifact_id="question")))
        with pytest.raises(FileExistsError):
            journal.append(self._record(1, RunArtifact(artifact_id="raw_data")))

    def test_latest_seq_reads_max_entry_without_state_manifest(self, workspace):
        journal = EpisodeJournal(workspace)
        assert journal.latest_seq() == 0
        journal.append(self._record(3, WriteArtifact(artifact_id="question")))
        assert journal.latest_seq() == 3


class TestDerivedCurrentState:
    def _append(self, workspace, seq, move, *, produced=None, retracted=None, status="applied"):
        EpisodeJournal(workspace).append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                move=move,
                status=status,
                produced=produced or [],
                retracted=retracted or [],
                trace_ids=[],
                resume=None,
            )
        )

    def test_current_state_replays_only_applied_versions(self, workspace):
        store = ArtifactStore(workspace)
        first = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "first"}},
        )
        self._append(
            workspace,
            1,
            WriteArtifact(artifact_id="question"),
            produced=[first],
        )
        second = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "second"}},
        )

        # Persisting a version is not the commit boundary. Until an applied
        # transition records it, readers continue to see the prior state.
        assert derive_current_state(workspace).get("question") == first

        self._append(
            workspace,
            2,
            WriteArtifact(artifact_id="question"),
            produced=[second],
        )

        assert derive_current_state(workspace).get("question") == second

    def test_rejected_and_raised_effects_are_not_current(self, workspace):
        store = ArtifactStore(workspace)
        rejected = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "rejected"}},
        )
        raised = store.write_version(
            "raw_data",
            provenance="computed",
            derived_from={},
            produced_by="run:raw_data",
            json_files={"raw-data.json": {}},
        )
        self._append(
            workspace,
            1,
            WriteArtifact(artifact_id="question"),
            produced=[rejected],
            status="rejected",
        )
        self._append(
            workspace,
            2,
            RunArtifact(artifact_id="raw_data"),
            produced=[raised],
            status="raised",
        )

        assert derive_current_state(workspace).current == {}

    def test_applied_retraction_removes_optional_output(self, workspace):
        store = ArtifactStore(workspace)
        measurements_v1 = store.write_version(
            "measurements",
            provenance="computed",
            derived_from={},
            produced_by="run:measurements",
            json_files={"measurements.json": {"workers": []}},
        )
        panel_v1 = store.write_version(
            "panel",
            provenance="computed",
            derived_from={},
            produced_by="run:measurements",
            parquet_files={"panel.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
        )
        self._append(
            workspace,
            1,
            RunArtifact(artifact_id="measurements"),
            produced=[measurements_v1, panel_v1],
        )

        state_with_panel = derive_current_state(workspace)
        assert state_with_panel.get("measurements") == measurements_v1
        assert state_with_panel.get("panel") == panel_v1

        measurements_v2 = store.write_version(
            "measurements",
            provenance="computed",
            derived_from={},
            produced_by="run:measurements",
            json_files={"measurements.json": {"workers": []}},
        )
        self._append(
            workspace,
            2,
            RunArtifact(artifact_id="measurements"),
            produced=[measurements_v2],
            retracted=[
                RetractedArtifact(
                    artifact_id="panel",
                    reason_ref="measurements.produces_optional.panel",
                )
            ],
        )

        state_without_panel = derive_current_state(workspace)
        assert state_without_panel.get("measurements") == measurements_v2
        assert state_without_panel.get("panel") is None
