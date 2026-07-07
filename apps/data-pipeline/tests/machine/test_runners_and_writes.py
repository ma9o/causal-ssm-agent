"""Runner/write executors: the produced-when-nonempty gates and write fan-out."""

import polars as pl
import pytest

from nof1_causal_lab.flows.stages.stage1b.result import split_stage1b_result
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.runners import execute_stage_locally
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import execute_write


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def _valid_causal_design(*, include_identifiability=True, non_identifiable=None) -> dict:
    spec = {
        "latent": {
            "constructs": [
                {
                    "name": "Perf",
                    "description": "Performance",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                },
                {
                    "name": "Stress",
                    "description": "Stress level",
                    "role": "endogenous",
                    "is_outcome": False,
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause": "Stress",
                    "effect": "Perf",
                    "description": "Stress reduces performance",
                    "lagged": True,
                }
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "construct_name": "Stress",
                    "construct_polarity": "positive",
                    "how_to_measure": "Self-reported stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
        },
        "estimation": {
            "state_order": ["Stress", "Perf"],
            "edges": [
                {
                    "cause": "Stress",
                    "effect": "Perf",
                    "description": "Stress reduces performance",
                    "lagged": True,
                }
            ],
            "known_inputs": [],
        },
    }
    if include_identifiability:
        identifiable = (
            {}
            if non_identifiable
            else {
                "Stress": {
                    "method": "do_calculus",
                    "estimand": "P(Perf | do(Stress))",
                    "marginalized_confounders": [],
                    "instruments": [],
                }
            }
        )
        spec["identifiability"] = {
            "identifiable_treatments": identifiable,
            "non_identifiable_treatments": non_identifiable or {},
        }
    return spec


class TestStage1bSplit:
    def test_explicit_identifiable_treatments_produce_identification_report(self):
        artifacts = split_stage1b_result({"causal_design": _valid_causal_design()})
        assert artifacts.identification_report["estimable_treatments"] == ["Stress"]

    def test_missing_identifiability_withholds_identification_report(self):
        artifacts = split_stage1b_result(
            {"causal_design": _valid_causal_design(include_identifiability=False)}
        )
        assert artifacts.identification_report is None

    def test_all_non_identifiable_withholds_identification_report(self):
        """The epistemic gate: no positive report means the fit chain is disabled."""
        artifacts = split_stage1b_result(
            {
                "causal_design": _valid_causal_design(
                    non_identifiable={"Stress": {"confounders": ["U"]}}
                )
            }
        )
        assert artifacts.identification_report is None


class TestStage2Gate:
    async def _run_stage2(self, workspace, monkeypatch, observation_rows):
        from nof1_causal_lab.flows.stages.stage2 import flow as stage2_flow
        from nof1_causal_lab.flows.stages.stage2 import materialization

        store = ArtifactStore(workspace)
        state = EpisodeState().with_versions(
            [
                store.write_version(
                    "question",
                    provenance="human",
                    derived_from={},
                    produced_by=None,
                    json_files={"question.json": {"text": "does stress hurt performance?"}},
                ),
                store.write_version(
                    "raw_data",
                    provenance="computed",
                    derived_from={},
                    produced_by="stage-0",
                    json_files={"profile.json": {"column_descriptions": []}},
                    parquet_files={"raw.parquet": pl.DataFrame({"timestamp": ["2026-01-01"]})},
                ),
                store.write_version(
                    "causal_design",
                    provenance="computed",
                    derived_from={},
                    produced_by="stage-1b",
                    json_files={"causal_design.json": {"causal_design": _valid_causal_design()}},
                ),
            ]
        )

        async def fake_extraction(raw_df, question, causal_design, **kwargs):
            return {"observation_rows": observation_rows, "worker_statuses": []}

        def fake_materialize(result, causal_design):
            return {
                "data_for_model": pl.DataFrame(result["observation_rows"]),
                "worker_statuses": result["worker_statuses"],
            }

        monkeypatch.setattr(stage2_flow, "run_stage2_extraction", fake_extraction)
        monkeypatch.setattr(materialization, "materialize_stage2_outputs", fake_materialize)

        from nof1_causal_lab.machine.moves import ExecOptions

        return await execute_stage_locally(workspace, "stage-2", _pins(state), ExecOptions())

    def test_empty_extraction_withholds_model_data(self, workspace, monkeypatch):
        import asyncio

        effects = asyncio.run(self._run_stage2(workspace, monkeypatch, []))
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"extraction_report"}

    def test_nonempty_extraction_produces_model_data(self, workspace, monkeypatch):
        import asyncio

        rows = [{"indicator": "stress_score", "value": 3.0, "timestamp": "2026-01-01"}]
        effects = asyncio.run(self._run_stage2(workspace, monkeypatch, rows))
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"extraction_report", "model_data"}
        model_data = next(i for i in effects.produced if i.artifact_id == "model_data")
        # derived_from pins the exact inputs consumed
        assert set(model_data.derived_from) == {"question", "raw_data", "causal_design"}


def _pins(state):
    from nof1_causal_lab.machine.graph import stage_spec
    from nof1_causal_lab.machine.moves import input_pins

    return input_pins(state, stage_spec("stage-2"))


class TestCausalDesignWrite:
    def test_write_fans_out_positive_identification_report(self, workspace):
        effects = execute_write(
            workspace,
            "causal_design",
            {"causal_design": _valid_causal_design()},
            "human",
        )
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"causal_design", "identification_report"}
        assert effects.retracted == []
        for info in effects.produced:
            assert info.provenance == "human"
        derived = {
            info.artifact_id: info.derived_from
            for info in effects.produced
            if info.artifact_id != "causal_design"
        }
        spec_version = next(
            info.version for info in effects.produced if info.artifact_id == "causal_design"
        )
        assert derived["identification_report"] == {"causal_design": spec_version}

    def test_write_with_nothing_estimable_retracts_identification_report(self, workspace):
        effects = execute_write(
            workspace,
            "causal_design",
            {
                "causal_design": _valid_causal_design(
                    non_identifiable={"Stress": {"confounders": ["U"]}}
                )
            },
            "human",
        )
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"causal_design"}
        assert effects.retracted == ["identification_report"]

    def test_invalid_payload_rejected(self, workspace):
        with pytest.raises(ArtifactWriteRejected):
            execute_write(workspace, "causal_design", {"causal_design": {"nope": True}}, "human")

    def test_question_write_requires_text(self, workspace):
        with pytest.raises(ArtifactWriteRejected):
            execute_write(workspace, "question", {"text": "   "}, "human")

    def test_binary_artifacts_not_directly_writable(self, workspace):
        with pytest.raises(ArtifactWriteRejected, match="no write executor"):
            execute_write(workspace, "posterior", {"anything": 1}, "human")
