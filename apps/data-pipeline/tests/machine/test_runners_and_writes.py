"""Runner/write executors: optional outputs and derivation cascade."""

import json

import polars as pl
import pytest

from nof1_causal_lab.flows.transitions.measurement_structure.identification import (
    derive_identification_report,
)
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import apply_transition, input_pins
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import execute_write


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def _valid_causal_design(*, include_identifiability=True, non_identifiable=None) -> dict:
    spec = {
        "latent": _latent_structure(),
        "measurement": _measurement_structure(),
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


def _latent_structure() -> dict:
    return {
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
    }


def _measurement_structure() -> dict:
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "stress_score",
                "construct_name": "Stress",
                "construct_polarity": "positive",
                "how_to_measure": "Self-reported stress",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "perf_score",
                "construct_name": "Perf",
                "construct_polarity": "positive",
                "how_to_measure": "Self-reported performance",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ],
    }


class TestIdentificationReportDerivation:
    def test_explicit_identifiable_treatments_produce_identification_report(self):
        report = derive_identification_report(_valid_causal_design())
        assert report is not None
        assert report["estimable_treatments"] == ["Stress"]

    def test_missing_identifiability_withholds_identification_report(self):
        report = derive_identification_report(_valid_causal_design(include_identifiability=False))
        assert report is None

    def test_all_non_identifiable_withholds_identification_report(self):
        report = derive_identification_report(
            _valid_causal_design(non_identifiable={"Stress": {"confounders": ["U"]}})
        )
        assert report is None


class TestStage2Gate:
    async def _finalize_measurements(self, workspace, observation_rows):
        from nof1_causal_lab.machine.temporal.measurement_activities import (
            finalize_measurements_activity,
        )
        from nof1_causal_lab.machine.temporal.messages import MeasurementsFinalizeInput
        from nof1_causal_lab.utils import data as data_module
        from nof1_causal_lab.utils import storage

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
                    produced_by="run:raw_data",
                    json_files={"profile.json": {"column_descriptions": []}},
                    parquet_files={"raw.parquet": pl.DataFrame({"timestamp": ["2026-01-01"]})},
                ),
                store.write_version(
                    "measurement_structure",
                    provenance="computed",
                    derived_from={},
                    produced_by="run:measurement_structure",
                    json_files={
                        "measurement_structure.json": {
                            "measurement_structure": _measurement_structure(),
                            "known_inputs": [],
                        }
                    },
                ),
            ]
        )

        spec = transition_spec("measurements")
        pins = input_pins(state, spec)
        run_id = "run-1"
        plan_ref = storage.join(
            data_module.scratch_run_dir(workspace, run_id),
            "extraction",
            "plan.json",
        )
        storage.write_text(
            plan_ref,
            json.dumps(
                {
                    "measurement_structure": _measurement_structure(),
                    "computed_dicts": observation_rows,
                    "chunks": [],
                }
            ),
        )

        return await finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id=workspace,
                state=state,
                run_id=run_id,
                plan_ref=plan_ref,
                pins=pins,
            )
        )

    def test_empty_extraction_withholds_panel(self, workspace):
        import asyncio

        effects = asyncio.run(self._finalize_measurements(workspace, []))
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"measurements"}

    def test_nonempty_extraction_produces_panel(self, workspace):
        import asyncio

        rows = [{"indicator": "stress_score", "value": 3.0, "timestamp": "2026-01-01"}]
        effects = asyncio.run(self._finalize_measurements(workspace, rows))
        produced = {info.artifact_id for info in effects.produced}
        assert produced == {"measurements", "panel"}
        panel = next(info for info in effects.produced if info.artifact_id == "panel")
        assert set(panel.derived_from) == {"question", "raw_data", "measurement_structure"}


class TestMeasurementStructureArtifactWrite:
    def _state_with_latent(self, store: ArtifactStore):
        latent_info = store.write_version(
            "latent_structure",
            provenance="llm",
            derived_from={},
            produced_by=None,
            json_files={"latent-structure.json": {"latent_structure": _latent_structure()}},
        )
        return EpisodeState().with_versions([latent_info])

    def test_write_cascades_causal_design_and_identification_report(self, workspace):
        store = ArtifactStore(workspace)
        state = self._state_with_latent(store)

        effects = execute_write(
            workspace,
            "measurement_structure",
            {"measurement_structure": _measurement_structure(), "known_inputs": []},
            "human",
            state,
        )

        produced = {info.artifact_id for info in effects.produced}
        assert produced == {
            "measurement_structure",
            "causal_design",
            "identification_report",
        }
        assert effects.retracted == []
        derived = {
            info.artifact_id: info.derived_from
            for info in effects.produced
            if info.artifact_id != "measurement_structure"
        }
        measurement_version = next(
            info.version for info in effects.produced if info.artifact_id == "measurement_structure"
        )
        assert derived["causal_design"] == {
            "latent_structure": 1,
            "measurement_structure": measurement_version,
        }
        assert derived["identification_report"] == {"causal_design": 1}

    def test_write_routes_known_input_into_derived_causal_design(self, workspace):
        from nof1_causal_lab.models.ssm.compile.spec_translation import (
            get_estimation_input_layout,
        )

        store = ArtifactStore(workspace)
        state = self._state_with_latent(store)

        effects = execute_write(
            workspace,
            "measurement_structure",
            {
                "measurement_structure": _measurement_structure(),
                "known_inputs": [
                    {
                        "construct": "Stress",
                        "source_indicator": "stress_score",
                        "scale": 2.0,
                        "missing_policy": "forward_fill",
                    }
                ],
            },
            "human",
            state,
        )

        causal_design_info = next(
            info for info in effects.produced if info.artifact_id == "causal_design"
        )
        causal_design = store.read_json_file(
            "causal_design",
            causal_design_info.version,
            "causal_design.json",
        )["causal_design"]
        assert causal_design["estimation"]["state_order"] == ["Perf"]
        assert causal_design["estimation"]["known_inputs"] == [
            {
                "construct": "Stress",
                "source_indicator": "stress_score",
                "scale": 2.0,
                "missing_policy": "forward_fill",
            }
        ]
        assert [
            (edge["cause"], edge["effect"]) for edge in causal_design["estimation"]["edges"]
        ] == [("Stress", "Perf")]
        assert get_estimation_input_layout(causal_design) == (
            ["Stress"],
            ["stress_score"],
            [2.0],
            ["forward_fill"],
            [True],
        )

    def test_embedded_authoring_finalizer_persists_known_inputs(self, workspace, tmp_path):
        import asyncio

        from nof1_causal_lab.machine.temporal.measurement_structure_activities import (
            finalize_measurement_structure_activity,
        )
        from nof1_causal_lab.machine.temporal.messages import SingleLLMTransitionFinalizeInput
        from nof1_causal_lab.utils import storage

        store = ArtifactStore(workspace)
        question = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "does stress hurt performance?"}},
        )
        raw_data = store.write_version(
            "raw_data",
            provenance="computed",
            derived_from={},
            produced_by="run:raw_data",
            json_files={"profile.json": {"column_descriptions": []}},
        )
        latent_structure = store.write_version(
            "latent_structure",
            provenance="computed",
            derived_from={"question": question.version},
            produced_by="run:latent_structure",
            json_files={"latent-structure.json": {"latent_structure": _latent_structure()}},
        )
        state = EpisodeState().with_versions([question, raw_data, latent_structure])
        result_ref = str(tmp_path / "measurement-result.json")
        storage.write_text(
            result_ref,
            json.dumps(
                {
                    "measurement_structure": _measurement_structure(),
                    "known_inputs": [
                        {
                            "construct": "Stress",
                            "source_indicator": "stress_score",
                            "scale": 2.0,
                            "missing_policy": "forward_fill",
                        }
                    ],
                }
            ),
        )

        effects = asyncio.run(
            finalize_measurement_structure_activity(
                SingleLLMTransitionFinalizeInput(
                    workspace_id=workspace,
                    transition_id="measurement_structure",
                    state=state,
                    pins={
                        "question": question.version,
                        "raw_data": raw_data.version,
                        "latent_structure": latent_structure.version,
                    },
                    context_ref="unused-context.json",
                    result_ref=result_ref,
                    trace_ref="trace://measurement",
                )
            )
        )

        measurement_info = next(
            info for info in effects.produced if info.artifact_id == "measurement_structure"
        )
        measurement_payload = store.read_json_file(
            "measurement_structure",
            measurement_info.version,
            "measurement_structure.json",
        )
        assert measurement_payload["known_inputs"][0]["construct"] == "Stress"
        assert "llm_trace_ref" not in measurement_payload

        causal_design_info = next(
            info for info in effects.produced if info.artifact_id == "causal_design"
        )
        causal_design = store.read_json_file(
            "causal_design",
            causal_design_info.version,
            "causal_design.json",
        )["causal_design"]
        assert causal_design["estimation"]["state_order"] == ["Perf"]
        assert causal_design["estimation"]["known_inputs"] == measurement_payload["known_inputs"]

    def test_write_retracts_derivations_with_stale_non_cascading_parents(self, workspace):
        store = ArtifactStore(workspace)
        question = store.write_version(
            "question",
            provenance="human",
            derived_from={},
            produced_by=None,
            json_files={"question.json": {"text": "does stress hurt performance?"}},
        )
        raw_data = store.write_version(
            "raw_data",
            provenance="computed",
            derived_from={},
            produced_by="run:raw_data",
            json_files={"profile.json": {"column_descriptions": []}},
            parquet_files={"raw.parquet": pl.DataFrame({"timestamp": ["2026-01-01"]})},
        )
        latent_structure = store.write_version(
            "latent_structure",
            provenance="computed",
            derived_from={"question": question.version},
            produced_by="run:latent_structure",
            json_files={"latent-structure.json": {"latent_structure": _latent_structure()}},
        )
        old_measurement = store.write_version(
            "measurement_structure",
            provenance="computed",
            derived_from={
                "question": question.version,
                "raw_data": raw_data.version,
                "latent_structure": latent_structure.version,
            },
            produced_by="run:measurement_structure",
            json_files={
                "measurement_structure.json": {
                    "measurement_structure": _measurement_structure(),
                    "known_inputs": [],
                }
            },
        )
        old_causal_design = store.write_version(
            "causal_design",
            provenance="computed",
            derived_from={
                "latent_structure": latent_structure.version,
                "measurement_structure": old_measurement.version,
            },
            produced_by="derive:causal_design",
            json_files={"causal_design.json": {"causal_design": _valid_causal_design()}},
        )
        old_identification = store.write_version(
            "identification_report",
            provenance="computed",
            derived_from={"causal_design": old_causal_design.version},
            produced_by="derive:identification_report",
            json_files={
                "identification_report.json": {
                    "outcome_name": "Perf",
                    "estimable_treatments": ["Stress"],
                    "non_identifiable_treatments": {},
                }
            },
        )
        panel = store.write_version(
            "panel",
            provenance="computed",
            derived_from={
                "question": question.version,
                "raw_data": raw_data.version,
                "measurement_structure": old_measurement.version,
            },
            produced_by="run:measurements",
            parquet_files={
                "panel.parquet": pl.DataFrame(
                    {
                        "indicator": ["stress_score"],
                        "value": [3.0],
                        "anchor_time": ["2026-01-01"],
                    }
                )
            },
        )
        validation = store.write_version(
            "validation_report",
            provenance="computed",
            derived_from={"panel": panel.version, "causal_design": old_causal_design.version},
            produced_by="derive:validation_report",
            json_files={
                "validation_report.json": {
                    "is_valid": True,
                    "indicators": {},
                    "dataset_issues": [],
                }
            },
        )
        statistical_model_spec = store.write_version(
            "statistical_model_spec",
            provenance="computed",
            derived_from={
                "question": question.version,
                "causal_design": old_causal_design.version,
                "identification_report": old_identification.version,
                "panel": panel.version,
                "validation_report": validation.version,
            },
            produced_by="run:statistical_model_spec",
            json_files={
                "statistical_model_spec.json": {
                    "statistical_model_spec": {"likelihoods": [], "parameters": []},
                    "authored_priors": {},
                    "resolved_priors": [],
                    "prior_predictive_samples": {},
                }
            },
        )
        compiled = store.write_version(
            "compiled_ssm",
            provenance="computed",
            derived_from={
                "statistical_model_spec": statistical_model_spec.version,
                "causal_design": old_causal_design.version,
            },
            produced_by="derive:compiled_ssm",
            json_files={"compiled-ssm.json": {"spec": {}}, "report.json": {}},
        )
        state = EpisodeState().with_versions(
            [
                question,
                raw_data,
                latent_structure,
                old_measurement,
                old_causal_design,
                old_identification,
                panel,
                validation,
                statistical_model_spec,
                compiled,
            ]
        )

        effects = execute_write(
            workspace,
            "measurement_structure",
            {"measurement_structure": _measurement_structure(), "known_inputs": []},
            "human",
            state,
        )

        produced = {info.artifact_id for info in effects.produced}
        retracted = {item.artifact_id: item.reason_ref for item in effects.retracted}
        assert produced == {
            "measurement_structure",
            "causal_design",
            "identification_report",
        }
        assert retracted == {
            "validation_report": "validation_report.parents_stale.panel",
            "compiled_ssm": "compiled_ssm.parents_stale.statistical_model_spec",
        }
        next_state = apply_transition(state, effects.produced, effects.retracted)
        assert not next_state.has("validation_report")
        assert not next_state.has("compiled_ssm")

    def test_failed_cascade_removes_written_versions(self, workspace, monkeypatch):
        from nof1_causal_lab.utils import identifiability

        store = ArtifactStore(workspace)
        state = self._state_with_latent(store)

        def fail_identifiability(*_args, **_kwargs):
            raise RuntimeError("identification failed")

        monkeypatch.setattr(identifiability, "check_identifiability", fail_identifiability)

        with pytest.raises(RuntimeError, match="identification failed"):
            execute_write(
                workspace,
                "measurement_structure",
                {"measurement_structure": _measurement_structure(), "known_inputs": []},
                "human",
                state,
            )

        assert store.list_versions("measurement_structure") == []
        assert store.list_versions("causal_design") == []

    def test_invalid_measurement_payload_rejected(self, workspace):
        with pytest.raises(ArtifactWriteRejected):
            execute_write(
                workspace,
                "measurement_structure",
                {"measurement_structure": {"nope": True}, "known_inputs": []},
                "human",
                EpisodeState(),
            )

    def test_question_write_requires_text(self, workspace):
        with pytest.raises(ArtifactWriteRejected):
            execute_write(workspace, "question", {"text": "   "}, "human", EpisodeState())

    def test_binary_artifacts_not_directly_writable(self, workspace):
        with pytest.raises(ArtifactWriteRejected, match="no write executor"):
            execute_write(workspace, "posterior", {"anything": 1}, "human", EpisodeState())
