"""Tests for the run lineage validator script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, ClassVar


def _load_validate_run() -> Any:
    module_name = "validate_run_under_test"
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_run.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_source_column_rules_use_raw_parquet_schema_when_descriptions_are_absent() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        artifacts={
            "raw_data": {"column_descriptions": []},
            "causal_design": {
                "causal_design": {
                    "measurement": {
                        "indicators": [
                            {"name": "observed_value", "source_columns": ["value"]},
                        ],
                    },
                },
            },
        },
        artifact_paths={},
        model_indicators=None,
        raw_input_columns={"timestamp", "value"},
    )

    assert validate_run.rule_raw_data_columns_match_raw_parquet(ctx) == []
    assert validate_run.rule_source_columns_in_raw_data(ctx) == []

    bad_ctx = validate_run.RunContext(
        workspace_id="ws",
        artifacts={
            **ctx.artifacts,
            "causal_design": {
                "causal_design": {
                    "measurement": {
                        "indicators": [
                            {"name": "missing_value", "source_columns": ["missing"]},
                        ],
                    },
                },
            },
        },
        artifact_paths={},
        model_indicators=None,
        raw_input_columns={"timestamp", "value"},
    )

    issues = validate_run.rule_source_columns_in_raw_data(bad_ctx)
    assert len(issues) == 1
    assert issues[0].rule == "source-columns-in-raw-data"
    assert "missing" in issues[0].message


def test_baseline_report_treatments_must_be_explicitly_identified() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        artifacts={
            "causal_design": {
                "causal_design": {
                    "identifiability": {
                        "identifiable_treatments": {"identified_treatment": {}},
                        "non_identifiable_treatments": {"blocked_treatment": {}},
                    },
                },
            },
            "baseline_report": {
                "intervention_results": [
                    {"treatment": "identified_treatment"},
                    {"treatment": "blocked_treatment"},
                    {"treatment": "unclassified_treatment"},
                ],
            },
        },
        artifact_paths={},
        model_indicators=None,
        raw_input_columns=None,
    )

    issues = validate_run.rule_baseline_report_treatments_identifiable(ctx)
    assert len(issues) == 1
    assert issues[0].rule == "baseline-report-treatments-identifiable"
    assert "blocked_treatment" in issues[0].message
    assert "unclassified_treatment" in issues[0].message
    assert "identified_treatment" not in issues[0].message


def test_baseline_report_treatments_fail_when_identifiability_verdicts_are_missing() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        artifacts={
            "causal_design": {"causal_design": {}},
            "baseline_report": {"intervention_results": [{"treatment": "treatment"}]},
        },
        artifact_paths={},
        model_indicators=None,
        raw_input_columns=None,
    )

    issues = validate_run.rule_baseline_report_treatments_identifiable(ctx)
    assert len(issues) == 1
    assert "no identifiability verdicts" in issues[0].message


def test_indicators_in_panel_ignores_future_measurements_artifacts() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        artifacts={
            "causal_design": {
                "causal_design": {
                    "measurement": {
                        "indicators": [{"name": "declared_indicator"}],
                    },
                },
            },
        },
        artifact_paths={},
        model_indicators=set(),
        raw_input_columns=None,
    )

    assert validate_run.rule_indicators_in_panel(ctx) == []


def test_load_run_context_respects_up_to_when_loading_measurements_artifacts(monkeypatch) -> None:
    validate_run = _load_validate_run()

    class FakeRawDataFrame:
        columns: ClassVar[list[str]] = ["timestamp", "value"]

    monkeypatch.setattr(
        validate_run,
        "_result_artifact_order",
        lambda: ("raw_data", "latent_structure", "causal_design", "measurements"),
    )

    class FakeInfo:
        version = 1

    class FakeState:
        def get(self, artifact_id: str) -> FakeInfo | None:
            present = {"raw_data", "latent_structure", "causal_design"}
            return FakeInfo() if artifact_id in present else None

    class FakeStore:
        def __init__(self, _workspace_id: str) -> None:
            pass

        def read_json_file(self, artifact_id: str, _version: int, _filename: str) -> dict[str, str]:
            return {"artifact_id": artifact_id}

        def file_path(self, artifact_id: str, _version: int, filename: str) -> str:
            return f"/tmp/ws/store/{artifact_id}/v1/{filename}"

    monkeypatch.setattr(validate_run, "derive_current_state", lambda _workspace_id: FakeState())
    monkeypatch.setattr(validate_run, "ArtifactStore", FakeStore)

    def fake_current_artifact_file(_workspace_id: str, artifact_id: str, _filename: str) -> str:
        if artifact_id == "raw_data":
            return "/tmp/raw.parquet"
        raise AssertionError(f"{artifact_id} should not be loaded past --up-to causal_design")

    monkeypatch.setattr(validate_run, "current_artifact_file", fake_current_artifact_file)
    monkeypatch.setattr(validate_run, "load_parquet", lambda _path: FakeRawDataFrame())

    ctx = validate_run.load_run_context("ws", up_to="causal_design")

    assert set(ctx.artifacts) == {"raw_data", "latent_structure", "causal_design"}
    assert ctx.model_indicators is None
    assert ctx.raw_input_columns == {"timestamp", "value"}
