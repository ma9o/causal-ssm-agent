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
        stages={
            "stage-0": {"column_descriptions": []},
            "stage-1b": {
                "causal_spec": {
                    "measurement": {
                        "indicators": [
                            {"name": "observed_value", "source_columns": ["value"]},
                        ],
                    },
                },
            },
        },
        stage_paths={},
        model_indicators=None,
        raw_input_columns={"timestamp", "value"},
    )

    assert validate_run.rule_stage0_columns_match_raw_parquet(ctx) == []
    assert validate_run.rule_source_columns_in_stage0(ctx) == []

    bad_ctx = validate_run.RunContext(
        workspace_id="ws",
        stages={
            **ctx.stages,
            "stage-1b": {
                "causal_spec": {
                    "measurement": {
                        "indicators": [
                            {"name": "missing_value", "source_columns": ["missing"]},
                        ],
                    },
                },
            },
        },
        stage_paths={},
        model_indicators=None,
        raw_input_columns={"timestamp", "value"},
    )

    issues = validate_run.rule_source_columns_in_stage0(bad_ctx)
    assert len(issues) == 1
    assert issues[0].rule == "source-columns-in-stage0"
    assert "missing" in issues[0].message


def test_stage6_treatments_must_be_explicitly_identified() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        stages={
            "stage-1b": {
                "causal_spec": {
                    "identifiability": {
                        "identifiable_treatments": {"identified_treatment": {}},
                        "non_identifiable_treatments": {"blocked_treatment": {}},
                    },
                },
            },
            "stage-6": {
                "intervention_results": [
                    {"treatment": "identified_treatment"},
                    {"treatment": "blocked_treatment"},
                    {"treatment": "unclassified_treatment"},
                ],
            },
        },
        stage_paths={},
        model_indicators=None,
        raw_input_columns=None,
    )

    issues = validate_run.rule_stage6_treatments_identifiable(ctx)
    assert len(issues) == 1
    assert issues[0].rule == "stage6-treatments-identifiable"
    assert "blocked_treatment" in issues[0].message
    assert "unclassified_treatment" in issues[0].message
    assert "identified_treatment" not in issues[0].message


def test_stage6_treatments_fail_when_identifiability_verdicts_are_missing() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        stages={
            "stage-1b": {"causal_spec": {}},
            "stage-6": {"intervention_results": [{"treatment": "treatment"}]},
        },
        stage_paths={},
        model_indicators=None,
        raw_input_columns=None,
    )

    issues = validate_run.rule_stage6_treatments_identifiable(ctx)
    assert len(issues) == 1
    assert "no identifiability verdicts" in issues[0].message


def test_indicators_in_model_data_ignores_future_stage2_artifacts() -> None:
    validate_run = _load_validate_run()
    ctx = validate_run.RunContext(
        workspace_id="ws",
        stages={
            "stage-1b": {
                "causal_spec": {
                    "measurement": {
                        "indicators": [{"name": "declared_indicator"}],
                    },
                },
            },
        },
        stage_paths={},
        model_indicators=set(),
        raw_input_columns=None,
    )

    assert validate_run.rule_indicators_in_model_data(ctx) == []


def test_load_run_context_respects_up_to_when_loading_stage2_artifacts(monkeypatch) -> None:
    validate_run = _load_validate_run()

    class FakeRawDataFrame:
        columns: ClassVar[list[str]] = ["timestamp", "value"]

    monkeypatch.setattr(
        validate_run,
        "topological_stage_order",
        lambda: ("stage-0", "stage-1a", "stage-1b", "stage-2"),
    )
    monkeypatch.setattr(validate_run, "runs_dir", lambda workspace_id: f"/tmp/{workspace_id}/run")
    monkeypatch.setattr(
        validate_run.storage,
        "exists",
        lambda path: path.endswith(("stage-0.json", "stage-1a.json", "stage-1b.json")),
    )
    monkeypatch.setattr(
        validate_run,
        "load_public_payload",
        lambda _workspace_id, stage_id: {"stage_id": stage_id},
    )

    def fake_current_artifact_file(_workspace_id: str, artifact_id: str, _filename: str) -> str:
        if artifact_id == "raw_data":
            return "/tmp/raw.parquet"
        raise AssertionError(
            f"{artifact_id} should not be loaded past --up-to stage-1b"
        )

    monkeypatch.setattr(validate_run, "current_artifact_file", fake_current_artifact_file)
    monkeypatch.setattr(validate_run, "load_parquet", lambda _path: FakeRawDataFrame())

    ctx = validate_run.load_run_context("ws", up_to="stage-1b")

    assert set(ctx.stages) == {"stage-0", "stage-1a", "stage-1b"}
    assert ctx.model_indicators is None
    assert ctx.raw_input_columns == {"timestamp", "value"}
