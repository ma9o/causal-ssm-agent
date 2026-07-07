"""Fixture-backed integration tests for production stage runners."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import re
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import polars as pl

from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
from nof1_causal_lab.flows.stage_contracts import validate_stage_payload
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import ExecOptions, input_pins
from nof1_causal_lab.machine.runners import execute_transition_locally
from nof1_causal_lab.utils import data as data_module
from tests.helpers import run_async
from tests.integration import stage_runner_fixtures as fx

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


class _LocalSandbox:
    """Tiny local stand-in for the ingestion code sandbox."""

    def __init__(self, extract_dir: Path) -> None:
        self._extract_dir = extract_dir

    def execute(self, code: str) -> tuple[str, pl.DataFrame | None]:
        ns = {
            "__builtins__": __builtins__,
            "pl": pl,
            "polars": pl,
            "csv": csv,
            "json": json,
            "Path": Path,
            "datetime": dt,
            "re": re,
            "math": math,
            "io": io,
            "DATA_DIR": str(self._extract_dir),
        }
        try:
            exec(code, ns)
        except Exception:  # noqa: BLE001 - mirrors the production tool surface.
            return f"Execution error:\n{traceback.format_exc()}", None

        result_df = ns.get("result_df")
        if not isinstance(result_df, pl.DataFrame):
            return "No Polars result_df produced.", None
        if result_df.is_empty():
            return "Warning: result_df is empty.", None
        return "Success", result_df


class _LocalSandboxContext:
    def __init__(self, extract_dir: Path, **_kwargs: Any) -> None:
        self._sandbox = _LocalSandbox(extract_dir)

    def __enter__(self) -> _LocalSandbox:
        return self._sandbox

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def _run_transition(
    workspace_id: str,
    state,
    artifact_id: str,
    options: ExecOptions | None = None,
):
    spec = transition_spec(artifact_id)
    pins = input_pins(state, spec)
    return run_async(
        execute_transition_locally(
            workspace_id,
            artifact_id,
            pins,
            state,
            options or ExecOptions(),
        )
    )


def _produced(effects, artifact_id: str):
    return next(info for info in effects.produced if info.artifact_id == artifact_id)


def _assert_contract(store: ArtifactStore, artifact_id: str, version: int, stage_id: str) -> dict:
    key_by_stage = {
        "stage-0": ("raw_data", "profile"),
        "stage-1a": ("latent_structure", "latent_structure"),
        "stage-1b": ("measurement_structure", "measurement_structure"),
        "stage-2": ("measurements", "measurements"),
        "stage-3": ("validation_report", "validation_report"),
        "stage-4": ("statistical_model_spec", "statistical_model_spec"),
        "stage-5b": ("posterior", "diagnostics"),
        "stage-6": ("baseline_report", "baseline_report"),
    }[stage_id]
    expected_artifact, key = key_by_stage
    assert artifact_id == expected_artifact
    payload = store.read_json_file(artifact_id, version, json_filename(artifact_id, key))
    if stage_id == "stage-1b":
        MeasurementStructure.model_validate(payload["measurement_structure"])
        return payload
    validate_stage_payload(stage_id, payload)
    return payload


_ACCEPT_ALL_STAGE4_SOFT_CHECKS = {
    "C1b confinement": "fixture",
    "C2 latent scale": "fixture",
    "C3 resolvability": "fixture",
    "C4b edge overwhelm": "fixture",
    "C4c saturation": "fixture",
    "C5b width": "fixture",
    "C5c transmission": "fixture",
}


def _normal_prior(parameter: str, mu: float, sigma: float) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "distribution": "Normal",
        "params": {"mu": mu, "sigma": sigma},
        "sources": [],
        "reasoning": "Fixture prior.",
    }


def _halfnormal_prior(parameter: str, sigma: float) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "distribution": "HalfNormal",
        "params": {"sigma": sigma},
        "sources": [],
        "reasoning": "Fixture prior.",
    }


def _stage4_construct_payload(construct: str) -> dict[str, Any]:
    payloads = {
        "Stress": {
            "construct": "Stress",
            "indicators": [{"variable": "stress_score", "family": "gaussian", "link": "identity"}],
            "priors": {
                "rho_Stress": _normal_prior("rho_Stress", 0.5, 0.2),
                "sigma_Stress": _halfnormal_prior("sigma_Stress", 0.5),
                "manifest_mean_stress_score": _normal_prior("manifest_mean_stress_score", 0.0, 2.0),
            },
            "accept": _ACCEPT_ALL_STAGE4_SOFT_CHECKS,
        },
        "Sleep": {
            "construct": "Sleep",
            "indicators": [{"variable": "sleep_score", "family": "gaussian", "link": "identity"}],
            "priors": {
                "rho_Sleep": _normal_prior("rho_Sleep", 0.5, 0.2),
                "sigma_Sleep": _halfnormal_prior("sigma_Sleep", 0.5),
                "beta_Stress_Sleep": _normal_prior("beta_Stress_Sleep", -0.2, 0.1),
                "manifest_mean_sleep_score": _normal_prior("manifest_mean_sleep_score", 0.0, 2.0),
            },
            "accept": _ACCEPT_ALL_STAGE4_SOFT_CHECKS,
        },
    }
    return payloads[construct]


def test_stage0_ingests_uploaded_file_through_runner(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage0 import flow as stage0_flow

    input_path = Path(data_module.input_dir(integration_workspace))
    input_path.mkdir(parents=True)
    (input_path / "observations.csv").write_text(
        "timestamp,stress_score,sleep_score\n2024-01-01T08:00:00,2,8\n2024-01-02T08:00:00,4,6\n"
    )
    monkeypatch.setattr(stage0_flow, "ModalCodeSandbox", _LocalSandboxContext)

    async def handler(tools: list[Any], _user_message: str) -> str:
        tool_map = {tool.name: tool for tool in tools}
        await tool_map["list_files"](path=".")
        await tool_map["read_file_sample"](path="observations.csv", n_lines=3)
        await tool_map["execute_python"](
            code=(
                'result_df = pl.read_csv(Path(DATA_DIR) / "observations.csv")\n'
                'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
            )
        )
        await tool_map["submit_table"](
            column_descriptions_json=json.dumps(
                {
                    "timestamp": "Observation timestamp.",
                    "stress_score": "Daily stress score.",
                    "sleep_score": "Daily sleep score.",
                }
            )
        )
        return ""

    install_scripted_stage_factory(handler)

    effects = _run_transition(integration_workspace, fx.state_from(), "raw_data")

    assert {info.artifact_id for info in effects.produced} == {"raw_data"}
    raw_info = _produced(effects, "raw_data")
    assert raw_info.derived_from == {}
    _assert_contract(artifact_store, "raw_data", raw_info.version, "stage-0")
    raw = artifact_store.read_parquet_file(
        "raw_data", raw_info.version, parquet_filename("raw_data", "raw")
    )
    assert raw.shape == (2, 3)


def test_stage1a_reads_question_and_persists_latent_structure(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
) -> None:
    question = fx.seed_question(artifact_store)

    async def handler(tools: list[Any], _user_message: str) -> str:
        await tools[0](structure_json=json.dumps(fx.latent_structure()))
        return ""

    install_scripted_stage_factory(handler)
    state = fx.state_from(question)

    effects = _run_transition(integration_workspace, state, "latent_structure")

    info = _produced(effects, "latent_structure")
    assert info.derived_from == {"question": question.version}
    payload = _assert_contract(artifact_store, "latent_structure", info.version, "stage-1a")
    assert payload["latent_structure"]["constructs"][1]["is_outcome"] is True


def test_stage1b_reads_upstream_artifacts_and_persists_measurement_derivations(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
) -> None:
    question = fx.seed_question(artifact_store)
    raw_data = fx.seed_raw_data(artifact_store)
    latent_structure = fx.seed_latent_structure(artifact_store, question_version=question.version)

    async def handler(tools: list[Any], _user_message: str) -> str:
        await tools[0](measurement_json=json.dumps(fx.measurement_structure()))
        return ""

    install_scripted_stage_factory(handler)
    state = fx.state_from(question, raw_data, latent_structure)

    effects = _run_transition(integration_workspace, state, "measurement_structure")

    produced_ids = {info.artifact_id for info in effects.produced}
    assert produced_ids == {
        "measurement_structure",
        "causal_design",
        "identification_report",
    }
    measurement_info = _produced(effects, "measurement_structure")
    spec_info = _produced(effects, "causal_design")
    id_info = _produced(effects, "identification_report")
    expected_pins = {
        "question": question.version,
        "raw_data": raw_data.version,
        "latent_structure": latent_structure.version,
    }
    assert measurement_info.derived_from == expected_pins
    assert spec_info.derived_from == {
        "latent_structure": latent_structure.version,
        "measurement_structure": measurement_info.version,
    }
    assert id_info.derived_from == {"causal_design": spec_info.version}
    _assert_contract(
        artifact_store,
        "measurement_structure",
        measurement_info.version,
        "stage-1b",
    )
    id_payload = artifact_store.read_json_file(
        "identification_report",
        id_info.version,
        json_filename("identification_report", "identification_report"),
    )
    assert id_payload["estimable_treatments"] == ["Stress"]


def test_stage2_runs_computed_extraction_from_seeded_artifacts(
    integration_workspace: str,
    artifact_store: ArtifactStore,
) -> None:
    question = fx.seed_question(artifact_store)
    raw_data = fx.seed_raw_data(artifact_store)
    latent_structure = fx.seed_latent_structure(artifact_store, question_version=question.version)
    measurement_structure = fx.seed_measurement_structure(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        latent_structure_version=latent_structure.version,
    )
    causal_design = fx.seed_causal_design(
        artifact_store,
        latent_structure_version=latent_structure.version,
        measurement_structure_version=measurement_structure.version,
    )
    state = fx.state_from(
        question, raw_data, latent_structure, measurement_structure, causal_design
    )

    effects = _run_transition(integration_workspace, state, "measurements")

    produced_ids = {info.artifact_id for info in effects.produced}
    assert produced_ids == {"measurements", "panel", "validation_report"}
    report_info = _produced(effects, "measurements")
    panel_info = _produced(effects, "panel")
    validation_info = _produced(effects, "validation_report")
    expected_pins = {
        "question": question.version,
        "raw_data": raw_data.version,
        "measurement_structure": measurement_structure.version,
    }
    assert report_info.derived_from == expected_pins
    assert panel_info.derived_from == expected_pins
    assert validation_info.derived_from == {
        "causal_design": causal_design.version,
        "panel": panel_info.version,
    }
    _assert_contract(artifact_store, "measurements", report_info.version, "stage-2")
    panel = artifact_store.read_parquet_file(
        "panel", panel_info.version, parquet_filename("panel", "panel")
    )
    assert set(panel["indicator"].unique()) == {"sleep_score", "stress_score"}
    payload = _assert_contract(
        artifact_store,
        "validation_report",
        validation_info.version,
        "stage-3",
    )
    assert set(payload["indicators"]) == {"sleep_score", "stress_score"}


def test_stage4_persists_compiled_ssm_from_seeded_artifacts(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage4.agentic import (
        stage4_construct_flow,
    )

    question = fx.seed_question(artifact_store)
    raw_data = fx.seed_raw_data(artifact_store)
    latent_structure = fx.seed_latent_structure(artifact_store, question_version=question.version)
    measurement_structure = fx.seed_measurement_structure(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        latent_structure_version=latent_structure.version,
    )
    causal_design = fx.seed_causal_design(
        artifact_store,
        latent_structure_version=latent_structure.version,
        measurement_structure_version=measurement_structure.version,
    )
    identification_report = fx.seed_identification_report(
        artifact_store,
        causal_design_version=causal_design.version,
    )
    panel = fx.seed_panel(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        measurement_structure_version=measurement_structure.version,
    )
    validation_report = fx.seed_validation_report(
        artifact_store,
        causal_design_version=causal_design.version,
        panel_version=panel.version,
    )

    original_build = stage4_construct_flow.run_stage4_construct_build

    async def fast_construct_build(**kwargs: Any):
        kwargs["n_draws"] = 8
        return await original_build(**kwargs)

    monkeypatch.setattr(stage4_construct_flow, "run_stage4_construct_build", fast_construct_build)

    async def handler(tools: list[Any], user_message: str) -> str:
        match = re.search(r"Active construct: `([^`]+)`", user_message)
        assert match is not None
        payload = _stage4_construct_payload(match.group(1))
        await {tool.name: tool for tool in tools}["submit_construct"].execute(**payload)
        return ""

    install_scripted_stage_factory(handler)
    state = fx.state_from(
        question,
        raw_data,
        latent_structure,
        measurement_structure,
        causal_design,
        identification_report,
        panel,
        validation_report,
    )

    effects = _run_transition(
        integration_workspace,
        state,
        "statistical_model_spec",
        ExecOptions(enable_literature=False),
    )

    produced_ids = {info.artifact_id for info in effects.produced}
    assert produced_ids == {"statistical_model_spec", "compiled_ssm"}
    spec_info = _produced(effects, "statistical_model_spec")
    compiled_info = _produced(effects, "compiled_ssm")
    assert spec_info.derived_from == {
        "question": question.version,
        "causal_design": causal_design.version,
        "identification_report": identification_report.version,
        "panel": panel.version,
        "validation_report": validation_report.version,
    }
    assert compiled_info.derived_from == {
        "statistical_model_spec": spec_info.version,
        "causal_design": causal_design.version,
    }
    _assert_contract(artifact_store, "statistical_model_spec", spec_info.version, "stage-4")
    compiled = artifact_store.read_json_file(
        "compiled_ssm",
        compiled_info.version,
        json_filename("compiled_ssm", "compiled_ssm"),
    )
    assert "spec" in compiled


def test_stage5b_persists_posterior_from_seeded_model_artifacts(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage5b import fit as stage5_fit

    compiled_ssm = fx.seed_compiled_ssm(artifact_store)
    panel = fx.seed_panel(artifact_store)

    def fake_fit_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "fitted": True,
            "inference_type": "marginal_particle_gibbs",
            "n_samples": 4,
            "duration_seconds": 0.01,
            "result": None,
            "spec": None,
            "runtime": SimpleNamespace(observation_support=None),
            "times": [0.0, 1.0],
            "mcmc_diagnostics": None,
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        }

    monkeypatch.setattr(stage5_fit, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        stage5_fit,
        "run_ppc",
        lambda _fitted: {
            "per_variable_warnings": [],
            "checked": True,
            "overlays": [],
            "test_stats": [],
        },
    )
    state = fx.state_from(compiled_ssm, panel)

    effects = _run_transition(integration_workspace, state, "posterior")

    info = _produced(effects, "posterior")
    assert info.derived_from == {
        "compiled_ssm": compiled_ssm.version,
        "panel": panel.version,
    }
    payload = _assert_contract(artifact_store, "posterior", info.version, "stage-5b")
    assert payload["inference_metadata"]["n_samples"] == 4


def test_stage6_persists_baseline_report_from_seeded_posterior(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage6 import flow as stage6_flow

    causal_design = fx.seed_causal_design(artifact_store)
    identification_report = fx.seed_identification_report(artifact_store)
    posterior = fx.seed_posterior(artifact_store, fitted_artifact={"fixture": "fit"})

    monkeypatch.setattr(
        stage6_flow,
        "run_interventions",
        lambda *_args, **_kwargs: [
            {"treatment": "Stress", "posterior_draws": [0.1, 0.2, 0.0, 0.3]}
        ],
    )

    async def handler(_tools: list[Any], _user_message: str) -> str:
        return "Stress has the strongest positive estimated effect."

    install_scripted_stage_factory(handler)
    state = fx.state_from(posterior, causal_design, identification_report)

    effects = _run_transition(integration_workspace, state, "baseline_report")

    info = _produced(effects, "baseline_report")
    assert info.derived_from == {
        "posterior": posterior.version,
        "causal_design": causal_design.version,
        "identification_report": identification_report.version,
    }
    payload = _assert_contract(artifact_store, "baseline_report", info.version, "stage-6")
    assert payload["intervention_results"][0]["treatment"] == "Stress"
