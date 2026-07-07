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

from nof1_causal_lab.flows.stage_contracts import validate_stage_payload
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.graph import stage_spec
from nof1_causal_lab.machine.moves import ExecOptions, input_pins
from nof1_causal_lab.machine.runners import execute_stage_locally
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


def _run_stage(
    workspace_id: str,
    state,
    stage_id: str,
    options: ExecOptions | None = None,
):
    pins = input_pins(state, stage_spec(stage_id))
    return run_async(execute_stage_locally(workspace_id, stage_id, pins, options or ExecOptions()))


def _produced(effects, artifact_id: str):
    return next(info for info in effects.produced if info.artifact_id == artifact_id)


def _assert_contract(store: ArtifactStore, artifact_id: str, version: int, stage_id: str) -> dict:
    key_by_stage = {
        "stage-0": ("raw_data", "profile"),
        "stage-1a": ("constructs", "constructs"),
        "stage-1b": ("causal_spec", "causal_spec"),
        "stage-2": ("extraction_report", "extraction_report"),
        "stage-3": ("validation_report", "validation_report"),
        "stage-4": ("compiled_ssm", "report"),
        "stage-5b": ("posterior", "diagnostics"),
        "stage-6": ("baseline_ranking", "baseline_ranking"),
    }[stage_id]
    expected_artifact, key = key_by_stage
    assert artifact_id == expected_artifact
    payload = store.read_json_file(artifact_id, version, json_filename(artifact_id, key))
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

    effects = _run_stage(integration_workspace, fx.state_from(), "stage-0")

    assert {info.artifact_id for info in effects.produced} == {"raw_data"}
    raw_info = _produced(effects, "raw_data")
    assert raw_info.derived_from == {}
    _assert_contract(artifact_store, "raw_data", raw_info.version, "stage-0")
    raw = artifact_store.read_parquet_file(
        "raw_data", raw_info.version, parquet_filename("raw_data", "raw")
    )
    assert raw.shape == (2, 3)


def test_stage1a_reads_question_and_persists_constructs(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
) -> None:
    question = fx.seed_question(artifact_store)

    async def handler(tools: list[Any], _user_message: str) -> str:
        await tools[0](structure_json=json.dumps(fx.latent_model()))
        return ""

    install_scripted_stage_factory(handler)
    state = fx.state_from(question)

    effects = _run_stage(integration_workspace, state, "stage-1a")

    info = _produced(effects, "constructs")
    assert info.derived_from == {"question": question.version}
    payload = _assert_contract(artifact_store, "constructs", info.version, "stage-1a")
    assert payload["latent_model"]["constructs"][1]["is_outcome"] is True


def test_stage1b_reads_upstream_artifacts_and_persists_identification(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
) -> None:
    question = fx.seed_question(artifact_store)
    raw_data = fx.seed_raw_data(artifact_store)
    constructs = fx.seed_constructs(artifact_store, question_version=question.version)

    async def handler(tools: list[Any], _user_message: str) -> str:
        await tools[0](measurement_json=json.dumps(fx.measurement_model()))
        return ""

    install_scripted_stage_factory(handler)
    state = fx.state_from(question, raw_data, constructs)

    effects = _run_stage(integration_workspace, state, "stage-1b")

    produced_ids = {info.artifact_id for info in effects.produced}
    assert produced_ids == {"causal_spec", "identification_report"}
    spec_info = _produced(effects, "causal_spec")
    id_info = _produced(effects, "identification_report")
    expected_pins = {
        "question": question.version,
        "raw_data": raw_data.version,
        "constructs": constructs.version,
    }
    assert spec_info.derived_from == expected_pins
    assert id_info.derived_from == expected_pins
    _assert_contract(artifact_store, "causal_spec", spec_info.version, "stage-1b")
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
    causal_spec = fx.seed_causal_spec(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
    )
    state = fx.state_from(question, raw_data, causal_spec)

    effects = _run_stage(integration_workspace, state, "stage-2")

    produced_ids = {info.artifact_id for info in effects.produced}
    assert produced_ids == {"extraction_report", "model_data"}
    report_info = _produced(effects, "extraction_report")
    model_data_info = _produced(effects, "model_data")
    expected_pins = {
        "question": question.version,
        "raw_data": raw_data.version,
        "causal_spec": causal_spec.version,
    }
    assert report_info.derived_from == expected_pins
    assert model_data_info.derived_from == expected_pins
    _assert_contract(artifact_store, "extraction_report", report_info.version, "stage-2")
    model_data = artifact_store.read_parquet_file(
        "model_data", model_data_info.version, parquet_filename("model_data", "model_data")
    )
    assert set(model_data["indicator"].unique()) == {"sleep_score", "stress_score"}


def test_stage3_validates_seeded_model_data_through_runner(
    integration_workspace: str,
    artifact_store: ArtifactStore,
) -> None:
    causal_spec = fx.seed_causal_spec(artifact_store)
    model_data = fx.seed_model_data(artifact_store, causal_spec_version=causal_spec.version)
    state = fx.state_from(causal_spec, model_data)

    effects = _run_stage(integration_workspace, state, "stage-3")

    info = _produced(effects, "validation_report")
    assert info.derived_from == {
        "causal_spec": causal_spec.version,
        "model_data": model_data.version,
    }
    payload = _assert_contract(
        artifact_store,
        "validation_report",
        info.version,
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
    constructs = fx.seed_constructs(artifact_store, question_version=question.version)
    causal_spec = fx.seed_causal_spec(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        constructs_version=constructs.version,
    )
    identification_report = fx.seed_identification_report(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        constructs_version=constructs.version,
    )
    model_data = fx.seed_model_data(
        artifact_store,
        question_version=question.version,
        raw_data_version=raw_data.version,
        causal_spec_version=causal_spec.version,
    )
    validation_report = fx.seed_validation_report(
        artifact_store,
        causal_spec_version=causal_spec.version,
        model_data_version=model_data.version,
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
        causal_spec,
        identification_report,
        model_data,
        validation_report,
    )

    effects = _run_stage(
        integration_workspace,
        state,
        "stage-4",
        ExecOptions(enable_literature=False),
    )

    info = _produced(effects, "compiled_ssm")
    assert info.derived_from == {
        "question": question.version,
        "causal_spec": causal_spec.version,
        "identification_report": identification_report.version,
        "model_data": model_data.version,
        "validation_report": validation_report.version,
    }
    _assert_contract(artifact_store, "compiled_ssm", info.version, "stage-4")
    compiled = artifact_store.read_json_file(
        "compiled_ssm", info.version, json_filename("compiled_ssm", "compiled_ssm")
    )
    assert "spec" in compiled


def test_stage5b_persists_posterior_from_seeded_model_artifacts(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage5b import fit as stage5_fit

    compiled_ssm = fx.seed_compiled_ssm(artifact_store)
    model_data = fx.seed_model_data(artifact_store)

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
    state = fx.state_from(compiled_ssm, model_data)

    effects = _run_stage(integration_workspace, state, "stage-5b")

    info = _produced(effects, "posterior")
    assert info.derived_from == {
        "compiled_ssm": compiled_ssm.version,
        "model_data": model_data.version,
    }
    payload = _assert_contract(artifact_store, "posterior", info.version, "stage-5b")
    assert payload["inference_metadata"]["n_samples"] == 4


def test_stage6_persists_baseline_ranking_from_seeded_posterior(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    install_scripted_stage_factory,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.stages.stage6 import flow as stage6_flow

    causal_spec = fx.seed_causal_spec(artifact_store)
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
    state = fx.state_from(posterior, causal_spec, identification_report)

    effects = _run_stage(integration_workspace, state, "stage-6")

    info = _produced(effects, "baseline_ranking")
    assert info.derived_from == {
        "posterior": posterior.version,
        "causal_spec": causal_spec.version,
        "identification_report": identification_report.version,
    }
    payload = _assert_contract(artifact_store, "baseline_ranking", info.version, "stage-6")
    assert payload["intervention_results"][0]["treatment"] == "Stress"
