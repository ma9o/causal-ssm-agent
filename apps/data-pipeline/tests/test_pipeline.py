import asyncio
import json
from types import SimpleNamespace

import cloudpickle
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.flows import dag, pipeline, stage_registry
from causal_ssm_agent.flows import run_store as run_store_module
from causal_ssm_agent.utils import data as data_module
from causal_ssm_agent.utils.causal_spec import get_all_treatments


def _redirect_storage(monkeypatch, tmp_path, workspace_id: str = "test_workspace") -> None:
    """Point runs_dir and input_dir to tmp_path so tests don't touch real data/."""
    base = str(tmp_path / "data")

    def _mock_runs_dir(c: str) -> str:
        return f"{base}/{c}/run"

    monkeypatch.setattr(run_store_module, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(data_module, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(data_module, "DATA_URI", base)
    monkeypatch.setattr(pipeline, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(pipeline, "DATA_URI", base)


def _stub_config() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(override_gates=False),
        stage4_prior_elicitation=SimpleNamespace(literature_search=SimpleNamespace(enabled=True)),
    )


def _noop_artifact(**_kwargs) -> None:
    return None


def _stage1a_latent_model(treatment: str = "treatment", outcome: str = "outcome") -> dict:
    return {
        "constructs": [
            {
                "name": treatment,
                "description": f"{treatment} construct",
                "role": "endogenous",
                "is_outcome": False,
                "temporal_status": "time_varying",
            },
            {
                "name": outcome,
                "description": f"{outcome} construct",
                "role": "endogenous",
                "is_outcome": True,
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause": treatment,
                "effect": outcome,
                "description": f"{treatment} affects {outcome}",
                "lagged": True,
            }
        ],
    }


def _write_public_result(tmp_path, workspace_id: str, stage_id: str, payload: dict) -> None:
    run_dir = tmp_path / "data" / workspace_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{stage_id}.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "result": json.dumps(payload),
            }
        )
    )


def _reset_stage_registry(monkeypatch):
    """Reset lazily-initialized stage registry so monkeypatched dag functions are picked up."""
    from causal_ssm_agent.flows import stage_registry

    monkeypatch.setattr(stage_registry, "_registry", None)
    monkeypatch.setattr(stage_registry, "_execution_order", None)


def _patch_common_stage_stubs(monkeypatch, calls: list):
    # Parameter names must match the bare computation function signatures in dag.py

    async def stage0(workspace_id: str) -> dict:
        calls.append(("stage0", workspace_id))
        return {
            "_df": pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}),
            "_column_descriptions": {},
        }

    def stage1b_gate(stage1a: dict, stage1b: dict, override_gates: bool) -> dict:
        calls.append(("stage1b_gate", stage1a, stage1b, override_gates))
        return {
            "treatments": get_all_treatments(stage1a["latent_model"]),
            "gate_failed": False,
            "gate_overridden": False,
            "web_outcome": "success",
            "non_identifiable": {},
        }

    async def stage2(question: str, stage0: dict, stage1b: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0, stage1b))
        raw_data = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["1.0"],
                "anchor_time": ["2024-01-01"],
                "support_start": ["2024-01-01"],
                "support_end": ["2024-01-01"],
            }
        )
        return {"_data_for_model": raw_data, "_raw_data": raw_data}

    def stage3(stage1b: dict, stage2: dict) -> dict:
        calls.append(("stage3", stage1b, stage2))
        return {
            "is_valid": True,
            "indicators": {},
            "dataset_issues": [],
            "outcome": "success",
        }

    def stage4b(stage4: dict, stage2: dict, ssm_builder=None):
        calls.append(("stage4b", stage4, stage2, ssm_builder))
        return {"parametric_id": {}}

    def stage4b_gate(stage4b: dict, override_gates: bool) -> dict:
        calls.append(("stage4b_gate", stage4b, override_gates))
        return {
            "gate_failed": False,
            "gate_overridden": False,
            "outcome": "success",
            "t_rule": {},
        }

    def stage5b(
        stage4: dict,
        stage2: dict,
        inference_method: str | None,
    ) -> dict:
        calls.append(("stage5b", stage4, stage2, inference_method))
        return {
            "_fitted_artifact": None,
            "power_scaling": [],
            "ppc": {},
            "inference_metadata": {},
            "mcmc_diagnostics": None,
            "svi_diagnostics": None,
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
            "outcome": "success",
        }

    def stage6(
        stage5b: dict,
        stage1a: dict,
        stage1b: dict,
        stage1b_gate: dict,
        question: str | None = None,
    ) -> dict:
        calls.append(("stage6", stage5b, stage1a, stage1b, stage1b_gate, question))
        return {"intervention_results": [], "outcome": "success"}

    def persist_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
        calls.append(("persist_web_result", stage_id, data, workspace_id))
        if stage_id == "stage-5b":
            return {"stage5b": True}
        if stage_id == "stage-6":
            return {"stage6": True}
        return data

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1b_gate", stage1b_gate)
    monkeypatch.setattr(dag, "stage2", stage2)
    monkeypatch.setattr(dag, "stage3", stage3)
    monkeypatch.setattr(dag, "stage4b", stage4b)
    monkeypatch.setattr(dag, "stage4b_gate", stage4b_gate)
    monkeypatch.setattr(dag, "stage5b", stage5b)
    monkeypatch.setattr(dag, "stage6", stage6)
    monkeypatch.setattr("causal_ssm_agent.flows.stages.persist_web_result", persist_web_result)
    _reset_stage_registry(monkeypatch)


def test_production_registry_offloads_stage4_to_modal(monkeypatch):
    pytest.importorskip("modal")
    from causal_ssm_agent.flows import modal_runners

    _reset_stage_registry(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")

    async def fake_stage4_runner(**kwargs):
        return kwargs

    monkeypatch.setattr(modal_runners, "modal_stage4_runner", fake_stage4_runner)

    registry = stage_registry.get_stage_registry()

    assert registry["stage-4"].runner is fake_stage4_runner


def test_build_main_deployment_enforces_schema_and_serial_concurrency():
    deployment = pipeline.build_main_deployment()

    assert deployment.name == "causal-inference"
    assert deployment.enforce_parameter_schema is True
    assert deployment.concurrency_limit == 1
    assert deployment.concurrency_options.collision_strategy.value == "ENQUEUE"


def test_stage1a_override_skips_recomputation_and_replays_downstream(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model("generated-treatment", "generated-outcome")}

    async def stage1b(question: str, stage0: dict, stage1a: dict) -> dict:
        calls.append(("stage1b", question, stage0, stage1a))
        return {
            "causal_spec": {
                "latent": {"constructs": [], "edges": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            }
        }

    async def stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
    ) -> dict:
        calls.append(("stage4", question, stage1b, stage2, stage3, enable_literature))
        return {
            "model_spec": {},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": stage1b["causal_spec"],
        }

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage4", stage4)
    _reset_stage_registry(monkeypatch)

    override_payload = {
        "latent_model": _stage1a_latent_model("override-treatment", "override-outcome"),
    }

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            stage_overrides={"stage-1a": override_payload},
        )
    )

    assert ("stage1a", "why is this happening?") not in calls
    stage1b_calls = [entry for entry in calls if entry[0] == "stage1b"]
    assert len(stage1b_calls) == 1
    assert stage1b_calls[0][3] == override_payload
    assert any(
        entry[0] == "persist_web_result" and entry[1] == "stage-1a" and entry[2] == override_payload
        for entry in calls
    )
    assert result == {"stage5b": True, "stage6": True}


def test_stage4_override_preserves_replay_contract_for_downstream_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model()}

    causal_spec = {
        "latent": {"constructs": [{"name": "L"}], "edges": []},
        "measurement": {"model_clock": "1d", "indicators": [{"name": "m"}]},
    }

    async def stage1b(question: str, stage0: dict, stage1a: dict) -> dict:
        calls.append(("stage1b", question, stage0, stage1a))
        return {"causal_spec": causal_spec}

    async def stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
    ) -> dict:
        raise AssertionError("stage4 should be skipped when an override is provided")

    def stage4b(stage4: dict, stage2: dict, ssm_builder=None):
        calls.append(("stage4b", stage4, stage2, ssm_builder))
        assert stage4["causal_spec"] == causal_spec
        return {"parametric_id": {}}

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage4", stage4)
    monkeypatch.setattr(dag, "stage4b", stage4b)
    _reset_stage_registry(monkeypatch)

    override_payload = {
        "model_spec": {"parameters": []},
        "authored_priors": {},
        "resolved_priors": [],
    }

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            stage_overrides={"stage-4": override_payload},
        )
    )

    assert any(entry[0] == "persist_web_result" and entry[1] == "stage-4" for entry in calls)
    assert result == {"stage5b": True, "stage6": True}


def test_stage6_runs_interventions_from_fitted_artifact(monkeypatch):
    from types import SimpleNamespace

    from causal_ssm_agent.models.ssm.inference import FittedArtifact

    # Build a minimal FittedArtifact with mock result and builder
    mock_samples = {"latent": np.array([[0.1, 0.2]])}
    mock_result = SimpleNamespace(get_samples=lambda: mock_samples)
    mock_spec = SimpleNamespace(
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=[],
    )
    mock_builder = SimpleNamespace(_spec=mock_spec)

    fitted_artifact = FittedArtifact(
        result=mock_result,
        builder=mock_builder,
        times=np.array([0.0, 1.0]),
        ppc_result={"checked": True, "per_variable_warnings": []},
        power_scaling_result={"checked": True, "diagnosis": {}},
    )
    stage5b_result = {
        "_fitted_result_path": "unused.pkl",
    }

    captured: dict[str, object] = {}

    monkeypatch.setattr(dag, "load_pickle", lambda _path: fitted_artifact)
    monkeypatch.setattr("prefect.artifacts.create_table_artifact", lambda **_kwargs: None)

    def fake_compute_interventions(**kwargs):
        captured.update(kwargs)
        return [{"treatment": "screen_time", "effect_size": 1.0, "identifiable": True}]

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.counterfactual.compute_interventions",
        fake_compute_interventions,
    )

    result = asyncio.run(
        dag.stage6(
            stage5b_result,
            {"latent_model": _stage1a_latent_model("screen_time", "sleep_quality")},
            {"causal_spec": {}},
            {"treatments": ["screen_time"]},
        )
    )

    assert result["intervention_results"][0]["treatment"] == "screen_time"
    assert captured["treatments"] == ["screen_time"]
    assert captured["outcome"] == "sleep_quality"
    assert captured["latent_names"] == ["screen_time", "sleep_quality"]


def test_stage3_awaits_async_validation_artifact(monkeypatch, tmp_path):
    raw_path = tmp_path / "stage2-raw-data.parquet"
    model_path = tmp_path / "stage2-model-data.parquet"
    raw_data = pl.DataFrame(
        {
            "indicator": ["stress_score"],
            "value": ["1.0"],
            "anchor_time": ["2024-01-01"],
        }
    )
    raw_data.write_parquet(raw_path)
    raw_data.write_parquet(model_path)

    captured: dict[str, object] = {"awaited": False}

    async def fake_create_table_artifact(**kwargs):
        captured["awaited"] = True
        captured["table"] = kwargs["table"]

    monkeypatch.setattr("prefect.artifacts.create_table_artifact", fake_create_table_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.validate_extraction",
        lambda *_args, **_kwargs: {
            "is_valid": True,
            "indicators": {
                "stress_score": {
                    "validation": {
                        "issues": [
                            {
                                "indicator": "stress_score",
                                "issue_type": "outlier",
                                "severity": "warning",
                                "message": "Outlier detected",
                            }
                        ]
                    }
                }
            },
            "dataset_issues": [],
        },
    )

    result = asyncio.run(
        dag.stage3(
            {
                "causal_spec": {
                    "measurement": {"model_clock": "1d", "indicators": [{"name": "stress_score"}]}
                }
            },
            {
                "_raw_data_path": str(raw_path),
                "_data_for_model_path": str(model_path),
            },
        )
    )

    assert result["outcome"] == "warn"
    assert captured["awaited"] is True
    assert captured["table"] == [
        {
            "indicator": "stress_score",
            "type": "outlier",
            "severity": "warning",
            "message": "Outlier detected",
        }
    ]


def test_resume_from_stage2_loads_existing_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    workspace_id = "test_workspace"
    run_dir = tmp_path / "data" / workspace_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    df_path = run_dir / "stage0-raw-input.parquet"
    pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}).write_parquet(df_path)

    _write_public_result(
        tmp_path,
        workspace_id,
        "stage-0",
        {
            "outcome": "success",
            "column_descriptions": [
                {"name": "timestamp", "description": "ts"},
                {"name": "value", "description": "val"},
            ],
        },
    )
    _write_public_result(
        tmp_path,
        workspace_id,
        "stage-1a",
        {"latent_model": _stage1a_latent_model()},
    )
    _write_public_result(
        tmp_path,
        workspace_id,
        "stage-1b",
        {
            "outcome": "success",
            "causal_spec": {
                "latent": {"constructs": [], "edges": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        },
    )

    async def stage0(_workspace_id: str) -> dict:
        raise AssertionError("stage0 should be restored, not rerun")

    async def stage1a(_question: str) -> dict:
        raise AssertionError("stage1a should be restored, not rerun")

    async def stage1b(_question: str, _stage0: dict, _stage1a: dict) -> dict:
        raise AssertionError("stage1b should be restored, not rerun")

    captured: dict = {}

    async def stage2(question: str, stage0: dict, stage1b: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0, stage1b))
        captured["question"] = question
        captured["stage0_df_path"] = stage0["_df_path"]
        captured["stage1b_result"] = stage1b
        raw_data = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["1.0"],
                "anchor_time": ["2024-01-01"],
                "support_start": ["2024-01-01"],
                "support_end": ["2024-01-01"],
            }
        )
        return {
            "_data_for_model": raw_data,
            "_raw_data": raw_data,
            "_worker_statuses": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
            "workers": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
        }

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage2", stage2)
    _reset_stage_registry(monkeypatch)

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            start_stage="stage-2",
            end_stage="stage-2",
        )
    )

    assert result["final_stage"] == "stage-2"
    assert result["workspace_id"] == workspace_id
    assert captured["question"] == "why is this happening?"
    assert captured["stage1b_result"]["causal_spec"]["measurement"]["indicators"] == []
    # Artifacts stay in place — df_path points to the same run dir
    assert captured["stage0_df_path"] == str(df_path)
    assert (run_dir / "stage-2-state.pkl").exists()


def test_load_stage2_snapshot_rehydrates_current_run_artifact_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)

    workspace_id = "test_workspace"
    run_dir = tmp_path / "data" / workspace_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "stage2-raw-data.parquet"
    model_path = run_dir / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(raw_path)
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(model_path)

    web_payload = {
        "outcome": "success",
        "workers": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
    }
    _write_public_result(tmp_path, workspace_id, "stage-2", web_payload)
    run_store_module.save_stage_snapshot(
        "stage-2",
        {
            "result": {
                "_raw_data_path": "/dead/run/stage2-raw-data.parquet",
                "_data_for_model_path": "/dead/run/stage2-model-data.parquet",
                "workers": [{"worker_id": 999, "status": "stale", "n_extractions": 0}],
                "preserved_field": "kept-from-snapshot",
            },
            "web": web_payload,
        },
        workspace_id,
    )

    state = stage_registry.load_stage_state(workspace_id, "stage-2")

    assert state["result"]["_raw_data_path"] == str(raw_path)
    assert state["result"]["_data_for_model_path"] == str(model_path)
    assert state["result"]["workers"] == web_payload["workers"]
    assert state["result"]["_worker_statuses"] == web_payload["workers"]
    assert state["result"]["preserved_field"] == "kept-from-snapshot"


def test_pipeline_emits_stage_progress_events(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    emitted: list[tuple[str, dict, dict | None]] = []
    monkeypatch.setattr(
        pipeline,
        "emit_event",
        lambda event, resource, payload=None, **_kwargs: emitted.append((event, resource, payload)),
    )

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model("generated-treatment", "generated-outcome")}

    monkeypatch.setattr(dag, "stage1a", stage1a)

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            end_stage="stage-1a",
        )
    )

    assert result["final_stage"] == "stage-1a"
    assert [(event, payload["stage_id"], payload["status"]) for event, _, payload in emitted] == [
        ("causal-ssm.pipeline-stage.running", "stage-0", "running"),
        ("causal-ssm.pipeline-stage.completed", "stage-0", "completed"),
        ("causal-ssm.pipeline-stage.running", "stage-1a", "running"),
        ("causal-ssm.pipeline-stage.completed", "stage-1a", "completed"),
    ]
    assert all(
        resource["prefect.resource.id"].startswith("prefect.flow-run.")
        for _, resource, _ in emitted
    )


def test_pipeline_emits_failed_stage_event(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        pipeline,
        "emit_event",
        lambda event, resource, **_kwargs: emitted.append((event, resource)),
    )

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(dag, "stage1a", stage1a)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            pipeline.causal_inference_pipeline(
                query="why is this happening?",
                end_stage="stage-1a",
            )
        )

    assert [event for event, _ in emitted] == [
        "causal-ssm.pipeline-stage.running",
        "causal-ssm.pipeline-stage.completed",
        "causal-ssm.pipeline-stage.running",
        "causal-ssm.pipeline-stage.failed",
    ]


def test_load_stage5b_state_reconstructs_from_public_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)

    workspace_id = "test_workspace"
    run_dir = tmp_path / "data" / workspace_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage5b-fitted-result.pkl").write_bytes(
        cloudpickle.dumps({"samples": {"x": [1, 2, 3]}})
    )
    _write_public_result(
        tmp_path,
        workspace_id,
        "stage-5b",
        {
            "outcome": "warn",
            "power_scaling": [
                {
                    "parameter": "beta_x",
                    "diagnosis": "prior_dominated",
                    "prior_sensitivity": 0.8,
                    "likelihood_sensitivity": 0.2,
                    "psis_k_hat": 0.4,
                }
            ],
            "ppc": {"checked": True, "per_variable_warnings": [{"variable": "y", "message": "m"}]},
            "inference_metadata": {"method": "svi"},
            "mcmc_diagnostics": None,
            "svi_diagnostics": {"loss": [1.0]},
            "smc_diagnostics": {"beta_schedule": [0.1, 1.0]},
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        },
    )

    state = stage_registry.load_stage_state(workspace_id, "stage-5b")

    assert state["result"]["_fitted_result_path"].endswith("stage5b-fitted-result.pkl")
    assert state["result"]["power_scaling"][0]["diagnosis"] == "prior_dominated"
    assert state["result"]["ppc"]["checked"] is True
    assert state["result"]["smc_diagnostics"] == {"beta_schedule": [0.1, 1.0]}


def test_stage4_override_compiles_artifact_for_downstream_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("prefect.artifacts.create_markdown_artifact", _noop_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.persist_web_result",
        lambda _stage_id, data, _workspace_id: data,
    )

    causal_spec = {
        "latent": {
            "constructs": [
                {
                    "name": "stress",
                    "role": "endogenous",
                    "description": "Stress state",
                    "temporal_status": "time_varying",
                }
            ],
            "edges": [],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "construct_name": "stress",
                    "how_to_measure": "Stress rating",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
        },
    }
    data_for_model = pl.DataFrame(
        {
            "indicator": ["stress_score"] * 5,
            "value": ["1.0", "2.0", "3.0", "4.0", "5.0"],
            "anchor_time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
        }
    )
    data_path = tmp_path / "stage2-data.parquet"
    data_for_model.write_parquet(data_path)

    override_payload = {
        "model_spec": {
            "likelihoods": [
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "continuous stress rating",
                }
            ],
            "parameters": [
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                },
                {
                    "name": "sigma_stress",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "Measurement noise",
                },
            ],
        },
        "authored_priors": {
            "rho_stress": {
                "parameter": "rho_stress",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "reasoning": "Reasonable persistence prior",
                "sources": [],
            },
            "sigma_stress": {
                "parameter": "sigma_stress",
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
                "reasoning": "Positive measurement noise prior",
                "sources": [],
            },
        },
        "causal_spec": {"measurement": {"indicators": [{"name": "stale"}]}},
        "model_info": {"model_built": False, "error": "stale compile result"},
        "_compiled_ssm": {"manifest_names": ["stale"]},
    }

    ctx = stage_registry.PipelineContext(
        workspace_id="test-workspace_id",
        prefect_run_id="test-run-id",
        question="why is this happening?",
        gates_overridden=False,
        lit_enabled=True,
        inference_method=None,
        supported_overrides={"stage-4": override_payload},
        is_byok=False,
    )
    stage_state = asyncio.run(
        stage_registry.run_stage_flow(
            stage_registry.get_stage_registry()["stage-4"],
            ctx,
            {
                "stage-1b": {"result": {"causal_spec": causal_spec}},
                "stage-2": {"result": {"_data_for_model_path": str(data_path)}},
                "stage-3": {"result": {"indicators": {}, "dataset_issues": [], "is_valid": True}},
            },
        )
    )

    stage4_result = stage_state["result"]
    assert stage4_result["causal_spec"] == causal_spec
    assert stage4_result["model_info"]["model_built"] is True
    assert stage4_result["model_info"] != override_payload["model_info"]
    assert stage4_result["_compiled_ssm"] != override_payload["_compiled_ssm"]
    assert "_compiled_ssm" in stage4_result


class _AsyncSubflowStub:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []
        self.fn_calls: list[tuple[tuple, dict]] = []
        self.with_options_calls: list[dict] = []

    def with_options(self, **kwargs):
        self.with_options_calls.append(kwargs)
        return self

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    async def fn(self, *args, **kwargs):
        self.fn_calls.append((args, kwargs))
        raise AssertionError("subflow should be invoked directly, not via .fn")


class _SyncSubflowStub:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []
        self.fn_calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def fn(self, *args, **kwargs):
        self.fn_calls.append((args, kwargs))
        raise AssertionError("subflow should be invoked directly, not via .fn")


def test_stage2_calls_subflow_directly(monkeypatch, tmp_path):
    stub = _AsyncSubflowStub(
        {
            "raw_data": [
                {
                    "indicator": "stress_score",
                    "value": "1.0",
                    "anchor_time": "2024-01-02T00:00:00",
                    "support_kind": "interval",
                    "summary_operator": "mean",
                    "anchor_policy": "support_end",
                    "observation_window": "1d",
                    "support_start": "2024-01-01T00:00:00",
                    "support_end": "2024-01-02T00:00:00",
                }
            ],
            "worker_statuses": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
            "n_total_extractions": 1,
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )
    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            {"_df_path": str(tmp_path / "input.parquet")},
            {
                "causal_spec": {
                    "measurement": {
                        "model_clock": "1d",
                        "indicators": [
                            {
                                "name": "stress_score",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            }
                        ],
                    }
                }
            },
        )
    )

    assert len(stub.with_options_calls) == 1
    assert stub.with_options_calls[0]["task_runner"]._max_workers == 6
    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert result["_raw_data"].height == 1
    assert result["_raw_data"]["support_kind"][0] == "interval"
    assert result["_raw_data"]["summary_operator"][0] == "mean"
    assert result["_raw_data"]["anchor_policy"][0] == "support_end"
    assert result["_raw_data"]["anchor_time"][0] == "2024-01-02T00:00:00"
    assert result["_raw_data"]["support_start"][0] == "2024-01-01T00:00:00"
    assert result["_raw_data"]["support_end"][0] == "2024-01-02T00:00:00"
    assert result["workers"] == [{"worker_id": 0, "status": "completed", "n_extractions": 1}]


def test_stage2_preserves_null_values_for_inference(monkeypatch, tmp_path):
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
    from causal_ssm_agent.utils.data import pivot_to_wide

    stub = _AsyncSubflowStub(
        {
            "raw_data": [
                {
                    "indicator": "daytime_screen_events",
                    "value": "5",
                    "anchor_time": "2024-01-01T00:00:00",
                    "support_start": "2024-01-01T00:00:00",
                    "support_end": "2024-01-01T00:00:00",
                },
                {
                    "indicator": "last_evening_activity_hour",
                    "value": None,
                    "anchor_time": "2024-01-01T00:00:00",
                    "support_start": "2024-01-01T00:00:00",
                    "support_end": "2024-01-01T00:00:00",
                },
            ],
            "worker_statuses": [{"worker_id": 0, "status": "completed", "n_extractions": 2}],
            "n_total_extractions": 2,
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            {"_df_path": str(tmp_path / "input.parquet")},
            {
                "causal_spec": {
                    "measurement": {
                        "model_clock": "1d",
                        "indicators": [
                            {
                                "name": "daytime_screen_events",
                                "measurement_dtype": "count",
                            },
                            {
                                "name": "last_evening_activity_hour",
                                "measurement_dtype": "continuous",
                            },
                        ],
                    }
                }
            },
        )
    )

    data_for_model = result["_data_for_model"]
    assert data_for_model.height == 2
    assert (
        data_for_model.filter(pl.col("indicator") == "last_evening_activity_hour")[
            "value"
        ].null_count()
        == 1
    )

    observations, _times, manifest_names = SSMModelBuilder(
        sampler_config={"method": "auto"}
    ).prepare_fit_inputs(pivot_to_wide(data_for_model))
    assert manifest_names == ["daytime_screen_events", "last_evening_activity_hour"]
    assert jnp.isclose(observations[0, 0], 5.0)
    assert jnp.isnan(observations[0, 1])


def test_stage2_keeps_semantic_rows_in_model_data(monkeypatch, tmp_path):
    stub = _AsyncSubflowStub(
        {
            "raw_data": [
                {
                    "indicator": "stress_score",
                    "value": "4.0",
                    "anchor_time": "2024-01-02T00:00:00",
                    "support_kind": "interval",
                    "summary_operator": "mean",
                    "anchor_policy": "support_end",
                    "observation_window": "1d",
                    "support_start": "2024-01-01T00:00:00",
                    "support_end": "2024-01-02T00:00:00",
                },
                {
                    "indicator": "closing_mood",
                    "value": "1",
                    "anchor_time": "2024-01-02T00:00:00",
                    "support_kind": "point",
                    "summary_operator": "last",
                    "anchor_policy": "support_end",
                    "observation_window": "1d",
                    "support_start": "2024-01-01T00:00:00",
                    "support_end": "2024-01-02T00:00:00",
                },
            ],
            "worker_statuses": [{"worker_id": 0, "status": "completed", "n_extractions": 2}],
            "n_total_extractions": 2,
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            {"_df_path": str(tmp_path / "input.parquet")},
            {
                "causal_spec": {
                    "measurement": {
                        "model_clock": "1d",
                        "indicators": [
                            {
                                "name": "stress_score",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            {
                                "name": "closing_mood",
                                "measurement_dtype": "ordinal",
                                "aggregation": "last",
                                "ordinal_levels": ["bad", "good"],
                            },
                        ],
                    }
                }
            },
        )
    )

    data_for_model = result["_data_for_model"].sort("indicator")
    assert data_for_model.height == 2
    assert data_for_model["indicator"].to_list() == ["closing_mood", "stress_score"]
    assert data_for_model["support_kind"].to_list() == ["point", "interval"]
    assert data_for_model["summary_operator"].to_list() == ["last", "mean"]
    assert data_for_model["anchor_policy"].to_list() == ["support_end", "support_end"]
    assert str(data_for_model["anchor_time"][0]) == "2024-01-02 00:00:00"
    assert str(data_for_model["support_start"][0]) == "2024-01-01 00:00:00"
    assert str(data_for_model["support_end"][0]) == "2024-01-02 00:00:00"
    assert data_for_model.filter(pl.col("indicator") == "closing_mood")["value"][0] == 1.0
    assert data_for_model.filter(pl.col("indicator") == "stress_score")["value"][0] == 4.0


def test_stage4_calls_subflow_directly(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(data_path)

    stub = _AsyncSubflowStub(
        {
            "model_spec": {"parameters": []},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": {
                "latent": {"constructs": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4_agentic_flow", stub)

    result = asyncio.run(
        dag.stage4(
            "why is this happening?",
            {
                "causal_spec": {
                    "latent": {"constructs": []},
                    "measurement": {"model_clock": "1d", "indicators": []},
                }
            },
            {"_data_for_model_path": str(data_path)},
            {"indicators": {}, "dataset_issues": [], "is_valid": True},
            enable_literature=True,
        )
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert result["model_spec"] == {"parameters": []}


def test_stage4b_calls_subflow_directly(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(data_path)

    stub = _SyncSubflowStub({"parametric_id": {"checked": True}})
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4b_parametric_id_flow", stub)

    result = dag.stage4b(
        {"model_spec": {"parameters": []}},
        {"_data_for_model_path": str(data_path)},
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert result == {"parametric_id": {"checked": True}}
