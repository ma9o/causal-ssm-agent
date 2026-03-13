import asyncio
import json
from types import SimpleNamespace

import cloudpickle
import polars as pl
import pytest

from causal_ssm_agent.flows import dag, pipeline
from causal_ssm_agent.utils import data as data_module


def _redirect_storage(monkeypatch, tmp_path, user_id: str = "test_user") -> None:
    """Point runs_dir and input_dir to tmp_path so tests don't touch real data/."""
    run_dir = tmp_path / "data" / user_id / "run"

    def _mock_runs_dir(c: str) -> type(run_dir):
        return tmp_path / "data" / c / "run"

    monkeypatch.setattr(dag, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(data_module, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(pipeline, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path / "data")


def _stub_config() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(override_gates=False),
        stage4_prior_elicitation=SimpleNamespace(literature_search=SimpleNamespace(enabled=True)),
    )


def _noop_artifact(**_kwargs) -> None:
    return None


def _write_public_result(tmp_path, user_id: str, stage_id: str, payload: dict) -> None:
    run_dir = tmp_path / "data" / user_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{stage_id}.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "result": json.dumps(payload),
            }
        )
    )


def _patch_common_stage_stubs(monkeypatch, calls: list):
    async def stage0(user_id: str) -> dict:
        calls.append(("stage0", user_id))
        return {
            "_df": pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}),
            "_column_descriptions": {},
        }

    def stage1b_gate(stage1a_result: dict, stage1b_result: dict, override_gates: bool) -> dict:
        calls.append(("stage1b_gate", stage1a_result, stage1b_result, override_gates))
        return {
            "treatments": stage1a_result["treatments"],
            "gate_failed": False,
            "gate_overridden": False,
            "web_outcome": "success",
            "non_identifiable": {},
        }

    async def stage2(question: str, stage0_result: dict, stage1b_result: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0_result, stage1b_result))
        raw_data = pl.DataFrame(
            {"indicator": ["stress_score"], "value": ["1.0"], "timestamp": ["2024-01-01"]}
        )
        return {"_data_for_model": raw_data, "_raw_data": raw_data}

    def stage3(stage1b_result: dict, stage2_result: dict) -> dict:
        calls.append(("stage3", stage1b_result, stage2_result))
        return {"validation_report": {}, "outcome": "success"}

    def stage4b(stage4_result: dict, stage2_result: dict, builder=None):
        calls.append(("stage4b", stage4_result, stage2_result, builder))
        return {"parametric_id": {}}

    def stage4b_gate(stage4b_result: dict, override_gates: bool) -> dict:
        calls.append(("stage4b_gate", stage4b_result, override_gates))
        return {
            "gate_failed": False,
            "gate_overridden": False,
            "outcome": "success",
            "t_rule": {},
        }

    def stage5b(
        stage4_result: dict,
        stage1b_result: dict,
        stage2_result: dict,
        inference_method: str | None,
    ) -> dict:
        calls.append(("stage5b", stage4_result, stage1b_result, stage2_result, inference_method))
        return {
            "_fitted_result": {"fitted": True},
            "_ps_result": {},
            "_ppc_result": {},
            "power_scaling": [],
            "ppc": {},
            "inference_metadata": {},
            "mcmc_diagnostics": None,
            "svi_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
            "outcome": "success",
        }

    def stage6(
        stage5_result: dict,
        stage1a_result: dict,
        stage1b_result: dict,
        stage1b_gate_result: dict,
    ) -> dict:
        calls.append(("stage6", stage5_result, stage1a_result, stage1b_result, stage1b_gate_result))
        return {"intervention_results": [], "outcome": "success"}

    def persist_web_result(stage_id: str, data: dict, user_id: str) -> dict:
        calls.append(("persist_web_result", stage_id, data, user_id))
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
        return {
            "latent_model": {"constructs": []},
            "outcome_name": "generated-outcome",
            "treatments": ["generated-treatment"],
        }

    async def stage1b(question: str, stage0_result: dict, stage1a_result: dict) -> dict:
        calls.append(("stage1b", question, stage0_result, stage1a_result))
        return {
            "causal_spec": {
                "latent": {"constructs": [], "edges": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            }
        }

    async def stage4(
        question: str, stage1b_result: dict, stage2_result: dict, enable_literature: bool
    ) -> dict:
        calls.append(("stage4", question, stage1b_result, stage2_result, enable_literature))
        return {
            "model_spec": {},
            "priors": {},
            "causal_spec": stage1b_result["causal_spec"],
        }

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage4", stage4)

    override_payload = {
        "latent_model": {"constructs": [{"name": "Overridden"}]},
        "outcome_name": "override-outcome",
        "treatments": ["override-treatment"],
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
        return {
            "latent_model": {"constructs": []},
            "outcome_name": "outcome",
            "treatments": ["treatment"],
        }

    causal_spec = {
        "latent": {"constructs": [{"name": "L"}], "edges": []},
        "measurement": {"model_clock": "1d", "indicators": [{"name": "m"}]},
    }

    async def stage1b(question: str, stage0_result: dict, stage1a_result: dict) -> dict:
        calls.append(("stage1b", question, stage0_result, stage1a_result))
        return {"causal_spec": causal_spec}

    async def stage4(
        question: str, stage1b_result: dict, stage2_result: dict, enable_literature: bool
    ) -> dict:
        raise AssertionError("stage4 should be skipped when an override is provided")

    def stage4b(stage4_result: dict, stage2_result: dict, builder=None):
        calls.append(("stage4b", stage4_result, stage2_result, builder))
        assert stage4_result["causal_spec"] == causal_spec
        return {"parametric_id": {}}

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage4", stage4)
    monkeypatch.setattr(dag, "stage4b", stage4b)

    override_payload = {
        "model_spec": {"parameters": []},
        "priors": {},
    }

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            stage_overrides={"stage-4": override_payload},
        )
    )

    assert any(entry[0] == "persist_web_result" and entry[1] == "stage-4" for entry in calls)
    assert result == {"stage5b": True, "stage6": True}


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

    user_id = "test_user"
    run_dir = tmp_path / "data" / user_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    df_path = run_dir / "stage0-raw-input.parquet"
    pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}).write_parquet(df_path)

    _write_public_result(
        tmp_path,
        user_id,
        "stage-0",
        {
            "outcome": "success",
            "source_label": "prior",
            "n_records": 1,
            "n_columns": 2,
            "date_range": {"start": "2024-01-01", "end": "2024-01-01"},
            "sample": [],
            "column_descriptions": [
                {"name": "timestamp", "dtype": "String", "description": "ts"},
                {"name": "value", "dtype": "String", "description": "val"},
            ],
        },
    )
    _write_public_result(
        tmp_path,
        user_id,
        "stage-1a",
        {
            "latent_model": {"constructs": []},
            "outcome_name": "outcome",
            "treatments": ["treatment"],
        },
    )
    _write_public_result(
        tmp_path,
        user_id,
        "stage-1b",
        {
            "outcome": "success",
            "causal_spec": {
                "latent": {"constructs": [], "edges": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        },
    )

    async def stage0(_user_id: str) -> dict:
        raise AssertionError("stage0 should be restored, not rerun")

    async def stage1a(_question: str) -> dict:
        raise AssertionError("stage1a should be restored, not rerun")

    async def stage1b(_question: str, _stage0: dict, _stage1a: dict) -> dict:
        raise AssertionError("stage1b should be restored, not rerun")

    captured: dict = {}

    async def stage2(question: str, stage0_result: dict, stage1b_result: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0_result, stage1b_result))
        captured["question"] = question
        captured["stage0_df_path"] = stage0_result["_df_path"]
        captured["stage1b_result"] = stage1b_result
        raw_data = pl.DataFrame(
            {"indicator": ["stress_score"], "value": ["1.0"], "timestamp": ["2024-01-01"]}
        )
        return {
            "_data_for_model": raw_data,
            "_raw_data": raw_data,
            "_worker_statuses": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
            "workers": [{"worker_id": 0, "status": "completed", "n_extractions": 1}],
            "combined_extractions_sample": [],
            "per_indicator_counts": {},
        }

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage2", stage2)

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            start_stage="stage-2",
            end_stage="stage-2",
        )
    )

    assert result["final_stage"] == "stage-2"
    assert result["user_id"] == user_id
    assert captured["question"] == "why is this happening?"
    assert captured["stage1b_result"]["causal_spec"]["measurement"]["indicators"] == []
    # Artifacts stay in place — df_path points to the same run dir
    assert captured["stage0_df_path"] == str(df_path)
    assert (run_dir / "stage-2-state.pkl").exists()


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
        return {
            "latent_model": {"constructs": []},
            "outcome_name": "generated-outcome",
            "treatments": ["generated-treatment"],
        }

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

    async def stage1a(_question: str) -> dict:
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

    user_id = "test_user"
    run_dir = tmp_path / "data" / user_id / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage5b-fitted-result.pkl").write_bytes(
        cloudpickle.dumps({"samples": {"x": [1, 2, 3]}})
    )
    _write_public_result(
        tmp_path,
        user_id,
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
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        },
    )

    state = dag.load_stage_state(user_id, "stage-5b")

    assert state["result"]["_fitted_result_path"].endswith("stage5b-fitted-result.pkl")
    assert state["result"]["_ps_result"]["checked"] is True
    assert state["result"]["_ps_result"]["diagnosis"] == {"beta_x": "prior_dominated"}
    assert state["result"]["_ppc_result"]["checked"] is True


def test_stage4_override_compiles_artifact_for_downstream_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("prefect.artifacts.create_markdown_artifact", _noop_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.persist_web_result",
        lambda _stage_id, data, _user_id: data,
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
        {"indicator": ["stress_score"], "value": ["1.0"], "timestamp": ["2024-01-01"]}
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
                    "constraint": "correlation",
                    "description": "AR coefficient",
                    "search_context": "stress autocorrelation",
                },
                {
                    "name": "sigma_stress_score",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "Measurement noise",
                    "search_context": "stress score measurement error",
                },
            ],
        },
        "priors": {
            "rho_stress": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
            },
            "sigma_stress_score": {
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
            },
        },
    }

    stage_state = asyncio.run(
        dag.stage4_flow.fn(
            "why is this happening?",
            {"causal_spec": causal_spec},
            {"_data_for_model_path": str(data_path)},
            True,
            "test-user_id",
            override_payload=override_payload,
        )
    )

    stage4_result = stage_state["result"]
    assert stage4_result["causal_spec"] == causal_spec
    assert stage4_result["model_info"]["model_built"] is True
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
                    "timestamp": "2024-01-01T00:00:00Z",
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
            {"causal_spec": {"measurement": {"model_clock": "1d", "indicators": []}}},
        )
    )

    assert len(stub.with_options_calls) == 1
    assert stub.with_options_calls[0]["task_runner"]._max_workers == 6
    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert result["_raw_data"].height == 1
    assert result["workers"] == [{"worker_id": 0, "status": "completed", "n_extractions": 1}]


def test_stage4_calls_subflow_directly(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "timestamp": ["2024-01-01"]}
    ).write_parquet(data_path)

    stub = _AsyncSubflowStub(
        {
            "model_spec": {"parameters": []},
            "priors": {},
            "causal_spec": {
                "latent": {"constructs": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4_orchestrated_flow", stub)

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
            enable_literature=True,
        )
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert result["model_spec"] == {"parameters": []}


def test_stage4b_calls_subflow_directly(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "timestamp": ["2024-01-01"]}
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
