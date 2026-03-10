import asyncio
from types import SimpleNamespace

import polars as pl

from causal_ssm_agent.flows import dag, pipeline


def _stub_config() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(override_gates=False),
        stage4_prior_elicitation=SimpleNamespace(literature_search=SimpleNamespace(enabled=True)),
    )


def _noop_artifact(**_kwargs) -> None:
    return None


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

    async def stage2(question: str, stage0_result: dict, stage1b_result: dict) -> dict:
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

    def stage5(
        stage4_result: dict,
        stage1b_result: dict,
        stage2_result: dict,
        inference_method: str | None,
    ) -> dict:
        calls.append(("stage5", stage4_result, stage1b_result, stage2_result, inference_method))
        return {
            "_fitted_result": {"fitted": True},
            "ps_result": {},
            "ppc_result": {},
            "ps_list": [],
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

    def persist_web_result(stage_id: str, data: dict, run_id: str) -> dict:
        calls.append(("persist_web_result", stage_id, data, run_id))
        if stage_id == "stage-5":
            return {"stage5": True}
        if stage_id == "stage-6":
            return {"stage6": True}
        return data

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1b_gate", stage1b_gate)
    monkeypatch.setattr(dag, "stage2", stage2)
    monkeypatch.setattr(dag, "stage3", stage3)
    monkeypatch.setattr(dag, "stage4b", stage4b)
    monkeypatch.setattr(dag, "stage4b_gate", stage4b_gate)
    monkeypatch.setattr(dag, "stage5", stage5)
    monkeypatch.setattr(dag, "stage6", stage6)
    monkeypatch.setattr("causal_ssm_agent.flows.stages.persist_web_result", persist_web_result)


def test_stage1a_override_skips_recomputation_and_replays_downstream(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
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
                "measurement": {"indicators": []},
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
    assert result == {"stage5": True, "stage6": True}


def test_stage4_override_preserves_replay_contract_for_downstream_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
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
        "measurement": {"indicators": [{"name": "m"}]},
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
    assert result == {"stage5": True, "stage6": True}


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
    monkeypatch.setattr(
        "causal_ssm_agent.utils.aggregations.aggregate_worker_measurements",
        lambda worker_dfs, _causal_spec: {"daily": worker_dfs[0]},
    )
    monkeypatch.setattr(
        "causal_ssm_agent.utils.aggregations.flatten_aggregated_data",
        lambda aggregated_result: aggregated_result["daily"],
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            {"_df_path": str(tmp_path / "input.parquet")},
            {"causal_spec": {"measurement": {"indicators": []}}},
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
            "causal_spec": {"latent": {"constructs": []}, "measurement": {"indicators": []}},
        }
    )
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4_orchestrated_flow", stub)

    result = asyncio.run(
        dag.stage4(
            "why is this happening?",
            {"causal_spec": {"latent": {"constructs": []}, "measurement": {"indicators": []}}},
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
