import asyncio
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import cloudpickle
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.flows import dag, pipeline, stage_registry
from causal_ssm_agent.flows import run_store as run_store_module
from causal_ssm_agent.utils import data as data_module
from causal_ssm_agent.utils import openrouter_client


class _FakeModalRunnersModule(ModuleType):
    modal_stage4_runner: Any
    modal_stage5b_runner: Any
    persist_noop: Any


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


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

    async def stage0(workspace_id: str, openrouter_api_key: str | None = None) -> dict:
        calls.append(("stage0", workspace_id))
        return {
            "_df": pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}),
            "_column_descriptions": {},
        }

    async def stage2(question: str, stage0: dict, stage1b: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0, stage1b))
        data_for_model = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["1.0"],
                "anchor_time": ["2024-01-01"],
                "support_start": ["2024-01-01"],
                "support_end": ["2024-01-01"],
            }
        )
        return {"_data_for_model": data_for_model}

    def stage3(stage1b: dict, stage2: dict) -> dict:
        calls.append(("stage3", stage1b, stage2))
        return {
            "is_valid": True,
            "indicators": {},
            "dataset_issues": [],
            "outcome": "success",
        }

    def stage4b(stage4: dict, stage2: dict, ssm_builder=None, root_run_id: str | None = None):
        calls.append(("stage4b", stage4, stage2, ssm_builder, root_run_id))
        return {"parametric_id": {}}

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

    def stage5a(stage4: dict, stage2: dict) -> dict:
        calls.append(("stage5a", stage4, stage2))
        return {"outcome": "success"}

    def stage6(
        stage5b: dict,
        stage1b: dict,
        question: str | None = None,
    ) -> dict:
        calls.append(("stage6", stage5b, stage1b, question))
        return {"intervention_results": [], "outcome": "success"}

    def persist_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
        calls.append(("persist_web_result", stage_id, data, workspace_id))
        if stage_id == "stage-5b":
            return {"stage5b": True}
        if stage_id == "stage-6":
            return {"stage6": True}
        return data

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage2", stage2)
    monkeypatch.setattr(dag, "stage3", stage3)
    monkeypatch.setattr(dag, "stage4b", stage4b)
    monkeypatch.setattr(dag, "stage5a", stage5a)
    monkeypatch.setattr(dag, "stage5b", stage5b)
    monkeypatch.setattr(dag, "stage6", stage6)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stage_persistence.persist_web_result", persist_web_result
    )
    _reset_stage_registry(monkeypatch)


def test_production_registry_routes_stage4_by_access_mode(monkeypatch):
    _reset_stage_registry(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")

    async def fake_stage4_runner(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
        workspace_id: str,
        root_run_id: str | None = None,
    ) -> dict:
        return {
            "runner": "modal",
            "workspace_id": workspace_id,
            "openrouter_api_key": openrouter_client.get_openrouter_api_key(),
            "root_run_id": root_run_id,
        }

    async def fake_local_stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
        workspace_id: str | None = None,
        root_run_id: str | None = None,
    ) -> dict:
        return {
            "runner": "local",
            "workspace_id": workspace_id,
            "openrouter_api_key": openrouter_client.get_openrouter_api_key(),
            "root_run_id": root_run_id,
        }

    fake_modal_runners = _FakeModalRunnersModule("causal_ssm_agent.flows.modal_runners")
    fake_modal_runners.modal_stage4_runner = fake_stage4_runner
    fake_modal_runners.modal_stage5b_runner = lambda *_args, **_kwargs: None
    fake_modal_runners.persist_noop = lambda _result, _workspace_id: None
    monkeypatch.setitem(sys.modules, "causal_ssm_agent.flows.modal_runners", fake_modal_runners)
    monkeypatch.setattr(dag, "stage4", fake_local_stage4)

    registry = stage_registry.get_stage_registry()

    with openrouter_client.use_openrouter_api_key("user-key"):
        user_result = asyncio.run(
            _resolve_maybe_awaitable(
                registry["stage-4"].runner(
                    question="why",
                    stage1b={},
                    stage2={},
                    stage3={},
                    enable_literature=True,
                    workspace_id="workspace-user",
                    openrouter_access_mode="user",
                    root_run_id="root-run-user",
                )
            )
        )
        local_result = asyncio.run(
            _resolve_maybe_awaitable(
                registry["stage-4"].runner(
                    question="why",
                    stage1b={},
                    stage2={},
                    stage3={},
                    enable_literature=True,
                    workspace_id="workspace-local",
                    openrouter_access_mode="local",
                    root_run_id="root-run-local",
                )
            )
        )
    modal_result = asyncio.run(
        _resolve_maybe_awaitable(
            registry["stage-4"].runner(
                question="why",
                stage1b={},
                stage2={},
                stage3={},
                enable_literature=True,
                workspace_id="workspace-modal",
                openrouter_access_mode=None,
                root_run_id="root-run-modal",
            )
        )
    )

    assert user_result == {
        "runner": "modal",
        "workspace_id": "workspace-user",
        "openrouter_api_key": "user-key",
        "root_run_id": "root-run-user",
    }
    assert local_result == {
        "runner": "local",
        "workspace_id": "workspace-local",
        "openrouter_api_key": "user-key",
        "root_run_id": "root-run-local",
    }
    assert modal_result == {
        "runner": "modal",
        "workspace_id": "workspace-modal",
        "openrouter_api_key": None,
        "root_run_id": "root-run-modal",
    }


def test_stage2_binding_uses_access_mode_for_free_window_limit(monkeypatch):
    from causal_ssm_agent.utils.config import get_config

    MAX_FREE_WINDOWS = get_config().stage2_workers.max_free_windows

    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    states = {
        "stage-0": {"result": {}},
        "stage-1b": {"result": {}},
    }
    user_ctx = stage_registry.PipelineContext(
        workspace_id="workspace-user",
        prefect_run_id="run-user",
        question="why",
        lit_enabled=True,
        inference_method=None,
        supported_overrides={},
        openrouter_api_key="user-key",
        openrouter_access_mode="user",
    )
    anonymous_ctx = stage_registry.PipelineContext(
        workspace_id="workspace-anonymous",
        prefect_run_id="run-anonymous",
        question="why",
        lit_enabled=True,
        inference_method=None,
        supported_overrides={},
        openrouter_api_key="anonymous-key",
        openrouter_access_mode="anonymous",
    )
    local_ctx = stage_registry.PipelineContext(
        workspace_id="workspace-local",
        prefect_run_id="run-local",
        question="why",
        lit_enabled=True,
        inference_method=None,
        supported_overrides={},
        openrouter_api_key=None,
        openrouter_access_mode="local",
    )

    user_inputs = stage_registry._bind_stage2(user_ctx, states)
    anonymous_inputs = stage_registry._bind_stage2(anonymous_ctx, states)
    local_inputs = stage_registry._bind_stage2(local_ctx, states)

    assert user_inputs["max_windows"] is None
    assert anonymous_inputs["max_windows"] == MAX_FREE_WINDOWS
    assert local_inputs["max_windows"] is None


def test_interactive_overrideable_stages_declare_materialization_policy():
    from causal_ssm_agent.flows.stage_contracts import INTERACTIVE_STAGES

    registry = stage_registry.get_stage_registry()

    for stage_id in INTERACTIVE_STAGES:
        defn = registry[stage_id]
        if not defn.override_eligible:
            continue
        assert defn.override_adapter is not None


def test_run_stage_flow_rejects_override_without_materialization_policy():
    contract = stage_registry.get_stage_registry()["stage-1a"].contract
    defn = stage_registry.StageDefinition(
        stage_id="stage-test",
        depends_on=frozenset(),
        contract=contract,
        bind_inputs=lambda _ctx, _states: {},
        runner=lambda: {"latent_model": _stage1a_latent_model()},
        override_eligible=True,
    )
    ctx = stage_registry.PipelineContext(
        workspace_id="workspace",
        prefect_run_id="run",
        question="why",
        lit_enabled=True,
        inference_method=None,
        supported_overrides={"stage-test": {"latent_model": _stage1a_latent_model()}},
        openrouter_api_key=None,
        openrouter_access_mode=None,
    )

    with pytest.raises(ValueError, match="explicit materialization policy"):
        asyncio.run(stage_registry.run_stage_flow(defn, ctx, {}))


def test_run_stage_flow_emits_stage4_initial_replay_state_before_runner(monkeypatch):
    events: list[tuple[str, object] | tuple[str, object, object]] = []

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_runtime_projections.project_stage4_initial_state",
        lambda _causal_spec: (
            {"nodes": [{"id": "indicator:x"}], "edges": [], "phases": []},
            {
                "cursor": {"kind": "block", "block_id": "indicator:x"},
                "block_status": {"indicator:x": "pending"},
                "model_spec_locked": False,
                "repair_campaign": None,
                "phase": "model_decisions",
            },
        ),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.runtime_events.emit_stage4_graph_event",
        lambda root_run_id, *, graph: events.append(("graph", root_run_id, graph)),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.runtime_events.emit_stage4_snapshot_event",
        lambda root_run_id, *, snapshot: events.append(("snapshot", root_run_id, snapshot)),
    )
    monkeypatch.setattr(
        stage_registry,
        "finalize_stage",
        lambda stage_id, result, workspace_id, extras=None, contract=None: {
            "stage_id": stage_id,
            "workspace_id": workspace_id,
            "result": result,
            "extras": extras,
            "contract": contract,
            "web": {"outcome": "success"},
        },
    )

    async def _runner(**_inputs):
        events.append(("runner", _inputs["root_run_id"]))
        return {"ok": True}

    defn = stage_registry.StageDefinition(
        stage_id="stage-4",
        depends_on=frozenset(),
        contract=stage_registry.get_stage_registry()["stage-4"].contract,
        bind_inputs=lambda _ctx, _states: {
            "question": "why",
            "stage1b": {
                "causal_spec": {
                    "latent": {"constructs": []},
                    "measurement": {"model_clock": "1d", "indicators": []},
                    "estimation": {"state_order": [], "edges": [], "induced_dependencies": []},
                }
            },
            "stage2": {"_data_for_model_path": "ignored"},
            "stage3": {"indicators": {}, "dataset_issues": [], "is_valid": True},
            "enable_literature": True,
            "workspace_id": "workspace",
            "root_run_id": "root-run-123",
        },
        runner=_runner,
    )
    ctx = stage_registry.PipelineContext(
        workspace_id="workspace",
        prefect_run_id="root-run-123",
        question="why",
        lit_enabled=True,
        inference_method=None,
        supported_overrides={},
        openrouter_api_key=None,
        openrouter_access_mode=None,
    )

    stage_state = asyncio.run(stage_registry.run_stage_flow(defn, ctx, {}))

    assert stage_state["result"] == {"ok": True}
    assert events == [
        ("graph", "root-run-123", {"nodes": [{"id": "indicator:x"}], "edges": [], "phases": []}),
        (
            "snapshot",
            "root-run-123",
            {
                "cursor": {"kind": "block", "block_id": "indicator:x"},
                "block_status": {"indicator:x": "pending"},
                "model_spec_locked": False,
                "repair_campaign": None,
                "phase": "model_decisions",
            },
        ),
        ("runner", "root-run-123"),
    ]


def _stub_stage0_result():
    return {
        "outcome": "success",
        "source_label": "stub",
        "n_records": 1,
        "n_columns": 2,
        "date_range": {"start": "2024-01-01", "end": "2024-01-01"},
        "sample": [],
        "column_descriptions": [
            {"name": "timestamp", "description": "ts"},
            {"name": "value", "description": "val"},
        ],
        "_df": pl.DataFrame({"timestamp": ["2024-01-01"], "value": ["1"]}),
        "_column_descriptions": {},
    }


def _stub_stage1a_result():
    return {
        "latent_model": {
            "constructs": [
                {
                    "name": "travel",
                    "description": "Travel exposure",
                    "role": "exogenous",
                    "is_outcome": False,
                    "temporal_status": "time_varying",
                },
                {
                    "name": "sleep_quality",
                    "description": "Observed sleep quality",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause": "travel",
                    "effect": "sleep_quality",
                    "description": "Travel affects sleep quality",
                    "lagged": True,
                }
            ],
        },
        "outcome_name": "sleep_quality",
        "treatments": ["travel"],
    }


@pytest.mark.parametrize(
    ("access_mode", "expected_key", "extra_setup"),
    [
        pytest.param("user", "user-key", "byok", id="user-byok"),
        pytest.param("local", "local-key", "env", id="local-env"),
    ],
)
def test_pipeline_threads_openrouter_key_by_access_mode(
    monkeypatch,
    tmp_path,
    access_mode,
    expected_key,
    extra_setup,
):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("causal_ssm_agent.utils.config.get_config", _stub_config)
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    pipeline_kwargs: dict = {
        "query": "why is this happening?",
        "end_stage": "stage-1a",
        "openrouter_access_mode": access_mode,
    }

    if extra_setup == "byok":
        monkeypatch.setattr(
            pipeline,
            "consume_byok_secret_ref",
            lambda ref: "user-key" if ref == "ref-123" else None,
        )
        pipeline_kwargs["openrouter_secret_ref"] = "ref-123"
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", "local-key")

    seen: list[tuple[str, str | None]] = []

    async def stage0(workspace_id: str) -> dict:
        seen.append(("stage0", openrouter_client.get_openrouter_api_key()))
        return _stub_stage0_result()

    async def stage1a(question: str) -> dict:
        seen.append(("stage1a", openrouter_client.get_openrouter_api_key()))
        return _stub_stage1a_result()

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1a", stage1a)
    _reset_stage_registry(monkeypatch)

    result = asyncio.run(pipeline.causal_inference_pipeline.fn(**pipeline_kwargs))

    assert result["final_stage"] == "stage-1a"
    assert seen == [("stage0", expected_key), ("stage1a", expected_key)]


@pytest.mark.parametrize("access_mode", [None, "local"])
def test_production_pipeline_requires_explicit_production_access_mode(monkeypatch, access_mode):
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")

    with pytest.raises(
        ValueError,
        match="Production runs must set openrouter_access_mode to 'anonymous' or 'user'",
    ):
        asyncio.run(
            pipeline.causal_inference_pipeline(
                query="why is this happening?",
                end_stage="stage-1a",
                openrouter_access_mode=access_mode,
            )
        )


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

    async def stage1a(question: str, openrouter_api_key: str | None = None) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model("generated-treatment", "generated-outcome")}

    async def stage1b(
        question: str,
        stage0: dict,
        stage1a: dict,
        openrouter_api_key: str | None = None,
    ) -> dict:
        calls.append(("stage1b", question, stage0, stage1a))
        return {
            "causal_spec": {
                "latent": stage1a["latent_model"],
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "treatment_score",
                            "construct_name": "override-treatment",
                            "how_to_measure": "Measure override-treatment",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "construct_polarity": "positive",
                        }
                    ],
                },
                "estimation": {
                    "state_order": ["override-treatment", "override-outcome"],
                    "edges": [
                        {
                            "cause": "override-treatment",
                            "effect": "override-outcome",
                            "description": ("override-treatment affects override-outcome"),
                            "lagged": True,
                        }
                    ],
                    "induced_dependencies": [],
                },
            }
        }

    async def stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
        workspace_id: str | None = None,
        openrouter_api_key: str | None = None,
        root_run_id: str | None = None,
    ) -> dict:
        calls.append(
            (
                "stage4",
                question,
                stage1b,
                stage2,
                stage3,
                enable_literature,
                workspace_id,
                root_run_id,
            )
        )
        return {
            "model_spec": {},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": stage1b["causal_spec"],
            "_compiled_ssm": {},
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


def test_pipeline_stops_cleanly_on_completed_fail_outcome(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str, openrouter_api_key: str | None = None) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model()}

    async def stage1b(
        question: str,
        stage0: dict,
        stage1a: dict,
        openrouter_api_key: str | None = None,
    ) -> dict:
        calls.append(("stage1b", question, stage0, stage1a))
        return {
            "causal_spec": {
                "latent": stage1a["latent_model"],
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "stress_score",
                            "construct_name": "treatment",
                            "how_to_measure": "Measure treatment",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
            }
        }

    def stage3(stage1b: dict, stage2: dict) -> dict:
        calls.append(("stage3", stage1b, stage2))
        return {
            "is_valid": False,
            "indicators": {},
            "dataset_issues": [
                {
                    "issue_type": "no_numeric",
                    "severity": "error",
                    "message": "No numeric observations survived validation.",
                }
            ],
            "outcome": "fail",
            "fail_reason": "data_validation_failed",
        }

    async def stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
        workspace_id: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> dict:
        raise AssertionError("stage4 should not run after a terminal stage outcome")

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage1b", stage1b)
    monkeypatch.setattr(dag, "stage3", stage3)
    monkeypatch.setattr(dag, "stage4", stage4)
    _reset_stage_registry(monkeypatch)

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
        )
    )

    assert result["final_stage"] == "stage-3"
    assert result["stage"]["outcome"] == "fail"
    assert result["stage"]["fail_reason"] == "data_validation_failed"
    assert not any(entry[0] == "stage4" for entry in calls)


def test_pipeline_materializes_stage1b_override_before_stage6(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str, openrouter_api_key: str | None = None) -> dict:
        calls.append(("stage1a", question))
        return {"latent_model": _stage1a_latent_model("override_treatment", "override_outcome")}

    async def stage4(
        question: str,
        stage1b: dict,
        stage2: dict,
        stage3: dict,
        enable_literature: bool,
        workspace_id: str,
        root_run_id: str | None = None,
    ) -> dict:
        calls.append(("stage4", question, stage1b, stage2, stage3, enable_literature, workspace_id))
        return {
            "model_spec": {},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": stage1b["causal_spec"],
            "_compiled_ssm": {},
        }

    monkeypatch.setattr(dag, "stage1a", stage1a)
    monkeypatch.setattr(dag, "stage4", stage4)
    _reset_stage_registry(monkeypatch)

    override_payload = {
        "causal_spec": {
            "latent": _stage1a_latent_model("override_treatment", "override_outcome"),
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "construct_name": "override_treatment",
                        "how_to_measure": "Measure override_treatment",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "construct_polarity": "positive",
                    }
                ],
            },
            "estimation": {
                "state_order": ["override_treatment", "override_outcome"],
                "edges": [
                    {
                        "cause": "override_treatment",
                        "effect": "override_outcome",
                        "description": "override_treatment affects override_outcome",
                        "lagged": True,
                    }
                ],
                "induced_dependencies": [],
            },
        }
    }

    result = asyncio.run(
        pipeline.causal_inference_pipeline(
            query="why is this happening?",
            stage_overrides={"stage-1b": override_payload},
        )
    )

    stage6_calls = [entry for entry in calls if entry[0] == "stage6"]
    assert len(stage6_calls) == 1
    materialized_stage1b = stage6_calls[0][2]
    assert materialized_stage1b["_identified_treatments"] == ["override_treatment"]
    assert materialized_stage1b["outcome"] == "success"
    assert result == {"stage5b": True, "stage6": True}


def test_stage6_runs_interventions_from_fitted_artifact(monkeypatch):
    from types import SimpleNamespace

    from causal_ssm_agent.models.ssm.inference import FittedArtifact

    # Build a minimal FittedArtifact with mock result and builder
    mock_spec = SimpleNamespace(
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=[],
    )
    mock_builder = SimpleNamespace(_spec=mock_spec)

    fitted_artifact = FittedArtifact(
        result=None,
        builder=mock_builder,
        times=np.array([0.0, 1.0]),
        ppc_result={"checked": True, "per_variable_warnings": []},
        power_scaling_result={"checked": True, "diagnosis": {}},
    )
    stage5b_result = {
        "_fitted_result_path": "unused.pkl",
    }

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage6.flow.load_pickle",
        lambda _path: fitted_artifact,
    )
    monkeypatch.setattr("prefect.artifacts.create_table_artifact", lambda **_kwargs: None)

    class _FakeLLMStageContext:
        def __init__(self, *_args, **_kwargs):
            self.trace_capture = {"trace": None}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def make_generate(self, _model: str, **_kwargs):
            async def _generate(messages: list[dict], label: str | None = None):
                captured["commentary_messages"] = messages
                captured["commentary_label"] = label
                self.trace_capture["trace"] = SimpleNamespace(
                    messages=[SimpleNamespace(role="assistant", content="stubbed summary")]
                )
                return {"content": "stubbed"}

            return _generate

        def finalize(self, result: dict) -> dict:
            return result

    monkeypatch.setattr("causal_ssm_agent.utils.llm.LLMStageContext", _FakeLLMStageContext)

    def fake_compute_interventions(**kwargs):
        captured.update(kwargs)
        return [{"treatment": "screen_time", "posterior_draws": [0.9, 1.0, 1.1]}]

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.counterfactual.compute_interventions",
        fake_compute_interventions,
    )

    latent_model = _stage1a_latent_model("screen_time", "sleep_quality")
    result = asyncio.run(
        dag.stage6(
            stage5b_result,
            {
                "causal_spec": {"latent": latent_model},
                "_identified_treatments": ["screen_time"],
            },
        )
    )

    assert result["intervention_results"][0]["treatment"] == "screen_time"
    assert captured["treatments"] == ["screen_time"]
    assert captured["outcome"] == "sleep_quality"
    assert captured["latent_names"] == ["screen_time", "sleep_quality"]
    assert captured["commentary_label"] == "comment-results"


def test_stage3_awaits_async_validation_artifact(monkeypatch, tmp_path):
    model_path = tmp_path / "stage2-model-data.parquet"
    data_for_model = pl.DataFrame(
        {
            "indicator": ["stress_score"],
            "value": ["1.0"],
            "anchor_time": ["2024-01-01"],
        }
    )
    data_for_model.write_parquet(model_path)

    captured: dict[str, object] = {"awaited": False}

    async def fake_create_table_artifact(**kwargs):
        captured["awaited"] = True
        captured["table"] = kwargs["table"]

    monkeypatch.setattr("prefect.artifacts.create_table_artifact", fake_create_table_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage3.flow.validate_extraction",
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


def test_stage3_normalizes_global_status_from_local_issue_severity(monkeypatch, tmp_path):
    model_path = tmp_path / "stage2-model-data.parquet"
    data_for_model = pl.DataFrame(
        {
            "indicator": ["stress_score"],
            "value": ["1.0"],
            "anchor_time": ["2024-01-01"],
        }
    )
    data_for_model.write_parquet(model_path)

    async def fake_create_table_artifact(**_kwargs):
        return None

    monkeypatch.setattr("prefect.artifacts.create_table_artifact", fake_create_table_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage3.flow.validate_extraction",
        lambda *_args, **_kwargs: {
            "is_valid": False,
            "indicators": {
                "stress_score": {
                    "validation": {
                        "issues": [
                            {
                                "indicator": "stress_score",
                                "issue_type": "low_n",
                                "severity": "warning",
                                "message": "Only 1 observation remains.",
                            }
                        ],
                        "checks": {},
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
                "_data_for_model_path": str(model_path),
            },
        )
    )

    assert result["is_valid"] is True
    assert result["outcome"] == "warn"
    assert result["fail_reason"] is None


def test_stage1b_filters_stage6_targets_to_estimable_states(monkeypatch):
    latent_model = {
        "constructs": [
            {
                "name": "screen_time",
                "description": "Screen time",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
            {
                "name": "age",
                "description": "Age",
                "role": "exogenous",
                "temporal_status": "time_invariant",
            },
            {
                "name": "sleep",
                "description": "Sleep quality",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
        ],
        "edges": [
            {"cause": "screen_time", "effect": "sleep", "description": "Screen time affects sleep"},
            {"cause": "age", "effect": "sleep", "description": "Age affects sleep"},
        ],
    }
    causal_spec = {
        "latent": latent_model,
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "daily_event_count",
                    "construct_name": "screen_time",
                    "how_to_measure": "Measure screen time",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_issue_searches",
                    "construct_name": "sleep",
                    "how_to_measure": "Measure sleep",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        },
        "identifiability": {
            "identifiable_treatments": {
                "screen_time": {"method": "do_calculus"},
                "age": {"method": "do_calculus"},
            },
            "non_identifiable_treatments": {},
        },
        "estimation": {
            "state_order": ["screen_time", "sleep"],
            "edges": [
                {
                    "cause": "screen_time",
                    "effect": "sleep",
                    "description": "Screen time affects sleep",
                }
            ],
            "induced_dependencies": [],
        },
    }

    monkeypatch.setattr(dag, "load_parquet", lambda _path: pl.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(
        "causal_ssm_agent.flows.pipeline_helpers.format_schema_for_llm",
        lambda *_args, **_kwargs: "schema",
    )

    async def fake_propose_measurement_with_identifiability_fix(*_args, **_kwargs):
        return {"causal_spec": causal_spec}

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage1b.flow.propose_measurement_with_identifiability_fix",
        fake_propose_measurement_with_identifiability_fix,
    )

    result = asyncio.run(
        dag.stage1b(
            "Does screen time affect sleep?",
            {"_df_path": "/tmp/ignored.parquet", "_column_descriptions": {}},
            {"latent_model": latent_model},
        )
    )

    assert result["_identified_treatments"] == ["screen_time"]
    assert result["outcome"] == "success"


def test_fitted_artifact_pickles_without_live_jax_caches():
    from causal_ssm_agent.models.ssm.inference import FittedArtifact, InferenceResult

    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("cannot pickle runtime cache")

    spec = SimpleNamespace(
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=["screen_time_obs"],
    )
    builder = SimpleNamespace(_spec=spec, _model=_Unpicklable())
    result = InferenceResult(
        _samples={"drift": jnp.array([[[-0.5, 0.1], [0.0, -0.3]]], dtype=jnp.float32)},
        method="laplace_em",
        diagnostics={"likelihood_backend": _Unpicklable()},
    )
    artifact = FittedArtifact(
        result=result,
        builder=builder,
        times=jnp.array([0.0, 1.0], dtype=jnp.float32),
        observation_support=SimpleNamespace(manifest_names=["screen_time_obs"]),
        ppc_result={"checked": True, "per_variable_warnings": []},
        power_scaling_result={"checked": True, "diagnosis": {}},
    )

    restored = cloudpickle.loads(cloudpickle.dumps(artifact))

    assert restored.result is not None
    assert restored.result.method == "laplace_em"
    np.testing.assert_allclose(
        np.asarray(restored.result.get_samples()["drift"]),
        np.asarray(result.get_samples()["drift"]),
    )
    assert restored.builder is not None
    assert restored.builder._spec.latent_names == ["screen_time", "sleep_quality"]
    assert restored.ppc_result == {"checked": True, "per_variable_warnings": []}


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

    async def stage0(_workspace_id: str, openrouter_api_key: str | None = None) -> dict:
        raise AssertionError("stage0 should be restored, not rerun")

    async def stage1a(_question: str, openrouter_api_key: str | None = None) -> dict:
        raise AssertionError("stage1a should be restored, not rerun")

    async def stage1b(
        _question: str,
        _stage0: dict,
        _stage1a: dict,
        openrouter_api_key: str | None = None,
    ) -> dict:
        raise AssertionError("stage1b should be restored, not rerun")

    captured: dict = {}

    async def stage2(question: str, stage0: dict, stage1b: dict, **_kw) -> dict:
        calls.append(("stage2", question, stage0, stage1b))
        captured["question"] = question
        captured["stage0_df_path"] = stage0["_df_path"]
        captured["stage1b_result"] = stage1b
        data_for_model = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["1.0"],
                "anchor_time": ["2024-01-01"],
                "support_start": ["2024-01-01"],
                "support_end": ["2024-01-01"],
            }
        )
        return {
            "_data_for_model": data_for_model,
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

    model_path = run_dir / "stage2-model-data.parquet"
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
                "_data_for_model_path": "/dead/run/stage2-model-data.parquet",
                "workers": [{"worker_id": 999, "status": "stale", "n_extractions": 0}],
                "preserved_field": "kept-from-snapshot",
            },
            "web": web_payload,
        },
        workspace_id,
    )

    state = stage_registry.load_stage_state(workspace_id, "stage-2")

    assert state["result"]["_data_for_model_path"] == str(model_path)
    assert state["result"]["workers"] == web_payload["workers"]
    assert state["result"]["preserved_field"] == "kept-from-snapshot"


def test_stage4_checkpoints_append_in_incremental_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)

    workspace_id = "test_workspace"
    first_runtime = {"cursor": "first"}
    second_runtime = {"cursor": "second"}

    first_path = run_store_module.save_stage4_checkpoint(first_runtime, workspace_id)
    second_path = run_store_module.save_stage4_checkpoint(second_runtime, workspace_id)

    checkpoint_dir = tmp_path / "data" / workspace_id / "run" / "stage-4-checkpoints"
    checkpoint_files = sorted(path.name for path in checkpoint_dir.iterdir())

    assert first_path.endswith("stage-4-checkpoints/000001.pkl")
    assert second_path.endswith("stage-4-checkpoints/000002.pkl")
    assert checkpoint_files == ["000001.pkl", "000002.pkl"]
    assert run_store_module.load_stage4_checkpoint(workspace_id) == second_runtime

    run_store_module.clear_stage4_checkpoint(workspace_id)

    assert not checkpoint_dir.exists()


def test_pipeline_emits_stage_progress_events(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    emitted: list[tuple[str, str, str, dict | None]] = []
    monkeypatch.setattr(
        pipeline,
        "emit_stage_progress_event",
        lambda run_id, stage_id, status, **kwargs: emitted.append(
            (run_id, stage_id, status, kwargs.get("error"))
        ),
    )

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str, openrouter_api_key: str | None = None) -> dict:
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
    assert [(stage_id, status) for _, stage_id, status, _ in emitted] == [
        ("stage-0", "running"),
        ("stage-0", "completed"),
        ("stage-1a", "running"),
        ("stage-1a", "completed"),
    ]
    assert all(run_id for run_id, _, _, _ in emitted)


def test_pipeline_emits_failed_stage_event(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    emitted: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        pipeline,
        "emit_stage_progress_event",
        lambda run_id, stage_id, status, **_kwargs: emitted.append((run_id, stage_id, status)),
    )

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str, openrouter_api_key: str | None = None) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(dag, "stage1a", stage1a)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            pipeline.causal_inference_pipeline(
                query="why is this happening?",
                end_stage="stage-1a",
            )
        )

    assert [status for _, _, status in emitted] == [
        "running",
        "completed",
        "running",
        "failed",
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


def test_load_stage4b_state_reconstructs_inference_structure_from_public_payload(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)

    workspace_id = "test_workspace"
    _write_public_result(
        tmp_path,
        workspace_id,
        "stage-4b",
        {
            "outcome": "warn",
            "parametric_id": {
                "checked": True,
                "summary": {
                    "structural_issues": ["beta_x"],
                    "boundary_issues": [],
                    "weak_params": ["beta_y"],
                },
            },
            "inference_structure": {
                "likelihood_path": "composed",
                "auto_method": "laplace_em",
                "first_pass_rb": {
                    "status": "active",
                    "inactive_reason": None,
                    "latent_variables": [{"name": "x", "method": "kalman"}],
                    "obs_variables": [{"name": "y", "method": "kalman"}],
                },
            },
        },
    )

    state = stage_registry.load_stage_state(workspace_id, "stage-4b")

    assert state["result"]["parametric_id"]["checked"] is True
    assert state["result"]["inference_structure"]["likelihood_path"] == "composed"
    assert state["web"]["inference_structure"]["auto_method"] == "laplace_em"


def test_stage5a_uses_fit_metadata(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.build_stage5a_svi_attempts",
        lambda: [{"method": "svi", "guide_type": "mvn"}],
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": True,
            "n_samples": 321,
            "duration_seconds": 4.25,
            "svi_diagnostics": {"elbo_losses": [3.0, 2.0, 1.0]},
            "posterior_marginals": [],
            "posterior_pairs": [],
        },
    )

    result = dag.stage5a(
        {"model_spec": {}}, {"_data_for_model_path": "/tmp/stage2-model-data.parquet"}
    )

    assert result["inference_metadata"] == {
        "method": "svi",
        "n_samples": 321,
        "duration_seconds": 4.25,
    }


def test_stage5a_failed_fit_returns_warn(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.build_stage5a_svi_attempts",
        lambda: [
            {"method": "svi", "guide_type": "mvn"},
            {"method": "svi", "guide_type": "normal"},
        ],
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": False,
            "error": "fit exploded",
            "duration_seconds": 1.25,
        },
    )

    result = dag.stage5a(
        {"model_spec": {}}, {"_data_for_model_path": "/tmp/stage2-model-data.parquet"}
    )

    assert result["outcome"] == "warn"
    assert result["inference_metadata"] == {
        "method": "svi",
        "n_samples": 0,
        "duration_seconds": 2.5,
    }
    assert result["svi_diagnostics"] is None


def test_stage5a_retries_with_safer_svi_attempt(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5a.flow.build_stage5a_svi_attempts",
        lambda: [
            {
                "method": "svi",
                "guide_type": "mvn",
                "learning_rate": 0.003,
            },
            {
                "method": "svi",
                "guide_type": "normal",
                "learning_rate": 0.001,
            },
        ],
    )

    calls: list[dict] = []

    def _fit_model(_compiled, _data, sampler_config=None, **_kwargs):
        calls.append(dict(sampler_config or {}))
        if len(calls) == 1:
            return {
                "fitted": False,
                "error": "SVI produced non-finite losses",
                "duration_seconds": 1.25,
            }
        return {
            "fitted": True,
            "n_samples": 123,
            "duration_seconds": 2.5,
            "svi_diagnostics": {"elbo_losses": [2.0, 1.0]},
            "posterior_marginals": [],
            "posterior_pairs": [],
        }

    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage5b.fit.fit_model", _fit_model)

    result = dag.stage5a(
        {"model_spec": {}}, {"_data_for_model_path": "/tmp/stage2-model-data.parquet"}
    )

    assert [call["guide_type"] for call in calls] == ["mvn", "normal"]
    assert result["outcome"] == "success"
    assert result["inference_metadata"] == {
        "method": "svi",
        "n_samples": 123,
        "duration_seconds": 3.75,
    }


def test_stage5b_uses_fit_metadata(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": True,
            "n_samples": 654,
            "duration_seconds": 7.5,
            "inference_type": "svi",
            "result": None,
            "builder": None,
            "runtime": SimpleNamespace(observation_support=None),
            "times": np.array([0.0]),
            "mcmc_diagnostics": None,
            "svi_diagnostics": {"elbo_losses": [1.0]},
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": [],
            "posterior_pairs": [],
        },
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.run_power_scaling",
        lambda *_args, **_kwargs: {"checked": False, "error": "skip"},
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.run_ppc",
        lambda *_args, **_kwargs: {"checked": False, "per_variable_warnings": []},
    )
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(
                to_sampler_config=lambda method_override=None: {"method": method_override or "auto"}
            )
        ),
    )

    result = dag.stage5b(
        {"model_spec": {}},
        {"_data_for_model_path": "/tmp/stage2-model-data.parquet"},
        inference_method="svi",
    )

    assert result["inference_metadata"] == {
        "method": "svi",
        "n_samples": 654,
        "duration_seconds": 7.5,
    }


def test_stage5b_failed_fit_returns_fail_without_postfit_diagnostics(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": False,
            "error": "fit exploded",
            "duration_seconds": 2.5,
        },
    )

    def _unexpected_power_scaling(*_args, **_kwargs):
        raise AssertionError("run_power_scaling should not run after a failed fit")

    def _unexpected_ppc(*_args, **_kwargs):
        raise AssertionError("run_ppc should not run after a failed fit")

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.run_power_scaling",
        _unexpected_power_scaling,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage5b.fit.run_ppc",
        _unexpected_ppc,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.utils.config.get_config",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(
                to_sampler_config=lambda method_override=None: {"method": method_override or "auto"}
            )
        ),
    )

    result = dag.stage5b(
        {"model_spec": {}},
        {"_data_for_model_path": "/tmp/stage2-model-data.parquet"},
        inference_method="svi",
    )

    assert result["outcome"] == "fail"
    assert result["fail_reason"] == "model_fit_failed"
    assert result["power_scaling"] == []
    assert result["ppc"] == {"checked": False, "per_variable_warnings": []}
    assert result["inference_metadata"] == {
        "method": "svi",
        "n_samples": 0,
        "duration_seconds": 2.5,
    }
    assert result["_fitted_artifact"].result is None
    assert result["_fitted_artifact"].power_scaling_result == {
        "checked": False,
        "error": "fit exploded",
    }


def test_stage4_override_compiles_artifact_for_downstream_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("prefect.artifacts.create_markdown_artifact", _noop_artifact)
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stage_persistence.persist_web_result",
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
                    "construct_polarity": "positive",
                }
            ],
        },
        "estimation": {
            "state_order": ["stress"],
            "edges": [],
            "induced_dependencies": [],
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
        lit_enabled=True,
        inference_method=None,
        supported_overrides={"stage-4": override_payload},
        openrouter_api_key=None,
        openrouter_access_mode=None,
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
    assert stage4_result["_causal_spec"] == causal_spec
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
            "observation_rows": [
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
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2.flow.stage2_extraction_flow", stub)
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
    assert result["_data_for_model"].height == 1
    assert result["_data_for_model"]["support_kind"][0] == "interval"
    assert result["_data_for_model"]["summary_operator"][0] == "mean"
    assert result["_data_for_model"]["anchor_policy"][0] == "support_end"
    assert str(result["_data_for_model"]["anchor_time"][0]) == "2024-01-02 00:00:00"
    assert str(result["_data_for_model"]["support_start"][0]) == "2024-01-01 00:00:00"
    assert str(result["_data_for_model"]["support_end"][0]) == "2024-01-02 00:00:00"
    assert result["workers"] == [{"worker_id": 0, "status": "completed", "n_extractions": 1}]


def test_stage2_preserves_null_values_for_inference(monkeypatch, tmp_path):
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
    from causal_ssm_agent.utils.data import pivot_to_wide

    stub = _AsyncSubflowStub(
        {
            "observation_rows": [
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
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2.flow.stage2_extraction_flow", stub)
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
            "observation_rows": [
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
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage2.flow.stage2_extraction_flow", stub)
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


def test_stage4_loads_model_data_and_forwards_subflow_inputs(monkeypatch, tmp_path):
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
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4.flow.stage4_agentic_flow", stub)

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
            workspace_id="workspace-123",
        )
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    args, kwargs = stub.calls[0]
    assert args == ()
    assert kwargs["causal_spec"] == {
        "latent": {"constructs": []},
        "measurement": {"model_clock": "1d", "indicators": []},
    }
    assert kwargs["question"] == "why is this happening?"
    assert kwargs["indicator_audits"] == {}
    assert kwargs["enable_literature"] is True
    assert kwargs["workspace_id"] == "workspace-123"
    assert kwargs["openrouter_api_key"] is None
    assert kwargs["root_run_id"] is None
    assert kwargs["data_for_model"].to_dicts() == [
        {"indicator": "stress_score", "value": "1.0", "anchor_time": "2024-01-01"}
    ]
    assert result["model_spec"] == {"parameters": []}


def test_stage4_accepts_explicit_openrouter_api_key(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
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
    monkeypatch.setattr("causal_ssm_agent.flows.stages.stage4.flow.stage4_agentic_flow", stub)

    with openrouter_client.use_openrouter_api_key("context-key"):
        asyncio.run(
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
                openrouter_api_key="explicit-key",
            )
        )

    assert len(stub.calls) == 1
    assert stub.calls[0][1]["openrouter_api_key"] == "explicit-key"


def test_stage4b_loads_model_data_and_forwards_subflow_inputs(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(data_path)

    stub = _SyncSubflowStub({"parametric_id": {"checked": True}})
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4b.flow.stage4b_parametric_id_flow", stub
    )
    builder = object()

    result = dag.stage4b(
        {"model_spec": {"parameters": []}, "_compiled_ssm": "compiled-ssm"},
        {"_data_for_model_path": str(data_path)},
        ssm_builder=builder,
        root_run_id="root-123",
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    args, kwargs = stub.calls[0]
    assert args == ()
    assert kwargs["compiled_ssm"] == "compiled-ssm"
    assert kwargs["builder"] is builder
    assert kwargs["root_run_id"] == "root-123"
    assert kwargs["data_for_model"].to_dicts() == [
        {"indicator": "stress_score", "value": "1.0", "anchor_time": "2024-01-01"}
    ]
    assert result == {"parametric_id": {"checked": True}, "outcome": "success"}
