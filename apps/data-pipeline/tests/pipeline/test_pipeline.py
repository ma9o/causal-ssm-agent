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

from nof1_causal_lab.flows import dag, pipeline, stage_registry
from nof1_causal_lab.flows import run_store as run_store_module
from nof1_causal_lab.flows.stage_contracts import (
    Stage0Contract,
    Stage1aContract,
    Stage1bContract,
    Stage2Contract,
    Stage3Contract,
    Stage4Contract,
    Stage5bContract,
    Stage6Contract,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_state import (
    Stage4DomainState,
    Stage4Runtime,
)
from nof1_causal_lab.utils import openrouter_client
from tests.pipeline._support import noop_artifact as _noop_artifact
from tests.pipeline._support import redirect_storage as _redirect_storage


class _FakeModalRunnersModule(ModuleType):
    modal_stage4_runner: Any
    modal_stage5b_runner: Any


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _stub_config() -> SimpleNamespace:
    return SimpleNamespace(
        stage4_prior_elicitation=SimpleNamespace(literature_search=SimpleNamespace(enabled=True)),
    )


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
                "sources": [],
            }
        ],
    }


def _minimal_causal_spec(
    treatment: str = "treatment",
    outcome: str = "outcome",
) -> dict:
    """Build the minimum valid CausalSpec dict (with one outcome construct)."""
    return {
        "latent": _stage1a_latent_model(treatment, outcome),
        "measurement": {"model_clock": "1d", "indicators": []},
    }


def _minimal_stage1b_contract(
    treatment: str = "treatment",
    outcome: str = "outcome",
    **extra_causal_spec_fields: Any,
) -> Stage1bContract:
    """Build a minimal valid Stage1bContract for test stubs."""
    spec = _minimal_causal_spec(treatment, outcome)
    spec.update(extra_causal_spec_fields)
    return Stage1bContract(causal_spec=spec)


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
    from nof1_causal_lab.flows import stage_registry

    monkeypatch.setattr(stage_registry, "_registry", None)
    monkeypatch.setattr(stage_registry, "_execution_order", None)


def _patch_common_stage_stubs(monkeypatch, calls: list):
    # Parameter names must match the bare computation function signatures in dag.py
    # Runners must return contract instances (not dicts) because finalize_stage
    # calls contract.model_dump() and the pipeline accesses contract.outcome.

    async def stage0(workspace_id: str) -> Stage0Contract:
        calls.append(("stage0", workspace_id))
        return Stage0Contract(
            column_descriptions=[
                {"name": "timestamp", "description": "ts"},
                {"name": "value", "description": "val"},
            ],
        )

    async def stage2(question: str, stage0, stage1b, workspace_id: str, **_kw) -> Stage2Contract:
        calls.append(("stage2", question, stage0, stage1b))
        return Stage2Contract(
            workers=[{"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 1}],
        )

    def stage3(stage1b, stage2, workspace_id: str) -> Stage3Contract:
        calls.append(("stage3", stage1b, stage2))
        return Stage3Contract(
            is_valid=True,
            indicators={},
            dataset_issues=[],
        )

    def stage5b(
        stage4,
        stage2,
        workspace_id: str,
        inference_method: str | None = None,
    ) -> Stage5bContract:
        calls.append(("stage5b", stage4, stage2, inference_method))
        return Stage5bContract(
            ppc={"checked": False, "per_variable_warnings": []},
            inference_metadata={
                "method": "marginal_particle_gibbs",
                "n_samples": 0,
                "duration_seconds": 0.0,
            },
        )

    async def stage6(
        stage5b,
        stage1b,
        workspace_id: str,
        question: str | None = None,
    ) -> Stage6Contract:
        calls.append(("stage6", stage5b, stage1b, question))
        return Stage6Contract(intervention_results=[])

    def persist_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
        calls.append(("persist_web_result", stage_id, data, workspace_id))
        return data

    # Stub out persistence so finalize_stage doesn't touch the filesystem
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stage_persistence.persist_contract",
        lambda _stage_id, contract, _workspace_id: calls.append(
            ("persist_web_result", _stage_id, contract.model_dump(mode="json"), _workspace_id)
        ),
    )
    monkeypatch.setattr(
        run_store_module,
        "save_stage_snapshot",
        lambda _stage_id, _contract, _workspace_id: None,
    )

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage2", stage2)
    monkeypatch.setattr(dag, "stage3", stage3)
    monkeypatch.setattr(dag, "stage5b", stage5b)
    monkeypatch.setattr(dag, "stage6", stage6)
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stage_persistence.persist_web_result", persist_web_result
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

    fake_modal_runners = _FakeModalRunnersModule("nof1_causal_lab.flows.modal_runners")
    fake_modal_runners.modal_stage4_runner = fake_stage4_runner
    fake_modal_runners.modal_stage5b_runner = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "nof1_causal_lab.flows.modal_runners", fake_modal_runners)
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
    from nof1_causal_lab.flows.stages.stage2.definition import _bind_stage2
    from nof1_causal_lab.utils.config import get_config

    MAX_FREE_WINDOWS = get_config().stage2_workers.max_free_windows

    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    states = {
        "stage-0": Stage0Contract(column_descriptions=[]),
        "stage-1b": _minimal_stage1b_contract(),
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

    user_inputs = _bind_stage2(user_ctx, states)
    anonymous_inputs = _bind_stage2(anonymous_ctx, states)
    local_inputs = _bind_stage2(local_ctx, states)

    assert user_inputs["max_windows"] is None
    assert anonymous_inputs["max_windows"] == MAX_FREE_WINDOWS
    assert local_inputs["max_windows"] is None


def test_interactive_overrideable_stages_declare_materialization_policy():
    from nof1_causal_lab.flows.stage_contracts import INTERACTIVE_STAGES

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
    from nof1_causal_lab.flows.stages.stage4.definition import (
        _emit_stage4_initial_replay_state,
    )

    events: list[tuple[str, object] | tuple[str, object, object]] = []

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage4.agentic.stage4_runtime_projections.project_stage4_initial_state",
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
        "nof1_causal_lab.flows.runtime_events.emit_stage4_graph_event",
        lambda root_run_id, *, graph: events.append(("graph", root_run_id, graph)),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.runtime_events.emit_stage4_snapshot_event",
        lambda root_run_id, *, snapshot: events.append(("snapshot", root_run_id, snapshot)),
    )
    _runner_result = Stage4Contract.model_validate(
        {
            "model_spec": {"parameters": [], "likelihoods": []},
            "authored_priors": {},
            "resolved_priors": [],
        }
    )

    async def _runner(**_inputs):
        events.append(("runner", _inputs["root_run_id"]))
        return _runner_result

    _stage1b_contract = _minimal_stage1b_contract(
        estimation={"state_order": [], "edges": [], "induced_dependencies": []},
    )

    defn = stage_registry.StageDefinition(
        stage_id="stage-4",
        depends_on=frozenset(),
        contract=stage_registry.get_stage_registry()["stage-4"].contract,
        bind_inputs=lambda _ctx, _states: {
            "question": "why",
            "stage1b": _stage1b_contract,
            "stage2": Stage2Contract(workers=[]),
            "stage3": Stage3Contract(is_valid=True, indicators={}, dataset_issues=[]),
            "enable_literature": True,
            "workspace_id": "workspace",
            "root_run_id": "root-run-123",
        },
        runner=_runner,
        before_run=_emit_stage4_initial_replay_state,
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

    stage_state = asyncio.run(
        stage_registry.run_stage_flow(
            defn,
            ctx,
            {},
            finalize=lambda _stage_id, contract, _workspace_id: contract,
        )
    )

    assert stage_state is _runner_result
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
    monkeypatch.setattr("nof1_causal_lab.utils.config.get_config", _stub_config)
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

    async def stage0(workspace_id: str) -> Stage0Contract:
        seen.append(("stage0", openrouter_client.get_openrouter_api_key()))
        return Stage0Contract(
            column_descriptions=[
                {"name": "timestamp", "description": "ts"},
                {"name": "value", "description": "val"},
            ],
        )

    async def stage1a(question: str) -> Stage1aContract:
        seen.append(("stage1a", openrouter_client.get_openrouter_api_key()))
        return Stage1aContract.model_validate(_stub_stage1a_result())

    # Stub out persistence so finalize_stage doesn't touch the filesystem
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stage_persistence.persist_contract",
        lambda _stage_id, _contract, _workspace_id: None,
    )
    monkeypatch.setattr(
        run_store_module,
        "save_stage_snapshot",
        lambda _stage_id, _contract, _workspace_id: None,
    )

    monkeypatch.setattr(dag, "stage0", stage0)
    monkeypatch.setattr(dag, "stage1a", stage1a)
    _reset_stage_registry(monkeypatch)

    result = asyncio.run(pipeline.causal_inference_pipeline(**pipeline_kwargs))

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
        "nof1_causal_lab.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> Stage1aContract:
        calls.append(("stage1a", question))
        return Stage1aContract(
            latent_model=_stage1a_latent_model("generated-treatment", "generated-outcome")
        )

    async def stage1b(
        question: str,
        stage0,
        stage1a,
        workspace_id: str,
    ) -> Stage1bContract:
        calls.append(("stage1b", question, stage0, stage1a))
        latent_model = stage1a.latent_model.model_dump()
        return Stage1bContract(
            causal_spec={
                "latent": latent_model,
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
        )

    async def stage4(
        question: str,
        stage1b,
        stage2,
        stage3,
        enable_literature: bool,
        workspace_id: str,
        root_run_id: str | None = None,
    ) -> Stage4Contract:
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
        return Stage4Contract(
            model_spec={"parameters": [], "likelihoods": []},
            authored_priors={},
            resolved_priors=[],
        )

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
    # The override is materialized as a Stage1aContract, check it was passed to stage1b
    stage1b_stage1a_arg = stage1b_calls[0][3]
    assert isinstance(stage1b_stage1a_arg, Stage1aContract)
    assert stage1b_stage1a_arg.latent_model.model_dump() == override_payload["latent_model"]
    # Check persist was called for stage-1a with the override payload
    assert any(entry[0] == "persist_web_result" and entry[1] == "stage-1a" for entry in calls)
    # Pipeline returns merged stage-5b + stage-6 contract dicts
    assert "intervention_results" in result


def test_pipeline_stops_cleanly_on_completed_fail_outcome(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> Stage1aContract:
        calls.append(("stage1a", question))
        return Stage1aContract(latent_model=_stage1a_latent_model())

    async def stage1b(
        question: str,
        stage0,
        stage1a,
        workspace_id: str,
    ) -> Stage1bContract:
        calls.append(("stage1b", question, stage0, stage1a))
        return Stage1bContract(
            causal_spec={
                "latent": stage1a.latent_model.model_dump(),
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "stress_score",
                            "construct_name": "treatment",
                            "how_to_measure": "Measure treatment",
                            "construct_polarity": "positive",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
            }
        )

    def stage3(stage1b, stage2, workspace_id: str) -> Stage3Contract:
        calls.append(("stage3", stage1b, stage2))
        return Stage3Contract(
            is_valid=False,
            indicators={},
            dataset_issues=[
                {
                    "issue_type": "no_numeric",
                    "severity": "error",
                    "message": "No numeric observations survived validation.",
                }
            ],
            outcome="fail",
            fail_reason="data_validation_failed",
        )

    async def stage4(
        question: str,
        stage1b,
        stage2,
        stage3,
        enable_literature: bool,
        workspace_id: str,
        root_run_id: str | None = None,
    ) -> Stage4Contract:
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
        "nof1_causal_lab.utils.config.get_config",
        _stub_config,
    )
    monkeypatch.setattr(pipeline, "create_markdown_artifact", _noop_artifact)

    calls: list = []
    _patch_common_stage_stubs(monkeypatch, calls)

    async def stage1a(question: str) -> Stage1aContract:
        calls.append(("stage1a", question))
        return Stage1aContract(
            latent_model=_stage1a_latent_model("override_treatment", "override_outcome")
        )

    async def stage4(
        question: str,
        stage1b,
        stage2,
        stage3,
        enable_literature: bool,
        workspace_id: str,
        root_run_id: str | None = None,
    ) -> Stage4Contract:
        calls.append(("stage4", question, stage1b, stage2, stage3, enable_literature, workspace_id))
        return Stage4Contract(
            model_spec={"parameters": [], "likelihoods": []},
            authored_priors={},
            resolved_priors=[],
        )

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
    # stage6 receives stage1b as a Stage1bContract
    materialized_stage1b = stage6_calls[0][2]
    assert isinstance(materialized_stage1b, Stage1bContract)
    assert materialized_stage1b.outcome == "success"
    # Pipeline returns merged stage-5b + stage-6 contract dicts
    assert "intervention_results" in result


def test_stage6_runs_interventions_from_fitted_artifact(monkeypatch):
    from types import SimpleNamespace

    from nof1_causal_lab.models.ssm.inference import FittedArtifact

    # Build a minimal FittedArtifact with mock result and compiled spec.
    mock_spec = SimpleNamespace(
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=[],
    )

    fitted_artifact = FittedArtifact(
        result=None,
        spec=mock_spec,
        times=np.array([0.0, 1.0]),
        ppc_result={"checked": True, "per_variable_warnings": []},
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage6.flow.load_pickle",
        lambda _path: fitted_artifact,
    )
    monkeypatch.setattr("prefect.artifacts.create_table_artifact", lambda **_kwargs: None)
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: "unused.pkl",
    )

    from contextlib import asynccontextmanager

    from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
    from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage

    class _FakeAgentSession:
        async def turn(self, _user_message):
            return TurnResult(completion="stubbed summary")

        @property
        def result(self):
            return AgentResult(
                completion="stubbed summary",
                trace=LLMTrace(
                    messages=[TraceMessage(role="assistant", content="stubbed summary")]
                ),
            )

    class _FakeStageSessionFactory:
        def __init__(self, *_args, **_kwargs):
            self.accumulated_trace = LLMTrace(
                messages=[TraceMessage(role="assistant", content="stubbed summary")]
            )

        @asynccontextmanager
        async def open(self, *, system_prompt=None, tools=None, log_label=None):
            captured["commentary_label"] = log_label
            yield _FakeAgentSession()

    monkeypatch.setattr(
        "nof1_causal_lab.flows.llm_stage_runtime.build_stage_session_factory",
        lambda _config: _FakeStageSessionFactory(),
    )

    def fake_compute_interventions(**kwargs):
        captured.update(kwargs)
        return [{"treatment": "screen_time", "posterior_draws": [0.9, 1.0, 1.1]}]

    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.counterfactual.compute_interventions",
        fake_compute_interventions,
    )

    stage1b_contract = _minimal_stage1b_contract(
        "screen_time",
        "sleep_quality",
        identifiability={
            "identifiable_treatments": {
                "screen_time": {
                    "method": "do_calculus",
                    "estimand": "P(sleep_quality|do(screen_time))",
                },
            },
            "non_identifiable_treatments": {},
        },
        estimation={
            "state_order": ["screen_time", "sleep_quality"],
            "edges": [
                {
                    "cause": "screen_time",
                    "effect": "sleep_quality",
                    "description": "screen_time affects sleep_quality",
                    "lagged": True,
                }
            ],
            "induced_dependencies": [],
        },
    )
    stage5b_contract = Stage5bContract(
        ppc={"checked": True, "per_variable_warnings": []},
        inference_metadata={
            "method": "marginal_particle_gibbs",
            "n_samples": 100,
            "duration_seconds": 1.0,
        },
    )
    result = asyncio.run(
        dag.stage6(
            stage5b_contract,
            stage1b_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage6Contract)
    result_dict = result.model_dump(mode="json")
    assert result_dict["intervention_results"][0]["treatment"] == "screen_time"
    # FittedArtifact.result is None, so run_interventions returns early stubs
    # and compute_interventions is not called. The commentary label is still set.
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
        "nof1_causal_lab.flows.stages.stage3.flow.validate_extraction",
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
                        ],
                        "checks": {},
                    }
                }
            },
            "dataset_issues": [],
        },
    )

    # Monkeypatch find_run_artifact to return our test path
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(model_path),
    )

    stage1b_contract = _minimal_stage1b_contract()
    stage2_contract = Stage2Contract(workers=[])

    result = asyncio.run(
        dag.stage3(
            stage1b_contract,
            stage2_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage3Contract)
    assert result.outcome == "warn"
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
        "nof1_causal_lab.flows.stages.stage3.flow.validate_extraction",
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

    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(model_path),
    )

    stage1b_contract = _minimal_stage1b_contract()
    stage2_contract = Stage2Contract(workers=[])

    result = asyncio.run(
        dag.stage3(
            stage1b_contract,
            stage2_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage3Contract)
    assert result.is_valid is True
    assert result.outcome == "warn"
    assert result.fail_reason is None


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
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_issue_searches",
                    "construct_name": "sleep",
                    "how_to_measure": "Measure sleep",
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        },
        "identifiability": {
            "identifiable_treatments": {
                "screen_time": {"method": "do_calculus", "estimand": "P(sleep|do(screen_time))"},
                "age": {"method": "do_calculus", "estimand": "P(sleep|do(age))"},
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
        "nof1_causal_lab.flows.pipeline_helpers.format_schema_for_llm",
        lambda *_args, **_kwargs: "schema",
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: "/tmp/ignored.parquet",
    )

    async def fake_propose_measurement_with_identifiability_fix(*_args, **_kwargs):
        return {"causal_spec": causal_spec}

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage1b.flow.propose_measurement_with_identifiability_fix",
        fake_propose_measurement_with_identifiability_fix,
    )

    stage0_contract = Stage0Contract(
        column_descriptions=[],
    )
    stage1a_contract = Stage1aContract(latent_model=latent_model)

    result = asyncio.run(
        dag.stage1b(
            "Does screen time affect sleep?",
            stage0_contract,
            stage1a_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage1bContract)
    assert result.outcome == "success"


def test_fitted_artifact_pickles_without_live_jax_caches():
    from nof1_causal_lab.models.ssm.inference import FittedArtifact, InferenceResult

    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("cannot pickle runtime cache")

    spec = SimpleNamespace(
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=["screen_time_obs"],
    )
    result = InferenceResult(
        _samples={"dynamics": jnp.array([[[-0.5, 0.1], [0.0, -0.3]]], dtype=jnp.float32)},
        method="marginal_particle_gibbs",
        diagnostics={"likelihood_backend": _Unpicklable()},
    )
    artifact = FittedArtifact(
        result=result,
        spec=spec,
        times=jnp.array([0.0, 1.0], dtype=jnp.float32),
        observation_support=SimpleNamespace(manifest_names=["screen_time_obs"]),
        ppc_result={"checked": True, "per_variable_warnings": []},
    )

    restored = cloudpickle.loads(cloudpickle.dumps(artifact))

    assert restored.result is not None
    assert restored.result.method == "marginal_particle_gibbs"
    np.testing.assert_allclose(
        np.asarray(restored.result.get_samples()["dynamics"]),
        np.asarray(result.get_samples()["dynamics"]),
    )
    assert restored.spec is not None
    assert restored.spec.latent_names == ["screen_time", "sleep_quality"]
    assert restored.ppc_result == {"checked": True, "per_variable_warnings": []}


def test_resume_from_stage2_loads_existing_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
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
        _minimal_stage1b_contract().model_dump(mode="json"),
    )

    async def stage0(_workspace_id: str) -> Stage0Contract:
        raise AssertionError("stage0 should be restored, not rerun")

    async def stage1a(_question: str) -> Stage1aContract:
        raise AssertionError("stage1a should be restored, not rerun")

    async def stage1b(
        _question: str,
        _stage0,
        _stage1a,
        workspace_id: str,
    ) -> Stage1bContract:
        raise AssertionError("stage1b should be restored, not rerun")

    captured: dict = {}

    async def stage2(question: str, stage0, stage1b, workspace_id: str, **_kw) -> Stage2Contract:
        calls.append(("stage2", question, stage0, stage1b))
        captured["question"] = question
        captured["stage0"] = stage0
        captured["stage1b"] = stage1b
        return Stage2Contract(
            workers=[{"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 1}],
        )

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
    # stage1b is now a Stage1bContract with empty indicators
    assert isinstance(captured["stage1b"], Stage1bContract)
    assert captured["stage1b"].causal_spec.measurement.indicators == []
    # stage0 is now a Stage0Contract
    assert isinstance(captured["stage0"], Stage0Contract)


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
        "workers": [{"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 1}],
    }
    _write_public_result(tmp_path, workspace_id, "stage-2", web_payload)

    # Save a Stage2Contract as the snapshot (new format: contract instance)
    snapshot_contract = Stage2Contract.model_validate(web_payload)
    run_store_module.save_stage_snapshot("stage-2", snapshot_contract, workspace_id)

    state = stage_registry.load_stage_state(workspace_id, "stage-2")

    assert isinstance(state, Stage2Contract)
    assert state.outcome == "success"
    assert len(state.workers) == 1
    assert state.workers[0].worker_id == 0


def test_stage4_checkpoints_overwrite_per_block_and_track_cursor(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)

    workspace_id = "test_workspace"
    first_runtime = Stage4Runtime(
        domain=Stage4DomainState(active_block_id="review:model_spec"),
    )
    second_runtime = Stage4Runtime(
        domain=Stage4DomainState(active_block_id="observation:obs_ordered_base"),
    )
    third_runtime = Stage4Runtime(
        domain=Stage4DomainState(active_block_id="observation:obs_ordered_base"),
    )
    done_runtime = Stage4Runtime(
        domain=Stage4DomainState(done=True),
    )

    first_path = run_store_module.save_stage4_checkpoint(first_runtime, workspace_id)
    second_path = run_store_module.save_stage4_checkpoint(second_runtime, workspace_id)
    third_path = run_store_module.save_stage4_checkpoint(third_runtime, workspace_id)
    done_path = run_store_module.save_stage4_checkpoint(done_runtime, workspace_id)

    checkpoint_dir = tmp_path / "data" / workspace_id / "run" / "stage-4-checkpoints"
    checkpoint_files = sorted(path.name for path in checkpoint_dir.iterdir())
    cursor_payload = json.loads((checkpoint_dir / "cursor.json").read_text())

    assert first_path.endswith("stage-4-checkpoints/review%3Amodel_spec.pkl")
    assert second_path.endswith("stage-4-checkpoints/observation%3Aobs_ordered_base.pkl")
    assert third_path.endswith("stage-4-checkpoints/observation%3Aobs_ordered_base.pkl")
    assert done_path.endswith("stage-4-checkpoints/__done__.pkl")
    assert checkpoint_files == [
        "__done__.pkl",
        "cursor.json",
        "observation%3Aobs_ordered_base.pkl",
        "review%3Amodel_spec.pkl",
    ]
    assert cursor_payload == {"kind": "done"}
    assert run_store_module.load_stage4_checkpoint(workspace_id) == done_runtime

    run_store_module.clear_stage4_checkpoint(workspace_id)

    assert checkpoint_dir.exists()
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "__done__.pkl",
        "observation%3Aobs_ordered_base.pkl",
        "review%3Amodel_spec.pkl",
    ]
    with pytest.raises(FileNotFoundError):
        run_store_module.load_stage4_checkpoint(workspace_id)


def test_pipeline_emits_stage_progress_events(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _redirect_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
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

    async def stage1a(question: str) -> Stage1aContract:
        calls.append(("stage1a", question))
        return Stage1aContract(
            latent_model=_stage1a_latent_model("generated-treatment", "generated-outcome")
        )

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
        "nof1_causal_lab.utils.config.get_config",
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

    async def stage1a(question: str) -> Stage1aContract:
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
            "ppc": {
                "checked": True,
                "per_variable_warnings": [
                    {
                        "variable": "y",
                        "check_type": "calibration",
                        "message": "m",
                        "value": 0.5,
                    }
                ],
            },
            "inference_metadata": {
                "method": "marginal_particle_gibbs",
                "n_samples": 100,
                "duration_seconds": 5.0,
            },
            "mcmc_diagnostics": None,
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        },
    )

    state = stage_registry.load_stage_state(workspace_id, "stage-5b")

    assert isinstance(state, Stage5bContract)
    assert state.ppc.checked is True


def test_stage5b_uses_fit_metadata(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": True,
            "n_samples": 654,
            "duration_seconds": 7.5,
            "inference_type": "marginal_particle_gibbs",
            "result": None,
            "builder": None,
            "runtime": SimpleNamespace(observation_support=None),
            "times": np.array([0.0]),
            "mcmc_diagnostics": None,
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": [],
            "posterior_pairs": [],
        },
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.fit.run_ppc",
        lambda *_args, **_kwargs: {"checked": False, "per_variable_warnings": []},
    )
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(
                to_sampler_config=lambda method_override=None: {
                    "method": method_override or "marginal_particle_gibbs"
                },
                compute_loo_diagnostics=False,
            )
        ),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag._load_compiled_ssm",
        lambda _workspace_id: None,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag._load_data_for_model_path",
        lambda _workspace_id: "/tmp/stage2-model-data.parquet",
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_pickle",
        lambda _obj, _workspace_id, _filename: None,
    )

    _s4 = Stage4Contract(
        model_spec={"parameters": [], "likelihoods": []},
        authored_priors={},
        resolved_priors=[],
    )
    _s2 = Stage2Contract(workers=[])

    result = dag.stage5b(
        _s4,
        _s2,
        workspace_id="test-workspace",
        inference_method="marginal_particle_gibbs",
    )

    assert isinstance(result, Stage5bContract)
    result_dict = result.model_dump(mode="json")
    assert result_dict["inference_metadata"] == {
        "method": "marginal_particle_gibbs",
        "n_samples": 654,
        "duration_seconds": 7.5,
    }


def test_stage5b_failed_fit_returns_fail_without_postfit_diagnostics(monkeypatch):
    data_for_model = pl.DataFrame(
        {"indicator": ["y"], "value": ["1"], "anchor_time": ["2024-01-01"]}
    )

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.flow.load_parquet",
        lambda _path: data_for_model,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.fit.fit_model",
        lambda *_args, **_kwargs: {
            "fitted": False,
            "error": "fit exploded",
            "duration_seconds": 2.5,
        },
    )

    def _unexpected_ppc(*_args, **_kwargs):
        raise AssertionError("run_ppc should not run after a failed fit")

    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage5b.fit.run_ppc",
        _unexpected_ppc,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(
                to_sampler_config=lambda method_override=None: {
                    "method": method_override or "marginal_particle_gibbs"
                },
                compute_loo_diagnostics=False,
            )
        ),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag._load_compiled_ssm",
        lambda _workspace_id: None,
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag._load_data_for_model_path",
        lambda _workspace_id: "/tmp/stage2-model-data.parquet",
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_pickle",
        lambda _obj, _workspace_id, _filename: None,
    )

    _s4 = Stage4Contract(
        model_spec={"parameters": [], "likelihoods": []},
        authored_priors={},
        resolved_priors=[],
    )
    _s2 = Stage2Contract(workers=[])

    result = dag.stage5b(
        _s4,
        _s2,
        workspace_id="test-workspace",
        inference_method="marginal_particle_gibbs",
    )

    assert isinstance(result, Stage5bContract)
    assert result.outcome == "fail"
    assert result.fail_reason == "model_fit_failed"
    result_dict = result.model_dump(mode="json")
    assert result_dict["ppc"] == {
        "checked": False,
        "per_variable_warnings": [],
        "overlays": [],
        "test_stats": [],
        "n_subsample": None,
    }
    assert result_dict["inference_metadata"] == {
        "method": "marginal_particle_gibbs",
        "n_samples": 0,
        "duration_seconds": 2.5,
    }


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
            "worker_statuses": [
                {"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 1}
            ],
            "n_total_extractions": 1,
        }
    )
    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage2.flow.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(tmp_path / "input.parquet"),
    )

    saved_parquets: list[pl.DataFrame] = []
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_parquet",
        lambda df, _workspace_id, _filename: saved_parquets.append(df),
    )

    stage0_contract = Stage0Contract(column_descriptions=[])
    stage1b_contract = Stage1bContract(
        causal_spec={
            "latent": _stage1a_latent_model("stress", "outcome"),
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "construct_name": "stress",
                        "how_to_measure": "Measure stress",
                        "construct_polarity": "positive",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
        }
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            stage0_contract,
            stage1b_contract,
            workspace_id="test-workspace",
        )
    )

    assert len(stub.with_options_calls) == 1
    assert stub.with_options_calls[0]["task_runner"]._max_workers == 6
    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    assert isinstance(result, Stage2Contract)
    # Verify data_for_model was saved via save_parquet
    assert len(saved_parquets) == 1
    data_for_model = saved_parquets[0]
    assert data_for_model.height == 1
    assert data_for_model["support_kind"][0] == "interval"
    assert data_for_model["summary_operator"][0] == "mean"
    assert data_for_model["anchor_policy"][0] == "support_end"
    assert str(data_for_model["anchor_time"][0]) == "2024-01-02 00:00:00"
    assert str(data_for_model["support_start"][0]) == "2024-01-01 00:00:00"
    assert str(data_for_model["support_end"][0]) == "2024-01-02 00:00:00"
    workers = result.model_dump(mode="json")["workers"]
    assert workers == [
        {"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 1, "error": None}
    ]


def test_stage2_preserves_null_values_for_inference(monkeypatch, tmp_path):
    from nof1_causal_lab.models.ssm.runtime import prepare_fit_inputs
    from nof1_causal_lab.utils.data import pivot_to_wide

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
            "worker_statuses": [
                {"worker_id": 0, "status": "completed", "n_extractions": 2, "n_windows": 1}
            ],
            "n_total_extractions": 2,
        }
    )
    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage2.flow.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(tmp_path / "input.parquet"),
    )

    saved_parquets: list[pl.DataFrame] = []
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_parquet",
        lambda df, _workspace_id, _filename: saved_parquets.append(df),
    )

    stage0_contract = Stage0Contract(column_descriptions=[])
    stage1b_contract = Stage1bContract(
        causal_spec={
            "latent": _stage1a_latent_model("screen_time", "sleep"),
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "daytime_screen_events",
                        "construct_name": "screen_time",
                        "how_to_measure": "Count events",
                        "construct_polarity": "positive",
                        "measurement_dtype": "count",
                        "aggregation": "sum",
                    },
                    {
                        "name": "last_evening_activity_hour",
                        "construct_name": "sleep",
                        "how_to_measure": "Measure hour",
                        "construct_polarity": "positive",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
        }
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            stage0_contract,
            stage1b_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage2Contract)
    assert len(saved_parquets) == 1
    data_for_model = saved_parquets[0]
    assert data_for_model.height == 2
    assert (
        data_for_model.filter(pl.col("indicator") == "last_evening_activity_hour")[
            "value"
        ].null_count()
        == 1
    )

    observations, _times, manifest_names, _wide = prepare_fit_inputs(
        SimpleNamespace(manifest_names=[], manifest_centered=None),
        pivot_to_wide(data_for_model),
    )
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
            "worker_statuses": [
                {"worker_id": 0, "status": "completed", "n_extractions": 2, "n_windows": 1}
            ],
            "n_total_extractions": 2,
        }
    )
    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage2.flow.stage2_extraction_flow", stub)
    monkeypatch.setattr(
        "nof1_causal_lab.utils.config.get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(max_concurrent_workers=6)),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(tmp_path / "input.parquet"),
    )

    saved_parquets: list[pl.DataFrame] = []
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_parquet",
        lambda df, _workspace_id, _filename: saved_parquets.append(df),
    )

    stage0_contract = Stage0Contract(column_descriptions=[])
    stage1b_contract = Stage1bContract(
        causal_spec={
            "latent": _stage1a_latent_model("stress", "mood"),
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "construct_name": "stress",
                        "how_to_measure": "Measure stress",
                        "construct_polarity": "positive",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "closing_mood",
                        "construct_name": "mood",
                        "how_to_measure": "Measure mood",
                        "construct_polarity": "positive",
                        "measurement_dtype": "ordinal",
                        "aggregation": "last",
                        "ordinal_levels": ["bad", "good"],
                    },
                ],
            },
        }
    )

    result = asyncio.run(
        dag.stage2(
            "why is this happening?",
            stage0_contract,
            stage1b_contract,
            workspace_id="test-workspace",
        )
    )

    assert isinstance(result, Stage2Contract)
    assert len(saved_parquets) == 1
    data_for_model = saved_parquets[0].sort("indicator")
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
            "model_spec": {"parameters": [], "likelihoods": []},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": {
                "latent": {"constructs": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        }
    )
    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage4.flow.stage4_agentic_flow", stub)
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(data_path),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_json",
        lambda _obj, _workspace_id, _filename: None,
    )

    stage1b_contract = _minimal_stage1b_contract()
    stage2_contract = Stage2Contract(workers=[])
    stage3_contract = Stage3Contract(is_valid=True, indicators={}, dataset_issues=[])

    result = asyncio.run(
        dag.stage4(
            "why is this happening?",
            stage1b_contract,
            stage2_contract,
            stage3_contract,
            enable_literature=True,
            workspace_id="workspace-123",
        )
    )

    assert len(stub.calls) == 1
    assert stub.fn_calls == []
    args, kwargs = stub.calls[0]
    assert args == ()
    assert kwargs["causal_spec"] == stage1b_contract.causal_spec.model_dump()
    assert kwargs["question"] == "why is this happening?"
    assert kwargs["indicator_audits"] == {}
    assert kwargs["enable_literature"] is True
    assert kwargs["workspace_id"] == "workspace-123"
    assert kwargs["openrouter_api_key"] is None
    assert kwargs["root_run_id"] is None
    assert kwargs["data_for_model"].to_dicts() == [
        {"indicator": "stress_score", "value": "1.0", "anchor_time": "2024-01-01"}
    ]
    assert isinstance(result, Stage4Contract)


def test_stage4_accepts_explicit_openrouter_api_key(monkeypatch, tmp_path):
    data_path = tmp_path / "stage2-model-data.parquet"
    pl.DataFrame(
        {"indicator": ["stress_score"], "value": ["1.0"], "anchor_time": ["2024-01-01"]}
    ).write_parquet(data_path)

    stub = _AsyncSubflowStub(
        {
            "model_spec": {"parameters": [], "likelihoods": []},
            "priors": {},
            "authored_priors": {},
            "resolved_priors": [],
            "causal_spec": {
                "latent": {"constructs": []},
                "measurement": {"model_clock": "1d", "indicators": []},
            },
        }
    )
    monkeypatch.setattr("nof1_causal_lab.flows.stages.stage4.flow.stage4_agentic_flow", stub)
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.find_run_artifact",
        lambda _workspace_id, _filenames: str(data_path),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.flows.dag.save_json",
        lambda _obj, _workspace_id, _filename: None,
    )

    stage1b_contract = _minimal_stage1b_contract()
    stage2_contract = Stage2Contract(workers=[])
    stage3_contract = Stage3Contract(is_valid=True, indicators={}, dataset_issues=[])

    with openrouter_client.use_openrouter_api_key("context-key"):
        asyncio.run(
            dag.stage4(
                "why is this happening?",
                stage1b_contract,
                stage2_contract,
                stage3_contract,
                enable_literature=True,
                workspace_id="workspace-123",
                openrouter_api_key="explicit-key",
            )
        )

    assert len(stub.calls) == 1
    assert stub.calls[0][1]["openrouter_api_key"] == "explicit-key"
