from datetime import UTC, datetime
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

import causal_ssm_agent.tool_server as tool_server


def test_execute_tool_rejects_invalid_input_before_invoking_tool(monkeypatch):
    client = TestClient(tool_server.app)
    called = False

    def fake_impl(_ctx, _args):
        nonlocal called
        called = True
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("stage-1a", "validate_latent_model"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/stage-1a/validate_latent_model",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_execute_tool_surfaces_unexpected_exception_detail(monkeypatch):
    client = TestClient(tool_server.app)

    monkeypatch.setattr(tool_server, "_build_context", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("stage-1a", "validate_latent_model"),
        lambda _ctx, _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/api/tools/stage-1a/validate_latent_model",
        json={"workspace_id": "user-123", "input": {"structure_json": "{}"}},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "message": "boom",
            "exception_type": "RuntimeError",
            "stage_id": "stage-1a",
            "tool_name": "validate_latent_model",
        }
    }


def test_persist_stage_web_patch_uses_shared_persistence_helper(monkeypatch):
    client = TestClient(tool_server.app)

    called_with = {}

    def fake_persist_web_patch(stage_id, patch, workspace_id):
        called_with["stage_id"] = stage_id
        called_with["patch"] = patch
        called_with["workspace_id"] = workspace_id
        return {"outcome": "success", **patch}

    monkeypatch.setattr(tool_server, "persist_web_patch", fake_persist_web_patch)

    response = client.post(
        "/api/stages/stage-6/persist-web-patch",
        json={"workspace_id": "user-123", "patch": {"llm_trace": {"messages": []}}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "payload": {"outcome": "success", "llm_trace": {"messages": []}},
    }
    assert called_with == {
        "stage_id": "stage-6",
        "patch": {"llm_trace": {"messages": []}},
        "workspace_id": "user-123",
    }


def test_build_stage6_context_rehydrates_builder_from_persisted_spec(monkeypatch):
    spec = SimpleNamespace(latent_names=["screen_time", "sleep_quality"], manifest_names=["sleep_obs"])
    fitted_artifact = SimpleNamespace(
        builder=SimpleNamespace(_spec=spec),
        observation_support=None,
    )
    rebuilt_builder = SimpleNamespace(_spec=spec, _model=object())
    rebuilt_runtime = SimpleNamespace(
        builder=rebuilt_builder,
        observation_support="support-runtime",
        observation_data=None,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool_server,
        "_load_stage_result",
        lambda _workspace_id, stage_id: (
            {"causal_spec": {"identifiability": {}, "measurement": {}}, "outcome": "warn"}
            if stage_id == "stage-1b"
            else {"outcome": "warn"}
        ),
    )
    monkeypatch.setattr(tool_server, "_load_optional_stage_result", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        tool_server,
        "_load_runtime_stage_result",
        lambda _workspace_id, stage_id: {
            "_data_for_model_path": "/tmp/stage2.parquet"
        }
        if stage_id == "stage-2"
        else {"_fitted_result_path": "/tmp/stage5b.pkl"},
    )
    monkeypatch.setattr(tool_server, "load_pickle", lambda _path: fitted_artifact)
    monkeypatch.setattr(tool_server, "load_parquet", lambda _path: object())
    monkeypatch.setattr(tool_server, "_extract_observation_timestamps", lambda _obs: [])
    monkeypatch.setattr(tool_server, "get_outcome_name", lambda _spec: "sleep_quality")
    monkeypatch.setattr(tool_server, "get_estimable_treatments", lambda _spec: ["screen_time"])

    def fake_prepare_model_runtime(*, data_for_model, builder, compiled_ssm=None, sampler_config=None):
        del data_for_model, compiled_ssm, sampler_config
        captured["builder"] = builder
        return rebuilt_runtime

    monkeypatch.setattr(tool_server, "prepare_model_runtime", fake_prepare_model_runtime)

    ctx = tool_server._build_stage6_context("user-123")

    assert isinstance(captured["builder"], tool_server.SSMModelBuilder)
    assert captured["builder"]._ssm_spec is spec
    assert ctx["_fitted_artifact"].builder is rebuilt_builder
    assert ctx["_fitted_artifact"].observation_support == "support-runtime"
    assert ctx["_prepared_runtime"] is rebuilt_runtime


def test_execute_validate_model_loads_stage2_runtime_via_stage_registry(monkeypatch):
    import causal_ssm_agent.flows.stage_registry as stage_registry

    expected_data_for_model = object()
    captured: dict[str, object] = {}

    def fake_load_stage_state(workspace_id, stage_id, prior_states=None):
        del prior_states
        assert workspace_id == "user-123"
        assert stage_id == "stage-2"
        return {"result": {"_data_for_model_path": "/run/stage2-model-data.parquet"}}

    def fake_load_parquet(path):
        assert path == "/run/stage2-model-data.parquet"
        return expected_data_for_model

    def fake_stage4_grounding(
        _data,
        causal_spec,
        *,
        current=None,
        data_for_model=None,
        indicator_audits=None,
    ):
        captured["causal_spec"] = causal_spec
        captured["current"] = current
        captured["data_for_model"] = data_for_model
        captured["indicator_audits"] = indicator_audits
        return {"model_spec": {}}, "VALID"

    monkeypatch.setattr(stage_registry, "load_stage_state", fake_load_stage_state)
    monkeypatch.setattr(tool_server, "load_parquet", fake_load_parquet)
    monkeypatch.setattr(
        tool_server,
        "_load_stage4_current",
        lambda workspace_id: {"workspace_id": workspace_id, "model_spec": {"parameters": []}},
    )
    monkeypatch.setattr(tool_server, "stage4_grounding", fake_stage4_grounding)

    result = tool_server._execute_validate_model(
        {
            "_workspace_id": "user-123",
            "stage-1b": {"causal_spec": {"latent": {"constructs": []}}},
        },
        {"model_json": "{}"},
    )

    assert result == {"result": "VALID", "stage_output": {"model_spec": {}}}
    assert captured == {
        "causal_spec": {"latent": {"constructs": []}},
        "current": {"workspace_id": "user-123", "model_spec": {"parameters": []}},
        "data_for_model": expected_data_for_model,
        "indicator_audits": None,
    }


def test_simulate_counterfactual_respects_estimand_shape(monkeypatch):
    class FakeResult:
        def __init__(self, samples):
            self._samples = samples

        def get_samples(self):
            return self._samples

    samples = {
        "drift": jnp.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[3.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "cint": jnp.zeros((2, 2)),
    }

    monkeypatch.setattr(
        tool_server,
        "approximate_abducted_state",
        lambda *_args, **_kwargs: {
            "state": jnp.zeros(2),
            "method": "kalman_smoother",
            "warning": None,
        },
    )

    def fake_forward(
        drift,
        cint,
        initial_state,
        treat_idx,
        outcome_idx,
        *,
        mode,
        value=None,
        amount=None,
        dt,
        horizon_steps,
    ):
        del cint, initial_state, treat_idx, outcome_idx, mode, value, amount, dt
        baseline = jnp.arange(1, horizon_steps + 1, dtype=jnp.float32) + drift[0, 0]
        counterfactual = baseline + 10.0
        effect = jnp.full((horizon_steps,), drift[0, 0] + 1.0)
        return baseline, counterfactual, effect

    def fake_forward_latent(
        drift,
        cint,
        initial_state,
        treat_idx,
        *,
        mode,
        value=None,
        amount=None,
        dt,
        horizon_steps,
    ):
        del cint, initial_state, treat_idx, mode, value, amount, dt
        baseline = jnp.stack(
            [
                jnp.full((horizon_steps,), 0.5, dtype=jnp.float32),
                jnp.full((horizon_steps,), 1.0, dtype=jnp.float32),
            ],
            axis=1,
        )
        effect = jnp.stack(
            [
                jnp.full((horizon_steps,), 1.5, dtype=jnp.float32),
                jnp.full((horizon_steps,), drift[0, 0] + 1.0, dtype=jnp.float32),
            ],
            axis=1,
        )
        counterfactual = baseline + effect
        return baseline, counterfactual, effect

    monkeypatch.setattr(tool_server, "forward_simulate_action_from_state", fake_forward)
    monkeypatch.setattr(
        tool_server,
        "forward_simulate_latent_action_from_state",
        fake_forward_latent,
    )

    ctx = {
        "_fitted_artifact": SimpleNamespace(
            result=FakeResult(samples),
            builder=SimpleNamespace(
                _spec=SimpleNamespace(latent_names=["treat", "outcome"], manifest_names=[]),
                _model=object(),
            ),
            observation_support=None,
        ),
        "_prepared_runtime": SimpleNamespace(
            observations=jnp.zeros((3, 1)),
            times=jnp.array([0.0, 1.0, 2.0]),
        ),
        "_identifiable_treatments": ["treat"],
        "_outcome_name": "outcome",
        "_observation_timestamps": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
        "stage-1b": {"causal_spec": {"measurement": {"model_clock": "1d"}}},
        "stage-6": {},
    }
    args = {
        "action": {"variable": "treat", "mode": "shift", "amount": 1.0},
        "query": {"horizon_days": 3},
    }

    end_state = tool_server._execute_simulate_counterfactual(
        ctx,
        {
            **args,
            "query": {**args["query"], "estimand": "end_state"},
        },
    )["result"]
    trajectory = tool_server._execute_simulate_counterfactual(
        ctx,
        {
            **args,
            "query": {**args["query"], "estimand": "trajectory"},
        },
    )["result"]

    assert end_state["estimand"] == "end_state"
    assert end_state["summary"]["mean"] == pytest.approx(3.0)
    assert end_state["baseline_forecast_mean"] == pytest.approx(5.0)
    assert end_state["effect_trajectory"] is None
    assert "counterfactual_forecast_mean" not in end_state
    assert "temporal" not in end_state
    assert end_state["evidence"] == {
        "start_time": "2024-01-01T00:00:00+00:00",
        "end_time": "2024-01-03T00:00:00+00:00",
        "n_timepoints": 3,
        "variables": [],
        "conditioning_method": "kalman_smoother",
    }
    assert end_state["visualization"] == {
        "reference_node_trajectories": None,
        "action_node_trajectories": None,
        "node_effect_trajectories": None,
        "abducted_state": {"treat": 0.0, "outcome": 0.0},
    }

    assert trajectory["estimand"] == "trajectory"
    assert trajectory["summary"]["mean"] == pytest.approx(3.0)
    assert trajectory["effect_trajectory"] == [
        {"day": 1.0, "effect": 3.0},
        {"day": 2.0, "effect": 3.0},
        {"day": 3.0, "effect": 3.0},
    ]
    assert "temporal" not in trajectory
    assert trajectory["visualization"] == {
        "reference_node_trajectories": {
            "treat": [0.5, 0.5, 0.5],
            "outcome": [1.0, 1.0, 1.0],
        },
        "action_node_trajectories": {
            "treat": [2.0, 2.0, 2.0],
            "outcome": [4.0, 4.0, 4.0],
        },
        "node_effect_trajectories": {
            "treat": [1.5, 1.5, 1.5],
            "outcome": [3.0, 3.0, 3.0],
        },
        "abducted_state": {"treat": 0.0, "outcome": 0.0},
    }


def test_get_tool_schemas_exposes_declared_result_schema():
    client = TestClient(tool_server.app)

    response = client.get("/api/tools/stage-6")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["get_model_info"]["result"] is None
    assert tools["simulate_intervention"]["result"] is not None
    assert tools["simulate_counterfactual"]["result"] is not None


def test_manifest_effects_include_interval_supported_outcome_indicators():
    samples = {
        "lambda": jnp.array(
            [
                [
                    [0.0, -1.0],
                    [0.0, 0.5],
                ]
            ]
        )
    }

    effects = tool_server._manifest_effects(
        samples,
        outcome_idx=1,
        effect_mean=0.25,
        manifest_names=["sleep_problem_search_count", "sleep_duration_hours"],
    )

    assert effects == {
        "sleep_problem_search_count": pytest.approx(-0.25),
        "sleep_duration_hours": pytest.approx(0.125),
    }


def test_get_model_info_uses_estimation_projection_for_variables_and_treatments():
    ctx = {
        "stage-1b": {
            "causal_spec": {
                "latent": {
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
                        {"cause": "screen_time", "effect": "sleep"},
                        {"cause": "age", "effect": "sleep"},
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "daily_event_count",
                            "construct_name": "screen_time",
                            "measurement_dtype": "continuous",
                            "support_kind": "real",
                            "summary_operator": "mean",
                            "observation_window": "1d",
                        },
                        {
                            "name": "sleep_issue_searches",
                            "construct_name": "sleep",
                            "measurement_dtype": "continuous",
                            "support_kind": "real",
                            "summary_operator": "mean",
                            "observation_window": "1d",
                        },
                    ],
                },
                "estimation": {
                    "state_order": ["screen_time", "sleep"],
                    "edges": [{"cause": "screen_time", "effect": "sleep"}],
                    "induced_dependencies": [],
                },
            }
        },
        "stage-4b": {},
        "stage-5b": {"inference_metadata": {"method": "svi"}},
        "stage-6": {},
        "_prepared_runtime": SimpleNamespace(
            manifest_names=["daily_event_count", "sleep_issue_searches"]
        ),
        "_fitted_artifact": SimpleNamespace(
            builder=SimpleNamespace(_spec=SimpleNamespace(latent_names=["screen_time", "sleep"]))
        ),
        "_identifiable_treatments": ["screen_time"],
        "_outcome_name": "sleep",
        "_observation_timestamps": [],
    }

    payload = tool_server._build_model_info_payload(
        ctx,
        {"sections": ["overview", "variables", "capabilities"]},
    )

    assert payload["overview"]["treatments"] == ["screen_time"]
    assert [item["name"] for item in payload["variables"]["constructs"]] == ["screen_time", "sleep"]
    assert [item["name"] for item in payload["variables"]["indicators"]] == [
        "daily_event_count",
        "sleep_issue_searches",
    ]
    assert payload["capabilities"]["intervention"]["supported_treatments"] == ["screen_time"]
