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

    monkeypatch.setattr(tool_server, "forward_simulate_action_from_state", fake_forward)

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
    assert end_state["counterfactual_forecast_mean"] == pytest.approx(15.0)
    assert end_state["temporal"] is None
    assert end_state["effect_trajectory"] is None

    assert trajectory["estimand"] == "trajectory"
    assert trajectory["summary"]["mean"] == pytest.approx(3.0)
    assert trajectory["temporal"]["peak_effect"] == pytest.approx(3.0)
    assert trajectory["effect_trajectory"] == [
        {"day": 1.0, "effect": 3.0},
        {"day": 2.0, "effect": 3.0},
        {"day": 3.0, "effect": 3.0},
    ]
