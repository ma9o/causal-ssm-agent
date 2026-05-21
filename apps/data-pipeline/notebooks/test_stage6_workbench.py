from types import SimpleNamespace

import pytest
import stage6_workbench
from stage6_workbench import Stage6Session, Stage6Workbench


def _session() -> Stage6Session:
    artifact = SimpleNamespace(
        builder=SimpleNamespace(
            spec=SimpleNamespace(
                latent_names=["screen_time", "sleep_quality"],
                manifest_names=["sleep_obs"],
            )
        )
    )
    return Stage6Session.from_context(
        {
            "_identifiable_treatments": ["screen_time"],
            "_outcome_name": "sleep_quality",
            "_fitted_artifact": artifact,
        }
    )


def test_stage6_session_runs_intervention_scenario(monkeypatch):
    session = _session()
    captured: dict[str, object] = {}

    def fake_execute(_ctx, args):
        captured.update(args)
        return {
            "result": {
                "rung": 2,
                "action": args["action"],
                "outcome": args["outcome"],
                "estimand": args["query"]["estimand"],
                "baseline_treatment_mean": 0.0,
                "summary": {
                    "mean": 0.2,
                    "median": 0.2,
                    "lower_95": 0.1,
                    "upper_95": 0.3,
                    "prob_positive": 1.0,
                },
            }
        }

    monkeypatch.setattr(
        stage6_workbench.tool_server, "_execute_simulate_intervention", fake_execute
    )

    scenario = session.intervention(
        name="less screen time",
        action=session.variable("screen_time").shift(-0.5),
        outcome="sleep_quality",
        estimand="trajectory",
        horizon_days=14,
        projection="both",
    )
    result = session.run(scenario)

    assert captured == {
        "action": {
            "variable": "screen_time",
            "mode": "shift",
            "value": None,
            "amount": -0.5,
        },
        "outcome": "sleep_quality",
        "query": {"estimand": "trajectory", "horizon_days": 14, "projection": "both"},
    }
    assert result.summary["mean"] == 0.2


def test_stage6_session_runs_counterfactual_scenario(monkeypatch):
    session = _session()
    captured: dict[str, object] = {}

    def fake_execute(_ctx, args):
        captured.update(args)
        return {
            "result": {
                "rung": 3,
                "evidence": {
                    "start_time": "2026-01-01T00:00:00+00:00",
                    "end_time": "2026-01-07T00:00:00+00:00",
                    "n_timepoints": 7,
                    "variables": ["screen_time"],
                    "conditioning_method": "kalman_smoother",
                },
                "action": args["action"],
                "outcome": args["outcome"],
                "estimand": args["query"]["estimand"],
                "baseline_forecast_mean": 0.0,
                "summary": {
                    "mean": -0.1,
                    "median": -0.1,
                    "lower_95": -0.2,
                    "upper_95": 0.0,
                    "prob_positive": 0.05,
                },
            }
        }

    monkeypatch.setattr(
        stage6_workbench.tool_server, "_execute_simulate_counterfactual", fake_execute
    )

    scenario = session.counterfactual(
        name="what if lower screen time",
        action=session.variable("screen_time").set(0.0),
        outcome="sleep_quality",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-07T00:00:00Z",
        variables=["screen_time"],
        estimand="end_state",
        horizon_days=30,
        projection="latent",
    )
    result = session.run(scenario)

    assert captured == {
        "evidence": {
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-07T00:00:00Z",
            "variables": ["screen_time"],
        },
        "action": {
            "variable": "screen_time",
            "mode": "set",
            "value": 0.0,
            "amount": None,
        },
        "outcome": "sleep_quality",
        "query": {"estimand": "end_state", "horizon_days": 30, "projection": "latent"},
    }
    assert result.summary["prob_positive"] == 0.05


def test_stage6_session_raises_tool_error(monkeypatch):
    session = _session()
    monkeypatch.setattr(
        stage6_workbench.tool_server,
        "_execute_simulate_intervention",
        lambda _ctx, _args: {"result": {"error": "not identifiable"}},
    )

    scenario = session.intervention(
        action=session.variable("screen_time").shift(1.0),
        outcome="sleep_quality",
    )

    with pytest.raises(ValueError, match="not identifiable"):
        session.run(scenario)


def test_stage6_workbench_constructs_widget_tree():
    workbench = Stage6Workbench(_session())

    assert workbench.treatment.value == "screen_time"
    assert workbench.outcome.value == "sleep_quality"
    assert workbench.kind.value == "intervention"
