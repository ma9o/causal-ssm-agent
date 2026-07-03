from datetime import UTC, datetime
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

import nof1_causal_lab.tool_server as tool_server


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


def test_build_stage6_context_rehydrates_runtime_from_persisted_spec(monkeypatch, tmp_path):
    import polars as pl

    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.moves import RunStage
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.models.ssm.testing import block_ssm_spec, full_dense_matrix_dynamics_spec
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))

    spec = block_ssm_spec(
        n_latent=2,
        n_manifest=1,
        dynamics_spec=full_dense_matrix_dynamics_spec(2),
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=["sleep_obs"],
    )
    fitted_artifact = SimpleNamespace(
        spec=spec,
        observation_support=None,
    )
    model_data = pl.DataFrame(
        {"indicator": ["sleep_obs"], "value": [1.0], "timestamp": ["2024-01-01T00:00:00"]}
    )

    store = ArtifactStore("user-123")
    causal_spec_info = store.write_version(
        "causal_spec",
        provenance="llm",
        derived_from={},
        produced_by="stage-1b",
        json_files={
            "causal_spec.json": {"causal_spec": {"identifiability": {}, "measurement": {}}}
        },
    )
    model_data_info = store.write_version(
        "model_data",
        provenance="computed",
        derived_from={"causal_spec": 1},
        produced_by="stage-2",
        parquet_files={"model_data.parquet": model_data},
    )
    estimands_info = store.write_version(
        "estimands",
        provenance="computed",
        derived_from={"causal_spec": 1},
        produced_by="stage-1b",
        json_files={"estimands.json": {"treatments": ["screen_time"], "outcome": "sleep_quality"}},
    )
    posterior_info = store.write_version(
        "posterior",
        provenance="computed",
        derived_from={"causal_spec": 1, "model_data": 1},
        produced_by="stage-5b",
        json_files={"diagnostics.json": {"outcome": "warn"}},
        pickle_files={"fitted.pkl": fitted_artifact},
    )
    EpisodeJournal("user-123").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-03T00:00:00+00:00",
            move=RunStage(stage_id="stage-5b"),
            status="applied",
            produced=[posterior_info],
            state_after=EpisodeState().with_versions(
                [causal_spec_info, model_data_info, estimands_info, posterior_info]
            ),
        )
    )

    rebuilt_runtime = SimpleNamespace(
        observation_support="support-runtime",
        observation_data=None,
    )
    captured: dict[str, object] = {}

    def fake_prepare_model_runtime(
        *, data_for_model, model, compiled_ssm=None, sampler_config=None
    ):
        del compiled_ssm, sampler_config
        captured["data_for_model"] = data_for_model
        captured["model"] = model
        return rebuilt_runtime

    monkeypatch.setattr(tool_server, "prepare_model_runtime", fake_prepare_model_runtime)
    monkeypatch.setattr(tool_server, "get_outcome_name", lambda _spec: "sleep_quality")

    ctx = tool_server._build_stage6_context("user-123")

    assert isinstance(captured["model"], tool_server.SSMModel)
    # The runtime is rebuilt from the unpickled fitted artifact's spec, on the
    # model_data version pinned by the posterior's derived_from.
    assert captured["model"].spec is ctx["_fitted_artifact"].spec
    assert list(captured["model"].spec.latent_names) == ["screen_time", "sleep_quality"]
    assert captured["data_for_model"].equals(model_data)
    assert ctx["_fitted_artifact"].observation_support == "support-runtime"
    assert ctx["_prepared_runtime"] is rebuilt_runtime
    assert ctx["stage-1b"] == {"causal_spec": {"identifiability": {}, "measurement": {}}}
    assert ctx["stage-5b"] == {"outcome": "warn"}
    assert ctx["_outcome_name"] == "sleep_quality"
    assert ctx["_identifiable_treatments"] == ["screen_time"]
    # Every serving-chain artifact pins current versions: nothing is stale.
    assert ctx["_stale_artifacts"] == []


def test_execute_submit_priors_loads_stage2_runtime_via_stage_registry(monkeypatch, tmp_path):
    import polars as pl

    import nof1_causal_lab.flows.stages.stage4.grounding as stage4_grounding_module
    import nof1_causal_lab.flows.stages.stage4.tool_registry as stage4_tool_registry
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_feedback import (
        make_stage4_grounding_result,
    )
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.moves import RunStage
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))

    store = ArtifactStore("user-123")
    model_data_info = store.write_version(
        "model_data",
        provenance="computed",
        derived_from={},
        produced_by="stage-2",
        parquet_files={"model_data.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
    )
    EpisodeJournal("user-123").append(
        TransitionRecord(
            seq=1,
            ts="2026-07-03T00:00:00+00:00",
            move=RunStage(stage_id="stage-2"),
            status="applied",
            produced=[model_data_info],
            state_after=EpisodeState().with_versions([model_data_info]),
        )
    )

    expected_data_for_model = object()
    captured: dict[str, object] = {}

    def fake_load_parquet(path):
        # The registry resolves the episode's CURRENT model_data version.
        assert path == store.file_path("model_data", 1, "model_data.parquet")
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
        return make_stage4_grounding_result(
            stage_output={"model_spec": {}},
            status="accepted",
            feedback="VALID",
            retain_for_next_prompt=False,
            capture_stage_output=True,
        )

    monkeypatch.setattr(stage4_tool_registry, "load_parquet", fake_load_parquet)
    monkeypatch.setattr(
        stage4_tool_registry,
        "_load_stage4_current",
        lambda workspace_id: {"workspace_id": workspace_id, "model_spec": {"parameters": []}},
    )
    monkeypatch.setattr(stage4_grounding_module, "stage4_grounding", fake_stage4_grounding)

    result = tool_server._execute_submit_priors(
        {
            "_workspace_id": "user-123",
            "stage-1b": {"causal_spec": {"latent": {"constructs": []}}},
        },
        {"priors_json": "{}"},
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
            self.method = "marginal_particle_gibbs"
            self._latent_paths = jnp.array(
                [
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]],
                    [[0.0, 0.0], [4.0, 5.0], [6.0, 7.0]],
                ]
            )
            self.diagnostics = {
                "latent_paths": self._latent_paths,
            }

        def get_samples(self):
            return self._samples

        def get_latent_paths(self):
            return self._latent_paths

    from nof1_causal_lab.models.ssm.dynamics import (
        DiagonalDecaySpec,
        DynamicsSpec,
        HillEdgeSpec,
    )

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            DiagonalDecaySpec(),
            HillEdgeSpec(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 2
    samples = {
        "vf_0_decay": jnp.tile(jnp.array([0.5, 0.5]), (n_draws, 1)),
        "vf_1_Emax": jnp.full((n_draws,), 1.5),
        "vf_1_EC50": jnp.full((n_draws,), 1.0),
        "vf_1_n": jnp.full((n_draws,), 2.0),
    }
    captured_initial_states: list[jnp.ndarray] = []

    def fake_vmap_simulate(
        vector_field,
        param_samples,
        initial_states,
        clamps,
        *,
        time_grid,
    ):
        del vector_field, clamps
        captured_initial_states.append(initial_states)
        n_draws = len(param_samples)
        n_t = time_grid.shape[0]
        # n_latent = 2 in this test setup
        n_latent = 2
        baseline_per_t = jnp.array([0.5, 5.0], dtype=jnp.float32)
        effect_per_t = jnp.array([1.5, 3.0], dtype=jnp.float32)
        baseline = jnp.broadcast_to(baseline_per_t, (n_draws, n_t, n_latent))
        effect = jnp.broadcast_to(effect_per_t, (n_draws, n_t, n_latent))
        counterfactual = baseline + effect
        return baseline, counterfactual, effect

    monkeypatch.setattr(tool_server, "vmap_simulate_clamps_from_state", fake_vmap_simulate)

    ctx = {
        "_fitted_artifact": SimpleNamespace(
            result=FakeResult(samples),
            spec=SimpleNamespace(
                dynamics_spec=spec,
                latent_names=["treat", "outcome"],
                manifest_names=[],
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
        "start": {"kind": "abducted"},
        "clamps": [{"variable": "treat", "mode": "shift", "amount": 1.0}],
        "query": {"horizon_days": 3},
    }

    expected_visualization = {
        "reference_node_trajectories": {
            "treat": [0.5, 0.5, 0.5],
            "outcome": [5.0, 5.0, 5.0],
        },
        "action_node_trajectories": {
            "treat": [2.0, 2.0, 2.0],
            "outcome": [8.0, 8.0, 8.0],
        },
        "node_effect_trajectories": {
            "treat": [1.5, 1.5, 1.5],
            "outcome": [3.0, 3.0, 3.0],
        },
        "start_state": {"treat": 4.0, "outcome": 5.0},
    }
    expected_start = {
        "kind": "abducted",
        "time_index": 2,
        "time": "2024-01-03T00:00:00+00:00",
        "state_source": "fitted_latent_paths",
    }

    end_state = tool_server._execute_simulate(
        ctx,
        {**args, "query": {**args["query"], "estimand": "end_state"}},
    )["result"]
    trajectory = tool_server._execute_simulate(
        ctx,
        {**args, "query": {**args["query"], "estimand": "trajectory"}},
    )["result"]

    assert end_state["estimand"] == "end_state"
    assert end_state["summary"]["mean"] == pytest.approx(3.0)
    assert end_state["reference_mean"] == pytest.approx(5.0)
    assert end_state["effect_trajectory"] is None
    assert "rung" not in end_state
    assert end_state["start"] == expected_start
    assert end_state["clamps"] == args["clamps"]
    assert end_state["visualization"] == expected_visualization
    assert all(
        jnp.array_equal(initial_states, jnp.array([[2.0, 3.0], [6.0, 7.0]]))
        for initial_states in captured_initial_states
    )

    assert trajectory["estimand"] == "trajectory"
    assert trajectory["summary"]["mean"] == pytest.approx(3.0)
    assert trajectory["reference_mean"] == pytest.approx(5.0)
    assert trajectory["effect_trajectory"] == [
        {"day": 1.0, "effect": 3.0},
        {"day": 2.0, "effect": 3.0},
        {"day": 3.0, "effect": 3.0},
    ]
    assert trajectory["start"] == expected_start
    assert trajectory["visualization"] == expected_visualization


def test_simulate_intervention_dispatches_to_vector_field_path():
    """End-to-end: a vector-field-fitted InferenceResult flows through
    ``_prepare_stage6_simulation`` and ``_execute_simulate_intervention``
    without touching the affine deterministic sample shape.

    Pins the Phase E dispatch: the tool_server endpoint inspects
    ``result.method`` and routes to ``vmap_*_dynamics`` helpers when
    the fit came from the vector-field driver.
    """

    from nof1_causal_lab.models.ssm.dynamics import (
        DiagonalDecaySpec,
        DynamicsSpec,
        HillEdgeSpec,
    )

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            DiagonalDecaySpec(),
            HillEdgeSpec(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 4
    samples = {
        "vf_0_decay": jnp.tile(jnp.array([0.5, 0.5]), (n_draws, 1)),
        "vf_1_Emax": jnp.full((n_draws,), 1.5),
        "vf_1_EC50": jnp.full((n_draws,), 1.0),
        "vf_1_n": jnp.full((n_draws,), 2.0),
    }

    fake_result = SimpleNamespace(
        method="marginal_particle_gibbs",
        diagnostics={},
        get_samples=lambda: samples,
    )

    ctx = {
        "_fitted_artifact": SimpleNamespace(
            result=fake_result,
            spec=SimpleNamespace(
                dynamics_spec=spec,
                latent_names=["src", "tgt"],
                manifest_names=[],
            ),
            observation_support=None,
        ),
        "_prepared_runtime": SimpleNamespace(
            observations=jnp.zeros((3, 1)),
            times=jnp.array([0.0, 1.0, 2.0]),
        ),
        "_identifiable_treatments": ["src"],
        "_outcome_name": "tgt",
        "_observation_timestamps": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
        "stage-1b": {"causal_spec": {"measurement": {"model_clock": "1d"}}},
        "stage-6": {},
    }
    args = {
        "start": {"kind": "baseline"},
        "clamps": [{"variable": "src", "mode": "shift", "amount": 0.5}],
        "query": {"horizon_days": 3, "estimand": "end_state"},
    }

    response = tool_server._execute_simulate(ctx, args)
    result = response["result"]
    assert "error" not in result, f"vector-field dispatch failed: {result}"
    assert result["start"]["kind"] == "baseline"
    assert result["estimand"] == "end_state"
    # Effect on tgt from shifting src up should be positive (Hill saturates).
    summary = result["summary"]
    assert summary["mean"] > 0
    # Almost-certainly-positive contrast — Hill is monotonic in src
    assert summary["prob_positive"] == pytest.approx(1.0)


def test_simulate_counterfactual_dispatches_to_vector_field_path():
    """Rung-3 on vector-field starts from retained fitted trajectory draws."""

    from nof1_causal_lab.models.ssm.dynamics import (
        DiagonalDecaySpec,
        DynamicsSpec,
        HillEdgeSpec,
    )

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            DiagonalDecaySpec(),
            HillEdgeSpec(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 3
    samples = {
        "vf_0_decay": jnp.tile(jnp.array([0.5, 0.5]), (n_draws, 1)),
        "vf_1_Emax": jnp.full((n_draws,), 1.5),
        "vf_1_EC50": jnp.full((n_draws,), 1.0),
        "vf_1_n": jnp.full((n_draws,), 2.0),
    }
    latent_paths = jnp.tile(jnp.array([[1.0, 0.5], [0.9, 0.6], [0.8, 0.7]]), (n_draws, 1, 1))
    fake_result = SimpleNamespace(
        method="marginal_particle_gibbs",
        diagnostics={
            "latent_paths": latent_paths,
        },
        get_samples=lambda: samples,
    )

    ctx = {
        "_fitted_artifact": SimpleNamespace(
            result=fake_result,
            spec=SimpleNamespace(
                dynamics_spec=spec,
                latent_names=["src", "tgt"],
                manifest_names=[],
            ),
            observation_support=None,
        ),
        "_prepared_runtime": SimpleNamespace(
            observations=jnp.zeros((3, 1)),
            times=jnp.array([0.0, 1.0, 2.0]),
        ),
        "_identifiable_treatments": ["src"],
        "_outcome_name": "tgt",
        "_observation_timestamps": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
        "stage-1b": {"causal_spec": {"measurement": {"model_clock": "1d"}}},
        "stage-6": {},
    }
    args = {
        "start": {"kind": "abducted"},
        "clamps": [{"variable": "src", "mode": "shift", "amount": 0.5}],
        "query": {"horizon_days": 3, "estimand": "end_state"},
    }

    response = tool_server._execute_simulate(ctx, args)
    result = response["result"]
    assert "error" not in result, f"vector-field abducted start failed: {result}"
    assert result["estimand"] == "end_state"
    assert result["start"] == {
        "kind": "abducted",
        "time_index": 2,
        "time": "2024-01-03T00:00:00+00:00",
        "state_source": "fitted_latent_paths",
    }
    # Shift on src should produce a positive effect on tgt via the Hill chain
    assert result["summary"]["mean"] > 0


def test_get_tool_schemas_exposes_declared_result_schema():
    client = TestClient(tool_server.app)

    response = client.get("/api/tools/stage-6")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["get_model_info"]["result"] is None
    assert tools["simulate"]["result"] is not None


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
        "stage-5b": {"inference_metadata": {"method": "marginal_particle_gibbs"}},
        "stage-6": {},
        "_prepared_runtime": SimpleNamespace(
            manifest_names=["daily_event_count", "sleep_issue_searches"]
        ),
        "_fitted_artifact": SimpleNamespace(
            spec=SimpleNamespace(latent_names=["screen_time", "sleep"])
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
    assert payload["capabilities"]["simulate"]["supported_targets"] == ["screen_time"]
