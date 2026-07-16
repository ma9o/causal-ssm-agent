from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

import nof1_causal_lab.tool_server as tool_server
from nof1_causal_lab.artifacts import (
    CausalDesign,
    CausalEdge,
    Construct,
    IdentifiabilityStatus,
    IdentifiedTreatmentStatus,
    LatentStructure,
    MeasurementStructure,
    Role,
    TemporalStatus,
)
from nof1_causal_lab.models.causal_proofs import (
    CausalDesignRef,
    CertifiedCausalAnalysis,
    PosteriorProvenance,
    certify_identified_estimand,
    certify_reportable_posterior,
)
from nof1_causal_lab.models.ssm.inference import FittedArtifact, ParticleMCMCPosterior

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.types import InferenceDiagnostics


def _identified_causal_design(treatment: str, outcome: str) -> CausalDesign:
    return CausalDesign(
        latent=LatentStructure(
            constructs=[
                Construct(
                    name=treatment,
                    description="Treatment",
                    role=Role.EXOGENOUS,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
                Construct(
                    name=outcome,
                    description="Outcome",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
            ],
            edges=[CausalEdge(cause=treatment, effect=outcome, description="Test edge")],
        ),
        measurement=MeasurementStructure(indicators=[], model_clock="1d"),
        identifiability=IdentifiabilityStatus(
            identifiable_treatments={
                treatment: IdentifiedTreatmentStatus(
                    method="do_calculus",
                    estimand=f"E[{outcome} | do({treatment})]",
                )
            }
        ),
    )


def _certified_simulation_context(
    *,
    samples: dict[str, jnp.ndarray],
    spec: Any,
    runtime: Any,
    treatment: str,
    outcome: str,
    latent_paths: jnp.ndarray | None = None,
    timestamps: list[datetime] | None = None,
) -> dict[str, Any]:
    design = _identified_causal_design(treatment, outcome)
    design_ref = CausalDesignRef(workspace_id="test-workspace", version=1)
    diagnostics: InferenceDiagnostics = {}
    if latent_paths is not None:
        diagnostics["latent_paths"] = latent_paths
    artifact = FittedArtifact(
        result=ParticleMCMCPosterior(_samples=samples, diagnostics=diagnostics),
        spec=spec,
        times=runtime.times,
        provenance=PosteriorProvenance(
            causal_design=design_ref,
            compiled_ssm_version=1,
            panel_version=1,
        ),
    )
    analysis = CertifiedCausalAnalysis(
        causal_design=design,
        causal_design_ref=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                causal_design_ref=design_ref,
                treatment=treatment,
                outcome=outcome,
            ),
        ),
        posterior=certify_reportable_posterior(artifact),
    )
    return {
        "_fitted_artifact": artifact,
        "_causal_analysis": analysis,
        "_prepared_runtime": runtime,
        "_identifiable_treatments": [treatment],
        "_outcome_name": outcome,
        "_observation_timestamps": timestamps or [],
        "causal_design": {"causal_design": design.model_dump(mode="json")},
        "baseline_report": {},
    }


def test_execute_tool_rejects_invalid_input_before_invoking_tool(monkeypatch):
    client = TestClient(tool_server.app)
    called = False

    def fake_impl(_ctx, _args):
        nonlocal called
        called = True
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_execute_tool_surfaces_unexpected_exception_detail(monkeypatch):
    client = TestClient(tool_server.app)

    monkeypatch.setattr(tool_server, "_build_context", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        lambda _ctx, _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {"structure_json": "{}"}},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "message": "boom",
            "exception_type": "RuntimeError",
            "context_id": "latent-structure",
            "tool_name": "validate_latent_structure",
        }
    }


def test_build_ranking_context_rehydrates_runtime_from_persisted_spec(monkeypatch, tmp_path):
    import polars as pl

    from nof1_causal_lab.machine.moves import RunArtifact
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.utils import data as data_module
    from tests.ssm_spec_fixtures import block_ssm_spec, full_dense_matrix_dynamics_spec

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))

    spec = block_ssm_spec(
        n_latent=2,
        n_manifest=1,
        dynamics_spec=full_dense_matrix_dynamics_spec(2),
        latent_names=["screen_time", "sleep_quality"],
        manifest_names=["sleep_obs"],
    )
    design = _identified_causal_design("screen_time", "sleep_quality")
    fitted_artifact = FittedArtifact(
        result=ParticleMCMCPosterior(_samples={"vf_0_decay": jnp.zeros((1, 2), dtype=jnp.float32)}),
        spec=spec,
        times=jnp.array([0.0]),
        provenance=PosteriorProvenance(
            causal_design=CausalDesignRef(workspace_id="user-123", version=1),
            compiled_ssm_version=1,
            panel_version=1,
        ),
    )
    model_data = pl.DataFrame(
        {"indicator": ["sleep_obs"], "value": [1.0], "timestamp": ["2024-01-01T00:00:00"]}
    )

    store = ArtifactStore("user-123")
    latent_structure = store.write_version(
        "latent_structure",
        provenance="llm",
        derived_from={},
        produced_by="run:latent_structure",
        json_files={"latent-structure.json": {"latent_structure": {"constructs": []}}},
    )
    measurement_structure = store.write_version(
        "measurement_structure",
        provenance="llm",
        derived_from={},
        produced_by="run:measurement_structure",
        json_files={"measurement_structure.json": {"measurement_structure": {"indicators": []}}},
    )
    causal_design = store.write_version(
        "causal_design",
        provenance="llm",
        derived_from={
            "latent_structure": latent_structure.version,
            "measurement_structure": measurement_structure.version,
        },
        produced_by="run:measurement_structure",
        json_files={"causal_design.json": {"causal_design": design.model_dump(mode="json")}},
    )
    measurements = store.write_version(
        "measurements",
        provenance="computed",
        derived_from={},
        produced_by="run:measurements",
        json_files={"measurements.json": {"workers": []}},
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"causal_design": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": model_data},
    )
    identification_report = store.write_version(
        "identification_report",
        provenance="computed",
        derived_from={"causal_design": 1},
        produced_by="run:measurement_structure",
        json_files={
            "identification_report.json": {
                "outcome_name": "sleep_quality",
                "estimable_treatments": ["screen_time"],
                "non_identifiable_treatments": {},
            }
        },
    )
    posterior = store.write_version(
        "posterior",
        provenance="computed",
        derived_from={"panel": 1},
        produced_by="run:posterior",
        json_files={"diagnostics.json": {"outcome": "warn"}},
        pickle_files={"fitted.pkl": fitted_artifact},
    )
    journal = EpisodeJournal("user-123")
    for seq, move, produced in (
        (1, RunArtifact(artifact_id="latent_structure"), [latent_structure]),
        (
            2,
            RunArtifact(artifact_id="measurement_structure"),
            [measurement_structure, causal_design, identification_report],
        ),
        (3, RunArtifact(artifact_id="measurements"), [measurements, panel]),
        (4, RunArtifact(artifact_id="posterior"), [posterior]),
    ):
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                move=move,
                status="applied",
                produced=produced,
                trace_ids=[],
                resume=None,
            )
        )

    rebuilt_runtime = SimpleNamespace(
        observation_support="support-runtime",
        observation_data=None,
    )
    captured: dict[str, Any] = {}

    def fake_prepare_model_runtime(
        *, data_for_model, model, compiled_ssm=None, sampler_config=None
    ):
        del compiled_ssm, sampler_config
        captured["data_for_model"] = data_for_model
        captured["model"] = model
        return rebuilt_runtime

    monkeypatch.setattr(tool_server, "prepare_model_runtime", fake_prepare_model_runtime)

    ctx = tool_server._build_ranking_context("user-123")

    assert isinstance(captured["model"], tool_server.SSMModel)
    # The runtime is rebuilt from the unpickled fitted artifact's spec, on the
    # panel version pinned by the posterior's derived_from.
    assert captured["model"].spec is ctx["_fitted_artifact"].spec
    assert list(captured["model"].spec.latent_names) == ["screen_time", "sleep_quality"]
    assert captured["data_for_model"].equals(model_data)
    assert ctx["_fitted_artifact"].observation_support == "support-runtime"
    assert ctx["_prepared_runtime"] is rebuilt_runtime
    assert ctx["causal_design"] == {"causal_design": design.model_dump(mode="json")}
    assert ctx["posterior"] == {"outcome": "warn"}
    assert ctx["_outcome_name"] == "sleep_quality"
    assert ctx["_identifiable_treatments"] == ["screen_time"]
    # Every serving-chain artifact pins current versions: nothing is stale.
    assert ctx["_stale_artifacts"] == []


def test_simulate_counterfactual_respects_estimand_shape(monkeypatch):
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

    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=SimpleNamespace(
            dynamics_spec=spec,
            latent_names=["treat", "outcome"],
            manifest_names=[],
        ),
        runtime=runtime,
        treatment="treat",
        outcome="outcome",
        latent_paths=jnp.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]],
                [[0.0, 0.0], [4.0, 5.0], [6.0, 7.0]],
            ]
        ),
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
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
    """End-to-end: a vector-field particle posterior flows through
    ``_prepare_analysis_simulation`` and ``_execute_simulate_intervention``
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

    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=SimpleNamespace(
            dynamics_spec=spec,
            latent_names=["src", "tgt"],
            manifest_names=[],
        ),
        runtime=runtime,
        treatment="src",
        outcome="tgt",
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
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
    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=SimpleNamespace(
            dynamics_spec=spec,
            latent_names=["src", "tgt"],
            manifest_names=[],
        ),
        runtime=runtime,
        treatment="src",
        outcome="tgt",
        latent_paths=latent_paths,
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
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

    response = client.get("/api/tools/ranking")

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
        "causal_design": {
            "causal_design": {
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
        "posterior": {"inference_metadata": {"method": "marginal_particle_gibbs"}},
        "baseline_report": {},
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
