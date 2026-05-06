"""Slow pipeline integration tests."""

import asyncio

import polars as pl
import pytest

from causal_ssm_agent.flows import stage_registry
from tests.pipeline.test_pipeline import _noop_artifact, _redirect_storage

pytestmark = pytest.mark.slow


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
        "estimation": {"state_order": ["stress"], "edges": [], "induced_dependencies": []},
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
