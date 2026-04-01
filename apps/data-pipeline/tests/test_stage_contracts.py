"""Tests for stage payload contracts persisted to the web layer."""

from __future__ import annotations

import json
import logging
from copy import deepcopy

import pytest
from pydantic import ValidationError

from causal_ssm_agent.flows.stages.contracts import (
    STAGE_TOOLS,
    validate_stage_payload,
)
from causal_ssm_agent.flows.stages.persist import persist_web_result


@pytest.fixture
def valid_stage_payloads() -> dict[str, dict]:
    """Minimal valid payload for each persisted stage id."""
    return {
        "stage-0": {
            "column_descriptions": [
                {"name": "date", "description": "Date of observation"},
                {"name": "value", "description": "Numeric value"},
                {"name": "category", "description": "Category label"},
            ],
        },
        "stage-1a": {
            "latent_model": {
                "constructs": [
                    {
                        "name": "Perf",
                        "description": "Performance",
                        "role": "endogenous",
                        "is_outcome": True,
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "Stress",
                        "description": "Stress level",
                        "role": "endogenous",
                        "is_outcome": False,
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause": "Stress",
                        "effect": "Perf",
                        "description": "Stress reduces performance",
                        "lagged": True,
                    }
                ],
            },
        },
        "stage-1b": {
            "causal_spec": {
                "latent": {
                    "constructs": [
                        {
                            "name": "Perf",
                            "description": "Performance",
                            "role": "endogenous",
                            "is_outcome": True,
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "Stress",
                            "description": "Stress level",
                            "role": "endogenous",
                            "is_outcome": False,
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [
                        {
                            "cause": "Stress",
                            "effect": "Perf",
                            "description": "Stress reduces performance",
                            "lagged": True,
                        }
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "stress_score",
                            "construct_name": "Stress",
                            "how_to_measure": "Self-reported stress",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
            },
        },
        "stage-2": {
            "workers": [
                {
                    "worker_id": 0,
                    "status": "completed",
                    "n_extractions": 3,
                    "n_windows": 7,
                }
            ],
        },
        "stage-3": {
            "is_valid": True,
            "indicators": {
                "stress_score": {
                    "profile": {
                        "measurement_dtype": "continuous",
                        "n_obs": 10,
                        "mean": 3.2,
                        "std": 1.1,
                        "min": 1.0,
                        "max": 5.0,
                        "q25": 2.0,
                        "q50": 3.0,
                        "q75": 4.0,
                        "variance": 1.2,
                        "time_coverage_ratio": 1.0,
                        "max_gap_ratio": 0.2,
                        "dtype_violations": 0,
                        "duplicate_pct": 0.1,
                        "arithmetic_sequence_detected": False,
                        "n_unparseable_timestamps": 0,
                        "zero_fraction": 0.0,
                        "is_nonnegative": True,
                        "is_unit_interval": False,
                        "looks_integer_valued": True,
                        "variance_to_mean_ratio": 0.375,
                    },
                    "validation": {
                        "issues": [],
                        "checks": {
                            "n_obs": "ok",
                            "variance": "ok",
                            "n_unparseable_timestamps": "ok",
                            "time_coverage_ratio": "ok",
                            "max_gap_ratio": "ok",
                            "dtype_violations": "ok",
                            "duplicate_pct": "ok",
                            "arithmetic_sequence_detected": "ok",
                        },
                    },
                }
            },
            "dataset_issues": [],
        },
        "stage-4": {
            "model_spec": {
                "likelihoods": [
                    {
                        "variable": "stress_score",
                        "distribution": "gaussian",
                        "link": "identity",
                        "reasoning": "continuous variable",
                    }
                ],
                "parameters": [
                    {
                        "name": "rho_Stress",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR coefficient",
                    }
                ],
            },
            "authored_priors": {
                "rho_Stress": {
                    "parameter": "rho_Stress",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "sources": [],
                    "reasoning": "weakly informative",
                }
            },
            "resolved_priors": [
                {
                    "parameter": "rho_Stress",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "sources": [],
                    "reasoning": "weakly informative",
                },
                {
                    "parameter": "t0_mean_Stress",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 2.0},
                    "sources": [],
                    "reasoning": "Default weakly informative prior for the initial state mean of Stress.",
                },
                {
                    "parameter": "t0_sd_Stress",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 2.0},
                    "sources": [],
                    "reasoning": (
                        "Default weakly informative prior for the initial state standard deviation "
                        "of Stress."
                    ),
                },
            ],
            "prior_predictive_samples": {"stress_score": [0.1, -0.2, 0.3]},
        },
        "stage-4b": {
            "parametric_id": {
                "checked": True,
                "t_rule": {
                    "satisfies": True,
                    "n_free_params": 2,
                    "n_manifest": 1,
                    "n_timepoints": 10,
                    "n_moments": 10,
                    "param_counts": {"ar_coefficient": 1, "residual_sd": 1},
                },
                "summary": {"structural_issues": [], "boundary_issues": [], "weak_params": []},
            }
        },
        "stage-5b": {
            "power_scaling": [
                {
                    "parameter": "rho_Stress",
                    "diagnosis": "well_identified",
                    "prior_sensitivity": 0.2,
                    "likelihood_sensitivity": 0.8,
                }
            ],
            "ppc": {
                "per_variable_warnings": [],
                "checked": True,
                "overlays": [],
                "test_stats": [],
            },
            "inference_metadata": {
                "method": "svi",
                "n_samples": 1000,
                "duration_seconds": 1.2,
            },
            "mcmc_diagnostics": None,
            "svi_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        },
        "stage-6": {
            "intervention_results": [
                {
                    "treatment": "Stress",
                    "posterior_draws": [0.08, 0.11, 0.14, 0.09, 0.15, 0.12, 0.10, 0.13],
                }
            ],
            "saved_scenarios": [
                {
                    "label": "Stress shift",
                    "query": "simulate_intervention(shift=-0.5)",
                    "summary": "Negative stress shift improves the outcome in the forward simulation.",
                }
            ],
            "final_summary": "Stress reduction remains the dominant actionable lever.",
        },
    }


def test_persist_web_result_normalizes_nonfinite_numbers(tmp_path, monkeypatch, valid_stage_payloads):
    from causal_ssm_agent.flows.stages import persist as persist_module

    captured: dict[str, str] = {}

    monkeypatch.setattr(
        persist_module.storage,
        "join",
        lambda *parts: "/".join(str(part).strip("/") for part in parts if part),
    )
    monkeypatch.setattr(persist_module.storage, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        persist_module.storage,
        "write_text",
        lambda path, text: captured.setdefault(path, text),
    )

    payload = deepcopy(valid_stage_payloads["stage-5b"])
    payload["power_scaling"][0]["psis_k_hat"] = float("inf")

    result = persist_web_result.fn("stage-5b", payload, "run-123")

    assert result["power_scaling"][0]["psis_k_hat"] is None
    written = json.loads(next(iter(captured.values())))
    assert written["power_scaling"][0]["psis_k_hat"] is None


def test_tool_server_registry_matches_served_tool_contracts() -> None:
    """Served tool contracts should match the tool server registry exactly."""
    from causal_ssm_agent.tool_server import _TOOL_IMPLS

    served_stage_ids = {stage_id for stage_id, _tool_name in _TOOL_IMPLS}
    assert served_stage_ids == {"stage-1a", "stage-1b", "stage-2", "stage-4", "stage-6"}

    for stage_id in served_stage_ids:
        contract_names = {tool.name for tool in STAGE_TOOLS[stage_id]}
        runtime_names = {
            tool_name for served_stage_id, tool_name in _TOOL_IMPLS if served_stage_id == stage_id
        }
        assert runtime_names == contract_names


def test_validate_stage_payload_accepts_all_stages(valid_stage_payloads: dict[str, dict]):
    """Each stage payload validates and round-trips to a JSON-serializable dict."""
    for stage_id, payload in valid_stage_payloads.items():
        validated = validate_stage_payload(stage_id, payload)
        assert isinstance(validated, dict)


def test_validate_stage_payload_rejects_unknown_stage():
    """Unknown stage ids should fail fast."""
    with pytest.raises(ValueError, match="Unknown stage_id"):
        validate_stage_payload("stage-x", {})


def test_persist_web_result_rejects_missing_required_fields(valid_stage_payloads: dict[str, dict]):
    """Persistence task should fail on contract violations."""
    bad = deepcopy(valid_stage_payloads["stage-2"])
    bad.pop("workers")
    with pytest.raises(ValidationError):
        persist_web_result.fn("stage-2", bad, "run-123")


def test_persist_web_result_logs_stage5b_summary(
    valid_stage_payloads: dict[str, dict], caplog: pytest.LogCaptureFixture
):
    """Persistence task should emit a compact stage summary for UI-visible logs."""
    with caplog.at_level(logging.INFO, logger="causal_ssm_agent.flows.stages.persist"):
        persist_web_result.fn("stage-5b", valid_stage_payloads["stage-5b"], "run-123")

    assert any(
        record.levelno == logging.INFO
        and record.message
        == "Stage 5b summary: method=svi samples=1000 power_scaling_issues=0 ppc_warnings=0 outcome=success"
        for record in caplog.records
    )


def test_persist_web_result_logs_warning_summary_for_warn_stage(
    valid_stage_payloads: dict[str, dict], caplog: pytest.LogCaptureFixture
):
    """Warn/fail outcomes should be surfaced at warning level in persisted stage logs."""
    payload = deepcopy(valid_stage_payloads["stage-3"])
    payload["outcome"] = "warn"
    payload["indicators"]["stress_score"]["validation"]["issues"] = [
        {
            "indicator": "stress_score",
            "issue_type": "insufficient_coverage",
            "severity": "warning",
            "message": "Too few daily periods",
        }
    ]

    with caplog.at_level(logging.INFO, logger="causal_ssm_agent.flows.stages.persist"):
        persist_web_result.fn("stage-3", payload, "run-123")

    assert any(
        record.levelno == logging.WARNING
        and record.message
        == "Stage 3 summary: is_valid=True issues=1 errors=0 warnings=1 outcome=warn"
        for record in caplog.records
    )


def test_stage6_rejects_extra_fields(valid_stage_payloads: dict[str, dict]):
    """Extra fields on intervention results should be rejected (extra=forbid)."""
    bad = deepcopy(valid_stage_payloads["stage-6"])
    bad["intervention_results"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-6", bad)


def test_stage6_saved_scenarios_reject_extra_fields(valid_stage_payloads: dict[str, dict]):
    """Saved stage-6 scenario entries should remain schema-checked."""
    bad = deepcopy(valid_stage_payloads["stage-6"])
    bad["saved_scenarios"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-6", bad)


def test_outcome_warn_and_fail_accepted(valid_stage_payloads: dict[str, dict]):
    """Warn outcomes and fail outcomes with fail_reason should be accepted."""
    for stage_id, payload in valid_stage_payloads.items():
        warn_payload = deepcopy(payload)
        warn_payload["outcome"] = "warn"
        validated_warn = validate_stage_payload(stage_id, warn_payload)
        assert validated_warn["outcome"] == "warn"

        fail_payload = deepcopy(payload)
        fail_payload["outcome"] = "fail"
        fail_payload["fail_reason"] = "test_failure"
        validated_fail = validate_stage_payload(stage_id, fail_payload)
        assert validated_fail["outcome"] == "fail"
        assert validated_fail["fail_reason"] == "test_failure"


def test_outcome_invalid_value_rejected(valid_stage_payloads: dict[str, dict]):
    """outcome with an invalid literal should be rejected."""
    bad = deepcopy(valid_stage_payloads["stage-0"])
    bad["outcome"] = "invalid"
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-0", bad)


def test_fail_reason_must_match_fail_outcome(valid_stage_payloads: dict[str, dict]):
    missing_reason = deepcopy(valid_stage_payloads["stage-1b"])
    missing_reason["outcome"] = "fail"
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-1b", missing_reason)

    stray_reason = deepcopy(valid_stage_payloads["stage-1b"])
    stray_reason["outcome"] = "warn"
    stray_reason["fail_reason"] = "should_not_be_here"
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-1b", stray_reason)
