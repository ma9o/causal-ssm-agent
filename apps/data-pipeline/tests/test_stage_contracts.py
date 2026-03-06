"""Tests for stage payload contracts persisted to the web layer."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from causal_ssm_agent.flows.stages.contracts import validate_stage_payload
from causal_ssm_agent.flows.stages.persist import persist_web_result


@pytest.fixture
def valid_stage_payloads() -> dict[str, dict]:
    """Minimal valid payload for each persisted stage id."""
    return {
        "stage-0": {
            "source_label": "Test data export",
            "n_records": 10,
            "n_columns": 3,
            "date_range": {"start": "2024-01-01", "end": "2024-01-10"},
            "sample": [{"date": "2024-01-01", "value": "1.5", "category": "A"}],
            "column_descriptions": [
                {"name": "date", "dtype": "Utf8", "description": "Date of observation"},
                {"name": "value", "dtype": "Float64", "description": "Numeric value"},
                {"name": "category", "dtype": "Utf8", "description": "Category label"},
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
                        "temporal_scale": "daily",
                    },
                    {
                        "name": "Stress",
                        "description": "Stress level",
                        "role": "endogenous",
                        "is_outcome": False,
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
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
            "outcome_name": "Perf",
            "treatments": ["Stress"],
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
                            "temporal_scale": "daily",
                        },
                        {
                            "name": "Stress",
                            "description": "Stress level",
                            "role": "endogenous",
                            "is_outcome": False,
                            "temporal_status": "time_varying",
                            "temporal_scale": "daily",
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
            "measurement_model": {
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
        "stage-2": {
            "workers": [
                {
                    "worker_id": 0,
                    "status": "completed",
                    "n_extractions": 3,
                    "chunk_size": 20,
                }
            ],
            "combined_extractions_sample": [
                {"indicator": "stress_score", "value": 1.2, "timestamp": "2024-01-01T00:00:00Z"},
                {"indicator": "late_night", "value": True, "timestamp": "2024-01-02T00:00:00Z"},
            ],
            "per_indicator_counts": {"stress_score": 2, "late_night": 1},
        },
        "stage-3": {
            "validation_report": {
                "is_valid": True,
                "issues": [],
                "per_indicator_health": [
                    {
                        "indicator": "stress_score",
                        "n_obs": 10,
                        "variance": 1.2,
                        "time_coverage_ratio": 1.0,
                        "max_gap_ratio": 0.2,
                        "dtype_violations": 0,
                        "duplicate_pct": 0.1,
                        "arithmetic_sequence_detected": False,
                        "cell_statuses": {
                            "n_obs": "ok",
                            "variance": "ok",
                            "time_coverage_ratio": "ok",
                            "max_gap_ratio": "ok",
                            "dtype_violations": "ok",
                            "duplicate_pct": "ok",
                            "arithmetic_sequence_detected": "ok",
                        },
                    }
                ],
            }
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
                        "search_context": "stress autoregression",
                    }
                ],
            },
            "priors": {
                "rho_Stress": {
                    "parameter": "rho_Stress",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "sources": [],
                    "reasoning": "weakly informative",
                }
            },
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
        "stage-5": {
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
                    "effect_size": 0.12,
                    "posterior_draws": [0.08, 0.11, 0.14, 0.09, 0.15, 0.12, 0.10, 0.13],
                    "prob_positive": 0.97,
                    "identifiable": True,
                }
            ],
        },
    }


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
        persist_web_result.fn("stage-2", bad)


def test_stage6_rejects_extra_fields(valid_stage_payloads: dict[str, dict]):
    """Extra fields on intervention results should be rejected (extra=forbid)."""
    bad = deepcopy(valid_stage_payloads["stage-6"])
    bad["intervention_results"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-6", bad)


def test_outcome_warn_and_fail_accepted(valid_stage_payloads: dict[str, dict]):
    """outcome: 'warn' and 'fail' should be accepted on every stage."""
    for stage_id, payload in valid_stage_payloads.items():
        for value in ("warn", "fail"):
            p = deepcopy(payload)
            p["outcome"] = value
            validated = validate_stage_payload(stage_id, p)
            assert validated["outcome"] == value


def test_outcome_invalid_value_rejected(valid_stage_payloads: dict[str, dict]):
    """outcome with an invalid literal should be rejected."""
    bad = deepcopy(valid_stage_payloads["stage-0"])
    bad["outcome"] = "invalid"
    with pytest.raises(ValidationError):
        validate_stage_payload("stage-0", bad)


def test_gate_failed_removed_from_stage1b_and_stage4b(valid_stage_payloads: dict[str, dict]):
    """gate_failed field should no longer be accepted (extra=forbid)."""
    for stage_id in ("stage-1b", "stage-4b"):
        bad = deepcopy(valid_stage_payloads[stage_id])
        bad["gate_failed"] = True
        with pytest.raises(ValidationError):
            validate_stage_payload(stage_id, bad)
