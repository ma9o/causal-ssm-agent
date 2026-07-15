"""Tests for artifact payload contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from nof1_causal_lab.flows.artifact_contracts import CONTEXT_TOOLS
from tests.artifact_contract_support import validate_artifact_payload


@pytest.fixture
def valid_artifact_payloads() -> dict[str, dict]:
    """Minimal valid payload for each persisted artifact id."""
    return {
        "raw_data": {
            "column_descriptions": [
                {"name": "date", "description": "Date of observation"},
                {"name": "value", "description": "Numeric value"},
                {"name": "category", "description": "Category label"},
            ],
        },
        "latent_structure": {
            "latent_structure": {
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
        "measurement_structure": {
            "measurement_structure": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "construct_name": "Stress",
                        "construct_polarity": "positive",
                        "how_to_measure": "Self-reported stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
        },
        "measurements": {
            "workers": [
                {
                    "worker_id": 0,
                    "status": "completed",
                    "n_extractions": 3,
                    "n_windows": 7,
                }
            ],
        },
        "validation_report": {
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
        "statistical_model_spec": {
            "statistical_model_spec": {
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
        "posterior": {
            "ppc": {
                "per_variable_warnings": [],
                "checked": True,
                "overlays": [],
                "test_stats": [],
            },
            "inference_metadata": {
                "method": "marginal_particle_gibbs",
                "n_samples": 1000,
                "duration_seconds": 1.2,
            },
            "mcmc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        },
        "baseline_report": {
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


def test_tool_server_registry_matches_served_tool_contracts() -> None:
    """Served tool contracts should match the tool server registry exactly."""
    from nof1_causal_lab.tool_server import _TOOL_IMPLS

    served_context_ids = {context_id for context_id, _tool_name in _TOOL_IMPLS}
    assert served_context_ids == {
        "latent-structure",
        "measurement-structure",
        "measurement",
        "statistical-model-spec",
        "ranking",
    }

    for context_id in served_context_ids:
        contract_names = {tool.name for tool in CONTEXT_TOOLS[context_id]}
        runtime_names = {
            tool_name
            for served_context_id, tool_name in _TOOL_IMPLS
            if served_context_id == context_id
        }
        assert runtime_names == contract_names


def test_validate_artifact_payload_accepts_all_artifacts(
    valid_artifact_payloads: dict[str, dict],
):
    """Each artifact payload validates and round-trips to a JSON-serializable dict."""
    for artifact_id, payload in valid_artifact_payloads.items():
        validated = validate_artifact_payload(artifact_id, payload)
        assert isinstance(validated, dict)


def test_validate_artifact_payload_rejects_unknown_artifact():
    """Unknown artifact ids should fail fast."""
    with pytest.raises(ValueError, match="Unknown artifact_id"):
        validate_artifact_payload("unknown_artifact", {})


def test_validate_artifact_payload_rejects_missing_required_fields(
    valid_artifact_payloads: dict[str, dict],
):
    """Artifact contract validation should fail on contract violations."""
    bad = deepcopy(valid_artifact_payloads["measurements"])
    bad.pop("workers")
    with pytest.raises(ValidationError):
        validate_artifact_payload("measurements", bad)


def test_baseline_report_rejects_extra_fields(valid_artifact_payloads: dict[str, dict]):
    """Extra fields on intervention results should be rejected (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["baseline_report"])
    bad["intervention_results"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_artifact_payload("baseline_report", bad)


def test_saved_scenarios_reject_extra_fields(valid_artifact_payloads: dict[str, dict]):
    """Saved scenario entries should remain schema-checked."""
    bad = deepcopy(valid_artifact_payloads["baseline_report"])
    bad["saved_scenarios"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_artifact_payload("baseline_report", bad)


def test_outcome_enum_no_longer_exists(valid_artifact_payloads: dict[str, dict]):
    """Contracts are pure artifacts: execution failure is a typed exception on
    the transition, never an outcome flag on the payload (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["measurement_structure"])
    bad["outcome"] = "fail"
    with pytest.raises(ValidationError):
        validate_artifact_payload("measurement_structure", bad)

    stray = deepcopy(valid_artifact_payloads["raw_data"])
    stray["fail_reason"] = "nope"
    with pytest.raises(ValidationError):
        validate_artifact_payload("raw_data", stray)
