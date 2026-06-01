"""Tests for prior predictive validation checks.

Covers: _check_nan_inf, _check_constraint_violations, _check_extreme_values,
compute_data_stats, format_validation_report, format_parameter_feedback,
get_failed_parameters.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily, PriorDistributionFamily
from nof1_causal_lab.models.prior_predictive import (
    _check_constraint_violations,
    _check_extreme_values,
    _check_lagged_response_plausibility,
    _check_nan_inf,
    _check_scale_plausibility,
    _infer_dynamics_repair_scope,
    compute_data_stats,
    format_parameter_feedback,
    format_validation_report,
    get_failed_parameters,
    resolve_scale_target_parameters,
)
from nof1_causal_lab.models.ssm.compile.artifact import serialize_edge_lag_days, serialize_ssm_spec
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
)
from nof1_causal_lab.models.ssm.parameterization import compile_prior_semantics
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    sample_prior_predictive_from_compiled_semantics,
)
from nof1_causal_lab.models.ssm.priors import PriorSpec
from nof1_causal_lab.models.ssm.structure import SparseMatrixBlockSpec, T0CholBlockSpec
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.testing import (
    block_ssm_spec,
    dense_matrix_dynamics_spec,
    diagonal_diffusion_block,
    full_diagonal_support,
    prior_registry,
)
from nof1_causal_lab.workers.schemas_prior import PriorValidationResult


def _require_result(result: PriorValidationResult | None) -> PriorValidationResult:
    assert result is not None
    return result


def _require_text(value: str | None) -> str:
    assert value is not None
    return value


# =============================================================================
# _check_nan_inf
# =============================================================================


class TestCheckNanInf:
    def test_clean_samples_no_issue(self):
        samples = {"vf_0_decay": jnp.array([1.0, 2.0, 3.0])}
        assert _check_nan_inf(samples) is None

    def test_nan_detected(self):
        samples = {"vf_0_decay": jnp.array([1.0, float("nan"), 3.0])}
        result = _require_result(_check_nan_inf(samples))
        assert not result.is_valid
        assert "vf_0_decay" in _require_text(result.issue)

    def test_inf_detected(self):
        samples = {"x": jnp.array([float("inf")])}
        result = _require_result(_check_nan_inf(samples))
        assert not result.is_valid
        assert "x" in _require_text(result.issue)

    def test_multiple_bad_sites(self):
        samples = {
            "a": jnp.array([float("nan")]),
            "b": jnp.array([float("inf")]),
            "c": jnp.array([1.0]),
        }
        result = _require_result(_check_nan_inf(samples))
        assert "a" in _require_text(result.issue)
        assert "b" in _require_text(result.issue)

    def test_likelihood_diagnostics_ignored(self):
        samples = {
            "ll_per_timestep": jnp.array([float("-inf")]),
            "log_likelihood": jnp.array([float("nan")]),
            "vf_0_decay": jnp.array([0.1, 0.2]),
        }
        assert _check_nan_inf(samples) is None

    def test_observations_report_failure_stage_and_manifest_details(self):
        observations = jnp.asarray(
            [
                [[1.0, 2.0], [3.0, float("inf")], [5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [float("nan"), 6.0]],
            ]
        )
        result = _check_nan_inf(
            {"observations": observations},
            manifest_names=["activity_vas", "sleep_quality"],
        )

        assert result is not None
        assert result.failure_stage == "observation_sample"
        assert result.bad_sample_sites == ["observations"]
        assert result.bad_manifest_names == ["activity_vas", "sleep_quality"]
        assert result.failing_draw_indices == [0, 1]
        assert result.first_bad_time_index == 1
        assert result.pathology_certificate is not None
        assert result.pathology_certificate.kind == "nonfinite_samples"
        assert result.pathology_certificate.primary_score == pytest.approx(1.0)

    def test_observations_mask_ignores_structural_nans(self):
        observations = jnp.asarray(
            [
                [[float("nan"), float("nan")], [1.0, 2.0]],
                [[float("nan"), float("nan")], [3.0, 4.0]],
            ]
        )
        observations_mask = jnp.asarray(
            [
                [[False, False], [True, True]],
                [[False, False], [True, True]],
            ]
        )

        result = _check_nan_inf(
            {
                "observations": observations,
                "observations_mask": observations_mask,
            },
            manifest_names=["activity_vas", "sleep_quality"],
        )

        assert result is None


# =============================================================================
# _check_constraint_violations
# =============================================================================


class TestCheckConstraintViolations:
    def test_all_positive_no_issue(self):
        samples = {"diffusion_diag_free": jnp.array([0.1, 0.5, 1.0])}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_negative_values_detected(self):
        # 50% negative → above 5% threshold
        samples = {"diffusion_diag_free": jnp.array([-1.0, -2.0, 1.0, 2.0])}
        results = _check_constraint_violations(samples)
        assert len(results) == 1
        assert "diffusion_diag_free" in results[0].parameter
        assert "negative" in _require_text(results[0].issue)

    def test_below_threshold_no_issue(self):
        # 1% negative → below 5% threshold
        values = jnp.concatenate([jnp.array([-0.001]), jnp.ones(99)])
        samples = {"manifest_var_diag_free": values}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_non_positive_site_ignored(self):
        samples = {"vf_1_weight": jnp.array([-1.0, -2.0])}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_missing_site_ignored(self):
        results = _check_constraint_violations({"other_site": jnp.ones(5)})
        assert results == []

    def test_custom_threshold(self):
        # 25% negative → above 10% custom threshold
        samples = {"t0_var_diag_free": jnp.array([-1.0, 1.0, 1.0, 1.0])}
        results = _check_constraint_violations(samples, threshold=0.1)
        assert len(results) == 1


# =============================================================================
# _check_extreme_values
# =============================================================================


class TestCheckExtremeValues:
    def test_normal_values_no_issue(self):
        samples = {"vf_0_decay": jnp.array([0.5, 0.3, 1.0])}
        results = _check_extreme_values(samples)
        assert results == []

    def test_extreme_values_detected(self):
        # All extreme
        samples = {"vf_0_decay": jnp.array([1e7, 1e8, 1e9])}
        results = _check_extreme_values(samples)
        assert len(results) == 1
        assert "extreme" in _require_text(results[0].issue).lower()

    def test_below_threshold_no_issue(self):
        # Only 1 out of 100 extreme → 1% < 10% threshold
        values = jnp.concatenate([jnp.array([1e7]), jnp.ones(99)])
        samples = {"vf_0_decay": values}
        results = _check_extreme_values(samples)
        assert results == []

    def test_non_param_site_ignored(self):
        # Sites not matching param patterns are skipped
        samples = {"dynamics": jnp.array([1e7, 1e8])}
        results = _check_extreme_values(samples)
        assert results == []


# =============================================================================
# format_validation_report
# =============================================================================


class TestFormatValidationReport:
    def test_passed_report_contains_no_failures(self):
        report = format_validation_report(True, [])
        assert "PASSED" in report
        assert "FAILED" not in report

    def test_failed_report_includes_each_issue_and_parameter(self):
        results = [
            PriorValidationResult(
                parameter="vf_0_decay",
                is_valid=False,
                issue="Too extreme",
                suggested_adjustment="Fix it",
            ),
            PriorValidationResult(
                parameter="diffusion_diag_free",
                is_valid=False,
                issue="Negative values",
                suggested_adjustment="Use positive prior",
            ),
        ]
        report = format_validation_report(False, results)
        assert "FAILED" in report
        assert "vf_0_decay" in report
        assert "Too extreme" in report
        assert "diffusion_diag_free" in report
        assert "Negative values" in report
        assert "Use positive prior" in report or "Negative values" in report


# =============================================================================
# format_parameter_feedback
# =============================================================================


class TestFormatParameterFeedback:
    def test_no_relevant_failures(self):
        results = [
            PriorValidationResult(parameter="other_param", is_valid=False, issue="unrelated")
        ]
        feedback = format_parameter_feedback("my_param", results)
        assert feedback == ""

    def test_relevant_failure_included(self):
        results = [
            PriorValidationResult(
                parameter="prior_predictive",
                is_valid=False,
                issue="NaN detected",
                suggested_adjustment="Fix priors",
            )
        ]
        feedback = format_parameter_feedback("dynamics", results)
        assert "NaN detected" in feedback
        assert "Fix priors" in feedback

    def test_prior_shown_when_provided(self):
        results = [
            PriorValidationResult(
                parameter="prior_predictive",
                is_valid=False,
                issue="NaN",
            )
        ]
        feedback = format_parameter_feedback(
            "dynamics",
            results,
            prior={"distribution": "Normal", "params": {"mu": 0.0, "sigma": 1.0}},
        )
        assert "Normal" in feedback
        assert "mu=0.0" in feedback

    def test_scale_mismatch_only_surfaces_for_targeted_scale_parameter(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "monthly_eveningness_activity_timing",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                }
            ],
            "parameters": [
                {
                    "name": "t0_mean_chronotype",
                    "role": "initial_state_mean",
                    "construct": "chronotype",
                },
                {
                    "name": "t0_sd_chronotype",
                    "role": "initial_state_sd",
                    "construct": "chronotype",
                },
                {
                    "name": "manifest_mean_monthly_eveningness_activity_timing",
                    "role": "observation_intercept",
                    "indicator": "monthly_eveningness_activity_timing",
                    "construct": "chronotype",
                },
            ],
        }
        results = [
            PriorValidationResult(
                parameter="scale_monthly_eveningness_activity_timing",
                is_valid=False,
                issue="Scale mismatch for monthly_eveningness_activity_timing",
                suggested_adjustment="Adjust diffusion/dynamics priors to match data scale",
            )
        ]

        assert (
            format_parameter_feedback(
                "t0_mean_chronotype",
                results,
                model_spec=model_spec,
            )
            == ""
        )
        assert (
            format_parameter_feedback(
                "manifest_mean_monthly_eveningness_activity_timing",
                results,
                model_spec=model_spec,
            )
            == ""
        )
        feedback = format_parameter_feedback(
            "t0_sd_chronotype",
            results,
            model_spec=model_spec,
        )
        assert "Scale mismatch for monthly_eveningness_activity_timing" in feedback


# =============================================================================
# get_failed_parameters
# =============================================================================


class TestGetFailedParameters:
    def test_no_failures_returns_empty(self):
        results = [PriorValidationResult(parameter="x", is_valid=True)]
        assert get_failed_parameters(results, ["param_a", "param_b"]) == []

    def test_global_failure_returns_all(self):
        results = [PriorValidationResult(parameter="prior_predictive", is_valid=False, issue="NaN")]
        params = ["alpha", "beta"]
        failed = get_failed_parameters(results, params)
        assert set(failed) == {"alpha", "beta"}

    def test_scale_mismatch_with_model_spec_targets_variance_parameter(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "monthly_eveningness_activity_timing",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                }
            ],
            "parameters": [
                {
                    "name": "t0_mean_chronotype",
                    "role": "initial_state_mean",
                    "construct": "chronotype",
                },
                {
                    "name": "t0_sd_chronotype",
                    "role": "initial_state_sd",
                    "construct": "chronotype",
                },
                {
                    "name": "beta_chronotype_sleep_quality",
                    "role": "fixed_effect",
                    "cause": "chronotype",
                    "effect": "sleep_quality",
                },
            ],
        }
        causal_spec = {
            "measurement": {
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                    }
                ]
            }
        }
        results = [
            PriorValidationResult(
                parameter="scale_monthly_eveningness_activity_timing",
                is_valid=False,
                issue="Scale mismatch for monthly_eveningness_activity_timing",
            )
        ]

        failed = get_failed_parameters(
            results,
            [parameter["name"] for parameter in model_spec["parameters"]],
            causal_spec=causal_spec,
            model_spec=model_spec,
        )

        assert failed == ["t0_sd_chronotype"]

    def test_scale_mismatch_with_sparse_model_spec_targets_variance_parameter(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "monthly_eveningness_activity_timing",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                }
            ],
            "parameters": [
                {
                    "name": "t0_mean_chronotype",
                    "role": "initial_state_mean",
                },
                {
                    "name": "t0_sd_chronotype",
                    "role": "initial_state_sd",
                },
                {
                    "name": "beta_chronotype_sleep_quality",
                    "role": "fixed_effect",
                },
                {
                    "name": "manifest_mean_monthly_eveningness_activity_timing",
                    "role": "observation_intercept",
                },
            ],
        }
        causal_spec = {
            "measurement": {
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                    }
                ]
            }
        }
        results = [
            PriorValidationResult(
                parameter="scale_monthly_eveningness_activity_timing",
                is_valid=False,
                issue="Scale mismatch for monthly_eveningness_activity_timing",
            )
        ]

        failed = get_failed_parameters(
            results,
            [parameter["name"] for parameter in model_spec["parameters"]],
            causal_spec=causal_spec,
            model_spec=model_spec,
        )

        assert failed == ["t0_sd_chronotype"]


class TestResolveScaleTargetParameters:
    def test_sparse_model_spec_infers_construct_from_parameter_names(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "monthly_eveningness_activity_timing",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                }
            ],
            "parameters": [
                {
                    "name": "t0_mean_chronotype",
                    "role": "initial_state_mean",
                },
                {
                    "name": "t0_sd_chronotype",
                    "role": "initial_state_sd",
                },
                {
                    "name": "manifest_mean_monthly_eveningness_activity_timing",
                    "role": "observation_intercept",
                },
            ],
        }

        resolved = resolve_scale_target_parameters(
            "monthly_eveningness_activity_timing",
            model_spec,
            indicator_to_construct={"monthly_eveningness_activity_timing": "chronotype"},
        )

        assert resolved == ["t0_sd_chronotype"]


# =============================================================================
# _check_lagged_response_plausibility
# =============================================================================


class TestCheckLaggedResponsePlausibility:
    def test_infer_dynamics_repair_scope_localizes_unstable_scc(self):
        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.array([True, True]),
                edge_support=np.array([[False, False], [True, False]]),
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            latent_names=["activity", "sleep"],
        )
        compiled_ssm = {
            "spec": serialize_ssm_spec(spec),
            "edge_lag_days": serialize_edge_lag_days({(1, 0): 1.0}),
        }
        causal_spec = {
            "latent": {"constructs": [], "edges": []},
            "measurement": {"model_clock": "1d"},
            "estimation": {
                "state_order": ["activity", "sleep"],
                "edges": [{"cause": "activity", "effect": "sleep", "lagged": True}],
            },
        }
        dynamics_samples = np.asarray(
            [
                [[-0.5, 0.0], [0.05, 0.1]],
                [[-0.4, 0.0], [0.03, 0.2]],
            ],
            dtype=np.float32,
        )

        scope = _infer_dynamics_repair_scope(
            dynamics_samples,
            [0, 1],
            compiled_ssm=compiled_ssm,
            causal_spec=causal_spec,
        )

        assert scope is not None
        assert scope.kind == "dynamics_scc"
        assert scope.construct_names == ["sleep"]

    def test_near_zero_one_lag_response_yields_warning(self):
        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.array([True, True]),
                edge_support=np.array([[False, False], [True, False]]),
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            latent_names=["stress", "sleep"],
        )
        samples = {
            "drift": jnp.asarray(
                [
                    [[-0.5, 0.0], [0.001, -0.5]],
                    [[-0.6, 0.0], [0.002, -0.4]],
                    [[-0.4, 0.0], [0.0015, -0.6]],
                ],
                dtype=jnp.float32,
            )
        }
        compiled_ssm = {
            "spec": serialize_ssm_spec(spec),
            "edge_lag_days": serialize_edge_lag_days({(1, 0): 1.0}),
        }
        causal_spec = {
            "latent": {"constructs": [], "edges": []},
            "measurement": {"model_clock": "1d"},
            "estimation": {
                "state_order": ["stress", "sleep"],
                "edges": [{"cause": "stress", "effect": "sleep", "lagged": True}],
            },
        }

        results = _check_lagged_response_plausibility(samples, compiled_ssm, causal_spec)

        assert len(results) == 1
        assert results[0].parameter == "beta_stress_sleep"
        assert results[0].severity == "warning"
        assert results[0].is_valid is True
        assert "one-lag response" in (results[0].issue or "")


class TestScalePlausibilityDiagnostics:
    def test_observation_samples_drive_scale_check_when_available(self):
        samples = {
            "drift": jnp.asarray([[[-1.0]]], dtype=jnp.float32),
            "diffusion": jnp.asarray([[[0.1]]], dtype=jnp.float32),
            "observations": jnp.asarray(
                [[[0.0], [300.0], [600.0], [900.0]]],
                dtype=jnp.float32,
            ),
            "observations_mask": jnp.asarray(
                [[[True], [True], [True], [True]]],
                dtype=bool,
            ),
        }

        results = _check_scale_plausibility(
            samples,
            data_stats={"monthly_eveningness_activity_timing": {"std": 1.0}},
            manifest_names=["monthly_eveningness_activity_timing"],
            n_subsample=1,
        )

        assert len(results) == 1
        assert results[0].code == "scale_mismatch"
        assert results[0].parameter == "scale_monthly_eveningness_activity_timing"

    def test_missing_observations_emits_harness_error(self):
        samples = {
            "drift": jnp.asarray([[[-1.0]]], dtype=jnp.float32),
            "diffusion": jnp.asarray([[[0.1]]], dtype=jnp.float32),
        }

        results = _check_scale_plausibility(
            samples,
            data_stats={"monthly_eveningness_activity_timing": {"std": 1.0}},
            manifest_names=["monthly_eveningness_activity_timing"],
            n_subsample=1,
        )

        assert len(results) == 1
        assert results[0].code == "prior_predictive_missing_observations"
        assert results[0].failure_stage == "observation_sample"

    def test_unstable_dynamics_emits_stage_and_certificate(self):
        samples = {
            "drift": jnp.asarray([[[0.1]], [[0.2]], [[-1.0]]], dtype=jnp.float32),
            "diffusion": jnp.asarray([[[0.1]], [[0.1]], [[0.1]]], dtype=jnp.float32),
            "observations": jnp.asarray(
                [
                    [[0.0]],
                    [[0.0]],
                    [[0.0]],
                ],
                dtype=jnp.float32,
            ),
            "observations_mask": jnp.asarray(
                [
                    [[True]],
                    [[True]],
                    [[True]],
                ],
                dtype=bool,
            ),
        }

        results = _check_scale_plausibility(
            samples,
            data_stats={},
            manifest_names=["dummy_manifest"],
            n_subsample=3,
        )

        assert len(results) == 1
        result = results[0]
        assert result.code == "dynamics_stability"
        assert result.failure_stage == "latent_dynamics"
        assert result.failing_draw_indices == [0, 1]
        assert result.pathology_certificate is not None
        assert result.pathology_certificate.kind == "dynamics_stability"
        assert result.pathology_certificate.primary_score == pytest.approx(2 / 3)

    def test_overwhelmingly_unstable_dynamics_raise_runtime_error(self, monkeypatch):
        samples = {
            "drift": jnp.asarray(
                [[[-1.0]], [[-1.0]], [[-1.0]], [[-1.0]], [[-1.0]]], dtype=jnp.float32
            ),
            "diffusion": jnp.asarray(
                [[[0.1]], [[0.1]], [[0.1]], [[0.1]], [[0.1]]], dtype=jnp.float32
            ),
            "observations": jnp.asarray(
                [
                    [[0.0]],
                    [[0.0]],
                    [[0.0]],
                    [[0.0]],
                    [[0.0]],
                ],
                dtype=jnp.float32,
            ),
            "observations_mask": jnp.asarray(
                [
                    [[True]],
                    [[True]],
                    [[True]],
                    [[True]],
                    [[True]],
                ],
                dtype=bool,
            ),
        }

        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.discretization.solve_lyapunov",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("solver failed")),
        )

        results = _check_scale_plausibility(
            samples,
            data_stats={},
            manifest_names=["dummy_manifest"],
            n_subsample=5,
        )

        assert len(results) == 1
        result = results[0]
        assert result.code == "dynamics_stability"
        assert result.failure_stage == "latent_dynamics"
        assert result.failing_draw_indices == [0, 1, 2, 3, 4]

    def test_nuisance_site_skipped(self):
        results = [PriorValidationResult(parameter="vf_1_cint", is_valid=False, issue="something")]
        failed = get_failed_parameters(results, ["alpha", "beta"])
        # vf_1_cint is nuisance → skipped, no other matches → falls back to all params
        assert set(failed) == {"alpha", "beta"}

    def test_keyword_matching(self):
        results = [
            PriorValidationResult(
                parameter="vf_0_decay",
                is_valid=False,
                issue="constraint",
            )
        ]
        params = ["rho_X", "beta_X_Y", "sigma_X"]
        failed = get_failed_parameters(results, params)
        # dynamics decay sites → keywords ["rho", "ar"]
        assert "rho_X" in failed
        assert "beta_X_Y" not in failed

    def test_scale_mismatch_without_causal_spec_returns_all(self):
        results = [
            PriorValidationResult(parameter="scale_mood", is_valid=False, issue="Scale mismatch")
        ]
        params = ["alpha", "beta"]
        failed = get_failed_parameters(results, params)
        assert set(failed) == {"alpha", "beta"}


# =============================================================================
# compute_data_stats
# =============================================================================


class TestComputeDataStats:
    def test_basic_stats(self):
        df = pl.DataFrame({"indicator": ["mood", "mood", "mood"], "value": [1.0, 2.0, 3.0]})
        stats = compute_data_stats(df)
        assert "mood" in stats
        assert abs(stats["mood"]["mean"] - 2.0) < 1e-6
        assert stats["mood"]["min"] == 1.0
        assert stats["mood"]["max"] == 3.0

    def test_multiple_indicators(self):
        df = pl.DataFrame(
            {
                "indicator": ["mood", "mood", "sleep", "sleep"],
                "value": [1.0, 3.0, 10.0, 20.0],
            }
        )
        stats = compute_data_stats(df)
        assert len(stats) == 2
        assert abs(stats["mood"]["mean"] - 2.0) < 1e-6
        assert abs(stats["sleep"]["mean"] - 15.0) < 1e-6

    def test_std_computed(self):
        df = pl.DataFrame({"indicator": ["x", "x", "x", "x"], "value": [0.0, 0.0, 10.0, 10.0]})
        stats = compute_data_stats(df)
        assert stats["x"]["std"] is not None
        assert stats["x"]["std"] > 0

    def test_single_value(self):
        df = pl.DataFrame({"indicator": ["a"], "value": [5.0]})
        stats = compute_data_stats(df)
        assert stats["a"]["mean"] == 5.0
        assert stats["a"]["min"] == 5.0
        assert stats["a"]["max"] == 5.0

    def test_string_values_cast(self):
        """String-typed values should be cast to float."""
        df = pl.DataFrame({"indicator": ["x", "x"], "value": ["1.5", "2.5"]})
        stats = compute_data_stats(df)
        assert abs(stats["x"]["mean"] - 2.0) < 1e-6

    def test_centered_indicators_are_centered_before_summary_stats(self):
        """Centered indicators should have mean zeroed before scale checks."""
        df = pl.DataFrame(
            {
                "indicator": ["x", "x", "x", "y", "y"],
                "value": [10.0, 11.0, 12.0, 5.0, 7.0],
            }
        )

        stats = compute_data_stats(df, centered_indicators={"x"})

        assert abs(stats["x"]["mean"]) < 1e-6
        assert stats["x"]["min"] == -1.0
        assert stats["x"]["max"] == 1.0
        assert stats["y"]["mean"] == 6.0


# =============================================================================
# Compiled prior predictive runtime
# =============================================================================


class TestCompiledPriorPredictiveRuntime:
    def test_known_inputs_are_threaded_into_compiled_prior_predictive(self):
        """Input-driven prior predictive simulation uses prepared transition inputs."""
        spec = block_ssm_spec(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=1,
                decay_support=full_diagonal_support(1),
                edge_support=np.zeros((1, 1), dtype=bool),
                coupling_template=jnp.zeros((1, 1), dtype=jnp.float32),
                intercept_support=np.zeros(1, dtype=bool),
                cint_template=jnp.zeros(1, dtype=jnp.float32),
            ),
            diffusion_block=diagonal_diffusion_block(1),
            input_effect_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=1,
                free_support=np.array([[True]]),
                template=jnp.zeros((1, 1), dtype=jnp.float32),
                free_site_name="input_effect_free",
                det_site_name="input_effect",
                support=SupportClass.REAL,
                site_kind=SiteKind.INPUT_EFFECT,
                assembly_group="input_effect",
                fixed_spec_field="input_effect",
                priors_field="input_effect",
            ),
            input_names=["dose"],
            input_source_indicators=["dose_mg"],
            input_scales=[10.0],
            input_missing_policies=["forward_fill"],
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
        )
        semantics = compile_prior_semantics(spec)

        samples = sample_prior_predictive_from_compiled_semantics(
            spec,
            semantics,
            jnp.arange(3, dtype=jnp.float32),
            transition_inputs=jnp.array([[0.0], [0.0], [1.0]], dtype=jnp.float32),
            num_samples=3,
            seed=0,
        )

        assert samples["input_effect"].shape == (3, 1, 1)
        assert samples["observations"].shape == (3, 3, 1)
        assert bool(jnp.isfinite(samples["observations"]).all())

    def test_nonlinear_dynamics_uses_compiled_prior_predictive_runtime(self):
        """Component dynamics specs should sample prior predictive without an affine view."""
        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=DynamicsSpec(
                n_latent=2,
                components=(
                    DiagonalDecaySpec(),
                    HillEdgeSpec(source=0, target=1),
                ),
            ),
            diffusion_block=diagonal_diffusion_block(2),
            manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        )
        semantics = compile_prior_semantics(spec)

        samples = sample_prior_predictive_from_compiled_semantics(
            spec,
            semantics,
            jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32),
            num_samples=3,
            seed=0,
        )

        assert samples["vf_0_decay"].shape == (3, 2)
        assert samples["vf_1_Emax"].shape == (3,)
        assert samples["latents"].shape == (3, 4, 2)
        assert samples["linear_predictors"].shape == (3, 4, 2)
        assert samples["observations"].shape == (3, 4, 2)
        assert bool(jnp.isfinite(samples["observations"]).all())

    def test_ordered_likelihood_requires_hydrated_level_counts(self):
        """Discrete emissions fail clearly until hydration provides level counts."""
        spec = block_ssm_spec(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=1,
                decay_support=full_diagonal_support(1),
                edge_support=np.zeros((1, 1), dtype=bool),
                coupling_template=jnp.zeros((1, 1), dtype=jnp.float32),
                intercept_support=np.zeros(1, dtype=bool),
                cint_template=jnp.zeros(1, dtype=jnp.float32),
            ),
            diffusion_block=diagonal_diffusion_block(1),
            manifest_dists=[DistributionFamily.ORDERED_LOGISTIC],
        )
        semantics = compile_prior_semantics(spec)

        with pytest.raises(ValueError, match="manifest_level_counts"):
            sample_prior_predictive_from_compiled_semantics(
                spec,
                semantics,
                jnp.arange(4, dtype=jnp.float32),
                num_samples=3,
                seed=0,
            )

    def test_initial_state_correlation_runtime_repairs_invalid_draws(self):
        """Compiled runtime should stabilize invalid multi-pair initial correlations."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        mask[2, 0] = True
        mask[2, 1] = True
        spec = block_ssm_spec(
            n_latent=3,
            n_manifest=3,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=3,
                decay_support=full_diagonal_support(3),
                edge_support=np.zeros((3, 3), dtype=bool),
                coupling_template=jnp.zeros((3, 3), dtype=jnp.float32),
                intercept_support=np.zeros(3, dtype=bool),
                cint_template=jnp.zeros(3, dtype=jnp.float32),
            ),
            diffusion_block=diagonal_diffusion_block(3),
            t0_chol_block=T0CholBlockSpec(
                n_latent=3,
                diag_support=full_diagonal_support(3),
                correlation_support=mask,
                template=jnp.eye(3, dtype=jnp.float32),
            ),
            manifest_dists=[DistributionFamily.GAUSSIAN] * 3,
            manifest_links=[LinkFunction.IDENTITY] * 3,
        )
        priors = prior_registry(
            t0_var_lower_free=PriorSpec(
                PriorDistributionFamily.TRUNCATED_NORMAL,
                {"mu": 0.8, "sigma": 0.1, "lower": -1.0, "upper": 1.0},
            )
        )
        semantics = compile_prior_semantics(spec, priors)

        samples = sample_prior_predictive_from_compiled_semantics(
            spec,
            semantics,
            jnp.arange(5, dtype=jnp.float32),
            num_samples=20,
            seed=0,
        )

        min_eigs = jnp.linalg.eigvalsh(samples["t0_cov"])[..., 0]
        assert bool(jnp.isfinite(samples["observations"]).all())
        assert bool(samples["observations_mask"].all())
        assert bool(jnp.isfinite(samples["t0_cov"]).all())
        assert bool((min_eigs > -1e-6).all())
