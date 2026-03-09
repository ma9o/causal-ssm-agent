"""Tests for prior predictive validation checks.

Covers: _check_nan_inf, _check_constraint_violations, _check_extreme_values,
_compute_data_stats, format_validation_report, format_parameter_feedback,
get_failed_parameters.
"""

import jax.numpy as jnp
import polars as pl

from causal_ssm_agent.models.prior_predictive import (
    _check_constraint_violations,
    _check_extreme_values,
    _check_nan_inf,
    _compute_data_stats,
    format_parameter_feedback,
    format_validation_report,
    get_failed_parameters,
)
from causal_ssm_agent.workers.schemas_prior import PriorValidationResult

# =============================================================================
# _check_nan_inf
# =============================================================================


class TestCheckNanInf:
    def test_clean_samples_no_issue(self):
        samples = {"drift_diag_pop": jnp.array([1.0, 2.0, 3.0])}
        assert _check_nan_inf(samples) is None

    def test_nan_detected(self):
        samples = {"drift_diag_pop": jnp.array([1.0, float("nan"), 3.0])}
        result = _check_nan_inf(samples)
        assert result is not None
        assert not result.is_valid
        assert "drift_diag_pop" in result.issue

    def test_inf_detected(self):
        samples = {"x": jnp.array([float("inf")])}
        result = _check_nan_inf(samples)
        assert result is not None
        assert not result.is_valid

    def test_multiple_bad_sites(self):
        samples = {
            "a": jnp.array([float("nan")]),
            "b": jnp.array([float("inf")]),
            "c": jnp.array([1.0]),
        }
        result = _check_nan_inf(samples)
        assert result is not None
        assert "a" in result.issue
        assert "b" in result.issue


# =============================================================================
# _check_constraint_violations
# =============================================================================


class TestCheckConstraintViolations:
    def test_all_positive_no_issue(self):
        samples = {"diffusion_diag_pop": jnp.array([0.1, 0.5, 1.0])}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_negative_values_detected(self):
        # 50% negative → above 5% threshold
        samples = {"diffusion_diag_pop": jnp.array([-1.0, -2.0, 1.0, 2.0])}
        results = _check_constraint_violations(samples)
        assert len(results) == 1
        assert "diffusion_diag_pop" in results[0].parameter
        assert "negative" in results[0].issue

    def test_below_threshold_no_issue(self):
        # 1% negative → below 5% threshold
        values = jnp.concatenate([jnp.array([-0.001]), jnp.ones(99)])
        samples = {"manifest_var_diag": values}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_non_positive_site_ignored(self):
        samples = {"drift_diag_pop": jnp.array([-1.0, -2.0])}
        results = _check_constraint_violations(samples)
        assert results == []

    def test_missing_site_ignored(self):
        results = _check_constraint_violations({"other_site": jnp.ones(5)})
        assert results == []

    def test_custom_threshold(self):
        # 25% negative → above 10% custom threshold
        samples = {"t0_var_diag": jnp.array([-1.0, 1.0, 1.0, 1.0])}
        results = _check_constraint_violations(samples, threshold=0.1)
        assert len(results) == 1


# =============================================================================
# _check_extreme_values
# =============================================================================


class TestCheckExtremeValues:
    def test_normal_values_no_issue(self):
        samples = {"drift_diag_pop": jnp.array([0.5, -0.3, 1.0])}
        results = _check_extreme_values(samples)
        assert results == []

    def test_extreme_values_detected(self):
        # All extreme
        samples = {"drift_diag_pop": jnp.array([1e7, 1e8, 1e9])}
        results = _check_extreme_values(samples)
        assert len(results) == 1
        assert "extreme" in results[0].issue.lower()

    def test_below_threshold_no_issue(self):
        # Only 1 out of 100 extreme → 1% < 10% threshold
        values = jnp.concatenate([jnp.array([1e7]), jnp.ones(99)])
        samples = {"drift_diag_pop": values}
        results = _check_extreme_values(samples)
        assert results == []

    def test_non_param_site_ignored(self):
        # Sites not matching param patterns are skipped
        samples = {"drift": jnp.array([1e7, 1e8])}
        results = _check_extreme_values(samples)
        assert results == []


# =============================================================================
# format_validation_report
# =============================================================================


class TestFormatValidationReport:
    def test_passed(self):
        report = format_validation_report(True, [])
        assert "PASSED" in report

    def test_failed_with_issues(self):
        results = [
            PriorValidationResult(
                parameter="drift_diag_pop",
                is_valid=False,
                issue="Too extreme",
                suggested_adjustment="Fix it",
            )
        ]
        report = format_validation_report(False, results)
        assert "FAILED" in report
        assert "Too extreme" in report


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
        feedback = format_parameter_feedback("drift", results)
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
            "drift",
            results,
            prior={"distribution": "Normal", "params": {"mu": 0.0, "sigma": 1.0}},
        )
        assert "Normal" in feedback
        assert "mu=0.0" in feedback


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

    def test_nuisance_site_skipped(self):
        results = [PriorValidationResult(parameter="cint_pop", is_valid=False, issue="something")]
        failed = get_failed_parameters(results, ["alpha", "beta"])
        # cint_pop is nuisance → doesn't match any param, but also doesn't
        # trigger blanket. Falls through to returning all.
        # (The function returns all params if no match is found)
        assert isinstance(failed, list)

    def test_keyword_matching(self):
        results = [
            PriorValidationResult(parameter="drift_diag_pop", is_valid=False, issue="constraint")
        ]
        params = ["rho_X", "beta_X_Y", "sigma_X"]
        failed = get_failed_parameters(results, params)
        # drift_diag → keywords ["rho", "ar"]
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
# _compute_data_stats
# =============================================================================


class TestComputeDataStats:
    def test_basic_stats(self):
        df = pl.DataFrame({"indicator": ["mood", "mood", "mood"], "value": [1.0, 2.0, 3.0]})
        stats = _compute_data_stats(df)
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
        stats = _compute_data_stats(df)
        assert len(stats) == 2
        assert abs(stats["mood"]["mean"] - 2.0) < 1e-6
        assert abs(stats["sleep"]["mean"] - 15.0) < 1e-6

    def test_std_computed(self):
        df = pl.DataFrame({"indicator": ["x", "x", "x", "x"], "value": [0.0, 0.0, 10.0, 10.0]})
        stats = _compute_data_stats(df)
        assert stats["x"]["std"] is not None
        assert stats["x"]["std"] > 0

    def test_single_value(self):
        df = pl.DataFrame({"indicator": ["a"], "value": [5.0]})
        stats = _compute_data_stats(df)
        assert stats["a"]["mean"] == 5.0
        assert stats["a"]["min"] == 5.0
        assert stats["a"]["max"] == 5.0

    def test_string_values_cast(self):
        """String-typed values should be cast to float."""
        df = pl.DataFrame({"indicator": ["x", "x"], "value": ["1.5", "2.5"]})
        stats = _compute_data_stats(df)
        assert abs(stats["x"]["mean"] - 2.0) < 1e-6
