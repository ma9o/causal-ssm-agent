"""Tests for prior predictive validation checks.

Covers: _check_nan_inf, _check_constraint_violations, _check_extreme_values,
compute_data_stats, format_validation_report, format_parameter_feedback,
get_failed_parameters.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.models.prior_predictive import (
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
)
from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm.parameterization import compile_prior_semantics
from causal_ssm_agent.models.ssm.prior_predictive_runtime import (
    sample_prior_predictive_from_compiled_semantics,
)
from causal_ssm_agent.models.ssm_compiler import serialize_ssm_spec
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction
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
        assert not result.is_valid
        assert "x" in result.issue

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

    def test_likelihood_diagnostics_ignored(self):
        samples = {
            "ll_per_timestep": jnp.array([float("-inf")]),
            "log_likelihood": jnp.array([float("nan")]),
            "drift_diag_pop": jnp.array([0.1, 0.2]),
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
    def test_passed_report_contains_no_failures(self):
        report = format_validation_report(True, [])
        assert "PASSED" in report
        assert "FAILED" not in report

    def test_failed_report_includes_each_issue_and_parameter(self):
        results = [
            PriorValidationResult(
                parameter="drift_diag_pop",
                is_valid=False,
                issue="Too extreme",
                suggested_adjustment="Fix it",
            ),
            PriorValidationResult(
                parameter="diffusion_diag_pop",
                is_valid=False,
                issue="Negative values",
                suggested_adjustment="Use positive prior",
            ),
        ]
        report = format_validation_report(False, results)
        assert "FAILED" in report
        assert "drift_diag_pop" in report
        assert "Too extreme" in report
        assert "diffusion_diag_pop" in report
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


# =============================================================================
# _check_lagged_response_plausibility
# =============================================================================


class TestCheckLaggedResponsePlausibility:
    def test_infer_dynamics_repair_scope_localizes_unstable_scc(self):
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["activity", "sleep"],
            drift_mask=np.array([[True, False], [True, True]]),
        )
        compiled_ssm = {"spec": serialize_ssm_spec(spec)}
        causal_spec = {
            "latent": {"constructs": [], "edges": []},
            "measurement": {"model_clock": "1d"},
            "estimation": {
                "state_order": ["activity", "sleep"],
                "edges": [{"cause": "activity", "effect": "sleep", "lagged": True}],
            },
        }
        drift_samples = np.asarray(
            [
                [[-0.5, 0.0], [0.05, 0.1]],
                [[-0.4, 0.0], [0.03, 0.2]],
            ],
            dtype=np.float32,
        )

        scope = _infer_dynamics_repair_scope(
            drift_samples,
            [0, 1],
            compiled_ssm=compiled_ssm,
            causal_spec=causal_spec,
        )

        assert scope is not None
        assert scope.kind == "dynamics_scc"
        assert scope.construct_names == ["sleep"]

    def test_near_zero_one_lag_response_yields_warning(self):
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            drift_mask=np.array([[True, False], [True, True]]),
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
        compiled_ssm = {"spec": serialize_ssm_spec(spec)}
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
            "causal_ssm_agent.models.ssm.discretization.solve_lyapunov",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("solver failed")),
        )

        with pytest.raises(RuntimeError, match="draws unstable"):
            _check_scale_plausibility(
                samples,
                data_stats={},
                manifest_names=["dummy_manifest"],
                n_subsample=5,
            )

    def test_nuisance_site_skipped(self):
        results = [PriorValidationResult(parameter="cint_pop", is_valid=False, issue="something")]
        failed = get_failed_parameters(results, ["alpha", "beta"])
        # cint_pop is nuisance → skipped, no other matches → falls back to all params
        assert set(failed) == {"alpha", "beta"}

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


# =============================================================================
# Compiled prior predictive runtime
# =============================================================================


def _complex_mixed_runtime_spec() -> SSMSpec:
    return SSMSpec(
        n_latent=4,
        n_manifest=10,
        drift=jnp.array(
            [
                [-0.45, 0.0, 0.0, 0.0],
                [0.08, -0.35, 0.0, 0.0],
                [0.02, 0.06, -0.4, 0.0],
                [0.0, 0.03, 0.05, -0.3],
            ],
            dtype=jnp.float32,
        ),
        diffusion=jnp.diag(jnp.array([0.2, 0.18, 0.16, 0.14], dtype=jnp.float32)),
        cint=jnp.zeros(4, dtype=jnp.float32),
        lambda_mat=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.3, 0.4, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.2, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.5, 0.3],
                [0.0, 0.0, 0.7, 0.0],
                [0.0, 0.2, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.9],
                [0.1, 0.0, 0.2, 0.8],
            ],
            dtype=jnp.float32,
        ),
        manifest_means=jnp.array(
            [0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1],
            dtype=jnp.float32,
        ),
        manifest_var=jnp.diag(
            jnp.array([0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12], dtype=jnp.float32)
            ** 2
        ),
        t0_means=jnp.zeros(4, dtype=jnp.float32),
        t0_var=jnp.eye(4, dtype=jnp.float32) * 0.25,
        manifest_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.STUDENT_T,
            DistributionFamily.GAMMA,
            DistributionFamily.BETA,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[
            LinkFunction.IDENTITY,
            LinkFunction.LOGIT,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
            LinkFunction.LOG,
            LinkFunction.LOGIT,
            LinkFunction.CUMULATIVE_LOGIT,
            LinkFunction.SOFTMAX,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
        ],
        manifest_level_counts=[0, 0, 0, 0, 0, 0, 4, 4, 0, 0],
        latent_names=["stress", "adherence", "sleep", "focus"],
        manifest_names=[
            "stress_cont",
            "adherence_flag",
            "steps_count",
            "fatigue_t",
            "screen_gap",
            "sleep_efficiency",
            "symptom_severity",
            "coping_style",
            "rumination_count",
            "focus_cont",
        ],
    )


class TestCompiledPriorPredictiveRuntime:
    def test_mixed_likelihood_samples_are_finite(self):
        """Compiled runtime handles a larger mixed-family model without tracing."""
        spec = _complex_mixed_runtime_spec()
        semantics = compile_prior_semantics(spec, SSMPriors())
        samples = sample_prior_predictive_from_compiled_semantics(
            spec,
            semantics,
            jnp.linspace(0.0, 5.5, 12, dtype=jnp.float32),
            num_samples=10,
            seed=7,
        )

        assert samples["observations"].shape == (10, 12, 10)
        assert samples["observations_mask"].shape == (10, 12, 10)
        assert bool(jnp.isfinite(samples["drift"]).all())
        assert bool(jnp.isfinite(samples["diffusion"]).all())
        assert bool(jnp.isfinite(samples["observations"]).all())
        assert bool(samples["observations_mask"].all())
        assert bool(
            (
                (samples["observations"][:, :, 1] == 0) | (samples["observations"][:, :, 1] == 1)
            ).all()
        )
        assert bool((samples["observations"][:, :, 2] >= 0).all())
        assert bool((samples["observations"][:, :, 4] > 0).all())
        assert bool(
            (
                (samples["observations"][:, :, 5] >= 0) & (samples["observations"][:, :, 5] <= 1)
            ).all()
        )
        assert bool(
            (
                (samples["observations"][:, :, 6] >= 0) & (samples["observations"][:, :, 6] <= 3)
            ).all()
        )
        assert bool(
            (
                (samples["observations"][:, :, 7] >= 0) & (samples["observations"][:, :, 7] <= 3)
            ).all()
        )
        assert bool((samples["observations"][:, :, 8] >= 0).all())

    def test_ordered_likelihood_requires_hydrated_level_counts(self):
        """Discrete emissions fail clearly until hydration provides level counts."""
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1, dtype=jnp.float32),
            diffusion="diag",
            manifest_dist=DistributionFamily.ORDERED_LOGISTIC,
        )
        semantics = compile_prior_semantics(spec, SSMPriors())

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
        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3, dtype=jnp.float32),
            diffusion="diag",
            t0_var="free",
            t0_correlation_mask=mask,
            manifest_dists=[DistributionFamily.GAUSSIAN] * 3,
            manifest_links=[LinkFunction.IDENTITY] * 3,
        )
        priors = SSMPriors(
            t0_var_offdiag={"mu": 0.8, "sigma": 0.1, "lower": -1.0, "upper": 1.0, "family": 1}
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
