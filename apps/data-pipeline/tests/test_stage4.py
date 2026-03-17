"""Tests for Stage 4: Model Specification & Prior Elicitation.

Unit tests for prior validation helpers, default priors, aggregation, and
prompt generation live in their dedicated files:
- test_prior_predictive.py (NaN/constraint/extreme checks, format functions)
- test_prior_aggregation.py (simple/GMM aggregation)
- test_prior_research_prompts.py (paraphrase generation)
- test_get_default_prior.py (constraint→distribution mapping)
- test_model_spec_validation.py (validate_model_spec domain rules)

This file tests stage4-specific orchestration: SSMModelBuilder wiring,
prior predictive end-to-end, failed parameter identification with
causal_spec context, SSM prior conversion, sparsity, trial compile,
and compile ownership.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from causal_ssm_agent.models.prior_predictive import (
    get_failed_parameters,
    validate_prior_predictive,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_priors as compile_ssm_priors,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_ssm_inputs,
)
from causal_ssm_agent.workers.schemas_prior import (
    PriorValidationResult,
)

# --- Fixtures ---


@pytest.fixture
def simple_model_spec() -> dict:
    """A minimal model spec for testing."""
    return {
        "likelihoods": [
            {
                "variable": "mood_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            }
        ],
        "parameters": [
            {
                "name": "intercept_mood_score",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Intercept for mood",
                "search_context": "mood baseline population mean",
            },
            {
                "name": "rho_mood",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) coefficient for mood",
                "search_context": "mood autocorrelation daily",
            },
            {
                "name": "sigma_mood_score",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for mood",
                "search_context": "mood variability within-person",
            },
        ],
    }


@pytest.fixture
def simple_priors() -> dict:
    """Simple priors matching the model spec."""
    return {
        "intercept_mood_score": {
            "parameter": "intercept_mood_score",
            "distribution": "Normal",
            "params": {"mu": 5.0, "sigma": 1.0},
            "sources": [],
            "reasoning": "Centered on scale midpoint",
        },
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "reasoning": "Weakly informative for AR coefficient",
        },
        "sigma_mood_score": {
            "parameter": "sigma_mood_score",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative for residual SD",
        },
    }


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Simple test data with lagged columns."""
    n = 50
    return pd.DataFrame(
        {
            "mood_score": np.random.randn(n) * 1.5 + 5,
            "mood_score_lag1": np.random.randn(n) * 1.5 + 5,
            "subject_id": np.repeat(np.arange(5), 10),
        }
    )


# --- SSMModelBuilder Tests ---


class TestSSMModelBuilder:
    """Test SSM model building."""

    def test_builder_init(self, simple_model_spec, simple_priors):
        """Builder initializes with spec and priors."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=simple_model_spec,
            priors=simple_priors,
        )
        assert builder._model_type == "SSM"
        assert builder.version == "0.1.0"

    def test_builder_builds_model(self, simple_model_spec, simple_priors, simple_data):
        """Builder creates an SSMModel."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=simple_model_spec,
            priors=simple_priors,
        )
        model = builder.build_model(simple_data)
        assert model is not None
        assert model.spec.n_manifest == 1  # mood_score only


# --- Prior Predictive Validation Tests ---


def _make_polars_data() -> pl.DataFrame:
    """Create polars long-format data for validation tests."""
    rng = np.random.default_rng(42)
    n = 30
    times = list(range(n))
    return pl.DataFrame(
        {
            "indicator": ["mood_score"] * n,
            "value": (rng.standard_normal(n) * 1.5 + 5).tolist(),
            "timestamp": times,
        }
    )


class TestPriorPredictiveValidation:
    """Test prior predictive validation end-to-end."""

    def test_valid_priors_pass(self, simple_model_spec, simple_priors):
        """Simple spec + priors + polars data -> is_valid=True."""
        raw_data = _make_polars_data()
        is_valid, results, _samples = validate_prior_predictive(
            simple_model_spec, simple_priors, raw_data, n_samples=10
        )
        assert is_valid is True
        assert len(results) > 0

    def test_model_build_failure(self, simple_priors):
        """Broken spec -> is_valid=False, error in results."""
        broken_spec = {
            "likelihoods": [
                {
                    "variable": "nonexistent_col",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coeff",
                    "search_context": "",
                }
            ],
        }
        # This should still build (builder is tolerant), but let's test
        # with a truly broken spec by patching build_model to raise
        with patch(
            "causal_ssm_agent.models.ssm_builder.SSMModelBuilder.build_model",
            side_effect=ValueError("deliberate test failure"),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                broken_spec, simple_priors, None, n_samples=10
            )
            assert is_valid is False
            assert any("model_build" in r.parameter for r in results)
            assert any("deliberate test failure" in (r.issue or "") for r in results)

    def test_no_data_still_validates(self, simple_model_spec, simple_priors):
        """raw_data=None -> NaN/constraint/extreme checks run, scale skipped."""
        is_valid, results, _samples = validate_prior_predictive(
            simple_model_spec, simple_priors, None, n_samples=10
        )
        # Should still produce results (pass or fail) without crashing
        assert isinstance(is_valid, bool)
        assert isinstance(results, list)

    def test_no_data_uses_support_compatible_dummy_build_data(self):
        """Support-restricted likelihoods should still validate without raw data."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "screen_gap",
                    "distribution": "gamma",
                    "link": "log",
                    "reasoning": "Positive continuous gap",
                }
            ],
            "parameters": [
                {
                    "name": "rho_screen_gap",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                    "search_context": "",
                }
            ],
        }
        priors = {
            "rho_screen_gap": {
                "parameter": "rho_screen_gap",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "Weakly informative",
            }
        }

        with patch(
            "causal_ssm_agent.models.ssm_builder.SSMModelBuilder.sample_prior_predictive",
            return_value={"drift_diag_pop": np.ones((2, 1))},
        ):
            is_valid, results, _samples = validate_prior_predictive(
                model_spec, priors, None, n_samples=2
            )

        assert is_valid is True
        assert not any(r.parameter == "model_build" for r in results)

    def test_build_validation_payload_from_assembly(self, simple_model_spec, simple_priors):
        """Shared Stage 4 assembly helpers return the expected payload shape."""
        from causal_ssm_agent.flows.stages.stage4_assembly import (
            build_validation_payload,
            validate_assembly,
        )

        raw_data = _make_polars_data()
        validation = validate_assembly(simple_model_spec, simple_priors, raw_data, None)
        result = build_validation_payload(validation, simple_model_spec)
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "results" in result
        assert "issues" in result

    def test_validate_assembly_reuses_compiled_artifact_for_prior_checks(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Stage 4 should compile once per validation attempt and pass that artifact through."""
        from causal_ssm_agent.flows.stages.stage4_assembly import validate_assembly

        compiled_artifact = {"schema_version": 1}
        seen_compiled: list[dict] = []

        def stub_validate_prior_predictive(*args, compiled_ssm=None, **kwargs):
            seen_compiled.append(compiled_ssm)
            return True, [], {}

        with (
            patch(
                "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
                return_value=compiled_artifact,
            ) as compile_mock,
            patch(
                "causal_ssm_agent.models.prior_predictive.validate_prior_predictive",
                side_effect=stub_validate_prior_predictive,
            ),
        ):
            validation = validate_assembly(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                None,
            )

        assert compile_mock.call_count == 1
        assert seen_compiled == [compiled_artifact]
        assert validation.compiled_ssm == compiled_artifact

    def test_validate_prior_predictive_skips_recompile_when_artifact_provided(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Explicit compiled_ssm should bypass compile_ssm_artifact entirely."""

        class _DummyBuilder:
            def sample_prior_predictive(self, samples: int = 500):
                return {"drift_diag_pop": np.ones((samples, 1))}

        runtime = SimpleNamespace(builder=_DummyBuilder())

        with (
            patch(
                "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
                side_effect=AssertionError("compile should not be called"),
            ),
            patch(
                "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
                return_value=runtime,
            ),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                n_samples=3,
                compiled_ssm={"schema_version": 1},
            )

        assert is_valid is True
        assert results


class TestFailedParameters:
    """Test failed parameter identification."""

    def test_scale_mismatch_with_causal_spec_targets_construct(self):
        """Scale mismatch with causal_spec targets only the affected construct."""
        results = [
            PriorValidationResult(
                parameter="scale_mood_score",
                is_valid=False,
                issue="Scale mismatch for mood_score",
                suggested_adjustment=None,
            ),
        ]
        causal_spec = {
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {"name": "mood_score", "construct_name": "mood"},
                    {"name": "stress_score", "construct_name": "stress"},
                ],
            },
        }
        all_params = ["rho_mood", "sigma_mood", "rho_stress", "sigma_stress", "beta_stress_mood"]
        failed = get_failed_parameters(results, all_params, causal_spec=causal_spec)
        # Only mood-related params should be re-elicited
        assert "rho_mood" in failed
        assert "sigma_mood" in failed
        assert "beta_stress_mood" in failed  # contains "mood"
        assert "rho_stress" not in failed
        assert "sigma_stress" not in failed


class TestRetryFeedback:
    """Test Stage 4 retry feedback shaping."""

    def test_build_retry_feedback_global_failure_returns_shared_summary(self):
        from causal_ssm_agent.flows.stages.stage4_assembly import build_retry_feedback

        validation = SimpleNamespace(
            pp_valid=False,
            pp_results=[
                PriorValidationResult(
                    parameter="model_build",
                    is_valid=False,
                    issue=(
                        "Model build failed:\n"
                        "Observation support check failed:\n"
                        "- 'ide_focus_gaps' uses gamma emission but 29/125 "
                        "observations are outside support (gamma requires y > 0; min=0, max=24)"
                    ),
                    suggested_adjustment="Fix model_spec or priors to enable model construction",
                )
            ],
        )
        priors = {
            "rho_focus_time": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
            },
            "sigma_focus_time": {
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
            },
        }

        failed, feedbacks, global_summary = build_retry_feedback(validation, priors)

        assert set(failed) == set(priors)
        assert feedbacks == {}
        assert global_summary is not None
        assert "Validation FAILED (global issue" in global_summary
        assert "'ide_focus_gaps' uses gamma emission" in global_summary
        assert "Model build failed" not in global_summary
        assert "Consider changing the distribution family" in global_summary


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_prior_converts_to_mu_sigma(self, simple_model_spec):
        """Beta(2,2) AR prior converts via AR-to-drift transform."""
        import math

        from causal_ssm_agent.models.ssm import SSMSpec

        priors = {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_spec = SSMSpec(n_latent=1, n_manifest=1, latent_names=["mood"])
        ssm_priors, _idx = compile_ssm_priors(priors, simple_model_spec, ssm_spec=ssm_spec)

        # Beta(2,2): E[X] = 0.5 → drift mu = -ln(0.5)/1.0 ≈ 0.693
        # Per-element with 1 entry: mu is a list [0.693]
        expected_mu = -math.log(0.5) / 1.0
        mu = ssm_priors.drift_diag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.01
        sigma = ssm_priors.drift_diag["sigma"]
        sigma_val = sigma[0] if isinstance(sigma, list) else sigma
        assert sigma_val > 0.4  # delta method sigma

    def test_halfnormal_prior_preserves_sigma(self, simple_model_spec):
        """HalfNormal(0.5) prior preserves sigma."""
        priors = {
            "sigma_mood_score": {
                "parameter": "sigma_mood_score",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_priors, _idx = compile_ssm_priors(priors, simple_model_spec, ssm_spec=None)
        assert ssm_priors.diffusion_diag["sigma"] == 0.5

    def test_uniform_prior_converts(self):
        """Uniform(-1, 1) converts to Normal(0, 0.5)."""
        from causal_ssm_agent.models.ssm_compilation import normalize_prior_params

        result = normalize_prior_params("Uniform", {"lower": -1.0, "upper": 1.0})
        assert result["mu"] == 0.0
        assert result["sigma"] == 0.5

    def test_compile_ssm_inputs_validates_dict_once(self, simple_model_spec, simple_priors):
        """Compilation should validate a dict spec once, then pass the parsed object through."""
        from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

        with patch.object(ModelSpec, "model_validate", wraps=ModelSpec.model_validate) as validate:
            compile_ssm_inputs(simple_model_spec, simple_priors)

        assert validate.call_count == 1

    def test_role_based_mapping_covers_loading(self, simple_model_spec):
        """LOADING role maps to lambda_free SSMPriors field."""
        spec = dict(simple_model_spec)
        spec["parameters"] = [
            {
                "name": "lambda_mood",
                "role": "loading",
                "constraint": "positive",
                "description": "Factor loading",
                "search_context": "test",
            },
        ]
        priors = {
            "lambda_mood": {
                "parameter": "lambda_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.8},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_priors, _idx = compile_ssm_priors(priors, spec, ssm_spec=None)
        assert ssm_priors.lambda_free["sigma"] == 0.8

    def test_keyword_fallback_without_model_spec(self):
        """Without ModelSpec, keywords still map priors (no AR transform)."""
        priors = {
            "rho_x": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.5},
            },
        }
        ssm_priors, _idx = compile_ssm_priors(priors, {}, ssm_spec=None)
        assert ssm_priors.drift_diag["mu"] == -0.3
        assert ssm_priors.drift_diag["sigma"] == 0.5

    def test_multiple_ar_params_produce_per_element_drift_diag(self):
        """Multiple AR params map to separate drift_diag array entries."""
        import math

        from causal_ssm_agent.models.ssm import SSMSpec

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 5.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 5.0}},
        }
        ssm_spec = SSMSpec(n_latent=2, n_manifest=2, latent_names=["mood", "stress"])
        ssm_priors, _idx = compile_ssm_priors(priors, model_spec, ssm_spec=ssm_spec)

        # Both should produce per-element arrays (lists), not scalars
        assert isinstance(ssm_priors.drift_diag["mu"], list)
        assert len(ssm_priors.drift_diag["mu"]) == 2

        # Beta(5,2) → E=5/7≈0.714, Beta(2,5) → E=2/7≈0.286
        mu_ar_mood = 5.0 / 7.0
        mu_ar_stress = 2.0 / 7.0
        expected_mood = -math.log(mu_ar_mood) / 1.0
        expected_stress = -math.log(mu_ar_stress) / 1.0
        assert abs(ssm_priors.drift_diag["mu"][0] - expected_mood) < 0.01
        assert abs(ssm_priors.drift_diag["mu"][1] - expected_stress) < 0.01

    def test_ar_transform_respects_granularity(self):
        """Hourly construct → dt=1/24, producing larger drift magnitude."""
        import math

        from causal_ssm_agent.models.ssm import SSMSpec

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }
        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "heart_rate",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [],
            },
            "measurement": {"model_clock": "1h", "indicators": []},
        }
        ssm_spec = SSMSpec(n_latent=1, n_manifest=1, latent_names=["heart_rate"])
        ssm_priors, _idx = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Beta(2,2) → E=0.5; hourly dt = 1/24
        # drift mu = -ln(0.5) / (1/24) = 0.693 * 24 ≈ 16.64
        dt_hourly = 1.0 / 24.0
        expected_mu = -math.log(0.5) / dt_hourly
        mu = ssm_priors.drift_diag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.1

    def test_beta_prior_dt_to_ct_transform(self):
        """FIXED_EFFECT beta priors are converted via element-wise beta/dt scaling."""
        from causal_ssm_agent.models.ssm import SSMSpec

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                    "search_context": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        # drift_mask enables off-diagonal at [mood, stress] position
        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_mask=drift_mask,
        )
        ssm_priors, _idx = compile_ssm_priors(priors, model_spec, ssm_spec=ssm_spec)

        # Daily default: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu = ssm_priors.drift_offdiag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - 0.3) < 0.01

    def test_beta_prior_dt_to_ct_respects_granularity(self):
        """FIXED_EFFECT beta transform uses effect construct's granularity."""
        from causal_ssm_agent.models.ssm import SSMSpec

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
                {
                    "variable": "act",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                    "search_context": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
        }
        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "heart_rate",
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "activity",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [{"cause": "activity", "effect": "heart_rate"}],
            },
            "measurement": {"model_clock": "1h", "indicators": []},
        }
        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["heart_rate", "activity"],
            drift_mask=drift_mask,
        )
        ssm_priors, _idx = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Hourly dt = 1/24 → beta_CT = 0.3 / (1/24) = 7.2
        dt_hourly = 1.0 / 24.0
        expected_mu = 0.3 / dt_hourly  # 7.2
        mu = ssm_priors.drift_offdiag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - expected_mu) < 0.5


# --- Trial Compile Tests ---


class TestTrialCompile:
    """Test trial_compile_model_spec catches structural errors early."""

    def test_valid_spec_returns_none(self, simple_model_spec):
        """A well-formed spec compiles successfully with default priors."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        result = trial_compile_model_spec(simple_model_spec)
        assert result is None

    def test_compile_failure_returns_error(self):
        """When compilation raises, trial_compile returns the error string."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                    "search_context": "",
                }
            ],
        }
        with patch(
            "causal_ssm_agent.models.ssm_compiler._compile_validated_ssm_artifact",
            side_effect=ValueError("dimension mismatch in drift matrix"),
        ):
            result = trial_compile_model_spec(spec)
        assert result is not None
        assert "dimension mismatch" in result

    def test_role_constraint_mismatch_returns_error(self):
        """Compiler should reject parameter-role constraint mismatches."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                    "search_context": "",
                },
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "none",
                    "description": "test",
                    "search_context": "",
                },
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "constraint 'none' unexpected for role 'residual_sd'" in result

    def test_missing_ar_parameters_returns_error(self):
        """Compiler should reject ModelSpecs with no latent dimensionality signal."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "test",
                    "search_context": "",
                }
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "No AR_COEFFICIENT parameters found" in result

    def test_rank_deficient_structure_returns_error(self):
        """Compiler should reject model specs with fewer manifests than latents."""
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "outcome_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                    "search_context": "",
                }
            ],
        }
        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "Treatment",
                        "role": "exogenous",
                        "description": "Treatment",
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "Outcome",
                        "role": "endogenous",
                        "description": "Outcome",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "outcome_score",
                        "construct_name": "Outcome",
                        "how_to_measure": "Use the outcome column directly",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
        }

        result = trial_compile_model_spec(spec, causal_spec)

        assert result is not None
        assert "Loading matrix is rank-deficient" in result


class TestStage4CompileOwnership:
    """Stage 4 should assert compilation, not retry via LLM feedback."""

    def test_stage4_compile_failure_raises_immediately(self, monkeypatch):
        """Compile failures should stop stage 4 instead of triggering retries."""
        from causal_ssm_agent.flows.stages.stage4_model import stage4_orchestrated_flow

        async def stub_propose_model_task(causal_spec: dict, question: str, raw_data: pl.DataFrame):
            return {"likelihoods": [], "parameters": [], "llm_trace": {"messages": []}}

        def stub_inject_marginalized_correlations(model_spec: dict, causal_spec: dict) -> None:
            return None

        def stub_trial_compile_model_spec(model_spec: dict, causal_spec: dict) -> str:
            return "dimension mismatch in drift matrix"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_model.propose_model_task",
            stub_propose_model_task,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.identifiability.inject_marginalized_correlations",
            stub_inject_marginalized_correlations,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_compiler.trial_compile_model_spec",
            stub_trial_compile_model_spec,
        )

        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "Outcome",
                        "role": "endogenous",
                        "description": "Outcome",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    }
                ],
                "edges": [],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "outcome_score",
                        "construct_name": "Outcome",
                        "how_to_measure": "Use the outcome column directly",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
            "identifiability": {
                "identifiable_treatments": {},
                "non_identifiable_treatments": {},
            },
        }

        with pytest.raises(ValueError, match="Stage 4 model spec failed compilation"):
            asyncio.run(
                stage4_orchestrated_flow(
                    causal_spec=causal_spec,
                    question="Does treatment affect outcome?",
                    raw_data=pl.DataFrame(schema={"indicator": pl.Utf8, "value": pl.Float64}),
                    enable_literature=False,
                )
            )

    def test_stage4_global_validation_failure_skips_prior_retries(self, monkeypatch):
        """Global validation failures should stop the prior retry loop immediately."""
        from causal_ssm_agent.flows.stages.stage4_assembly import (
            AssemblyValidation,
            materialize_stage4_result,
        )
        from causal_ssm_agent.flows.stages.stage4_model import stage4_orchestrated_flow

        proposed_spec = {
            "likelihoods": [
                {
                    "variable": "outcome_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                    "search_context": "outcome autocorrelation",
                }
            ],
            "llm_trace": {"messages": []},
        }
        global_failure = PriorValidationResult(
            parameter="model_build",
            is_valid=False,
            issue=(
                "Model build failed:\n"
                "Observation support check failed:\n"
                "- 'outcome_score' uses gamma emission but 1/10 observations are outside support"
            ),
            suggested_adjustment="Fix model_spec or priors to enable model construction",
        )

        async def stub_propose_model_task(causal_spec: dict, question: str, raw_data: pl.DataFrame):
            return proposed_spec

        def stub_validate_assembly(model_spec: dict, priors, raw_data, causal_spec):
            if priors is None:
                return AssemblyValidation(
                    normalized_model_spec=model_spec,
                    compile_ok=True,
                )
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                pp_checked=True,
                pp_valid=False,
                pp_results=[global_failure],
            )

        class _MapResult:
            def __init__(self, results: list[dict]):
                self._results = results

            def result(self) -> list[dict]:
                return self._results

        class _FakeElicitTask:
            def __init__(self):
                self.calls: list[list[str]] = []

            def map(self, parameter_specs, **_kwargs):
                names = [spec["name"] for spec in parameter_specs]
                self.calls.append(names)
                if len(self.calls) > 1:
                    raise AssertionError("unexpected retry on global validation failure")
                return _MapResult(
                    [
                        {
                            "parameter": name,
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "test",
                        }
                        for name in names
                    ]
                )

        fake_elicit_task = _FakeElicitTask()

        def stub_compile_model_artifact(*_args, **_kwargs):
            return {"model_built": False, "error": "global validation failure"}

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_model.propose_model_task",
            stub_propose_model_task,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_model.elicit_prior_task",
            fake_elicit_task,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_assembly.compile_model_artifact",
            stub_compile_model_artifact,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.config.get_config",
            lambda: SimpleNamespace(
                pipeline=SimpleNamespace(max_prior_retries=3),
                stage4_prior_elicitation=SimpleNamespace(
                    paraphrasing=SimpleNamespace(enabled=False, n_paraphrases=1)
                ),
            ),
        )

        authored_state = asyncio.run(
            stage4_orchestrated_flow(
                causal_spec={
                    "measurement": {
                        "model_clock": "1d",
                        "indicators": [
                            {
                                "name": "outcome_score",
                                "construct_name": "outcome",
                            }
                        ],
                    }
                },
                question="Does treatment affect outcome?",
                raw_data=pl.DataFrame({"indicator": ["outcome_score"], "value": [1.0]}),
                enable_literature=False,
            )
        )
        result = materialize_stage4_result(
            authored_state=authored_state,
            raw_data=pl.DataFrame({"indicator": ["outcome_score"], "value": [1.0]}),
            causal_spec={
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "outcome_score",
                            "construct_name": "outcome",
                        }
                    ],
                }
            },
        )

        assert fake_elicit_task.calls == [["rho_outcome"]]
        assert result["validation_retries"] is None
        assert result["is_valid"] is False
        assert result["validation"]["issues"] == [
            "Validation FAILED (global issue — affects all parameters):\n"
            "- Observation support check failed: 'outcome_score' uses gamma emission "
            "but 1/10 observations are outside support\n"
            "  Suggested: Fix model_spec or priors to enable model construction\n\n"
            "NOTE: This is a model_spec issue (likelihood family incompatible "
            "with observed data). Consider changing the distribution family "
            "rather than adjusting priors."
        ]
