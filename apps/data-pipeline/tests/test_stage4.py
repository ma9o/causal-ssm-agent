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
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from causal_ssm_agent.models.prior_predictive import (
    get_failed_parameters,
    validate_prior_predictive,
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

    def test_validate_priors_task_delegates(self, simple_model_spec, simple_priors):
        """Prefect task.fn() -> returns dict with expected keys."""
        from causal_ssm_agent.flows.stages.stage4_model import validate_priors_task

        raw_data = _make_polars_data()
        result = validate_priors_task.fn(simple_model_spec, simple_priors, raw_data)
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "results" in result
        assert "issues" in result


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


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_prior_converts_to_mu_sigma(self, simple_model_spec):
        """Beta(2,2) AR prior converts via AR-to-drift transform."""
        import math

        from causal_ssm_agent.models.ssm import SSMSpec
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=simple_model_spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, simple_model_spec, ssm_spec=ssm_spec)

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
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        priors = {
            "sigma_mood_score": {
                "parameter": "sigma_mood_score",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }
        builder = SSMModelBuilder(model_spec=simple_model_spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, simple_model_spec)
        assert ssm_priors.diffusion_diag["sigma"] == 0.5

    def test_uniform_prior_converts(self):
        """Uniform(-1, 1) converts to Normal(0, 0.5)."""
        from causal_ssm_agent.models.ssm_builder import _normalize_prior_params

        result = _normalize_prior_params("Uniform", {"lower": -1.0, "upper": 1.0})
        assert result["mu"] == 0.0
        assert result["sigma"] == 0.5

    def test_role_based_mapping_covers_loading(self, simple_model_spec):
        """LOADING role maps to lambda_free SSMPriors field."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, spec)
        assert ssm_priors.lambda_free["sigma"] == 0.8

    def test_keyword_fallback_without_model_spec(self):
        """Without ModelSpec, keywords still map priors (no AR transform)."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        priors = {
            "rho_x": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.5},
            },
        }
        builder = SSMModelBuilder(priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, None)
        assert ssm_priors.drift_diag["mu"] == -0.3
        assert ssm_priors.drift_diag["sigma"] == 0.5

    def test_multiple_ar_params_produce_per_element_drift_diag(self):
        """Multiple AR params map to separate drift_diag array entries."""
        import math

        from causal_ssm_agent.models.ssm import SSMSpec
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

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
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors, causal_spec=causal_spec)
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

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
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

        # Daily default: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu = ssm_priors.drift_offdiag["mu"]
        mu_val = mu[0] if isinstance(mu, list) else mu
        assert abs(mu_val - 0.3) < 0.01

    def test_beta_prior_dt_to_ct_respects_granularity(self):
        """FIXED_EFFECT beta transform uses effect construct's granularity."""
        from causal_ssm_agent.models.ssm import SSMSpec
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors, causal_spec=causal_spec)
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

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
            "causal_ssm_agent.models.ssm_compiler.compile_ssm_artifact",
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
