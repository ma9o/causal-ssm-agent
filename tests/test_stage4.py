"""Tests for Stage 4: Model Specification & Prior Elicitation."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pytest

from causal_ssm_agent.models.prior_predictive import (
    _check_constraint_violations,
    _check_extreme_values,
    _check_nan_inf,
    format_parameter_feedback,
    format_validation_report,
    get_failed_parameters,
    validate_prior_predictive,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    EXPECTED_CONSTRAINT_FOR_ROLE,
    VALID_LINKS_FOR_DISTRIBUTION,
    DistributionFamily,
    LikelihoodSpec,
    LinkFunction,
    ModelSpec,
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    validate_model_spec,
)
from causal_ssm_agent.workers.prior_research import (
    aggregate_prior_samples,
    get_default_prior,
)
from causal_ssm_agent.workers.prompts.prior_research import generate_paraphrased_prompts
from causal_ssm_agent.workers.schemas_prior import (
    PriorProposal,
    PriorSource,
    PriorValidationResult,
    RawPriorSample,
)

# --- Fixtures ---


@pytest.fixture
def simple_model_spec() -> dict:
    """A minimal model spec for testing."""
    return {
        "likelihoods": [
            {
                "variable": "mood_score",
                "distribution": "Normal",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            }
        ],
        "random_effects": [],
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
        "model_clock": "daily",
        "reasoning": "Simple AR(1) model for mood",
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
            "confidence": 0.5,
            "reasoning": "Centered on scale midpoint",
        },
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "confidence": 0.5,
            "reasoning": "Weakly informative for AR coefficient",
        },
        "sigma_mood_score": {
            "parameter": "sigma_mood_score",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "confidence": 0.5,
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


# --- Schema Tests ---


class TestSchemas:
    """Test model and prior schemas."""

    def test_parameter_spec_validation(self):
        """ParameterSpec validates correctly."""
        spec = ParameterSpec(
            name="beta_stress_mood",
            role=ParameterRole.FIXED_EFFECT,
            constraint=ParameterConstraint.NONE,
            description="Effect of stress on mood",
            search_context="stress mood effect size",
        )
        assert spec.name == "beta_stress_mood"
        assert spec.role == ParameterRole.FIXED_EFFECT

    def test_likelihood_spec_validation(self):
        """LikelihoodSpec validates correctly."""
        spec = LikelihoodSpec(
            variable="mood_score",
            distribution=DistributionFamily.NORMAL,
            link=LinkFunction.IDENTITY,
            reasoning="Continuous outcome",
        )
        assert spec.distribution == DistributionFamily.NORMAL

    def test_model_spec_validation(self, simple_model_spec):
        """ModelSpec validates from dict."""
        spec = ModelSpec.model_validate(simple_model_spec)
        assert len(spec.likelihoods) == 1
        assert len(spec.parameters) == 3

    def test_prior_proposal_validation(self):
        """PriorProposal validates correctly."""
        proposal = PriorProposal(
            parameter="beta_x",
            distribution="Normal",
            params={"mu": 0.3, "sigma": 0.1},
            sources=[
                PriorSource(
                    title="Meta-analysis",
                    url="https://example.com",
                    snippet="Effect size r=0.3",
                    effect_size="r=0.3",
                )
            ],
            confidence=0.8,
            reasoning="Based on meta-analysis",
        )
        assert proposal.confidence == 0.8
        assert len(proposal.sources) == 1


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


# --- Prior Validation Tests ---


class TestPriorValidation:
    """Test prior predictive validation helpers."""

    def test_format_validation_report_passed(self):
        """Report formats correctly for passed validation."""
        results = [
            PriorValidationResult(
                parameter="x", is_valid=True, issue=None, suggested_adjustment=None
            ),
        ]
        report = format_validation_report(True, results)
        assert "PASSED" in report

    def test_format_validation_report_failed(self):
        """Report formats correctly for failed validation."""
        results = [
            PriorValidationResult(
                parameter="x", is_valid=False, issue="Bad values", suggested_adjustment=None
            ),
        ]
        report = format_validation_report(False, results)
        assert "FAILED" in report
        assert "x: Bad values" in report


# --- Default Prior Tests ---


class TestDefaultPriors:
    """Test default prior generation."""

    def test_default_prior_positive_constraint(self):
        """Positive constraint yields HalfNormal."""
        param = ParameterSpec(
            name="sigma_x",
            role=ParameterRole.RESIDUAL_SD,
            constraint=ParameterConstraint.POSITIVE,
            description="Residual SD",
            search_context="",
        )
        prior = get_default_prior(param)
        assert prior.distribution == "HalfNormal"

    def test_default_prior_unit_interval(self):
        """Unit interval constraint yields Beta."""
        param = ParameterSpec(
            name="rho_x",
            role=ParameterRole.AR_COEFFICIENT,
            constraint=ParameterConstraint.UNIT_INTERVAL,
            description="AR coefficient",
            search_context="",
        )
        prior = get_default_prior(param)
        assert prior.distribution == "Beta"

    def test_default_prior_correlation(self):
        """Correlation constraint yields Uniform(-1, 1)."""
        param = ParameterSpec(
            name="cor_xy",
            role=ParameterRole.CORRELATION,
            constraint=ParameterConstraint.CORRELATION,
            description="Correlation",
            search_context="",
        )
        prior = get_default_prior(param)
        assert prior.distribution == "Uniform"
        assert prior.params["lower"] == -1.0
        assert prior.params["upper"] == 1.0

    def test_default_prior_unconstrained(self):
        """Unconstrained parameter yields Normal."""
        param = ParameterSpec(
            name="beta_x",
            role=ParameterRole.FIXED_EFFECT,
            constraint=ParameterConstraint.NONE,
            description="Fixed effect",
            search_context="",
        )
        prior = get_default_prior(param)
        assert prior.distribution == "Normal"

    def test_default_prior_low_confidence(self):
        """Default priors have low confidence."""
        param = ParameterSpec(
            name="beta_x",
            role=ParameterRole.FIXED_EFFECT,
            constraint=ParameterConstraint.NONE,
            description="Fixed effect",
            search_context="",
        )
        prior = get_default_prior(param)
        assert prior.confidence == 0.3


# --- AutoElicit Tests ---


class TestAutoElicit:
    """Test AutoElicit-style paraphrased prompting and aggregation."""

    def test_aggregate_unimodal_uses_simple(self):
        """GMM with unimodal data falls back to simple pooling."""
        # All samples identical -> GMM should select K=1 -> simple pooling
        samples = [
            RawPriorSample(paraphrase_id=i, mu=0.3, sigma=0.1, confidence=0.8, reasoning="")
            for i in range(5)
        ]

        result = aggregate_prior_samples(samples)

        # Should fall back to simple (K=1 detected)
        assert result.method == "simple"
        assert result.mixture_weights is None
        assert np.isclose(result.mu, 0.3)

    def test_aggregate_bimodal_detects_mixture(self):
        """GMM detects multimodal distribution (K >= 2)."""
        # Create clearly bimodal samples with more separation
        samples = [
            RawPriorSample(paraphrase_id=0, mu=-2.0, sigma=0.1, confidence=0.8, reasoning=""),
            RawPriorSample(paraphrase_id=1, mu=-2.0, sigma=0.1, confidence=0.7, reasoning=""),
            RawPriorSample(paraphrase_id=2, mu=-2.0, sigma=0.1, confidence=0.9, reasoning=""),
            RawPriorSample(paraphrase_id=3, mu=2.0, sigma=0.1, confidence=0.8, reasoning=""),
            RawPriorSample(paraphrase_id=4, mu=2.0, sigma=0.1, confidence=0.7, reasoning=""),
            RawPriorSample(paraphrase_id=5, mu=2.0, sigma=0.1, confidence=0.9, reasoning=""),
        ]

        result = aggregate_prior_samples(samples)

        # Should detect K >= 2 mixture (BIC may select 2 or 3)
        assert result.method == "gmm"
        assert result.mixture_weights is not None
        assert len(result.mixture_weights) >= 2
        assert len(result.mixture_means) >= 2
        assert len(result.mixture_stds) >= 2
        # Means should span negative and positive values
        assert min(result.mixture_means) < 0
        assert max(result.mixture_means) > 0

    def test_aggregate_too_few_samples_uses_simple(self):
        """GMM with <3 samples falls back to simple."""
        samples = [
            RawPriorSample(paraphrase_id=0, mu=0.2, sigma=0.1, confidence=0.8, reasoning=""),
            RawPriorSample(paraphrase_id=1, mu=0.3, sigma=0.1, confidence=0.7, reasoning=""),
        ]

        result = aggregate_prior_samples(samples)

        assert result.method == "simple"
        # Simple pooling formula
        assert np.isclose(result.mu, 0.25)

    def test_paraphrase_generation_count(self):
        """generate_paraphrased_prompts returns N prompts."""
        prompts = generate_paraphrased_prompts(
            parameter_name="beta_stress_mood",
            parameter_role="fixed_effect",
            parameter_constraint="none",
            parameter_description="Effect of stress on mood",
            question="How does stress affect mood?",
            literature_context="No literature found.",
            n_paraphrases=10,
        )

        assert len(prompts) == 10

    def test_paraphrase_generation_fewer_than_10(self):
        """generate_paraphrased_prompts handles n < 10."""
        prompts = generate_paraphrased_prompts(
            parameter_name="beta_x",
            parameter_role="fixed_effect",
            parameter_constraint="none",
            parameter_description="Test param",
            question="Test question",
            literature_context="",
            n_paraphrases=3,
        )

        assert len(prompts) == 3

    def test_paraphrase_generation_content_varies(self):
        """Each paraphrase template produces different content."""
        prompts = generate_paraphrased_prompts(
            parameter_name="beta_x",
            parameter_role="fixed_effect",
            parameter_constraint="none",
            parameter_description="Test param",
            question="Test question",
            literature_context="",
            n_paraphrases=10,
        )

        # Each prompt should be unique (different template endings)
        assert len(set(prompts)) == 10

    def test_paraphrase_generation_contains_parameter_info(self):
        """Paraphrased prompts contain parameter info."""
        prompts = generate_paraphrased_prompts(
            parameter_name="beta_stress_mood",
            parameter_role="fixed_effect",
            parameter_constraint="none",
            parameter_description="Effect of stress on mood",
            question="How does stress affect mood?",
            literature_context="No literature found.",
            n_paraphrases=1,
        )

        assert "beta_stress_mood" in prompts[0]
        assert "fixed_effect" in prompts[0]
        assert "Effect of stress on mood" in prompts[0]

    def test_single_sample_returns_input(self):
        """With n=1, aggregation returns same values as input."""
        samples = [
            RawPriorSample(paraphrase_id=0, mu=0.3, sigma=0.15, confidence=0.8, reasoning="test"),
        ]

        result = aggregate_prior_samples(samples)

        # With single sample: mu = mu, sigma = sigma (var(mu) = 0)
        assert np.isclose(result.mu, 0.3)
        assert np.isclose(result.sigma, 0.15)
        assert result.n_samples == 1


# --- Domain Validation Tests ---


class TestDomainValidation:
    """Test domain validation rules from schemas_model."""

    def test_valid_spec_no_issues(self, simple_model_spec):
        """A well-formed ModelSpec produces no validation issues."""
        spec = ModelSpec.model_validate(simple_model_spec)
        issues = validate_model_spec(spec)
        assert issues == []

    def test_wrong_link_for_distribution(self):
        """Bernoulli + identity link is flagged as error."""
        spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="x",
                    distribution=DistributionFamily.BERNOULLI,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                )
            ],
            parameters=[],
            model_clock="daily",
            reasoning="test",
        )
        issues = validate_model_spec(spec)
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert "identity" in issues[0]["issue"]

    def test_wrong_constraint_for_role(self):
        """ar_coefficient + none constraint is flagged as warning."""
        spec = ModelSpec(
            likelihoods=[],
            parameters=[
                ParameterSpec(
                    name="rho_x",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.NONE,
                    description="test",
                    search_context="test",
                )
            ],
            model_clock="daily",
            reasoning="test",
        )
        issues = validate_model_spec(spec)
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "unit_interval" in issues[0]["issue"]

    def test_wrong_distribution_for_dtype(self):
        """Normal for binary dtype is flagged when indicators provided."""
        spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="flag",
                    distribution=DistributionFamily.NORMAL,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                )
            ],
            parameters=[],
            model_clock="daily",
            reasoning="test",
        )
        indicators = [{"name": "flag", "measurement_dtype": "binary"}]
        issues = validate_model_spec(spec, indicators=indicators)
        assert any(i["severity"] == "error" and "binary" in i["issue"] for i in issues)

    def test_all_distributions_have_link_rules(self):
        """Every DistributionFamily member has an entry in VALID_LINKS_FOR_DISTRIBUTION."""
        for dist in DistributionFamily:
            assert dist in VALID_LINKS_FOR_DISTRIBUTION, f"Missing link rule for {dist}"

    def test_all_roles_have_constraint_rules(self):
        """Every ParameterRole member has an entry in EXPECTED_CONSTRAINT_FOR_ROLE."""
        for role in ParameterRole:
            assert role in EXPECTED_CONSTRAINT_FOR_ROLE, f"Missing constraint rule for {role}"


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
        is_valid, results = validate_prior_predictive(
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
                    "distribution": "Normal",
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
            "model_clock": "daily",
            "reasoning": "test",
        }
        # This should still build (builder is tolerant), but let's test
        # with a truly broken spec by patching build_model to raise
        with patch(
            "causal_ssm_agent.models.ssm_builder.SSMModelBuilder.build_model",
            side_effect=ValueError("deliberate test failure"),
        ):
            is_valid, results = validate_prior_predictive(
                broken_spec, simple_priors, None, n_samples=10
            )
            assert is_valid is False
            assert any("model_build" in r.parameter for r in results)
            assert any("deliberate test failure" in (r.issue or "") for r in results)

    def test_nan_detection(self):
        """Mock samples with NaN -> caught."""
        samples = {
            "drift_diag_pop": jnp.array([1.0, float("nan"), 3.0]),
            "diffusion_diag_pop": jnp.array([1.0, 2.0, 3.0]),
        }
        result = _check_nan_inf(samples)
        assert result is not None
        assert result.is_valid is False
        assert "NaN" in result.issue
        assert "drift_diag_pop" in result.issue

    def test_extreme_values_detection(self):
        """Mock samples > 1e6 -> caught."""
        # 50% of values are extreme (above 10% threshold)
        samples = {
            "drift_diag_pop": jnp.array([1e7, 1e8, 0.5, 0.3]),
        }
        results = _check_extreme_values(samples)
        assert len(results) == 1
        assert results[0].is_valid is False
        assert "Extreme" in results[0].issue

    def test_constraint_violations_detection(self):
        """Positive-constrained sites with negative values -> caught."""
        # >1% negative
        vals = jnp.concatenate([jnp.ones(90), -jnp.ones(10)])
        samples = {"diffusion_diag_pop": vals}
        results = _check_constraint_violations(samples)
        assert len(results) == 1
        assert results[0].is_valid is False
        assert "negative" in results[0].issue

    def test_no_data_still_validates(self, simple_model_spec, simple_priors):
        """raw_data=None -> NaN/constraint/extreme checks run, scale skipped."""
        is_valid, results = validate_prior_predictive(
            simple_model_spec, simple_priors, None, n_samples=10
        )
        # Should still produce results (pass or fail) without crashing
        assert isinstance(is_valid, bool)
        assert isinstance(results, list)

    def test_format_report_integrates(self, simple_model_spec, simple_priors):
        """validate + format_validation_report -> well-formed string."""
        is_valid, results = validate_prior_predictive(
            simple_model_spec, simple_priors, None, n_samples=10
        )
        report = format_validation_report(is_valid, results)
        assert isinstance(report, str)
        assert "Prior predictive validation" in report
        if is_valid:
            assert "PASSED" in report

    def test_validate_priors_task_delegates(self, simple_model_spec, simple_priors):
        """Prefect task.fn() -> returns dict with expected keys."""
        from causal_ssm_agent.flows.stages.stage4_model import validate_priors_task

        raw_data = _make_polars_data()
        result = validate_priors_task.fn(simple_model_spec, simple_priors, raw_data)
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "results" in result
        assert "issues" in result


# --- Validation Feedback Tests ---


class TestValidationFeedback:
    """Test per-parameter validation feedback formatting."""

    def test_format_parameter_feedback_with_issues(self):
        """Feedback includes issue and suggestion for failed parameter."""
        results = [
            PriorValidationResult(
                parameter="scale_mood",
                is_valid=False,
                issue="Scale mismatch for mood: implied std (450.2) vs data std (1.3), ratio=346",
                suggested_adjustment="Adjust diffusion/drift priors to match data scale",
            ),
        ]
        prior = {"distribution": "Normal", "params": {"mu": 5.0, "sigma": 10.0}}

        # scale_mood is a global failure that maps to all params
        feedback = format_parameter_feedback("sigma_mood_score", results, prior=prior)

        assert "Normal(mu=5.0, sigma=10.0)" in feedback
        assert "Scale mismatch" in feedback
        assert "Adjust diffusion" in feedback

    def test_format_parameter_feedback_with_data_stats(self):
        """Feedback includes data scale reference."""
        results = [
            PriorValidationResult(
                parameter="dynamics_stability",
                is_valid=False,
                issue="Unstable dynamics: 8/10 prior draws have unstable drift",
                suggested_adjustment="Tighten drift_diag prior",
            ),
        ]
        data_stats = {"mood": {"mean": 5.0, "std": 1.3, "min": 1.0, "max": 9.0}}

        # dynamics_stability is a global failure
        feedback = format_parameter_feedback("rho_mood", results, data_stats=data_stats)

        assert "mood" in feedback
        assert "std=1.3" in feedback

    def test_format_parameter_feedback_empty_for_passing(self):
        """No feedback for parameters that passed."""
        results = [
            PriorValidationResult(
                parameter="x", is_valid=True, issue=None, suggested_adjustment=None
            ),
        ]
        feedback = format_parameter_feedback("x", results)
        assert feedback == ""


class TestFailedParameters:
    """Test failed parameter identification."""

    def test_global_failure_returns_all(self):
        """Model build failure affects all parameters."""
        results = [
            PriorValidationResult(
                parameter="model_build",
                is_valid=False,
                issue="Build failed",
                suggested_adjustment=None,
            ),
        ]
        failed = get_failed_parameters(results, ["rho_mood", "sigma_mood", "beta_stress"])
        assert set(failed) == {"rho_mood", "sigma_mood", "beta_stress"}

    def test_scale_mismatch_returns_all(self):
        """Scale mismatch affects all parameters."""
        results = [
            PriorValidationResult(
                parameter="scale_mood",
                is_valid=False,
                issue="Scale mismatch",
                suggested_adjustment=None,
            ),
        ]
        failed = get_failed_parameters(results, ["rho_mood", "sigma_mood"])
        assert set(failed) == {"rho_mood", "sigma_mood"}

    def test_drift_diag_failure_maps_to_ar(self):
        """drift_diag failure maps to AR coefficient parameters."""
        results = [
            PriorValidationResult(
                parameter="drift_diag_pop",
                is_valid=False,
                issue="Extreme values",
                suggested_adjustment=None,
            ),
        ]
        failed = get_failed_parameters(results, ["rho_mood", "sigma_mood", "beta_stress"])
        assert "rho_mood" in failed
        assert "beta_stress" not in failed

    def test_no_failures_returns_empty(self):
        """All passing -> empty list."""
        results = [
            PriorValidationResult(
                parameter="x", is_valid=True, issue=None, suggested_adjustment=None
            ),
        ]
        assert get_failed_parameters(results, ["x", "y"]) == []


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_prior_converts_to_mu_sigma(self, simple_model_spec):
        """Beta(2,2) prior for AR coefficient converts to valid mu/sigma."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        priors = {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "confidence": 0.5,
                "reasoning": "test",
            },
        }
        builder = SSMModelBuilder(model_spec=simple_model_spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, simple_model_spec)

        # Beta(2,2): E[X] = 0.5, Var[X] = 1/20 -> sigma ~ 0.224
        assert abs(ssm_priors.drift_diag["mu"] - 0.5) < 0.01
        assert abs(ssm_priors.drift_diag["sigma"] - 0.2236) < 0.01

    def test_halfnormal_prior_preserves_sigma(self, simple_model_spec):
        """HalfNormal(0.5) prior preserves sigma."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        priors = {
            "sigma_mood_score": {
                "parameter": "sigma_mood_score",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "confidence": 0.5,
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
                "confidence": 0.5,
                "reasoning": "test",
            },
        }
        builder = SSMModelBuilder(model_spec=spec, priors=priors)
        ssm_priors = builder._convert_priors_to_ssm(priors, spec)
        assert ssm_priors.lambda_free["sigma"] == 0.8

    def test_keyword_fallback_without_model_spec(self):
        """Without ModelSpec, keywords still map priors."""
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
