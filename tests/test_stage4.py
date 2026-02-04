"""Tests for Stage 4: Model Specification & Prior Elicitation."""

import numpy as np
import pandas as pd
import pytest

from dsem_agent.models.prior_predictive import (
    _get_constraint_from_distribution,
    _validate_parameter,
    _validate_prior_predictive_samples,
    format_validation_report,
)
from dsem_agent.orchestrator.schemas_glmm import (
    DistributionFamily,
    GLMMSpec,
    LikelihoodSpec,
    LinkFunction,
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from dsem_agent.workers.prior_research import (
    aggregate_prior_samples,
    get_default_prior,
)
from dsem_agent.workers.prompts.prior_research import generate_paraphrased_prompts
from dsem_agent.workers.schemas_prior import (
    PriorProposal,
    PriorSource,
    PriorValidationResult,
    RawPriorSample,
)


# --- Fixtures ---


@pytest.fixture
def simple_glmm_spec() -> dict:
    """A minimal GLMM spec for testing."""
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
    """Simple priors matching the GLMM spec."""
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
    return pd.DataFrame({
        "mood_score": np.random.randn(n) * 1.5 + 5,
        "mood_score_lag1": np.random.randn(n) * 1.5 + 5,
        "subject_id": np.repeat(np.arange(5), 10),
    })


# --- Schema Tests ---


class TestSchemas:
    """Test GLMM and prior schemas."""

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

    def test_glmm_spec_validation(self, simple_glmm_spec):
        """GLMMSpec validates from dict."""
        spec = GLMMSpec.model_validate(simple_glmm_spec)
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


# --- CTSEMModelBuilder Tests ---
# NOTE: Full tests will be added when implementation is merged from numpyro-ctsem


class TestCTSEMModelBuilder:
    """Test CT-SEM model building."""

    def test_builder_init(self, simple_glmm_spec, simple_priors):
        """Builder initializes with spec and priors."""
        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        builder = CTSEMModelBuilder(
            glmm_spec=simple_glmm_spec,
            priors=simple_priors,
        )
        assert builder._model_type == "CT-SEM"
        assert builder.version == "0.1.0"

    def test_builder_builds_model(self, simple_glmm_spec, simple_priors, simple_data):
        """Builder creates a CT-SEM model."""
        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        builder = CTSEMModelBuilder(
            glmm_spec=simple_glmm_spec,
            priors=simple_priors,
        )
        model = builder.build_model(simple_data)
        assert model is not None
        assert model.spec.n_manifest == 1  # mood_score only


# --- Prior Validation Tests ---


class TestPriorValidation:
    """Test prior predictive validation helpers."""

    def test_get_constraint_positive(self):
        """HalfNormal implies positive constraint."""
        assert _get_constraint_from_distribution("HalfNormal") == "positive"
        assert _get_constraint_from_distribution("Gamma") == "positive"
        assert _get_constraint_from_distribution("Exponential") == "positive"

    def test_get_constraint_unit_interval(self):
        """Beta implies unit_interval constraint."""
        assert _get_constraint_from_distribution("Beta") == "unit_interval"

    def test_get_constraint_none(self):
        """Normal has no implicit constraint."""
        assert _get_constraint_from_distribution("Normal") == "none"
        assert _get_constraint_from_distribution("Unknown") == "none"

    def test_validate_prior_predictive_samples_valid(self):
        """Valid samples pass validation."""
        samples = np.random.randn(100) + 5
        result = _validate_prior_predictive_samples("mood", samples, "Normal")
        assert result.is_valid
        assert result.issue is None

    def test_validate_prior_predictive_samples_nan(self):
        """NaN samples fail validation."""
        samples = np.array([1.0, 2.0, np.nan, 4.0])
        result = _validate_prior_predictive_samples("mood", samples, "Normal")
        assert not result.is_valid
        assert "NaN/Inf" in result.issue

    def test_validate_prior_predictive_samples_negative_poisson(self):
        """Negative samples fail for Poisson."""
        samples = np.array([1.0, 2.0, -1.0, 4.0])
        result = _validate_prior_predictive_samples("count", samples, "Poisson")
        assert not result.is_valid
        assert "negative" in result.issue

    def test_validate_prior_predictive_samples_outside_beta(self):
        """Samples outside [0,1] fail for Beta."""
        samples = np.array([0.5, 0.8, 1.5, 0.2])
        result = _validate_prior_predictive_samples("prob", samples, "Beta")
        assert not result.is_valid
        assert "outside [0, 1]" in result.issue

    def test_format_validation_report_passed(self):
        """Report formats correctly for passed validation."""
        results = [
            PriorValidationResult(parameter="x", is_valid=True, issue=None, suggested_adjustment=None),
        ]
        report = format_validation_report(True, results)
        assert "PASSED" in report

    def test_format_validation_report_failed(self):
        """Report formats correctly for failed validation."""
        results = [
            PriorValidationResult(parameter="x", is_valid=False, issue="Bad values", suggested_adjustment=None),
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
