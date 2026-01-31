"""Tests for Stage 4: Model Specification."""

import pytest

from dsem_agent.flows.stages.stage4_model import (
    DEFAULT_PRIORS,
    DTYPE_TO_DISTRIBUTION,
    specify_model,
    elicit_priors_sync,
)


@pytest.fixture
def simple_dsem_model():
    """A simple DSEM model for testing."""
    return {
        "latent": {
            "constructs": [
                {
                    "name": "stress",
                    "description": "Psychological stress level",
                    "role": "endogenous",
                    "is_outcome": False,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "is_outcome": False,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "mood",
                    "description": "Emotional state",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "causal_granularity": "daily",
                },
                {
                    "name": "trait_anxiety",
                    "description": "Stable anxiety disposition",
                    "role": "exogenous",
                    "is_outcome": False,
                    "temporal_status": "time_invariant",
                    "causal_granularity": None,
                },
            ],
            "edges": [
                {
                    "cause": "stress",
                    "effect": "mood",
                    "description": "Stress reduces mood",
                    "lagged": True,
                },
                {
                    "cause": "sleep",
                    "effect": "mood",
                    "description": "Poor sleep worsens mood",
                    "lagged": True,
                },
                {
                    "cause": "trait_anxiety",
                    "effect": "stress",
                    "description": "Anxious people experience more stress",
                    "lagged": False,
                },
            ],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "hrv",
                    "construct": "stress",
                    "how_to_measure": "Heart rate variability",
                    "measurement_granularity": "hourly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "cortisol",
                    "construct": "stress",
                    "how_to_measure": "Cortisol level",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_hours",
                    "construct": "sleep",
                    "how_to_measure": "Hours of sleep",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "sum",
                },
                {
                    "name": "mood_rating",
                    "construct": "mood",
                    "how_to_measure": "Self-reported mood 1-10",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "name": "stai_score",
                    "construct": "trait_anxiety",
                    "how_to_measure": "STAI trait score",
                    "measurement_granularity": "daily",
                    "measurement_dtype": "continuous",
                    "aggregation": "first",
                },
            ],
        },
        "identifiability": None,
    }


class TestSpecifyModel:
    """Tests for specify_model()."""

    def test_model_clock_is_finest_granularity(self, simple_dsem_model):
        """Model clock should be the finest endogenous granularity."""
        spec = specify_model({}, simple_dsem_model)
        assert spec["time_index"] == "daily"

    def test_constructs_have_required_fields(self, simple_dsem_model):
        """Each construct spec should have required fields."""
        spec = specify_model({}, simple_dsem_model)

        for name, construct in spec["constructs"].items():
            assert "name" in construct
            assert "role" in construct
            assert "temporal_status" in construct
            assert "is_outcome" in construct

    def test_endogenous_time_varying_get_ar_prior(self, simple_dsem_model):
        """Endogenous time-varying constructs should get AR priors."""
        spec = specify_model({}, simple_dsem_model)

        # stress, sleep, mood are endogenous time-varying
        assert "ar_prior" in spec["constructs"]["stress"]
        assert "ar_prior" in spec["constructs"]["sleep"]
        assert "ar_prior" in spec["constructs"]["mood"]

        # trait_anxiety is exogenous, no AR
        assert "ar_prior" not in spec["constructs"]["trait_anxiety"]

    def test_endogenous_get_sigma_prior(self, simple_dsem_model):
        """Endogenous constructs should get residual variance priors."""
        spec = specify_model({}, simple_dsem_model)

        assert "sigma_prior" in spec["constructs"]["stress"]
        assert "sigma_prior" in spec["constructs"]["mood"]
        assert "sigma_prior" not in spec["constructs"]["trait_anxiety"]

    def test_edges_have_coefficient_priors(self, simple_dsem_model):
        """Each edge should have a beta coefficient prior."""
        spec = specify_model({}, simple_dsem_model)

        assert len(spec["edges"]) == 3

        for param_name, edge in spec["edges"].items():
            assert param_name.startswith("beta_")
            assert "prior" in edge
            assert "cause" in edge
            assert "effect" in edge
            assert "lagged" in edge

    def test_single_indicator_loading_fixed(self, simple_dsem_model):
        """Single-indicator constructs should have loading=1.0."""
        spec = specify_model({}, simple_dsem_model)

        # mood has only mood_rating
        assert spec["measurement"]["mood_rating"]["loading"] == 1.0
        assert spec["measurement"]["mood_rating"]["loading_prior"] is None

    def test_multi_indicator_first_loading_fixed(self, simple_dsem_model):
        """Multi-indicator: first loading fixed, rest estimated."""
        spec = specify_model({}, simple_dsem_model)

        # stress has hrv and cortisol
        # First one (hrv) should be fixed
        assert spec["measurement"]["hrv"]["loading"] == 1.0
        assert spec["measurement"]["hrv"]["loading_prior"] is None

        # Second one (cortisol) should be estimated
        assert spec["measurement"]["cortisol"]["loading"] is None
        assert spec["measurement"]["cortisol"]["loading_prior"] is not None

    def test_dtype_to_distribution_mapping(self, simple_dsem_model):
        """Indicators should get correct distribution based on dtype."""
        spec = specify_model({}, simple_dsem_model)

        # All indicators are continuous
        for ind_spec in spec["measurement"].values():
            assert ind_spec["distribution"] == "Normal"
            assert ind_spec["link"] == "identity"


class TestElicitPriorsSync:
    """Tests for elicit_priors_sync() (default priors)."""

    def test_returns_all_parameters(self, simple_dsem_model):
        """Should return priors for all parameters."""
        spec = specify_model({}, simple_dsem_model)
        priors = elicit_priors_sync(spec)

        # AR priors for endogenous time-varying
        assert "rho_stress" in priors
        assert "rho_sleep" in priors
        assert "rho_mood" in priors

        # Sigma priors for endogenous
        assert "sigma_stress" in priors
        assert "sigma_mood" in priors

        # Beta priors for edges
        assert "beta_mood_stress" in priors
        assert "beta_mood_sleep" in priors
        assert "beta_stress_trait_anxiety" in priors

    def test_priors_have_required_fields(self, simple_dsem_model):
        """Each prior should have mean, std, reasoning, source."""
        spec = specify_model({}, simple_dsem_model)
        priors = elicit_priors_sync(spec)

        for param_name, prior in priors.items():
            assert "mean" in prior
            assert "std" in prior
            assert "reasoning" in prior
            assert "source" in prior
            assert prior["source"] == "default"

    def test_loading_priors_for_multi_indicator(self, simple_dsem_model):
        """Should have loading priors for non-reference indicators."""
        spec = specify_model({}, simple_dsem_model)
        priors = elicit_priors_sync(spec)

        # cortisol is 2nd indicator for stress, should have loading prior
        assert "lambda_cortisol" in priors


class TestDtypeMapping:
    """Tests for dtype to distribution mapping."""

    def test_all_dtypes_mapped(self):
        """All valid dtypes should have mappings."""
        valid_dtypes = ["continuous", "binary", "count", "ordinal", "categorical"]

        for dtype in valid_dtypes:
            assert dtype in DTYPE_TO_DISTRIBUTION
            assert "dist" in DTYPE_TO_DISTRIBUTION[dtype]
            assert "link" in DTYPE_TO_DISTRIBUTION[dtype]

    def test_continuous_is_normal(self):
        """Continuous should map to Normal with identity link."""
        assert DTYPE_TO_DISTRIBUTION["continuous"]["dist"] == "Normal"
        assert DTYPE_TO_DISTRIBUTION["continuous"]["link"] == "identity"

    def test_binary_is_bernoulli(self):
        """Binary should map to Bernoulli with logit link."""
        assert DTYPE_TO_DISTRIBUTION["binary"]["dist"] == "Bernoulli"
        assert DTYPE_TO_DISTRIBUTION["binary"]["link"] == "logit"

    def test_count_is_poisson(self):
        """Count should map to Poisson with log link."""
        assert DTYPE_TO_DISTRIBUTION["count"]["dist"] == "Poisson"
        assert DTYPE_TO_DISTRIBUTION["count"]["link"] == "log"


class TestDefaultPriors:
    """Tests for default prior specifications."""

    def test_ar_bounded_zero_one(self):
        """AR prior should be bounded [0, 1]."""
        ar_prior = DEFAULT_PRIORS["ar"]
        assert ar_prior["dist"] == "Uniform"
        assert ar_prior["lower"] == 0.0
        assert ar_prior["upper"] == 1.0

    def test_beta_centered_at_zero(self):
        """Beta prior should be centered at zero."""
        beta_prior = DEFAULT_PRIORS["beta"]
        assert beta_prior["dist"] == "Normal"
        assert beta_prior["mean"] == 0.0

    def test_sigma_is_positive(self):
        """Sigma prior should be positive (HalfNormal)."""
        sigma_prior = DEFAULT_PRIORS["sigma"]
        assert sigma_prior["dist"] == "HalfNormal"
