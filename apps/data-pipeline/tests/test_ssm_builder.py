"""Tests for SSMModelBuilder helper functions.

Covers: normalize_prior_params, split_compound_name, fit-input preparation.
"""

import jax.numpy as jnp
import polars as pl
import pytest

from causal_ssm_agent.models.ssm.model import SSMSpec
from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
from causal_ssm_agent.models.ssm_compilation import (
    compile_priors,
    normalize_prior_params,
    split_compound_name,
)

# =============================================================================
# normalize_prior_params
# =============================================================================


class TestNormalizePriorParams:
    def test_normal_returns_mu_sigma(self):
        """Normal distribution passes through mu/sigma."""
        result = normalize_prior_params("Normal", {"mu": 1.0, "sigma": 2.0})
        assert result == {"mu": 1.0, "sigma": 2.0}

    def test_normal_defaults(self):
        """Normal with no params should use defaults."""
        result = normalize_prior_params("Normal", {})
        assert result == {"mu": 0.0, "sigma": 1.0}

    def test_truncatednormal(self):
        """TruncatedNormal should behave like Normal."""
        result = normalize_prior_params("TruncatedNormal", {"mu": 3.0, "sigma": 0.5})
        assert result == {"mu": 3.0, "sigma": 0.5}

    def test_halfnormal(self):
        """HalfNormal should only return sigma."""
        result = normalize_prior_params("HalfNormal", {"sigma": 2.5})
        assert result == {"sigma": 2.5}

    def test_halfnormal_default(self):
        """HalfNormal with no sigma should default to 1.0."""
        result = normalize_prior_params("HalfNormal", {})
        assert result == {"sigma": 1.0}

    def test_beta_conversion(self):
        """Beta(2, 2) should give mu=0.5, sigma=sqrt(1/20)."""
        result = normalize_prior_params("Beta", {"alpha": 2.0, "beta": 2.0})
        expected_mu = 2.0 / 4.0  # 0.5
        expected_var = (2.0 * 2.0) / (16.0 * 5.0)  # 0.05
        assert abs(result["mu"] - expected_mu) < 1e-10
        assert abs(result["sigma"] - expected_var**0.5) < 1e-10

    def test_beta_asymmetric(self):
        """Beta(1, 3) should give correct mean."""
        result = normalize_prior_params("Beta", {"alpha": 1.0, "beta": 3.0})
        assert abs(result["mu"] - 0.25) < 1e-10

    def test_beta_defaults(self):
        """Beta with no params should default to alpha=2, beta=2."""
        result = normalize_prior_params("Beta", {})
        assert abs(result["mu"] - 0.5) < 1e-10

    def test_uniform_conversion(self):
        """Uniform(0, 1) should give mu=0.5, sigma=0.25, and bounds."""
        result = normalize_prior_params("Uniform", {"lower": 0.0, "upper": 1.0})
        assert result["mu"] == 0.5
        assert result["sigma"] == 0.25
        assert result["lower"] == 0.0
        assert result["upper"] == 1.0

    def test_uniform_symmetric(self):
        """Uniform(-2, 2) should give mu=0, sigma=1."""
        result = normalize_prior_params("Uniform", {"lower": -2.0, "upper": 2.0})
        assert result["mu"] == 0.0
        assert result["sigma"] == 1.0
        assert result["lower"] == -2.0
        assert result["upper"] == 2.0

    def test_uniform_defaults(self):
        """Uniform with no bounds should default to -1, 1."""
        result = normalize_prior_params("Uniform", {})
        assert result["mu"] == 0.0
        assert result["sigma"] == 0.5
        assert result["lower"] == -1.0
        assert result["upper"] == 1.0

    def test_case_insensitive(self):
        """Distribution name matching should be case-insensitive."""
        r1 = normalize_prior_params("normal", {"mu": 1.0, "sigma": 2.0})
        r2 = normalize_prior_params("NORMAL", {"mu": 1.0, "sigma": 2.0})
        assert r1 == r2

    def test_unknown_distribution_fallback(self):
        """Unknown distribution should fall back to mu/sigma extraction."""
        result = normalize_prior_params("Cauchy", {"mu": 1.0, "sigma": 2.0})
        assert result == {"mu": 1.0, "sigma": 2.0}

    def test_unknown_distribution_defaults(self):
        """Unknown distribution with no params should use defaults."""
        result = normalize_prior_params("SomeDistribution", {})
        assert result == {"mu": 0.0, "sigma": 1.0}


# =============================================================================
# split_compound_name
# =============================================================================


class TestSplitCompoundName:
    def test_simple_split(self):
        """Should split 'a_b' into ('a', 'b')."""
        result = split_compound_name("a_b", {"a"}, {"b"})
        assert result == ("a", "b")

    def test_multi_word_first(self):
        """Should handle multi-word first part."""
        result = split_compound_name(
            "stress_level_focus",
            {"stress_level"},
            {"focus"},
        )
        assert result == ("stress_level", "focus")

    def test_multi_word_second(self):
        """Should handle multi-word second part."""
        result = split_compound_name(
            "stress_focus_quality",
            {"stress"},
            {"focus_quality"},
        )
        assert result == ("stress", "focus_quality")

    def test_multi_word_both(self):
        """Should handle multi-word in both parts."""
        result = split_compound_name(
            "stress_level_focus_quality",
            {"stress_level"},
            {"focus_quality"},
        )
        assert result == ("stress_level", "focus_quality")

    def test_no_valid_split(self):
        """Should return None when no valid split exists."""
        result = split_compound_name("a_b_c", {"x"}, {"y"})
        assert result is None

    def test_single_word_no_split(self):
        """Single word with no underscore should return None."""
        result = split_compound_name("single", {"single"}, {"single"})
        assert result is None

    def test_first_valid_split_wins(self):
        """Should return the first valid split found (left to right)."""
        result = split_compound_name(
            "a_b_c",
            {"a", "a_b"},
            {"b_c", "c"},
        )
        # First split tried: ("a", "b_c") — both valid
        assert result == ("a", "b_c")

    def test_only_second_split_valid(self):
        """Should find a later split if the first isn't valid."""
        result = split_compound_name(
            "a_b_c",
            {"a_b"},
            {"c"},
        )
        assert result == ("a_b", "c")

    @pytest.mark.parametrize(
        "compound",
        ["", "_", "__"],
    )
    def test_edge_cases(self, compound):
        """Edge cases with empty strings should not crash."""
        result = split_compound_name(compound, {"", "_"}, {"", "_"})
        # Just verify it doesn't crash; result depends on valid sets
        assert result is None or isinstance(result, tuple)


class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        """AR priors must stay on the DT persistence scale in (0, 1)."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                }
            ],
        }
        priors = {
            "rho_mood": {
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 1.0},
            }
        }
        ssm_spec = SSMSpec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match="DT persistence scale"):
            compile_priors(priors, model_spec, ssm_spec=ssm_spec)


class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "screen_gap",
                    "distribution": "gamma",
                    "link": "log",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "name": "rho_screen_gap",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                }
            ],
        }
        builder = SSMModelBuilder(model_spec=model_spec, priors={})
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})

        with pytest.raises(ValueError, match="Observation support check failed"):
            builder.build_model(X)


class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        builder = SSMModelBuilder()
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "x": [10.0, None],
                "y": [None, 30.0],
            }
        )

        observations, times, manifest_names = builder.prepare_fit_inputs(wide)

        assert manifest_names == ["x", "y"]
        assert jnp.allclose(times, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.isclose(observations[0, 0], 10.0)
        assert jnp.isnan(observations[0, 1])
        assert jnp.isnan(observations[1, 0])
        assert jnp.isclose(observations[1, 1], 30.0)
