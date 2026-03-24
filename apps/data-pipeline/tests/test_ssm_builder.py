"""Tests for SSMModelBuilder helper functions.

Covers: normalize_prior_params, split_compound_name, fit-input preparation.
"""

import jax.numpy as jnp
import polars as pl
import pytest

from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm_builder import SSMModelBuilder, prepare_model_runtime
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

    def test_gamma(self):
        """Gamma should preserve positive-support family metadata."""
        result = normalize_prior_params("Gamma", {"concentration": 3.0, "rate": 2.0})
        assert result == {"family": 1, "concentration": 3.0, "rate": 2.0}

    def test_unknown_distribution_raises(self):
        """Unknown prior distributions should fail early."""
        with pytest.raises(ValueError, match="Unsupported prior distribution family"):
            normalize_prior_params("Cauchy", {"mu": 1.0, "sigma": 2.0})


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


class TestPrepareModelRuntime:
    def test_preserves_long_observation_metadata_and_augments_support_boundaries(self, caplog):
        data_for_model = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.likelihood = "particle"
                self.spec = SSMSpec(
                    n_latent=1,
                    n_manifest=1,
                    lambda_mat=jnp.eye(1, dtype=jnp.float32),
                    manifest_names=["stress_score"],
                )

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

        class StubBuilder:
            def __init__(self):
                self._model = StubModel()
                self._spec = self._model.spec

            def prepare_fit_inputs(self, wide_data: pl.DataFrame):
                return (
                    jnp.array([[jnp.nan], [1.0]], dtype=jnp.float32),
                    jnp.array(wide_data["time"].to_list(), dtype=jnp.float32),
                    ["stress_score"],
                )

        with caplog.at_level("INFO"):
            runtime = prepare_model_runtime(data_for_model, builder=StubBuilder())

        assert runtime.observation_data is not None
        assert runtime.observation_data.columns == data_for_model.columns
        assert runtime.observation_data["observation_window"][0] == "1mo"
        assert runtime.observation_data["support_end"][0] == "2024-02-01T00:00:00"
        assert runtime.observation_data["anchor_time"][0] == "2024-02-01T00:00:00"
        assert runtime.wide_data["time"].to_list() == [-31.0, 0.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.manifest_names == ["stress_score"]
        assert runtime.observation_support.support_kinds == ["interval"]
        assert runtime.observation_support.summary_operators == ["mean"]
        assert runtime.observation_support.anchor_policies == ["support_end"]
        assert runtime.observation_support.observation_windows == ["1mo"]
        assert runtime.observation_support.requires_interval_summary_handling is True
        assert runtime.observation_support.interval_summary_manifest_names == ["stress_score"]
        assert runtime.observation_support.support_start_times.shape == (2, 1)
        assert runtime.observation_support.support_end_times.shape == (2, 1)
        assert runtime.observation_support.support_start_times[1, 0] == pytest.approx(-31.0)
        assert runtime.observation_support.support_end_times[1, 0] == pytest.approx(0.0)
        assert runtime.observation_support.interval_prev_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_curr_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_weights.shape == (2, 1, 1)
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [0]]
        assert runtime.observation_support.interval_prev_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_curr_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(31.0)
        assert runtime.manifest_names == ["stress_score"]
        assert runtime.builder._model.observation_support is runtime.observation_support
        assert runtime.inference_structure.likelihood_path == "particle"
        assert runtime.inference_structure.auto_method == "svi"
        assert (
            runtime.inference_structure.first_pass_rb.inactive_reason == "interval_summary_support"
        )
        assert "support-aware observation semantics" in caplog.text

    def test_compiles_overlapping_interval_windows_into_concurrent_slots(self):
        data_for_model = pl.DataFrame(
            {
                "indicator": ["stress_score", "stress_score"],
                "value": [3.0, 5.0],
                "anchor_time": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
                "support_kind": ["interval", "interval"],
                "summary_operator": ["mean", "mean"],
                "anchor_policy": ["support_end", "support_end"],
                "observation_window": ["2d", "2d"],
                "support_start": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                "support_end": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.likelihood = "particle"
                self.spec = SSMSpec(
                    n_latent=1,
                    n_manifest=1,
                    lambda_mat=jnp.eye(1, dtype=jnp.float32),
                    manifest_names=["stress_score"],
                )

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

        class StubBuilder:
            def __init__(self):
                self._model = StubModel()
                self._spec = self._model.spec

            def prepare_fit_inputs(self, wide_data: pl.DataFrame):
                return (
                    jnp.array([[jnp.nan], [jnp.nan], [3.0], [5.0]], dtype=jnp.float32),
                    jnp.array(wide_data["time"].to_list(), dtype=jnp.float32),
                    ["stress_score"],
                )

        runtime = prepare_model_runtime(data_for_model, builder=StubBuilder())

        assert runtime.wide_data["time"].to_list() == [-2.0, -1.0, 0.0, 1.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
        assert runtime.inference_structure.likelihood_path == "particle"
        assert (
            runtime.inference_structure.first_pass_rb.inactive_reason == "interval_summary_support"
        )
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [-1], [0], [1]]
        assert runtime.observation_support.interval_weights.shape == (4, 1, 2)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 1] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[3, 0, 1] == pytest.approx(1.0)

    def test_builder_prior_predictive_reuses_prepared_support_schedule(self):
        data_for_model = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )
        builder = SSMModelBuilder(
            ssm_spec=SSMSpec(
                n_latent=1,
                n_manifest=1,
                lambda_mat=jnp.eye(1, dtype=jnp.float32),
                diffusion="diag",
                manifest_names=["stress_score"],
            ),
            ssm_priors=SSMPriors(),
        )
        runtime = prepare_model_runtime(data_for_model, builder=builder)

        samples = runtime.builder.sample_prior_predictive(samples=3)

        assert samples["observations"].shape == (3, 2, 1)
        assert jnp.isnan(samples["observations"][:, 0, 0]).all()
        assert jnp.isfinite(samples["observations"][:, 1, 0]).all()
