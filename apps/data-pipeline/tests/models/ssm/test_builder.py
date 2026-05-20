"""Tests for SSMModelBuilder helper functions.

Covers: normalize_prior_params, split_compound_name, fit-input preparation.
"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.models.ssm.builder import SSMModelBuilder, prepare_model_runtime
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors,
    compile_ssm_inputs_from_spec,
    normalize_prior_params,
    split_compound_name,
)
from nof1_causal_lab.models.ssm.dynamics.composite import (
    default_linear_drift_spec,
    linear_drift_spec,
)
from nof1_causal_lab.models.ssm.model import (
    SSMSpec,
    full_diagonal_mask,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    SparseMatrixBlockSpec,
    T0CholBlockSpec,
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
)

# =============================================================================
# normalize_prior_params
# =============================================================================


def _make_spec(
    *,
    n_latent: int = 1,
    n_manifest: int = 1,
    drift_spec=None,
    diffusion_block=None,
    lambda_block=None,
    manifest_means_block=None,
    manifest_chol_block=None,
    t0_means_block=None,
    t0_chol_block=None,
    input_effect_block=None,
    static_state_sd_block=None,
    **kwargs,
) -> SSMSpec:
    """Build an SSMSpec from explicit block specs for tests."""
    if drift_spec is None:
        drift_spec = default_linear_drift_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_spec=drift_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        input_effect_block=input_effect_block or default_input_effect_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        **kwargs,
    )


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
        """TruncatedNormal should preserve bounds."""
        result = normalize_prior_params(
            "TruncatedNormal",
            {"mu": 3.0, "sigma": 0.5, "lower": 0.0, "upper": 5.0},
        )
        assert result == {
            "family": 1,
            "mu": 3.0,
            "sigma": 0.5,
            "lower": 0.0,
            "upper": 5.0,
        }

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
        """Uniform(0, 1) should preserve uniform family metadata and bounds."""
        result = normalize_prior_params("Uniform", {"lower": 0.0, "upper": 1.0})
        assert result["family"] == 2
        assert result["mu"] == 0.5
        assert result["sigma"] == 0.25
        assert result["lower"] == 0.0
        assert result["upper"] == 1.0

    def test_uniform_symmetric(self):
        """Uniform(-2, 2) should preserve its midpoint/width summary and bounds."""
        result = normalize_prior_params("Uniform", {"lower": -2.0, "upper": 2.0})
        assert result["family"] == 2
        assert result["mu"] == 0.0
        assert result["sigma"] == 1.0
        assert result["lower"] == -2.0
        assert result["upper"] == 2.0

    def test_uniform_defaults(self):
        """Uniform with no bounds should default to -1, 1."""
        result = normalize_prior_params("Uniform", {})
        assert result["family"] == 2
        assert result["mu"] == 0.0
        assert result["sigma"] == 0.5
        assert result["lower"] == -1.0
        assert result["upper"] == 1.0

    def test_non_canonical_name_raises(self):
        """Only canonical prior distribution spellings should be accepted."""
        with pytest.raises(ValueError, match="Unsupported prior distribution family"):
            normalize_prior_params("normal", {"mu": 1.0, "sigma": 2.0})
        with pytest.raises(ValueError, match="Unsupported prior distribution family"):
            normalize_prior_params("NORMAL", {"mu": 1.0, "sigma": 2.0})
        with pytest.raises(ValueError, match="Unsupported prior distribution family"):
            normalize_prior_params("half_normal", {"sigma": 1.0})

    def test_gamma(self):
        """Gamma should preserve positive-support family metadata."""
        result = normalize_prior_params("Gamma", {"concentration": 3.0, "rate": 2.0})
        assert result == {"family": 1, "concentration": 3.0, "rate": 2.0}

    def test_lognormal(self):
        """LogNormal should serialize to log-scale loc/sigma runtime params."""
        result = normalize_prior_params("LogNormal", {"mu": 0.2, "sigma": 0.7})
        assert result == {"family": 2, "loc": 0.2, "sigma": 0.7}

    def test_exponential(self):
        """Exponential should preserve its own positive-support family metadata."""
        result = normalize_prior_params("Exponential", {"rate": 2.5})
        assert result == {"family": 3, "rate": 2.5}

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

    def test_empty_string_returns_none(self):
        """Empty string has no separator to split on."""
        result = split_compound_name("", {"x"}, {"y"})
        assert result is None

    def test_no_matching_prefix_returns_none(self):
        """When no valid prefix matches, returns None."""
        result = split_compound_name("foo_bar", {"baz"}, {"bar"})
        assert result is None

    def test_no_matching_suffix_returns_none(self):
        """When no valid suffix matches, returns None."""
        result = split_compound_name("foo_bar", {"foo"}, {"qux"})
        assert result is None


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
        ssm_spec = _make_spec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match="DT persistence scale"):
            compile_priors(priors, model_spec, ssm_spec=ssm_spec)

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        """Initial-state correlations should compile to bounded correlation priors."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "cor0_mood_sleep",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                }
            ],
        }
        priors = {
            "cor0_mood_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.2, "sigma": 0.8},
            }
        }
        t0_mask = np.zeros((2, 2), dtype=bool)
        t0_mask[1, 0] = True
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "sleep"],
            manifest_names=["mood", "sleep"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_mask=full_diagonal_mask(2),
                correlation_mask=t0_mask,
                template=jnp.eye(2),
            ),
        )

        prior_registry, _index_maps, _diagnostics = compile_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
        )
        t0_corr_prior = prior_registry.priors_by_site["t0_var_lower_free"]

        assert t0_corr_prior.params["mu"] == [0.2]
        assert t0_corr_prior.params["sigma"] == [0.8]
        assert t0_corr_prior.params["lower"] == [-1.0]
        assert t0_corr_prior.params["upper"] == [1.0]

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        """Authored initial-state priors should compile to the t0 mean/diag sites."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "t0_mean_mood",
                    "role": "initial_state_mean",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "t0_mean_sleep",
                    "role": "initial_state_mean",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "t0_sd_mood",
                    "role": "initial_state_sd",
                    "constraint": "positive",
                    "description": "",
                },
                {
                    "name": "t0_sd_sleep",
                    "role": "initial_state_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
        }
        priors = {
            "t0_mean_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.2},
            },
            "t0_mean_sleep": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.4},
            },
            "t0_sd_mood": {
                "distribution": "HalfNormal",
                "params": {"sigma": 0.7},
            },
            "t0_sd_sleep": {
                "distribution": "HalfNormal",
                "params": {"sigma": 0.9},
            },
        }
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "sleep"],
            manifest_names=["mood", "sleep"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_mask=full_diagonal_mask(2),
                correlation_mask=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )

        prior_registry, index_maps, _diagnostics = compile_priors(
            priors, model_spec, ssm_spec=ssm_spec
        )
        t0_mean_prior = prior_registry.priors_by_site["t0_means_free"]
        t0_diag_prior = prior_registry.priors_by_site["t0_var_diag_free"]

        assert t0_mean_prior.params["mu"] == [0.1, -0.3]
        assert t0_mean_prior.params["sigma"] == [0.2, 0.4]
        assert t0_diag_prior.params["sigma"] == [0.7, 0.9]
        assert index_maps[7]["t0_mean_mood"] == ("t0_means", 0)
        assert index_maps[7]["t0_mean_sleep"] == ("t0_means", 1)
        assert index_maps[8]["t0_sd_mood"] == ("t0_var_diag", 0)
        assert index_maps[8]["t0_sd_sleep"] == ("t0_var_diag", 1)

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        """Filtered initial-state pairs should not leave holes in prior arrays."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "a",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "b",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "c",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "cor0_A_B",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                },
                {
                    "name": "cor0_C_B",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                },
            ],
        }
        priors = {
            "cor0_C_B": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.2},
            }
        }
        t0_mask = np.zeros((3, 3), dtype=bool)
        t0_mask[2, 1] = True
        ssm_spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            latent_names=["A", "B", "C"],
            manifest_names=["a", "b", "c"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=3,
                diag_mask=full_diagonal_mask(3),
                correlation_mask=t0_mask,
                template=jnp.eye(3),
            ),
        )

        prior_registry, index_maps, _diagnostics = compile_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
        )
        t0_corr_prior = prior_registry.priors_by_site["t0_var_lower_free"]

        assert index_maps[6]["cor0_C_B"] == ("t0_var_offdiag", 0)
        assert t0_corr_prior.params["mu"] == [0.1]
        assert t0_corr_prior.params["sigma"] == [0.2]
        assert t0_corr_prior.params["lower"] == [-1.0]
        assert t0_corr_prior.params["upper"] == [1.0]

    def test_cross_lag_prior_requires_resolved_interval_metadata(self):
        """Cross-lag priors should fail instead of silently defaulting to 1 day."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                }
            ],
        }
        priors = {
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            }
        }
        drift_offdiag_mask = np.zeros((2, 2), dtype=bool)
        drift_offdiag_mask[0, 1] = True
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            manifest_names=["mood", "stress"],
            drift_spec=linear_drift_spec(
                n_latent=2,
                drift_diag_mask=full_diagonal_mask(2),
                drift_offdiag_mask=drift_offdiag_mask,
                drift_template=jnp.zeros((2, 2)),
                cint_mask=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
        )

        with pytest.raises(ValueError, match="could not resolve an authoring interval"):
            compile_priors(priors, model_spec, ssm_spec=ssm_spec)

    def test_compile_inputs_from_spec_requires_model_spec_for_semantic_priors(self):
        """Direct SSMSpec compilation should reject raw semantic priors without model_spec."""
        ssm_spec = _make_spec(n_latent=1, n_manifest=1, latent_names=["mood"])
        priors = {
            "rho_mood": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
            }
        }

        with pytest.raises(ValueError, match="requires model_spec to compile semantic prior"):
            compile_ssm_inputs_from_spec(ssm_spec=ssm_spec, priors=priors)


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

    def test_manifest_centering_applies_only_to_centered_channels(self):
        """prepare_fit_inputs should deterministically center only marked manifests."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_names=["x", "y"],
            manifest_centered=[True, False],
        )
        builder = SSMModelBuilder(ssm_spec=spec)
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "x": [10.0, 11.0, 12.0],
                "y": [5.0, 6.0, 7.0],
            }
        )

        observations, times, manifest_names = builder.prepare_fit_inputs(wide)

        assert manifest_names == ["x", "y"]
        np.testing.assert_allclose(np.asarray(times), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(observations[:, 1]), np.array([5.0, 6.0, 7.0]))

    def test_transition_inputs_are_scaled_filled_and_shifted_to_interval_start(self):
        """Known inputs are deterministic transition covariates aligned to interval starts."""
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            input_effect_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=1,
                mask=np.array([[True]]),
                template=jnp.zeros((1, 1)),
                free_site_name="input_effect_free",
                det_site_name="input_effect",
            ),
            input_names=["dose"],
            input_source_indicators=["dose_mg"],
            input_scales=[10.0],
            input_missing_policies=["forward_fill"],
        )
        builder = SSMModelBuilder(ssm_spec=spec)
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "dose_mg": [0.0, 20.0, None, 30.0],
                "mood_rating": [1.0, 2.0, 3.0, 4.0],
            }
        )

        transition_inputs = builder.prepare_transition_inputs(wide)

        assert transition_inputs is not None
        np.testing.assert_allclose(
            np.asarray(transition_inputs),
            np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32),
        )


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
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

            def set_transition_inputs(self, transition_inputs):
                self.transition_inputs = transition_inputs

        class StubBuilder:
            def __init__(self):
                self._attached_model = StubModel()

            def prepare_fit_inputs(self, wide_data: pl.DataFrame):
                return (
                    jnp.array([[jnp.nan], [1.0]], dtype=jnp.float32),
                    jnp.array(wide_data["time"].to_list(), dtype=jnp.float32),
                    ["stress_score"],
                )

            def prepare_transition_inputs(self, _wide_data: pl.DataFrame):
                return None

            @property
            def has_model(self) -> bool:
                return True

            @property
            def model(self):
                return self._attached_model

            @property
            def spec(self):
                return self._attached_model.spec

        with caplog.at_level("INFO"):
            runtime = prepare_model_runtime(
                data_for_model, builder=cast("SSMModelBuilder", StubBuilder())
            )

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
        assert runtime.model.observation_support is runtime.observation_support
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "aux_kalman_mcmc"
        assert runtime.inference_structure.method_override is None
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
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

            def set_transition_inputs(self, transition_inputs):
                self.transition_inputs = transition_inputs

        class StubBuilder:
            def __init__(self):
                self._attached_model = StubModel()

            def prepare_fit_inputs(self, wide_data: pl.DataFrame):
                return (
                    jnp.array([[jnp.nan], [jnp.nan], [3.0], [5.0]], dtype=jnp.float32),
                    jnp.array(wide_data["time"].to_list(), dtype=jnp.float32),
                    ["stress_score"],
                )

            def prepare_transition_inputs(self, _wide_data: pl.DataFrame):
                return None

            @property
            def has_model(self) -> bool:
                return True

            @property
            def model(self):
                return self._attached_model

            @property
            def spec(self):
                return self._attached_model.spec

        runtime = prepare_model_runtime(
            data_for_model, builder=cast("SSMModelBuilder", StubBuilder())
        )

        assert runtime.wide_data["time"].to_list() == [-2.0, -1.0, 0.0, 1.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "aux_kalman_mcmc"
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
            ssm_spec=_make_spec(
                n_latent=1,
                n_manifest=1,
                diffusion_block=DiffusionBlockSpec(
                    n_latent=1,
                    diffusion_chol_mask=np.diag(full_diagonal_mask(1)),
                    diffusion_chol_template=jnp.eye(1, dtype=jnp.float32),
                ),
                manifest_names=["stress_score"],
            ),
        )
        runtime = prepare_model_runtime(data_for_model, builder=builder)

        samples = runtime.builder.sample_prior_predictive(samples=3)

        assert samples["observations"].shape == (3, 2, 1)
        assert samples["observations_mask"].shape == (3, 2, 1)
        assert jnp.isnan(samples["observations"][:, 0, 0]).all()
        assert jnp.isfinite(samples["observations"][:, 1, 0]).all()
        assert (~samples["observations_mask"][:, 0, 0]).all()
        assert samples["observations_mask"][:, 1, 0].all()
