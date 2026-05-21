"""End-to-end tests: CausalSpec → Model Spec → Prior Conversion → Discretization.

These tests verify the full chain from a realistic causal specification
through DT→CT prior conversion and CT→DT discretization, checking that
the mathematical roundtrip is consistent.

Phase 1 tests:
- reference_interval_days precedence chain for DT→CT conversion
- SSMSpec structure (drift_mask, lambda_mask) from DAG
- First-order DT→CT→DT roundtrip consistency
- Prior predictive produces finite, stable samples

Phase 2 tests:
- Exact matrix logarithm DT→CT conversion
- Embeddability conditions for the transition matrix
- First-order vs exact approximation error bounds
"""

import math
from copy import deepcopy

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.models.ssm import SSMSpec, discretize_system
from nof1_causal_lab.models.ssm.builder import SSMModelBuilder
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors as compile_ssm_priors,
)
from nof1_causal_lab.models.ssm.compile.inputs import (
    translate_spec as translate_ssm_spec,
)
from tests.ssm_test_utils import block_ssm_spec, structural_dense_drift_spec


def _block_spec_with_drift_offdiag(
    *,
    n_latent: int,
    n_manifest: int,
    drift_offdiag_mask: np.ndarray,
    latent_names: list[str] | None = None,
) -> SSMSpec:
    return block_ssm_spec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_spec=structural_dense_drift_spec(
            n_latent=n_latent,
            drift_diag_mask=np.ones(n_latent, dtype=bool),
            drift_offdiag_mask=drift_offdiag_mask,
            drift_template=jnp.zeros((n_latent, n_latent)),
            cint_mask=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        latent_names=latent_names,
    )


def _with_estimation_projection(causal_spec: dict) -> dict:
    causal_spec = deepcopy(causal_spec)
    measurement = causal_spec.get("measurement", {})
    indicators = measurement.get("indicators", [])
    for indicator in indicators:
        if isinstance(indicator, dict) and "construct_polarity" not in indicator:
            indicator["construct_polarity"] = "positive"
    latent = causal_spec.get("latent", {})
    causal_spec["estimation"] = {
        "state_order": [
            construct["name"]
            for construct in latent.get("constructs", [])
            if isinstance(construct, dict) and isinstance(construct.get("name"), str)
        ],
        "edges": deepcopy(latent.get("edges", [])),
        "induced_dependencies": [],
    }
    return causal_spec


def _translate_spec_for_test(model_spec: dict, causal_spec: dict | None = None):
    return translate_ssm_spec(model_spec, causal_spec=causal_spec)


def _compile_priors_for_test(
    priors: dict[str, dict],
    model_spec: dict,
    *,
    ssm_spec: SSMSpec | None = None,
    causal_spec: dict | None = None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
):
    prior_registry, index_maps, _diagnostics = compile_ssm_priors(
        priors,
        model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )
    return prior_registry, index_maps


def _prior_vector(value, *, dtype=float):
    array = np.asarray(value if isinstance(value, list | tuple) else [value], dtype=dtype)
    return array.reshape(-1)


def _prior_params(prior_registry, site_name: str):
    return prior_registry.priors_by_site[site_name].params


def _base_decay_means(prior_registry) -> np.ndarray:
    prior = _prior_params(prior_registry, "vf_0_base_decay")
    if "concentration" in prior and "rate" in prior:
        return _prior_vector(prior["concentration"]) / _prior_vector(prior["rate"])
    if "value" in prior:
        return _prior_vector(prior["value"])
    if "sigma" in prior:
        return _prior_vector(prior["sigma"]) * math.sqrt(2.0 / math.pi)
    raise AssertionError(f"Unsupported drift_base_decay prior payload: {prior}")


def _mean_drift_from_priors(spec: SSMSpec, prior_registry) -> jnp.ndarray:
    base_decay = jnp.asarray(_base_decay_means(prior_registry), dtype=jnp.float32)
    offdiag = jnp.asarray(
        _prior_vector(_prior_params(prior_registry, "vf_0_offdiag")["mu"]),
        dtype=jnp.float32,
    )
    drift_component = spec.drift_spec.components[0]
    return drift_component.assemble_drift(base_decay, offdiag)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def two_construct_causal_spec() -> dict:
    """Realistic 2-construct causal spec: stress → mood.

    - Both constructs are daily time-varying
    - 3 indicators: mood_rating, stress_self_report, stress_cortisol
    - stress_cortisol is a second indicator for stress (free loading)
    """
    return _with_estimation_projection(
        {
            "latent": {
                "constructs": [
                    {
                        "name": "mood",
                        "description": "Daily mood state",
                        "role": "endogenous",
                        "is_outcome": True,
                        "temporal_status": "time_varying",
                    },
                    {
                        "name": "stress",
                        "description": "Daily stress level",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause": "stress",
                        "effect": "mood",
                        "description": "Stress impairs mood",
                        "lagged": True,
                    },
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "mood_rating",
                        "construct_name": "mood",
                        "how_to_measure": "Self-reported mood (1-10)",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "stress_self_report",
                        "construct_name": "stress",
                        "how_to_measure": "Self-reported stress (1-10)",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "stress_cortisol",
                        "construct_name": "stress",
                        "how_to_measure": "Salivary cortisol (nmol/L)",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
        }
    )


@pytest.fixture
def two_construct_model_spec() -> dict:
    """ModelSpec matching the 2-construct causal spec."""
    return {
        "likelihoods": [
            {
                "variable": "mood_rating",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            },
            {
                "variable": "stress_self_report",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            },
            {
                "variable": "stress_cortisol",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous biomarker",
            },
        ],
        "parameters": [
            {
                "name": "rho_mood",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for mood",
            },
            {
                "name": "rho_stress",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for stress",
            },
            {
                "name": "beta_stress_mood",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Cross-lagged effect of stress on mood",
            },
            {
                "name": "sigma_mood",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for mood",
            },
            {
                "name": "sigma_stress",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for stress",
            },
            {
                "name": "lambda_stress_cortisol_stress",
                "role": "loading",
                "constraint": "positive",
                "description": "Loading: stress → stress_cortisol",
            },
            {
                "name": "obs_sd_stress_self_report",
                "role": "measurement_error_sd",
                "constraint": "positive",
                "description": "Measurement error SD for stress self-report",
            },
            {
                "name": "obs_sd_stress_cortisol",
                "role": "measurement_error_sd",
                "constraint": "positive",
                "description": "Measurement error SD for stress cortisol",
            },
        ],
    }


@pytest.fixture
def weekly_study_priors() -> dict[str, dict]:
    """Priors from a weekly-interval study (reference_interval_days=7).

    AR coefficients: Beta(3,2) → E=0.6 (mood), Beta(2,2) → E=0.5 (stress)
    Cross-lag: Normal(0.3, 0.15) at weekly scale
    """
    return {
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 3.0, "beta": 2.0},
            "sources": [
                {
                    "title": "Weekly mood dynamics meta-analysis",
                    "snippet": "AR(1) ≈ 0.6 at weekly interval",
                    "study_interval_days": 7.0,
                }
            ],
            "reasoning": "Meta-analysis of weekly diary studies",
            "reference_interval_days": 7.0,
        },
        "rho_stress": {
            "parameter": "rho_stress",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "reasoning": "No literature; weakly informative",
            # No reference_interval_days → falls back to dt=1
        },
        "beta_stress_mood": {
            "parameter": "beta_stress_mood",
            "distribution": "Normal",
            "params": {"mu": 0.3, "sigma": 0.15},
            "sources": [
                {
                    "title": "Stress-mood cross-lag study",
                    "snippet": "β = 0.3 at weekly interval",
                    "study_interval_days": 7.0,
                }
            ],
            "reasoning": "Weekly cross-lagged panel study",
            "reference_interval_days": 7.0,
        },
        "sigma_mood": {
            "parameter": "sigma_mood",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "sigma_stress": {
            "parameter": "sigma_stress",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "lambda_stress_cortisol_stress": {
            "parameter": "lambda_stress_cortisol_stress",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.8},
            "sources": [],
            "reasoning": "Weakly informative free loading",
        },
        "obs_sd_stress_self_report": {
            "parameter": "obs_sd_stress_self_report",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.5},
            "sources": [],
            "reasoning": "Weakly informative measurement error",
        },
        "obs_sd_stress_cortisol": {
            "parameter": "obs_sd_stress_cortisol",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.5},
            "sources": [],
            "reasoning": "Weakly informative measurement error",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: First-order DT→CT with reference_interval_days
# ═══════════════════════════════════════════════════════════════════════


class TestE2ESpecToDiscretization:
    """End-to-end: CausalSpec → SSMSpec → PriorRegistry → discretize → roundtrip."""

    def test_ssm_spec_structure_from_dag(self, two_construct_causal_spec, two_construct_model_spec):
        """SSMModelBuilder produces correct SSMSpec from DAG structure."""
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )

        # Dimensions
        assert spec.n_latent == 2  # mood, stress
        assert spec.n_manifest == 3  # mood_rating, stress_self_report, stress_cortisol
        assert spec.latent_names == ["mood", "stress"]

        # Drift mask: diagonal (AR) + stress→mood off-diagonal
        drift_component = spec.drift_spec.components[0]
        assert drift_component.drift_diag_mask[0]  # mood AR
        assert drift_component.drift_diag_mask[1]  # stress AR
        assert drift_component.drift_offdiag_mask[
            0, 1
        ]  # stress→mood coupling (effect=mood row, cause=stress col)
        assert not drift_component.drift_offdiag_mask[1, 0]  # no mood→stress edge

        # Lambda mask: stress_cortisol has free loading for stress
        assert spec.lambda_block.mask is not None
        # mood_rating loads on mood (fixed=1.0), stress_self_report loads on stress (fixed=1.0)
        # stress_cortisol loads on stress (free)
        manifest_names = spec.manifest_names
        assert manifest_names is not None
        stress_cortisol_idx = manifest_names.index("stress_cortisol")
        assert spec.latent_names is not None
        stress_latent_idx = spec.latent_names.index("stress")
        assert spec.lambda_block.mask[stress_cortisol_idx, stress_latent_idx]

    def test_causal_spec_owns_latent_identity(self):
        """Latent identity comes from causal_spec, not from AR parameter count."""
        causal_spec = _with_estimation_projection(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "mood",
                            "description": "Daily mood",
                            "role": "endogenous",
                            "is_outcome": True,
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "stress",
                            "description": "Daily stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "trait_vulnerability",
                            "description": "Stable vulnerability factor",
                            "role": "exogenous",
                            "temporal_status": "time_invariant",
                        },
                    ],
                    "edges": [
                        {
                            "cause": "stress",
                            "effect": "mood",
                            "description": "Stress impairs mood",
                            "lagged": True,
                        }
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "mood_rating",
                            "construct_name": "mood",
                            "how_to_measure": "Mood rating",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "stress_rating",
                            "construct_name": "stress",
                            "how_to_measure": "Stress rating",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "vulnerability_score",
                            "construct_name": "trait_vulnerability",
                            "how_to_measure": "Vulnerability questionnaire",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
            }
        )
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "vulnerability_score",
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
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.1}},
        }

        spec, _elags = _translate_spec_for_test(model_spec, causal_spec=causal_spec)
        ssm_priors, _idx = _compile_priors_for_test(
            priors,
            model_spec,
            ssm_spec=spec,
            causal_spec=causal_spec,
        )

        assert spec.latent_names == ["mood", "stress", "trait_vulnerability"]
        assert spec.n_latent == 3
        drift_component = spec.drift_spec.components[0]

        cint_component = spec.drift_spec.components[1]
        assert drift_component.time_invariant_mask is not None
        np.testing.assert_array_equal(drift_component.time_invariant_mask, [False, False, True])
        np.testing.assert_array_equal(drift_component.drift_diag_mask, [True, True, False])
        np.testing.assert_array_equal(cint_component.cint_mask, [False, False, False])
        np.testing.assert_array_equal(
            spec.diffusion_block.diffusion_chol_mask,
            [
                [True, False, False],
                [False, True, False],
                [False, False, False],
            ],
        )
        assert _base_decay_means(ssm_priors).shape == (2,)

    def test_time_invariant_states_drop_static_target_drift_and_diffusion_support(self):
        """Time-invariant states should not expose drift, diffusion, or cint support."""
        causal_spec = _with_estimation_projection(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "mood",
                            "description": "Daily mood",
                            "role": "endogenous",
                            "is_outcome": True,
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "stress",
                            "description": "Daily stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "trait_vulnerability",
                            "description": "Stable vulnerability factor",
                            "role": "exogenous",
                            "temporal_status": "time_invariant",
                        },
                    ],
                    "edges": [
                        {
                            "cause": "stress",
                            "effect": "mood",
                            "description": "Stress impairs mood",
                            "lagged": True,
                        },
                        {
                            "cause": "trait_vulnerability",
                            "effect": "mood",
                            "description": "Stable vulnerability shifts mood dynamics",
                            "lagged": False,
                        },
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "mood_rating",
                            "construct_name": "mood",
                            "how_to_measure": "Mood rating",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "stress_rating",
                            "construct_name": "stress",
                            "how_to_measure": "Stress rating",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "vulnerability_score",
                            "construct_name": "trait_vulnerability",
                            "how_to_measure": "Vulnerability questionnaire",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
            }
        )
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "vulnerability_score",
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
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "beta_trait_vulnerability_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "cor_mood_stress",
                    "role": "correlation",
                    "constraint": "correlation",
                    "description": "",
                },
            ],
        }

        spec, _elags = _translate_spec_for_test(model_spec, causal_spec=causal_spec)

        assert spec.latent_names == ["mood", "stress", "trait_vulnerability"]
        drift_component = spec.drift_spec.components[0]

        cint_component = spec.drift_spec.components[1]
        np.testing.assert_array_equal(drift_component.drift_diag_mask, [True, True, False])
        assert drift_component.drift_offdiag_mask[0, 1]
        assert drift_component.drift_offdiag_mask[0, 2]
        assert not drift_component.drift_offdiag_mask[2, 0]
        assert not drift_component.drift_offdiag_mask[2, 1]
        np.testing.assert_array_equal(cint_component.cint_mask, [False, False, False])
        np.testing.assert_array_equal(
            spec.diffusion_block.diffusion_chol_mask,
            [
                [True, False, False],
                [True, True, False],
                [False, False, False],
            ],
        )

    def test_builder_rejects_parameter_names_not_grounded_in_causal_spec(
        self, two_construct_causal_spec
    ):
        """Bad parameter names should fail instead of compiling a different model."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_self_report",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_cortisol",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_affect",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_affect": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }

        spec, _elags = _translate_spec_for_test(
            model_spec,
            causal_spec=two_construct_causal_spec,
        )

        with pytest.raises(ValueError, match="does not correspond to a free dynamics decay"):
            _compile_priors_for_test(
                priors,
                model_spec,
                ssm_spec=spec,
                causal_spec=two_construct_causal_spec,
            )

    def test_compiled_artifact_roundtrips_grounded_structure(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Compiled artifacts preserve the grounded latent and measurement layout."""
        from nof1_causal_lab.models.ssm.compile.artifact import (
            build_compiled_ssm_builder,
            compile_ssm_artifact,
        )
        from nof1_causal_lab.utils.data import pivot_to_wide

        compiled = compile_ssm_artifact(
            two_construct_model_spec,
            weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )

        assert compiled["schema_version"] == 1
        assert compiled["spec"]["latent_names"] == ["mood", "stress"]
        assert compiled["spec"]["manifest_names"] == [
            "mood_rating",
            "stress_self_report",
            "stress_cortisol",
        ]
        parameter_bindings = [
            {
                "parameter": binding["parameter"],
                "site_name": binding["site_name"],
                "flat_index": binding["flat_index"],
            }
            for binding in compiled["parameter_bindings"]
        ]
        assert parameter_bindings == [
            {"parameter": "beta_stress_mood", "site_name": "vf_0_offdiag", "flat_index": 0},
            {
                "parameter": "lambda_stress_cortisol_stress",
                "site_name": "lambda_free",
                "flat_index": 0,
            },
            {
                "parameter": "obs_sd_stress_cortisol",
                "site_name": "manifest_var_diag_free",
                "flat_index": 1,
            },
            {
                "parameter": "obs_sd_stress_self_report",
                "site_name": "manifest_var_diag_free",
                "flat_index": 0,
            },
            {"parameter": "rho_mood", "site_name": "vf_0_base_decay", "flat_index": 0},
            {"parameter": "rho_stress", "site_name": "vf_0_base_decay", "flat_index": 1},
            {"parameter": "sigma_mood", "site_name": "diffusion_diag_free", "flat_index": 0},
            {"parameter": "sigma_stress", "site_name": "diffusion_diag_free", "flat_index": 1},
        ]

        data_for_model = pl.DataFrame(
            {
                "indicator": [
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                ],
                "value": [6.0, 4.0, 10.0, 7.0, 5.0, 11.0],
                "anchor_time": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                ],
            }
        )

        builder = build_compiled_ssm_builder(compiled, pivot_to_wide(data_for_model))
        spec = builder.spec
        assert spec.latent_names == ["mood", "stress"]
        drift_component = spec.drift_spec.components[0]
        assert drift_component.drift_offdiag_mask[0, 1]
        assert not drift_component.drift_offdiag_mask[1, 0]
        assert spec.lambda_block.mask is not None
        assert spec.lambda_block.mask[2, 1]
        runtime = builder.model.get_prior_runtime_bundle()
        assert runtime.prior_state["vf_0_base_decay"]["concentration"].shape == (2,)
        assert builder.model.parameter_bindings == compiled["parameter_bindings"]

    def test_residual_sd_priors_are_construct_specific(
        self, two_construct_causal_spec, two_construct_model_spec
    ):
        """Construct-specific sigma priors compile to per-latent diffusion scales."""
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 3.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
            "sigma_mood": {"distribution": "HalfNormal", "params": {"sigma": 0.1}},
            "sigma_stress": {"distribution": "HalfNormal", "params": {"sigma": 0.9}},
            "lambda_stress_cortisol_stress": {
                "distribution": "Normal",
                "params": {"mu": 0.8, "sigma": 0.2},
            },
        }

        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        assert _prior_params(ssm_priors, "diffusion_diag_free") == {"sigma": [0.1, 0.9]}

    def test_dt_to_ct_uses_reference_interval_days(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Priors with reference_interval_days use that dt.

        rho_mood has reference_interval_days=7 → dt=7
        rho_stress has no reference_interval_days → falls back to dt=1
        beta_stress_mood has reference_interval_days=7 → dt=7
        """
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        # --- rho_mood: Beta(3,2) → E=0.6, reference_interval_days=7 ---
        # drift_base_decay[0] = -ln(0.6) / 7 ≈ 0.073
        mu_ar_mood = 3.0 / 5.0  # E[Beta(3,2)] = 0.6
        expected_drift_mood = -math.log(mu_ar_mood) / 7.0
        mu_drift = _base_decay_means(ssm_priors)
        mu_mood = mu_drift[0]
        assert abs(mu_mood - expected_drift_mood) < 0.01, (
            f"mood drift: got {mu_mood}, expected {expected_drift_mood} "
            f"(using reference_interval_days=7)"
        )

        # --- rho_stress: Beta(2,2) → E=0.5, no reference_interval_days → daily dt=1 ---
        # drift_base_decay[1] = -ln(0.5) / 1.0 ≈ 0.693
        mu_ar_stress = 0.5
        expected_drift_stress = -math.log(mu_ar_stress) / 1.0
        mu_stress = mu_drift[1]
        assert abs(mu_stress - expected_drift_stress) < 0.01, (
            f"stress drift: got {mu_stress}, expected {expected_drift_stress} "
            f"(fallback to daily dt=1)"
        )

        # --- beta_stress_mood: Normal(0.3, 0.15), reference_interval_days=7 ---
        # drift_offdiag[0] = 0.3 / 7 ≈ 0.043
        expected_offdiag = 0.3 / 7.0
        mu_offdiag = _prior_params(ssm_priors, "vf_0_offdiag")["mu"]
        mu_offdiag_val = mu_offdiag[0] if isinstance(mu_offdiag, list) else mu_offdiag
        assert abs(mu_offdiag_val - expected_offdiag) < 0.01, (
            f"stress→mood drift: got {mu_offdiag_val}, expected {expected_offdiag} "
            f"(using reference_interval_days=7)"
        )

    def test_ct_drift_is_stable(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """The CT drift matrix from converted priors has all eigenvalues with Re < 0."""
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        drift = np.asarray(_mean_drift_from_priors(spec, ssm_priors))

        # All eigenvalues must have negative real parts (stability)
        eigenvalues = np.linalg.eigvals(drift)
        max_real = np.max(np.real(eigenvalues))
        assert max_real < 0, f"Drift matrix is unstable: max Re(eigenvalue) = {max_real}"

    def test_first_order_roundtrip_ar(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Resolved AR persistence includes base decay, incoming mass, and margin."""
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        drift = _mean_drift_from_priors(spec, ssm_priors)

        # Discretize at dt=7 (weekly). The resolved diagonal damping is stronger
        # than the elicited baseline persistence when incoming coupling is present.
        dt_weekly = 7.0
        F_weekly = jla.expm(drift * dt_weekly)

        base_decay = _base_decay_means(ssm_priors)
        offdiag = _prior_vector(_prior_params(ssm_priors, "vf_0_offdiag")["mu"])
        drift_component = spec.drift_spec.components[0]
        baseline_ar_mood = 3.0 / 5.0  # Beta(3,2) mean = 0.6
        expected_resolved_mood = math.exp(
            -(base_decay[0] + abs(offdiag[0]) + drift_component.stability_margin) * dt_weekly
        )
        recovered_ar_mood = float(F_weekly[0, 0])
        assert recovered_ar_mood < baseline_ar_mood
        assert abs(recovered_ar_mood - expected_resolved_mood) < 0.05, (
            f"Weekly resolved mood AR: got {recovered_ar_mood:.4f}, "
            f"expected ≈{expected_resolved_mood:.4f}"
        )

        # stress: dt=1 for this prior, with only the stability margin added.
        recovered_ar_stress = float(F_weekly[1, 1])
        assert recovered_ar_stress < 0.05, (
            f"Stress AR at weekly interval should be very low (daily-derived rate), "
            f"got {recovered_ar_stress:.4f}"
        )

        # Discretize at dt=1 (daily) for stress resolved persistence.
        F_daily = jla.expm(drift * 1.0)
        expected_daily_stress = math.exp(-(base_decay[1] + drift_component.stability_margin))
        recovered_daily_stress = float(F_daily[1, 1])
        assert abs(recovered_daily_stress - expected_daily_stress) < 0.05, (
            f"Daily roundtrip stress AR: got {recovered_daily_stress:.4f}, "
            f"expected ≈{expected_daily_stress:.4f}"
        )

    def test_first_order_roundtrip_cross_lag(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """DT→CT→DT roundtrip for cross-lagged coefficient.

        beta_stress_mood = 0.3 from weekly study
        → CT rate = 0.3/7 → discretize at dt=7 → F[mood,stress] ≈ 0.3
        (first-order approximation; exact requires matrix exponential)
        """
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        # Build drift matrix
        drift = _mean_drift_from_priors(spec, ssm_priors)

        # Discretize at weekly interval
        dt_weekly = 7.0
        F_weekly = jla.expm(drift * dt_weekly)

        # NOTE: F[0,1] ≠ β_DT because the matrix exponential mixes terms:
        #   F[0,1] = A[0,1] * (exp(A[0,0]*dt) - exp(A[1,1]*dt)) / (A[1,1] - A[0,0])
        # For different diagonal entries, this is NOT simply A[0,1]*dt.
        # The exact DT→CT→DT roundtrip requires the matrix logarithm (Phase 2).
        #
        # What we CAN verify at first order:
        # 1. The CT rate was computed correctly (tested in test_dt_to_ct_uses_reference_interval_days)
        # 2. The coupling direction is preserved (F[0,1] > 0 since A[0,1] > 0)
        # 3. The exact logm(F)/dt recovers the original A (tested in Phase 2 tests)
        recovered_coupling = float(F_weekly[0, 1])
        assert recovered_coupling > 0, (
            f"Coupling direction should be positive (stress→mood), got {recovered_coupling:.4f}"
        )
        # Verify via exact logm roundtrip
        from scipy.linalg import logm

        A_recovered = logm(np.array(F_weekly)).real / dt_weekly
        ct_rate = float(drift[0, 1])  # the CT rate we set
        assert abs(A_recovered[0, 1] - ct_rate) < 1e-6, (
            f"Exact logm roundtrip: got {A_recovered[0, 1]:.6f}, expected {ct_rate:.6f}"
        )

    def test_discretize_produces_valid_system(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """discretize_system produces valid F, Q, c from converted priors."""
        spec, _elags = _translate_spec_for_test(
            two_construct_model_spec,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model_spec,
            ssm_spec=spec,
            causal_spec=two_construct_causal_spec,
        )

        # Build drift and diffusion at prior means
        n = spec.n_latent
        drift = _mean_drift_from_priors(spec, ssm_priors)

        # Simple diagonal diffusion
        diff_sd = _prior_params(ssm_priors, "diffusion_diag_free").get("sigma", 1.0)
        diff_sd_arr = jnp.asarray(diff_sd, dtype=jnp.float32)
        diffusion_cov = jnp.diag(diff_sd_arr**2)

        # CINT (zeros)
        cint = jnp.zeros(n)

        # Discretize at dt=1 (daily)
        F, Q, c = discretize_system(drift, diffusion_cov, cint, dt=1.0)

        # F should be a valid transition matrix (all eigenvalues < 1 in abs)
        eigs_F = jnp.linalg.eigvals(F)
        assert jnp.all(jnp.abs(eigs_F) < 1.0 + 1e-6), (
            f"F has eigenvalues outside unit circle: {eigs_F}"
        )

        # Q should be symmetric positive semi-definite
        assert jnp.allclose(Q, Q.T, atol=1e-6), "Q is not symmetric"
        eigs_Q = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigs_Q >= -1e-6), f"Q has negative eigenvalues: {eigs_Q}"

        # No NaN/Inf
        assert jnp.all(jnp.isfinite(F)), "F contains NaN/Inf"
        assert jnp.all(jnp.isfinite(Q)), "Q contains NaN/Inf"
        assert c is not None, "c should not be None when cint is provided"
        assert jnp.all(jnp.isfinite(c)), "c contains NaN/Inf"

    def test_prior_predictive_produces_finite_samples(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Prior predictive sampling produces finite, bounded outputs."""
        import polars as pl

        from nof1_causal_lab.models.ssm.inference import prior_predictive

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )

        # Build model with minimal mock data
        n_time = 30
        rng = np.random.default_rng(0)
        mock_data = pl.DataFrame(
            {
                "mood_rating": rng.normal(5, 1.5, n_time),
                "stress_self_report": rng.normal(5, 1.5, n_time),
                "stress_cortisol": rng.normal(10, 2, n_time),
                "time": np.arange(n_time, dtype=float),
            }
        )
        model = builder.build_model(mock_data)

        # Sample from prior predictive
        times = jnp.arange(n_time, dtype=jnp.float32)
        samples = prior_predictive(model, times, num_samples=20, seed=42)

        # Check key deterministic sites exist and are finite
        assert "vf_0_drift" in samples, "Missing 'vf_0_drift' in prior predictive samples"
        drift_samples = samples["vf_0_drift"]
        assert jnp.all(jnp.isfinite(drift_samples)), "vf_0_drift samples contain NaN/Inf"

        if "diffusion" in samples:
            diff_samples = samples["diffusion"]
            assert jnp.all(jnp.isfinite(diff_samples)), "diffusion samples contain NaN/Inf"

        # Drift diag should be negative (stability)
        if drift_samples.ndim == 3:  # (n_samples, n_latent, n_latent)
            for i in range(drift_samples.shape[0]):
                diag = jnp.diag(drift_samples[i])
                assert jnp.all(diag < 0), f"Sample {i} has non-negative drift diagonal: {diag}"

    def test_different_intervals_produce_different_rates(self, two_construct_model_spec):
        """Same DT beta at different study intervals → different CT rates.

        beta=0.3 from weekly (dt=7) → CT rate ≈ 0.043
        beta=0.3 from daily  (dt=1) → CT rate ≈ 0.300
        This is the Kuiper & Ryan (2018) sign-reversal effect in action.
        """
        causal_spec = _with_estimation_projection(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "mood",
                            "description": "Mood",
                            "role": "endogenous",
                            "is_outcome": True,
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "stress",
                            "description": "Stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [
                        {
                            "cause": "stress",
                            "effect": "mood",
                            "description": "test",
                            "lagged": True,
                        },
                    ],
                },
                "measurement": {"model_clock": "1d", "indicators": []},
            }
        )

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
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }

        # Weekly study priors
        priors_weekly = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
            },
        }
        # Daily study priors (same beta value, different interval)
        priors_daily = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 1.0,
            },
        }

        drift_offdiag_mask = np.array([[False, True], [False, False]])
        ssm_spec = _block_spec_with_drift_offdiag(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_offdiag_mask=drift_offdiag_mask,
        )

        ssm_priors_w, _idx = _compile_priors_for_test(
            priors_weekly,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        ssm_priors_d, _idx = _compile_priors_for_test(
            priors_daily,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Weekly: mixed intervals (beta=7d, rho=1d) → first-order: 0.3 / 7 ≈ 0.043
        mu_w = _prior_params(ssm_priors_w, "vf_0_offdiag")["mu"]
        mu_w_val = mu_w[0] if isinstance(mu_w, list) else mu_w
        assert abs(mu_w_val - 0.3 / 7.0) < 0.01

        # Daily: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu_d = _prior_params(ssm_priors_d, "vf_0_offdiag")["mu"]
        mu_d_val = mu_d[0] if isinstance(mu_d, list) else mu_d
        expected_daily = 0.3
        assert abs(mu_d_val - expected_daily) < 0.05, (
            f"Daily case uses beta/dt scaling: got {mu_d_val}, expected {expected_daily}"
        )

        # Rates should differ significantly because beta/dt depends on the interval.
        assert mu_d_val > mu_w_val, "Daily rate should be larger than weekly rate"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Exact Matrix Logarithm DT→CT
# ═══════════════════════════════════════════════════════════════════════


class TestExactMatrixLogConversion:
    """Phase 2: Exact A = logm(Φ)/dt conversion and embeddability checks.

    These tests validate the mathematical properties independently of
    the pipeline, operating directly on transition matrices.
    """

    def test_scalar_logm_matches_first_order(self):
        """For a 1D system, logm(Phi)/dt gives the same drift magnitude.

        logm([[rho]]) = [[ln(rho)]] (negative for rho < 1).
        Our pipeline stores baseline decay as a positive magnitude. So:
          base_decay_mu = -ln(rho)/dt  (positive)
          actual_drift  = -base_decay_mu = ln(rho)/dt  (negative without coupling)
          logm(Phi)/dt  = ln(rho)/dt  (negative, matches actual_drift)
        """
        rho = 0.7
        dt = 1.0
        Phi = np.array([[rho]])

        # Pipeline convention: positive magnitude (gets negated by model)
        drift_mag = -math.log(rho) / dt  # positive

        # Exact (logm): gives the actual (negative) drift
        from scipy.linalg import logm

        A_exact = logm(Phi).real / dt

        # logm gives ln(rho)/dt which equals -drift_mag
        assert abs(A_exact[0, 0] - (-drift_mag)) < 1e-10

    def test_exact_roundtrip_2d_system(self):
        """Exact logm roundtrip: A → Φ = exp(A*dt) → logm(Φ)/dt → A.

        Build a known 2D CT drift, discretize, then recover via logm.
        """
        from scipy.linalg import expm, logm

        # Known stable drift
        A = np.array(
            [
                [-0.5, 0.1],
                [-0.2, -0.8],
            ]
        )
        dt = 1.0

        # Forward: CT → DT
        Phi = expm(A * dt)

        # Backward: DT → CT (exact)
        A_recovered = logm(Phi).real / dt

        np.testing.assert_allclose(A_recovered, A, atol=1e-10)

    def test_first_order_error_grows_with_dt(self):
        """First-order β/dt approximation error grows with observation interval.

        For a triangular system, the relative error of the first-order
        off-diagonal recovery depends on the eigenvalue spread and dt,
        NOT on the coupling magnitude itself.

        Longer observation intervals → more eigenvalue mixing → larger error.
        """
        from scipy.linalg import expm, logm

        # Fixed system
        A = np.array([[-0.3, 0.15], [-0.1, -0.5]])

        # Short interval (dt=0.5): first-order should be decent
        dt_short = 0.5
        Phi_short = expm(A * dt_short)
        A_first_short = np.zeros_like(A)
        A_first_short[0, 0] = -math.log(abs(Phi_short[0, 0])) / dt_short
        A_first_short[1, 1] = -math.log(abs(Phi_short[1, 1])) / dt_short
        A_first_short[0, 1] = Phi_short[0, 1] / dt_short
        A_first_short[1, 0] = Phi_short[1, 0] / dt_short
        error_short = np.linalg.norm(A_first_short - A) / np.linalg.norm(A)

        # Long interval (dt=7): first-order should be much worse
        dt_long = 7.0
        Phi_long = expm(A * dt_long)
        A_first_long = np.zeros_like(A)
        A_first_long[0, 0] = -math.log(abs(Phi_long[0, 0])) / dt_long
        A_first_long[1, 1] = -math.log(abs(Phi_long[1, 1])) / dt_long
        A_first_long[0, 1] = Phi_long[0, 1] / dt_long
        A_first_long[1, 0] = Phi_long[1, 0] / dt_long
        error_long = np.linalg.norm(A_first_long - A) / np.linalg.norm(A)

        # Error should be larger for longer intervals
        assert error_long > error_short, (
            f"First-order error should grow with dt: "
            f"short(dt={dt_short})={error_short:.4f}, long(dt={dt_long})={error_long:.4f}"
        )

        # Exact logm should have near-zero error for both
        A_exact_short = logm(Phi_short).real / dt_short
        A_exact_long = logm(Phi_long).real / dt_long
        np.testing.assert_allclose(A_exact_short, A, atol=1e-8)
        np.testing.assert_allclose(A_exact_long, A, atol=1e-8)

    def test_embeddability_positive_eigenvalues(self):
        """A DT transition matrix Φ is embeddable iff all eigenvalues are positive real.

        Ref: Higham (2008), Ch. 11 — principal matrix logarithm exists when
        Φ has no eigenvalues on the closed negative real axis.
        """
        from scipy.linalg import logm

        # Embeddable: stable 2D system with positive eigenvalues
        Phi_good = np.array(
            [
                [0.8, 0.1],
                [0.05, 0.7],
            ]
        )
        eigs = np.linalg.eigvals(Phi_good)
        assert np.all(np.real(eigs) > 0), "Expected positive real eigenvalues"

        A_good = logm(Phi_good).real
        # Recovered A should be stable (negative diagonal)
        assert np.all(np.diag(A_good) < 0), (
            f"Recovered drift should be stable, got diagonal: {np.diag(A_good)}"
        )

        # Non-embeddable: negative eigenvalue
        Phi_bad = np.array(
            [
                [-0.5, 0.0],
                [0.0, 0.8],
            ]
        )
        eigs_bad = np.linalg.eigvals(Phi_bad)
        has_negative = np.any(np.real(eigs_bad) <= 0)
        assert has_negative, "This matrix should have a non-positive eigenvalue"

        # logm of non-embeddable matrix produces complex result
        A_bad = logm(Phi_bad)
        has_complex = np.any(np.abs(np.imag(A_bad)) > 1e-10)
        assert has_complex, "logm of non-embeddable Φ should have imaginary components"

    def test_exact_logm_recovers_cross_lag_better_than_first_order(self):
        """For a realistic 2-construct system, logm recovers cross-lag
        more accurately than the first-order β/dt approximation.

        This is the core Phase 2 improvement.
        """
        from scipy.linalg import expm, logm

        # True CT system: stress → mood with moderate coupling
        A_true = np.array(
            [
                [-0.3, 0.15],  # mood: AR drift -0.3, stress coupling 0.15
                [0.0, -0.5],  # stress: AR drift -0.5, no reverse coupling
            ]
        )
        dt = 7.0  # weekly observation interval

        # Generate "observed" DT transition matrix
        Phi = expm(A_true * dt)

        # First-order recovery
        A_first = np.zeros_like(A_true)
        A_first[0, 0] = -math.log(Phi[0, 0]) / dt
        A_first[1, 1] = -math.log(Phi[1, 1]) / dt
        A_first[0, 1] = Phi[0, 1] / dt  # β/dt approximation
        error_first = np.linalg.norm(A_first - A_true) / np.linalg.norm(A_true)

        # Exact logm recovery
        A_exact = logm(Phi).real / dt
        error_exact = np.linalg.norm(A_exact - A_true) / np.linalg.norm(A_true)

        # Exact should be much better
        assert error_exact < error_first, (
            f"logm error ({error_exact:.6f}) should be less than "
            f"first-order error ({error_first:.6f})"
        )
        # logm should be essentially perfect
        assert error_exact < 1e-8, f"logm error unexpectedly large: {error_exact}"

    def test_discretize_at_multiple_intervals(self):
        """Discretizing at different intervals from the same CT drift
        produces different but consistent DT parameters.

        Key property: F(dt1) * F(dt2) == F(dt1 + dt2) (semi-group property).
        """
        # Stable 2D drift
        drift = jnp.array(
            [
                [-0.3, 0.05],
                [-0.1, -0.5],
            ]
        )
        diffusion_cov = jnp.eye(2) * 0.1

        # Discretize at dt=1 and dt=2
        F1, _Q1, _ = discretize_system(drift, diffusion_cov, None, dt=1.0)
        F2, _Q2, _ = discretize_system(drift, diffusion_cov, None, dt=2.0)

        # Semi-group property: F(2) == F(1) @ F(1)
        F1_squared = F1 @ F1
        np.testing.assert_allclose(
            np.array(F2),
            np.array(F1_squared),
            atol=1e-5,
            err_msg="Semi-group property F(2dt) = F(dt)^2 violated",
        )

        # F(1) should have larger eigenvalues than F(2) — less decay at shorter interval
        eigs_1 = jnp.abs(jnp.linalg.eigvals(F1))
        eigs_2 = jnp.abs(jnp.linalg.eigvals(F2))
        assert jnp.all(eigs_1 > eigs_2), (
            f"Shorter interval should have less decay: |eigs(F1)|={eigs_1}, |eigs(F2)|={eigs_2}"
        )

    def test_builder_keeps_elementwise_priors_when_intervals_match(self, two_construct_causal_spec):
        """SSMModelBuilder keeps factorized DT→CT priors even when dt values match."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_self_report",
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
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }

        # All parameters at dt=7 (weekly)
        priors = {
            "rho_mood": {
                "distribution": "Beta",
                "params": {"alpha": 3.0, "beta": 2.0},
                "reference_interval_days": 7.0,
            },
            "rho_stress": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "reference_interval_days": 7.0,
            },
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
            },
        }

        drift_offdiag_mask = np.array([[False, True], [False, False]])
        ssm_spec = _block_spec_with_drift_offdiag(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_offdiag_mask=drift_offdiag_mask,
        )

        ssm_priors, _idx = _compile_priors_for_test(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=two_construct_causal_spec,
        )

        drift_base_decay = _base_decay_means(ssm_priors)
        drift_offdiag = _prior_params(ssm_priors, "vf_0_offdiag")["mu"]

        assert abs(drift_base_decay[0] - (-math.log(0.6) / 7.0)) < 0.01
        assert abs(drift_base_decay[1] - (-math.log(0.5) / 7.0)) < 0.01
        assert abs(drift_offdiag[0] - (0.3 / 7.0)) < 0.01

    def test_edge_lag_days_populated(self, two_construct_causal_spec):
        """Builder stores edge lag metadata from causal spec during support building."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_causal_spec,
        )

        _dm, _input_mask, _lm, _lmask, edge_lag_days = build_structural_support_from_causal_spec(
            ["mood", "stress"],
            ["mood_rating", "stress_self_report"],
            2,
            2,
            causal_spec=two_construct_causal_spec,
        )

        # stress -> mood edge, both daily, lagged=True: lag = 24h = 1.0 day
        assert len(edge_lag_days) == 1
        # effect_idx=0 (mood), cause_idx=1 (stress)
        assert (0, 1) in edge_lag_days
        assert abs(edge_lag_days[(0, 1)] - 1.0) < 0.01

    def test_drift_lag_consistency_warns(self, two_construct_causal_spec, caplog):
        """Builder warns when CT drift implies timescale far from edge lag."""
        import logging

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_rating",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_self_report",
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
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        # Very large beta → CT rate implies very fast coupling (short timescale)
        # but edge lag is 1 day → should warn about mismatch
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 6.0, "sigma": 1.0},
            },
        }
        drift_offdiag_mask = np.array([[False, True], [False, False]])
        ssm_spec = _block_spec_with_drift_offdiag(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_offdiag_mask=drift_offdiag_mask,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_causal_spec,
        )

        _dm, _input_mask, _lm, _lmask, edge_lag_days = build_structural_support_from_causal_spec(
            ["mood", "stress"],
            ["mood_rating", "stress_self_report"],
            2,
            2,
            causal_spec=two_construct_causal_spec,
        )
        with caplog.at_level(logging.WARNING, logger="nof1_causal_lab.models.ssm.compile.inputs"):
            _compile_priors_for_test(
                priors,
                model_spec,
                ssm_spec=ssm_spec,
                causal_spec=two_construct_causal_spec,
                edge_lag_days=edge_lag_days,
            )

        # Large beta_CT → implied timescale << 1 day, edge lag = 1 day → warning
        lag_warnings = [r for r in caplog.records if "mismatch" in r.message.lower()]
        assert len(lag_warnings) >= 1, (
            f"Expected lag mismatch warning, got: {[r.message for r in caplog.records]}"
        )
