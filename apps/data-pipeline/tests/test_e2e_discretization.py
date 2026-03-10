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

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.models.ssm import SSMSpec, discretize_system
from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

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
    return {
        "latent": {
            "constructs": [
                {
                    "name": "mood",
                    "description": "Daily mood state",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "stress",
                    "description": "Daily stress level",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
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
                "constraint": "correlation",
                "description": "AR(1) for mood",
                "search_context": "mood autocorrelation",
            },
            {
                "name": "rho_stress",
                "role": "ar_coefficient",
                "constraint": "correlation",
                "description": "AR(1) for stress",
                "search_context": "stress autocorrelation",
            },
            {
                "name": "beta_stress_mood",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Cross-lagged effect of stress on mood",
                "search_context": "stress mood cross-lagged",
            },
            {
                "name": "sigma_mood_rating",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for mood_rating",
                "search_context": "",
            },
            {
                "name": "sigma_stress_self_report",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for stress_self_report",
                "search_context": "",
            },
            {
                "name": "sigma_stress_cortisol",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for stress_cortisol",
                "search_context": "",
            },
            {
                "name": "lambda_stress_cortisol_stress",
                "role": "loading",
                "constraint": "positive",
                "description": "Loading: stress → stress_cortisol",
                "search_context": "",
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
            # No reference_interval_days → falls back to temporal_scale (daily → dt=1)
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
        "sigma_mood_rating": {
            "parameter": "sigma_mood_rating",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "sigma_stress_self_report": {
            "parameter": "sigma_stress_self_report",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "sigma_stress_cortisol": {
            "parameter": "sigma_stress_cortisol",
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
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: First-order DT→CT with reference_interval_days
# ═══════════════════════════════════════════════════════════════════════


class TestE2ESpecToDiscretization:
    """End-to-end: CausalSpec → SSMSpec → SSMPriors → discretize → roundtrip."""

    def test_ssm_spec_structure_from_dag(self, two_construct_causal_spec, two_construct_model_spec):
        """SSMModelBuilder produces correct SSMSpec from DAG structure."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors={},
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)

        # Dimensions
        assert spec.n_latent == 2  # mood, stress
        assert spec.n_manifest == 3  # mood_rating, stress_self_report, stress_cortisol
        assert spec.latent_names == ["mood", "stress"]

        # Drift mask: diagonal (AR) + stress→mood off-diagonal
        assert spec.drift_mask is not None
        assert spec.drift_mask[0, 0]  # mood AR
        assert spec.drift_mask[1, 1]  # stress AR
        assert spec.drift_mask[0, 1]  # stress→mood coupling (effect=mood row, cause=stress col)
        assert not spec.drift_mask[1, 0]  # no mood→stress edge

        # Lambda mask: stress_cortisol has free loading for stress
        assert spec.lambda_mask is not None
        # mood_rating loads on mood (fixed=1.0), stress_self_report loads on stress (fixed=1.0)
        # stress_cortisol loads on stress (free)
        manifest_names = spec.manifest_names
        assert manifest_names is not None
        stress_cortisol_idx = manifest_names.index("stress_cortisol")
        assert spec.latent_names is not None
        stress_latent_idx = spec.latent_names.index("stress")
        assert spec.lambda_mask[stress_cortisol_idx, stress_latent_idx]

    def test_causal_spec_owns_latent_identity(self):
        """Latent identity comes from causal_spec, not from AR parameter count."""
        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "mood",
                        "description": "Daily mood",
                        "role": "endogenous",
                        "is_outcome": True,
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
                    },
                    {
                        "name": "stress",
                        "description": "Daily stress",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
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
                ]
            },
        }
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
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
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
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.1}},
        }

        builder = SSMModelBuilder(model_spec=model_spec, priors=priors, causal_spec=causal_spec)
        spec = builder._convert_spec_to_ssm(model_spec)
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=spec)

        assert spec.latent_names == ["mood", "stress", "trait_vulnerability"]
        assert spec.n_latent == 3
        assert spec.time_invariant_mask is not None
        np.testing.assert_array_equal(spec.time_invariant_mask, [False, False, True])
        assert isinstance(ssm_priors.drift_diag["mu"], list)
        assert len(ssm_priors.drift_diag["mu"]) == 3

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
            ],
            "parameters": [
                {
                    "name": "rho_affect",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
            ],
        }
        priors = {
            "rho_affect": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }

        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors=priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(model_spec)

        with pytest.raises(ValueError, match="AR parameter does not reference a construct"):
            builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=spec)

    def test_compiled_artifact_roundtrips_grounded_structure(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Compiled artifacts preserve the grounded latent and measurement layout."""
        from causal_ssm_agent.models.ssm_compiler import (
            build_compiled_ssm_builder,
            compile_ssm_artifact,
        )

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

        raw_data = pl.DataFrame(
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
                "timestamp": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                ],
            }
        )

        builder = build_compiled_ssm_builder(compiled, raw_data)
        assert builder._spec is not None
        assert builder._spec.latent_names == ["mood", "stress"]
        assert builder._spec.drift_mask is not None
        assert builder._spec.drift_mask[0, 1]
        assert not builder._spec.drift_mask[1, 0]
        assert builder._spec.lambda_mask is not None
        assert builder._spec.lambda_mask[2, 1]
        assert builder._model is not None
        assert isinstance(builder._model.priors.drift_diag["mu"], list)
        assert len(builder._model.priors.drift_diag["mu"]) == 2

    def test_dt_to_ct_uses_reference_interval_days(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """Priors with reference_interval_days use that dt, not temporal_scale.

        rho_mood has reference_interval_days=7 → dt=7
        rho_stress has no reference_interval_days → falls back to temporal_scale=daily → dt=1
        beta_stress_mood has reference_interval_days=7 → dt=7
        """
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)
        ssm_priors = builder._convert_priors_to_ssm(
            weekly_study_priors, two_construct_model_spec, ssm_spec=spec
        )

        # --- rho_mood: Beta(3,2) → E=0.6, reference_interval_days=7 ---
        # drift_diag[0] = -ln(0.6) / 7 ≈ 0.073
        mu_ar_mood = 3.0 / 5.0  # E[Beta(3,2)] = 0.6
        expected_drift_mood = -math.log(mu_ar_mood) / 7.0
        mu_drift = ssm_priors.drift_diag["mu"]
        mu_mood = mu_drift[0] if isinstance(mu_drift, list) else mu_drift
        assert abs(mu_mood - expected_drift_mood) < 0.01, (
            f"mood drift: got {mu_mood}, expected {expected_drift_mood} "
            f"(using reference_interval_days=7)"
        )

        # --- rho_stress: Beta(2,2) → E=0.5, no reference_interval_days → daily dt=1 ---
        # drift_diag[1] = -ln(0.5) / 1.0 ≈ 0.693
        mu_ar_stress = 0.5
        expected_drift_stress = -math.log(mu_ar_stress) / 1.0
        mu_stress = mu_drift[1] if isinstance(mu_drift, list) else mu_drift
        assert abs(mu_stress - expected_drift_stress) < 0.01, (
            f"stress drift: got {mu_stress}, expected {expected_drift_stress} "
            f"(fallback to daily dt=1)"
        )

        # --- beta_stress_mood: Normal(0.3, 0.15), reference_interval_days=7 ---
        # drift_offdiag[0] = 0.3 / 7 ≈ 0.043
        expected_offdiag = 0.3 / 7.0
        mu_offdiag = ssm_priors.drift_offdiag["mu"]
        mu_offdiag_val = mu_offdiag[0] if isinstance(mu_offdiag, list) else mu_offdiag
        assert abs(mu_offdiag_val - expected_offdiag) < 0.01, (
            f"stress→mood drift: got {mu_offdiag_val}, expected {expected_offdiag} "
            f"(using reference_interval_days=7)"
        )

    def test_ct_drift_is_stable(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """The CT drift matrix from converted priors has all eigenvalues with Re < 0."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)
        ssm_priors = builder._convert_priors_to_ssm(
            weekly_study_priors, two_construct_model_spec, ssm_spec=spec
        )

        # Build the drift matrix from priors (using mu values)
        n = spec.n_latent
        drift = np.zeros((n, n))

        # Diagonal (negative for stability)
        mu_diag = ssm_priors.drift_diag["mu"]
        if isinstance(mu_diag, list):
            for i, val in enumerate(mu_diag):
                drift[i, i] = -abs(val)  # model enforces negative diagonal
        else:
            np.fill_diagonal(drift, -abs(mu_diag))

        # Off-diagonal from drift mask
        if spec.drift_mask is not None:
            mu_offdiag = ssm_priors.drift_offdiag["mu"]
            offdiag_vals = mu_offdiag if isinstance(mu_offdiag, list) else [mu_offdiag]
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        drift[i, j] = offdiag_vals[idx]
                        idx += 1

        # All eigenvalues must have negative real parts (stability)
        eigenvalues = np.linalg.eigvals(drift)
        max_real = np.max(np.real(eigenvalues))
        assert max_real < 0, f"Drift matrix is unstable: max Re(eigenvalue) = {max_real}"

    def test_first_order_roundtrip_ar(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """DT→CT→DT roundtrip for AR coefficient at original study interval.

        rho_mood = 0.6 from weekly study → CT drift → discretize at dt=7 → recover ≈ 0.6
        """
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)
        ssm_priors = builder._convert_priors_to_ssm(
            weekly_study_priors, two_construct_model_spec, ssm_spec=spec
        )

        # Build drift matrix at prior means
        n = spec.n_latent
        drift = jnp.zeros((n, n))
        mu_diag = ssm_priors.drift_diag["mu"]
        if isinstance(mu_diag, list):
            drift = drift.at[jnp.diag_indices(n)].set(jnp.array([-abs(v) for v in mu_diag]))
        else:
            drift = drift.at[jnp.diag_indices(n)].set(-abs(mu_diag))

        if spec.drift_mask is not None:
            mu_offdiag = ssm_priors.drift_offdiag["mu"]
            offdiag_vals = mu_offdiag if isinstance(mu_offdiag, list) else [mu_offdiag]
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        drift = drift.at[i, j].set(offdiag_vals[idx])
                        idx += 1

        # Discretize at dt=7 (weekly) — should recover original DT AR
        dt_weekly = 7.0
        F_weekly = jla.expm(drift * dt_weekly)

        # Diagonal of F ≈ exp(a_ii * dt) ≈ original AR coefficient
        # mood: exp(-0.073 * 7) ≈ exp(-0.511) ≈ 0.6
        original_ar_mood = 3.0 / 5.0  # Beta(3,2) mean = 0.6
        recovered_ar_mood = float(F_weekly[0, 0])
        assert abs(recovered_ar_mood - original_ar_mood) < 0.05, (
            f"Weekly roundtrip mood AR: got {recovered_ar_mood:.4f}, "
            f"expected ≈{original_ar_mood:.4f}"
        )

        # stress: dt=1 for this prior, so at dt=7 it's exp(-0.693*7) ≈ 0.008
        # This is correct — the CT rate was derived from daily, so weekly persistence is very low
        recovered_ar_stress = float(F_weekly[1, 1])
        assert recovered_ar_stress < 0.05, (
            f"Stress AR at weekly interval should be very low (daily-derived rate), "
            f"got {recovered_ar_stress:.4f}"
        )

        # Discretize at dt=1 (daily) for stress roundtrip
        F_daily = jla.expm(drift * 1.0)
        original_ar_stress = 0.5  # Beta(2,2) mean
        recovered_daily_stress = float(F_daily[1, 1])
        assert abs(recovered_daily_stress - original_ar_stress) < 0.05, (
            f"Daily roundtrip stress AR: got {recovered_daily_stress:.4f}, "
            f"expected ≈{original_ar_stress:.4f}"
        )

    def test_first_order_roundtrip_cross_lag(
        self, two_construct_causal_spec, two_construct_model_spec, weekly_study_priors
    ):
        """DT→CT→DT roundtrip for cross-lagged coefficient.

        beta_stress_mood = 0.3 from weekly study
        → CT rate = 0.3/7 → discretize at dt=7 → F[mood,stress] ≈ 0.3
        (first-order approximation; exact requires matrix exponential)
        """
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)
        ssm_priors = builder._convert_priors_to_ssm(
            weekly_study_priors, two_construct_model_spec, ssm_spec=spec
        )

        # Build drift matrix
        n = spec.n_latent
        drift = jnp.zeros((n, n))
        mu_diag = ssm_priors.drift_diag["mu"]
        if isinstance(mu_diag, list):
            drift = drift.at[jnp.diag_indices(n)].set(jnp.array([-abs(v) for v in mu_diag]))

        if spec.drift_mask is not None:
            mu_offdiag = ssm_priors.drift_offdiag["mu"]
            offdiag_vals = mu_offdiag if isinstance(mu_offdiag, list) else [mu_offdiag]
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        drift = drift.at[i, j].set(offdiag_vals[idx])
                        idx += 1

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
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )
        spec = builder._convert_spec_to_ssm(two_construct_model_spec)
        ssm_priors = builder._convert_priors_to_ssm(
            weekly_study_priors, two_construct_model_spec, ssm_spec=spec
        )

        # Build drift and diffusion at prior means
        n = spec.n_latent
        drift = jnp.zeros((n, n))
        mu_diag = ssm_priors.drift_diag["mu"]
        if isinstance(mu_diag, list):
            drift = drift.at[jnp.diag_indices(n)].set(jnp.array([-abs(v) for v in mu_diag]))

        if spec.drift_mask is not None:
            mu_offdiag = ssm_priors.drift_offdiag["mu"]
            offdiag_vals = mu_offdiag if isinstance(mu_offdiag, list) else [mu_offdiag]
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        drift = drift.at[i, j].set(offdiag_vals[idx])
                        idx += 1

        # Simple diagonal diffusion
        diff_sd = ssm_priors.diffusion_diag.get("sigma", 1.0)
        diffusion_cov = jnp.eye(n) * diff_sd**2

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

        from causal_ssm_agent.models.ssm.inference import prior_predictive
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        builder = SSMModelBuilder(
            model_spec=two_construct_model_spec,
            priors=weekly_study_priors,
            causal_spec=two_construct_causal_spec,
        )

        # Build model with minimal mock data
        n_time = 30
        mock_data = pl.DataFrame(
            {
                "mood_rating": np.random.randn(n_time) * 1.5 + 5,
                "stress_self_report": np.random.randn(n_time) * 1.5 + 5,
                "stress_cortisol": np.random.randn(n_time) * 2 + 10,
                "time": np.arange(n_time, dtype=float),
            }
        )
        model = builder.build_model(mock_data)

        # Sample from prior predictive
        times = jnp.arange(n_time, dtype=jnp.float32)
        samples = prior_predictive(model, times, num_samples=20, seed=42)

        # Check key deterministic sites exist and are finite
        assert "drift" in samples, "Missing 'drift' in prior predictive samples"
        drift_samples = samples["drift"]
        assert jnp.all(jnp.isfinite(drift_samples)), "drift samples contain NaN/Inf"

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
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        causal_spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "mood",
                        "description": "Mood",
                        "role": "endogenous",
                        "is_outcome": True,
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
                    },
                    {
                        "name": "stress",
                        "description": "Stress",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
                    },
                ],
                "edges": [
                    {"cause": "stress", "effect": "mood", "description": "test", "lagged": True},
                ],
            },
            "measurement": {"indicators": []},
        }

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
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
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

        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_mask=drift_mask,
        )

        builder_w = SSMModelBuilder(
            model_spec=model_spec, priors=priors_weekly, causal_spec=causal_spec
        )
        ssm_priors_w = builder_w._convert_priors_to_ssm(
            priors_weekly, model_spec, ssm_spec=ssm_spec
        )

        builder_d = SSMModelBuilder(
            model_spec=model_spec, priors=priors_daily, causal_spec=causal_spec
        )
        ssm_priors_d = builder_d._convert_priors_to_ssm(priors_daily, model_spec, ssm_spec=ssm_spec)

        # Weekly: mixed intervals (beta=7d, rho=1d) → first-order: 0.3 / 7 ≈ 0.043
        mu_w = ssm_priors_w.drift_offdiag["mu"]
        mu_w_val = mu_w[0] if isinstance(mu_w, list) else mu_w
        assert abs(mu_w_val - 0.3 / 7.0) < 0.01

        # Daily: uniform intervals (all 1d) → exact logm is applied.
        # Phi = [[0.5, 0.3], [0, 0.5]] (identical AR eigenvalues)
        # logm off-diagonal: 0.3 / 0.5 = 0.6 (exact CT coupling rate)
        mu_d = ssm_priors_d.drift_offdiag["mu"]
        mu_d_val = mu_d[0] if isinstance(mu_d, list) else mu_d
        expected_logm = 0.3 / 0.5  # c / a for repeated eigenvalue a
        assert abs(mu_d_val - expected_logm) < 0.05, (
            f"Daily case uses exact logm: got {mu_d_val}, expected {expected_logm}"
        )

        # Rates should differ significantly (exact logm vs first-order)
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
        Our pipeline stores drift_diag as a positive magnitude that gets
        negated via -abs() in the model. So:
          drift_diag_mu = -ln(rho)/dt  (positive)
          actual_drift  = -drift_diag_mu = ln(rho)/dt  (negative)
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

    def test_builder_uses_logm_when_intervals_match(self, two_construct_causal_spec):
        """SSMModelBuilder applies exact logm when all parameters have the same dt.

        When all reference_interval_days values are equal, the builder should
        assemble the full DT transition matrix Phi and apply logm(Phi)/dt
        instead of the first-order element-wise approximation.
        """
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
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
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

        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_mask=drift_mask,
        )

        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors=priors,
            causal_spec=two_construct_causal_spec,
        )
        ssm_priors = builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

        # Verify logm was applied by checking roundtrip consistency:
        # Reconstruct Phi from the exact CT drift, then check it matches original DT params
        from scipy.linalg import expm

        drift_diag = ssm_priors.drift_diag["mu"]
        drift_offdiag = ssm_priors.drift_offdiag["mu"]

        A = np.zeros((2, 2))
        A[0, 0] = -abs(drift_diag[0])  # model negates diagonal
        A[1, 1] = -abs(drift_diag[1])
        A[0, 1] = drift_offdiag[0]

        Phi_reconstructed = expm(A * 7.0)

        # Should recover original DT values closely
        rho_mood_original = 3.0 / 5.0  # E[Beta(3,2)] = 0.6
        rho_stress_original = 0.5  # E[Beta(2,2)] = 0.5
        beta_original = 0.3

        assert abs(Phi_reconstructed[0, 0] - rho_mood_original) < 0.01, (
            f"Roundtrip rho_mood: got {Phi_reconstructed[0, 0]:.4f}, expected {rho_mood_original}"
        )
        assert abs(Phi_reconstructed[1, 1] - rho_stress_original) < 0.01, (
            f"Roundtrip rho_stress: got {Phi_reconstructed[1, 1]:.4f}, expected {rho_stress_original}"
        )
        assert abs(Phi_reconstructed[0, 1] - beta_original) < 0.01, (
            f"Roundtrip beta: got {Phi_reconstructed[0, 1]:.4f}, expected {beta_original}"
        )

    def test_edge_lag_days_populated(self, two_construct_causal_spec):
        """Builder stores edge lag metadata from causal spec during mask building."""
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
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
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
        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors={},
            causal_spec=two_construct_causal_spec,
        )
        # Building masks populates _edge_lag_days
        builder._build_masks_from_causal_spec(
            ["mood", "stress"], ["mood_rating", "stress_self_report"], 2, 2
        )

        # stress -> mood edge, both daily, lagged=True: lag = 24h = 1.0 day
        assert len(builder._edge_lag_days) == 1
        # effect_idx=0 (mood), cause_idx=1 (stress)
        assert (0, 1) in builder._edge_lag_days
        assert abs(builder._edge_lag_days[(0, 1)] - 1.0) < 0.01

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
                    "constraint": "correlation",
                    "description": "",
                    "search_context": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
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
        # Very large beta → CT rate implies very fast coupling (short timescale)
        # but edge lag is 1 day → should warn about mismatch
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 5.0, "sigma": 1.0},
            },
        }
        drift_mask = np.array([[True, True], [False, True]])
        ssm_spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            drift_mask=drift_mask,
        )
        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors=priors,
            causal_spec=two_construct_causal_spec,
        )
        # Must build masks first to populate _edge_lag_days
        builder._build_masks_from_causal_spec(
            ["mood", "stress"], ["mood_rating", "stress_self_report"], 2, 2
        )
        with caplog.at_level(logging.WARNING, logger="causal_ssm_agent.models.ssm_builder"):
            builder._convert_priors_to_ssm(priors, model_spec, ssm_spec=ssm_spec)

        # Large beta_CT → implied timescale << 1 day, edge lag = 1 day → warning
        lag_warnings = [r for r in caplog.records if "mismatch" in r.message.lower()]
        assert len(lag_warnings) >= 1, (
            f"Expected lag mismatch warning, got: {[r.message for r in caplog.records]}"
        )
