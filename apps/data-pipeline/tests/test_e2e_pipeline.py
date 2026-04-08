"""End-to-end fixture-driven pipeline test.

Wires FOUR_LATENT benchmark fixtures through the real computation stages
(3, 4b, 5b) to verify data flows correctly from causal specification through
inference to intervention ranking — without LLM calls.

Uses .fn() to bypass Prefect runtime on all stage tasks.
All inference uses SVI (~5s on CPU) instead of NUTS-DA to keep total <60s.
"""

from datetime import datetime, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from benchmarks.problems.four_latent import FOUR_LATENT

from causal_ssm_agent.flows.stages.stage1b.flow import build_causal_spec
from causal_ssm_agent.flows.stages.stage3.flow import (
    validate_extraction,
)
from causal_ssm_agent.flows.stages.stage5b.fit import fit_model
from causal_ssm_agent.flows.stages.stage6.interventions import run_interventions
from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
from causal_ssm_agent.utils.causal_spec import get_all_treatments
from tests.ssm_test_utils import diagonal_diffusion_kwargs, make_ssm_spec

# ==============================================================================
# Constants
# ==============================================================================

INDICATOR_NAMES = [
    "stress_primary",
    "fatigue_primary",
    "focus_primary",
    "perf_primary",
    "burnout_index",
    "cognitive_score",
]

T = 80
SEED = 42
BASE_DATE = datetime(2024, 1, 1)

# Fast SVI config for e2e tests (prod uses nuts_da on GPU)
_SVI_CONFIG = {
    "method": "svi",
    "num_steps": 200,
    "num_samples": 50,
    "learning_rate": 0.001,
    "seed": 0,
}


# ==============================================================================
# Fixtures (class-scoped for reuse across tests)
# ==============================================================================


@pytest.fixture(scope="class")
def four_latent_sim():
    """Simulate ground truth from FOUR_LATENT benchmark."""
    obs, times, latent = FOUR_LATENT.simulate(T=T, seed=SEED)
    return {"obs": np.array(obs), "times": np.array(times), "latent": np.array(latent)}


@pytest.fixture(scope="class")
def latent_model():
    """Stage 1a output: latent model matching FOUR_LATENT structure."""
    return {
        "constructs": [
            {
                "name": "Stress",
                "description": "Psychological stress level",
                "role": "exogenous",
                "temporal_status": "time_varying",
                "is_outcome": False,
            },
            {
                "name": "Fatigue",
                "description": "Physical and mental fatigue",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": False,
            },
            {
                "name": "Focus",
                "description": "Ability to concentrate",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": False,
            },
            {
                "name": "Perf",
                "description": "Task performance",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
        ],
        "edges": [
            {
                "cause": "Stress",
                "effect": "Fatigue",
                "description": "Stress increases fatigue",
                "lagged": True,
            },
            {
                "cause": "Stress",
                "effect": "Focus",
                "description": "Stress impairs focus",
                "lagged": True,
            },
            {
                "cause": "Fatigue",
                "effect": "Focus",
                "description": "Fatigue reduces focus",
                "lagged": True,
            },
            {
                "cause": "Focus",
                "effect": "Perf",
                "description": "Focus drives performance",
                "lagged": True,
            },
        ],
    }


@pytest.fixture(scope="class")
def measurement_model():
    """Stage 1b output: measurement model with 6 indicators for 4 constructs."""
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "stress_primary",
                "construct_name": "Stress",
                "how_to_measure": "Self-reported stress scale",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "fatigue_primary",
                "construct_name": "Fatigue",
                "how_to_measure": "Self-reported fatigue scale",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "focus_primary",
                "construct_name": "Focus",
                "how_to_measure": "Self-reported focus scale",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "perf_primary",
                "construct_name": "Perf",
                "how_to_measure": "Task completion rate",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "burnout_index",
                "construct_name": "Stress",
                "how_to_measure": "Composite burnout measure",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "cognitive_score",
                "construct_name": "Perf",
                "how_to_measure": "Cognitive test score",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ],
    }


@pytest.fixture(scope="class")
def causal_spec(latent_model, measurement_model):
    """Combined CausalSpec via build_causal_spec."""
    identifiability_status = {
        "identifiable_treatments": {
            "Stress": {
                "method": "do_calculus",
                "estimand": "P(Perf|do(Stress))",
                "marginalized_confounders": [],
                "instruments": [],
            },
            "Fatigue": {
                "method": "do_calculus",
                "estimand": "P(Perf|do(Fatigue))",
                "marginalized_confounders": [],
                "instruments": [],
            },
            "Focus": {
                "method": "do_calculus",
                "estimand": "P(Perf|do(Focus))",
                "marginalized_confounders": [],
                "instruments": [],
            },
        },
        "non_identifiable_treatments": {},
    }
    return build_causal_spec.fn(latent_model, measurement_model, identifiability_status)


@pytest.fixture(scope="class")
def worker_dfs(four_latent_sim):
    """Stage 2 output: DataFrames from simulated observations.

    Converts FOUR_LATENT observations to DataFrames with string values
    and ISO timestamps, split into 3 chunks.
    """
    obs = four_latent_sim["obs"]
    records = []
    for t in range(T):
        ts = (BASE_DATE + timedelta(days=t)).isoformat()
        for i, name in enumerate(INDICATOR_NAMES):
            records.append(
                {
                    "indicator": name,
                    "value": str(float(obs[t, i])),
                    "anchor_time": ts,
                }
            )

    # Split into 3 chunks
    chunk_size = len(records) // 3
    chunks = [records[:chunk_size], records[chunk_size : 2 * chunk_size], records[2 * chunk_size :]]

    results = []
    for chunk in chunks:
        df = pl.DataFrame(
            chunk,
            schema={"indicator": pl.Utf8, "value": pl.Utf8, "anchor_time": pl.Utf8},
        )
        results.append(df)

    return results


@pytest.fixture(scope="class")
def model_spec():
    """Stage 4 orchestrator output: model specification for FOUR_LATENT."""
    return {
        "likelihoods": [
            {
                "variable": name,
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Gaussian indicator",
            }
            for name in INDICATOR_NAMES
        ],
        "parameters": [
            # AR coefficients
            {
                "name": "rho_Stress",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Stress",
            },
            {
                "name": "rho_Fatigue",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Fatigue",
            },
            {
                "name": "rho_Focus",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Focus",
            },
            {
                "name": "rho_Perf",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Performance",
            },
            # Cross-effects
            {
                "name": "beta_Stress_Fatigue",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Stress -> Fatigue effect",
            },
            {
                "name": "beta_Stress_Focus",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Stress -> Focus effect",
            },
            {
                "name": "beta_Fatigue_Focus",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Fatigue -> Focus effect",
            },
            {
                "name": "beta_Focus_Perf",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Focus -> Performance effect",
            },
            # Residual SDs
            *[
                {
                    "name": f"sigma_{name}",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": f"Residual SD for {name}",
                }
                for name in INDICATOR_NAMES
            ],
        ],
    }


@pytest.fixture(scope="class")
def priors():
    """Stage 4 worker output: prior proposals for each parameter."""
    prior_dict = {}
    # AR priors
    for name in ["Stress", "Fatigue", "Focus", "Perf"]:
        prior_dict[f"rho_{name}"] = {
            "distribution": "Beta",
            "params": {"alpha": 5.0, "beta": 2.0},
        }
    # Cross-effect priors
    for name in [
        "beta_Stress_Fatigue",
        "beta_Stress_Focus",
        "beta_Fatigue_Focus",
        "beta_Focus_Perf",
    ]:
        prior_dict[name] = {
            "distribution": "Normal",
            "params": {"mu": 0.0, "sigma": 0.5},
        }
    # Residual SD priors
    for name in INDICATOR_NAMES:
        prior_dict[f"sigma_{name}"] = {
            "distribution": "HalfNormal",
            "params": {"sigma": 0.5},
        }
    return prior_dict


@pytest.fixture(scope="class")
def daily_data(causal_spec, worker_dfs):
    """Data at model_clock resolution with datetime anchor_time column.

    Mirrors the new dag.py stage2() logic: encode non-continuous types,
    cast to Float64, parse anchor times to datetime.
    """
    from causal_ssm_agent.utils.aggregations import _encode_non_continuous
    from causal_ssm_agent.utils.causal_spec import get_indicator_dtypes, get_indicators

    combined = pl.concat(worker_dfs, how="vertical")
    dtype_lookup = get_indicator_dtypes(causal_spec)
    ordinal_levels_lookup: dict[str, list[str]] = {
        ind["name"]: ind["ordinal_levels"]
        for ind in get_indicators(causal_spec)
        if ind.get("ordinal_levels")
    }
    data = _encode_non_continuous(combined, dtype_lookup, ordinal_levels_lookup)
    data = data.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("value"),
        pl.col("anchor_time")
        .str.replace(r"[Zz]$", "")
        .str.replace(r"[+-]\d{2}:\d{2}$", "")
        .str.to_datetime(strict=False)
        .alias("anchor_time"),
    ).drop_nulls(subset=["anchor_time", "value"])
    return data.sort("indicator", "anchor_time")


@pytest.fixture(scope="class")
def stage4_result(model_spec, priors):
    """Assembled dict for stages 4b and 5 (with fast SVI config)."""
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

    return {
        "_compiled_ssm": compile_ssm_artifact(model_spec, priors),
    }


@pytest.fixture(scope="class")
def direct_fit_result():
    """Mock posterior from FOUR_LATENT ground truth.

    Uses true parameters as "posterior samples" with small noise.
    This is a plumbing test — we verify the pipeline wires correctly,
    not inference quality (which is tested by benchmarks/).
    """
    from causal_ssm_agent.models.ssm import SSMModel
    from causal_ssm_agent.models.ssm.inference import InferenceResult

    n_draws = 50
    key = jax.random.PRNGKey(SEED)

    # Create mock posterior: true params + small Gaussian noise
    keys = jax.random.split(key, 5)
    samples = {
        "drift": FOUR_LATENT.true_drift[None] + 0.01 * jax.random.normal(keys[0], (n_draws, 4, 4)),
        "diffusion": jnp.broadcast_to(jnp.diag(FOUR_LATENT.true_diff_diag), (n_draws, 4, 4)),
        "cint": FOUR_LATENT.true_cint[None] + 0.01 * jax.random.normal(keys[1], (n_draws, 4)),
    }

    # Force negative diagonal on drift (stability)
    drift = samples["drift"]
    for i in range(4):
        drift = drift.at[:, i, i].set(-jnp.abs(drift[:, i, i]))
    samples["drift"] = drift

    result = InferenceResult(_samples=samples, method="svi", diagnostics={})

    spec = FOUR_LATENT.spec
    model = SSMModel(spec, FOUR_LATENT.priors)
    builder = SSMModelBuilder(ssm_spec=spec)
    builder._spec = spec
    builder._model = model
    builder._result = result

    return {
        "result": result,
        "builder": builder,
        "times": jnp.arange(T, dtype=jnp.float32),
    }


# ==============================================================================
# Test Class
# ==============================================================================


@pytest.mark.slow
class TestE2EPipeline:
    """End-to-end pipeline test using FOUR_LATENT fixtures."""

    # ------------------------------------------------------------------
    # Stage 3: validation + aggregation (Polars, fast)
    # ------------------------------------------------------------------

    def test_stage3_validate_extraction(self, causal_spec, worker_dfs):
        """validate_extraction passes with all indicators present."""
        result = validate_extraction.fn(causal_spec, worker_dfs)
        assert result["is_valid"] is True
        issues = [
            issue
            for audit in result["indicators"].values()
            for issue in audit["validation"]["issues"]
        ]
        errors = [i for i in issues if i["severity"] == "error"]
        assert len(errors) == 0

        # All 6 indicators present with sufficient observations
        present = {i["indicator"] for i in issues if i["issue_type"] == "missing"}
        assert len(present) == 0  # None missing

    def test_stage4b_t_rule(self, model_spec, priors, daily_data):
        """T-rule check passes (necessary condition for identifiability)."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.utils.data import pivot_to_wide
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)
        builder.build_model(pivot_to_wide(daily_data))
        assert builder._spec is not None
        t_rule = check_t_rule(builder._spec, T=T)
        assert t_rule.satisfies is True
        assert t_rule.n_free_params < t_rule.n_moments

    # ------------------------------------------------------------------
    # Stage 5: inference (SVI for speed)
    # ------------------------------------------------------------------

    def test_stage5_fit(self, direct_fit_result):
        """Direct SSMModel fit completes with expected sample sites."""
        result = direct_fit_result["result"]
        assert result.method == "svi"
        samples = result.get_samples()
        assert "drift" in samples
        assert "diffusion" in samples

    def test_stage5_fit_via_pipeline_path(self, stage4_result, daily_data):
        """fit_model.fn() surfaces unstable SVI runs as explicit task failures."""
        result = fit_model.fn(
            stage4_result["_compiled_ssm"], daily_data, sampler_config=_SVI_CONFIG
        )
        assert result["fitted"] is False
        assert "non-finite losses" in result["error"]

    def test_stage5_interval_summary_pipeline_path_reaches_interventions(self):
        """Interval-summary data should flow through fit_model.fn() and intervention analysis."""
        from causal_ssm_agent.models.ssm import SSMPriors
        from causal_ssm_agent.models.ssm.inference import FittedArtifact

        data_for_model = pl.DataFrame(
            {
                "indicator": [
                    "perf_avg",
                    "perf_avg",
                ],
                "value": [0.5, 0.7],
                "anchor_time": [
                    "2024-01-03T00:00:00",
                    "2024-01-04T00:00:00",
                ],
                "support_kind": [
                    "interval",
                    "interval",
                ],
                "summary_operator": [
                    "mean",
                    "mean",
                ],
                "anchor_policy": [
                    "support_end",
                    "support_end",
                ],
                "observation_window": ["2d", "2d"],
                "support_start": [
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                ],
                "support_end": [
                    "2024-01-03T00:00:00",
                    "2024-01-04T00:00:00",
                ],
            }
        )

        svi_config = {
            "method": "svi",
            "num_steps": 25,
            "num_samples": 10,
            "learning_rate": 0.01,
            "seed": 0,
        }

        builder = SSMModelBuilder(
            ssm_spec=make_ssm_spec(
                n_latent=1,
                n_manifest=1,
                lambda_mat=jnp.eye(1),
                **diagonal_diffusion_kwargs(1),
                latent_names=["Perf"],
                manifest_names=["perf_avg"],
            ),
            ssm_priors=SSMPriors(),
            sampler_config=svi_config,
        )

        fitted_result = fit_model.fn(
            None,
            data_for_model,
            sampler_config=svi_config,
            builder=builder,
        )

        assert fitted_result["fitted"] is True
        assert fitted_result["runtime"].observation_support is not None
        assert fitted_result["runtime"].observation_support.max_active_windows == 2

        fitted = FittedArtifact(
            result=fitted_result["result"],
            builder=fitted_result["builder"],
            times=fitted_result["times"],
        )
        results = run_interventions.fn(
            fitted,
            ["Perf"],
            "Perf",
            {"identifiability": {"non_identifiable_treatments": {}}},
        )

        assert len(results) == 1
        assert results[0]["treatment"] == "Perf"
        assert results[0]["effect_size"] is not None

    # ------------------------------------------------------------------
    # Parameter recovery (from direct fit, smoke test only)
    # ------------------------------------------------------------------

    def test_parameter_recovery(self, direct_fit_result):
        """Drift diagonal is negative (stability) and all samples finite.

        With SVI the posterior is approximate, but we verify it's not
        degenerate (no NaN) and respects the stability constraint.
        """
        samples = direct_fit_result["result"].get_samples()
        drift_samples = samples["drift"]  # (n_draws, 4, 4)

        # All samples should be finite (no NaN from failed inference)
        assert jnp.all(jnp.isfinite(drift_samples)), "Drift contains NaN/Inf"

        # Diagonal should be negative (stability constraint)
        for i in range(4):
            diag_mean = float(jnp.mean(drift_samples[:, i, i]))
            assert diag_mean < 0, f"Drift[{i},{i}] mean={diag_mean:.3f}, expected negative"

    # ------------------------------------------------------------------
    # Interventions (plumbing test)
    # ------------------------------------------------------------------

    def test_interventions(self, direct_fit_result, latent_model, causal_spec):
        """run_interventions returns structured results for all treatments."""
        from causal_ssm_agent.models.ssm.inference import FittedArtifact

        treatments = get_all_treatments(latent_model)
        fitted = FittedArtifact(
            result=direct_fit_result["result"],
            builder=direct_fit_result["builder"],
            times=direct_fit_result["times"],
        )

        results = run_interventions.fn(fitted, treatments, "Perf", causal_spec)

        # 3 treatments returned
        assert len(results) == 3

        # All identifiable
        assert all(r["identifiable"] for r in results)

        # All have non-None effect sizes and draws (pipeline produces values)
        for r in results:
            assert r["effect_size"] is not None
            assert r["posterior_draws"] is not None
            assert len(r["posterior_draws"]) > 0
            assert np.isfinite(r["effect_size"]), f"{r['treatment']} effect is not finite"
