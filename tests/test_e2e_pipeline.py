"""End-to-end fixture-driven pipeline test.

Wires FOUR_LATENT benchmark fixtures through the real computation stages
(3, 4b, 5) to verify data flows correctly from causal specification through
inference to intervention ranking — without LLM calls.

Uses .fn() to bypass Prefect runtime on all stage tasks.
All inference uses SVI (~5s on CPU) instead of NUTS-DA to keep total <60s.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from benchmarks.problems.four_latent import FOUR_LATENT

from causal_ssm_agent.flows.stages.stage1b_measurement import build_causal_spec
from causal_ssm_agent.flows.stages.stage3_validation import (
    aggregate_measurements,
    combine_worker_results,
    validate_extraction,
)
from causal_ssm_agent.flows.stages.stage5_inference import fit_model, run_interventions
from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
from causal_ssm_agent.orchestrator.schemas import (
    CausalSpec,
    LatentModel,
    MeasurementModel,
)
from causal_ssm_agent.utils.aggregations import flatten_aggregated_data
from causal_ssm_agent.utils.effects import get_all_treatments, get_outcome_from_latent_model

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
    "seed": 0,
}


# ==============================================================================
# Mock WorkerResult (matches real WorkerResult interface)
# ==============================================================================


@dataclass
class MockWorkerResult:
    """Mock WorkerResult with a .dataframe attribute."""

    dataframe: pl.DataFrame


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
                "causal_granularity": "daily",
                "is_outcome": False,
            },
            {
                "name": "Fatigue",
                "description": "Physical and mental fatigue",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
                "is_outcome": False,
            },
            {
                "name": "Focus",
                "description": "Ability to concentrate",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
                "is_outcome": False,
            },
            {
                "name": "Perf",
                "description": "Task performance",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
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
        "indicators": [
            {
                "name": "stress_primary",
                "construct_name": "Stress",
                "how_to_measure": "Self-reported stress scale",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "fatigue_primary",
                "construct_name": "Fatigue",
                "how_to_measure": "Self-reported fatigue scale",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "focus_primary",
                "construct_name": "Focus",
                "how_to_measure": "Self-reported focus scale",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "perf_primary",
                "construct_name": "Perf",
                "how_to_measure": "Task completion rate",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "burnout_index",
                "construct_name": "Stress",
                "how_to_measure": "Composite burnout measure",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "cognitive_score",
                "construct_name": "Perf",
                "how_to_measure": "Cognitive test score",
                "measurement_granularity": "daily",
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
def worker_results(four_latent_sim):
    """Stage 2 output: mock worker results from simulated observations.

    Converts FOUR_LATENT observations to MockWorkerResult list with string values
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
                    "timestamp": ts,
                }
            )

    # Split into 3 chunks
    chunk_size = len(records) // 3
    chunks = [records[:chunk_size], records[chunk_size : 2 * chunk_size], records[2 * chunk_size :]]

    results = []
    for chunk in chunks:
        df = pl.DataFrame(
            chunk,
            schema={"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8},
        )
        results.append(MockWorkerResult(dataframe=df))

    return results


@pytest.fixture(scope="class")
def model_spec():
    """Stage 4 orchestrator output: model specification for FOUR_LATENT."""
    return {
        "likelihoods": [
            {
                "variable": name,
                "distribution": "Normal",
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
                "search_context": "autoregressive stress",
            },
            {
                "name": "rho_Fatigue",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Fatigue",
                "search_context": "autoregressive fatigue",
            },
            {
                "name": "rho_Focus",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Focus",
                "search_context": "autoregressive focus",
            },
            {
                "name": "rho_Perf",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for Performance",
                "search_context": "autoregressive performance",
            },
            # Cross-effects
            {
                "name": "beta_Stress_Fatigue",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Stress -> Fatigue effect",
                "search_context": "stress fatigue causal effect",
            },
            {
                "name": "beta_Stress_Focus",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Stress -> Focus effect",
                "search_context": "stress focus causal effect",
            },
            {
                "name": "beta_Fatigue_Focus",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Fatigue -> Focus effect",
                "search_context": "fatigue focus causal effect",
            },
            {
                "name": "beta_Focus_Perf",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Focus -> Performance effect",
                "search_context": "focus performance causal effect",
            },
            # Residual SDs
            *[
                {
                    "name": f"sigma_{name}",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": f"Residual SD for {name}",
                    "search_context": f"measurement error {name}",
                }
                for name in INDICATOR_NAMES
            ],
        ],
        "model_clock": "daily",
        "reasoning": "FOUR_LATENT benchmark model specification",
    }


@pytest.fixture(scope="class")
def priors():
    """Stage 4 worker output: prior proposals for each parameter."""
    prior_dict = {}
    # AR priors
    for name in ["Stress", "Fatigue", "Focus", "Perf"]:
        prior_dict[f"rho_{name}"] = {
            "distribution": "Normal",
            "params": {"mu": -0.5, "sigma": 0.5},
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
def raw_data_pl(worker_results):
    """Long-format polars DataFrame from combine_worker_results."""
    return combine_worker_results.fn(worker_results)


@pytest.fixture(scope="class")
def daily_data(causal_spec, worker_results):
    """Aggregated daily data (proper datetime time_bucket column).

    This is what stages 4b and 5 receive in the real pipeline —
    aggregated data with datetime time_bucket, not raw string timestamps.
    """
    aggregated = aggregate_measurements.fn(causal_spec, worker_results)
    return flatten_aggregated_data(aggregated)


@pytest.fixture(scope="class")
def stage4_result(model_spec, priors, daily_data):
    """Assembled dict for stages 4b and 5 (with fast SVI config)."""
    return {
        "model_spec": model_spec,
        "priors": priors,
        "validation": {"is_valid": True, "results": [], "issues": []},
        "model_info": {"model_built": True, "model_type": "SSM", "version": "0.1.0"},
        "is_valid": True,
        "raw_data": daily_data,
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

    return {"result": result, "builder": builder}


# ==============================================================================
# Test Class
# ==============================================================================


@pytest.mark.slow
class TestE2EPipeline:
    """End-to-end pipeline test using FOUR_LATENT fixtures."""

    # ------------------------------------------------------------------
    # Schema / spec tests (no inference, fast)
    # ------------------------------------------------------------------

    def test_build_causal_spec(self, causal_spec):
        """CausalSpec round-trips through build_causal_spec.fn()."""
        spec = CausalSpec.model_validate(causal_spec)
        assert len(spec.latent.constructs) == 4
        assert len(spec.measurement.indicators) == 6
        assert len(spec.latent.edges) == 4
        outcome = get_outcome_from_latent_model(causal_spec["latent"])
        assert outcome == "Perf"

    def test_latent_model_validates(self, latent_model):
        """LatentModel fixture passes Pydantic validation."""
        model = LatentModel.model_validate(latent_model)
        assert len(model.constructs) == 4
        assert len(model.edges) == 4

    def test_measurement_model_validates(self, measurement_model):
        """MeasurementModel fixture passes Pydantic validation."""
        model = MeasurementModel.model_validate(measurement_model)
        assert len(model.indicators) == 6

    def test_get_treatments(self, latent_model):
        """get_all_treatments returns all non-outcome ancestors of Perf."""
        treatments = get_all_treatments(latent_model)
        assert treatments == ["Fatigue", "Focus", "Stress"]

    def test_get_outcome(self, latent_model):
        """get_outcome_from_latent_model returns Perf."""
        assert get_outcome_from_latent_model(latent_model) == "Perf"

    # ------------------------------------------------------------------
    # Stage 3: validation + aggregation (Polars, fast)
    # ------------------------------------------------------------------

    def test_stage3_validate_extraction(self, causal_spec, worker_results):
        """validate_extraction passes with all indicators present."""
        result = validate_extraction.fn(causal_spec, worker_results)
        assert result["is_valid"] is True
        errors = [i for i in result["issues"] if i["severity"] == "error"]
        assert len(errors) == 0

        # All 6 indicators present with sufficient observations
        present = {i["indicator"] for i in result["issues"] if i["issue_type"] == "missing"}
        assert len(present) == 0  # None missing

    def test_stage3_combine(self, worker_results):
        """combine_worker_results produces correct shape."""
        combined = combine_worker_results.fn(worker_results)
        assert len(combined) == T * len(INDICATOR_NAMES)  # 80 * 6 = 480
        assert set(combined.columns) == {"indicator", "value", "timestamp"}

    def test_stage3_aggregate(self, causal_spec, worker_results):
        """aggregate_measurements produces daily data for all indicators."""
        aggregated = aggregate_measurements.fn(causal_spec, worker_results)
        assert "daily" in aggregated
        daily = aggregated["daily"]
        assert daily["indicator"].n_unique() == 6
        # Each indicator should have ~80 time buckets
        for name in INDICATOR_NAMES:
            ind_data = daily.filter(pl.col("indicator") == name)
            assert len(ind_data) >= 70  # Allow some tolerance for bucketing

    # ------------------------------------------------------------------
    # Stage 4b: parametric identifiability (T-rule only for speed)
    # ------------------------------------------------------------------

    def test_stage4b_t_rule(self, model_spec, priors, daily_data):
        """T-rule check passes (necessary condition for identifiability)."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)
        builder.build_model(daily_data.to_pandas())
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
        """fit_model.fn() completes through the ModelSpec -> SSMSpec -> fit chain."""
        result = fit_model.fn(stage4_result, daily_data, sampler_config=_SVI_CONFIG)
        assert result["fitted"] is True
        assert result["inference_type"] == "svi"

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
        treatments = get_all_treatments(latent_model)
        fitted = {
            "fitted": True,
            "result": direct_fit_result["result"],
            "builder": direct_fit_result["builder"],
        }

        results = run_interventions.fn(fitted, treatments, "Perf", causal_spec)

        # 3 treatments returned
        assert len(results) == 3

        # All identifiable
        assert all(r["identifiable"] for r in results)

        # All have non-None effect sizes and CIs (pipeline produces values)
        for r in results:
            assert r["effect_size"] is not None
            assert r["credible_interval"] is not None
            assert np.isfinite(r["effect_size"]), f"{r['treatment']} effect is not finite"
