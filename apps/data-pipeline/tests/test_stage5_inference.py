"""Tests for Stage 5 inference task logging and orchestration."""

import logging
from types import SimpleNamespace

import jax.numpy as jnp
import polars as pl

from causal_ssm_agent.flows.stages import stage5_inference
from causal_ssm_agent.models.ssm.inference_structure import (
    FirstPassRBPlan,
    InferenceStructurePlan,
)
from causal_ssm_agent.models.ssm_builder import PreparedModelRuntime


class _FakeResult:
    method = "laplace_em"

    def __init__(self) -> None:
        self.diagnostics = {}

    def get_mcmc_diagnostics(self):
        return None

    def get_svi_diagnostics(self):
        return None

    def get_smc_diagnostics(self):
        return {"n_levels": 3}

    def get_loo_diagnostics(self, *, model_fn, observations, times):
        return {"elpd_loo": -12.3}

    def get_posterior_marginals(self):
        return [{"parameter": "theta"}]

    def get_posterior_pairs(self):
        return []

    def get_samples(self):
        return {"theta": jnp.zeros((4, 1), dtype=jnp.float32)}


class _FakeBuilder:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

        def _model(*_args, **_kwargs):
            return None

        self._model = SimpleNamespace(
            model=_model,
            make_likelihood_backend=lambda: "particle-backend",
        )

    def fit_prepared(self, observations, times):
        return self._result


def _make_runtime(fake_builder: _FakeBuilder) -> PreparedModelRuntime:
    return PreparedModelRuntime(
        builder=fake_builder,
        wide_data=pl.DataFrame(
            {
                "time": [0.0, 1.5],
                "sleep_avg": [0.2, None],
                "energy": [0.8, 0.5],
            }
        ),
        observation_data=None,
        observation_support=SimpleNamespace(
            requires_interval_summary_handling=True,
            interval_summary_manifest_names=["sleep_avg"],
            max_active_windows=2,
        ),
        inference_structure=InferenceStructurePlan(
            likelihood_path="particle",
            auto_method="laplace_em",
            first_pass_rb=FirstPassRBPlan(
                status="inactive",
                inactive_reason="interval_summary_support",
            ),
        ),
        observations=jnp.array([[0.2, 0.8], [jnp.nan, 0.5]], dtype=jnp.float32),
        times=jnp.array([0.0, 1.5], dtype=jnp.float32),
        manifest_names=["sleep_avg", "energy"],
    )


def test_fit_model_logs_runtime_summary_and_diagnostic_boundaries(monkeypatch, caplog):
    fake_result = _FakeResult()
    fake_builder = _FakeBuilder(fake_result)
    runtime = _make_runtime(fake_builder)

    monkeypatch.setattr(stage5_inference, "prepare_model_runtime", lambda **_kwargs: runtime)

    data_for_model = pl.DataFrame(
        {
            "indicator": ["sleep_avg", "energy", "energy"],
            "value": [0.2, 0.8, 0.5],
            "anchor_time": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                "2024-01-02T12:00:00",
            ],
        }
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.fit_model.fn(
            None,
            data_for_model,
            sampler_config={"method": "auto"},
            builder=fake_builder,
        )

    assert result["fitted"] is True
    assert "Prepared runtime in" in caplog.text
    assert "support=interval(1: sleep_avg) max_active_windows=2" in caplog.text
    assert "Manifest order: sleep_avg, energy" in caplog.text
    assert (
        "Inference route: requested_method=auto auto_method=laplace_em "
        "likelihood_path=particle first_pass_rb=inactive "
        "inactive_reason=interval_summary_support"
    ) in caplog.text
    assert "Starting inference kernel..." in caplog.text
    assert "Collecting sampler diagnostics..." in caplog.text
    assert "Computing LOO diagnostics..." in caplog.text
    assert "Extracting posterior summaries..." in caplog.text
    assert "Posterior summaries ready in" in caplog.text


def test_run_power_scaling_logs_completion_summary(monkeypatch, caplog):
    fake_result = _FakeResult()
    fake_builder = _FakeBuilder(fake_result)
    runtime = _make_runtime(fake_builder)

    class _FakePowerScalingResult:
        def __init__(self) -> None:
            self.prior_sensitivity = {"theta": 0.1}
            self.likelihood_sensitivity = {"theta": 0.2}
            self.diagnosis = {"theta": "well_identified"}
            self.psis_k_hat = {"theta": 0.05}

        def print_report(self):
            return None

    monkeypatch.setattr(
        "causal_ssm_agent.utils.parametric_id_postfit.power_scaling_sensitivity",
        lambda **_kwargs: _FakePowerScalingResult(),
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.run_power_scaling.fn(
            {"fitted": True, "result": fake_result, "runtime": runtime}
        )

    assert result["checked"] is True
    assert "Running power-scaling sensitivity" in caplog.text
    assert "Power-scaling complete in" in caplog.text
