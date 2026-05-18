"""Tests for Stage 5 inference task logging and orchestration."""

import logging

import jax.numpy as jnp
import numpy as np
import polars as pl

from causal_ssm_agent.flows.stages.stage5b import fit as stage5_inference
from causal_ssm_agent.models.ssm.inference import InferenceResult
from causal_ssm_agent.models.ssm.inference.structure import InferenceStructurePlan
from causal_ssm_agent.models.ssm.model import SSMModel
from causal_ssm_agent.models.ssm_builder import PreparedModelRuntime, SSMModelBuilder
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.ssm_test_utils import make_ssm_spec


class _FakeResult(InferenceResult):
    def __init__(self) -> None:
        self.method = "map"
        self.diagnostics = {}
        self._samples = {"theta": jnp.zeros((4, 1), dtype=jnp.float32)}

    def get_smc_diagnostics(self):
        return {"n_levels": 3}

    def get_loo_diagnostics(
        self,
        model_fn=None,
        observations: jnp.ndarray | None = None,
        times: jnp.ndarray | None = None,
    ):
        return {"elpd_loo": -12.3}

    def get_posterior_marginals(self, n_bins: int = 50):
        del n_bins
        return [{"parameter": "theta"}]

    def get_posterior_pairs(self, max_params: int = 6, max_samples: int = 200):
        del max_params, max_samples
        return []


class _FakeBuilder(SSMModelBuilder):
    def __init__(self, result: _FakeResult) -> None:
        super().__init__()
        self._result = result
        model = SSMModel(
            make_ssm_spec(
                n_latent=1,
                n_manifest=2,
                latent_names=["sleep_state"],
                manifest_names=["sleep_avg", "energy"],
            )
        )
        self.attach_runtime_artifacts(model, result=result)

    def fit_prepared(
        self, observations: jnp.ndarray, times: jnp.ndarray, **_kwargs
    ) -> InferenceResult:
        result = self._result
        assert result is not None
        return result


def _make_observation_support_runtime() -> ObservationSupportRuntime:
    return ObservationSupportRuntime(
        anchor_times=np.array([0.0, 1.5]),
        manifest_names=["sleep_avg", "energy"],
        support_kinds=["interval", "point"],
        summary_operators=["mean", None],
        anchor_policies=["end", "end"],
        observation_windows=["1d", None],
        support_start_times=np.array([[np.nan, np.nan], [0.0, np.nan]]),
        support_end_times=np.array([[np.nan, np.nan], [1.5, np.nan]]),
        interval_prev_coeffs=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.0]],
            ]
        ),
        interval_curr_coeffs=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.0]],
            ]
        ),
        interval_weights=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        emission_slot_indices=np.array([[-1, -1], [0, -1]]),
    )


def _make_runtime(fake_builder: _FakeBuilder) -> PreparedModelRuntime:
    return PreparedModelRuntime(
        builder=fake_builder,
        model=fake_builder.model,
        spec=fake_builder.spec,
        structure_runtime=fake_builder.model.structure_runtime,
        wide_data=pl.DataFrame(
            {
                "time": [0.0, 1.5],
                "sleep_avg": [0.2, None],
                "energy": [0.8, 0.5],
            }
        ),
        observation_data=None,
        observation_support=_make_observation_support_runtime(),
        inference_structure=InferenceStructurePlan(
            structural_backend="particle",
            resolved_method="map",
            method_override=None,
            first_pass_partition=None,
        ),
        observations=jnp.array([[0.2, 0.8], [jnp.nan, 0.5]], dtype=jnp.float32),
        times=jnp.array([0.0, 1.5], dtype=jnp.float32),
        transition_inputs=None,
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
            sampler_config={"method": "map"},
            builder=fake_builder,
        )

    assert result["fitted"] is True
    assert "Prepared runtime in" in caplog.text
    assert "support=interval(1: sleep_avg) max_active_windows=2" in caplog.text
    assert "Manifest order: sleep_avg, energy" in caplog.text
    assert (
        "Inference route: requested_method=map resolved_method=map "
        "structural_backend=particle method_override=none first_pass_partition=none"
    ) in caplog.text
    assert "Starting inference kernel..." in caplog.text
    assert "Collecting sampler diagnostics..." in caplog.text
    assert "Computing LOO diagnostics..." in caplog.text
    assert "Extracting posterior summaries..." in caplog.text
    assert "Posterior summaries ready in" in caplog.text


def test_fit_model_can_skip_loo_diagnostics(monkeypatch, caplog):
    fake_result = _FakeResult()
    fake_builder = _FakeBuilder(fake_result)
    runtime = _make_runtime(fake_builder)

    monkeypatch.setattr(stage5_inference, "prepare_model_runtime", lambda **_kwargs: runtime)

    data_for_model = pl.DataFrame(
        {
            "indicator": ["sleep_avg"],
            "value": [0.2],
            "anchor_time": ["2024-01-01T00:00:00"],
        }
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.fit_model.fn(
            None,
            data_for_model,
            sampler_config={"method": "map"},
            builder=fake_builder,
            compute_loo_diagnostics=False,
        )

    assert result["fitted"] is True
    assert result["loo_diagnostics"] is None
    assert "Skipping LOO diagnostics by configuration." in caplog.text
    assert "Computing LOO diagnostics..." not in caplog.text


def test_fit_model_restores_compile_cache_before_preparing_runtime(monkeypatch):
    fake_result = _FakeResult()
    fake_builder = _FakeBuilder(fake_result)
    runtime = _make_runtime(fake_builder)
    restore_calls: list[tuple[str | None, dict | None, bool]] = []

    monkeypatch.setattr(
        stage5_inference,
        "restore_stage4_compile_cache",
        lambda workspace_id, compiled_ssm, *, wait_for_pending: (
            restore_calls.append((workspace_id, compiled_ssm, wait_for_pending)) or True
        ),
    )
    monkeypatch.setattr(stage5_inference, "prepare_model_runtime", lambda **_kwargs: runtime)

    data_for_model = pl.DataFrame(
        {
            "indicator": ["sleep_avg"],
            "value": [0.2],
            "anchor_time": ["2024-01-01T00:00:00"],
        }
    )
    compiled_ssm = {"spec": {"n_latent": 1}}

    result = stage5_inference.fit_model.fn(
        compiled_ssm,
        data_for_model,
        sampler_config={"method": "map"},
        workspace_id="workspace-123",
        wait_for_compile_cache=True,
    )

    assert restore_calls == [("workspace-123", compiled_ssm, True)]
    assert result["fitted"] is True


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
        "causal_ssm_agent.models.ssm.diagnostics.power_scaling_sensitivity",
        lambda **_kwargs: _FakePowerScalingResult(),
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.run_power_scaling.fn(
            {"fitted": True, "result": fake_result, "runtime": runtime}
        )

    assert result["checked"] is True
    assert "Running power-scaling sensitivity" in caplog.text
    assert "Power-scaling complete in" in caplog.text
