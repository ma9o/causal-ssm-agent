"""Tests for Stage 4b inference-structure payload planning."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.flows.dag import stage4b
from causal_ssm_agent.flows.stages.stage4b.flow import parametric_id_task
from causal_ssm_agent.models.ssm.inference.structure import (
    build_inference_structure_payload,
    plan_inference_structure,
)
from causal_ssm_agent.models.ssm.model import SSMSpec
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

pytestmark = pytest.mark.slow


def _support_runtime() -> ObservationSupportRuntime:
    return ObservationSupportRuntime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["y0", "y1", "y2"],
        support_kinds=["point", "point", "interval"],
        summary_operators=[None, None, "window_average"],
        anchor_policies=[None, None, "end"],
        observation_windows=["1d", "1d", "1d"],
        support_start_times=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, 0.0]]),
        support_end_times=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, 1.0]]),
        interval_prev_coeffs=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.5]]]),
        interval_curr_coeffs=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.5]]]),
        interval_weights=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [1.0]]]),
        emission_slot_indices=np.array([[-1, -1, -1], [-1, -1, 0]], dtype=np.int32),
    )


def _make_model(spec: SSMSpec, observation_support=None):
    return SimpleNamespace(
        spec=spec, observation_support=observation_support, likelihood="particle"
    )


def _make_separable_spec(first_pass_rb: bool = True) -> SSMSpec:
    return SSMSpec(
        n_latent=3,
        n_manifest=3,
        drift=jnp.diag(jnp.array([-0.5, -0.5, -0.5])),
        lambda_mat=jnp.eye(3),
        diffusion=jnp.eye(3) * 0.3,
        manifest_var=jnp.eye(3) * 0.1,
        manifest_means=jnp.zeros(3),
        t0_means=jnp.zeros(3),
        t0_var=jnp.eye(3),
        diffusion_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
            DistributionFamily.STUDENT_T,
        ],
        first_pass_rb=first_pass_rb,
        latent_names=["g0", "g1", "s0"],
        manifest_names=["yg0", "yg1", "ys0"],
    )


class TestStage4bInferenceStructurePayload:
    @pytest.mark.parametrize(
        "label, spec_factory, obs_support, expected_reason",
        [
            (
                "disabled_in_spec",
                lambda: _make_separable_spec(first_pass_rb=False),
                None,
                "disabled_in_spec",
            ),
            (
                "interval_summary_support",
                _make_separable_spec,
                _support_runtime,
                "interval_summary_support",
            ),
            (
                "no_executable_partition",
                lambda: SSMSpec(
                    n_latent=2,
                    n_manifest=2,
                    drift=jnp.diag(jnp.array([-0.5, -0.3])),
                    lambda_mat=jnp.ones((2, 2)),
                    diffusion_dists=[
                        DistributionFamily.GAUSSIAN,
                        DistributionFamily.STUDENT_T,
                    ],
                    latent_names=["g0", "s0"],
                    manifest_names=["y0", "y1"],
                ),
                None,
                "no_executable_partition",
            ),
        ],
        ids=["disabled_in_spec", "interval_summary", "no_executable_partition"],
    )
    def test_payload_marks_first_pass_inactive(
        self, label, spec_factory, obs_support, expected_reason
    ):
        spec = spec_factory()
        model = _make_model(spec, observation_support=obs_support() if obs_support else None)
        plan = plan_inference_structure(
            model.spec,
            likelihood=model.likelihood,
            observation_support=model.observation_support,
        )
        payload = build_inference_structure_payload(spec, plan)

        assert payload["likelihood_path"] == "particle"
        assert payload["auto_method"] == "laplace_em"
        assert payload["first_pass_rb"]["status"] == "inactive"
        assert payload["first_pass_rb"]["inactive_reason"] == expected_reason

    def test_payload_includes_active_first_pass_assignments(self):
        spec = _make_separable_spec()
        model = _make_model(spec)
        plan = plan_inference_structure(
            model.spec,
            likelihood=model.likelihood,
            observation_support=model.observation_support,
        )
        payload = build_inference_structure_payload(spec, plan)

        assert payload["likelihood_path"] == "composed"
        assert payload["auto_method"] == "laplace_em"
        assert payload["first_pass_rb"]["status"] == "active"
        assert payload["first_pass_rb"]["inactive_reason"] is None
        assert payload["first_pass_rb"]["latent_variables"] == [
            {"name": "g0", "method": "kalman"},
            {"name": "g1", "method": "kalman"},
            {"name": "s0", "method": "particle"},
        ]
        assert payload["first_pass_rb"]["obs_variables"] == [
            {"name": "yg0", "method": "kalman"},
            {"name": "yg1", "method": "kalman"},
            {"name": "ys0", "method": "particle"},
        ]

    def test_t_rule_failure_keeps_nested_stage4b_contract(self, monkeypatch):
        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(spec),
        )

        class StubTRule:
            satisfies = False
            n_free_params = 12
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        assert set(result.keys()) == {"parametric_id", "inference_structure"}
        assert result["parametric_id"]["checked"] is True
        assert result["parametric_id"]["t_rule"]["satisfies"] is False
        assert result["inference_structure"]["likelihood_path"] == "composed"

    def test_profile_diagnostics_aggregate_into_contract_summary(self, monkeypatch):
        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(spec),
        )

        class StubTRule:
            satisfies = True
            n_free_params = 4
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        class StubSensitivityResult:
            def __init__(self):
                self.singular_values = [1.0, 0.1]
                self.condition_number = 10.0
                self.per_parameter = [
                    {
                        "parameter": "drift_offdiag_pop[0]",
                        "interpretable_parameter": "beta_g1_g0",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    },
                    {
                        "parameter": "lambda_free",
                        "interpretable_parameter": "lambda_free",
                        "sensitivity_norm": 0.1,
                        "effective_sv": 1e-5,
                        "sv_status": "warn",
                        "normalized_effective_sv": 1e-5,
                        "normalized_sv_status": "warn",
                        "identifiable": True,
                    },
                ]
                self.n_draws = 8
                self.n_observations = 12
                self.n_parameters = 2

            def print_report(self):
                return None

        class StubProfileLikelihoodResult:
            def __init__(self):
                self.parameter_names = ["drift_offdiag_pop[0]", "lambda_free"]
                self.parameter_profiles = {
                    "drift_offdiag_pop[0]": {
                        "grid_con": [-1.0, 0.0, 1.0],
                        "profile_ll": jnp.array([-0.1, 0.0, -0.1]),
                    },
                    "lambda_free": {
                        "grid_con": [-1.0, 0.0, 1.0],
                        "profile_ll": jnp.array([-3.0, 0.0, -0.2]),
                    },
                }
                self.threshold = 1.92

            def print_report(self):
                return None

            def summary(self):
                return {
                    "drift_offdiag_pop[0]": "structurally_unidentifiable",
                    "lambda_free": "practically_unidentifiable",
                }

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.get_stage4b_sweep_context",
            lambda *_args, **_kwargs: SimpleNamespace(
                scalar_names=["drift_offdiag_pop[0]", "lambda_free"]
            ),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.output_sensitivity_analysis",
            lambda *_args, **_kwargs: StubSensitivityResult(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.graph_analysis.kalman_block_profile_indices",
            lambda *_args, **_kwargs: [0, 1],
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.profile_likelihood",
            lambda *_args, **_kwargs: StubProfileLikelihoodResult(),
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        pid = result["parametric_id"]
        assert pid["summary"] == {
            "structural_issues": ["drift_offdiag_pop[0]"],
            "boundary_issues": [],
            "weak_params": ["lambda_free"],
        }
        assert [entry["classification"] for entry in pid["per_param_classification"]] == [
            "structurally_unidentifiable",
            "practically_unidentifiable",
        ]
        assert "n_parameters" not in pid
        assert "parameter_names" not in pid

    def test_skips_profile_likelihood_when_only_nuisance_sensitivity_fails(self, monkeypatch):
        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(spec),
        )

        class StubTRule:
            satisfies = True
            n_free_params = 4
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        class StubSensitivityResult:
            def __init__(self):
                self.singular_values = [1.0, 0.1]
                self.condition_number = 10.0
                self.per_parameter = [
                    {
                        "parameter": "diffusion_diag_pop[0]",
                        "interpretable_parameter": "sigma_g0",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    },
                    {
                        "parameter": "drift_diag_pop[0]",
                        "interpretable_parameter": "rho_g0",
                        "sensitivity_norm": 0.1,
                        "effective_sv": 1e-5,
                        "sv_status": "warn",
                        "normalized_effective_sv": 1e-5,
                        "normalized_sv_status": "warn",
                        "identifiable": True,
                    },
                ]
                self.n_draws = 8
                self.n_observations = 12
                self.n_parameters = 2

            def print_report(self):
                return None

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.get_stage4b_sweep_context",
            lambda *_args, **_kwargs: SimpleNamespace(
                scalar_names=["diffusion_diag_pop[0]", "drift_diag_pop[0]"]
            ),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.output_sensitivity_analysis",
            lambda *_args, **_kwargs: StubSensitivityResult(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.profile_likelihood",
            lambda *_args, **_kwargs: pytest.fail("profile_likelihood should be skipped"),
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        pid = result["parametric_id"]
        assert pid["per_param_classification"] is None
        assert pid["threshold"] is None
        assert pid["summary"]["weak_params"] == [
            "diffusion_diag_pop[0]",
            "drift_diag_pop[0]",
        ]

    def test_profiles_only_substantive_raw_failures_after_sensitivity_gate(self, monkeypatch):
        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(spec),
        )

        class StubTRule:
            satisfies = True
            n_free_params = 4
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        class StubSensitivityResult:
            def __init__(self):
                self.singular_values = [1.0, 0.1]
                self.condition_number = 10.0
                self.per_parameter = [
                    {
                        "parameter": "diffusion_diag_pop[0]",
                        "interpretable_parameter": "sigma_g0",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    },
                    {
                        "parameter": "drift_offdiag_pop[0]",
                        "interpretable_parameter": "beta_g1_g0",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    },
                    {
                        "parameter": "lambda_free",
                        "interpretable_parameter": "lambda_free",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    },
                ]
                self.n_draws = 8
                self.n_observations = 12
                self.n_parameters = 3

            def print_report(self):
                return None

        class StubProfileLikelihoodResult:
            def __init__(self):
                self.parameter_names = ["lambda_free"]
                self.parameter_profiles = {
                    "lambda_free": {
                        "grid_con": [-1.0, 0.0, 1.0],
                        "profile_ll": jnp.array([-3.0, 0.0, -0.2]),
                    },
                }
                self.threshold = 1.92

            def print_report(self):
                return None

            def summary(self):
                return {
                    "lambda_free": "practically_unidentifiable",
                }

        captured = {}

        def _profile_likelihood_stub(*_args, **kwargs):
            captured["profile_indices"] = kwargs["profile_indices"]
            return StubProfileLikelihoodResult()

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.get_stage4b_sweep_context",
            lambda *_args, **_kwargs: SimpleNamespace(
                scalar_names=["diffusion_diag_pop[0]", "drift_offdiag_pop[0]", "lambda_free"]
            ),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.output_sensitivity_analysis",
            lambda *_args, **_kwargs: StubSensitivityResult(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.graph_analysis.kalman_block_profile_indices",
            lambda *_args, **_kwargs: [2],
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.profile_likelihood",
            _profile_likelihood_stub,
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        pid = result["parametric_id"]
        assert captured["profile_indices"] == [2]
        assert [entry["name"] for entry in pid["per_param_classification"]] == ["lambda_free"]
        assert pid["threshold"] == pytest.approx(1.92)

    def test_skips_profile_likelihood_on_particle_only_path(self, monkeypatch):
        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(
                spec,
                observation_support=_support_runtime(),
            ),
        )

        class StubTRule:
            satisfies = True
            n_free_params = 4
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        class StubSensitivityResult:
            def __init__(self):
                self.singular_values = [1.0, 0.1]
                self.condition_number = 10.0
                self.per_parameter = [
                    {
                        "parameter": "lambda_free",
                        "interpretable_parameter": "lambda_free",
                        "sensitivity_norm": 0.01,
                        "effective_sv": 1e-8,
                        "sv_status": "fail",
                        "normalized_effective_sv": 1e-8,
                        "normalized_sv_status": "fail",
                        "identifiable": False,
                    }
                ]
                self.n_draws = 8
                self.n_observations = 8
                self.n_parameters = 1

            def print_report(self):
                return None

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.get_stage4b_sweep_context",
            lambda *_args, **_kwargs: SimpleNamespace(scalar_names=["lambda_free"]),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.output_sensitivity_analysis",
            lambda *_args, **_kwargs: StubSensitivityResult(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.profile_likelihood",
            lambda *_args, **_kwargs: pytest.fail(
                "profile_likelihood should be skipped on particle-only paths"
            ),
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        pid = result["parametric_id"]
        assert pid["per_param_classification"] is None
        assert pid["threshold"] is None
        assert pid["summary"]["weak_params"] == ["lambda_free"]

    def test_unsupported_sensitivity_screen_does_not_emit_false_weak_params(self, monkeypatch):
        from causal_ssm_agent.utils.parametric_id import OutputSensitivityUnsupportedError

        spec = _make_separable_spec()
        model = _make_model(spec)
        runtime = SimpleNamespace(
            builder=SimpleNamespace(_model=model),
            observations=jnp.zeros((4, spec.n_manifest)),
            times=jnp.arange(4.0),
            inference_structure=plan_inference_structure(
                spec,
                observation_support=_support_runtime(),
            ),
        )

        class StubTRule:
            satisfies = True
            n_free_params = 4
            n_moments = 8

            def print_report(self):
                return None

            def model_dump(self):
                return {
                    "satisfies": self.satisfies,
                    "n_free_params": self.n_free_params,
                    "n_moments": self.n_moments,
                }

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_builder.prepare_model_runtime",
            lambda **_: runtime,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.check_t_rule",
            lambda *_args, **_kwargs: StubTRule(),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.get_stage4b_sweep_context",
            lambda *_args, **_kwargs: SimpleNamespace(scalar_names=["lambda_free"]),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.output_sensitivity_analysis",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OutputSensitivityUnsupportedError("unsupported")
            ),
        )
        monkeypatch.setattr(
            "causal_ssm_agent.utils.parametric_id.profile_likelihood",
            lambda *_args, **_kwargs: pytest.fail(
                "profile_likelihood should be skipped on particle-only paths"
            ),
        )

        result = parametric_id_task.fn(
            data_for_model=pl.DataFrame(),
        )

        pid = result["parametric_id"]
        assert pid["sensitivity_analysis"] is None
        assert pid["per_param_classification"] is None
        assert pid["threshold"] is None
        assert pid["summary"]["weak_params"] == []

    def test_stage4b_demotes_t_rule_failure_to_warning(self, monkeypatch):
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4b.flow.stage4b_parametric_id_flow",
            lambda *_args, **_kwargs: {
                "parametric_id": {
                    "checked": True,
                    "t_rule": {
                        "satisfies": False,
                        "n_free_params": 12,
                        "n_moments": 8,
                    },
                    "summary": {
                        "structural_issues": [],
                        "boundary_issues": [],
                        "weak_params": [],
                    },
                }
            },
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4b.flow.load_parquet",
            lambda _path: pl.DataFrame(),
        )

        result = stage4b({}, {"_data_for_model_path": "unused"})

        assert result["outcome"] == "warn"
