"""Tests for vector-field transition construction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.discretization import (
    discretize_at_states_batched,
    discretize_system_with_inputs_batched,
)
from nof1_causal_lab.models.ssm.dynamics.edges import (
    DenseLinear,
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    StateDecay,
    StateIntercept,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import (
    VectorField,
    VectorFieldArgs,
)
from nof1_causal_lab.models.ssm.inference.methods.map import _build_map_laplace_bundle
from nof1_causal_lab.models.ssm.inference.targets.affine import derive_affine_dynamics
from nof1_causal_lab.models.ssm.inference.targets.base import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    InitialStateParams,
    MeasurementParams,
    RuntimeDynamics,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood
from nof1_causal_lab.models.ssm.inference.targets.laplace.point import (
    _transition_start_linearization_states,
)
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions
from nof1_causal_lab.models.ssm.model import SSMModel, SSMSpec
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass


def _constant_runtime_dynamics(
    *,
    input_effect: jnp.ndarray | None = None,
) -> RuntimeDynamics:
    drift = jnp.array(
        [
            [-0.30, 0.05, 0.00],
            [0.00, -0.40, 0.02],
            [0.00, 0.00, -0.50],
        ],
        dtype=jnp.float32,
    )
    cint = jnp.array([0.03, -0.04, 0.02], dtype=jnp.float32)
    return RuntimeDynamics(
        vector_field=VectorField(
            n_latent=3,
            components=(
                DenseLinear(),
                DiagonalDecay(),
                StateDecay(target=2),
                Intercept(),
                StateIntercept(target=0),
                LinearEdge(source=0, target=1),
            ),
        ),
        vf_params=(
            {"drift": drift},
            {"decay": jnp.array([0.20, 0.15, 0.10], dtype=jnp.float32)},
            {"decay": jnp.array(0.05, dtype=jnp.float32)},
            {"cint": cint},
            {"cint": jnp.array(0.07, dtype=jnp.float32)},
            {"weight": jnp.array(-0.10, dtype=jnp.float32)},
        ),
        diffusion_cov=jnp.diag(jnp.array([0.08, 0.06, 0.04], dtype=jnp.float32)),
        input_effect=input_effect,
    )


def _trajectory_runtime_dynamics() -> RuntimeDynamics:
    return RuntimeDynamics(
        vector_field=VectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                HillEdge(source=0, target=1),
            ),
        ),
        vf_params=(
            {"decay": jnp.array([0.35, 0.45], dtype=jnp.float32)},
            {
                "Emax": jnp.array(0.80, dtype=jnp.float32),
                "EC50": jnp.array(1.20, dtype=jnp.float32),
                "n": jnp.array(2.0, dtype=jnp.float32),
            },
        ),
        diffusion_cov=jnp.diag(jnp.array([0.05, 0.07], dtype=jnp.float32)),
    )


def _interval_mean_support_runtime() -> ObservationSupportRuntime:
    return ObservationSupportRuntime(
        anchor_times=np.array([0.0, 0.50, 1.25], dtype=np.float64),
        manifest_names=["mean_signal"],
        support_kinds=["interval"],
        summary_operators=["mean"],
        anchor_policies=["support_end"],
        observation_windows=["previous_interval"],
        support_start_times=np.array([[np.nan], [0.0], [0.50]], dtype=np.float64),
        support_end_times=np.array([[np.nan], [0.50], [1.25]], dtype=np.float64),
        interval_prev_coeffs=np.array([[[0.0]], [[0.25]], [[0.375]]], dtype=np.float64),
        interval_curr_coeffs=np.array([[[0.0]], [[0.25]], [[0.375]]], dtype=np.float64),
        interval_weights=np.array([[[0.0]], [[0.50]], [[0.75]]], dtype=np.float64),
        emission_slot_indices=np.array([[-1], [0], [0]], dtype=np.int64),
    )


def _long_interval_mean_support_runtime(n_time: int) -> ObservationSupportRuntime:
    anchor_times = np.arange(n_time, dtype=np.float64)
    support_start = np.full((n_time, 1), np.nan, dtype=np.float64)
    support_end = np.full((n_time, 1), np.nan, dtype=np.float64)
    interval_prev = np.zeros((n_time, 1, 1), dtype=np.float64)
    interval_curr = np.zeros((n_time, 1, 1), dtype=np.float64)
    interval_weights = np.zeros((n_time, 1, 1), dtype=np.float64)
    emission_slots = np.full((n_time, 1), -1, dtype=np.int64)
    support_start[1:, 0] = anchor_times[:-1]
    support_end[1:, 0] = anchor_times[1:]
    interval_prev[1:, 0, 0] = 0.5
    interval_curr[1:, 0, 0] = 0.5
    interval_weights[1:, 0, 0] = 1.0
    emission_slots[1:, 0] = 0
    return ObservationSupportRuntime(
        anchor_times=anchor_times,
        manifest_names=["mean_signal"],
        support_kinds=["interval"],
        summary_operators=["mean"],
        anchor_policies=["support_end"],
        observation_windows=["previous_interval"],
        support_start_times=support_start,
        support_end_times=support_end,
        interval_prev_coeffs=interval_prev,
        interval_curr_coeffs=interval_curr,
        interval_weights=interval_weights,
        emission_slot_indices=emission_slots,
    )


def _nonlinear_point_ssm_spec() -> SSMSpec:
    n_latent = 2
    n_manifest = 1
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=DynamicsSpec(
            n_latent=n_latent,
            components=(
                DiagonalDecaySpec(),
                HillEdgeSpec(
                    source=0,
                    target=1,
                ),
            ),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=np.zeros((n_latent, n_latent), dtype=bool),
            diffusion_chol_template=jnp.diag(jnp.array([0.22, 0.26], dtype=jnp.float32)),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=np.zeros(n_manifest, dtype=bool),
            template=jnp.array([0.05], dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=np.zeros(n_manifest, dtype=bool),
            template=jnp.array([[0.45]], dtype=jnp.float32),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.zeros(n_latent, dtype=bool),
            template=jnp.array([0.20, -0.10], dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.zeros(n_latent, dtype=bool),
            correlation_support=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.diag(jnp.array([0.55, 0.65], dtype=jnp.float32)),
        ),
        input_effect_block=SparseMatrixBlockSpec(
            n_rows=n_latent,
            n_cols=0,
            free_support=np.zeros((n_latent, 0), dtype=bool),
            template=jnp.zeros((n_latent, 0), dtype=jnp.float32),
            free_site_name="input_effect_free",
            det_site_name="input_effect",
            support=SupportClass.REAL,
            site_kind=SiteKind.INPUT_EFFECT,
            assembly_group="input_effect",
            fixed_spec_field="input_effect",
            priors_field="input_effect",
        ),
        static_state_sd_block=SparseVectorBlockSpec(
            n=0,
            free_support=np.zeros(0, dtype=bool),
            template=jnp.zeros(0, dtype=jnp.float32),
            free_site_name="static_state_sd_free",
            det_site_name="static_state_sds",
            support=SupportClass.POSITIVE,
            site_kind=SiteKind.STATIC_STATE_SD,
            assembly_group="t0",
            fixed_spec_field="static_state_sds",
            priors_field="static_state_sd",
        ),
        diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
    )


def test_constant_vector_field_local_linearization_matches_affine_view():
    dynamics = _constant_runtime_dynamics()
    affine = derive_affine_dynamics(dynamics)
    args = VectorFieldArgs(params=dynamics.vf_params, intervention=Intervention.none())
    x_lin = jnp.array([0.70, -0.20, 0.40], dtype=jnp.float32)

    drift, cint = dynamics.vector_field.linearize(x_lin, args)

    np.testing.assert_allclose(drift, affine.drift, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cint, affine.cint, rtol=1e-6, atol=1e-6)


def test_constant_vector_field_local_discretization_matches_affine_discretization():
    dynamics = _constant_runtime_dynamics()
    affine = derive_affine_dynamics(dynamics)
    args = VectorFieldArgs(params=dynamics.vf_params, intervention=Intervention.none())
    time_intervals = jnp.array([0.20, 0.50, 1.10], dtype=jnp.float32)
    linearization_states = jnp.array(
        [
            [0.70, -0.20, 0.40],
            [0.20, 0.10, -0.30],
            [1.10, -0.70, 0.05],
        ],
        dtype=jnp.float32,
    )

    local_Ad, local_Qd, local_cd = discretize_at_states_batched(
        dynamics.vector_field,
        linearization_states,
        args,
        dynamics.diffusion_cov,
        time_intervals,
    )
    affine_Ad, affine_Qd, affine_cd = discretize_system_with_inputs_batched(
        affine.drift,
        affine.diffusion_cov,
        affine.cint,
        affine.input_effect,
        None,
        time_intervals,
    )

    np.testing.assert_allclose(local_Ad, affine_Ad, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(local_Qd, affine_Qd, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(local_cd, affine_cd, rtol=2e-5, atol=2e-5)


def test_transition_builder_preserves_affine_input_discretization():
    input_effect = jnp.array(
        [
            [0.10, -0.20],
            [0.00, 0.30],
            [-0.05, 0.02],
        ],
        dtype=jnp.float32,
    )
    dynamics = _constant_runtime_dynamics(input_effect=input_effect)
    affine = derive_affine_dynamics(dynamics)
    time_intervals = jnp.array([0.20, 0.50, 1.10], dtype=jnp.float32)
    transition_inputs = jnp.array(
        [
            [1.0, 0.0],
            [0.5, -1.0],
            [0.0, 0.25],
        ],
        dtype=jnp.float32,
    )

    transitions = build_discrete_transitions(
        dynamics,
        time_intervals,
        transition_inputs=transition_inputs,
    )
    expected_Ad, expected_Qd, expected_cd = discretize_system_with_inputs_batched(
        affine.drift,
        affine.diffusion_cov,
        affine.cint,
        affine.input_effect,
        transition_inputs,
        time_intervals,
    )

    np.testing.assert_allclose(transitions.Ad, expected_Ad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transitions.Qd, expected_Qd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transitions.cd, expected_cd, rtol=1e-6, atol=1e-6)


def test_transition_builder_requires_states_for_trajectory_dependent_dynamics():
    dynamics = _trajectory_runtime_dynamics()
    time_intervals = jnp.array([0.20, 0.50, 1.10], dtype=jnp.float32)

    with pytest.raises(ValueError, match="linearization_states"):
        build_discrete_transitions(dynamics, time_intervals)


def test_transition_builder_discretizes_trajectory_dependent_dynamics_at_states():
    dynamics = _trajectory_runtime_dynamics()
    time_intervals = jnp.array([0.20, 0.50, 1.10], dtype=jnp.float32)
    linearization_states = jnp.array(
        [
            [0.30, 0.10],
            [1.00, -0.20],
            [1.60, 0.50],
        ],
        dtype=jnp.float32,
    )
    args = VectorFieldArgs(params=dynamics.vf_params, intervention=Intervention.none())

    transitions = build_discrete_transitions(
        dynamics,
        time_intervals,
        linearization_states=linearization_states,
    )
    expected_Ad, expected_Qd, expected_cd = discretize_at_states_batched(
        dynamics.vector_field,
        linearization_states,
        args,
        dynamics.diffusion_cov,
        time_intervals,
    )

    np.testing.assert_allclose(transitions.Ad, expected_Ad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transitions.Qd, expected_Qd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transitions.cd, expected_cd, rtol=1e-6, atol=1e-6)


def test_laplace_point_backend_accepts_trajectory_dependent_dynamics():
    dynamics = _trajectory_runtime_dynamics()
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
        n_ieks_iters=2,
    )
    measurement_params = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial_state = InitialStateParams(
        mean=jnp.array([0.20, -0.10], dtype=jnp.float32),
        cov=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.array([[0.05], [0.20], [0.12]], dtype=jnp.float32)
    time_intervals = jnp.array([0.01, 0.50, 0.75], dtype=jnp.float32)

    log_lik, aux = backend.compute_log_likelihood_with_aux(
        dynamics,
        measurement_params,
        initial_state,
        observations,
        time_intervals,
    )

    assert bool(jnp.isfinite(log_lik))
    assert aux["latent_mode"].shape == (3, 2)
    assert int(aux["n_iterations"]) == 2


def test_point_dynamic_linearization_states_use_interval_starts():
    init_mean = jnp.array([0.20, -0.10], dtype=jnp.float32)
    latent_trajectory = jnp.array(
        [
            [0.30, 0.10],
            [1.00, -0.20],
            [1.60, 0.50],
        ],
        dtype=jnp.float32,
    )

    states = _transition_start_linearization_states(latent_trajectory, init_mean)

    expected = jnp.array(
        [
            [0.20, -0.10],
            [0.30, 0.10],
            [1.00, -0.20],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(states, expected)


def test_laplace_point_backend_differentiates_trajectory_dependent_dynamics():
    base_dynamics = _trajectory_runtime_dynamics()
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
        n_ieks_iters=2,
    )
    measurement_params = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial_state = InitialStateParams(
        mean=jnp.array([0.20, -0.10], dtype=jnp.float32),
        cov=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.array([[0.05], [0.20], [0.12]], dtype=jnp.float32)
    time_intervals = jnp.array([0.01, 0.50, 0.75], dtype=jnp.float32)

    def _objective(emax: jnp.ndarray) -> jnp.ndarray:
        hill_params = dict(base_dynamics.vf_params[1])
        hill_params["Emax"] = emax
        dynamics = base_dynamics._replace(vf_params=(base_dynamics.vf_params[0], hill_params))
        return backend.compute_log_likelihood(
            dynamics,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
        )

    grad = jax.grad(_objective)(jnp.array(0.80, dtype=jnp.float32))

    assert bool(jnp.isfinite(grad))


def test_laplace_interval_support_accepts_trajectory_dependent_dynamics_dense():
    dynamics = _trajectory_runtime_dynamics()
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
        n_ieks_iters=2,
        observation_support=_interval_mean_support_runtime(),
    )
    measurement_params = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial_state = InitialStateParams(
        mean=jnp.array([0.20, -0.10], dtype=jnp.float32),
        cov=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.array([[jnp.nan], [0.11], [0.18]], dtype=jnp.float32)
    time_intervals = jnp.array([0.01, 0.50, 0.75], dtype=jnp.float32)

    log_lik, aux = backend.compute_log_likelihood_with_aux(
        dynamics,
        measurement_params,
        initial_state,
        observations,
        time_intervals,
    )

    assert bool(jnp.isfinite(log_lik))
    assert aux["latent_mode"].shape == (3, 2)
    assert int(aux["solver_kind"]) == LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT


def test_laplace_interval_support_differentiates_trajectory_dependent_dynamics_dense():
    base_dynamics = _trajectory_runtime_dynamics()
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
        n_ieks_iters=1,
        observation_support=_interval_mean_support_runtime(),
    )
    measurement_params = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial_state = InitialStateParams(
        mean=jnp.array([0.20, -0.10], dtype=jnp.float32),
        cov=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.array([[jnp.nan], [0.11], [0.18]], dtype=jnp.float32)
    time_intervals = jnp.array([0.01, 0.50, 0.75], dtype=jnp.float32)

    def _objective(emax: jnp.ndarray) -> jnp.ndarray:
        hill_params = dict(base_dynamics.vf_params[1])
        hill_params["Emax"] = emax
        dynamics = base_dynamics._replace(vf_params=(base_dynamics.vf_params[0], hill_params))
        return backend.compute_log_likelihood(
            dynamics,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
        )

    grad = jax.grad(_objective)(jnp.array(0.80, dtype=jnp.float32))

    assert bool(jnp.isfinite(grad))


def test_laplace_interval_support_uses_banded_dynamic_path_for_large_problem():
    base_dynamics = _trajectory_runtime_dynamics()
    n_time = 81
    backend = LaplaceLikelihood(
        n_latent=2,
        n_manifest=1,
        manifest_dists=[DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.IDENTITY],
        n_ieks_iters=1,
        observation_support=_long_interval_mean_support_runtime(n_time),
    )
    measurement_params = MeasurementParams(
        lambda_mat=jnp.array([[1.0, 0.25]], dtype=jnp.float32),
        manifest_means=jnp.array([0.05], dtype=jnp.float32),
        manifest_cov=jnp.array([[0.20]], dtype=jnp.float32),
    )
    initial_state = InitialStateParams(
        mean=jnp.array([0.20, -0.10], dtype=jnp.float32),
        cov=jnp.diag(jnp.array([0.30, 0.40], dtype=jnp.float32)),
    )
    observations = jnp.linspace(0.05, 0.25, n_time, dtype=jnp.float32)[:, None]
    observations = observations.at[0, 0].set(jnp.nan)
    time_intervals = jnp.ones((n_time,), dtype=jnp.float32).at[0].set(0.01)

    log_lik, aux = backend.compute_log_likelihood_with_aux(
        base_dynamics,
        measurement_params,
        initial_state,
        observations,
        time_intervals,
    )

    def _objective(emax: jnp.ndarray) -> jnp.ndarray:
        hill_params = dict(base_dynamics.vf_params[1])
        hill_params["Emax"] = emax
        dynamics = base_dynamics._replace(vf_params=(base_dynamics.vf_params[0], hill_params))
        return backend.compute_log_likelihood(
            dynamics,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
        )

    grad = jax.grad(_objective)(jnp.array(0.80, dtype=jnp.float32))

    assert bool(jnp.isfinite(log_lik))
    assert bool(jnp.isfinite(grad))
    assert aux["latent_mode"].shape == (n_time, 2)
    assert int(aux["solver_kind"]) == LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS


def test_ssm_model_laplace_path_accepts_nonlinear_point_dynamics():
    model = SSMModel(_nonlinear_point_ssm_spec())
    observations = jnp.array([[0.05], [0.20], [0.12]], dtype=jnp.float32)
    times = jnp.array([0.0, 0.50, 1.25], dtype=jnp.float32)
    backend = model.make_laplace_backend(n_ieks_iters=2)
    condition_data = {
        "vf_0_decay": jnp.array([0.35, 0.45], dtype=jnp.float32),
        "vf_1_Emax": jnp.array(0.80, dtype=jnp.float32),
        "vf_1_EC50": jnp.array(1.20, dtype=jnp.float32),
        "vf_1_n": jnp.array(2.0, dtype=jnp.float32),
    }

    conditioned_model = handlers.condition(model.model, data=condition_data)
    trace = handlers.trace(handlers.seed(conditioned_model, rng_seed=0)).get_trace(
        observations,
        times,
        likelihood_backend=backend,
    )

    assert bool(jnp.isfinite(trace["log_likelihood"]["fn"].log_factor))


def test_map_laplace_bundle_evaluates_nonlinear_point_model():
    model = SSMModel(_nonlinear_point_ssm_spec())
    observations = jnp.array([[0.05], [0.20], [0.12]], dtype=jnp.float32)
    times = jnp.array([0.0, 0.50, 1.25], dtype=jnp.float32)
    backend = model.make_laplace_backend(n_ieks_iters=2)

    bundle = _build_map_laplace_bundle(
        model,
        observations,
        times,
        jax.random.PRNGKey(0),
        backend,
        reparam=None,
    )
    log_posterior = bundle["log_posterior_fn"](
        bundle["flat_example"],
        observations,
        times,
    )

    assert bundle["dim"] > 0
    assert bool(jnp.isfinite(log_posterior))
