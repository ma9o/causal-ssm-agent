"""Comprehensive tests for SSM inference backends.

Tests cover:
1. ParticleLikelihood: finite likelihood, determinism, gradient flow
2. SSMAdapter: observation models (Gaussian, Poisson, Student-t, Gamma)
3. Parameter recovery: simulate → fit() → check credible intervals
4. Hierarchical likelihood robustness
5. Edge cases and builder wiring
6. SVI inference backend

Test Matrix:
| Model Class                    | Noise Family         | Test Type          |
|--------------------------------|----------------------|--------------------|
| Linear-Gaussian                | gaussian             | LL finite, grad    |
| Linear, Poisson obs            | poisson              | Param recovery     |
| Linear, Student-t obs          | student_t            | Param recovery     |
| Linear, Student-t process      | student_t diffusion  | Variance calib     |
| High-dim, Poisson + Student-t  | poisson + student_t  | Stress test        |
"""

import functools

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from causal_ssm_agent.models.ssm import DistributionFamily, InferenceResult, SSMModel, fit
from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference import _apply_reparam, _eval_model
from causal_ssm_agent.models.ssm.inference.engines.tempered_smc import run_tempered_smc
from causal_ssm_agent.models.ssm.inference.methods.dpf import DPFLikelihood, ProposalNetwork
from causal_ssm_agent.models.ssm.inference.methods.laplace_em import (
    LaplaceLikelihood,
    _build_ieks_system,
    _dense_support_laplace_log_lik,
    _ieks_smooth,
    _infer_support_groups,
    _should_use_dense_support_laplace,
    _solve_block_tridiagonal,
)
from causal_ssm_agent.models.ssm.inference.methods.nuts_da import _da_model
from causal_ssm_agent.models.ssm.inference.methods.pgas import _csmc_sweep_support_aware
from causal_ssm_agent.models.ssm.inference.methods.structured_vi import StructuredVILikelihood
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.ssm.inference.targets.emissions import get_mean_param_log_prob_fn
from causal_ssm_agent.models.ssm.inference.targets.kalman import KalmanLikelihood
from causal_ssm_agent.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
    compile_measurement_semantics,
)
from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood, SSMAdapter
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    compile_observation_operator,
    expected_observation_mean,
    trajectory_observation_log_probs,
)
from causal_ssm_agent.models.ssm.inference.utils import _build_eval_fns, _discover_sites
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from causal_ssm_agent.orchestrator.schemas_model import LinkFunction
from tests.ssm_test_utils import make_ssm_spec


def _support_runtime(**kwargs) -> ObservationSupportRuntime:
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int32)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime(**kwargs)


def test_compile_observation_operator_keeps_point_support_without_interval_summary_mode():
    support = _support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["pulse"],
        support_kinds=["point"],
        observation_windows=[None],
        support_start_times=np.array([[0.0], [1.0]]),
        support_end_times=np.array([[0.0], [1.0]]),
        interval_prev_coeffs=np.zeros((2, 1)),
        interval_curr_coeffs=np.zeros((2, 1)),
        interval_weights=np.zeros((2, 1)),
    )

    operator = compile_observation_operator(support)

    assert operator.support_kind_codes is not None
    assert operator.summary_operator_codes is not None
    assert operator.requires_interval_summary_handling is False
    assert operator.interval_summary_indices == ()
    np.testing.assert_array_equal(
        np.asarray(operator.point_like_mask(jnp.float32)),
        np.array([1.0], dtype=np.float32),
    )


def test_expected_observation_mean_dispatches_by_summary_operator():
    support = _support_runtime(
        anchor_times=np.array([0.0]),
        manifest_names=["y_sum", "y_count", "y_mean", "y_std", "y_last"],
        support_kinds=["interval", "interval", "interval", "interval", "point"],
        summary_operators=["sum", "count", "mean", "std", "last"],
        observation_windows=["1d", "1d", "1d", "1d", None],
        support_start_times=np.zeros((1, 5)),
        support_end_times=np.zeros((1, 5)),
        interval_prev_coeffs=np.zeros((1, 5)),
        interval_curr_coeffs=np.zeros((1, 5)),
        interval_weights=np.ones((1, 5)),
    )
    operator = compile_observation_operator(support)

    expected = expected_observation_mean(
        response_t=jnp.array([7.0, 8.0, 9.0, 10.0, 11.0]),
        obs_sum=jnp.array([6.0, 4.0, 9.0, 8.0, 0.0]),
        obs_sumsq=jnp.array([36.0, 16.0, 41.0, 34.0, 0.0]),
        obs_weight=jnp.array([1.0, 1.0, 3.0, 2.0, 1.0]),
        summary_operator_codes=operator.summary_operator_codes,
    )

    np.testing.assert_allclose(expected, np.array([6.0, 4.0, 3.0, 1.0, 11.0]))


# =============================================================================
# ParticleLikelihood: Core Functionality
# =============================================================================


class TestLaplaceEMBlockSolver:
    """Numerical checks for the block-tridiagonal IEKS rewrite."""

    @staticmethod
    def _dense_block_matrix(
        lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray
    ) -> jnp.ndarray:
        n_blocks, block_dim = diag.shape[:2]
        mat = np.zeros((n_blocks * block_dim, n_blocks * block_dim), dtype=np.float64)
        lower_np = np.asarray(lower)
        diag_np = np.asarray(diag)
        upper_np = np.asarray(upper)
        for i in range(n_blocks):
            row = slice(i * block_dim, (i + 1) * block_dim)
            mat[row, row] = diag_np[i]
            if i > 0:
                prev = slice((i - 1) * block_dim, i * block_dim)
                mat[row, prev] = lower_np[i]
            if i + 1 < n_blocks:
                nxt = slice((i + 1) * block_dim, (i + 2) * block_dim)
                mat[row, nxt] = upper_np[i]
        return jnp.asarray(mat)

    def test_block_solver_matches_dense_reference(self):
        """Recursive block solver should agree with a dense solve on SPD systems."""
        key = random.PRNGKey(7)
        n_blocks = 7
        block_dim = 3

        key, diag_key, lower_key, x_key = random.split(key, 4)
        raw_diag = random.normal(diag_key, (n_blocks, block_dim, block_dim))
        diag = jnp.matmul(raw_diag, jnp.swapaxes(raw_diag, -1, -2)) + 4.0 * jnp.eye(block_dim)
        lower = jnp.zeros((n_blocks, block_dim, block_dim))
        lower_noise = random.normal(lower_key, (n_blocks - 1, block_dim, block_dim)) * 0.05
        lower = lower.at[1:].set(lower_noise)
        upper = jnp.zeros_like(lower).at[:-1].set(jnp.swapaxes(lower[1:], -1, -2))

        dense = self._dense_block_matrix(lower, diag, upper)
        x_true = random.normal(x_key, (n_blocks, block_dim))
        rhs = (dense @ x_true.reshape(-1)).reshape(n_blocks, block_dim)

        x_solved = _solve_block_tridiagonal(lower, diag, upper, rhs)
        np.testing.assert_allclose(x_solved, x_true, atol=1e-5, rtol=1e-5)

    def test_gaussian_ieks_mode_matches_dense_system(self):
        """For Gaussian observations, one IEKS step should equal the exact mode solve."""
        key = random.PRNGKey(11)
        T = 5
        D = 2
        M = 2

        observations = random.normal(key, (T, M)) * 0.2
        obs_mask = jnp.ones((T, M), dtype=bool)
        Ad = jnp.broadcast_to(jnp.array([[0.92, 0.05], [0.02, 0.88]]), (T, D, D))
        Qd = jnp.broadcast_to(jnp.array([[0.15, 0.01], [0.01, 0.12]]), (T, D, D))
        cd = jnp.zeros((T, D))
        H = jnp.array([[1.0, 0.1], [0.2, 1.0]])
        d = jnp.array([0.0, 0.1])
        R = jnp.array([[0.2, 0.02], [0.02, 0.25]])
        init_mean = jnp.array([0.05, -0.1])
        init_cov = jnp.array([[0.8, 0.05], [0.05, 0.7]])

        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
        )

        z_smooth, log_lik = _ieks_smooth(
            observations,
            obs_mask,
            Ad,
            Qd,
            cd,
            H,
            d,
            R,
            init_mean,
            init_cov,
            obs_kernel,
            n_ieks_iters=1,
        )

        z_init = jnp.broadcast_to(init_mean, (T, D))
        grads, J_t = jax.vmap(
            lambda y_t, z_t, mask_t: obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
        )(observations, z_init, obs_mask.astype(jnp.float32))
        tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_init, grads)
        lower, diag, upper, rhs = _build_ieks_system(Ad, Qd, cd, init_mean, init_cov, J_t, tilde_y)
        dense = self._dense_block_matrix(lower, diag, upper)
        z_dense = jnp.linalg.solve(dense, rhs.reshape(-1)).reshape(T, D)

        np.testing.assert_allclose(z_smooth, z_dense, atol=1e-5, rtol=1e-5)
        assert jnp.isfinite(log_lik).all()


class TestParticleLikelihoodCore:
    """Test ParticleLikelihood core functionality."""

    @pytest.fixture
    def linear_gaussian_params(self):
        """Standard test parameters for 2D linear-Gaussian model."""
        return {
            "ct_params": CTParams(
                drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
                diffusion_cov=jnp.array([[0.1, 0.02], [0.02, 0.1]]),
                cint=jnp.array([0.0, 0.0]),
            ),
            "meas_params": MeasurementParams(
                lambda_mat=jnp.eye(2),
                manifest_means=jnp.zeros(2),
                manifest_cov=jnp.eye(2) * 0.1,
            ),
            "init_params": InitialStateParams(
                mean=jnp.zeros(2),
                cov=jnp.eye(2),
            ),
        }

    @pytest.fixture
    def simple_observations(self):
        """Simple test observations and time intervals."""
        T = 15
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.5
        return observations, time_intervals

    def test_pf_produces_finite_likelihood(self, linear_gaussian_params, simple_observations):
        """PF log-likelihood should be finite for reasonable parameters."""
        observations, time_intervals = simple_observations

        backend = ParticleLikelihood(n_latent=2, n_manifest=2, n_particles=200)
        ll = backend.compute_log_likelihood(
            linear_gaussian_params["ct_params"],
            linear_gaussian_params["meas_params"],
            linear_gaussian_params["init_params"],
            observations,
            time_intervals,
        )

        assert jnp.all(jnp.isfinite(ll)), f"PF produced non-finite: {ll}"

    def test_pf_varies_with_params(self, simple_observations):
        """PF likelihood should vary with different parameters."""
        observations, time_intervals = simple_observations

        drift_values = [
            jnp.array([[-0.3, 0.0], [0.0, -0.3]]),
            jnp.array([[-0.5, 0.1], [0.1, -0.5]]),
            jnp.array([[-0.8, 0.0], [0.0, -0.8]]),
        ]

        likelihoods = []
        for drift in drift_values:
            ct_params = CTParams(
                drift=drift,
                diffusion_cov=jnp.eye(2) * 0.1,
                cint=jnp.zeros(2),
            )
            meas_params = MeasurementParams(
                lambda_mat=jnp.eye(2),
                manifest_means=jnp.zeros(2),
                manifest_cov=jnp.eye(2) * 0.1,
            )
            init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

            backend = ParticleLikelihood(
                n_latent=2,
                n_manifest=2,
                n_particles=200,
                rng_key=random.PRNGKey(42),
            )
            ll = backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                observations,
                time_intervals,
            )
            likelihoods.append(float(ll[-1]))

        assert all(np.isfinite(ll) for ll in likelihoods)
        assert len({round(ll, 2) for ll in likelihoods}) > 1

    def test_pf_support_aware_window_average_produces_finite_likelihood(self):
        """Interval-summary observations should run through the support-aware PF path."""
        ct_params = CTParams(
            drift=jnp.array([[-0.4]]),
            diffusion_cov=jnp.array([[0.1]]),
            cint=jnp.array([0.0]),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]]),
            manifest_means=jnp.array([0.0]),
            manifest_cov=jnp.array([[0.2]]),
        )
        init = InitialStateParams(mean=jnp.array([0.0]), cov=jnp.array([[1.0]]))
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        backend = ParticleLikelihood(
            n_latent=1,
            n_manifest=1,
            n_particles=80,
            rng_key=random.PRNGKey(0),
            observation_support=support,
            block_rb=False,
        )
        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert support.requires_interval_summary_handling is True
        assert jnp.all(jnp.isfinite(ll))

    def test_pf_support_aware_window_average_handles_mixed_diffusion(self):
        """Support-aware PF should keep mixed diffusion families instead of collapsing them."""
        ct_params = CTParams(
            drift=jnp.array([[-0.4, 0.1], [0.0, -0.3]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1, 0.02], [0.02, 0.15]], dtype=jnp.float32),
            cint=jnp.zeros(2, dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0, 0.5]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.zeros(2, dtype=jnp.float32),
            cov=jnp.eye(2, dtype=jnp.float32),
        )
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        backend = ParticleLikelihood(
            n_latent=2,
            n_manifest=1,
            n_particles=80,
            rng_key=random.PRNGKey(1),
            diffusion_dist=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
            observation_support=support,
            block_rb=False,
        )
        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params={"proc_df": 5.0},
        )

        assert backend.diffusion_dist == "mixed"
        assert jnp.all(jnp.isfinite(ll))


class TestSupportAwareTrajectoryObservationLogProb:
    def test_window_average_matches_manual_gaussian_average(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )
        latent = jnp.array([[1.0], [3.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [2.0]], dtype=jnp.float32)
        obs_mask = ~jnp.isnan(observations)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d_meas = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)
        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=R,
        )
        mean_log_prob_fn = get_mean_param_log_prob_fn(DistributionFamily.GAUSSIAN)

        ll = trajectory_observation_log_probs(
            latent,
            observations,
            obs_mask,
            H,
            d_meas,
            R,
            obs_kernel,
            mean_log_prob_fn,
            support,
        )

        manual = mean_log_prob_fn(
            jnp.array([2.0], dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert ll[0] == pytest.approx(0.0)
        assert ll[1] == pytest.approx(float(manual))

    def test_overlapping_window_averages_match_manual_gaussian_means(self):
        support = _support_runtime(
            anchor_times=np.array([-2.0, -1.0, 0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [-2.0], [-1.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [0.0], [1.0]]),
            interval_prev_coeffs=np.array(
                [
                    [[0.0, 0.0]],
                    [[0.5, 0.0]],
                    [[0.5, 0.5]],
                    [[0.0, 0.5]],
                ]
            ),
            interval_curr_coeffs=np.array(
                [
                    [[0.0, 0.0]],
                    [[0.5, 0.0]],
                    [[0.5, 0.5]],
                    [[0.0, 0.5]],
                ]
            ),
            interval_weights=np.array(
                [
                    [[0.0, 0.0]],
                    [[1.0, 0.0]],
                    [[1.0, 1.0]],
                    [[0.0, 1.0]],
                ]
            ),
            emission_slot_indices=np.array([[-1], [-1], [0], [1]], dtype=np.int32),
        )
        latent = jnp.array([[1.0], [3.0], [5.0], [7.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [jnp.nan], [3.0], [5.0]], dtype=jnp.float32)
        obs_mask = ~jnp.isnan(observations)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d_meas = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)
        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=R,
        )
        mean_log_prob_fn = get_mean_param_log_prob_fn(DistributionFamily.GAUSSIAN)

        ll = trajectory_observation_log_probs(
            latent,
            observations,
            obs_mask,
            H,
            d_meas,
            R,
            obs_kernel,
            mean_log_prob_fn,
            support,
        )

        manual_0 = mean_log_prob_fn(
            jnp.array([3.0], dtype=jnp.float32),
            jnp.array([3.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )
        manual_1 = mean_log_prob_fn(
            jnp.array([5.0], dtype=jnp.float32),
            jnp.array([5.0], dtype=jnp.float32),
            R,
            jnp.array([1.0], dtype=jnp.float32),
        )

        assert ll[0] == pytest.approx(0.0)
        assert ll[1] == pytest.approx(0.0)
        assert ll[2] == pytest.approx(float(manual_0))
        assert ll[3] == pytest.approx(float(manual_1))


class TestStructuredVISupportAware:
    def test_structured_vi_backend_handles_window_average(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = StructuredVILikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_vi_steps=2,
            n_mc_samples=1,
            vi_lr=0.01,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert jnp.isfinite(ll)


class TestLaplaceSupportAware:
    def test_infer_support_groups_ignores_reused_slot_history(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0], [np.nan], [2.0], [np.nan], [4.0]]),
            support_end_times=np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan], [5.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [-1], [0], [-1], [0]], dtype=np.int32),
        )

        windows, bandwidth = _infer_support_groups(support)

        np.testing.assert_array_equal(np.asarray(windows.anchor_indices), np.array([1, 3, 5]))
        np.testing.assert_array_equal(np.asarray(windows.start_indices), np.array([0, 2, 4]))
        np.testing.assert_array_equal(np.asarray(windows.state_lens), np.array([2, 2, 2]))
        assert windows.max_state_len == 2
        assert bandwidth == 1

    def test_dense_support_path_threshold_matches_smallgolden_regime(self):
        assert _should_use_dense_support_laplace(n_time=10, n_latent=12) is True
        assert _should_use_dense_support_laplace(n_time=20, n_latent=12) is False

    def test_laplace_backend_prefers_dense_path_for_small_support_models(self, monkeypatch):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        calls: list[str] = []

        def _fake_dense(*args, **kwargs):
            calls.append("dense")
            return jnp.array(-1.0, dtype=jnp.float32)

        def _fake_banded(*args, **kwargs):
            calls.append("banded")
            return jnp.array(-2.0, dtype=jnp.float32)

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.laplace_em._dense_support_laplace_log_lik",
            _fake_dense,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.laplace_em._support_aware_ieks_log_lik",
            _fake_banded,
        )

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert float(ll) == pytest.approx(-1.0)
        assert calls == ["dense"]

    def test_laplace_backend_handles_window_average(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert jnp.isfinite(ll)

    def test_laplace_banded_matches_dense_reference(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            time_intervals,
        )
        if cd.ndim == 1:
            cd = cd[:, None]
        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=meas_params.manifest_cov,
        )
        mean_log_prob_fn = get_mean_param_log_prob_fn(DistributionFamily.GAUSSIAN)

        banded = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        dense = _dense_support_laplace_log_lik(
            jnp.nan_to_num(observations, nan=0.0),
            ~jnp.isnan(observations),
            Ad,
            Qd,
            cd,
            meas_params.lambda_mat,
            meas_params.manifest_means,
            meas_params.manifest_cov,
            init.mean,
            init.cov,
            obs_kernel,
            mean_log_prob_fn,
            support,
            2,
        )

        assert banded == pytest.approx(float(dense), rel=1e-2, abs=1e-2)


class TestDPFSupportAware:
    def test_dpf_backend_handles_window_average(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = DPFLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_particles=16,
            proposal_net=ProposalNetwork(1, 1, key=random.PRNGKey(1)),
            rng_key=random.PRNGKey(2),
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert jnp.isfinite(ll)


class TestPGASSupportAware:
    def test_support_aware_csmc_sweep_produces_finite_trajectory(self):
        support = _support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )
        ref_traj = jnp.array([[0.0], [0.1]], dtype=jnp.float32)
        Ad = jnp.broadcast_to(jnp.array([[1.0]], dtype=jnp.float32), (2, 1, 1))
        Qd = jnp.broadcast_to(jnp.array([[0.1]], dtype=jnp.float32), (2, 1, 1))
        chol_Qd = jnp.broadcast_to(jnp.array([[jnp.sqrt(0.1)]], dtype=jnp.float32), (2, 1, 1))
        cd = jnp.zeros((2, 1), dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        obs_mask_float = (~jnp.isnan(observations)).astype(jnp.float32)
        measurement_semantics = compile_measurement_semantics(
            [DistributionFamily.GAUSSIAN],
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
            manifest_links=[LinkFunction.IDENTITY],
            observation_support=support,
        )

        traj, _ = _csmc_sweep_support_aware(
            random.PRNGKey(0),
            ref_traj,
            Ad,
            Qd,
            chol_Qd,
            cd,
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([[1.0]], dtype=jnp.float32),
            observations,
            obs_mask_float,
            4,
            0.0,
            measurement_semantics,
            jnp.array([[1.0]], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([[0.2]], dtype=jnp.float32),
        )

        assert traj.shape == (2, 1)
        assert jnp.all(jnp.isfinite(traj))


class TestNutsDASupportAware:
    def test_da_model_emits_support_aware_factor(self):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dists=[DistributionFamily.GAUSSIAN],
        )
        model = SSMModel(spec)
        model.set_observation_support(
            _support_runtime(
                anchor_times=np.array([0.0, 1.0]),
                manifest_names=["avg_signal"],
                support_kinds=["interval"],
                observation_windows=["1d"],
                support_start_times=np.array([[np.nan], [0.0]]),
                support_end_times=np.array([[np.nan], [1.0]]),
                interval_prev_coeffs=np.array([[0.0], [0.5]]),
                interval_curr_coeffs=np.array([[0.0], [0.5]]),
                interval_weights=np.array([[0.0], [1.0]]),
            )
        )
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(
                lambda obs, ts: _da_model(model, obs, ts, centered=True)
            ).get_trace(observations, times)

        assert "obs_support_loglik" in trace


class TestSupportAwareMethodSmoke:
    @staticmethod
    def _support() -> ObservationSupportRuntime:
        return _support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )

    @staticmethod
    def _model() -> SSMModel:
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dists=[DistributionFamily.GAUSSIAN],
        )
        model = SSMModel(spec, likelihood="particle", n_particles=16)
        model.set_observation_support(TestSupportAwareMethodSmoke._support())
        return model

    def test_fit_pgas_support_aware_smoke(self):
        model = self._model()
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="pgas",
            n_outer=2,
            n_warmup=1,
            n_csmc_particles=4,
            n_mh_steps=1,
            n_leapfrog=1,
            svi_warmstart=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "pgas"

    def test_fit_dpf_support_aware_smoke(self):
        model = self._model()
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="dpf",
            n_outer=2,
            n_csmc_particles=4,
            n_mh_steps=1,
            n_pf_particles=8,
            n_train_seqs=2,
            n_train_steps=1,
            n_particles_train=4,
            T_train=2,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "dpf"


# =============================================================================
# SSMAdapter: Observation Models
# =============================================================================


class TestParticleMissingData:
    """Tests for Gaussian observation masking in PF adapter."""

    def test_missing_dimension_not_penalized(self):
        """Missing dims should not incur huge log-det penalties."""
        n_latent, n_manifest = 1, 2
        adapter = SSMAdapter(
            n_latent,
            n_manifest,
            manifest_dists=[DistributionFamily.GAUSSIAN] * n_manifest,
            diffusion_dist=DistributionFamily.GAUSSIAN,
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        )

        params = {
            "lambda_mat": jnp.array([[1.0], [1.0]]),
            "manifest_means": jnp.array([0.0, 0.0]),
            "manifest_cov": jnp.diag(jnp.array([0.5, 0.5])),
        }
        x = jnp.array([0.0])
        y = jnp.array([1.0, -2.0])
        obs_mask = jnp.array([True, False])

        ll = adapter.observation_log_prob(y, x, params, obs_mask)

        # Manual univariate logpdf for observed dimension
        sigma2 = 0.5
        resid = y[0]
        manual = -0.5 * (jnp.log(2 * jnp.pi * sigma2) + (resid**2) / sigma2)

        assert jnp.isfinite(ll)
        assert jnp.allclose(ll, manual, atol=1e-5), f"{ll} vs {manual}"


class TestKalmanMissingData:
    """Regression tests for missing-data handling in the exact Kalman path."""

    @staticmethod
    def _simple_params(n_manifest: int) -> tuple[CTParams, MeasurementParams, InitialStateParams]:
        ct_params = CTParams(
            drift=jnp.array([[-0.5]]),
            diffusion_cov=jnp.array([[1e-6]]),
            cint=jnp.array([0.0]),
        )
        measurement_params = MeasurementParams(
            lambda_mat=jnp.ones((n_manifest, 1)),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.5,
        )
        initial_state = InitialStateParams(
            mean=jnp.array([0.0]),
            cov=jnp.array([[1.0]]),
        )
        return ct_params, measurement_params, initial_state

    def test_fully_missing_timestep_has_zero_increment(self):
        """A timestep with no observations should add zero log-likelihood."""
        backend = KalmanLikelihood(n_latent=1, n_manifest=1)
        ct_params, measurement_params, initial_state = self._simple_params(n_manifest=1)

        lnc = backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            jnp.array([[1.0], [jnp.nan]]),
            jnp.array([1.0, 1.0]),
        )

        ll_per_timestep = jnp.diff(lnc, prepend=0.0)
        assert jnp.isclose(ll_per_timestep[1], 0.0, atol=1e-5), ll_per_timestep

    def test_missing_channel_matches_observed_subsystem(self):
        """Masking out one manifest should match the corresponding 1D Kalman model."""
        dt = jnp.array([1.0])
        init_mask = InitialStateParams(mean=jnp.array([0.0]), cov=jnp.array([[1.0]]))
        ct_mask = CTParams(
            drift=jnp.array([[-0.5]]), diffusion_cov=jnp.array([[1e-6]]), cint=jnp.array([0.0])
        )

        masked_backend = KalmanLikelihood(n_latent=1, n_manifest=2)
        masked_meas = MeasurementParams(
            lambda_mat=jnp.array([[1.0], [1.0]]),
            manifest_means=jnp.array([0.0, 0.0]),
            manifest_cov=jnp.diag(jnp.array([0.5, 0.5])),
        )
        ll_from_nan = masked_backend.compute_log_likelihood(
            ct_mask,
            masked_meas,
            init_mask,
            jnp.array([[1.0, jnp.nan]]),
            dt,
        )
        ll_from_explicit_mask = masked_backend.compute_log_likelihood(
            ct_mask,
            masked_meas,
            init_mask,
            jnp.array([[1.0, 999.0]]),
            dt,
            obs_mask=jnp.array([[True, False]]),
        )

        single_backend = KalmanLikelihood(n_latent=1, n_manifest=1)
        single_meas = MeasurementParams(
            lambda_mat=jnp.array([[1.0]]),
            manifest_means=jnp.array([0.0]),
            manifest_cov=jnp.array([[0.5]]),
        )
        ll_single = single_backend.compute_log_likelihood(
            ct_mask,
            single_meas,
            init_mask,
            jnp.array([[1.0]]),
            dt,
        )

        assert jnp.allclose(ll_from_nan, ll_from_explicit_mask, atol=1e-4)
        assert jnp.allclose(ll_from_nan, ll_single, atol=1e-5)

    def test_skipped_empty_tick_matches_explicit_missing_timestep(self):
        """A fully unobserved clock tick should be equivalent to a longer dt gap."""
        backend = KalmanLikelihood(n_latent=1, n_manifest=1)
        ct_params, measurement_params, initial_state = self._simple_params(n_manifest=1)

        ll_skipped_tick = backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            jnp.array([[1.0], [2.0]]),
            jnp.array([1.0, 2.0]),
        )
        ll_explicit_missing_tick = backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            jnp.array([[1.0], [jnp.nan], [2.0]]),
            jnp.array([1.0, 1.0, 1.0]),
        )

        assert jnp.allclose(ll_skipped_tick[-1], ll_explicit_missing_tick[-1], atol=1e-5)


class TestStudentTProcessNoise:
    """Tests for Student-t process noise variance calibration."""

    @pytest.mark.slow
    def test_student_t_process_noise_variance_matches_qd(self):
        """Student-t noise should match Qd variance (df > 2)."""
        from causal_ssm_agent.models.ssm.discretization import discretize_system

        n_latent, n_manifest = 1, 1
        df = 5.0
        dt = 1.0

        drift = jnp.array([[-0.5]])
        diffusion_cov = jnp.array([[0.3**2]])
        _, Qd, _ = discretize_system(drift, diffusion_cov, None, dt)

        adapter = SSMAdapter(
            n_latent,
            n_manifest,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            diffusion_dist=DistributionFamily.STUDENT_T,
            manifest_links=[LinkFunction.IDENTITY],
        )

        params = {
            "drift": drift,
            "diffusion_cov": diffusion_cov,
            "lambda_mat": jnp.eye(1),
            "manifest_means": jnp.zeros(1),
            "manifest_cov": jnp.eye(1),
            "t0_mean": jnp.zeros(n_latent),
            "t0_cov": jnp.eye(n_latent),
            "proc_df": df,
        }

        key = random.PRNGKey(0)
        n_samples = 400
        keys = random.split(key, n_samples)
        x_prev = jnp.zeros(n_latent)

        samples = jax.vmap(lambda k: adapter.transition_sample(k, x_prev, params, dt))(keys)
        sample_var = jnp.var(samples, axis=0)[0]
        target_var = Qd[0, 0]

        assert jnp.isfinite(sample_var)
        assert jnp.allclose(sample_var, target_var, rtol=0.25, atol=0.05), (
            f"Sample var {float(sample_var):.4f} vs Qd {float(target_var):.4f}"
        )


# =============================================================================
# Parameter Recovery Tests (PF + NUTS)
# =============================================================================


class TestParameterRecoveryPF:
    """Parameter recovery tests using particle filter + NUTS.

    Simulate from known parameters → run PF+NUTS → verify true params
    fall within 90% credible intervals.
    """

    @pytest.mark.slow
    def test_drift_diagonal_recovery(self):
        """Recover drift diagonal parameters from simulated data."""
        true_drift_diag = jnp.array([-0.6, -0.9])

        key = random.PRNGKey(42)
        T = 60
        n_latent = 2
        dt = 0.5
        discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))
        process_noise = 0.3

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * process_noise
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.1
        times = jnp.arange(T, dtype=float) * dt

        spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
        )
        model = SSMModel(spec, n_particles=200)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="nuts",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = result.get_samples()
        drift_diag_samples = samples["drift_diag_pop"]

        for i, true_val in enumerate(true_drift_diag):
            posterior_mean = jnp.mean(drift_diag_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.5, (
                f"Drift[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )

    @pytest.mark.slow
    def test_diffusion_recovery(self):
        """Recover diffusion parameters from simulated data."""
        true_diffusion_diag = jnp.array([0.4, 0.4])
        true_drift_diag = jnp.array([-0.5, -0.5])

        key = random.PRNGKey(123)
        T = 80
        n_latent = 2
        dt = 0.5

        discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * true_diffusion_diag
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.05
        times = jnp.arange(T, dtype=float) * dt

        spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
        )
        model = SSMModel(spec, n_particles=200)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="nuts",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = result.get_samples()
        diffusion_samples = samples["diffusion_diag_pop"]

        for i, true_val in enumerate(true_diffusion_diag):
            posterior_mean = jnp.mean(diffusion_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.4, (
                f"Diffusion[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )


class TestHighDimNonlinear:
    """Test PF+NUTS on high-dimensional nonlinear model."""

    @pytest.mark.slow
    def test_high_dimensional_poisson_pf_nuts(self):
        """PF+NUTS should produce finite results on 6D Poisson model."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm.discretization import discretize_system

        n_latent, n_manifest = 6, 6
        T = 30
        dt = 0.4
        true_drift = -0.4
        proc_df = 4.0

        # Stable drift with mild cross-coupling
        drift = true_drift * jnp.eye(n_latent) + 0.05 * (
            jnp.ones((n_latent, n_latent)) - jnp.eye(n_latent)
        )
        diffusion_cov = jnp.eye(n_latent) * 0.2**2

        # Nonlinear observation: Poisson with log-link
        key = random.PRNGKey(123)
        key, key_cross, key_noise, key_obs = random.split(key, 4)
        lambda_base = 0.7 * jnp.eye(n_manifest, n_latent)
        cross = random.normal(key_cross, (n_manifest, n_latent)) * 0.05
        lambda_mat = lambda_base + cross
        manifest_means = jnp.ones(n_manifest) * jnp.log(5.0)

        # Simulate latent states with Student-t process noise
        Ad, Qd, _ = discretize_system(drift, diffusion_cov, None, dt)
        chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key_noise, key_z, key_chi2 = random.split(key_noise, 3)
            z = random.normal(key_z, (n_latent,))
            chi2 = random.gamma(key_chi2, proc_df / 2.0) * 2.0
            scale = jnp.sqrt((proc_df - 2.0) / chi2)
            noise = chol @ (z * scale)
            states.append(Ad @ states[-1] + noise)
        latent = jnp.stack(states)

        # Generate Poisson observations with log-link
        eta = jax.vmap(lambda x: lambda_mat @ x + manifest_means)(latent)
        eta = jnp.clip(eta, -10.0, 6.0)
        rates = jnp.exp(eta)
        observations = random.poisson(key_obs, rates).astype(jnp.float32)

        # Test PF likelihood is finite
        time_intervals = jnp.ones(T) * dt

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=None)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=300,
            manifest_dists=["poisson"],
            diffusion_dist="student_t",
            manifest_links=["log"],
        )
        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params={"proc_df": proc_df},
        )

        assert jnp.all(jnp.isfinite(ll)), f"Non-finite LL on high-dim model: {ll}"


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Test edge cases and robustness with ParticleLikelihood."""

    def test_single_observation(self):
        """Handle single observation gracefully."""
        ct_params = CTParams(
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=None,
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        observations = jnp.array([[0.5, -0.3]])
        time_intervals = jnp.array([1.0])

        backend = ParticleLikelihood(n_latent=2, n_manifest=2, n_particles=100)
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.all(jnp.isfinite(ll))

    def test_irregular_time_intervals(self):
        """Handle irregular time intervals."""
        ct_params = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.array([0.1, -0.1]),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        observations = jnp.array(
            [
                [0.1, 0.2],
                [0.3, 0.1],
                [0.2, 0.4],
                [0.5, 0.3],
                [0.4, 0.5],
            ]
        )
        time_intervals = jnp.array([0.1, 0.5, 0.2, 1.0, 0.3])

        backend = ParticleLikelihood(n_latent=2, n_manifest=2, n_particles=100)
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.all(jnp.isfinite(ll))

    def test_higher_dimensional_system(self):
        """Test 4-dimensional latent system."""
        n_latent = 4
        n_manifest = 4
        T = 30

        key = random.PRNGKey(42)
        observations = random.normal(key, (T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

        ct_params = CTParams(
            drift=jnp.diag(jnp.array([-0.5, -0.6, -0.7, -0.8])),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=None,
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=200,
        )
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.all(jnp.isfinite(ll))

    def test_non_identity_lambda(self):
        """Test with non-identity factor loading matrix."""
        n_latent = 2
        n_manifest = 3
        T = 20

        key = random.PRNGKey(42)
        observations = random.normal(key, (T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

        lambda_mat = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        )

        ct_params = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=None,
        )
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=200,
        )
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.all(jnp.isfinite(ll))


# =============================================================================
# fit() Integration Tests
# =============================================================================


class TestInferenceCaching:
    """Low-risk caching behavior for default inference helpers."""

    @staticmethod
    def _identity_transform():
        class _IdentityTransform:
            @staticmethod
            def inv(value):
                return value

        return _IdentityTransform()

    def test_model_reuses_backend_instances(self):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec)

        backend_a = model.make_likelihood_backend()
        backend_b = model.make_likelihood_backend()
        laplace_a = model.make_laplace_backend(3)
        laplace_b = model.make_laplace_backend(3)
        laplace_c = model.make_laplace_backend(5)

        assert backend_a is backend_b
        assert laplace_a is laplace_b
        assert laplace_a is not laplace_c

    def test_discover_sites_uses_dummy_backend_for_structural_trace(self):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec)
        observations = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        class _ExplodingBackend:
            def compute_log_likelihood(self, *_args, **_kwargs):
                raise AssertionError("site discovery should not evaluate the real likelihood")

        site_info = _discover_sites(
            model,
            observations,
            times,
            random.PRNGKey(0),
            _ExplodingBackend(),
        )

        assert "drift_diag_pop" in site_info
        assert "manifest_var_diag" in site_info

    def test_tempered_smc_reuses_bundle_cache_without_reparam(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec)
        observations = jnp.zeros((2, 1), dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)
        backend = object()
        build_calls = {"count": 0}

        class _ZeroDistribution:
            @staticmethod
            def sample(_key, shape):
                return jnp.zeros(shape, dtype=jnp.float32)

        def fake_bundle(*_args, **_kwargs):
            build_calls["count"] += 1
            return {
                "dim": 1,
                "site_info": {
                    "drift_diag_pop": {
                        "distribution": _ZeroDistribution(),
                        "transform": self._identity_transform(),
                        "value": jnp.asarray([0.0], dtype=jnp.float32),
                    }
                },
                "unravel_fn": lambda z: {"drift_diag_pop": z},
                "batch_lik_val_and_grad": lambda particles: (
                    jnp.zeros((particles.shape[0],), dtype=particles.dtype),
                    jnp.zeros_like(particles),
                ),
                "mutate_batch_jit": lambda _rng_key, particles, _beta, _eps, _chol_mass: (
                    particles,
                    jnp.zeros((particles.shape[0],), dtype=jnp.int32),
                ),
                "mutate_batch_wastefree_jit": lambda _rng_key, particles, _beta, _eps, _chol_mass: (
                    particles[:, None, :],
                    jnp.zeros((particles.shape[0],), dtype=jnp.int32),
                ),
                "pilot_adapt_jit": lambda rng_key, particles, eps, _chol_mass, _target_accept: (
                    rng_key,
                    particles,
                    eps,
                    jnp.asarray(True),
                    jnp.asarray(0.5, dtype=eps.dtype),
                ),
                "standard_mutation_rounds_jit": (
                    lambda rng_key, particles, _beta_k, eps, _chol_mass, _target_accept: (
                        rng_key,
                        particles,
                        eps,
                        jnp.asarray(0.5, dtype=eps.dtype),
                        jnp.asarray(1, dtype=jnp.int32),
                    )
                ),
            }

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.engines.tempered_smc._build_tempered_smc_bundle",
            fake_bundle,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.engines.tempered_smc.extract_constrained_samples",
            lambda *_args, **_kwargs: {"drift_diag_pop": jnp.zeros((1, 1), dtype=jnp.float32)},
        )

        for _ in range(2):
            result = run_tempered_smc(
                model,
                observations,
                times,
                n_outer=0,
                n_warmup=0,
                n_csmc_particles=2,
                n_mh_steps=1,
                n_leapfrog=1,
                likelihood_backend=backend,
            )
            assert result.method == "tempered_smc"

        assert build_calls["count"] == 1
        assert len(model._artifact_cache) == 1
        [cache_key] = model._artifact_cache.keys()
        assert cache_key[:6] == (
            "tempered_smc_core",
            id(backend),
            tuple(observations.shape),
            tuple(times.shape),
            1,
            1,
        )
        assert cache_key[-2:] == ("none", None)


class TestPureJaxLikelihoodEvaluator:
    """The pure-JAX likelihood path should match NumPyro replay exactly."""

    @staticmethod
    def _build_poisson_case():
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dists=[DistributionFamily.POISSON],
            manifest_links=[LinkFunction.LOG],
            manifest_means=jnp.array([jnp.log(4.0)], dtype=jnp.float32),
        )
        model = SSMModel(spec, n_particles=40)
        observations = jnp.array([[4.0], [3.0], [5.0], [6.0]], dtype=jnp.float32)
        times = jnp.arange(observations.shape[0], dtype=jnp.float32) * 0.5
        return model, observations, times

    @staticmethod
    def _assert_log_likelihood_match(reparam) -> None:
        model, observations, times = TestPureJaxLikelihoodEvaluator._build_poisson_case()
        backend = model.make_likelihood_backend()
        site_info = _discover_sites(
            model,
            observations,
            times,
            random.PRNGKey(0),
            backend,
            reparam=reparam,
        )
        example_unc = {
            name: info["transform"].inv(info["value"]) for name, info in site_info.items()
        }
        z0, unravel_fn = ravel_pytree(example_unc)
        log_lik_fn, _ = _build_eval_fns(
            model,
            observations,
            times,
            site_info,
            unravel_fn,
            likelihood_backend=backend,
            reparam=reparam,
        )

        base_model_fn = functools.partial(model.model, likelihood_backend=backend)
        replay_model_fn = _apply_reparam(base_model_fn, reparam)
        constrained = {
            name: site_info[name]["transform"](unravel_fn(z0)[name]) for name in site_info
        }
        replay_ll, _ = _eval_model(replay_model_fn, constrained, observations, times)

        np.testing.assert_allclose(
            np.asarray(log_lik_fn(z0)),
            np.asarray(replay_ll),
            rtol=1e-6,
            atol=1e-6,
        )
        grads = jax.grad(log_lik_fn)(z0)
        assert jnp.all(jnp.isfinite(grads))

    def test_log_likelihood_matches_model_replay_without_reparam(self):
        self._assert_log_likelihood_match(reparam=None)

    def test_log_likelihood_matches_model_replay_with_fixed_autoreparam(self):
        self._assert_log_likelihood_match(reparam=AutoReparam(centered=0.0))


# =============================================================================
# SVI-specific Tests
# =============================================================================


class TestSVIBackend:
    """Tests specific to SVI inference backend."""

    def test_svi_rejects_nonfinite_losses(self, monkeypatch):
        """SVI should fail fast instead of returning a numerically invalid fit."""
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="kalman")
        observations = jnp.zeros((4, 1), dtype=jnp.float32)
        times = jnp.arange(4, dtype=jnp.float32)

        class FakeSVI:
            def __init__(self, *_args, **_kwargs):
                pass

            def init(self, *_args, **_kwargs):
                return 0

            def update(self, state, *_args, **_kwargs):
                losses = jnp.array([1.0, jnp.nan], dtype=jnp.float32)
                return state + 1, losses[state]

            def get_params(self, *_args, **_kwargs):
                return {"loc": jnp.array([0.0], dtype=jnp.float32)}

        monkeypatch.setattr("causal_ssm_agent.models.ssm.inference.SVI", FakeSVI)

        with pytest.raises(FloatingPointError, match="non-finite losses"):
            fit(
                model,
                observations=observations,
                times=times,
                method="svi",
                num_steps=2,
                num_samples=2,
            )

    def test_svi_rejects_nonfinite_posterior_samples(self, monkeypatch):
        """SVI should fail fast when guide predictive samples contain NaNs."""
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="kalman")
        observations = jnp.zeros((4, 1), dtype=jnp.float32)
        times = jnp.arange(4, dtype=jnp.float32)

        class FakeSVI:
            def __init__(self, *_args, **_kwargs):
                pass

            def init(self, *_args, **_kwargs):
                return 0

            def update(self, state, *_args, **_kwargs):
                losses = jnp.array([1.0, 0.5], dtype=jnp.float32)
                return state + 1, losses[state]

            def get_params(self, *_args, **_kwargs):
                return {"loc": jnp.array([0.0], dtype=jnp.float32)}

        class FakePredictive:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, *_args, **_kwargs):
                return {"drift": jnp.array([[[jnp.nan]]], dtype=jnp.float32)}

        monkeypatch.setattr("causal_ssm_agent.models.ssm.inference.SVI", FakeSVI)
        monkeypatch.setattr("causal_ssm_agent.models.ssm.inference.Predictive", FakePredictive)

        with pytest.raises(FloatingPointError, match="non-finite posterior samples"):
            fit(
                model,
                observations=observations,
                times=times,
                method="svi",
                num_steps=2,
                num_samples=1,
            )

    @pytest.mark.slow
    def test_svi_losses_decrease(self):
        """ELBO loss should generally decrease during SVI training.

        Uses 1D model because multi-latent SVI has gradient instability
        from eigvals backward pass in the drift stability penalty.
        """
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="kalman")

        T = 15
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 1)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        result = fit(
            model,
            observations=observations,
            times=times,
            method="svi",
            num_steps=200,
            num_samples=10,
        )

        losses = result.diagnostics["losses"]
        # Compare first 10% mean to last 10% mean
        n = len(losses)
        early_mean = float(jnp.mean(losses[: n // 10]))
        late_mean = float(jnp.mean(losses[-n // 10 :]))
        assert late_mean < early_mean, (
            f"SVI loss did not decrease: early={early_mean:.1f}, late={late_mean:.1f}"
        )


# =============================================================================
# SVI Parameter Recovery
# =============================================================================


class TestAutoMethodConfigRouting:
    """Regression tests for backend-specific config propagation under method='auto'."""

    @staticmethod
    def _non_point_support() -> ObservationSupportRuntime:
        return _support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["y"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )

    def test_auto_routes_non_point_support_to_laplace_em(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="particle")
        model.set_observation_support(self._non_point_support())
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_laplace_em(_model, _observations, _times, **kwargs):
            return InferenceResult(
                _samples={"drift_diag_pop": jnp.zeros((1, 1), dtype=jnp.float32)},
                method="laplace_em",
                diagnostics={"kwargs": kwargs},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.laplace_em.fit_laplace_em",
            fake_fit_laplace_em,
        )

        result = fit(model, observations=observations, times=times, method="auto")

        assert result.method == "laplace_em"

    def test_non_point_support_allows_laplace_em(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="particle")
        model.set_observation_support(self._non_point_support())
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_laplace_em(_model, _observations, _times, **kwargs):
            return InferenceResult(
                _samples={"drift_diag_pop": jnp.zeros((1, 1), dtype=jnp.float32)},
                method="laplace_em",
                diagnostics={"kwargs": kwargs},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.laplace_em.fit_laplace_em",
            fake_fit_laplace_em,
        )

        result = fit(model, observations=observations, times=times, method="laplace_em")

        assert result.method == "laplace_em"

    @pytest.mark.parametrize(
        ("method", "target"),
        [
            ("dpf", "causal_ssm_agent.models.ssm.inference.methods.dpf.fit_dpf"),
            ("pgas", "causal_ssm_agent.models.ssm.inference.methods.pgas.fit_pgas"),
        ],
    )
    def test_non_point_support_allows_particle_trajectory_methods(
        self,
        monkeypatch,
        method,
        target,
    ):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="particle")
        model.set_observation_support(self._non_point_support())
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_method(_model, _observations, _times, **kwargs):
            return InferenceResult(
                _samples={"drift_diag_pop": jnp.zeros((1, 1), dtype=jnp.float32)},
                method=method,
                diagnostics={"kwargs": kwargs},
            )

        monkeypatch.setattr(target, fake_fit_method)

        result = fit(model, observations=observations, times=times, method=method)

        assert result.method == method

    def test_auto_passes_smc_config_to_laplace_em(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, likelihood="kalman")
        observations = jnp.zeros((3, 1), dtype=jnp.float32)
        times = jnp.arange(3, dtype=jnp.float32)
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.select_default_method",
            lambda _spec, _likelihood="particle", _support=None: "laplace_em",
        )

        def fake_fit_laplace_em(_model, _observations, _times, **kwargs):
            captured.update(kwargs)
            return InferenceResult(
                _samples={"drift_diag_pop": jnp.zeros((1, 1), dtype=jnp.float32)},
                method="laplace_em",
                diagnostics={},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.laplace_em.fit_laplace_em",
            fake_fit_laplace_em,
        )

        result = fit(
            model,
            observations=observations,
            times=times,
            method="auto",
            smc_config={"n_outer": 6, "n_csmc_particles": 8, "param_step_size": 0.05},
            nuts_config={"max_tree_depth": 99},
            svi_config={"num_steps": 1234},
        )

        assert result.method == "laplace_em"
        assert captured["n_outer"] == 6
        assert captured["n_csmc_particles"] == 8
        assert captured["param_step_size"] == 0.05
        assert "max_tree_depth" not in captured
        assert "num_steps" not in captured


# =============================================================================
# Builder Noise Family Wiring Tests
# =============================================================================


# =============================================================================
# Hess-MC² Unit Tests (Pure-JAX Components)
# =============================================================================


class TestHessMC2Proposals:
    """Test proposal functions are mathematically correct.

    All proposals are pure JAX: no model, no handlers, instant.
    """

    @pytest.fixture
    def particle_state(self):
        """Standard 3D particle state for proposal tests."""
        D = 3
        return {
            "x": jnp.array([1.0, -0.5, 0.3]),
            "grad": jnp.array([0.2, -0.1, 0.05]),
            "hessian": jnp.diag(jnp.array([-2.0, -1.5, -3.0])),  # negative definite
            "z": jnp.array([0.5, -0.3, 0.1]),
            "eps": 0.1,
            "eps_fb": 0.01,
            "D": D,
        }

    def test_rw_proposal_is_x_plus_eps_z(self, particle_state):
        """RW: x_new = x + eps * z (Eq 28)."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_rw

        s = particle_state
        x_new, v, v_half, chol_M, _ss = _propose_rw(
            s["x"], s["grad"], s["hessian"], s["z"], s["eps"], s["eps_fb"]
        )
        expected = s["x"] + s["eps"] * s["z"]
        assert jnp.allclose(x_new, expected)
        assert jnp.allclose(v, s["z"])
        assert jnp.allclose(v_half, s["z"])
        assert jnp.allclose(chol_M, jnp.eye(s["D"]))

    def test_fo_proposal_uses_gradient(self, particle_state):
        """FO/MALA: v_half = 0.5*eps*grad + z, x_new = x + eps*v_half (Eq 30-33)."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_fo

        s = particle_state
        x_new, v, v_half, _chol_M, _ss = _propose_fo(
            s["x"], s["grad"], s["hessian"], s["z"], s["eps"], s["eps_fb"]
        )
        expected_v_half = 0.5 * s["eps"] * s["grad"] + s["z"]
        expected_x = s["x"] + s["eps"] * expected_v_half
        assert jnp.allclose(v_half, expected_v_half)
        assert jnp.allclose(x_new, expected_x)
        assert jnp.allclose(v, s["z"])

    def test_fo_reduces_to_rw_when_grad_is_zero(self, particle_state):
        """With zero gradient, FO should behave like RW."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_fo, _propose_rw

        s = particle_state
        zero_grad = jnp.zeros(s["D"])
        x_fo, _, _, _, _ = _propose_fo(
            s["x"], zero_grad, s["hessian"], s["z"], s["eps"], s["eps_fb"]
        )
        x_rw, _, _, _, _ = _propose_rw(
            s["x"], zero_grad, s["hessian"], s["z"], s["eps"], s["eps_fb"]
        )
        assert jnp.allclose(x_fo, x_rw)

    def test_so_proposal_uses_hessian_when_psd(self, particle_state):
        """SO: with negative definite Hessian, uses full mass matrix M = -H."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_so

        s = particle_state
        x_new, _v, _v_half, chol_M, ss = _propose_so(
            s["x"], s["grad"], s["hessian"], s["z"], s["eps"], s["eps_fb"]
        )
        # M = -H = diag([2, 1.5, 3]), chol_M = diag([sqrt(2), sqrt(1.5), sqrt(3)])
        neg_hd = -jnp.diag(s["hessian"])  # [2.0, 1.5, 3.0]
        expected_v = s["z"] * jnp.sqrt(neg_hd)
        expected_v_half = 0.5 * s["eps"] * s["grad"] + expected_v
        expected_x = s["x"] + s["eps"] * (expected_v_half / neg_hd)
        assert jnp.allclose(x_new, expected_x, atol=1e-5)
        assert jnp.allclose(ss, s["eps"])  # used SO step size, not fallback
        # chol_M should be approximately diag(sqrt(neg_hd))
        assert chol_M.shape == (s["D"], s["D"])
        assert jnp.allclose(jnp.diag(chol_M), jnp.sqrt(neg_hd), atol=1e-3)

    def test_so_falls_back_to_fo_when_not_psd(self, particle_state):
        """SO: with non-negative-definite Hessian, falls back to FO."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_fo, _propose_so

        s = particle_state
        # Hessian with positive eigenvalue -> -H not PSD
        bad_hess = jnp.diag(jnp.array([1.0, -1.5, -3.0]))
        x_so, _, _, chol_M, ss = _propose_so(
            s["x"], s["grad"], bad_hess, s["z"], s["eps"], s["eps_fb"]
        )
        x_fo, _, _, _, _ = _propose_fo(
            s["x"], s["grad"], bad_hess, s["z"], s["eps_fb"], s["eps_fb"]
        )
        assert jnp.allclose(x_so, x_fo)
        assert jnp.allclose(chol_M, jnp.eye(s["D"]))  # identity mass
        assert jnp.allclose(ss, s["eps_fb"])  # used fallback step size


class TestHessMC2ReverseMomentum:
    """Test reverse momentum functions match paper equations."""

    @pytest.fixture
    def reverse_state(self):
        return {
            "v_half": jnp.array([0.3, -0.2, 0.1]),
            "grad_new": jnp.array([0.1, -0.3, 0.2]),
            "hessian_new": jnp.diag(jnp.array([-2.0, -1.0, -4.0])),
            "eps": 0.1,
            "eps_fb": 0.01,
        }

    def test_rw_reverse_is_identity(self, reverse_state):
        """RW reverse: v_new = v_half (symmetric)."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _reverse_rw

        s = reverse_state
        v_new, _chol_M, _ss = _reverse_rw(
            s["v_half"], s["grad_new"], s["hessian_new"], s["eps"], s["eps_fb"]
        )
        assert jnp.allclose(v_new, s["v_half"])

    def test_fo_reverse_applies_gradient_kick(self, reverse_state):
        """FO reverse: v_new = 0.5*eps*grad_new + v_half (Eq 34)."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _reverse_fo

        s = reverse_state
        v_new, _chol_M, _ss = _reverse_fo(
            s["v_half"], s["grad_new"], s["hessian_new"], s["eps"], s["eps_fb"]
        )
        expected = 0.5 * s["eps"] * s["grad_new"] + s["v_half"]
        assert jnp.allclose(v_new, expected)


class TestHessMC2Weights:
    """Test importance weight computation and CoV L-kernel."""

    def test_cov_density_is_finite(self):
        """CoV log-density should be finite for reasonable inputs."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _log_cov_density

        v = jnp.array([0.5, -0.3])
        chol_M = jnp.eye(2)
        eps = 0.1
        D = 2
        ld = _log_cov_density(v, chol_M, eps, D)
        assert jnp.isfinite(ld)

    def test_cov_density_higher_for_smaller_v(self):
        """Closer to mode (v=0) should give higher density."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _log_cov_density

        chol_M = jnp.eye(3)
        eps = 0.1
        D = 3
        ld_small = _log_cov_density(jnp.array([0.01, 0.01, 0.01]), chol_M, eps, D)
        ld_large = _log_cov_density(jnp.array([2.0, 2.0, 2.0]), chol_M, eps, D)
        assert ld_small > ld_large

    def test_weight_update_no_change_gives_zero_correction(self):
        """If proposal doesn't move and forward == reverse, weight unchanged."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _compute_weight

        D = 2
        logw_old = jnp.array(-1.0)
        log_post = jnp.array(-5.0)
        v = jnp.array([0.3, -0.1])
        chol_M = jnp.eye(D)
        ss = jnp.array(0.1)

        # Same post, same v forward and reverse, same mass and step size
        lw = _compute_weight(logw_old, log_post, log_post, v, v, chol_M, chol_M, ss, ss, D)
        # log_L - log_q cancels, log_post_new - log_post_old cancels
        assert jnp.allclose(lw, logw_old, atol=1e-5)

    def test_weight_increases_when_posterior_improves(self):
        """Moving to higher posterior should increase the weight."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _compute_weight

        D = 2
        logw_old = jnp.array(0.0)
        v = jnp.array([0.1, -0.1])
        chol_M = jnp.eye(D)
        ss = jnp.array(0.1)

        lw_better = _compute_weight(
            logw_old, jnp.array(-3.0), jnp.array(-5.0), v, v, chol_M, chol_M, ss, ss, D
        )
        lw_worse = _compute_weight(
            logw_old, jnp.array(-7.0), jnp.array(-5.0), v, v, chol_M, chol_M, ss, ss, D
        )
        assert lw_better > lw_worse

    def test_weight_neginf_for_invalid_posterior(self):
        """Non-finite posterior should give -inf weight."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _compute_weight

        D = 2
        lw = _compute_weight(
            jnp.array(0.0),
            jnp.array(jnp.nan),  # invalid new posterior
            jnp.array(-5.0),
            jnp.zeros(D),
            jnp.zeros(D),
            jnp.eye(D),
            jnp.eye(D),
            jnp.array(0.1),
            jnp.array(0.1),
            D,
        )
        assert lw == -jnp.inf


class TestHessMC2VmapBatching:
    """Test that proposals and weights work correctly under vmap."""

    def test_propose_fo_batch(self):
        """Vmapped FO proposal should give same result as sequential."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _propose_fo

        N, D = 4, 3
        key = random.PRNGKey(0)
        xs = random.normal(key, (N, D))
        grads = random.normal(random.PRNGKey(1), (N, D))
        zs = random.normal(random.PRNGKey(2), (N, D))
        hess = jnp.zeros((N, D, D))
        eps, eps_fb = 0.1, 0.01

        batch_fn = jax.vmap(_propose_fo, in_axes=(0, 0, 0, 0, None, None))
        x_batch, _, _, _, _ = batch_fn(xs, grads, hess, zs, eps, eps_fb)

        for i in range(N):
            x_single, _, _, _, _ = _propose_fo(xs[i], grads[i], hess[i], zs[i], eps, eps_fb)
            assert jnp.allclose(x_batch[i], x_single)

    def test_weight_batch(self):
        """Vmapped weight computation should match sequential."""
        from causal_ssm_agent.models.ssm.inference.methods.hessmc2 import _compute_weight

        N, D = 4, 3
        key = random.PRNGKey(42)
        keys = random.split(key, 8)
        logw_old = random.normal(keys[0], (N,))
        log_post_new = random.normal(keys[1], (N,)) - 5.0
        log_post_old = random.normal(keys[2], (N,)) - 5.0
        v = random.normal(keys[3], (N, D))
        v_new = random.normal(keys[4], (N, D))
        fwd_chol = jnp.broadcast_to(jnp.eye(D), (N, D, D)).copy()
        rev_chol = jnp.broadcast_to(jnp.eye(D), (N, D, D)).copy()
        fwd_ss = jnp.full((N,), 0.1)
        rev_ss = jnp.full((N,), 0.1)

        batch_fn = jax.vmap(_compute_weight, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))
        lw_batch = batch_fn(
            logw_old, log_post_new, log_post_old, v, v_new, fwd_chol, rev_chol, fwd_ss, rev_ss, D
        )

        for i in range(N):
            lw_single = _compute_weight(
                logw_old[i],
                log_post_new[i],
                log_post_old[i],
                v[i],
                v_new[i],
                fwd_chol[i],
                rev_chol[i],
                fwd_ss[i],
                rev_ss[i],
                D,
            )
            assert jnp.allclose(lw_batch[i], lw_single)


# =============================================================================
# Tempered SMC Upgrade Tests
# =============================================================================


class TestTemperedSMCVariants:
    """Parameterized smoke tests for tempered SMC variants."""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "variant, extra_kwargs",
        [
            pytest.param(
                "adaptive",
                {
                    "n_outer": 50,
                    "adaptive_tempering": True,
                    "target_ess_ratio": 0.5,
                    "waste_free": False,
                },
                id="adaptive",
            ),
            pytest.param(
                "waste_free",
                {
                    "n_outer": 10,
                    "waste_free": True,
                },
                id="waste_free",
            ),
            pytest.param(
                "multi_step_hmc",
                {
                    "n_outer": 6,
                    "n_leapfrog": 5,
                    "waste_free": False,
                },
                id="multi_step_hmc",
            ),
        ],
    )
    def test_tempered_smc_variant(self, lgss_data, variant, extra_kwargs):
        """Each tempered SMC variant should complete and pass its specific checks."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        common_kwargs = {
            "observations": lgss_data["observations"],
            "times": lgss_data["times"],
            "method": "tempered_smc",
            "n_csmc_particles": 10,
            "n_mh_steps": 5,
            "param_step_size": 0.01,
            "seed": 0,
        }
        result = fit(model, **{**common_kwargs, **extra_kwargs})

        assert isinstance(result, InferenceResult)

        # Variant-specific assertions
        if variant == "adaptive":
            beta_schedule = result.diagnostics["beta_schedule"]
            assert beta_schedule[-1] == 1.0, f"Final beta={beta_schedule[-1]}, expected 1.0"
        elif variant == "waste_free":
            assert result.diagnostics["waste_free"] is True
        elif variant == "multi_step_hmc":
            assert result.diagnostics["n_leapfrog"] == 5

    def test_waste_free_rejects_bad_n(self):
        """Waste-free should reject N % n_mh_steps != 0."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, n_particles=50)

        T = 10
        observations = jnp.zeros((T, 1))
        times = jnp.arange(T, dtype=float)

        with pytest.raises(ValueError, match="waste_free requires"):
            fit(
                model,
                observations=observations,
                times=times,
                method="tempered_smc",
                n_outer=5,
                n_csmc_particles=7,  # N=7, n_mh_steps=3 -> 7%3 != 0
                n_mh_steps=3,
                waste_free=True,
                seed=0,
            )


# =============================================================================
# PGAS Upgrade Tests
# =============================================================================


class TestPGASVariants:
    """Parameterized smoke tests for PGAS variants."""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "variant, extra_kwargs",
        [
            pytest.param(
                "preconditioned",
                {
                    "n_outer": 10,
                    "n_warmup": 5,
                    "block_sampling": False,
                },
                id="preconditioned",
            ),
            pytest.param(
                "optimal_proposal",
                {
                    "n_outer": 6,
                    "n_warmup": 3,
                },
                id="optimal_proposal",
            ),
            pytest.param(
                "block_sampling",
                {
                    "n_outer": 10,
                    "n_warmup": 5,
                    "block_sampling": True,
                },
                id="block_sampling",
            ),
        ],
    )
    def test_pgas_variant(self, lgss_data, variant, extra_kwargs):
        """Each PGAS variant should complete and pass its specific checks."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        common_kwargs = {
            "observations": lgss_data["observations"],
            "times": lgss_data["times"],
            "method": "pgas",
            "n_csmc_particles": 8,
            "n_mh_steps": 3,
            "param_step_size": 0.1,
            "seed": 0,
        }
        result = fit(model, **{**common_kwargs, **extra_kwargs})

        assert isinstance(result, InferenceResult)

        # Variant-specific assertions
        if variant == "preconditioned":
            assert result.method == "pgas"
            assert len(result.diagnostics["accept_rates"]) == 10
        elif variant == "optimal_proposal":
            assert result.diagnostics["gaussian_obs"] is True
        elif variant == "block_sampling":
            assert result.diagnostics["block_sampling"] is True
            assert "block_accept_rates" in result.diagnostics
            # Should have per-block rates for each parameter site
            block_rates = result.diagnostics["block_accept_rates"]
            assert len(block_rates) > 0
            for _name, rates in block_rates.items():
                assert len(rates) == 10  # n_outer iterations

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_fallback_for_poisson(self):
        """PGAS should fall back to gradient proposal for non-Gaussian obs."""
        from causal_ssm_agent.models.ssm import DistributionFamily, SSMModel, fit

        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dists=[DistributionFamily.POISSON],
            manifest_means=jnp.array([jnp.log(5.0)]),
        )
        model = SSMModel(spec, n_particles=50)

        T = 30
        key = random.PRNGKey(0)
        observations = random.poisson(key, jnp.ones((T, 1)) * 5.0).astype(jnp.float32)
        times = jnp.arange(T, dtype=float)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="pgas",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.diagnostics["gaussian_obs"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
