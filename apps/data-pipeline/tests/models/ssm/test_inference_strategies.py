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

import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.distributions import DistributionFamily
from causal_ssm_agent.models.ssm import AutoReparam, InferenceResult, SSMModel, fit
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.methods.map import (
    _build_map_laplace_bundle,
    fit_map,
)
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
from causal_ssm_agent.models.ssm.inference.targets.laplace import (
    LaplaceLikelihood,
    _assemble_support_aware_observation_system,
    _block_banded_logdet,
    _build_ieks_system_from_prior,
    _build_linear_summary_accumulator_plan,
    _build_linear_summary_augmented_system,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_banded_cholesky,
    _factor_block_profile_cholesky,
    _ieks_smooth,
    _infer_support_groups,
    _linear_summary_augmented_ieks_laplace,
    _make_support_window_derivatives,
    _point_ieks_mode,
    _point_laplace_from_mode,
    _predictive_latent_init,
    _should_use_dense_support_laplace,
    _solve_block_banded_from_cholesky,
    _solve_block_tridiagonal,
    _support_aware_ieks_laplace,
    _support_aware_ieks_mode,
    _support_aware_laplace_from_mode,
    _support_aware_step_halving_search,
    block_profile_logdet_packed_cotangent,
)
from causal_ssm_agent.models.ssm.inference.targets.linear_summary_augmentation import (
    lift_linear_summary_observation_trajectory,
    row_observation_log_prob,
)
from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood, SSMAdapter
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    compile_observation_operator,
    expected_observation_mean,
    get_point_like_mask,
    get_summary_operator_codes,
    get_support_kind_codes,
    trajectory_observation_log_prob,
    trajectory_observation_log_probs,
)
from causal_ssm_agent.models.ssm.inference.utils import _discover_sites
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.ssm_test_utils import (
    diagonal_diffusion_kwargs,
    make_observation_support_runtime,
    make_ssm_spec,
)


def test_compile_observation_operator_keeps_point_support_without_interval_summary_mode():
    support = make_observation_support_runtime(
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
    support = make_observation_support_runtime(
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
    assert operator.summary_operator_codes is not None

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

        H_rows = jnp.broadcast_to(H[None, :, :], (T, *H.shape))
        d_rows = jnp.broadcast_to(d[None, :], (T, *d.shape))
        z_smooth, log_lik, _inner_eval_aux = _ieks_smooth(
            observations,
            obs_mask,
            Ad,
            Qd,
            cd,
            H_rows,
            d_rows,
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
        prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )
        lower, diag, upper, rhs = _build_ieks_system_from_prior(
            prior_lower,
            prior_diag,
            prior_upper,
            prior_rhs,
            J_t,
            tilde_y,
        )
        dense = self._dense_block_matrix(lower, diag, upper)
        z_dense = jnp.linalg.solve(dense, rhs.reshape(-1)).reshape(T, D)

        np.testing.assert_allclose(z_smooth, z_dense, atol=1e-4, rtol=1e-4)
        assert jnp.isfinite(log_lik).all()

    def test_point_implicit_laplace_value_matches_direct_mode_evaluation(self):
        """Implicit point-path wrapper should match direct mode solve + Laplace eval."""
        observations = jnp.array([[0.1], [0.3], [-0.2], [0.15]], dtype=jnp.float32)
        obs_mask = jnp.ones_like(observations, dtype=bool)
        Ad = jnp.broadcast_to(jnp.array([[0.91]], dtype=jnp.float32), (4, 1, 1))
        Qd = jnp.broadcast_to(jnp.array([[0.07]], dtype=jnp.float32), (4, 1, 1))
        cd = jnp.zeros((4, 1), dtype=jnp.float32)
        H_rows = jnp.broadcast_to(jnp.array([[[1.0]]], dtype=jnp.float32), (4, 1, 1))
        d_rows = jnp.zeros((4, 1), dtype=jnp.float32)
        init_mean = jnp.array([0.05], dtype=jnp.float32)
        init_cov = jnp.array([[0.6]], dtype=jnp.float32)

        def _runtime(raw_params):
            obs_df = jnp.exp(raw_params[0]) + 2.5
            obs_var = jnp.exp(raw_params[1]) + 0.1
            obs_kernel = build_observation_kernel(
                DistributionFamily.STUDENT_T,
                LinkFunction.IDENTITY,
                {"obs_df": obs_df},
            )
            return jnp.array([[obs_var]], dtype=jnp.float32), obs_kernel

        def _implicit_objective(raw_params):
            R, obs_kernel = _runtime(raw_params)
            _z_mode, log_lik, _inner_eval_aux = _ieks_smooth(
                observations,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                R,
                init_mean,
                init_cov,
                obs_kernel,
                n_ieks_iters=12,
            )
            return log_lik

        def _direct_objective(raw_params):
            R, obs_kernel = _runtime(raw_params)
            z_mode, mode_aux = _point_ieks_mode(
                observations,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                R,
                init_mean,
                init_cov,
                obs_kernel,
                n_ieks_iters=12,
            )
            log_lik, _inner_eval_aux = _point_laplace_from_mode(
                z_mode,
                mode_aux,
                observations,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                R,
                init_mean,
                init_cov,
                obs_kernel,
            )
            return log_lik

        raw_params = jnp.array([0.35, -1.2], dtype=jnp.float32)
        implicit_value = _implicit_objective(raw_params)
        direct_value = _direct_objective(raw_params)

        np.testing.assert_allclose(
            np.asarray(implicit_value),
            np.asarray(direct_value),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_point_implicit_gradient_matches_finite_difference(self):
        """Implicit point-path gradient should agree with finite differences."""
        observations = jnp.array([[0.1], [0.3], [-0.2], [0.15]], dtype=jnp.float32)
        obs_mask = jnp.ones_like(observations, dtype=bool)
        Ad = jnp.broadcast_to(jnp.array([[0.91]], dtype=jnp.float32), (4, 1, 1))
        Qd = jnp.broadcast_to(jnp.array([[0.07]], dtype=jnp.float32), (4, 1, 1))
        cd = jnp.zeros((4, 1), dtype=jnp.float32)
        H_rows = jnp.broadcast_to(jnp.array([[[1.0]]], dtype=jnp.float32), (4, 1, 1))
        d_rows = jnp.zeros((4, 1), dtype=jnp.float32)
        init_mean = jnp.array([0.05], dtype=jnp.float32)
        init_cov = jnp.array([[0.6]], dtype=jnp.float32)

        def _build_measurement_objects(manifest_cov, runtime_extra_params):
            return compile_measurement_semantics(
                [DistributionFamily.STUDENT_T],
                manifest_cov=manifest_cov,
                extra_params=runtime_extra_params,
                manifest_links=[LinkFunction.IDENTITY],
                observation_support=None,
            )

        def _objective(raw_params):
            obs_df = jnp.exp(raw_params[0]) + 2.5
            obs_var = jnp.exp(raw_params[1]) + 0.1
            extra_params = {"obs_df": obs_df}
            R = jnp.array([[obs_var]], dtype=jnp.float32)
            measurement_semantics = _build_measurement_objects(R, extra_params)
            _z_mode, log_lik, _inner_eval_aux = _ieks_smooth(
                observations,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                R,
                init_mean,
                init_cov,
                measurement_semantics.obs_kernel,
                n_ieks_iters=12,
                build_measurement_objects=_build_measurement_objects,
                extra_params=extra_params,
            )
            return log_lik

        raw_params = jnp.array([0.35, -1.2], dtype=jnp.float32)
        implicit_grad = jax.grad(_objective)(raw_params)

        eps = 1e-3
        finite_diff = np.zeros((2,), dtype=np.float32)
        raw_params_np = np.asarray(raw_params)
        for idx in range(raw_params_np.shape[0]):
            step = np.zeros_like(raw_params_np)
            step[idx] = eps
            finite_diff[idx] = (
                float(_objective(jnp.asarray(raw_params_np + step, dtype=jnp.float32)))
                - float(_objective(jnp.asarray(raw_params_np - step, dtype=jnp.float32)))
            ) / (2.0 * eps)

        np.testing.assert_allclose(
            np.asarray(implicit_grad),
            finite_diff,
            rtol=5e-2,
            atol=5e-2,
        )

    def test_point_backend_gradient_supports_traced_observation_hyperparameters(self):
        """LaplaceLikelihood point path should differentiate through traced obs hyperparameters."""
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.STUDENT_T],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=12,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.09]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.07]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.05], dtype=jnp.float32),
            cov=jnp.array([[0.6]], dtype=jnp.float32),
        )
        observations = jnp.array([[0.1], [0.3], [-0.2], [0.15]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        def _objective(raw_params):
            obs_df = jnp.exp(raw_params[0]) + 2.5
            obs_var = jnp.exp(raw_params[1]) + 0.1
            meas_params = MeasurementParams(
                lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
                manifest_means=jnp.array([0.0], dtype=jnp.float32),
                manifest_cov=jnp.array([[obs_var]], dtype=jnp.float32),
            )
            return backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                observations,
                time_intervals,
                extra_params={"obs_df": obs_df},
            )

        raw_params = jnp.array([0.35, -1.2], dtype=jnp.float32)
        implicit_grad = jax.grad(_objective)(raw_params)

        eps = 1e-3
        finite_diff = np.zeros((2,), dtype=np.float32)
        raw_params_np = np.asarray(raw_params)
        for idx in range(raw_params_np.shape[0]):
            step = np.zeros_like(raw_params_np)
            step[idx] = eps
            finite_diff[idx] = (
                float(_objective(jnp.asarray(raw_params_np + step, dtype=jnp.float32)))
                - float(_objective(jnp.asarray(raw_params_np - step, dtype=jnp.float32)))
            ) / (2.0 * eps)

        np.testing.assert_allclose(
            np.asarray(implicit_grad),
            finite_diff,
            rtol=5e-2,
            atol=5e-2,
        )


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
        support = make_observation_support_runtime(
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
        support = make_observation_support_runtime(
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
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
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

        assert backend.transition_dispatch_mode == "mixed"
        assert jnp.all(jnp.isfinite(ll))


class TestSupportAwareTrajectoryObservationLogProb:
    def test_window_average_matches_manual_gaussian_average(self):
        support = make_observation_support_runtime(
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
        support = make_observation_support_runtime(
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
            emission_slot_indices=np.array([[-1], [-1], [0], [1]], dtype=np.int64),
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


class TestLaplaceSupportAware:
    def test_infer_support_groups_ignores_reused_slot_history(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0], [np.nan], [2.0], [np.nan], [4.0]]),
            support_end_times=np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan], [5.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.0], [0.5], [0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [-1], [0], [-1], [0]], dtype=np.int64),
        )

        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)
        assert len(window_batches) == 1
        windows = window_batches[0]

        np.testing.assert_array_equal(np.asarray(windows.anchor_indices), np.array([1, 3, 5]))
        np.testing.assert_array_equal(np.asarray(windows.start_indices), np.array([0, 2, 4]))
        np.testing.assert_array_equal(np.asarray(windows.state_lens), np.array([2, 2, 2]))
        np.testing.assert_array_equal(
            np.asarray(row_upper_bandwidths), np.array([1, 0, 1, 0, 1, 0])
        )
        assert windows.max_state_len == 2
        assert bandwidth == 1

    def test_infer_support_groups_buckets_by_state_length(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["3d"],
            support_start_times=np.array([[np.nan], [0.0], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0], [np.nan], [3.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [0.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [-1], [0]], dtype=np.int64),
        )

        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)

        assert [batch.max_state_len for batch in window_batches] == [2, 4]
        np.testing.assert_array_equal(np.asarray(window_batches[0].anchor_indices), np.array([1]))
        np.testing.assert_array_equal(np.asarray(window_batches[1].anchor_indices), np.array([3]))
        np.testing.assert_array_equal(np.asarray(row_upper_bandwidths), np.array([3, 2, 1, 0]))
        assert bandwidth == 3

    def test_linear_summary_accumulator_plan_reuses_slots_and_marks_resets(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            summary_operators=["mean"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0], [1.0], [2.0], [3.0]]),
            support_end_times=np.array([[np.nan], [1.0], [2.0], [3.0], [4.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0], [1.0], [1.0]]),
            emission_slot_indices=np.array([[-1], [0], [0], [0], [0]], dtype=np.int64),
        )

        plan = _build_linear_summary_accumulator_plan(
            support,
            [DistributionFamily.GAUSSIAN],
            [LinkFunction.IDENTITY],
        )

        assert plan is not None
        assert plan.n_accumulators == 1
        np.testing.assert_array_equal(
            np.asarray(plan.row_reset_mask[:, 0]),
            np.array([True, True, True, True, False]),
        )
        np.testing.assert_array_equal(
            np.asarray(plan.row_emission_accumulator_indices[:, 0]),
            np.array([-1, 0, 0, 0, 0]),
        )
        np.testing.assert_allclose(
            np.asarray(plan.row_emission_scales[1:, 0]),
            np.ones((4,), dtype=np.float32),
        )

    def test_linear_summary_accumulator_plan_rejects_nonlinear_interval_support(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["window_std"],
            support_kinds=["interval"],
            summary_operators=["std"],
            observation_windows=["1d"],
            support_start_times=np.array([[np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [1.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5]]),
            interval_weights=np.array([[0.0], [1.0]]),
        )

        plan = _build_linear_summary_accumulator_plan(
            support,
            [DistributionFamily.GAUSSIAN],
            [LinkFunction.IDENTITY],
        )

        assert plan is None

    def test_linear_summary_augmented_system_builds_rowwise_observations_and_resets(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["point_signal", "avg_signal"],
            support_kinds=["point", "interval"],
            summary_operators=["last", "mean"],
            observation_windows=["1d", "2d"],
            support_start_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.0]]),
            support_end_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 2.0]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_weights=np.array([[[0.0], [0.0]], [[0.0], [1.0]], [[0.0], [1.0]]]),
        )
        plan = _build_linear_summary_accumulator_plan(
            support,
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            [LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        )
        assert plan is not None

        Ad_aug, _Qd_aug, _cd_aug, init_mean_aug, init_cov_aug, H_rows, d_rows = (
            _build_linear_summary_augmented_system(
                plan=plan,
                time_intervals=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
                drift=jnp.array([[-0.4]], dtype=jnp.float32),
                diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
                cint=jnp.array([0.2], dtype=jnp.float32),
                H=jnp.array([[1.5], [2.0]], dtype=jnp.float32),
                d=jnp.array([0.3, -0.2], dtype=jnp.float32),
                init_mean=jnp.array([0.0], dtype=jnp.float32),
                init_cov=jnp.array([[1.0]], dtype=jnp.float32),
                support_kind_codes=get_support_kind_codes(support),
            )
        )

        assert init_mean_aug.shape == (2,)
        assert init_cov_aug.shape == (2, 2)
        assert H_rows.shape == (3, 2, 2)
        assert d_rows.shape == (3, 2)
        np.testing.assert_allclose(
            np.asarray(H_rows[:, 0, 0]), np.full((3,), 1.5, dtype=np.float32)
        )
        np.testing.assert_allclose(np.asarray(d_rows[:, 0]), np.full((3,), 0.3, dtype=np.float32))
        np.testing.assert_allclose(np.asarray(H_rows[2, 1]), np.array([0.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(d_rows[:, 1]), np.zeros((3,), dtype=np.float32))
        assert float(Ad_aug[1, 1, 1]) == pytest.approx(0.0)
        assert float(Ad_aug[2, 1, 1]) == pytest.approx(1.0)

    def test_linear_summary_augmented_row_likelihood_matches_support_aware_likelihood(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["point_signal", "avg_signal"],
            support_kinds=["point", "interval"],
            summary_operators=["last", "mean"],
            observation_windows=["1d", "2d"],
            support_start_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.0]]),
            support_end_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 2.0]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_weights=np.array([[[0.0], [0.0]], [[0.0], [1.0]], [[0.0], [1.0]]]),
        )
        plan = _build_linear_summary_accumulator_plan(
            support,
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            [LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        )
        assert plan is not None

        observations = jnp.array(
            [
                [0.50, jnp.nan],
                [0.88, jnp.nan],
                [1.34, 0.53],
            ],
            dtype=jnp.float32,
        )
        obs_mask = ~jnp.isnan(observations)
        latent_trajectory = jnp.array([[0.10], [0.40], [0.70]], dtype=jnp.float32)
        H = jnp.array([[1.5], [2.0]], dtype=jnp.float32)
        d = jnp.array([0.3, -0.2], dtype=jnp.float32)
        R = jnp.diag(jnp.array([0.04, 0.09], dtype=jnp.float32))

        _Ad_aug, _Qd_aug, _cd_aug, _init_mean_aug, _init_cov_aug, H_rows, d_rows = (
            _build_linear_summary_augmented_system(
                plan=plan,
                time_intervals=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
                drift=jnp.array([[-0.4]], dtype=jnp.float32),
                diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
                cint=jnp.array([0.2], dtype=jnp.float32),
                H=H,
                d=d,
                init_mean=jnp.array([0.0], dtype=jnp.float32),
                init_cov=jnp.array([[1.0]], dtype=jnp.float32),
                support_kind_codes=get_support_kind_codes(support),
            )
        )

        augmented_trajectory = lift_linear_summary_observation_trajectory(
            latent_trajectory,
            H=H,
            d=d,
            plan=plan,
            observation_support=support,
        )
        support_semantics = compile_measurement_semantics(
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            manifest_cov=R,
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
            observation_support=support,
        )
        augmented_semantics = compile_measurement_semantics(
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            manifest_cov=R,
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
            observation_support=None,
        )

        support_lp = trajectory_observation_log_prob(
            latent_trajectory,
            observations,
            obs_mask,
            H,
            d,
            R,
            support_semantics.obs_kernel,
            support_semantics.mean_log_prob_fn,
            support,
        )
        augmented_lp = row_observation_log_prob(
            augmented_trajectory,
            observations,
            obs_mask,
            H_rows,
            d_rows,
            R,
            augmented_semantics.obs_kernel,
        )

        assert float(augmented_lp) == pytest.approx(float(support_lp), abs=1e-6)

    def test_profile_masked_banded_cholesky_matches_full_banded_solver(self):
        row_upper_bandwidths = jnp.array([3, 2, 1, 1, 0], dtype=jnp.int32)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        diag = jnp.asarray(np.array([4.0, 4.5, 4.2, 4.3, 3.8], dtype=np.float32)[:, None, None])
        upper = jnp.zeros((3, 5, 1, 1), dtype=jnp.float32)
        upper = upper.at[0, 0, 0, 0].set(0.20)
        upper = upper.at[0, 1, 0, 0].set(0.15)
        upper = upper.at[0, 2, 0, 0].set(0.10)
        upper = upper.at[0, 3, 0, 0].set(0.08)
        upper = upper.at[1, 0, 0, 0].set(0.05)
        upper = upper.at[1, 1, 0, 0].set(0.04)
        upper = upper.at[2, 0, 0, 0].set(0.02)
        rhs = jnp.asarray(np.array([1.0, -0.5, 0.25, 0.75, -1.25], dtype=np.float32)[:, None])

        chol_full, lower_full = _factor_block_banded_cholesky(diag, upper)
        sol_full = _solve_block_banded_from_cholesky(chol_full, lower_full, rhs)

        chol_profile, lower_profile = _factor_block_banded_cholesky(
            diag,
            upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        sol_profile = _solve_block_banded_from_cholesky(
            chol_profile,
            lower_profile,
            rhs,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

        np.testing.assert_allclose(
            np.asarray(chol_profile), np.asarray(chol_full), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(lower_profile),
            np.asarray(lower_full),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(sol_profile), np.asarray(sol_full), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(_block_banded_logdet(chol_profile)),
            np.asarray(_block_banded_logdet(chol_full)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_predictive_latent_init_rolls_forward_mean_dynamics(self):
        Ad = jnp.array(
            [
                [[1.0]],
                [[2.0]],
                [[3.0]],
            ],
            dtype=jnp.float32,
        )
        cd = jnp.array([[0.5], [1.0], [-2.0]], dtype=jnp.float32)
        init_mean = jnp.array([2.0], dtype=jnp.float32)

        z_init = _predictive_latent_init(Ad, cd, init_mean)

        np.testing.assert_allclose(
            np.asarray(z_init[:, 0]),
            np.array([2.5, 6.0, 16.0], dtype=np.float32),
        )

    def test_dense_support_path_threshold_matches_smallgolden_regime(self):
        assert _should_use_dense_support_laplace(n_time=10, n_latent=12) is True
        assert _should_use_dense_support_laplace(n_time=20, n_latent=12) is False

    def test_support_window_gauss_newton_matches_linear_gaussian_exact_blocks(self):
        support = make_observation_support_runtime(
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
        window_batches, bandwidth, _row_upper_bandwidths = _infer_support_groups(support)
        assert len(window_batches) == 1
        windows = window_batches[0]
        observation_operator = compile_observation_operator(support)
        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        mean_log_prob_fn = get_mean_param_log_prob_fn(DistributionFamily.GAUSSIAN)
        window_derivatives = (
            _make_support_window_derivatives(
                max_state_len=windows.max_state_len,
                n_latent=1,
                n_manifest=1,
                summary_operator_codes=get_summary_operator_codes(support),
                obs_kernel=obs_kernel,
                mean_log_prob_fn=mean_log_prob_fn,
            ),
        )

        z_est = jnp.array([[0.4], [1.0]], dtype=jnp.float32)
        observations = jnp.array([[jnp.nan], [1.3]], dtype=jnp.float32)
        obs_mask = ~jnp.isnan(observations)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d = jnp.array([0.0], dtype=jnp.float32)
        R = jnp.array([[0.2]], dtype=jnp.float32)

        diag, upper, rhs = _assemble_support_aware_observation_system(
            z_est,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            window_batches,
            observation_operator.point_like_mask(z_est.dtype),
            window_derivatives,
            bandwidth,
        )

        state_len = int(windows.state_lens[0])
        prev_coeff = windows.prev_coeffs[0, 0, 0].astype(z_est.dtype)
        curr_coeff = windows.curr_coeffs[0, 0, 0].astype(z_est.dtype)
        weight = windows.weights[0, 0, 0].astype(z_est.dtype)
        summary_codes = get_summary_operator_codes(support)
        anchor_obs = jnp.nan_to_num(observations, nan=0.0)[1]
        anchor_mask = obs_mask[1].astype(z_est.dtype)

        def _exact_window_log_prob(segment_flat):
            states = segment_flat.reshape(windows.max_state_len, 1)
            responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d))(states)
            last_response = responses[state_len - 1]
            obs_sum = prev_coeff * responses[0] + curr_coeff * responses[1]
            obs_sumsq = prev_coeff * responses[0] ** 2 + curr_coeff * responses[1] ** 2
            obs_weight = jnp.full_like(obs_sum, weight)
            expected_mean = expected_observation_mean(
                last_response,
                obs_sum,
                obs_sumsq,
                obs_weight,
                summary_codes,
            )
            return mean_log_prob_fn(anchor_obs, expected_mean, R, anchor_mask)

        segment_flat = z_est.reshape(-1)
        grad = jax.grad(_exact_window_log_prob)(segment_flat)
        hess = jax.hessian(_exact_window_log_prob)(segment_flat)
        info = -0.5 * (hess + hess.T)
        taylor_rhs = info @ segment_flat + grad

        expected_diag = np.array([info[0, 0], info[1, 1]], dtype=np.float32)
        expected_upper = np.array([info[0, 1]], dtype=np.float32)
        expected_rhs = np.array(taylor_rhs.reshape(2, 1), dtype=np.float32)

        np.testing.assert_allclose(np.asarray(diag[:, 0, 0]), expected_diag, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(upper[0, 0, 0, 0]),
            expected_upper[0],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(np.asarray(rhs[:, 0]), expected_rhs[:, 0], rtol=1e-5, atol=1e-5)


class TestLaplaceSupportAwareGradients:
    """Implicit-mode gradient + autodiff equivalence checks for support-aware Laplace."""

    def test_support_aware_implicit_mode_gradient_matches_direct_autodiff(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]], dtype=np.float32),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]], dtype=np.float32),
            interval_weights=np.array([[0.0], [1.0], [1.0]], dtype=np.float32),
        )
        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        point_like_mask = get_point_like_mask(get_support_kind_codes(support), jnp.float32)
        summary_operator_codes = get_summary_operator_codes(support)
        clean_obs = jnp.array([[0.0], [0.0], [0.25]], dtype=jnp.float32)
        obs_mask = jnp.array([[False], [False], [True]])
        Ad = jnp.broadcast_to(jnp.array([[0.92]], dtype=jnp.float32), (3, 1, 1))
        Qd = jnp.broadcast_to(jnp.array([[0.08]], dtype=jnp.float32), (3, 1, 1))
        cd = jnp.zeros((3, 1), dtype=jnp.float32)
        H = jnp.array([[1.0]], dtype=jnp.float32)
        d = jnp.array([0.0], dtype=jnp.float32)
        init_mean = jnp.array([0.1], dtype=jnp.float32)
        init_cov = jnp.array([[0.7]], dtype=jnp.float32)

        def _build_measurement_objects(manifest_cov, runtime_extra_params):
            measurement_semantics = compile_measurement_semantics(
                [DistributionFamily.STUDENT_T],
                manifest_cov=manifest_cov,
                extra_params=runtime_extra_params,
                manifest_links=[LinkFunction.IDENTITY],
                observation_support=support,
            )
            window_derivatives = tuple(
                _make_support_window_derivatives(
                    max_state_len=batch.max_state_len,
                    n_latent=1,
                    n_manifest=1,
                    summary_operator_codes=summary_operator_codes,
                    obs_kernel=measurement_semantics.obs_kernel,
                    mean_log_prob_fn=measurement_semantics.mean_log_prob_fn,
                )
                for batch in window_batches
            )
            return measurement_semantics, window_derivatives

        def _runtime_params(raw_params):
            obs_df = jnp.exp(raw_params[0]) + 2.5
            obs_var = jnp.exp(raw_params[1]) + 0.1
            return (
                jnp.array([[obs_var]], dtype=jnp.float32),
                {"obs_df": obs_df},
            )

        def _implicit_objective(raw_params):
            R, extra_params = _runtime_params(raw_params)
            measurement_semantics, window_derivatives = _build_measurement_objects(R, extra_params)
            log_lik, _z_mode, _inner_eval_aux = _support_aware_ieks_laplace(
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                H,
                d,
                R,
                init_mean,
                init_cov,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                support,
                window_batches,
                bandwidth,
                row_upper_bandwidths,
                row_lower_bandwidths,
                window_derivatives,
                _build_measurement_objects,
                extra_params,
                n_ieks_iters=2,
            )
            return log_lik

        def _direct_objective(raw_params):
            R, extra_params = _runtime_params(raw_params)
            measurement_semantics, window_derivatives = _build_measurement_objects(R, extra_params)
            z_mode, mode_aux = _support_aware_ieks_mode(
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                H,
                d,
                R,
                init_mean,
                init_cov,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                support,
                window_batches,
                bandwidth,
                row_upper_bandwidths,
                row_lower_bandwidths,
                window_derivatives,
                n_ieks_iters=2,
                factor_block_cholesky_fn=_factor_block_banded_cholesky,
                solve_block_from_cholesky_fn=_solve_block_banded_from_cholesky,
            )
            log_lik, _inner_eval_aux = _support_aware_laplace_from_mode(
                z_mode,
                mode_aux,
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                H,
                d,
                R,
                init_mean,
                init_cov,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                support,
                window_batches,
                point_like_mask,
                window_derivatives,
                bandwidth,
                row_upper_bandwidths,
                row_lower_bandwidths,
                factor_block_cholesky_fn=_factor_block_banded_cholesky,
            )
            return log_lik

        raw_params = jnp.array([1.1, -0.7], dtype=jnp.float32)
        implicit_value, implicit_grad = jax.value_and_grad(_implicit_objective)(raw_params)
        direct_value, direct_grad = jax.value_and_grad(_direct_objective)(raw_params)

        np.testing.assert_allclose(
            np.asarray(implicit_value),
            np.asarray(direct_value),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(implicit_grad),
            np.asarray(direct_grad),
            rtol=5e-4,
            atol=5e-4,
        )

    def test_linear_summary_augmented_backend_tracks_support_aware_gaussian_mean(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["point_signal", "avg_signal"],
            support_kinds=["point", "interval"],
            summary_operators=["last", "mean"],
            observation_windows=["1d", "2d"],
            support_start_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.0]]),
            support_end_times=np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 2.0]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0]], [[0.0], [0.5]], [[0.0], [0.5]]]),
            interval_weights=np.array([[[0.0], [0.0]], [[0.0], [1.0]], [[0.0], [1.0]]]),
        )
        measurement_semantics = compile_measurement_semantics(
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            manifest_cov=jnp.array([[0.2, 0.05], [0.05, 0.3]], dtype=jnp.float32),
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
            observation_support=support,
        )
        window_batches, bandwidth, row_upper_bandwidths = _infer_support_groups(support)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        window_derivatives = tuple(
            _make_support_window_derivatives(
                max_state_len=batch.max_state_len,
                n_latent=1,
                n_manifest=2,
                summary_operator_codes=get_summary_operator_codes(support),
                obs_kernel=measurement_semantics.obs_kernel,
                mean_log_prob_fn=measurement_semantics.mean_log_prob_fn,
            )
            for batch in window_batches
        )

        ct_params = CTParams(
            drift=jnp.array([[-0.35]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.12]], dtype=jnp.float32),
            cint=jnp.array([0.05], dtype=jnp.float32),
        )
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            time_intervals,
        )
        assert cd is not None

        H = jnp.array([[1.0], [1.2]], dtype=jnp.float32)
        d = jnp.array([0.1, -0.2], dtype=jnp.float32)
        R = jnp.array([[0.2, 0.05], [0.05, 0.3]], dtype=jnp.float32)
        init_mean = jnp.array([0.2], dtype=jnp.float32)
        init_cov = jnp.array([[0.7]], dtype=jnp.float32)
        observations = jnp.array(
            [
                [0.3, jnp.nan],
                [0.7, jnp.nan],
                [0.2, 0.4],
            ],
            dtype=jnp.float32,
        )
        obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        plan = _build_linear_summary_accumulator_plan(
            support,
            [DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN],
            [LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        )
        assert plan is not None

        z_aug, log_lik_aug, _aug_aux = _linear_summary_augmented_ieks_laplace(
            clean_obs,
            obs_mask,
            time_intervals,
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            H,
            d,
            R,
            init_mean,
            init_cov,
            measurement_semantics.obs_kernel,
            plan,
            get_support_kind_codes(support),
            n_ieks_iters=3,
        )

        def _build_measurement_objects(manifest_cov, runtime_extra_params):
            del manifest_cov, runtime_extra_params
            return measurement_semantics, window_derivatives

        log_lik_support, z_mode_support, _support_aux = _support_aware_ieks_laplace(
            clean_obs,
            obs_mask,
            Ad,
            Qd,
            cd,
            H,
            d,
            R,
            init_mean,
            init_cov,
            measurement_semantics.obs_kernel,
            measurement_semantics.mean_log_prob_fn,
            support,
            window_batches,
            bandwidth,
            row_upper_bandwidths,
            row_lower_bandwidths,
            window_derivatives,
            _build_measurement_objects,
            None,
            n_ieks_iters=3,
        )

        # The augmented path uses exact CT discretization of the accumulator dynamics,
        # while the support-aware baseline uses the compiled window coefficients.
        np.testing.assert_allclose(
            np.asarray(log_lik_aug),
            np.asarray(log_lik_support),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(z_aug[:, :1]),
            np.asarray(z_mode_support),
            rtol=3e-2,
            atol=1e-2,
        )

    def test_linear_summary_augmented_backend_gradient_supports_traced_observation_hyperparameters(
        self,
    ):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]], dtype=np.float32),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]], dtype=np.float32),
            interval_weights=np.array([[0.0], [1.0], [1.0]], dtype=np.float32),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.STUDENT_T],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=8,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.15]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.08]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.05], dtype=jnp.float32),
            cov=jnp.array([[0.7]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.2]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        def _objective(raw_params):
            obs_df = jnp.exp(raw_params[0]) + 2.5
            obs_var = jnp.exp(raw_params[1]) + 0.1
            meas_params = MeasurementParams(
                lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
                manifest_means=jnp.array([0.0], dtype=jnp.float32),
                manifest_cov=jnp.array([[obs_var]], dtype=jnp.float32),
            )
            return backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                observations,
                time_intervals,
                extra_params={"obs_df": obs_df},
            )

        raw_params = jnp.array([0.25, -1.0], dtype=jnp.float32)
        implicit_grad = jax.grad(_objective)(raw_params)

        eps = 1e-3
        finite_diff = np.zeros((2,), dtype=np.float32)
        raw_params_np = np.asarray(raw_params)
        for idx in range(raw_params_np.shape[0]):
            step = np.zeros_like(raw_params_np)
            step[idx] = eps
            finite_diff[idx] = (
                float(_objective(jnp.asarray(raw_params_np + step, dtype=jnp.float32)))
                - float(_objective(jnp.asarray(raw_params_np - step, dtype=jnp.float32)))
            ) / (2.0 * eps)

        np.testing.assert_allclose(
            np.asarray(implicit_grad),
            finite_diff,
            rtol=7e-2,
            atol=7e-2,
        )


class TestLaplaceBackendCaching:
    """Backend-cache reuse, invalidation, and support-window derivative caching."""

    def test_block_profile_logdet_cotangent_matches_direct_autodiff(self):
        row_upper_bandwidths = jnp.array([2, 2, 1, 0], dtype=jnp.int32)
        row_lower_bandwidths = jnp.asarray(
            _compute_profile_lower_bandwidths(np.asarray(row_upper_bandwidths)),
            dtype=jnp.int32,
        )
        diag = jnp.array(
            [
                [[4.0, 0.2], [0.2, 3.5]],
                [[3.8, -0.1], [-0.1, 3.2]],
                [[3.4, 0.05], [0.05, 2.9]],
                [[3.1, 0.0], [0.0, 2.7]],
            ],
            dtype=jnp.float32,
        )
        upper = jnp.zeros((2, 4, 2, 2), dtype=jnp.float32)
        upper = upper.at[0, 0].set(jnp.array([[0.12, -0.03], [0.05, 0.08]], dtype=jnp.float32))
        upper = upper.at[1, 0].set(jnp.array([[0.04, 0.01], [-0.02, 0.03]], dtype=jnp.float32))
        upper = upper.at[0, 1].set(jnp.array([[0.09, 0.02], [0.01, 0.07]], dtype=jnp.float32))
        upper = upper.at[0, 2].set(jnp.array([[0.06, -0.01], [0.02, 0.05]], dtype=jnp.float32))

        def _packed_logdet(diag_blocks, upper_blocks):
            chol_diag, _lower = _factor_block_banded_cholesky(
                diag_blocks,
                upper_blocks,
                row_upper_bandwidths,
                row_lower_bandwidths,
            )
            return _block_banded_logdet(chol_diag)

        direct_diag_bar, direct_upper_bar = jax.grad(_packed_logdet, argnums=(0, 1))(diag, upper)
        chol_diag, lower = _factor_block_profile_cholesky(
            diag,
            upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        diag_bar, upper_bar = block_profile_logdet_packed_cotangent(
            chol_diag,
            lower,
            row_upper_bandwidths,
            row_lower_bandwidths,
            scale=jnp.array(1.0, dtype=diag.dtype),
        )

        np.testing.assert_allclose(
            np.asarray(diag_bar), np.asarray(direct_diag_bar), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(upper_bar),
            np.asarray(direct_upper_bar),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_laplace_backend_reuses_linear_summary_mode_cache_across_runtime_evals(
        self, monkeypatch
    ):
        support = make_observation_support_runtime(
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

        seen_inits: list[np.ndarray | None] = []
        returned_mode = jnp.array([[0.1, 0.0], [0.2, 0.1], [0.3, 0.4]], dtype=jnp.float32)

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._should_use_dense_support_laplace",
            lambda **_kwargs: False,
        )

        def _fake_linear_summary_laplace(*_args, z_init=None, **_kwargs):
            seen_inits.append(None if z_init is None else np.asarray(z_init))
            return returned_mode, jnp.array(-1.0, dtype=jnp.float32), {}

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._linear_summary_augmented_ieks_laplace",
            _fake_linear_summary_laplace,
        )

        ll_0, _aux_0 = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        ll_1, _aux_1 = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert float(ll_0) == pytest.approx(-1.0)
        assert float(ll_1) == pytest.approx(-1.0)
        assert seen_inits[0] is None
        np.testing.assert_allclose(seen_inits[1], np.asarray(returned_mode))

    def test_laplace_backend_linear_summary_mode_init_explicitly_overrides_cache(self, monkeypatch):
        support = make_observation_support_runtime(
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

        seen_inits: list[np.ndarray | None] = []
        cached_mode = jnp.array([[0.1, 0.0], [0.2, 0.1], [0.3, 0.4]], dtype=jnp.float32)
        explicit_mode = jnp.array([[0.9, 0.3], [0.8, 0.2], [0.7, 0.1]], dtype=jnp.float32)

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._should_use_dense_support_laplace",
            lambda **_kwargs: False,
        )

        def _fake_linear_summary_laplace(*_args, z_init=None, **_kwargs):
            seen_inits.append(None if z_init is None else np.asarray(z_init))
            return cached_mode, jnp.array(-1.0, dtype=jnp.float32), {}

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._linear_summary_augmented_ieks_laplace",
            _fake_linear_summary_laplace,
        )

        backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            latent_mode_init=explicit_mode,
        )

        assert seen_inits[0] is None
        np.testing.assert_allclose(seen_inits[1], np.asarray(explicit_mode))

    def test_laplace_backend_reuses_point_mode_cache_across_runtime_evals(self, monkeypatch):
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
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
        observations = jnp.array([[0.0], [0.25], [0.5]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        seen_inits: list[np.ndarray | None] = []
        returned_mode = jnp.array([[0.1], [0.2], [0.3]], dtype=jnp.float32)

        def _fake_ieks(*_args, z_init=None, **_kwargs):
            seen_inits.append(None if z_init is None else np.asarray(z_init))
            return returned_mode, jnp.array(-1.0, dtype=jnp.float32), {}

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._ieks_smooth",
            _fake_ieks,
        )

        ll_0, _aux_0 = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        ll_1, _aux_1 = backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert float(ll_0) == pytest.approx(-1.0)
        assert float(ll_1) == pytest.approx(-1.0)
        assert seen_inits[0] is None
        np.testing.assert_allclose(seen_inits[1], np.asarray(returned_mode))

    def test_laplace_backend_caches_support_window_derivative_builders(self, monkeypatch):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            summary_operators=["std"],
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

        derivative_calls: list[int] = []
        sentinel_window_derivatives = object()

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._should_use_dense_support_laplace",
            lambda **_kwargs: False,
        )

        def _fake_make_support_window_derivatives(**_kwargs):
            derivative_calls.append(1)
            return sentinel_window_derivatives

        def _fake_support_laplace(*_args, window_derivatives=None, z_init=None, **_kwargs):
            del z_init
            assert window_derivatives == (sentinel_window_derivatives,)
            return (
                jnp.array(-1.0, dtype=jnp.float32),
                jnp.array([[0.1], [0.2], [0.3]], dtype=jnp.float32),
                {},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._make_support_window_derivatives",
            _fake_make_support_window_derivatives,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._support_aware_ieks_laplace",
            _fake_support_laplace,
        )

        backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        backend.compute_log_likelihood_with_aux(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert derivative_calls == [1]

    def test_laplace_backend_prefers_linear_summary_augmentation_when_available(self, monkeypatch):
        support = make_observation_support_runtime(
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

        def _fake_linear(*args, **kwargs):
            calls.append("linear")
            return (
                jnp.zeros((3, 2), dtype=jnp.float32),
                jnp.array(-1.0, dtype=jnp.float32),
                {"solver_kind": jnp.asarray(2, dtype=jnp.int32)},
            )

        def _forbidden_dense(*args, **kwargs):
            raise AssertionError("eligible linear interval summaries should not use dense support")

        def _forbidden_banded(*args, **kwargs):
            calls.append("banded")
            raise AssertionError("eligible linear interval summaries should not use banded support")

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._linear_summary_augmented_ieks_laplace",
            _fake_linear,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._dense_support_laplace_log_lik",
            _forbidden_dense,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.targets.laplace._support_aware_ieks_laplace",
            _forbidden_banded,
        )

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert float(ll) == pytest.approx(-1.0)
        assert calls == ["linear"]

    def test_laplace_backend_interval_support_path_handles_large_float64_windows(self):
        n_latent = 10
        n_manifest = 10
        n_time = 18
        anchor_times = np.arange(n_time, dtype=np.float64)
        support_start = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
        support_end = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
        interval_prev = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
        interval_curr = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
        interval_weights = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
        emission_slots = np.full((n_time, n_manifest), -1, dtype=np.int64)

        for t in range(1, n_time):
            support_start[t, :] = anchor_times[t - 1]
            support_end[t, :] = anchor_times[t]
            interval_prev[t, :, 0] = 0.5
            interval_curr[t, :, 0] = 0.5
            interval_weights[t, :, 0] = 1.0
            emission_slots[t, :] = 0

        support = ObservationSupportRuntime(
            anchor_times=anchor_times,
            manifest_names=[f"y{i}" for i in range(n_manifest)],
            support_kinds=["interval"] * n_manifest,
            summary_operators=["mean"] * n_manifest,
            anchor_policies=["support_end"] * n_manifest,
            observation_windows=["1d"] * n_manifest,
            support_start_times=support_start,
            support_end_times=support_end,
            interval_prev_coeffs=interval_prev,
            interval_curr_coeffs=interval_curr,
            interval_weights=interval_weights,
            emission_slot_indices=emission_slots,
        )
        backend = LaplaceLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            manifest_dists=[DistributionFamily.GAUSSIAN] * n_manifest,
            manifest_links=[LinkFunction.IDENTITY] * n_manifest,
            n_ieks_iters=2,
            observation_support=support,
        )
        observations = jnp.zeros((n_time, n_manifest), dtype=jnp.float32).at[0].set(jnp.nan)
        time_intervals = jnp.ones((n_time,), dtype=jnp.float32)
        ct_params = CTParams(
            drift=-0.2 * jnp.eye(n_latent, dtype=jnp.float32),
            diffusion_cov=jnp.diag(jnp.linspace(0.01, 0.03, n_latent, dtype=jnp.float32)),
            cint=jnp.zeros(n_latent, dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent, dtype=jnp.float32),
            manifest_means=jnp.zeros(n_manifest, dtype=jnp.float32),
            manifest_cov=jnp.diag(jnp.linspace(0.05, 0.09, n_manifest, dtype=jnp.float32)),
        )
        init = InitialStateParams(
            mean=jnp.zeros(n_latent, dtype=jnp.float32),
            cov=jnp.eye(n_latent, dtype=jnp.float32),
        )

        assert _should_use_dense_support_laplace(n_time=n_time, n_latent=n_latent) is False

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert jnp.isfinite(ll)


class TestParticleMissingData:
    """Tests for Gaussian observation masking in PF adapter."""

    def test_missing_dimension_not_penalized(self):
        """Missing dims should not incur huge log-det penalties."""
        n_latent, n_manifest = 1, 2
        adapter = SSMAdapter(
            n_latent,
            n_manifest,
            manifest_dists=[DistributionFamily.GAUSSIAN] * n_manifest,
            diffusion_dists=[DistributionFamily.GAUSSIAN],
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
            **diagonal_diffusion_kwargs(1),
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
            **diagonal_diffusion_kwargs(1),
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

        assert "drift_base_decay_free" in site_info
        assert "manifest_var_diag_free" in site_info


class TestSVIBackend:
    """Tests specific to SVI inference backend."""

    def test_svi_rejects_nonfinite_losses(self, monkeypatch):
        """SVI should fail fast instead of returning a numerically invalid fit."""
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
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

        monkeypatch.setattr("causal_ssm_agent.models.ssm.inference.methods.svi.SVI", FakeSVI)

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
            **diagonal_diffusion_kwargs(1),
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

        monkeypatch.setattr("causal_ssm_agent.models.ssm.inference.methods.svi.SVI", FakeSVI)
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.svi.Predictive", FakePredictive
        )

        with pytest.raises(FloatingPointError, match="non-finite posterior samples"):
            fit(
                model,
                observations=observations,
                times=times,
                method="svi",
                num_steps=2,
                num_samples=1,
            )


# =============================================================================
# SVI Parameter Recovery
# =============================================================================


class TestDefaultMethodRouting:
    """Regression tests for default inference routing."""

    @staticmethod
    def _non_point_support() -> ObservationSupportRuntime:
        return make_observation_support_runtime(
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

    def test_default_always_routes_to_aux_gibbs(self):
        """Default routing resolves to auxiliary Gibbs for all model types."""
        from causal_ssm_agent.models.ssm.inference.structure import plan_inference_structure

        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
        )

        plan_kalman = plan_inference_structure(spec, likelihood="kalman")
        assert plan_kalman.resolved_method == "aux_gibbs"

        plan_particle = plan_inference_structure(spec, likelihood="particle")
        assert plan_particle.resolved_method == "aux_gibbs"

    def test_fit_without_method_dispatches_to_aux_gibbs(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
        )
        model = SSMModel(spec, likelihood="kalman")
        observations = jnp.zeros((2, 1), dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_aux_gibbs(_model, _observations, _times, **kwargs):
            return InferenceResult(
                _samples={"drift_base_decay_free": jnp.zeros((1, 1), dtype=jnp.float32)},
                method="aux_gibbs",
                diagnostics={"kwargs": kwargs},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.aux_gibbs.fit_aux_gibbs",
            fake_fit_aux_gibbs,
        )

        result = fit(model, observations=observations, times=times)

        assert result.method == "aux_gibbs"

    def test_non_point_support_allows_map(self, monkeypatch):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
        )
        model = SSMModel(spec, likelihood="particle")
        model.set_observation_support(self._non_point_support())
        observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
        times = jnp.array([0.0, 1.0], dtype=jnp.float32)

        def fake_fit_map(_model, _observations, _times, **kwargs):
            return InferenceResult(
                _samples={"drift_base_decay_free": jnp.zeros((1, 1), dtype=jnp.float32)},
                method="map",
                diagnostics={"kwargs": kwargs},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm.inference.methods.map.fit_map",
            fake_fit_map,
        )

        result = fit(model, observations=observations, times=times, method="map")

        assert result.method == "map"


def test_map_optimizer_smoke_on_small_kalman_model():
    spec = make_ssm_spec(
        n_latent=1,
        n_manifest=1,
        drift=jnp.array([[-0.4]], dtype=jnp.float32),
        drift_diag_mask=np.array([False]),
        drift_offdiag_mask=np.zeros((1, 1), dtype=bool),
        lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
        lambda_mask=np.zeros((1, 1), dtype=bool),
        manifest_means=jnp.array([0.0], dtype=jnp.float32),
        manifest_means_mask=np.array([False]),
        t0_means=jnp.array([0.0], dtype=jnp.float32),
        t0_means_mask=np.array([False]),
        t0_chol=jnp.array([[1.0]], dtype=jnp.float32),
        t0_chol_diag_mask=np.array([True]),
        t0_correlation_mask=np.zeros((1, 1), dtype=bool),
        manifest_chol=jnp.array([[0.0]], dtype=jnp.float32),
        manifest_chol_diag_mask=np.array([True]),
        **diagonal_diffusion_kwargs(1),
    )
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method="map",
        num_samples=6,
        n_ieks_iters=2,
        maxiter=5,
        n_init_samples=4,
        seed=0,
    )

    assert result.method == "map"
    assert result.diagnostics["optimizer"] == "L-BFGS-B"
    assert np.isfinite(result.diagnostics["mode_log_likelihood"])
    assert np.isfinite(result.diagnostics["mode_log_posterior"])

    samples = result.get_samples()
    assert samples["diffusion_diag_free"].shape == (6, 1)
    assert samples["manifest_var_diag_free"].shape == (6, 1)
    assert samples["t0_var_diag_free"].shape == (6, 1)
    assert bool(jnp.isfinite(samples["diffusion_diag_free"]).all())
    assert bool(jnp.isfinite(samples["manifest_var_diag_free"]).all())
    assert bool(jnp.isfinite(samples["t0_var_diag_free"]).all())


def _make_aux_gibbs_smoke_spec(**overrides):
    kwargs = {
        "n_latent": 1,
        "n_manifest": 1,
        "drift": jnp.array([[-0.4]], dtype=jnp.float32),
        "drift_diag_mask": np.array([False]),
        "drift_offdiag_mask": np.zeros((1, 1), dtype=bool),
        "lambda_mat": jnp.array([[1.0]], dtype=jnp.float32),
        "lambda_mask": np.zeros((1, 1), dtype=bool),
        "manifest_means": jnp.array([0.0], dtype=jnp.float32),
        "manifest_means_mask": np.array([False]),
        "t0_means": jnp.array([0.0], dtype=jnp.float32),
        "t0_means_mask": np.array([False]),
        "t0_chol": jnp.array([[1.0]], dtype=jnp.float32),
        "t0_chol_diag_mask": np.array([True]),
        "t0_correlation_mask": np.zeros((1, 1), dtype=bool),
        "manifest_chol": jnp.array([[0.0]], dtype=jnp.float32),
        "manifest_chol_diag_mask": np.array([True]),
        **diagonal_diffusion_kwargs(1),
    }
    kwargs.update(overrides)
    return make_ssm_spec(**kwargs)


def _small_kalman_observations_and_times():
    return (
        jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32),
        jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
    )


def _assert_small_particle_mcmc_result(result, *, method: str, num_samples: int) -> None:
    assert result.method == method
    assert method in result.diagnostics
    samples = result.get_samples()
    assert samples["diffusion_diag_free"].shape == (num_samples, 1)
    assert samples["manifest_var_diag_free"].shape == (num_samples, 1)
    assert samples["t0_var_diag_free"].shape == (num_samples, 1)
    assert bool(jnp.isfinite(samples["diffusion_diag_free"]).all())
    assert bool(jnp.isfinite(samples["manifest_var_diag_free"]).all())
    assert bool(jnp.isfinite(samples["t0_var_diag_free"]).all())
    latent_summary = result.get_latent_posterior_summary()
    assert latent_summary is not None
    assert latent_summary["mean"].shape == (3, 1)
    assert bool(jnp.isfinite(latent_summary["mean"]).all())
    latent_paths = result.get_latent_paths()
    assert latent_paths is not None
    assert latent_paths.shape == (1, num_samples, 3, 1)


@pytest.mark.parametrize(
    ("method", "num_warmup", "num_samples", "seed", "extra_kwargs"),
    [
        (
            "aux_gibbs",
            8,
            10,
            0,
            {"latent_delta": 0.2, "param_step_size": 0.03, "init_scale": 0.01},
        ),
        (
            "particle_mgrad",
            6,
            8,
            17,
            {
                "latent_delta": 0.2,
                "latent_target_accept": 0.5,
                "n_particles": 6,
                "param_step_size": 0.03,
                "init_scale": 0.01,
            },
        ),
    ],
    ids=["aux-gibbs", "particle-mgrad"],
)
def test_particle_mcmc_smoke_on_small_kalman_model(
    method,
    num_warmup,
    num_samples,
    seed,
    extra_kwargs,
):
    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations, times = _small_kalman_observations_and_times()

    result = fit(
        model,
        observations=observations,
        times=times,
        method=method,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        seed=seed,
        retain_latent_paths=True,
        **extra_kwargs,
    )

    _assert_small_particle_mcmc_result(result, method=method, num_samples=num_samples)


def test_aux_gibbs_eq10_11_smoke_on_small_kalman_model():
    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=8,
        num_samples=10,
        num_chains=1,
        seed=11,
        latent_proposal_family="eq10_11",
        latent_delta=0.2,
        param_step_size=0.03,
        init_scale=0.01,
        retain_latent_paths=True,
    )

    assert result.method == "aux_gibbs"
    aux_diag = result.diagnostics["aux_gibbs"]
    assert aux_diag["latent_proposal_family"] == "eq10_11"
    samples = result.get_samples()
    assert samples["diffusion_diag_free"].shape == (10, 1)
    assert samples["manifest_var_diag_free"].shape == (10, 1)
    assert samples["t0_var_diag_free"].shape == (10, 1)
    assert bool(jnp.isfinite(samples["diffusion_diag_free"]).all())
    assert bool(jnp.isfinite(samples["manifest_var_diag_free"]).all())
    assert bool(jnp.isfinite(samples["t0_var_diag_free"]).all())
    latent_summary = result.get_latent_posterior_summary()
    assert latent_summary is not None
    assert latent_summary["mean"].shape == (3, 1)
    assert bool(jnp.isfinite(latent_summary["mean"]).all())
    latent_paths = result.get_latent_paths()
    assert latent_paths is not None
    assert latent_paths.shape == (1, 10, 3, 1)


def test_aux_gibbs_can_skip_latent_posterior_summary():
    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=4,
        num_samples=6,
        num_chains=1,
        seed=7,
        latent_delta=0.2,
        param_step_size=0.03,
        init_scale=0.01,
        compute_latent_posterior_summary=False,
    )

    assert result.get_latent_posterior_summary() is None
    history = result.diagnostics["chain_complete_log_posterior_history"]
    assert history.shape == (1, 6)
    assert bool(jnp.isfinite(history).all())


def test_aux_gibbs_dual_averaging_smoke_on_small_kalman_model():
    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=4,
        num_samples=5,
        num_chains=2,
        seed=9,
        latent_delta=0.2,
        param_step_size=0.03,
        init_scale=0.01,
        adaptation_scheme="dual_averaging",
    )

    diag = result.diagnostics["aux_gibbs"]
    assert diag["adaptation_scheme"] == "dual_averaging"
    assert bool(jnp.isfinite(result.get_samples()["diffusion_diag_free"]).all())


def test_aux_gibbs_pathfinder_auto_preconditioner_reuses_pathfinder_run(monkeypatch):
    from causal_ssm_agent.models.ssm.inference.trajectory_mcmc import build_auxiliary_kalman_bundle

    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    aux_bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=random.PRNGKey(0),
        reparam=None,
    )
    flat_example = aux_bundle["flat_example"]
    dim = int(aux_bundle["dim"])
    fake_state = SimpleNamespace(
        mean=flat_example,
        chol=jnp.eye(dim, dtype=flat_example.dtype) * 0.25,
        best_elbo=3.0,
    )
    pathfinder_runs = {"count": 0}

    def _fake_run_pathfinder_approximations(*_args, **_kwargs):
        pathfinder_runs["count"] += 1
        return fake_state, {
            "n_pathfinder_starts": 2,
            "n_pathfinder_starts_finite": 1,
            "best_pathfinder_elbo": 3.0,
            "pathfinder_elbo": 3.0,
            "pathfinder_elbo_min": 3.0,
            "pathfinder_elbo_max": 3.0,
            "pathfinder_elbo_spread": 0.0,
        }

    def _unexpected_fit_map(*_args, **_kwargs):
        raise AssertionError("fit_map should not run when auto_preconditioner_method='pathfinder'.")

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.aux_gibbs._run_pathfinder_approximation",
        _fake_run_pathfinder_approximations,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.aux_gibbs.fit_map",
        _unexpected_fit_map,
    )

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=4,
        num_samples=5,
        num_chains=2,
        seed=13,
        latent_delta=0.2,
        param_step_size=0.03,
        init_scale=0.01,
        auto_preconditioner_method="pathfinder",
        pathfinder_preconditioner_low_rank_scale=0.5,
    )

    aux_diag = result.diagnostics["aux_gibbs"]
    assert pathfinder_runs["count"] == 1
    assert aux_diag["init_method"] == "pathfinder"
    assert aux_diag["pathfinder_sampling_mode"] == "pathfinder_gaussian"
    assert aux_diag["pathfinder_init_scale"] is None
    assert aux_diag["auto_preconditioner"] is True
    assert aux_diag["auto_preconditioner_method"] == "pathfinder"
    assert aux_diag["auto_preconditioner_n_pathfinder_starts"] == 2
    assert aux_diag["auto_preconditioner_n_pathfinder_starts_finite"] == 1
    assert aux_diag["auto_preconditioner_best_pathfinder_elbo"] == pytest.approx(3.0)


def test_aux_gibbs_pathfinder_init_scale_switches_sampling_mode(monkeypatch):
    from causal_ssm_agent.models.ssm.inference.trajectory_mcmc import build_auxiliary_kalman_bundle

    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    aux_bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=random.PRNGKey(0),
        reparam=None,
    )
    flat_example = aux_bundle["flat_example"]
    dim = int(aux_bundle["dim"])
    fake_state = SimpleNamespace(
        mean=flat_example + jnp.full((dim,), 1.5, dtype=flat_example.dtype),
        chol=jnp.eye(dim, dtype=flat_example.dtype) * 0.4,
        best_elbo=3.0,
    )
    pathfinder_runs = {"count": 0}

    def _fake_run_pathfinder_approximations(*_args, **_kwargs):
        pathfinder_runs["count"] += 1
        return fake_state, {
            "n_pathfinder_starts": 3,
            "n_pathfinder_starts_finite": 3,
            "best_pathfinder_elbo": 3.0,
            "pathfinder_elbo": 3.0,
            "pathfinder_elbo_min": 2.8,
            "pathfinder_elbo_max": 3.0,
            "pathfinder_elbo_spread": 0.2,
        }

    def _unexpected_fit_map(*_args, **_kwargs):
        raise AssertionError("fit_map should not run when auto_preconditioner_method='pathfinder'.")

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.aux_gibbs._run_pathfinder_approximation",
        _fake_run_pathfinder_approximations,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.aux_gibbs.fit_map",
        _unexpected_fit_map,
    )

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=4,
        num_samples=5,
        num_chains=4,
        seed=17,
        latent_delta=0.2,
        param_step_size=0.03,
        init_scale=0.01,
        auto_preconditioner_method="pathfinder",
        pathfinder_init_scale=0.15,
        n_pathfinder_starts=3,
    )

    aux_diag = result.diagnostics["aux_gibbs"]
    assert pathfinder_runs["count"] == 1
    assert aux_diag["pathfinder_sampling_mode"] == "mode_plus_scaled_normal"
    assert aux_diag["pathfinder_init_scale"] == pytest.approx(0.15)
    assert aux_diag["auto_preconditioner_n_pathfinder_starts"] == 3
    assert aux_diag["auto_preconditioner_n_pathfinder_starts_finite"] == 3
    assert aux_diag["auto_preconditioner_best_pathfinder_elbo"] == pytest.approx(3.0)
    assert aux_diag["auto_preconditioner_pathfinder_elbo_spread"] == pytest.approx(0.2)
    assert bool(jnp.isfinite(result.get_samples()["diffusion_diag_free"]).all())


def test_particle_mgrad_dual_averaging_supports_pathfinder_init(monkeypatch):
    spec = _make_aux_gibbs_smoke_spec()
    model = SSMModel(spec, likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    latent_delta_min = 0.05
    latent_delta_max = 0.12

    def _fake_pathfinder_approximate(_key, _log_posterior_fn, flat_example, **_kwargs):
        return SimpleNamespace(
            position=flat_example, elbo=jnp.array(0.0, dtype=flat_example.dtype)
        ), {}

    def _fake_pathfinder_sample(_key, state, *, num_samples):
        base = jnp.broadcast_to(state.position, (num_samples, state.position.shape[0]))
        offsets = jnp.arange(num_samples, dtype=base.dtype)[:, None] * 0.01
        return base + offsets, jnp.zeros((num_samples,), dtype=base.dtype)

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.particle_mgrad.pathfinder.approximate",
        _fake_pathfinder_approximate,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.particle_mgrad.pathfinder.sample",
        _fake_pathfinder_sample,
    )

    result = fit(
        model,
        observations=observations,
        times=times,
        method="particle_mgrad",
        num_warmup=4,
        num_samples=5,
        num_chains=2,
        seed=11,
        latent_delta=0.2,
        latent_target_accept=0.5,
        n_particles=6,
        param_step_size=0.03,
        init_scale=0.01,
        adaptation_scheme="dual_averaging",
        init_method="pathfinder",
        latent_delta_min=latent_delta_min,
        latent_delta_max=latent_delta_max,
    )

    diag = result.diagnostics["particle_mgrad"]
    assert diag["adaptation_scheme"] == "dual_averaging"
    assert diag["init_method"] == "pathfinder"
    final_latent_delta = np.asarray(diag["final_latent_delta"])
    assert final_latent_delta.shape == (2, 3)
    assert np.all(final_latent_delta >= latent_delta_min)
    assert np.all(final_latent_delta <= latent_delta_max)


@pytest.mark.parametrize(
    ("method", "num_warmup", "seed", "extra_kwargs"),
    [
        ("aux_gibbs", 6, 1, {"latent_delta": 0.2, "param_step_size": 0.03, "init_scale": 0.01}),
        (
            "particle_mgrad",
            4,
            18,
            {
                "latent_delta": 0.2,
                "latent_target_accept": 0.5,
                "n_particles": 6,
                "param_step_size": 0.03,
                "init_scale": 0.01,
            },
        ),
    ],
    ids=["aux-gibbs", "particle-mgrad"],
)
def test_particle_mcmc_multi_chain_diagnostics(method, num_warmup, seed, extra_kwargs):
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method=method,
        num_warmup=num_warmup,
        num_samples=5,
        num_chains=2,
        seed=seed,
        **extra_kwargs,
    )

    samples = result.get_samples()
    assert samples["diffusion_diag_free"].shape == (10, 1)
    diag = result.get_mcmc_diagnostics()
    assert diag is not None
    assert diag["num_chains"] == 2
    assert diag["num_samples"] == 5
    assert "latent_accept_prob_mean" in diag
    assert "parameter_accept_prob_mean" in diag
    assert "trace_data" in diag
    assert "rank_histograms" in diag


@pytest.mark.parametrize(
    ("method", "seed", "extra_kwargs"),
    [
        ("aux_gibbs", 2, {"latent_delta": 0.15, "param_step_size": 0.03, "init_scale": 0.01}),
        (
            "particle_mgrad",
            19,
            {
                "latent_delta": 0.15,
                "latent_target_accept": 0.5,
                "n_particles": 6,
                "param_step_size": 0.03,
                "init_scale": 0.01,
            },
        ),
    ],
    ids=["aux-gibbs", "particle-mgrad"],
)
def test_particle_mcmc_support_aware_interval_summary_smoke(method, seed, extra_kwargs):
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="particle")
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["y"],
        support_kinds=["interval"],
        observation_windows=["1d"],
        support_start_times=np.array([[np.nan], [0.0]]),
        support_end_times=np.array([[np.nan], [1.0]]),
        interval_prev_coeffs=np.array([[0.0], [0.5]], dtype=np.float32),
        interval_curr_coeffs=np.array([[0.0], [0.5]], dtype=np.float32),
        interval_weights=np.array([[0.0], [1.0]], dtype=np.float32),
    )
    model.set_observation_support(support)
    observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method=method,
        num_warmup=4,
        num_samples=6,
        num_chains=1,
        seed=seed,
        **extra_kwargs,
    )

    summary = result.get_latent_posterior_summary()
    assert summary is not None
    assert summary["mean"].shape == (2, 1)
    assert bool(jnp.isfinite(summary["mean"]).all())
    assert bool(jnp.isfinite(result.get_samples()["diffusion_diag_free"]).all())


def test_aux_gibbs_rejects_nonlinear_interval_summary_support():
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="particle")
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 1.0]),
        manifest_names=["y"],
        support_kinds=["interval"],
        summary_operators=["std"],
        observation_windows=["1d"],
        support_start_times=np.array([[np.nan], [0.0]]),
        support_end_times=np.array([[np.nan], [1.0]]),
        interval_prev_coeffs=np.array([[0.0], [0.5]], dtype=np.float32),
        interval_curr_coeffs=np.array([[0.0], [0.5]], dtype=np.float32),
        interval_weights=np.array([[0.0], [1.0]], dtype=np.float32),
    )
    model.set_observation_support(support)
    observations = jnp.array([[jnp.nan], [0.2]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="linear interval summaries"):
        fit(
            model,
            observations=observations,
            times=times,
            method="aux_gibbs",
            num_warmup=4,
            num_samples=6,
            num_chains=1,
            seed=2,
        )


def test_aux_gibbs_heterogeneous_observation_families_smoke():
    spec = _make_aux_gibbs_smoke_spec(
        n_manifest=2,
        lambda_mat=jnp.array([[1.0], [0.7]], dtype=jnp.float32),
        lambda_mask=np.zeros((2, 1), dtype=bool),
        manifest_means=jnp.array([0.0, 0.0], dtype=jnp.float32),
        manifest_means_mask=np.array([False, False]),
        manifest_chol=jnp.zeros((2, 2), dtype=jnp.float32),
        manifest_chol_diag_mask=np.array([True, True]),
        manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY],
    )
    model = SSMModel(spec, likelihood="particle")
    observations = jnp.array([[0.1, 0.2], [0.15, -0.1], [0.05, 0.12]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method="aux_gibbs",
        num_warmup=4,
        num_samples=6,
        num_chains=1,
        seed=3,
        latent_delta=0.15,
        param_step_size=0.03,
        init_scale=0.01,
    )

    summary = result.get_latent_posterior_summary()
    assert summary is not None
    assert summary["mean"].shape == (3, 1)
    assert bool(jnp.isfinite(summary["mean"]).all())
    assert bool(jnp.isfinite(result.get_samples()["diffusion_diag_free"]).all())


@pytest.mark.parametrize(
    ("method", "seed", "extra_kwargs"),
    [
        ("aux_gibbs", 4, {"latent_delta": 0.2, "param_step_size": 0.03, "init_scale": 0.01}),
        (
            "particle_mgrad",
            10,
            {
                "latent_delta": 0.2,
                "latent_target_accept": 0.5,
                "n_particles": 6,
                "param_step_size": 0.03,
                "init_scale": 0.01,
            },
        ),
    ],
    ids=["aux-gibbs", "particle-mgrad"],
)
def test_particle_mcmc_supports_fixed_centering_autoreparam(method, seed, extra_kwargs):
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    result = fit(
        model,
        observations=observations,
        times=times,
        method=method,
        num_warmup=4,
        num_samples=6,
        num_chains=1,
        seed=seed,
        reparam=AutoReparam(centered=0.0),
        **extra_kwargs,
    )

    sample_names = set(result.get_samples())
    assert all("_decentered" not in name for name in sample_names)
    diag = result.get_mcmc_diagnostics()
    assert diag is not None
    diag_names = {entry["parameter"] for entry in diag["per_parameter"]}
    assert all("_decentered" not in name for name in diag_names)


def test_aux_gibbs_rejects_unknown_latent_proposal_family():
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="latent proposal family"):
        fit(
            model,
            observations=observations,
            times=times,
            method="aux_gibbs",
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            seed=12,
            latent_proposal_family="bad_family",
        )


def test_aux_gibbs_rejects_learnable_centering_autoreparam():
    model = SSMModel(_make_aux_gibbs_smoke_spec(), likelihood="kalman")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="fixed centering"):
        fit(
            model,
            observations=observations,
            times=times,
            method="aux_gibbs",
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            seed=5,
            reparam=AutoReparam(),
        )


@pytest.mark.parametrize(
    ("method", "seed"),
    [("aux_gibbs", 6), ("particle_mgrad", 11)],
    ids=["aux-gibbs", "particle-mgrad"],
)
def test_particle_mcmc_rejects_student_t_diffusion(method, seed):
    spec = _make_aux_gibbs_smoke_spec(
        diffusion_dists=[DistributionFamily.STUDENT_T],
    )
    model = SSMModel(spec, likelihood="particle")
    observations = jnp.array([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="Gaussian latent diffusion"):
        fit(
            model,
            observations=observations,
            times=times,
            method=method,
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            seed=seed,
        )


def test_support_aware_step_halving_search_backtracks_to_improving_step():
    z_start = jnp.array([0.0], dtype=jnp.float32)
    step_direction = jnp.array([3.0], dtype=jnp.float32)

    def objective_fn(z):
        return -jnp.sum((z - 1.0) ** 2)

    z_next, objective_next, accepted, alpha = _support_aware_step_halving_search(
        z_start,
        step_direction,
        objective_fn(z_start),
        objective_fn,
        max_halvings=4,
    )

    assert bool(accepted)
    np.testing.assert_allclose(np.asarray(z_next), np.array([1.5], dtype=np.float32), atol=1e-6)
    assert float(alpha) == pytest.approx(0.5)
    assert float(objective_next) > float(objective_fn(z_start))


def test_map_support_aware_uses_exact_gradient_outer_optimizer(monkeypatch):
    observations = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)
    flat_example = jnp.array([0.25, -0.5], dtype=jnp.float32)
    captured: dict[str, object] = {}

    class _FakeModel:
        likelihood = "particle"
        observation_support = SimpleNamespace(requires_interval_summary_handling=True)
        spec = None
        _structure_runtime = None

        def make_laplace_backend(self, _n_ieks_iters):
            return SimpleNamespace()

    def fake_build_bundle(_model, _observations, _times, _trace_key, _backend, _reparam):
        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del runtime_observations, runtime_times, latent_mode_init
            return -jnp.sum((z - jnp.array([1.0, -2.0], dtype=z.dtype)) ** 2)

        def log_prior_unc_fn(z):
            return -0.1 * jnp.sum(z**2)

        def log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            ) + log_prior_unc_fn(z)

        def neg_log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return -log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )

        def neg_log_posterior_with_aux_fn(
            z,
            runtime_observations,
            runtime_times,
            latent_mode_init=None,
        ):
            del latent_mode_init
            return neg_log_posterior_fn(z, runtime_observations, runtime_times), {
                "log_posterior": log_posterior_fn(z, runtime_observations, runtime_times),
                "log_likelihood": log_lik_fn(z, runtime_observations, runtime_times),
                "log_prior": log_prior_unc_fn(z),
                "inner": {
                    "solver_kind": jnp.asarray(1, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(2, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(2, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(-3.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(-1.0, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(1e-3, dtype=jnp.float32),
                    "final_damping": jnp.asarray(1e-4, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(1.0, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.1, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(2.0, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.5, dtype=jnp.float32),
                },
                "latent_mode": jnp.asarray([[z[0], z[1]]], dtype=jnp.float32),
            }

        def forbidden_batch(_candidates):
            raise AssertionError("support-aware MAP should not batch-score init candidates")

        return {
            "dim": 2,
            "flat_example": flat_example,
            "site_info": {"theta": object()},
            "unravel_fn": lambda z: {"theta": z},
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": log_posterior_fn,
            "neg_log_posterior_fn": neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": neg_log_posterior_with_aux_fn,
            "batch_log_posterior_jit": forbidden_batch,
        }

    def forbidden_draw(*_args, **_kwargs):
        raise AssertionError("support-aware MAP should not draw init candidates")

    def fake_gradient_minimize(fun, x0, jac, method, tol, options, callback):
        captured["method"] = method
        captured["x0"] = np.asarray(x0)
        captured["tol"] = tol
        captured["options"] = dict(options)
        captured["fun_at_x0"] = float(fun(np.asarray(x0)))
        captured["jac_at_x0"] = np.asarray(jac(np.asarray(x0)))
        optimum = np.array([1.0 / 1.1, -2.0 / 1.1], dtype=np.float64)
        callback(optimum)
        return SimpleNamespace(
            x=optimum,
            fun=float(fun(optimum)),
            nit=3,
            nfev=5,
            status=0,
            success=True,
        )

    def fake_sample_posterior(
        _rng_key,
        z_mode,
        _neg_log_posterior_fn,
        _observations,
        _times,
        *,
        num_samples,
        hessian_jitter,
    ):
        del hessian_jitter
        unc_samples = jnp.broadcast_to(z_mode, (num_samples, z_mode.shape[0]))
        covariance = jnp.eye(z_mode.shape[0], dtype=z_mode.dtype)
        eigvals = jnp.ones((z_mode.shape[0],), dtype=z_mode.dtype)
        return unc_samples, covariance, eigvals

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_map_laplace_bundle",
        fake_build_bundle,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._draw_laplace_init_candidates",
        forbidden_draw,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.spo.minimize",
        fake_gradient_minimize,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._sample_laplace_parameter_posterior",
        fake_sample_posterior,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.extract_constrained_samples",
        lambda unc_samples, *_args, **_kwargs: {"theta": unc_samples},
    )

    result = fit_map(
        _FakeModel(),
        observations,
        times,
        num_samples=3,
        n_ieks_iters=2,
        maxiter=9,
        tol=1e-3,
        parameter_covariance_method="exact_hessian",
        seed=0,
    )

    assert result.method == "map"
    assert result.diagnostics["optimizer"] == "L-BFGS-B"
    np.testing.assert_allclose(captured["x0"], np.asarray(flat_example))
    assert captured["method"] == "L-BFGS-B"
    assert captured["tol"] == 1e-3
    assert captured["options"]["maxiter"] == 9
    np.testing.assert_allclose(
        captured["jac_at_x0"],
        np.array([-1.45, 2.9], dtype=np.float64),
        atol=1e-6,
    )
    assert result.diagnostics["n_function_evals"] == 5
    assert result.get_samples()["theta"].shape == (3, 2)


def test_map_generic_path_uses_multistart_lbfgsb(monkeypatch):
    observations = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)
    flat_example = jnp.array([0.25, -0.5], dtype=jnp.float32)
    captured: dict[str, object] = {}

    class _FakeModel:
        likelihood = "particle"
        observation_support = None
        spec = None
        _structure_runtime = None

        def make_laplace_backend(self, _n_ieks_iters):
            return SimpleNamespace()

    def fake_build_bundle(_model, _observations, _times, _trace_key, _backend, _reparam):
        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del runtime_observations, runtime_times, latent_mode_init
            return -jnp.sum((z - jnp.array([1.0, -2.0], dtype=z.dtype)) ** 2)

        def log_prior_unc_fn(z):
            return -0.1 * jnp.sum(z**2)

        def log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            ) + log_prior_unc_fn(z)

        def neg_log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return -log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )

        def neg_log_posterior_with_aux_fn(
            z,
            runtime_observations,
            runtime_times,
            latent_mode_init=None,
        ):
            del latent_mode_init
            return neg_log_posterior_fn(z, runtime_observations, runtime_times), {
                "log_posterior": log_posterior_fn(z, runtime_observations, runtime_times),
                "log_likelihood": log_lik_fn(z, runtime_observations, runtime_times),
                "log_prior": log_prior_unc_fn(z),
                "inner": {
                    "solver_kind": jnp.asarray(1, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(3, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(3, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(-4.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(-1.0, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(5e-4, dtype=jnp.float32),
                    "final_damping": jnp.asarray(1e-5, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(1.0, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.05, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(1.5, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.4, dtype=jnp.float32),
                },
                "latent_mode": jnp.asarray([[z[0], z[1]]], dtype=jnp.float32),
            }

        def batch_log_posterior_jit(candidates):
            return jnp.asarray([-50.0, -10.0, -1.0], dtype=jnp.float32)

        return {
            "dim": 2,
            "flat_example": flat_example,
            "site_info": {"theta": object()},
            "unravel_fn": lambda z: {"theta": z},
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": log_posterior_fn,
            "neg_log_posterior_fn": neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": neg_log_posterior_with_aux_fn,
            "batch_log_posterior_jit": batch_log_posterior_jit,
        }

    def fake_draw_candidates(_key, _site_info, *, dim, n_candidates, dtype):
        del dim, n_candidates, dtype
        return random.PRNGKey(123), jnp.array(
            [
                [0.0, 0.0],
                [4.0, 4.0],
                [1.0, -2.0],
            ],
            dtype=jnp.float32,
        )

    def fake_gradient_minimize(fun, x0, jac, method, tol, options, callback):
        captured["method"] = method
        captured["x0"] = np.asarray(x0)
        captured["tol"] = tol
        captured["options"] = dict(options)
        captured["fun_at_x0"] = float(fun(np.asarray(x0)))
        captured["jac_at_x0"] = np.asarray(jac(np.asarray(x0)))
        optimum = np.array([1.0 / 1.1, -2.0 / 1.1], dtype=np.float64)
        callback(optimum)
        return SimpleNamespace(
            x=optimum,
            fun=float(fun(optimum)),
            nit=4,
            nfev=6,
            status=0,
            success=True,
        )

    def fake_sample_posterior(
        _rng_key,
        z_mode,
        _neg_log_posterior_fn,
        _observations,
        _times,
        *,
        num_samples,
        hessian_jitter,
    ):
        del hessian_jitter
        unc_samples = jnp.broadcast_to(z_mode, (num_samples, z_mode.shape[0]))
        covariance = jnp.eye(z_mode.shape[0], dtype=z_mode.dtype)
        eigvals = jnp.ones((z_mode.shape[0],), dtype=z_mode.dtype)
        return unc_samples, covariance, eigvals

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_map_laplace_bundle",
        fake_build_bundle,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._draw_laplace_init_candidates",
        fake_draw_candidates,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.spo.minimize",
        fake_gradient_minimize,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._sample_laplace_parameter_posterior",
        fake_sample_posterior,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.extract_constrained_samples",
        lambda unc_samples, *_args, **_kwargs: {"theta": unc_samples},
    )

    result = fit_map(
        _FakeModel(),
        observations,
        times,
        num_samples=3,
        n_ieks_iters=2,
        maxiter=9,
        tol=1e-3,
        n_init_samples=2,
        parameter_covariance_method="exact_hessian",
        seed=0,
    )

    assert result.method == "map"
    assert result.diagnostics["optimizer"] == "L-BFGS-B"
    np.testing.assert_allclose(captured["x0"], np.array([1.0, -2.0], dtype=np.float64))
    assert captured["method"] == "L-BFGS-B"
    assert captured["tol"] == 1e-3
    assert captured["options"]["maxiter"] == 9
    np.testing.assert_allclose(
        captured["jac_at_x0"],
        np.array([0.2, -0.4], dtype=np.float64),
        atol=1e-6,
    )
    assert result.diagnostics["n_function_evals"] == 6


def test_map_emits_prefect_progress_logs(monkeypatch, caplog):
    observations = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)
    flat_example = jnp.array([0.25, -0.5], dtype=jnp.float32)

    class _FakeModel:
        likelihood = "particle"
        observation_support = None
        spec = None
        _structure_runtime = None

        def make_laplace_backend(self, _n_ieks_iters):
            return SimpleNamespace()

    def fake_build_bundle(_model, _observations, _times, _trace_key, _backend, _reparam):
        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del runtime_observations, runtime_times, latent_mode_init
            return -jnp.sum((z - jnp.array([1.0, -2.0], dtype=z.dtype)) ** 2)

        def log_prior_unc_fn(z):
            return -0.1 * jnp.sum(z**2)

        def log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            ) + log_prior_unc_fn(z)

        def neg_log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return -log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )

        def neg_log_posterior_with_aux_fn(
            z,
            runtime_observations,
            runtime_times,
            latent_mode_init=None,
        ):
            del latent_mode_init
            return neg_log_posterior_fn(z, runtime_observations, runtime_times), {
                "log_posterior": log_posterior_fn(z, runtime_observations, runtime_times),
                "log_likelihood": log_lik_fn(z, runtime_observations, runtime_times),
                "log_prior": log_prior_unc_fn(z),
                "inner": {
                    "solver_kind": jnp.asarray(1, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(3, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(2, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(-4.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(-1.5, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(2e-4, dtype=jnp.float32),
                    "final_damping": jnp.asarray(1e-5, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(1.0, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.05, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(1.25, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.35, dtype=jnp.float32),
                },
                "latent_mode": jnp.asarray([[z[0], z[1]]], dtype=jnp.float32),
            }

        return {
            "dim": 2,
            "flat_example": flat_example,
            "site_info": {"theta": object()},
            "unravel_fn": lambda z: {"theta": z},
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": log_posterior_fn,
            "neg_log_posterior_fn": neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": neg_log_posterior_with_aux_fn,
            "batch_log_posterior_jit": lambda _candidates: jnp.array(
                [-10.0, -1.0], dtype=jnp.float32
            ),
        }

    def fake_draw_candidates(_key, _site_info, *, dim, n_candidates, dtype):
        del dim, n_candidates, dtype
        return random.PRNGKey(123), jnp.array(
            [
                [0.0, 0.0],
                [1.0, -2.0],
            ],
            dtype=jnp.float32,
        )

    def fake_gradient_minimize(fun, x0, jac, method, tol, options, callback):
        del jac, method, tol, options
        mid = np.array([0.8, -1.6], dtype=np.float64)
        callback(mid)
        optimum = np.array([1.0 / 1.1, -2.0 / 1.1], dtype=np.float64)
        callback(optimum)
        return SimpleNamespace(
            x=optimum,
            fun=float(fun(optimum)),
            nit=2,
            nfev=4,
            status=0,
            success=True,
        )

    def fake_sample_posterior(
        _rng_key,
        z_mode,
        _neg_log_posterior_fn,
        _observations,
        _times,
        *,
        num_samples,
        hessian_jitter,
    ):
        del hessian_jitter
        unc_samples = jnp.broadcast_to(z_mode, (num_samples, z_mode.shape[0]))
        covariance = jnp.eye(z_mode.shape[0], dtype=z_mode.dtype)
        eigvals = jnp.array([0.25, 2.0], dtype=z_mode.dtype)
        return unc_samples, covariance, eigvals

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_map_laplace_bundle",
        fake_build_bundle,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._draw_laplace_init_candidates",
        fake_draw_candidates,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.spo.minimize",
        fake_gradient_minimize,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._sample_laplace_parameter_posterior",
        fake_sample_posterior,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.extract_constrained_samples",
        lambda unc_samples, *_args, **_kwargs: {"theta": unc_samples},
    )

    with caplog.at_level(
        logging.INFO,
        logger="causal_ssm_agent.models.ssm.inference.methods.map",
    ):
        fit_map(
            _FakeModel(),
            observations,
            times,
            num_samples=3,
            n_ieks_iters=2,
            maxiter=9,
            tol=1e-3,
            parameter_covariance_method="exact_hessian",
            seed=0,
        )

    assert "MAP phase start: phase=build_likelihood_backend" in caplog.text
    assert "MAP phase complete: phase=build_bundle" in caplog.text
    assert "MAP outer init:" in caplog.text
    assert "MAP outer iter 1:" in caplog.text
    assert "MAP inner iter 1:" in caplog.text
    assert "MAP outer mode:" in caplog.text
    assert "MAP parameter curvature exact_hessian:" in caplog.text


def test_map_can_skip_parameter_hessian(monkeypatch):
    observations = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)
    flat_example = jnp.array([0.25, -0.5], dtype=jnp.float32)

    class _FakeModel:
        likelihood = "particle"
        observation_support = None
        spec = None
        _structure_runtime = None

        def make_laplace_backend(self, _n_ieks_iters):
            return SimpleNamespace()

    def fake_build_bundle(_model, _observations, _times, _trace_key, _backend, _reparam):
        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del runtime_observations, runtime_times, latent_mode_init
            return -jnp.sum((z - jnp.array([1.0, -2.0], dtype=z.dtype)) ** 2)

        def log_prior_unc_fn(z):
            return -0.1 * jnp.sum(z**2)

        def log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            ) + log_prior_unc_fn(z)

        def neg_log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return -log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )

        def neg_log_posterior_with_aux_fn(
            z,
            runtime_observations,
            runtime_times,
            latent_mode_init=None,
        ):
            del latent_mode_init
            return neg_log_posterior_fn(z, runtime_observations, runtime_times), {
                "log_posterior": log_posterior_fn(z, runtime_observations, runtime_times),
                "log_likelihood": log_lik_fn(z, runtime_observations, runtime_times),
                "log_prior": log_prior_unc_fn(z),
                "inner": {
                    "solver_kind": jnp.asarray(1, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(1, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(1, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(-2.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(-1.0, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(1e-2, dtype=jnp.float32),
                    "final_damping": jnp.asarray(1e-3, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(0.5, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.2, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(1.0, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.25, dtype=jnp.float32),
                },
                "latent_mode": jnp.asarray([[z[0], z[1]]], dtype=jnp.float32),
            }

        return {
            "dim": 2,
            "flat_example": flat_example,
            "site_info": {"theta": object()},
            "unravel_fn": lambda z: {"theta": z},
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": log_posterior_fn,
            "neg_log_posterior_fn": neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": neg_log_posterior_with_aux_fn,
            "batch_log_posterior_jit": lambda _candidates: jnp.array(
                [-10.0, -1.0], dtype=jnp.float32
            ),
        }

    def fake_draw_candidates(_key, _site_info, *, dim, n_candidates, dtype):
        del dim, n_candidates, dtype
        return random.PRNGKey(123), jnp.array(
            [
                [0.0, 0.0],
                [1.0, -2.0],
            ],
            dtype=jnp.float32,
        )

    def fake_gradient_minimize(fun, x0, jac, method, tol, options, callback):
        del fun, jac, method, tol, options
        callback(np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64),
            fun=0.0,
            nit=1,
            nfev=1,
            status=0,
            success=True,
        )

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_map_laplace_bundle",
        fake_build_bundle,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._draw_laplace_init_candidates",
        fake_draw_candidates,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.spo.minimize",
        fake_gradient_minimize,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._sample_laplace_parameter_posterior",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parameter hessian path should be skipped")
        ),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.extract_constrained_samples",
        lambda unc_samples, *_args, **_kwargs: {"theta": unc_samples},
    )

    result = fit_map(
        _FakeModel(),
        observations,
        times,
        num_samples=4,
        compute_parameter_hessian=False,
        seed=0,
    )

    assert result.method == "map"
    assert result.diagnostics["compute_parameter_hessian"] is False
    assert result.diagnostics["parameter_posterior_strategy"] == "mode_only"
    assert result.diagnostics["hessian_condition_number"] is None
    np.testing.assert_allclose(
        np.asarray(result.diagnostics["covariance_diag"]),
        np.zeros((2,), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(result.get_samples()["theta"]),
        np.broadcast_to(np.array([1.0, -2.0], dtype=np.float32), (4, 2)),
    )


def test_map_can_use_optimizer_hess_inv_covariance(monkeypatch):
    observations = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    times = jnp.array([0.0, 1.0], dtype=jnp.float32)
    flat_example = jnp.array([0.25, -0.5], dtype=jnp.float32)

    class _FakeModel:
        likelihood = "particle"
        observation_support = None
        spec = None
        _structure_runtime = None

        def make_laplace_backend(self, _n_ieks_iters):
            return SimpleNamespace()

    def fake_build_bundle(_model, _observations, _times, _trace_key, _backend, _reparam):
        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del runtime_observations, runtime_times, latent_mode_init
            return -jnp.sum((z - jnp.array([1.0, -2.0], dtype=z.dtype)) ** 2)

        def log_prior_unc_fn(z):
            return -0.1 * jnp.sum(z**2)

        def log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            ) + log_prior_unc_fn(z)

        def neg_log_posterior_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            return -log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )

        def neg_log_posterior_with_aux_fn(
            z,
            runtime_observations,
            runtime_times,
            latent_mode_init=None,
        ):
            del latent_mode_init
            return neg_log_posterior_fn(z, runtime_observations, runtime_times), {
                "log_posterior": log_posterior_fn(z, runtime_observations, runtime_times),
                "log_likelihood": log_lik_fn(z, runtime_observations, runtime_times),
                "log_prior": log_prior_unc_fn(z),
                "inner": {
                    "solver_kind": jnp.asarray(1, dtype=jnp.int32),
                    "n_iterations": jnp.asarray(1, dtype=jnp.int32),
                    "n_accepted_steps": jnp.asarray(1, dtype=jnp.int32),
                    "init_log_joint": jnp.asarray(-2.0, dtype=jnp.float32),
                    "final_log_joint": jnp.asarray(-1.0, dtype=jnp.float32),
                    "final_rel_change": jnp.asarray(1e-2, dtype=jnp.float32),
                    "final_damping": jnp.asarray(1e-3, dtype=jnp.float32),
                    "final_step_alpha": jnp.asarray(0.5, dtype=jnp.float32),
                    "final_step_norm": jnp.asarray(0.2, dtype=jnp.float32),
                    "laplace_logdet": jnp.asarray(1.0, dtype=jnp.float32),
                    "min_chol_diag": jnp.asarray(0.25, dtype=jnp.float32),
                },
                "latent_mode": jnp.asarray([[z[0], z[1]]], dtype=jnp.float32),
            }

        return {
            "dim": 2,
            "flat_example": flat_example,
            "site_info": {"theta": object()},
            "unravel_fn": lambda z: {"theta": z},
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": log_posterior_fn,
            "neg_log_posterior_fn": neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": neg_log_posterior_with_aux_fn,
            "batch_log_posterior_jit": lambda _candidates: jnp.array(
                [-10.0, -1.0], dtype=jnp.float32
            ),
        }

    def fake_draw_candidates(_key, _site_info, *, dim, n_candidates, dtype):
        del dim, n_candidates, dtype
        return random.PRNGKey(123), jnp.array(
            [
                [0.0, 0.0],
                [1.0, -2.0],
            ],
            dtype=jnp.float32,
        )

    class _FakeInvHess:
        def todense(self):
            return np.array([[2.0, 0.0], [0.0, 0.5]], dtype=np.float64)

    def fake_gradient_minimize(fun, x0, jac, method, tol, options, callback):
        del fun, jac, method, tol, options
        callback(np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(
            x=np.asarray(x0, dtype=np.float64),
            fun=0.0,
            nit=1,
            nfev=1,
            status=0,
            success=True,
            hess_inv=_FakeInvHess(),
        )

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_map_laplace_bundle",
        fake_build_bundle,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._draw_laplace_init_candidates",
        fake_draw_candidates,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.spo.minimize",
        fake_gradient_minimize,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._sample_laplace_parameter_posterior",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact parameter Hessian path should be skipped")
        ),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map.extract_constrained_samples",
        lambda unc_samples, *_args, **_kwargs: {"theta": unc_samples},
    )

    result = fit_map(
        _FakeModel(),
        observations,
        times,
        num_samples=4,
        compute_parameter_hessian=True,
        parameter_covariance_method="optimizer_hess_inv",
        seed=0,
    )

    assert result.method == "map"
    assert result.diagnostics["compute_parameter_hessian"] is True
    assert result.diagnostics["parameter_posterior_strategy"] == "laplace_gaussian"
    assert result.diagnostics["parameter_covariance_method"] == "optimizer_hess_inv"
    assert result.diagnostics["hessian_condition_number"] is None
    np.testing.assert_allclose(
        np.asarray(result.diagnostics["covariance_diag"]),
        np.array([2.0001, 0.5001], dtype=np.float32),
        atol=1e-5,
    )
    assert result.get_samples()["theta"].shape == (4, 2)


def test_map_bundle_reuses_runtime_objectives_across_same_shape_datasets(monkeypatch):
    observations_a = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    observations_b = jnp.array([[2.0], [3.0]], dtype=jnp.float32)
    times_a = jnp.array([0.0, 1.0], dtype=jnp.float32)
    times_b = jnp.array([0.0, 2.0], dtype=jnp.float32)
    counters = {"discover": 0, "build_eval_fns": 0}

    class _IdentityTransform:
        def inv(self, value):
            return value

    class _FakeModel:
        def __init__(self):
            self._artifact_cache: dict[tuple[object, ...], object] = {}

        def get_cached_artifact(self, cache_key, factory):
            if cache_key not in self._artifact_cache:
                self._artifact_cache[cache_key] = factory()
            return self._artifact_cache[cache_key]

    def fake_discover_sites(
        _model,
        _observations,
        _times,
        _trace_key,
        _likelihood_backend,
        reparam=None,
    ):
        del reparam
        counters["discover"] += 1
        return {
            "theta": {
                "transform": _IdentityTransform(),
                "value": jnp.array([0.5, -0.25], dtype=jnp.float32),
                "distribution": object(),
            }
        }

    def fake_build_eval_fns(
        _model,
        _observations,
        _times,
        _site_info,
        _unravel_fn,
        likelihood_backend,
        reparam=None,
        *,
        include_likelihood_aux,
        runtime_observations_times,
    ):
        del likelihood_backend, reparam
        counters["build_eval_fns"] += 1
        assert include_likelihood_aux is True
        assert runtime_observations_times is True

        def log_lik_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del latent_mode_init
            return jnp.sum(z) + jnp.sum(runtime_observations) + jnp.sum(runtime_times)

        def log_prior_unc_fn(z):
            return -0.5 * jnp.sum(z**2)

        def log_lik_with_aux_fn(z, runtime_observations, runtime_times, latent_mode_init=None):
            del latent_mode_init
            return log_lik_fn(z, runtime_observations, runtime_times), {
                "solver_kind": jnp.asarray(0, dtype=jnp.int32),
                "n_iterations": jnp.asarray(0, dtype=jnp.int32),
                "n_accepted_steps": jnp.asarray(0, dtype=jnp.int32),
                "init_log_joint": jnp.asarray(0.0, dtype=jnp.float32),
                "final_log_joint": jnp.asarray(0.0, dtype=jnp.float32),
                "final_rel_change": jnp.asarray(0.0, dtype=jnp.float32),
                "final_damping": jnp.asarray(0.0, dtype=jnp.float32),
                "final_step_alpha": jnp.asarray(0.0, dtype=jnp.float32),
                "final_step_norm": jnp.asarray(0.0, dtype=jnp.float32),
                "laplace_logdet": jnp.asarray(0.0, dtype=jnp.float32),
                "min_chol_diag": jnp.asarray(0.0, dtype=jnp.float32),
            }

        return log_lik_fn, log_prior_unc_fn, log_lik_with_aux_fn

    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._discover_sites",
        fake_discover_sites,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.models.ssm.inference.methods.map._build_eval_fns",
        fake_build_eval_fns,
    )

    model = _FakeModel()
    backend = SimpleNamespace()
    bundle_a = _build_map_laplace_bundle(
        model,
        observations_a,
        times_a,
        random.PRNGKey(0),
        backend,
        None,
    )
    bundle_b = _build_map_laplace_bundle(
        model,
        observations_b,
        times_b,
        random.PRNGKey(1),
        backend,
        None,
    )

    assert counters["discover"] == 2
    assert counters["build_eval_fns"] == 1
    assert bundle_a["log_posterior_fn"] is bundle_b["log_posterior_fn"]
    assert bundle_a["neg_log_posterior_fn"] is bundle_b["neg_log_posterior_fn"]

    z = jnp.array([0.2, -0.1], dtype=jnp.float32)
    log_post_a = bundle_a["log_posterior_fn"](z, observations_a, times_a)
    log_post_b = bundle_b["log_posterior_fn"](z, observations_b, times_b)
    assert float(log_post_a) == pytest.approx(2.075)
    assert float(log_post_b) == pytest.approx(7.075)


# =============================================================================
# Builder Noise Family Wiring Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
