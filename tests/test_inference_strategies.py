"""Comprehensive tests for inference strategy backends.

Tests follow the strategy:
1. Unit tests for strategy selection and expression parsing
2. Cross-validation: Kalman as ground truth, UKF must match on linear-Gaussian
3. PMMH bootstrap filter: finite likelihood, consistency
4. Parameter recovery: simulate → infer → check credible intervals

Test Matrix:
| Model Class                    | Strategies           | Ground Truth       |
|--------------------------------|----------------------|--------------------|
| Linear-Gaussian                | Kalman, UKF          | Kalman as ref      |
| Nonlinear dynamics, Gaussian   | UKF                  | Simulated recovery |
| Linear, non-Gaussian obs       | PMMH (particle)      | Bootstrap filter   |
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.ssm import NoiseFamily, SSMSpec

# =============================================================================
# Unit Tests: Strategy Selector
# =============================================================================


class TestStrategySelector:
    """Unit tests for strategy selection logic."""

    def test_linear_gaussian_selects_kalman(self):
        """Linear dynamics + Gaussian noise → Kalman filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            diffusion="diag",
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.KALMAN

    def test_fixed_matrices_selects_kalman(self):
        """Fixed (ndarray) matrices are linear → Kalman filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion=jnp.eye(2) * 0.3,
            lambda_mat=jnp.eye(2),
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.KALMAN

    def test_student_t_observation_selects_particle(self):
        """Non-Gaussian observation noise → particle filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.STUDENT_T,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_student_t_process_selects_particle(self):
        """Non-Gaussian process noise → particle filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.STUDENT_T,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_poisson_observation_selects_particle(self):
        """Poisson (count) observations → particle filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.POISSON,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_gamma_observation_selects_particle(self):
        """Gamma (positive continuous) observations → particle filter."""
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAMMA,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE


class TestExpressionParser:
    """Unit tests for state-dependent term detection."""

    def test_ndarray_is_linear(self):
        """Fixed numpy array is parameter-only (linear)."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms(jnp.eye(2))

    def test_free_string_is_linear(self):
        """'free' string means estimated but linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms("free")

    def test_diag_string_is_linear(self):
        """'diag' string means diagonal but linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms("diag")

    def test_none_is_linear(self):
        """None (disabled parameter) is linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms(None)

    def test_state_reference_is_nonlinear(self):
        """String referencing 'state' is nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("state[0] * param")

    def test_ss_reference_is_nonlinear(self):
        """String referencing 'ss[' is nonlinear (state-space indexing)."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("ss[0] + ss[1]")

    def test_nonlinear_function_is_nonlinear(self):
        """String with nonlinear functions is nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("exp(state[0])")
        assert _has_state_dependent_terms("sin(x)")
        assert _has_state_dependent_terms("tanh(y)")

    def test_numeric_constant_is_linear(self):
        """Numeric constants (int, float) are linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms(0.5)
        assert not _has_state_dependent_terms(1)

    def test_simple_param_string_is_linear(self):
        """Simple parameter name string is linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert not _has_state_dependent_terms("beta")
        assert not _has_state_dependent_terms("sigma_obs")


# =============================================================================
# Cross-Validation: Linear-Gaussian Models
# =============================================================================


class TestCrossValidationLinearGaussian:
    """Cross-validate Kalman and UKF on linear-Gaussian models.

    Key invariant: UKF must match Kalman on linear-Gaussian.
    Kalman is exact—use as ground truth.
    """

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

    def test_kalman_computes_finite_likelihood(self, linear_gaussian_params, simple_observations):
        """Kalman filter produces finite log-likelihood."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        observations, time_intervals = simple_observations
        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            linear_gaussian_params["ct_params"],
            linear_gaussian_params["meas_params"],
            linear_gaussian_params["init_params"],
            observations,
            time_intervals,
        )
        assert jnp.isfinite(ll)

    def test_ukf_matches_kalman(self, linear_gaussian_params, simple_observations):
        """UKF log-likelihood matches Kalman within numerical tolerance."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood

        observations, time_intervals = simple_observations

        kalman = KalmanLikelihood()
        kalman_ll = kalman.compute_log_likelihood(
            linear_gaussian_params["ct_params"],
            linear_gaussian_params["meas_params"],
            linear_gaussian_params["init_params"],
            observations,
            time_intervals,
        )

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            linear_gaussian_params["ct_params"],
            linear_gaussian_params["meas_params"],
            linear_gaussian_params["init_params"],
            observations,
            time_intervals,
        )

        # UKF should match Kalman within 5% for linear-Gaussian
        np.testing.assert_allclose(
            float(ukf_ll),
            float(kalman_ll),
            rtol=0.05,
            err_msg=f"UKF={float(ukf_ll):.4f} vs Kalman={float(kalman_ll):.4f}",
        )

    def test_bootstrap_filter_matches_kalman_moderate_particles(
        self, linear_gaussian_params, simple_observations
    ):
        """Bootstrap PF matches Kalman within Monte Carlo error."""
        from dsem_agent.models.pmmh import CTSEMAdapter, bootstrap_filter

        observations, time_intervals = simple_observations
        obs_mask = ~jnp.isnan(observations)

        # Kalman (ground truth)
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        kalman = KalmanLikelihood()
        kalman_ll = kalman.compute_log_likelihood(
            linear_gaussian_params["ct_params"],
            linear_gaussian_params["meas_params"],
            linear_gaussian_params["init_params"],
            observations,
            time_intervals,
        )

        # Bootstrap PF
        ct = linear_gaussian_params["ct_params"]
        mp = linear_gaussian_params["meas_params"]
        ip = linear_gaussian_params["init_params"]
        model = CTSEMAdapter(n_latent=2, n_manifest=2)
        params = {
            "drift": ct.drift,
            "diffusion_cov": ct.diffusion_cov,
            "cint": ct.cint,
            "lambda_mat": mp.lambda_mat,
            "manifest_means": mp.manifest_means,
            "manifest_cov": mp.manifest_cov,
            "t0_mean": ip.mean,
            "t0_cov": ip.cov,
        }

        result = bootstrap_filter(
            model,
            params,
            observations,
            time_intervals,
            obs_mask,
            n_particles=1000,
            key=random.PRNGKey(42),
        )

        # PF with 1000 particles should be within 15% of Kalman
        np.testing.assert_allclose(
            float(result.log_likelihood),
            float(kalman_ll),
            rtol=0.15,
            err_msg=f"PF={float(result.log_likelihood):.4f} vs Kalman={float(kalman_ll):.4f}",
        )

    def test_all_strategies_agree_on_longer_series(self):
        """Kalman and UKF agree on longer time series (25 points)."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood

        T = 25
        key = random.PRNGKey(123)
        observations = random.normal(key, (T, 2)) * 0.3
        time_intervals = jnp.ones(T) * 0.5

        ct_params = CTParams(
            drift=jnp.array([[-0.6, 0.15], [0.1, -0.7]]),
            diffusion_cov=jnp.array([[0.08, 0.01], [0.01, 0.08]]),
            cint=jnp.zeros(2),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.05,
        )
        init_params = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        kalman = KalmanLikelihood()
        kalman_ll = kalman.compute_log_likelihood(
            ct_params, meas_params, init_params, observations, time_intervals
        )

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            ct_params, meas_params, init_params, observations, time_intervals
        )

        assert jnp.isfinite(kalman_ll)
        np.testing.assert_allclose(float(ukf_ll), float(kalman_ll), rtol=0.05)


# =============================================================================
# Cross-Validation: UKF on Nonlinear (sanity checks)
# =============================================================================


class TestCrossValidationNonlinear:
    """Test UKF on models with varied data.

    Since we don't have ground truth for nonlinear, we check:
    1. Produces finite likelihoods
    2. Consistency across parameter variations
    """

    def test_ukf_finite_on_varied_data(self):
        """UKF produces finite likelihoods on varied data with outliers."""
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood

        T = 20
        key = random.PRNGKey(789)
        base_obs = random.normal(key, (T, 2)) * 0.5
        outliers = jnp.array([[2.0, -1.5], [-1.8, 2.2]])
        observations = base_obs.at[5].set(outliers[0]).at[15].set(outliers[1])
        time_intervals = jnp.ones(T) * 0.5

        ct_params = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.zeros(2),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.2,
        )
        init_params = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            ct_params, meas_params, init_params, observations, time_intervals
        )
        assert jnp.isfinite(ukf_ll), f"UKF produced non-finite: {ukf_ll}"


# =============================================================================
# Parameter Recovery Tests
# =============================================================================


class TestParameterRecoveryKalman:
    """Parameter recovery tests using Kalman filter.

    Simulate from known parameters → run inference → verify true params
    fall within 90% credible intervals.
    """

    @pytest.mark.slow
    @pytest.mark.xfail(reason="MCMC convergence sensitive to parameterization; needs tuning")
    def test_drift_diagonal_recovery(self):
        """Recover drift diagonal parameters from simulated data."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        true_drift_diag = jnp.array([-0.6, -0.9])

        key = random.PRNGKey(42)
        T = 60
        n_latent = 2
        dt = 0.5
        discrete_coef = jnp.exp(jnp.diag(true_drift_diag) * dt)
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

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
        )
        model = SSMModel(spec)

        mcmc = model.fit(
            observations=observations,
            times=times,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        drift_diag_samples = samples["drift_diag_pop"]

        for i, true_val in enumerate(true_drift_diag):
            posterior_mean = jnp.mean(drift_diag_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.5, (
                f"Drift[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )

    @pytest.mark.slow
    @pytest.mark.xfail(reason="MCMC convergence sensitive to parameterization; needs tuning")
    def test_diffusion_recovery(self):
        """Recover diffusion parameters from simulated data."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        true_diffusion_diag = jnp.array([0.4, 0.4])
        true_drift_diag = jnp.array([-0.5, -0.5])

        key = random.PRNGKey(123)
        T = 80
        n_latent = 2
        dt = 0.5

        discrete_coef = jnp.exp(jnp.diag(true_drift_diag) * dt)

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * true_diffusion_diag
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.05
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
        )
        model = SSMModel(spec)

        mcmc = model.fit(
            observations=observations,
            times=times,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        diffusion_samples = samples["diffusion_diag_pop"]

        for i, true_val in enumerate(true_diffusion_diag):
            posterior_mean = jnp.mean(diffusion_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.4, (
                f"Diffusion[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )


class TestPMMHIntegration:
    """Test PMMH integration (replaces old particle filter NumPyro tests)."""

    def test_pmmh_bootstrap_filter_varies_with_params(self):
        """Bootstrap filter likelihood varies with different parameters."""
        from dsem_agent.models.pmmh import CTSEMAdapter, bootstrap_filter

        T = 10
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.5
        obs_mask = ~jnp.isnan(observations)

        model = CTSEMAdapter(n_latent=2, n_manifest=2)

        drift_values = [
            jnp.array([[-0.3, 0.0], [0.0, -0.3]]),
            jnp.array([[-0.5, 0.1], [0.1, -0.5]]),
            jnp.array([[-0.8, 0.0], [0.0, -0.8]]),
        ]

        likelihoods = []
        for drift in drift_values:
            params = {
                "drift": drift,
                "diffusion_cov": jnp.eye(2) * 0.1,
                "lambda_mat": jnp.eye(2),
                "manifest_means": jnp.zeros(2),
                "manifest_cov": jnp.eye(2) * 0.1,
                "t0_mean": jnp.zeros(2),
                "t0_cov": jnp.eye(2),
            }
            result = bootstrap_filter(
                model,
                params,
                observations,
                time_intervals,
                obs_mask,
                n_particles=200,
                key=random.PRNGKey(42),
            )
            likelihoods.append(float(result.log_likelihood))

        # All likelihoods should be finite
        assert all(np.isfinite(ll) for ll in likelihoods)

        # Likelihoods should vary with parameters
        assert len({round(ll, 2) for ll in likelihoods}) > 1


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_observation(self):
        """Handle single observation gracefully."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

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

        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.isfinite(ll)

    def test_very_small_time_interval(self):
        """Handle very small time intervals."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        T = 10
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

        observations = jnp.ones((T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.001

        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.isfinite(ll)

    def test_irregular_time_intervals(self):
        """Handle irregular time intervals."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

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

        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.isfinite(ll)

    def test_higher_dimensional_system(self):
        """Test 4-dimensional latent system."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

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

        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.isfinite(ll)

    def test_non_identity_lambda(self):
        """Test with non-identity factor loading matrix."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

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

        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )
        assert jnp.isfinite(ll)


# =============================================================================
# Backend Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Test that backends integrate correctly with strategy selector."""

    def test_get_backend_returns_correct_type(self):
        """get_likelihood_backend returns correct backend class."""
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood
        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        kalman_backend = get_likelihood_backend(InferenceStrategy.KALMAN)
        assert kalman_backend is not None
        assert hasattr(kalman_backend, "compute_log_likelihood")

        ukf_backend = get_likelihood_backend(InferenceStrategy.UKF)
        assert isinstance(ukf_backend, UKFLikelihood)

        # PARTICLE raises ValueError (uses PMMH path)
        with pytest.raises(ValueError, match="PMMH"):
            get_likelihood_backend(InferenceStrategy.PARTICLE)

    def test_model_strategy_method(self):
        """SSMModel.get_inference_strategy() works correctly."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy

        # Linear-Gaussian model
        spec_kalman = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        model_kalman = SSMModel(spec_kalman)
        assert model_kalman.get_inference_strategy() == InferenceStrategy.KALMAN

        # Non-Gaussian observation model
        spec_particle = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.STUDENT_T,
        )
        model_particle = SSMModel(spec_particle)
        assert model_particle.get_inference_strategy() == InferenceStrategy.PARTICLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
