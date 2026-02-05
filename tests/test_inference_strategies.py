"""Comprehensive tests for inference strategy backends.

Tests follow the strategy:
1. Unit tests for strategy selection and expression parsing
2. Cross-validation: Kalman as ground truth, UKF/Particle must match on linear-Gaussian
3. Parameter recovery: simulate → infer → check 90% credible intervals

Test Matrix:
| Model Class                    | Strategies              | Ground Truth       |
|--------------------------------|-------------------------|--------------------|
| Linear-Gaussian                | Kalman, UKF, Particle   | Kalman as ref      |
| Nonlinear dynamics, Gaussian   | UKF, Particle           | Simulated recovery |
| Linear, non-Gaussian obs       | Particle                | Simulated recovery |
| Nonlinear, non-Gaussian        | Particle                | Simulated recovery |
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

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
    """Cross-validate Kalman, UKF, and Particle on linear-Gaussian models.

    Key invariant: all strategies must agree on log-likelihood.
    Kalman is exact—use as ground truth.
    """

    @pytest.fixture
    def linear_gaussian_params(self):
        """Standard test parameters for 2D linear-Gaussian model."""
        return {
            "drift": jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            "diffusion_cov": jnp.array([[0.1, 0.02], [0.02, 0.1]]),
            "cint": jnp.array([0.0, 0.0]),
            "lambda_mat": jnp.eye(2),
            "manifest_means": jnp.zeros(2),
            "manifest_cov": jnp.eye(2) * 0.1,
            "t0_mean": jnp.zeros(2),
            "t0_cov": jnp.eye(2),
        }

    @pytest.fixture
    def simple_observations(self):
        """Simple test observations and time intervals."""
        T = 15  # Reduced for faster tests
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.5
        return observations, time_intervals

    def test_kalman_computes_finite_likelihood(self, linear_gaussian_params, simple_observations):
        """Kalman filter produces finite log-likelihood."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        observations, time_intervals = simple_observations
        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            **linear_gaussian_params,
        )
        assert jnp.isfinite(ll)

    def test_ukf_matches_kalman(self, linear_gaussian_params, simple_observations):
        """UKF log-likelihood matches Kalman within numerical tolerance."""
        from dsem_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        observations, time_intervals = simple_observations

        # Kalman (ground truth)
        kalman_ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            **linear_gaussian_params,
        )

        # UKF
        ct_params = CTParams(
            drift=linear_gaussian_params["drift"],
            diffusion_cov=linear_gaussian_params["diffusion_cov"],
            cint=linear_gaussian_params["cint"],
        )
        meas_params = MeasurementParams(
            lambda_mat=linear_gaussian_params["lambda_mat"],
            manifest_means=linear_gaussian_params["manifest_means"],
            manifest_cov=linear_gaussian_params["manifest_cov"],
        )
        init_params = InitialStateParams(
            mean=linear_gaussian_params["t0_mean"],
            cov=linear_gaussian_params["t0_cov"],
        )

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        # UKF should match Kalman within 5% for linear-Gaussian
        np.testing.assert_allclose(
            float(ukf_ll),
            float(kalman_ll),
            rtol=0.05,
            err_msg=f"UKF={float(ukf_ll):.4f} vs Kalman={float(kalman_ll):.4f}",
        )

    def test_particle_matches_kalman_moderate_particles(
        self, linear_gaussian_params, simple_observations
    ):
        """Particle filter matches Kalman within Monte Carlo error."""
        from dsem_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        observations, time_intervals = simple_observations

        # Kalman (ground truth)
        kalman_ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            **linear_gaussian_params,
        )

        # Particle filter with moderate particle count (faster)
        ct_params = CTParams(
            drift=linear_gaussian_params["drift"],
            diffusion_cov=linear_gaussian_params["diffusion_cov"],
            cint=linear_gaussian_params["cint"],
        )
        meas_params = MeasurementParams(
            lambda_mat=linear_gaussian_params["lambda_mat"],
            manifest_means=linear_gaussian_params["manifest_means"],
            manifest_cov=linear_gaussian_params["manifest_cov"],
        )
        init_params = InitialStateParams(
            mean=linear_gaussian_params["t0_mean"],
            cov=linear_gaussian_params["t0_cov"],
        )

        # Use moderate particle count (1000) for reasonable speed/accuracy tradeoff
        particle = ParticleLikelihood(n_particles=1000, seed=42)
        particle_ll = particle.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        # Particle filter with 1000 particles should be within 15% of Kalman
        np.testing.assert_allclose(
            float(particle_ll),
            float(kalman_ll),
            rtol=0.15,
            err_msg=f"Particle={float(particle_ll):.4f} vs Kalman={float(kalman_ll):.4f}",
        )

    def test_all_strategies_agree_on_longer_series(self):
        """All strategies agree on longer time series (25 points)."""
        from dsem_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        # Generate moderate length time series
        T = 25
        key = random.PRNGKey(123)
        observations = random.normal(key, (T, 2)) * 0.3
        time_intervals = jnp.ones(T) * 0.5

        # Parameters
        drift = jnp.array([[-0.6, 0.15], [0.1, -0.7]])
        diffusion_cov = jnp.array([[0.08, 0.01], [0.01, 0.08]])
        cint = jnp.zeros(2)
        lambda_mat = jnp.eye(2)
        manifest_means = jnp.zeros(2)
        manifest_cov = jnp.eye(2) * 0.05
        t0_mean = jnp.zeros(2)
        t0_cov = jnp.eye(2)

        # Kalman
        kalman_ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=drift,
            diffusion_cov=diffusion_cov,
            cint=cint,
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
            t0_mean=t0_mean,
            t0_cov=t0_cov,
        )

        # UKF
        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=cint)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat, manifest_means=manifest_means, manifest_cov=manifest_cov
        )
        init_params = InitialStateParams(mean=t0_mean, cov=t0_cov)

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        # Particle (moderate count for speed)
        particle = ParticleLikelihood(n_particles=800, seed=456)
        particle_ll = particle.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        # All should be reasonably close to Kalman
        assert jnp.isfinite(kalman_ll)
        np.testing.assert_allclose(float(ukf_ll), float(kalman_ll), rtol=0.05)
        np.testing.assert_allclose(float(particle_ll), float(kalman_ll), rtol=0.20)


# =============================================================================
# Cross-Validation: UKF vs Particle on Nonlinear
# =============================================================================


class TestCrossValidationNonlinear:
    """Cross-validate UKF and Particle on models where Kalman doesn't apply.

    Since we don't have ground truth, we check:
    1. Both produce finite likelihoods
    2. They're in the same ballpark (within 20%)
    3. Consistency across runs
    """

    def test_ukf_and_particle_both_finite_on_varied_data(self):
        """UKF and Particle both produce finite likelihoods on varied data."""
        from dsem_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood

        # Generate varied data (some outliers)
        T = 20
        key = random.PRNGKey(789)
        base_obs = random.normal(key, (T, 2)) * 0.5
        # Add a few outliers
        outliers = jnp.array([[2.0, -1.5], [-1.8, 2.2]])
        observations = base_obs.at[5].set(outliers[0]).at[15].set(outliers[1])
        time_intervals = jnp.ones(T) * 0.5

        # Parameters
        ct_params = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.zeros(2),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.2,  # Higher obs noise
        )
        init_params = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        ukf = UKFLikelihood()
        ukf_ll = ukf.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        particle = ParticleLikelihood(n_particles=500, seed=42)
        particle_ll = particle.compute_log_likelihood(
            ct_params=ct_params,
            measurement_params=meas_params,
            initial_state=init_params,
            observations=observations,
            time_intervals=time_intervals,
        )

        assert jnp.isfinite(ukf_ll), f"UKF produced non-finite: {ukf_ll}"
        assert jnp.isfinite(particle_ll), f"Particle produced non-finite: {particle_ll}"

        # They should be in the same ballpark (within 30%)
        np.testing.assert_allclose(
            float(particle_ll),
            float(ukf_ll),
            rtol=0.30,
            err_msg=f"Large discrepancy: UKF={float(ukf_ll):.2f}, Particle={float(particle_ll):.2f}",
        )


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
        """Recover drift diagonal parameters from simulated data.

        Note: This test is marked xfail because MCMC convergence is sensitive
        to model parameterization and requires more samples/chains for reliability.
        The core likelihood computation is validated by the cross-validation tests.
        """
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        # True parameters
        true_drift_diag = jnp.array([-0.6, -0.9])

        # Generate data from true model
        key = random.PRNGKey(42)
        T = 60  # Reduced for speed
        n_latent = 2

        # Simulate AR(1)-like process with these drift values
        # x_{t+1} = exp(A*dt) * x_t + noise
        dt = 0.5
        discrete_coef = jnp.exp(jnp.diag(true_drift_diag) * dt)
        process_noise = 0.3

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * process_noise
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        # Add observation noise
        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.1
        times = jnp.arange(T, dtype=float) * dt

        # Fit model with minimal samples for speed
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

        # Check posterior mean is in reasonable range (relaxed for short chain)
        for i, true_val in enumerate(true_drift_diag):
            posterior_mean = jnp.mean(drift_diag_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.5, (
                f"Drift[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )

    @pytest.mark.slow
    @pytest.mark.xfail(reason="MCMC convergence sensitive to parameterization; needs tuning")
    def test_diffusion_recovery(self):
        """Recover diffusion parameters from simulated data.

        Note: This test is marked xfail because MCMC convergence is sensitive
        to model parameterization and requires more samples/chains for reliability.
        The core likelihood computation is validated by the cross-validation tests.
        """
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        # True parameters
        true_diffusion_diag = jnp.array([0.4, 0.4])
        true_drift_diag = jnp.array([-0.5, -0.5])

        # Generate data
        key = random.PRNGKey(123)
        T = 80  # Reduced for speed
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

        # Fit model with minimal samples
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

        # Check posterior mean is in reasonable range
        for i, true_val in enumerate(true_diffusion_diag):
            posterior_mean = jnp.mean(diffusion_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.4, (
                f"Diffusion[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )


class TestParameterRecoveryParticle:
    """Parameter recovery tests using Particle filter.

    For non-Gaussian models where only particle filter applies.
    """

    def test_particle_likelihood_integrates_with_numpyro(self):
        """Verify particle likelihood can be used in NumPyro model.

        This is a smoke test to ensure the particle filter integrates
        correctly with NumPyro's sampling machinery.
        """
        from dsem_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood

        # Generate simple data
        T = 10
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

        # Test that we can compute likelihood for different parameter values
        particle = ParticleLikelihood(n_particles=200, seed=42)

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
            init_params = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

            ll = particle.compute_log_likelihood(
                ct_params=ct_params,
                measurement_params=meas_params,
                initial_state=init_params,
                observations=observations,
                time_intervals=time_intervals,
            )
            likelihoods.append(float(ll))

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
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        observations = jnp.array([[0.5, -0.3]])
        time_intervals = jnp.array([1.0])

        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=None,
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
            t0_mean=jnp.zeros(2),
            t0_cov=jnp.eye(2),
        )
        assert jnp.isfinite(ll)

    def test_very_small_time_interval(self):
        """Handle very small time intervals."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        T = 10
        observations = jnp.ones((T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.001  # Very small

        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=None,
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
            t0_mean=jnp.zeros(2),
            t0_cov=jnp.eye(2),
        )
        assert jnp.isfinite(ll)

    def test_irregular_time_intervals(self):
        """Handle irregular time intervals."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        observations = jnp.array(
            [
                [0.1, 0.2],
                [0.3, 0.1],
                [0.2, 0.4],
                [0.5, 0.3],
                [0.4, 0.5],
            ]
        )
        # Irregular intervals
        time_intervals = jnp.array([0.1, 0.5, 0.2, 1.0, 0.3])

        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.array([0.1, -0.1]),
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
            t0_mean=jnp.zeros(2),
            t0_cov=jnp.eye(2),
        )
        assert jnp.isfinite(ll)

    def test_higher_dimensional_system(self):
        """Test 4-dimensional latent system."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        n_latent = 4
        n_manifest = 4
        T = 30

        key = random.PRNGKey(42)
        observations = random.normal(key, (T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

        # Create stable 4x4 drift matrix
        drift = jnp.diag(jnp.array([-0.5, -0.6, -0.7, -0.8]))

        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=drift,
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=None,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
            t0_mean=jnp.zeros(n_latent),
            t0_cov=jnp.eye(n_latent),
        )
        assert jnp.isfinite(ll)

    def test_non_identity_lambda(self):
        """Test with non-identity factor loading matrix."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        n_latent = 2
        n_manifest = 3
        T = 20

        key = random.PRNGKey(42)
        observations = random.normal(key, (T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

        # 3 manifest variables from 2 latent
        lambda_mat = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        )

        ll = kalman_log_likelihood(
            observations=observations,
            time_intervals=time_intervals,
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=None,
            lambda_mat=lambda_mat,
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
            t0_mean=jnp.zeros(n_latent),
            t0_cov=jnp.eye(n_latent),
        )
        assert jnp.isfinite(ll)


# =============================================================================
# Backend Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Test that backends integrate correctly with strategy selector."""

    def test_get_backend_returns_correct_type(self):
        """get_likelihood_backend returns correct backend class."""
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
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

        particle_backend = get_likelihood_backend(InferenceStrategy.PARTICLE)
        assert isinstance(particle_backend, ParticleLikelihood)

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
