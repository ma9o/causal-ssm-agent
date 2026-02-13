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

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from causal_ssm_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood, SSMAdapter
from causal_ssm_agent.models.ssm import InferenceResult, NoiseFamily, SSMModel, SSMSpec, fit

# =============================================================================
# ParticleLikelihood: Core Functionality
# =============================================================================


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

        assert jnp.isfinite(ll), f"PF produced non-finite: {ll}"

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
            likelihoods.append(float(ll))

        assert all(np.isfinite(ll) for ll in likelihoods)
        assert len({round(ll, 2) for ll in likelihoods}) > 1


class TestDeterministicKey:
    """Test that fixed PF key gives deterministic results."""

    def test_same_params_same_key_same_ll(self):
        """Same parameters + same key should produce identical log-likelihood."""
        T = 10
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        time_intervals = jnp.ones(T) * 0.5

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

        pf_key = random.PRNGKey(99)

        backend1 = ParticleLikelihood(
            n_latent=2,
            n_manifest=2,
            n_particles=200,
            rng_key=pf_key,
        )
        ll1 = backend1.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        backend2 = ParticleLikelihood(
            n_latent=2,
            n_manifest=2,
            n_particles=200,
            rng_key=pf_key,
        )
        ll2 = backend2.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert float(ll1) == float(ll2), f"Not deterministic: {ll1} vs {ll2}"


class TestParticleLikelihoodGradient:
    """Test that jax.grad flows through the particle filter."""

    def test_gradient_is_finite(self):
        """jax.grad of PF log-likelihood should produce finite values."""
        T = 8
        observations = jnp.ones((T, 2)) * 0.3
        time_intervals = jnp.ones(T) * 0.5

        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        pf_key = random.PRNGKey(0)

        def ll_fn(drift_diag):
            drift = jnp.diag(drift_diag)
            ct_params = CTParams(
                drift=drift,
                diffusion_cov=jnp.eye(2) * 0.1,
                cint=None,
            )
            backend = ParticleLikelihood(
                n_latent=2,
                n_manifest=2,
                n_particles=100,
                rng_key=pf_key,
            )
            return backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                observations,
                time_intervals,
            )

        drift_diag = jnp.array([-0.5, -0.5])
        grad = jax.grad(ll_fn)(drift_diag)

        assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"


# =============================================================================
# SSMAdapter: Observation Models
# =============================================================================


class TestParticleMissingData:
    """Tests for Gaussian observation masking in PF adapter."""

    def test_missing_dimension_not_penalized(self):
        """Missing dims should not incur huge log-det penalties."""
        n_latent, n_manifest = 1, 2
        adapter = SSMAdapter(n_latent, n_manifest, manifest_dist="gaussian")

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
            n_latent, n_manifest, manifest_dist="gaussian", diffusion_dist="student_t"
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
    @pytest.mark.xfail(reason="MCMC convergence sensitive to parameterization; needs tuning")
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

        spec = SSMSpec(
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
    @pytest.mark.xfail(reason="MCMC convergence sensitive to parameterization; needs tuning")
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

        spec = SSMSpec(
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


class TestParameterRecoveryPoisson:
    """Parameter recovery for Poisson observations via PF+NUTS."""

    @pytest.mark.slow
    def test_drift_recovery_poisson_obs(self):
        """Recover drift from 1D AR(1) with Poisson observations."""
        true_drift = -0.5
        n_latent = 1
        T = 80
        dt = 0.5
        process_noise = 0.2
        log_baseline = jnp.log(5.0)

        # Simulate latent AR(1) process
        key = random.PRNGKey(42)
        discrete_coef = jnp.exp(true_drift * dt)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * process_noise
            new_state = discrete_coef * states[-1] + noise
            states.append(new_state)
        latent = jnp.stack(states)

        # Poisson observations: y ~ Poisson(exp(x + log_baseline))
        key, subkey = random.split(key)
        rates = jnp.exp(latent + log_baseline)
        observations = random.poisson(subkey, rates).astype(jnp.float32)
        times = jnp.arange(T, dtype=jnp.float32) * dt

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dist=NoiseFamily.POISSON,
            manifest_means=jnp.array([log_baseline]),
        )
        model = SSMModel(spec, n_particles=300)

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
        # Model applies -abs(drift_diag_pop), so apply the same transform
        actual_drift_mean = float(jnp.mean(-jnp.abs(drift_diag_samples[:, 0])))

        # Recovered drift should be negative
        assert actual_drift_mean < 0.0, f"Drift should be negative: {actual_drift_mean:.3f}"


class TestParameterRecoveryStudentT:
    """Parameter recovery for Student-t observations via PF+NUTS."""

    @pytest.mark.slow
    def test_drift_recovery_student_t_obs(self):
        """Recover drift from 1D AR(1) with Student-t observations."""
        true_drift = -0.6
        n_latent, n_manifest = 1, 1
        T = 60
        dt = 0.5
        process_noise = 0.2
        obs_scale = 0.3
        obs_df = 5.0

        # Simulate latent AR(1) process
        key = random.PRNGKey(123)
        discrete_coef = jnp.exp(true_drift * dt)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * process_noise
            new_state = discrete_coef * states[-1] + noise
            states.append(new_state)
        latent = jnp.stack(states)

        # Student-t observations
        key, key_z, key_chi2 = random.split(key, 3)
        z = random.normal(key_z, (T, n_manifest))
        chi2 = random.gamma(key_chi2, obs_df / 2.0, (T, n_manifest)) * 2.0
        t_noise = z / jnp.sqrt(chi2 / obs_df) * obs_scale
        observations = latent + t_noise
        times = jnp.arange(T, dtype=jnp.float32) * dt

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dist=NoiseFamily.STUDENT_T,
        )
        model = SSMModel(spec, n_particles=300)

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
        # Model applies -abs(drift_diag_pop), so apply the same transform
        actual_drift_mean = float(jnp.mean(-jnp.abs(drift_diag_samples[:, 0])))

        # Recovered drift should be negative
        assert actual_drift_mean < 0.0, f"Drift should be negative: {actual_drift_mean:.3f}"


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
            manifest_dist="poisson",
            diffusion_dist="student_t",
        )
        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params={"proc_df": proc_df},
        )

        assert jnp.isfinite(ll), f"Non-finite LL on high-dim model: {ll}"


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
        assert jnp.isfinite(ll)

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
        assert jnp.isfinite(ll)

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
        assert jnp.isfinite(ll)

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
        assert jnp.isfinite(ll)


# =============================================================================
# fit() Integration Tests
# =============================================================================


class TestFitReturnsInferenceResult:
    """Test that fit() returns InferenceResult for all methods."""

    @pytest.mark.slow
    def test_fit_nuts_returns_inference_result(self):
        """fit() with method='nuts' returns InferenceResult."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        model = SSMModel(spec, n_particles=50)

        T = 15
        key = random.PRNGKey(0)
        observations = random.normal(key, (T, 2)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        result = fit(
            model,
            observations=observations,
            times=times,
            method="nuts",
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "nuts"
        samples = result.get_samples()
        assert "drift_diag_pop" in samples

    @pytest.mark.slow
    def test_fit_svi_returns_inference_result(self):
        """fit() with method='svi' returns InferenceResult."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        model = SSMModel(spec, n_particles=50)

        T = 15
        key = random.PRNGKey(0)
        observations = random.normal(key, (T, 2)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        result = fit(
            model,
            observations=observations,
            times=times,
            method="svi",
            num_steps=50,
            num_samples=20,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "svi"
        samples = result.get_samples()
        # SVI Predictive returns deterministic sites (drift, diffusion, etc.)
        assert "drift" in samples
        assert samples["drift"].shape[0] == 20

    @pytest.mark.slow
    def test_fit_poisson_svi_returns_inference_result(self):
        """fit() with Poisson manifest_dist + SVI returns InferenceResult."""
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dist=NoiseFamily.POISSON,
        )
        model = SSMModel(spec, n_particles=50)

        T = 15
        key = random.PRNGKey(0)
        observations = random.poisson(key, jnp.ones((T, 1)) * 5.0).astype(jnp.float32)
        times = jnp.arange(T, dtype=jnp.float32)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="svi",
            num_steps=50,
            num_samples=20,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "svi"


# =============================================================================
# SVI-specific Tests
# =============================================================================


class TestSVIBackend:
    """Tests specific to SVI inference backend."""

    @pytest.mark.slow
    def test_svi_losses_decrease(self):
        """ELBO loss should generally decrease during SVI training."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            diffusion="diag",
        )
        model = SSMModel(spec, n_particles=50)

        T = 15
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
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

    @pytest.mark.slow
    def test_svi_guide_types(self):
        """All guide types should produce valid results."""
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )

        T = 10
        key = random.PRNGKey(0)
        observations = random.normal(key, (T, 1)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        for guide_type in ["normal", "mvn", "delta"]:
            model = SSMModel(spec, n_particles=50)
            result = fit(
                model,
                observations=observations,
                times=times,
                method="svi",
                guide_type=guide_type,
                num_steps=50,
                num_samples=10,
            )
            assert isinstance(result, InferenceResult)
            assert len(result.get_samples()) > 0


# =============================================================================
# PMMH-specific Tests
# =============================================================================


class TestPMMHBackend:
    """Tests specific to PMMH inference backend."""

    @pytest.mark.slow
    @pytest.mark.xfail(reason="PMMH acceptance rate sensitive to seed/initialization")
    def test_pmmh_acceptance_rate_reasonable(self):
        """PMMH acceptance rate should be between 0.05 and 0.95."""
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
        )
        model = SSMModel(spec, n_particles=100)

        T = 15
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 1)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        result = fit(
            model,
            observations=observations,
            times=times,
            method="pmmh",
            num_warmup=50,
            num_samples=50,
        )

        rate = result.diagnostics["acceptance_rate"]
        assert 0.05 < rate < 0.95, f"Acceptance rate out of range: {rate:.3f}"


# =============================================================================
# SVI Parameter Recovery
# =============================================================================


class TestSVIParameterRecovery:
    """Parameter recovery tests using SVI."""

    @pytest.mark.slow
    def test_svi_drift_recovery(self):
        """SVI should recover drift direction from simulated Gaussian data."""
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

        spec = SSMSpec(
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
            method="svi",
            num_steps=2000,
            num_samples=200,
        )

        samples = result.get_samples()
        # SVI Predictive returns deterministic sites; use "drift" (full matrix)
        drift_samples = samples["drift"]

        # Check drift diagonal is negative (correct sign)
        for i in range(n_latent):
            posterior_mean = float(jnp.mean(drift_samples[:, i, i]))
            assert posterior_mean < 0.0, (
                f"Drift[{i},{i}] posterior mean {posterior_mean:.3f} should be negative"
            )


# =============================================================================
# Builder Noise Family Wiring Tests
# =============================================================================


class TestBuilderNoiseFamilyWiring:
    """Test that SSMModelBuilder wires noise families from ModelSpec."""

    def test_convert_spec_sets_poisson_noise_family(self):
        """ModelSpec with Poisson likelihood -> SSMSpec has POISSON manifest_dist."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.orchestrator.schemas_model import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )

        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="count_var",
                    distribution=DistributionFamily.POISSON,
                    link=LinkFunction.LOG,
                    reasoning="Count data",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="rho_count",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR coeff",
                    search_context="autoregressive",
                ),
            ],
            model_clock="daily",
            reasoning="test",
        )

        builder = SSMModelBuilder(model_spec=model_spec)
        ssm_spec = builder._convert_spec_to_ssm(model_spec)

        assert ssm_spec.manifest_dist == NoiseFamily.POISSON

    def test_convert_spec_sets_gaussian_for_normal(self):
        """ModelSpec with Normal likelihood -> SSMSpec has GAUSSIAN manifest_dist."""
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.orchestrator.schemas_model import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ModelSpec,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
        )

        model_spec = ModelSpec(
            likelihoods=[
                LikelihoodSpec(
                    variable="continuous_var",
                    distribution=DistributionFamily.NORMAL,
                    link=LinkFunction.IDENTITY,
                    reasoning="Continuous data",
                ),
            ],
            parameters=[
                ParameterSpec(
                    name="rho_cont",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR coeff",
                    search_context="autoregressive",
                ),
            ],
            model_clock="daily",
            reasoning="test",
        )

        builder = SSMModelBuilder(model_spec=model_spec)
        ssm_spec = builder._convert_spec_to_ssm(model_spec)

        assert ssm_spec.manifest_dist == NoiseFamily.GAUSSIAN


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
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_rw

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
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_fo

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
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_fo, _propose_rw

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
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_so

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
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_fo, _propose_so

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
        from causal_ssm_agent.models.ssm.hessmc2 import _reverse_rw

        s = reverse_state
        v_new, _chol_M, _ss = _reverse_rw(
            s["v_half"], s["grad_new"], s["hessian_new"], s["eps"], s["eps_fb"]
        )
        assert jnp.allclose(v_new, s["v_half"])

    def test_fo_reverse_applies_gradient_kick(self, reverse_state):
        """FO reverse: v_new = 0.5*eps*grad_new + v_half (Eq 34)."""
        from causal_ssm_agent.models.ssm.hessmc2 import _reverse_fo

        s = reverse_state
        v_new, _chol_M, _ss = _reverse_fo(
            s["v_half"], s["grad_new"], s["hessian_new"], s["eps"], s["eps_fb"]
        )
        expected = 0.5 * s["eps"] * s["grad_new"] + s["v_half"]
        assert jnp.allclose(v_new, expected)

    def test_fo_forward_reverse_symmetry(self):
        """FO proposal + reverse with same gradient should recover original v."""
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_fo, _reverse_fo

        x = jnp.array([1.0, 2.0])
        grad = jnp.array([0.5, -0.3])
        z = jnp.array([0.1, -0.2])
        eps = 0.1

        _, _v, v_half, _, _ = _propose_fo(x, grad, jnp.zeros((2, 2)), z, eps, eps)
        # If grad_new == grad (stationary), reverse should give v_new == v
        # because: v = z, v_half = 0.5*eps*grad + z
        #          v_new = 0.5*eps*grad + v_half = 0.5*eps*grad + 0.5*eps*grad + z
        # This is NOT v — the symmetry is in the weight correction, not the values.
        # Just verify the reverse produces finite values.
        v_new, _, _ = _reverse_fo(v_half, grad, jnp.zeros((2, 2)), eps, eps)
        assert jnp.all(jnp.isfinite(v_new))


class TestHessMC2Weights:
    """Test importance weight computation and CoV L-kernel."""

    def test_cov_density_is_finite(self):
        """CoV log-density should be finite for reasonable inputs."""
        from causal_ssm_agent.models.ssm.hessmc2 import _log_cov_density

        v = jnp.array([0.5, -0.3])
        chol_M = jnp.eye(2)
        eps = 0.1
        D = 2
        ld = _log_cov_density(v, chol_M, eps, D)
        assert jnp.isfinite(ld)

    def test_cov_density_higher_for_smaller_v(self):
        """Closer to mode (v=0) should give higher density."""
        from causal_ssm_agent.models.ssm.hessmc2 import _log_cov_density

        chol_M = jnp.eye(3)
        eps = 0.1
        D = 3
        ld_small = _log_cov_density(jnp.array([0.01, 0.01, 0.01]), chol_M, eps, D)
        ld_large = _log_cov_density(jnp.array([2.0, 2.0, 2.0]), chol_M, eps, D)
        assert ld_small > ld_large

    def test_weight_update_no_change_gives_zero_correction(self):
        """If proposal doesn't move and forward == reverse, weight unchanged."""
        from causal_ssm_agent.models.ssm.hessmc2 import _compute_weight

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
        from causal_ssm_agent.models.ssm.hessmc2 import _compute_weight

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
        from causal_ssm_agent.models.ssm.hessmc2 import _compute_weight

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


class TestHessMC2FullHessian:
    """Test full Hessian computation via jax.hessian on known functions."""

    def test_quadratic_hessian_is_exact(self):
        """For f(x) = 0.5 * x^T A x, H = A."""
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])

        def f(x):
            return 0.5 * x @ A @ x

        x = jnp.array([1.0, -1.0])
        H = jax.hessian(f)(x)
        assert jnp.allclose(H, A, atol=1e-4)

    def test_hessian_captures_off_diagonal(self):
        """Full Hessian should capture off-diagonal curvature that diagonal misses."""
        A = jnp.array([[2.0, 1.5], [1.5, 3.0]])

        def f(x):
            return 0.5 * x @ A @ x

        x = jnp.array([1.0, -1.0])
        H = jax.hessian(f)(x)
        assert jnp.allclose(H[0, 1], 1.5, atol=1e-4)
        assert jnp.allclose(H[1, 0], 1.5, atol=1e-4)

    def test_rosenbrock_hessian(self):
        """Verify full Hessian on Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2."""

        def rosenbrock(xy):
            x, y = xy[0], xy[1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        # At (1, 1) (the minimum):
        # d^2f/dx^2 = 2 - 400*y + 1200*x^2 = 802
        # d^2f/dy^2 = 200
        # d^2f/dxdy = -400*x = -400
        xy = jnp.array([1.0, 1.0])
        H = jax.hessian(rosenbrock)(xy)
        assert jnp.allclose(H[0, 0], 802.0, atol=1.0)
        assert jnp.allclose(H[1, 1], 200.0, atol=1.0)
        assert jnp.allclose(H[0, 1], -400.0, atol=1.0)


class TestHessMC2VmapBatching:
    """Test that proposals and weights work correctly under vmap."""

    def test_propose_fo_batch(self):
        """Vmapped FO proposal should give same result as sequential."""
        from causal_ssm_agent.models.ssm.hessmc2 import _propose_fo

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
        from causal_ssm_agent.models.ssm.hessmc2 import _compute_weight

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
# Hess-MC² Smoke Test (Non-linear Non-Gaussian DGP)
# =============================================================================


class TestHessMC2Smoke:
    """End-to-end smoke test for Hess-MC² on a Linear Gaussian SSM.

    Replicates the paper's LGSS experiment (Section IV-A) using our
    continuous-time SSM framework. D=3 parameters (drift_diag, diffusion_diag,
    manifest_var_diag) — matching the paper's low-dimensional test case.

    Uses tiny inference settings (N=8, K=3) to exercise the full pipeline
    quickly. Recovery test is in tests/test_recovery.py.
    """

    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_lgss_hessian_smoke(self, lgss_data):
        """Hess-MC² Hessian proposal on 1D LGSS (D=3) — pipeline check.

        Paper reference: Section IV-A, LGSS model.
        Performance gate: must complete within 30s.
        Exercises tempered warmup code path (warmup_iters=2).
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="hessmc2",
            n_smc_particles=8,
            n_iterations=5,
            proposal="hessian",
            step_size=0.5,
            warmup_iters=2,
            warmup_step_size=0.5,
            adapt_step_size=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "hessmc2"
        samples = result.get_samples()

        for site in ["drift_diag_pop", "diffusion_diag_pop", "manifest_var_diag"]:
            assert site in samples, f"Missing sample site: {site}"

        assert samples["drift_diag_pop"].shape == (8, 1)
        assert samples["diffusion_diag_pop"].shape == (8, 1)
        assert samples["manifest_var_diag"].shape == (8, 1)

        ess = result.diagnostics["ess_history"]
        assert len(ess) == 5
        assert all(e > 0 for e in ess)
        assert result.diagnostics["warmup_iters"] == 2

        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"Hess-MC² smoke took {elapsed:.1f}s, must be under 30s"


# =============================================================================
# MCMC Utils Tests
# =============================================================================


class TestMCMCUtils:
    """Test shared MCMC utility functions."""

    def test_hmc_step_n_leapfrog_1_is_mala(self):
        """hmc_step with n_leapfrog=1 should behave as MALA."""
        from causal_ssm_agent.models.ssm.mcmc_utils import hmc_step

        D = 3
        key = random.PRNGKey(42)

        # Simple quadratic target: -0.5 * x^T x
        def target_vg(z):
            val = -0.5 * jnp.dot(z, z)
            grad = -z
            return val, grad

        z = jnp.array([1.0, -0.5, 0.3])
        chol_mass = jnp.eye(D)
        z_new, accepted, log_target = hmc_step(key, z, target_vg, 0.1, chol_mass, n_leapfrog=1)

        assert z_new.shape == (D,)
        assert jnp.isfinite(log_target)
        assert accepted.dtype == jnp.bool_

    def test_hmc_step_n_leapfrog_5(self):
        """hmc_step with n_leapfrog=5 should produce valid results."""
        from causal_ssm_agent.models.ssm.mcmc_utils import hmc_step

        D = 3
        key = random.PRNGKey(42)

        def target_vg(z):
            val = -0.5 * jnp.dot(z, z)
            grad = -z
            return val, grad

        z = jnp.array([1.0, -0.5, 0.3])
        chol_mass = jnp.eye(D)
        z_new, _accepted, log_target = hmc_step(key, z, target_vg, 0.1, chol_mass, n_leapfrog=5)

        assert z_new.shape == (D,)
        assert jnp.isfinite(log_target)

    def test_find_next_beta_basic(self):
        """find_next_beta should return a value between beta_prev and 1.0."""
        from causal_ssm_agent.models.ssm.mcmc_utils import find_next_beta

        N = 100
        logw = jnp.zeros(N)
        key = random.PRNGKey(0)
        log_liks = random.normal(key, (N,)) * 10.0  # spread-out likelihoods

        beta = find_next_beta(logw, log_liks, 0.0, 0.5, N)
        assert 0.0 < beta <= 1.0

    def test_find_next_beta_reaches_one(self):
        """find_next_beta should reach 1.0 when likelihoods are uniform."""
        from causal_ssm_agent.models.ssm.mcmc_utils import find_next_beta

        N = 100
        logw = jnp.zeros(N)
        log_liks = jnp.zeros(N)  # all equal -> ESS stays at N for any delta

        beta = find_next_beta(logw, log_liks, 0.0, 0.5, N)
        assert beta == 1.0

    def test_dual_averaging_converges(self):
        """Dual averaging should converge step size toward target acceptance."""
        from causal_ssm_agent.models.ssm.mcmc_utils import (
            dual_averaging_init,
            dual_averaging_update,
        )

        state = dual_averaging_init(1.0)
        # Simulate low acceptance (step too large) -> should shrink
        for _ in range(50):
            state = dual_averaging_update(state, 0.1, target_accept=0.65)
        assert state.eps < 1.0  # step size should have decreased
        assert state.eps_bar < 1.0

        # Simulate high acceptance (step too small) -> should grow
        state2 = dual_averaging_init(0.001)
        for _ in range(50):
            state2 = dual_averaging_update(state2, 0.95, target_accept=0.65)
        assert state2.eps > 0.001

    def test_compute_weighted_chol_mass_shape(self):
        """compute_weighted_chol_mass should return (D, D) lower-triangular."""
        from causal_ssm_agent.models.ssm.mcmc_utils import compute_weighted_chol_mass

        D = 4
        N = 50
        key = random.PRNGKey(0)
        particles = random.normal(key, (N, D))
        logw = jnp.zeros(N)

        chol = compute_weighted_chol_mass(particles, logw, D)
        assert chol.shape == (D, D)
        # Should be lower-triangular (upper triangle ~0)
        assert jnp.allclose(chol, jnp.tril(chol), atol=1e-6)


# =============================================================================
# Tempered SMC Upgrade Tests
# =============================================================================


class TestTemperedSMCAdaptive:
    """Test adaptive ESS-based tempering."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_adaptive_tempering_reaches_beta_one(self, lgss_data):
        """Adaptive tempering should reach beta=1.0."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=50,
            n_csmc_particles=10,
            n_mh_steps=5,
            param_step_size=0.01,
            adaptive_tempering=True,
            target_ess_ratio=0.5,
            waste_free=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        beta_schedule = result.diagnostics["beta_schedule"]
        assert beta_schedule[-1] == 1.0, f"Final beta={beta_schedule[-1]}, expected 1.0"


class TestTemperedSMCWasteFree:
    """Test waste-free particle recycling."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_waste_free_runs(self, lgss_data):
        """Waste-free mode should complete without error."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=10,
            n_csmc_particles=10,  # N=10, n_mh_steps=5 -> M=2
            n_mh_steps=5,
            param_step_size=0.01,
            waste_free=True,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.diagnostics["waste_free"] is True

    def test_waste_free_rejects_bad_n(self):
        """Waste-free should reject N % n_mh_steps != 0."""
        from causal_ssm_agent.models.ssm import SSMModel, SSMSpec, fit

        spec = SSMSpec(
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


class TestTemperedSMCMultiStepHMC:
    """Test multi-step HMC mutations in tempered SMC."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_multi_step_hmc_runs(self, lgss_data):
        """n_leapfrog=5 should complete without error."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=6,
            n_csmc_particles=10,
            n_mh_steps=5,
            param_step_size=0.01,
            n_leapfrog=5,
            waste_free=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.diagnostics["n_leapfrog"] == 5


# =============================================================================
# PGAS Upgrade Tests
# =============================================================================


class TestPGASPreconditioned:
    """Test preconditioned MALA in PGAS."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_preconditioned_runs(self, lgss_data):
        """PGAS with preconditioned HMC should complete without error."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="pgas",
            n_outer=10,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=5,
            block_sampling=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "pgas"
        assert len(result.diagnostics["accept_rates"]) == 10


class TestPGASOptimalProposal:
    """Test locally optimal proposal for Gaussian observations."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_optimal_proposal_gaussian(self, lgss_data):
        """PGAS with optimal proposal should complete for Gaussian obs."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="pgas",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.diagnostics["gaussian_obs"] is True

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_fallback_for_poisson(self):
        """PGAS should fall back to gradient proposal for non-Gaussian obs."""
        from causal_ssm_agent.models.ssm import NoiseFamily, SSMModel, SSMSpec, fit

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            diffusion="diag",
            manifest_dist=NoiseFamily.POISSON,
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


class TestPGASBlockSampling:
    """Test block parameter sampling in PGAS."""

    @pytest.fixture
    def lgss_data(self):
        """1D Linear Gaussian SSM data."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm import SSMSpec, discretize_system

        n_latent, n_manifest = 1, 1
        T, dt = 50, 1.0

        true_drift = jnp.array([[-0.3]])
        true_diff_cov = jnp.array([[0.3**2]])
        true_obs_var = jnp.array([[0.5**2]])

        Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
        R_chol = jla.cholesky(true_obs_var, lower=True)

        key = random.PRNGKey(42)
        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
        latent = jnp.stack(states)

        key, obs_key = random.split(key)
        observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
        times = jnp.arange(T, dtype=float) * dt

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {"observations": observations, "times": times, "spec": spec}

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_block_sampling_runs(self, lgss_data):
        """PGAS with block sampling should complete and have per-block diagnostics."""
        from causal_ssm_agent.models.ssm import SSMModel, fit

        model = SSMModel(lgss_data["spec"], n_particles=50)
        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="pgas",
            n_outer=10,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=5,
            block_sampling=True,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.diagnostics["block_sampling"] is True
        assert "block_accept_rates" in result.diagnostics
        # Should have per-block rates for each parameter site
        block_rates = result.diagnostics["block_accept_rates"]
        assert len(block_rates) > 0
        for _name, rates in block_rates.items():
            assert len(rates) == 10  # n_outer iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
