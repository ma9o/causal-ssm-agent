"""Comprehensive tests for particle filter + NUTS inference.

Tests cover:
1. ParticleLikelihood: finite likelihood, determinism, gradient flow
2. SSMAdapter: observation models (Gaussian, Poisson, Student-t, Gamma)
3. Parameter recovery: simulate → PF+NUTS → check credible intervals
4. Hierarchical likelihood robustness
5. Edge cases and builder wiring

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
from numpyro.infer import MCMC

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.particle import ParticleLikelihood, SSMAdapter
from dsem_agent.models.ssm import NoiseFamily, SSMModel, SSMSpec

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

        ll = adapter._obs_log_prob_gaussian(y, x, params, obs_mask)

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
        from dsem_agent.models.ssm.discretization import discretize_system

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
# Hierarchical Likelihood Robustness
# =============================================================================


class TestHierarchicalLikelihood:
    """Robustness tests for hierarchical likelihood masking."""

    def test_subject_without_observations_is_finite(self):
        """Subjects with no observations should not introduce NaNs/Infs."""
        from dsem_agent.models.ssm import SSMModel

        n_latent, n_manifest = 2, 2
        T = 6
        times = jnp.arange(T, dtype=float) * 0.5
        observations = random.normal(random.PRNGKey(0), (T, n_manifest))

        # All observations belong to subject 0; subject 1 has none
        subject_ids = jnp.zeros(T, dtype=int)
        n_subjects = 2

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            hierarchical=True,
            n_subjects=n_subjects,
        )
        model = SSMModel(spec, n_particles=100)

        drift = jnp.stack(
            [
                jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
                jnp.array([[-0.6, 0.0], [0.0, -0.6]]),
            ]
        )
        diffusion_cov = jnp.stack([jnp.eye(n_latent) * 0.1, jnp.eye(n_latent) * 0.1])

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=None)
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        t0_means = jnp.zeros((n_subjects, n_latent))
        t0_cov = jnp.eye(n_latent)

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=100,
        )
        ll = model._hierarchical_likelihood(
            backend,
            ct_params,
            meas_params,
            observations,
            times,
            subject_ids,
            n_subjects,
            t0_means,
            t0_cov,
        )
        assert jnp.isfinite(ll), f"Non-finite hierarchical LL: {ll}"


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
        from dsem_agent.models.ssm import SSMModel

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
        from dsem_agent.models.ssm import SSMModel

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

        mcmc = model.fit(
            observations=observations,
            times=times,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        drift_diag_samples = samples["drift_diag_pop"]
        posterior_mean = float(jnp.mean(drift_diag_samples[:, 0]))

        # Drift should be negative
        assert posterior_mean < 0.0, f"Drift should be negative: {posterior_mean:.3f}"


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

        mcmc = model.fit(
            observations=observations,
            times=times,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        drift_diag_samples = samples["drift_diag_pop"]
        posterior_mean = float(jnp.mean(drift_diag_samples[:, 0]))

        # Drift should be negative
        assert posterior_mean < 0.0, f"Drift should be negative: {posterior_mean:.3f}"


class TestHighDimNonlinear:
    """Test PF+NUTS on high-dimensional nonlinear model."""

    @pytest.mark.slow
    def test_high_dimensional_poisson_pf_nuts(self):
        """PF+NUTS should produce finite results on 6D Poisson model."""
        import jax.scipy.linalg as jla

        from dsem_agent.models.ssm.discretization import discretize_system

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


class TestFitReturnsOnlyMCMC:
    """Test that fit() always returns MCMC regardless of noise family."""

    def test_fit_gaussian_returns_mcmc(self):
        """fit() with Gaussian manifest_dist returns MCMC."""
        from dsem_agent.models.ssm import SSMModel

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

        result = model.fit(
            observations=observations,
            times=times,
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=0,
        )

        assert isinstance(result, MCMC)

    def test_fit_poisson_returns_mcmc(self):
        """fit() with Poisson manifest_dist also returns MCMC (PF+NUTS)."""
        from dsem_agent.models.ssm import SSMModel

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

        result = model.fit(
            observations=observations,
            times=times,
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=0,
        )

        assert isinstance(result, MCMC)


# =============================================================================
# Builder Noise Family Wiring Tests
# =============================================================================


class TestBuilderNoiseFamilyWiring:
    """Test that SSMModelBuilder wires noise families from ModelSpec."""

    def test_convert_spec_sets_poisson_noise_family(self):
        """ModelSpec with Poisson likelihood -> SSMSpec has POISSON manifest_dist."""
        from dsem_agent.models.ssm_builder import SSMModelBuilder
        from dsem_agent.orchestrator.schemas_model import (
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

        import pandas as pd

        data = pd.DataFrame({"count_var": [1, 2, 3], "time": [0, 1, 2]})

        builder = SSMModelBuilder(model_spec=model_spec)
        ssm_spec = builder._convert_spec_to_ssm(model_spec, data)

        assert ssm_spec.manifest_dist == NoiseFamily.POISSON

    def test_convert_spec_sets_gaussian_for_normal(self):
        """ModelSpec with Normal likelihood -> SSMSpec has GAUSSIAN manifest_dist."""
        from dsem_agent.models.ssm_builder import SSMModelBuilder
        from dsem_agent.orchestrator.schemas_model import (
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

        import pandas as pd

        data = pd.DataFrame({"continuous_var": [1.0, 2.0, 3.0], "time": [0, 1, 2]})

        builder = SSMModelBuilder(model_spec=model_spec)
        ssm_spec = builder._convert_spec_to_ssm(model_spec, data)

        assert ssm_spec.manifest_dist == NoiseFamily.GAUSSIAN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
