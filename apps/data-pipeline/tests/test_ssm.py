"""Tests for State-Space Model NumPyro implementation.

Tests core functionality:
1. Matrix utilities (expm, Lyapunov solver, discretization)
2. ParticleLikelihood correctness
3. NumPyro model compilation and sampling
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest


class TestCoreUtilities:
    """Test core matrix utilities for state-space models."""

    def test_solve_lyapunov_simple(self):
        """Test Lyapunov solver with simple 2x2 case."""
        from causal_ssm_agent.models.ssm.discretization import solve_lyapunov

        # Simple stable drift matrix
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify: A*X + X*A' = -Q
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6), f"Residual: {residual}"

    def test_solve_lyapunov_coupled(self):
        """Test Lyapunov solver with coupled system."""
        from causal_ssm_agent.models.ssm.discretization import solve_lyapunov

        # Coupled drift
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        Q = jnp.array([[1.0, 0.2], [0.2, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6)

        # X should be symmetric
        assert jnp.allclose(X, X.T, atol=1e-10)

    def test_discretize_system_identity(self):
        """Test that dt=0 gives identity transformation."""
        from causal_ssm_agent.models.ssm.discretization import discretize_system

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        c = jnp.array([0.1, 0.2])

        # Very small dt should give ~identity drift
        disc_drift, disc_Q, _ = discretize_system(A, Q, c, dt=1e-6)

        assert jnp.allclose(disc_drift, jnp.eye(2), atol=1e-5)
        assert jnp.allclose(disc_Q, jnp.zeros((2, 2)), atol=1e-5)

    def test_discretize_system_unit_time(self):
        """Test discretization at dt=1."""
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm.discretization import discretize_system

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        c = jnp.array([0.1, 0.2])

        disc_drift, disc_Q, _ = discretize_system(A, Q, c, dt=1.0)

        # Check discrete drift = exp(A)
        expected_drift = jla.expm(A)
        assert jnp.allclose(disc_drift, expected_drift, atol=1e-6)

        # Q should be positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(disc_Q)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_compute_asymptotic_diffusion(self):
        """Test asymptotic diffusion computation."""
        from causal_ssm_agent.models.ssm.discretization import compute_asymptotic_diffusion

        A = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        Q = G @ G.T

        Q_inf = compute_asymptotic_diffusion(A, Q)

        # For diagonal A=-I and Q=I, Q_inf should be 0.5*I
        # Since A*Q_inf + Q_inf*A' = -Q => -2*Q_inf = -I => Q_inf = 0.5*I
        expected = 0.5 * jnp.eye(2)
        assert jnp.allclose(Q_inf, expected, atol=1e-6)


class TestDiscretizationMomentMatching:
    """Validate discretize_system against fine-grained Euler-Maruyama simulation."""

    @pytest.mark.parametrize(
        "drift, diffusion_cov, cint, dt",
        [
            # 1) Diagonal drift, identity diffusion, no intercept
            (
                jnp.array([[-1.0, 0.0], [0.0, -2.0]]),
                jnp.eye(2),
                None,
                0.5,
            ),
            # 2) Coupled drift, correlated diffusion, with intercept
            (
                jnp.array([[-1.0, 0.3], [0.2, -1.5]]),
                jnp.array([[1.0, 0.4], [0.4, 0.8]]),
                jnp.array([0.5, -0.3]),
                1.0,
            ),
            # 3) Strongly coupled, large dt (stresses the Lyapunov approach)
            (
                jnp.array([[-0.5, 0.4], [0.3, -0.6]]),
                jnp.array([[0.6, 0.1], [0.1, 0.9]]),
                jnp.array([1.0, 0.0]),
                2.0,
            ),
            # 4) 3-dimensional system
            (
                jnp.array([[-1.0, 0.2, 0.0], [0.1, -1.5, 0.3], [0.0, 0.1, -0.8]]),
                jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.1], [0.0, 0.1, 0.7]]),
                jnp.array([0.2, -0.1, 0.4]),
                0.8,
            ),
        ],
        ids=["diagonal", "coupled-correlated", "large-dt", "3d"],
    )
    def test_moments_match_euler_maruyama(self, drift, diffusion_cov, cint, dt):
        """Analytic (Ad, Qd, cd) must match empirical moments from EM simulation.

        For the SDE  dx = (A x + c) dt + G dW,  starting from deterministic x0:
          - E[x(dt)] = Ad @ x0 + cd
          - Cov[x(dt)] = Qd   (process noise covariance over one step)

        We validate by running many EM trajectories with a very fine sub-step
        and comparing sample statistics against the analytic formulas.
        """
        from causal_ssm_agent.models.ssm.discretization import discretize_system

        n = drift.shape[0]
        n_trajectories = 50_000
        dt_sim = 1e-4  # fine Euler-Maruyama sub-step
        n_steps = round(dt / dt_sim)

        # Cholesky factor of diffusion_cov = GG'
        G = jnp.linalg.cholesky(diffusion_cov)
        sqrt_dt_sim = jnp.sqrt(dt_sim)

        # Deterministic starting point
        key = random.PRNGKey(42)
        x0 = jnp.ones(n) * 0.5

        # --- Euler-Maruyama simulation ---
        keys = random.split(key, n_steps)
        x = jnp.broadcast_to(x0, (n_trajectories, n))  # (N, n)

        for i in range(n_steps):
            z = random.normal(keys[i], shape=(n_trajectories, n))
            dx_det = (x @ drift.T + (cint if cint is not None else 0.0)) * dt_sim
            dx_stoch = (z @ G.T) * sqrt_dt_sim
            x = x + dx_det + dx_stoch

        # Empirical moments
        emp_mean = jnp.mean(x, axis=0)
        emp_cov = jnp.cov(x.T, bias=False)

        # --- Analytic moments from discretize_system ---
        Ad, Qd, cd = discretize_system(drift, diffusion_cov, cint, dt)
        analytic_mean = Ad @ x0 + (cd if cd is not None else 0.0)
        analytic_cov = Qd

        # --- Compare ---
        # Mean: Monte-Carlo SE ~ sqrt(diag(Qd)/N), allow ~5 sigma
        mc_se_mean = jnp.sqrt(jnp.diag(analytic_cov) / n_trajectories)
        mean_tol = jnp.maximum(5.0 * mc_se_mean, 1e-3)  # floor at 1e-3
        mean_err = jnp.abs(analytic_mean - emp_mean)
        assert jnp.all(mean_err < mean_tol), (
            f"Mean mismatch:\n  analytic={analytic_mean}\n  empirical={emp_mean}\n"
            f"  error={mean_err}\n  tolerance={mean_tol}"
        )

        # Covariance: use relative tolerance for entries, absolute for near-zero
        cov_err = jnp.abs(analytic_cov - emp_cov)
        cov_scale = jnp.maximum(jnp.abs(analytic_cov), 1e-3)
        # Allow ~5% relative error (Monte-Carlo variance of sample cov ~ 1/sqrt(N))
        rel_tol = 0.05
        assert jnp.all(cov_err < rel_tol * cov_scale + 1e-3), (
            f"Covariance mismatch:\n  analytic=\n{analytic_cov}\n  empirical=\n{emp_cov}\n"
            f"  error=\n{cov_err}\n  rel_scale=\n{cov_scale}"
        )


class TestParticleLikelihoodBackend:
    """Test particle filter likelihood backend."""

    def test_pf_log_likelihood_finite(self):
        """Test that PF log-likelihood is finite for reasonable data."""
        from causal_ssm_agent.models.likelihoods.base import (
            CTParams,
            InitialStateParams,
            MeasurementParams,
        )
        from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

        T, n_latent, n_manifest = 10, 2, 2

        ct_params = CTParams(
            drift=jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
            diffusion_cov=0.1 * jnp.eye(n_latent),
            cint=None,
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=0.1 * jnp.eye(n_manifest),
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        observations = jnp.ones((T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T)

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=200,
        )
        lnc = backend.compute_log_likelihood(
            ct_params, meas_params, init, observations, time_intervals
        )

        # Returns (T,) cumulative log-normalizing constants
        assert lnc.shape == (T,)
        assert jnp.all(jnp.isfinite(lnc))


class TestSSMModel:
    """Test NumPyro state-space model."""

    def test_model_compiles(self):
        """Test that model compiles without errors."""
        from causal_ssm_agent.models.ssm import SSMModel, SSMSpec

        spec = SSMSpec(n_latent=2, n_manifest=2)
        model = SSMModel(spec, n_particles=50)

        # Create dummy data
        T = 10
        observations = jnp.ones((T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Try to trace the model (this will fail if there are shape errors)
        import numpyro

        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as trace:
            model.model(observations, times, likelihood_backend=model.make_likelihood_backend())

        # Check that key sites exist
        assert "drift_diag_pop" in trace
        assert "diffusion_diag_pop" in trace

    def test_prior_predictive(self):
        """Test prior predictive sampling (should skip PF via handlers.block)."""
        from causal_ssm_agent.models.ssm import SSMModel, SSMSpec
        from causal_ssm_agent.models.ssm.inference import prior_predictive

        spec = SSMSpec(n_latent=2, n_manifest=2)
        model = SSMModel(spec, n_particles=50)

        times = jnp.arange(10, dtype=float)
        prior_samples = prior_predictive(model, times, num_samples=10)

        # Should have samples for key parameters
        assert "drift" in prior_samples
        assert prior_samples["drift"].shape == (10, 2, 2)

    @pytest.mark.slow
    def test_fit_runs(self):
        """Test that fitting runs without errors (minimal samples)."""
        from causal_ssm_agent.models.ssm import SSMModel, SSMSpec, fit

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),  # Fix loadings to simplify
        )
        model = SSMModel(spec, n_particles=50)

        # Generate simple data
        T = 20
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Run with minimal samples using NUTS
        result = fit(
            model, observations, times, method="nuts", num_warmup=10, num_samples=10, num_chains=1
        )

        samples = result.get_samples()
        assert "drift_diag_pop" in samples


class TestSSMModelBuilder:
    """Test SSMModelBuilder pipeline integration."""

    def test_builder_with_ssm_spec(self):
        """Test building SSM from SSMSpec directly."""
        import polars as pl

        from causal_ssm_agent.models.ssm import SSMSpec
        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        # Create a SSMSpec directly
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            manifest_names=["mood", "stress"],
        )

        # Create sample data
        T = 20
        X = pl.DataFrame(
            {
                "mood": np.random.randn(T),
                "stress": np.random.randn(T),
                "time": np.arange(T, dtype=float),
            }
        )

        builder = SSMModelBuilder(ssm_spec=spec)
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec is not None
        assert builder._spec.n_manifest == 2
        assert builder._spec.n_latent == 2

    def test_builder_auto_detect(self):
        """Test auto-detection of manifest columns."""
        import polars as pl

        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        T = 15
        X = pl.DataFrame(
            {
                "x": np.random.randn(T),
                "y": np.random.randn(T),
                "time": np.arange(T, dtype=float),
            }
        )

        builder = SSMModelBuilder()
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec is not None
        assert builder._spec.n_manifest == 2

    @pytest.mark.slow
    def test_builder_fit(self):
        """Test fitting via builder interface."""
        import polars as pl

        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

        T = 20
        X = pl.DataFrame(
            {
                "x": np.random.randn(T) * 0.5,
                "y": np.random.randn(T) * 0.5,
                "time": np.arange(T, dtype=float),
            }
        )

        builder = SSMModelBuilder(
            sampler_config={
                "num_warmup": 10,
                "num_samples": 10,
                "num_chains": 1,
            }
        )
        builder.fit(X)

        samples = builder.get_samples()
        # Builder defaults to SVI, which returns deterministic sites
        assert "drift" in samples


class TestDistributionFamily:
    """Test DistributionFamily enum works with SSMSpec."""

    def test_distribution_family_values(self):
        """DistributionFamily should have expected lowercase values."""
        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

        assert DistributionFamily.GAUSSIAN == "gaussian"
        assert DistributionFamily.STUDENT_T == "student_t"
        assert DistributionFamily.POISSON == "poisson"
        assert DistributionFamily.GAMMA == "gamma"
        assert DistributionFamily.BERNOULLI == "bernoulli"

    def test_distribution_family_in_spec(self):
        """SSMSpec should accept DistributionFamily enum values."""
        from causal_ssm_agent.models.ssm import SSMSpec
        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=DistributionFamily.GAUSSIAN,
            manifest_dist=DistributionFamily.POISSON,
        )
        assert spec.diffusion_dist == DistributionFamily.GAUSSIAN
        assert spec.manifest_dist == DistributionFamily.POISSON


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
