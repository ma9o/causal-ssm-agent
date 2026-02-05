"""Tests for State-Space Model NumPyro implementation.

Tests core functionality:
1. Matrix utilities (expm, Lyapunov solver, discretization)
2. Kalman filter correctness
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
        from dsem_agent.models.ssm.discretization import solve_lyapunov

        # Simple stable drift matrix
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify: A*X + X*A' = -Q
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6), f"Residual: {residual}"

    def test_solve_lyapunov_coupled(self):
        """Test Lyapunov solver with coupled system."""
        from dsem_agent.models.ssm.discretization import solve_lyapunov

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
        from dsem_agent.models.ssm.discretization import discretize_system

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

        from dsem_agent.models.ssm.discretization import discretize_system

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
        from dsem_agent.models.ssm.discretization import compute_asymptotic_diffusion

        A = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        Q = G @ G.T

        Q_inf = compute_asymptotic_diffusion(A, Q)

        # For diagonal A=-I and Q=I, Q_inf should be 0.5*I
        # Since A*Q_inf + Q_inf*A' = -Q => -2*Q_inf = -I => Q_inf = 0.5*I
        expected = 0.5 * jnp.eye(2)
        assert jnp.allclose(Q_inf, expected, atol=1e-6)


class TestKalmanFilter:
    """Test Kalman filter implementation."""

    def test_kalman_predict(self):
        """Test Kalman prediction step."""
        from dsem_agent.models.ssm.kalman import kalman_predict

        state_mean = jnp.array([1.0, 0.0])
        state_cov = jnp.eye(2)
        discrete_drift = 0.9 * jnp.eye(2)
        discrete_Q = 0.1 * jnp.eye(2)
        discrete_cint = jnp.array([0.1, 0.1])

        pred_mean, pred_cov = kalman_predict(
            state_mean, state_cov, discrete_drift, discrete_Q, discrete_cint
        )

        # Check mean prediction
        expected_mean = discrete_drift @ state_mean + discrete_cint
        assert jnp.allclose(pred_mean, expected_mean)

        # Check covariance prediction
        expected_cov = discrete_drift @ state_cov @ discrete_drift.T + discrete_Q
        assert jnp.allclose(pred_cov, expected_cov)

    def test_kalman_update_reduces_uncertainty(self):
        """Test that Kalman update reduces uncertainty."""
        from dsem_agent.models.ssm.kalman import kalman_update_simple

        pred_mean = jnp.array([0.0, 0.0])
        pred_cov = jnp.eye(2)
        observation = jnp.array([1.0, 0.5])
        lambda_mat = jnp.eye(2)
        manifest_means = jnp.zeros(2)
        manifest_cov = 0.5 * jnp.eye(2)

        upd_mean, upd_cov, ll = kalman_update_simple(
            pred_mean, pred_cov, observation, lambda_mat, manifest_means, manifest_cov
        )

        # Updated covariance should have smaller trace (less uncertainty)
        assert jnp.trace(upd_cov) < jnp.trace(pred_cov)

        # Updated mean should be closer to observation
        assert jnp.linalg.norm(upd_mean - observation) < jnp.linalg.norm(
            pred_mean - observation
        )

        # Log-likelihood should be finite
        assert jnp.isfinite(ll)

    def test_kalman_log_likelihood_finite(self):
        """Test that log-likelihood is finite for reasonable data."""
        from dsem_agent.models.ssm.kalman import kalman_log_likelihood

        T, n_latent, n_manifest = 10, 2, 2

        # Simple observations
        observations = jnp.ones((T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T)

        # Stable parameters
        drift = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        diffusion_cov = 0.1 * jnp.eye(n_latent)
        cint = None
        lambda_mat = jnp.eye(n_manifest, n_latent)
        manifest_means = jnp.zeros(n_manifest)
        manifest_cov = 0.1 * jnp.eye(n_manifest)
        t0_mean = jnp.zeros(n_latent)
        t0_cov = jnp.eye(n_latent)

        ll = kalman_log_likelihood(
            observations,
            time_intervals,
            drift,
            diffusion_cov,
            cint,
            lambda_mat,
            manifest_means,
            manifest_cov,
            t0_mean,
            t0_cov,
        )

        assert jnp.isfinite(ll)


class TestSSMModel:
    """Test NumPyro state-space model."""

    def test_model_compiles(self):
        """Test that model compiles without errors."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        spec = SSMSpec(n_latent=2, n_manifest=2)
        model = SSMModel(spec)

        # Create dummy data
        T = 10
        observations = jnp.ones((T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Try to trace the model (this will fail if there are shape errors)
        import numpyro

        with numpyro.handlers.seed(rng_seed=0), numpyro.handlers.trace() as trace:
            model.model(observations, times)

        # Check that key sites exist
        assert "drift_diag_pop" in trace
        assert "diffusion_diag_pop" in trace

    def test_prior_predictive(self):
        """Test prior predictive sampling."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        spec = SSMSpec(n_latent=2, n_manifest=2)
        model = SSMModel(spec)

        times = jnp.arange(10, dtype=float)
        prior_samples = model.prior_predictive(times, num_samples=10)

        # Should have samples for key parameters
        assert "drift" in prior_samples
        assert prior_samples["drift"].shape == (10, 2, 2)

    @pytest.mark.slow
    def test_fit_runs(self):
        """Test that fitting runs without errors (minimal samples)."""
        from dsem_agent.models.ssm import SSMModel, SSMSpec

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),  # Fix loadings to simplify
        )
        model = SSMModel(spec)

        # Generate simple data
        T = 20
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Run with minimal samples
        mcmc = model.fit(
            observations, times, num_warmup=10, num_samples=10, num_chains=1
        )

        samples = mcmc.get_samples()
        assert "drift_diag_pop" in samples


class TestSSMModelBuilder:
    """Test SSMModelBuilder pipeline integration."""

    def test_builder_with_ssm_spec(self):
        """Test building SSM from SSMSpec directly."""
        import pandas as pd

        from dsem_agent.models.ssm import SSMSpec
        from dsem_agent.models.ssm_builder import SSMModelBuilder

        # Create a SSMSpec directly
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            manifest_names=["mood", "stress"],
        )

        # Create sample data
        T = 20
        X = pd.DataFrame({
            "mood": np.random.randn(T),
            "stress": np.random.randn(T),
            "time": np.arange(T, dtype=float),
        })

        builder = SSMModelBuilder(ssm_spec=spec)
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2
        assert builder._spec.n_latent == 2

    def test_builder_auto_detect(self):
        """Test auto-detection of manifest columns."""
        import pandas as pd

        from dsem_agent.models.ssm_builder import SSMModelBuilder

        T = 15
        X = pd.DataFrame({
            "x": np.random.randn(T),
            "y": np.random.randn(T),
            "time": np.arange(T, dtype=float),
        })

        builder = SSMModelBuilder()
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2

    @pytest.mark.slow
    def test_builder_fit(self):
        """Test fitting via builder interface."""
        import pandas as pd

        from dsem_agent.models.ssm_builder import SSMModelBuilder

        T = 20
        X = pd.DataFrame({
            "x": np.random.randn(T) * 0.5,
            "y": np.random.randn(T) * 0.5,
            "time": np.arange(T, dtype=float),
        })

        builder = SSMModelBuilder(
            sampler_config={
                "num_warmup": 10,
                "num_samples": 10,
                "num_chains": 1,
            }
        )
        builder.fit(X)

        samples = builder.get_samples()
        assert "drift_diag_pop" in samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
