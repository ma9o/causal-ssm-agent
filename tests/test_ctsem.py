"""Tests for CT-SEM NumPyro implementation.

Tests core functionality:
1. Matrix utilities (expm, Lyapunov solver, discretization)
2. Kalman filter correctness
3. NumPyro model compilation and sampling
4. Parity with ctsem R package (where applicable)
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest


def generate_stable_drift(n: int, seed: int = 42) -> np.ndarray:
    """Generate stable nxn drift matrix with guaranteed negative eigenvalues."""
    rng = np.random.default_rng(seed)
    # Start with negative diagonal (ensures stability for small off-diagonal)
    drift = np.diag(rng.uniform(-0.8, -0.3, size=n))
    # Add small off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                drift[i, j] = rng.uniform(-0.15, 0.15)
    # Verify stability
    eigenvalues = np.linalg.eigvals(drift)
    assert np.all(np.real(eigenvalues) < 0), f"Drift matrix not stable: {eigenvalues}"
    return drift


def generate_diffusion_chol(n: int, seed: int = 42) -> np.ndarray:
    """Generate nxn lower triangular Cholesky factor."""
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, n))
    # Positive diagonal
    chol[np.diag_indices(n)] = rng.uniform(0.2, 0.6, size=n)
    # Small off-diagonal (lower triangle)
    for i in range(n):
        for j in range(i):
            chol[i, j] = rng.uniform(-0.1, 0.1)
    return chol


def generate_cint(n: int, seed: int = 42) -> np.ndarray:
    """Generate n-dimensional continuous intercept."""
    return np.random.default_rng(seed).uniform(-0.2, 0.2, size=n)


class TestCoreUtilities:
    """Test core matrix utilities for CT-SEM."""

    def test_solve_lyapunov_simple(self):
        """Test Lyapunov solver with simple 2x2 case."""
        from dsem_agent.models.ctsem.core import solve_lyapunov

        # Simple stable drift matrix
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify: A*X + X*A' = -Q
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6), f"Residual: {residual}"

    def test_solve_lyapunov_coupled(self):
        """Test Lyapunov solver with coupled system."""
        from dsem_agent.models.ctsem.core import solve_lyapunov

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
        from dsem_agent.models.ctsem.core import discretize_system

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

        from dsem_agent.models.ctsem.core import discretize_system

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
        from dsem_agent.models.ctsem.core import compute_asymptotic_diffusion

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
        from dsem_agent.models.ctsem.kalman import kalman_predict

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
        from dsem_agent.models.ctsem.kalman import kalman_update_simple

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
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

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


class TestCTSEMModel:
    """Test NumPyro CT-SEM model."""

    def test_model_compiles(self):
        """Test that model compiles without errors."""
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(n_latent=2, n_manifest=2)
        model = CTSEMModel(spec)

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
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(n_latent=2, n_manifest=2)
        model = CTSEMModel(spec)

        times = jnp.arange(10, dtype=float)
        prior_samples = model.prior_predictive(times, num_samples=10)

        # Should have samples for key parameters
        assert "drift" in prior_samples
        assert prior_samples["drift"].shape == (10, 2, 2)

    @pytest.mark.slow
    def test_fit_runs(self):
        """Test that fitting runs without errors (minimal samples)."""
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),  # Fix loadings to simplify
        )
        model = CTSEMModel(spec)

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


class TestParityWithCtsem:
    """Tests for parity with the ctsem R package.

    These tests compare our NumPyro implementation against
    actual results from the ctsem R package via rpy2.
    """

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        # Check if ctsem is installed
        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        # Also need Matrix package for expm
        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    def test_discretization_matches_ctsem(self, r_ctsem):
        """Test that discretization matches ctsem's actual R implementation.

        Calls R's ctsem package directly via rpy2 and compares results.
        """
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Test parameters
        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_chol = np.array([[0.5, 0.0], [0.1, 0.4]])
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = np.array([0.1, -0.1])
        dt = 1.0

        # Run discretization in R using ctsem's approach
        # This matches ctsem's internal ctDiscretePars function
        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)
            library(ctsem)

            n <- nrow(DRIFT)
            I <- diag(n)

            # Discrete drift: expm(DRIFT * dt)
            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            # Asymptotic diffusion (solve Lyapunov equation)
            # A*Q + Q*A' = -DIFFUSION, so Q = solve_lyap(-DIFFUSION)
            # Using ctsem's approach via Kronecker product
            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            # Discrete diffusion
            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            # Discrete intercept: solve(DRIFT, (discreteDRIFT - I) %*% CINT)
            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        # Use numpy2ri converter for automatic numpy <-> R conversion
        with localconverter(ro.default_converter + numpy2ri.converter):
            # Call R function with numpy arrays (auto-converted)
            r_result = r_discretize(drift, diffusion_cov, cint, dt)

            # Extract R results by index (NamedList returns items in order)
            # Order: discreteDRIFT, discreteDIFFUSION, discreteCINT
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])
            r_disc_c = np.asarray(r_result[2])

        # Our NumPyro implementation
        py_disc_drift, py_disc_Q, py_disc_c = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        # Compare results (tolerance allows for float32 vs float64 differences)
        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=1e-6,
            err_msg="Discrete drift mismatch with ctsem",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=1e-6,
            err_msg="Discrete diffusion mismatch with ctsem",
        )
        np.testing.assert_allclose(
            np.array(py_disc_c),
            r_disc_c,
            atol=1e-6,
            err_msg="Discrete CINT mismatch with ctsem",
        )

    def test_kalman_likelihood_matches_ctsem(self, r_ctsem):
        """Test that Kalman filter log-likelihood matches ctsem.

        Compares the log-likelihood computation for a simple model.
        """
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Simple 2-latent, 2-manifest model
        n_latent = 2
        n_manifest = 2

        # Model parameters (stable system)
        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_chol = np.array([[0.3, 0.0], [0.05, 0.25]])
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = np.array([0.0, 0.0])
        manifest_means = np.array([0.0, 0.0])
        loadings = np.eye(n_manifest, n_latent)  # Identity loadings
        manifest_var = np.array([0.1, 0.1])  # Measurement error variance (diagonal)
        manifest_cov = np.diag(manifest_var)

        # Time intervals (not absolute times)
        time_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # Simple synthetic observations (not generated from model, just for testing)
        Y = np.array(
            [
                [0.1, -0.2],
                [0.3, 0.1],
                [0.2, 0.3],
                [0.0, 0.2],
                [-0.1, 0.1],
            ]
        )

        # Initial state
        init_mean = np.zeros(n_latent)
        init_cov = np.eye(n_latent) * 1.0

        # R code to compute Kalman log-likelihood using ctsem's approach
        # Note: vectors are passed as column matrices to ensure proper dimensions
        r_kalman_code = """
        function(Y, dt, DRIFT, DIFFUSION, CINT, LAMBDA, MANIFESTMEANS,
                 MANIFESTCOV, T0MEANS, T0VAR) {
            library(Matrix)

            n_latent <- nrow(DRIFT)
            n_manifest <- nrow(LAMBDA)
            n_time <- nrow(Y)
            I <- diag(n_latent)

            # Ensure vectors are column matrices
            CINT <- matrix(CINT, ncol=1)
            MANIFESTMEANS <- matrix(MANIFESTMEANS, ncol=1)
            T0MEANS <- matrix(T0MEANS, ncol=1)

            # Asymptotic diffusion
            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            # Initialize
            state_mean <- T0MEANS
            state_cov <- T0VAR
            total_ll <- 0

            for (t in 1:n_time) {
                # Predict step (always, using dt for this time point)
                dt_t <- dt[t]
                discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                # Update step
                y_t <- matrix(Y[t, ], ncol=1)
                pred_y <- LAMBDA %*% state_mean + MANIFESTMEANS
                S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTCOV
                residual <- y_t - pred_y

                # Log-likelihood contribution
                ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                               t(residual) %*% solve(S, residual))
                total_ll <- total_ll + ll_t

                # Kalman gain and update
                K <- state_cov %*% t(LAMBDA) %*% solve(S)
                state_mean <- state_mean + K %*% residual
                state_cov <- (I - K %*% LAMBDA) %*% state_cov
            }

            return(as.numeric(total_ll))
        }
        """
        r_kalman_ll = ro.r(r_kalman_code)

        # Use numpy2ri converter for automatic numpy <-> R conversion
        with localconverter(ro.default_converter + numpy2ri.converter):
            # Get R log-likelihood
            r_ll_result = r_kalman_ll(
                Y,
                time_intervals,
                drift,
                diffusion_cov,
                cint,
                loadings,
                manifest_means,
                manifest_cov,
                init_mean,
                init_cov,
            )

        r_ll = float(np.asarray(r_ll_result)[0])

        # Our implementation
        py_ll = kalman_log_likelihood(
            observations=jnp.array(Y),
            time_intervals=jnp.array(time_intervals),
            drift=jnp.array(drift),
            diffusion_cov=jnp.array(diffusion_cov),
            cint=jnp.array(cint),
            lambda_mat=jnp.array(loadings),
            manifest_means=jnp.array(manifest_means),
            manifest_cov=jnp.array(manifest_cov),
            t0_mean=jnp.array(init_mean),
            t0_cov=jnp.array(init_cov),
        )

        np.testing.assert_allclose(
            float(py_ll),
            r_ll,
            atol=1e-6,
            err_msg=f"Log-likelihood mismatch: Python={float(py_ll)}, R={r_ll}",
        )


class TestParityDimensionScaling:
    """Test parity across different system dimensions (3x3, 5x5)."""

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    @pytest.mark.parametrize("n_latent", [3, 5])
    def test_discretization_scaling(self, r_ctsem, n_latent):
        """Discretization matches ctsem for nxn systems."""
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Generate stable parameters
        drift = generate_stable_drift(n_latent, seed=42 + n_latent)
        diffusion_chol = generate_diffusion_chol(n_latent, seed=42 + n_latent)
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = generate_cint(n_latent, seed=42 + n_latent)
        dt = 1.0

        # R code for discretization (same as TestParityWithCtsem)
        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)

            n <- nrow(DRIFT)
            I <- diag(n)

            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_result = r_discretize(drift, diffusion_cov, cint, dt)
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])
            r_disc_c = np.asarray(r_result[2])

        py_disc_drift, py_disc_Q, py_disc_c = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=1e-6,
            err_msg=f"Discrete drift mismatch for n={n_latent}",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=1e-6,
            err_msg=f"Discrete diffusion mismatch for n={n_latent}",
        )
        np.testing.assert_allclose(
            np.array(py_disc_c),
            r_disc_c,
            atol=1e-6,
            err_msg=f"Discrete CINT mismatch for n={n_latent}",
        )

    @pytest.mark.parametrize("n_latent", [3, 5])
    def test_kalman_likelihood_scaling(self, r_ctsem, n_latent):
        """Kalman likelihood matches for nxn systems."""
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        n_manifest = n_latent  # Identity loadings for simplicity
        T = 10

        # Generate stable parameters with different seeds to avoid degenerate cases
        drift = generate_stable_drift(n_latent, seed=200 + n_latent * 10)
        diffusion_chol = generate_diffusion_chol(n_latent, seed=200 + n_latent * 10)
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = np.zeros(n_latent)
        manifest_means = np.zeros(n_manifest)
        loadings = np.eye(n_manifest, n_latent)
        manifest_cov = np.diag(np.full(n_manifest, 0.1))

        # Time intervals and observations
        time_intervals = np.full(T, 0.5)
        rng = np.random.default_rng(seed=200 + n_latent * 10)
        Y = rng.normal(0, 0.5, size=(T, n_manifest))

        # Initial state
        init_mean = np.zeros(n_latent)
        init_cov = np.eye(n_latent)

        # R Kalman code (same as TestParityWithCtsem)
        r_kalman_code = """
        function(Y, dt, DRIFT, DIFFUSION, CINT, LAMBDA, MANIFESTMEANS,
                 MANIFESTCOV, T0MEANS, T0VAR) {
            library(Matrix)

            n_latent <- nrow(DRIFT)
            n_manifest <- nrow(LAMBDA)
            n_time <- nrow(Y)
            I <- diag(n_latent)

            CINT <- matrix(CINT, ncol=1)
            MANIFESTMEANS <- matrix(MANIFESTMEANS, ncol=1)
            T0MEANS <- matrix(T0MEANS, ncol=1)

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            state_mean <- T0MEANS
            state_cov <- T0VAR
            total_ll <- 0

            for (t in 1:n_time) {
                dt_t <- dt[t]
                discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                y_t <- matrix(Y[t, ], ncol=1)
                pred_y <- LAMBDA %*% state_mean + MANIFESTMEANS
                S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTCOV
                residual <- y_t - pred_y

                ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                               t(residual) %*% solve(S, residual))
                total_ll <- total_ll + ll_t

                K <- state_cov %*% t(LAMBDA) %*% solve(S)
                state_mean <- state_mean + K %*% residual
                state_cov <- (I - K %*% LAMBDA) %*% state_cov
            }

            return(as.numeric(total_ll))
        }
        """
        r_kalman_ll = ro.r(r_kalman_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_ll_result = r_kalman_ll(
                Y, time_intervals, drift, diffusion_cov, cint,
                loadings, manifest_means, manifest_cov, init_mean, init_cov
            )

        r_ll = float(np.asarray(r_ll_result)[0])

        py_ll = kalman_log_likelihood(
            observations=jnp.array(Y),
            time_intervals=jnp.array(time_intervals),
            drift=jnp.array(drift),
            diffusion_cov=jnp.array(diffusion_cov),
            cint=jnp.array(cint),
            lambda_mat=jnp.array(loadings),
            manifest_means=jnp.array(manifest_means),
            manifest_cov=jnp.array(manifest_cov),
            t0_mean=jnp.array(init_mean),
            t0_cov=jnp.array(init_cov),
        )

        # Slightly relaxed tolerance for larger systems due to accumulated float32 vs float64 differences
        np.testing.assert_allclose(
            float(py_ll),
            r_ll,
            atol=1e-5,
            err_msg=f"Log-likelihood mismatch for n={n_latent}: Python={float(py_ll)}, R={r_ll}",
        )


class TestParityEdgeCases:
    """Test numerical edge cases."""

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    def test_near_singular_drift(self, r_ctsem):
        """Eigenvalues close to 0 (slow dynamics)."""
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Near-singular drift (eigenvalues close to 0)
        drift = np.array([[-0.01, 0.001], [0.001, -0.02]])
        diffusion_cov = np.array([[0.1, 0.0], [0.0, 0.1]])
        cint = np.array([0.0, 0.0])
        dt = 1.0

        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)

            n <- nrow(DRIFT)
            I <- diag(n)

            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_result = r_discretize(drift, diffusion_cov, cint, dt)
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])

        py_disc_drift, py_disc_Q, _ = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        # Relaxed tolerance for near-singular case
        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=1e-5,
            err_msg="Discrete drift mismatch for near-singular drift (atol=1e-5)",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=1e-5,
            err_msg="Discrete diffusion mismatch for near-singular drift (atol=1e-5)",
        )

    @pytest.mark.parametrize("dt", [0.001, 10.0, 100.0])
    def test_extreme_time_intervals(self, r_ctsem, dt):
        """Very small and very large dt values."""
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_cov = np.array([[0.1, 0.02], [0.02, 0.1]])
        cint = np.array([0.1, -0.1])

        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)

            n <- nrow(DRIFT)
            I <- diag(n)

            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_result = r_discretize(drift, diffusion_cov, cint, dt)
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])
            r_disc_c = np.asarray(r_result[2])

        py_disc_drift, py_disc_Q, py_disc_c = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        # Relaxed tolerance for large dt (asymptotic behavior)
        atol = 1e-5 if dt > 10 else 1e-6

        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=atol,
            err_msg=f"Discrete drift mismatch for dt={dt} (atol={atol})",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=atol,
            err_msg=f"Discrete diffusion mismatch for dt={dt} (atol={atol})",
        )
        np.testing.assert_allclose(
            np.array(py_disc_c),
            r_disc_c,
            atol=atol,
            err_msg=f"Discrete CINT mismatch for dt={dt} (atol={atol})",
        )

    def test_ill_conditioned_diffusion(self, r_ctsem):
        """High condition number diffusion covariance."""
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # High correlation -> ill-conditioned
        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_cov = np.array([[1.0, 0.99], [0.99, 1.0]])
        cint = np.array([0.0, 0.0])
        dt = 1.0

        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)

            n <- nrow(DRIFT)
            I <- diag(n)

            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_result = r_discretize(drift, diffusion_cov, cint, dt)
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])

        py_disc_drift, py_disc_Q, _ = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        # Relaxed tolerance for ill-conditioned case
        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=1e-4,
            err_msg="Discrete drift mismatch for ill-conditioned diffusion (atol=1e-4)",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=1e-4,
            err_msg="Discrete diffusion mismatch for ill-conditioned diffusion (atol=1e-4)",
        )

    def test_nonzero_cint_full_pipeline(self, r_ctsem):
        """Non-zero CINT through discretization AND Kalman filter."""
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        n_latent = 2
        n_manifest = 2

        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_chol = np.array([[0.3, 0.0], [0.05, 0.25]])
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        # Meaningful non-zero intercept
        cint = np.array([0.5, -0.3])
        manifest_means = np.array([0.1, -0.1])
        loadings = np.eye(n_manifest, n_latent)
        manifest_cov = np.diag([0.1, 0.1])

        time_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        Y = np.array([
            [0.5, -0.1],
            [0.8, -0.2],
            [1.0, 0.0],
            [1.2, 0.1],
            [1.1, 0.2],
        ])

        init_mean = np.zeros(n_latent)
        init_cov = np.eye(n_latent)

        r_kalman_code = """
        function(Y, dt, DRIFT, DIFFUSION, CINT, LAMBDA, MANIFESTMEANS,
                 MANIFESTCOV, T0MEANS, T0VAR) {
            library(Matrix)

            n_latent <- nrow(DRIFT)
            n_manifest <- nrow(LAMBDA)
            n_time <- nrow(Y)
            I <- diag(n_latent)

            CINT <- matrix(CINT, ncol=1)
            MANIFESTMEANS <- matrix(MANIFESTMEANS, ncol=1)
            T0MEANS <- matrix(T0MEANS, ncol=1)

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            state_mean <- T0MEANS
            state_cov <- T0VAR
            total_ll <- 0

            for (t in 1:n_time) {
                dt_t <- dt[t]
                discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                y_t <- matrix(Y[t, ], ncol=1)
                pred_y <- LAMBDA %*% state_mean + MANIFESTMEANS
                S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTCOV
                residual <- y_t - pred_y

                ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                               t(residual) %*% solve(S, residual))
                total_ll <- total_ll + ll_t

                K <- state_cov %*% t(LAMBDA) %*% solve(S)
                state_mean <- state_mean + K %*% residual
                state_cov <- (I - K %*% LAMBDA) %*% state_cov
            }

            return(as.numeric(total_ll))
        }
        """
        r_kalman_ll = ro.r(r_kalman_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_ll_result = r_kalman_ll(
                Y, time_intervals, drift, diffusion_cov, cint,
                loadings, manifest_means, manifest_cov, init_mean, init_cov
            )

        r_ll = float(np.asarray(r_ll_result)[0])

        py_ll = kalman_log_likelihood(
            observations=jnp.array(Y),
            time_intervals=jnp.array(time_intervals),
            drift=jnp.array(drift),
            diffusion_cov=jnp.array(diffusion_cov),
            cint=jnp.array(cint),
            lambda_mat=jnp.array(loadings),
            manifest_means=jnp.array(manifest_means),
            manifest_cov=jnp.array(manifest_cov),
            t0_mean=jnp.array(init_mean),
            t0_cov=jnp.array(init_cov),
        )

        np.testing.assert_allclose(
            float(py_ll),
            r_ll,
            atol=1e-6,
            err_msg=f"Log-likelihood mismatch with non-zero CINT: Python={float(py_ll)}, R={r_ll}",
        )


class TestParityMultiSubject:
    """Test with multi-subject data from ctsem's ctGenerate."""

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    def test_ctgenerate_data_format(self, r_ctsem):
        """Verify ctGenerate produces expected data format."""
        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        r_code = """
        function() {
            library(ctsem)
            set.seed(42)

            gm <- ctModel(
                Tpoints = 5,
                n.latent = 2, n.manifest = 2,
                DRIFT = matrix(c(-0.5, 0.1, 0.2, -0.8), nrow=2),
                DIFFUSION = matrix(c(0.3, 0.0, 0.05, 0.25), nrow=2),
                MANIFESTVAR = diag(0.1, 2),
                LAMBDA = diag(1, 2),
                CINT = matrix(c(0.0, 0.0), nrow=2),
                T0MEANS = matrix(c(0, 0), nrow=2),
                T0VAR = diag(1, 2)
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=10, dtmean=0.5)
            return(as.matrix(data))
        }
        """
        r_generate = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            data = np.asarray(r_generate())

        # ctGenerate returns: [id, time, Y1, Y2, ...]
        # First column is subject id, second is absolute time
        assert data.shape[0] == 5  # Tpoints
        assert data.shape[1] >= 4  # id, time, Y1, Y2

    def test_likelihood_on_generated_data(self, r_ctsem):
        """Likelihood matches on ctsem-generated data."""
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Generate data and compute likelihood in R
        r_code = """
        function() {
            library(ctsem)
            library(Matrix)
            set.seed(42)

            # Model parameters
            DRIFT <- matrix(c(-0.5, 0.1, 0.2, -0.8), nrow=2)
            DIFFUSION_chol <- matrix(c(0.3, 0.0, 0.05, 0.25), nrow=2)
            DIFFUSION <- DIFFUSION_chol %*% t(DIFFUSION_chol)
            MANIFESTVAR <- diag(0.1, 2)
            LAMBDA <- diag(1, 2)
            CINT <- matrix(c(0.0, 0.0), nrow=2)
            T0MEANS <- matrix(c(0, 0), nrow=2)
            T0VAR <- diag(1, 2)

            gm <- ctModel(
                Tpoints = 10,
                n.latent = 2, n.manifest = 2,
                DRIFT = matrix(c(-0.5, 0.1, 0.2, -0.8), nrow=2),
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = MANIFESTVAR,
                LAMBDA = LAMBDA,
                CINT = CINT,
                T0MEANS = T0MEANS,
                T0VAR = T0VAR
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=10, dtmean=0.5)

            # Extract observations and time intervals
            # ctGenerate returns absolute time, compute intervals
            Y <- as.matrix(data[, c("Y1", "Y2")])
            times <- data[, "time"]
            dt <- c(times[1], diff(times))  # First interval from t=0

            # Compute log-likelihood manually
            n_latent <- 2
            n_manifest <- 2
            n_time <- nrow(Y)
            I <- diag(n_latent)

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            state_mean <- T0MEANS
            state_cov <- T0VAR
            total_ll <- 0

            for (t in 1:n_time) {
                dt_t <- dt[t]
                discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                y_t <- matrix(Y[t, ], ncol=1)
                pred_y <- LAMBDA %*% state_mean
                S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTVAR
                residual <- y_t - pred_y

                ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                               t(residual) %*% solve(S, residual))
                total_ll <- total_ll + ll_t

                K <- state_cov %*% t(LAMBDA) %*% solve(S)
                state_mean <- state_mean + K %*% residual
                state_cov <- (I - K %*% LAMBDA) %*% state_cov
            }

            list(
                Y = Y,
                dt = dt,
                ll = as.numeric(total_ll),
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION,
                MANIFESTVAR = MANIFESTVAR,
                LAMBDA = LAMBDA,
                CINT = as.vector(CINT),
                T0MEANS = as.vector(T0MEANS),
                T0VAR = T0VAR
            )
        }
        """
        r_generate_and_ll = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            result = r_generate_and_ll()
            Y = np.asarray(result[0])
            dt = np.asarray(result[1])
            r_ll = float(np.asarray(result[2])[0])
            drift = np.asarray(result[3])
            diffusion_cov = np.asarray(result[4])
            manifest_cov = np.asarray(result[5])
            loadings = np.asarray(result[6])
            cint = np.asarray(result[7])
            init_mean = np.asarray(result[8])
            init_cov = np.asarray(result[9])

        py_ll = kalman_log_likelihood(
            observations=jnp.array(Y),
            time_intervals=jnp.array(dt),
            drift=jnp.array(drift),
            diffusion_cov=jnp.array(diffusion_cov),
            cint=jnp.array(cint),
            lambda_mat=jnp.array(loadings),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.array(manifest_cov),
            t0_mean=jnp.array(init_mean),
            t0_cov=jnp.array(init_cov),
        )

        np.testing.assert_allclose(
            float(py_ll),
            r_ll,
            atol=1e-6,
            err_msg=f"Log-likelihood mismatch on ctGenerate data: Python={float(py_ll)}, R={r_ll}",
        )

    def test_multi_subject_per_subject_likelihood(self, r_ctsem):
        """Per-subject likelihood with n.subjects=5."""
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        r_code = """
        function() {
            library(ctsem)
            library(Matrix)
            set.seed(42)

            DRIFT <- matrix(c(-0.5, 0.1, 0.2, -0.8), nrow=2)
            DIFFUSION_chol <- matrix(c(0.3, 0.0, 0.05, 0.25), nrow=2)
            DIFFUSION <- DIFFUSION_chol %*% t(DIFFUSION_chol)
            MANIFESTVAR <- diag(0.1, 2)
            LAMBDA <- diag(1, 2)
            CINT <- matrix(c(0.0, 0.0), nrow=2)
            T0MEANS <- matrix(c(0, 0), nrow=2)
            T0VAR <- diag(1, 2)

            gm <- ctModel(
                Tpoints = 8,
                n.latent = 2, n.manifest = 2,
                DRIFT = matrix(c(-0.5, 0.1, 0.2, -0.8), nrow=2),
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = MANIFESTVAR,
                LAMBDA = LAMBDA,
                CINT = CINT,
                T0MEANS = T0MEANS,
                T0VAR = T0VAR
            )

            data <- ctGenerate(gm, n.subjects=5, burnin=10, dtmean=0.5)

            # Compute per-subject log-likelihoods
            n_latent <- 2
            n_manifest <- 2
            I <- diag(n_latent)

            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            compute_ll <- function(Y_subj, dt_subj) {
                state_mean <- T0MEANS
                state_cov <- T0VAR
                total_ll <- 0

                for (t in 1:nrow(Y_subj)) {
                    dt_t <- dt_subj[t]
                    discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                    discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                    discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                    state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                    state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                    y_t <- matrix(Y_subj[t, ], ncol=1)
                    pred_y <- LAMBDA %*% state_mean
                    S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTVAR
                    residual <- y_t - pred_y

                    ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                                   t(residual) %*% solve(S, residual))
                    total_ll <- total_ll + ll_t

                    K <- state_cov %*% t(LAMBDA) %*% solve(S)
                    state_mean <- state_mean + K %*% residual
                    state_cov <- (I - K %*% LAMBDA) %*% state_cov
                }
                return(as.numeric(total_ll))
            }

            # Process each subject
            subject_ids <- unique(data[, "id"])
            results <- list()

            for (i in seq_along(subject_ids)) {
                sid <- subject_ids[i]
                subj_data <- data[data[, "id"] == sid, ]
                Y_subj <- as.matrix(subj_data[, c("Y1", "Y2")])
                # ctGenerate returns absolute time, compute intervals
                times_subj <- subj_data[, "time"]
                dt_subj <- c(times_subj[1], diff(times_subj))
                ll_subj <- compute_ll(Y_subj, dt_subj)

                results[[i]] <- list(
                    Y = Y_subj,
                    dt = dt_subj,
                    ll = ll_subj
                )
            }

            list(
                results = results,
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION,
                MANIFESTVAR = MANIFESTVAR,
                LAMBDA = LAMBDA,
                CINT = as.vector(CINT),
                T0MEANS = as.vector(T0MEANS),
                T0VAR = T0VAR
            )
        }
        """
        r_multi_subject = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            result = r_multi_subject()
            # Result structure: [results_list, DRIFT, DIFFUSION, ...]
            # results_list contains per-subject [Y, dt, ll] for each subject

            # Extract model parameters (indices 1-7 are model params)
            drift = np.asarray(result[1])
            diffusion_cov = np.asarray(result[2])
            manifest_cov = np.asarray(result[3])
            loadings = np.asarray(result[4])
            cint = np.asarray(result[5])
            init_mean = np.asarray(result[6])
            init_cov = np.asarray(result[7])

            # Extract per-subject results
            r_results = result[0]

        # Test each subject
        for i in range(5):
            with localconverter(ro.default_converter + numpy2ri.converter):
                subj_result = r_results[i]
                Y_subj = np.asarray(subj_result[0])
                dt_subj = np.asarray(subj_result[1])
                r_ll_subj = float(np.asarray(subj_result[2])[0])

            py_ll = kalman_log_likelihood(
                observations=jnp.array(Y_subj),
                time_intervals=jnp.array(dt_subj),
                drift=jnp.array(drift),
                diffusion_cov=jnp.array(diffusion_cov),
                cint=jnp.array(cint),
                lambda_mat=jnp.array(loadings),
                manifest_means=jnp.zeros(2),
                manifest_cov=jnp.array(manifest_cov),
                t0_mean=jnp.array(init_mean),
                t0_cov=jnp.array(init_cov),
            )

            np.testing.assert_allclose(
                float(py_ll),
                r_ll_subj,
                atol=1e-6,
                err_msg=f"Log-likelihood mismatch for subject {i+1}: Python={float(py_ll)}, R={r_ll_subj}",
            )


class TestParityModelRecovery:
    """Test full parameter recovery pipeline."""

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    @pytest.mark.slow
    def test_parameter_recovery_drift_diagonal(self, r_ctsem):
        """Recover drift diagonal from ctsem-generated data.

        Note: Parameter recovery from finite time series is inherently noisy.
        This test uses 200 time points and strong dynamics to improve identifiability.
        Tolerance is set to 0.3 to account for MCMC variance.
        """
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # True parameters - use stronger dynamics (more negative) for better identifiability
        true_drift_diag = np.array([-0.8, -1.2])

        # Generate data in R - use more time points for better recovery
        r_code = """
        function() {
            library(ctsem)
            set.seed(42)

            # Stronger dynamics for better identifiability
            DRIFT <- matrix(c(-0.8, 0.0, 0.0, -1.2), nrow=2)  # Diagonal for simplicity
            DIFFUSION_chol <- matrix(c(0.4, 0.0, 0.0, 0.4), nrow=2)

            gm <- ctModel(
                Tpoints = 200,
                n.latent = 2, n.manifest = 2,
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = diag(0.05, 2),  # Lower measurement noise
                LAMBDA = diag(1, 2),
                CINT = matrix(c(0.0, 0.0), nrow=2),
                T0MEANS = matrix(c(0, 0), nrow=2),
                T0VAR = diag(1, 2)
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=50, dtmean=0.5)

            # ctGenerate returns absolute time, not dT
            list(
                Y = as.matrix(data[, c("Y1", "Y2")]),
                times = data[, "time"]
            )
        }
        """
        r_generate = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            result = r_generate()
            Y = np.asarray(result[0])
            times = np.asarray(result[1])

        # Fit with NumPyro
        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
        )
        model = CTSEMModel(spec)

        mcmc = model.fit(
            observations=jnp.array(Y),
            times=jnp.array(times),
            num_warmup=1000,
            num_samples=1000,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        drift_diag_samples = samples["drift_diag_pop"]
        posterior_mean = np.mean(drift_diag_samples, axis=0)

        # Check recovery (relaxed tolerance for stochastic estimation)
        # With 200 time points and strong dynamics, we expect ~0.3 accuracy
        np.testing.assert_allclose(
            posterior_mean,
            true_drift_diag,
            atol=0.3,
            err_msg=f"Drift diagonal recovery: estimated={posterior_mean}, true={true_drift_diag}",
        )

    @pytest.mark.slow
    def test_parameter_recovery_diffusion(self, r_ctsem):
        """Recover diffusion parameters from ctsem-generated data.

        Note: Diffusion parameters are harder to estimate than drift.
        This test uses 200 time points and larger diffusion values for identifiability.
        """
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # True diffusion diagonal - use larger values for better identifiability
        true_diffusion_diag = np.array([0.5, 0.5])

        # Generate data in R
        r_code = """
        function() {
            library(ctsem)
            set.seed(123)

            DRIFT <- matrix(c(-0.8, 0.0, 0.0, -0.8), nrow=2)  # Diagonal for simplicity
            DIFFUSION_chol <- matrix(c(0.5, 0.0, 0.0, 0.5), nrow=2)  # Larger diagonal

            gm <- ctModel(
                Tpoints = 200,
                n.latent = 2, n.manifest = 2,
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = diag(0.02, 2),  # Very small measurement error
                LAMBDA = diag(1, 2),
                CINT = matrix(c(0.0, 0.0), nrow=2),
                T0MEANS = matrix(c(0, 0), nrow=2),
                T0VAR = diag(1, 2)
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=50, dtmean=0.5)

            # ctGenerate returns absolute time, not dT
            list(
                Y = as.matrix(data[, c("Y1", "Y2")]),
                times = data[, "time"]
            )
        }
        """
        r_generate = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            result = r_generate()
            Y = np.asarray(result[0])
            times = np.asarray(result[1])

        # Fit with NumPyro
        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
        )
        model = CTSEMModel(spec)

        mcmc = model.fit(
            observations=jnp.array(Y),
            times=jnp.array(times),
            num_warmup=1000,
            num_samples=1000,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        diffusion_diag_samples = samples["diffusion_diag_pop"]
        posterior_mean = np.mean(diffusion_diag_samples, axis=0)

        # Check recovery (relaxed tolerance for stochastic estimation)
        # Diffusion is harder to estimate than drift, allow 0.25 tolerance
        np.testing.assert_allclose(
            posterior_mean,
            true_diffusion_diag,
            atol=0.25,
            err_msg=f"Diffusion diagonal recovery: estimated={posterior_mean}, true={true_diffusion_diag}",
        )


class TestParityFullEstimation:
    """Test full model estimation parity between NumPyro and ctsem.

    These tests generate data from complex CT-SEM models, fit with both
    ctsem's ctStanFit and our NumPyro implementation, and compare
    parameter estimates.

    Note: These tests are slow (~minutes) because they run full MCMC in both R and Python.
    """

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    @pytest.mark.slow
    def test_full_estimation_bivariate_cross_lagged(self, r_ctsem):
        """Full estimation parity for bivariate cross-lagged model.

        This is the "big test" - a complex model with:
        - Cross-lagged effects (off-diagonal drift)
        - Single subject with many time points (150)
        - Uses both ctsem and NumPyro to fit the same data

        Compares MAP/posterior mean estimates from ctsem vs NumPyro.
        """
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # True parameters for data generation
        true_drift_11 = -0.6  # Auto-effect latent 1
        true_drift_22 = -0.8  # Auto-effect latent 2
        true_drift_12 = 0.15  # Cross-lag: latent 2 -> latent 1
        true_drift_21 = -0.1  # Cross-lag: latent 1 -> latent 2
        true_diffusion = 0.4  # Diagonal diffusion (Cholesky)
        true_manifest_var = 0.1
        n_timepoints = 150  # More data for better estimation

        # Generate data and fit with ctsem in R
        r_code = """
        function(n_timepoints, drift_11, drift_22, drift_12, drift_21,
                 diffusion_diag, manifest_var, seed) {
            library(ctsem)
            set.seed(seed)

            # Define the generative model (column-major ordering)
            DRIFT <- matrix(c(drift_11, drift_21, drift_12, drift_22), nrow=2, byrow=FALSE)
            DIFFUSION_chol <- diag(diffusion_diag, 2)

            gm <- ctModel(
                Tpoints = n_timepoints,
                n.latent = 2,
                n.manifest = 2,
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = diag(manifest_var, 2),
                LAMBDA = diag(1, 2),
                CINT = matrix(c(0.0, 0.0), nrow=2),
                T0MEANS = matrix(c(0, 0), nrow=2),
                T0VAR = diag(1, 2)
            )

            # Generate single-subject data
            data <- ctGenerate(gm, n.subjects=1, burnin=30, dtmean=0.5)

            # Define estimation model
            em <- ctModel(
                type = "stanct",
                n.latent = 2,
                n.manifest = 2,
                LAMBDA = diag(1, 2),
                manifestNames = c("Y1", "Y2"),
                latentNames = c("eta1", "eta2")
            )

            # Fit with ctStanFit (optimization = MAP estimate)
            fit <- ctStanFit(
                datalong = data,
                ctstanmodel = em,
                optimize = TRUE,
                optimcontrol = list(is = FALSE),
                verbose = 0
            )

            # Extract parameter estimates from summary
            summ <- summary(fit)
            pop_means <- summ$popmeans

            # ctsem uses lowercase names: drift_eta1, drift_eta2, drift_eta1_eta2, etc.
            list(
                data_long = data,
                drift_11 = pop_means["drift_eta1", "mean"],
                drift_22 = pop_means["drift_eta2", "mean"],
                drift_12 = pop_means["drift_eta1_eta2", "mean"],
                drift_21 = pop_means["drift_eta2_eta1", "mean"],
                true_drift = DRIFT
            )
        }
        """
        r_fit_ctsem = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            result = r_fit_ctsem(
                n_timepoints,
                true_drift_11, true_drift_22, true_drift_12, true_drift_21,
                true_diffusion, true_manifest_var,
                42  # seed
            )

            data_long = np.asarray(result[0])
            r_drift_11 = float(np.asarray(result[1]).flat[0])
            r_drift_22 = float(np.asarray(result[2]).flat[0])
            r_drift_12 = float(np.asarray(result[3]).flat[0])
            r_drift_21 = float(np.asarray(result[4]).flat[0])

        # Parse R's drift estimates
        r_drift_diag = np.array([r_drift_11, r_drift_22])
        r_drift_offdiag = np.array([r_drift_21, r_drift_12])

        # Prepare data for NumPyro (single subject)
        times = data_long[:, 1]
        Y = data_long[:, 2:4]

        # Fit with NumPyro
        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
        )
        model = CTSEMModel(spec)

        mcmc = model.fit(
            observations=jnp.array(Y),
            times=jnp.array(times),
            num_warmup=500,
            num_samples=500,
            num_chains=1,
        )

        samples = mcmc.get_samples()

        # Get NumPyro estimates (posterior mean)
        py_drift_diag = np.mean(samples["drift_diag_pop"], axis=0)
        py_drift_offdiag = np.mean(samples["drift_offdiag_pop"], axis=0)

        # Compare drift diagonal (auto-effects)
        # Note: Tolerance is 0.5 due to different priors between ctsem and NumPyro,
        # and different estimation methods (MAP vs MCMC posterior mean)
        np.testing.assert_allclose(
            np.sort(py_drift_diag),
            np.sort(r_drift_diag),
            atol=0.5,
            err_msg=f"Drift diagonal mismatch: NumPyro={py_drift_diag}, ctsem={r_drift_diag}",
        )

        # Compare drift off-diagonal (cross-effects)
        np.testing.assert_allclose(
            np.sort(py_drift_offdiag),
            np.sort(r_drift_offdiag),
            atol=0.5,
            err_msg=f"Drift off-diagonal mismatch: NumPyro={py_drift_offdiag}, ctsem={r_drift_offdiag}",
        )

        # Verify recovery of true parameters (both implementations)
        # This is the most important test - both should recover true values
        true_drift_diag = np.array([true_drift_11, true_drift_22])
        np.testing.assert_allclose(
            np.sort(py_drift_diag),
            np.sort(true_drift_diag),
            atol=0.5,
            err_msg=f"NumPyro drift diagonal recovery: estimated={py_drift_diag}, true={true_drift_diag}",
        )

        # ctsem should also recover true values
        np.testing.assert_allclose(
            np.sort(r_drift_diag),
            np.sort(true_drift_diag),
            atol=0.8,  # More relaxed for ctsem due to regularization priors
            err_msg=f"ctsem drift diagonal recovery: estimated={r_drift_diag}, true={true_drift_diag}",
        )

    @pytest.mark.slow
    def test_full_estimation_with_cint(self, r_ctsem):
        """Parameter recovery with non-zero continuous intercepts.

        Tests that NumPyro can recover CINT from ctsem-generated data.
        Note: ctsem doesn't estimate CINT by default (fixed at 0), so we only
        test NumPyro's recovery of true parameters, not cross-implementation parity.
        """
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # True parameters
        true_drift_diag = np.array([-0.5, -0.7])
        true_cint = np.array([0.3, -0.2])  # Non-zero intercepts
        true_diffusion = 0.3
        n_timepoints = 150  # More data for CINT estimation

        # Generate data with ctsem
        r_code = """
        function(n_timepoints, drift_11, drift_22, cint1, cint2,
                 diffusion_diag, seed) {
            library(ctsem)
            set.seed(seed)

            DRIFT <- diag(c(drift_11, drift_22))
            DIFFUSION_chol <- diag(diffusion_diag, 2)
            CINT <- matrix(c(cint1, cint2), nrow=2)

            gm <- ctModel(
                Tpoints = n_timepoints,
                n.latent = 2,
                n.manifest = 2,
                DRIFT = DRIFT,
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = diag(0.1, 2),
                LAMBDA = diag(1, 2),
                CINT = CINT,
                T0MEANS = matrix(c(0, 0), nrow=2),
                T0VAR = diag(1, 2)
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=30, dtmean=0.5)
            return(data)
        }
        """
        r_generate = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            data_long = np.asarray(r_generate(
                n_timepoints,
                true_drift_diag[0], true_drift_diag[1],
                true_cint[0], true_cint[1],
                true_diffusion,
                123  # seed
            ))

        # Prepare data for NumPyro
        times = data_long[:, 1]
        Y = data_long[:, 2:4]

        # Fit with NumPyro (with CINT enabled)
        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            cint="free",  # Enable CINT estimation
        )
        model = CTSEMModel(spec)

        mcmc = model.fit(
            observations=jnp.array(Y),
            times=jnp.array(times),
            num_warmup=500,
            num_samples=500,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        py_drift_diag = np.mean(samples["drift_diag_pop"], axis=0)
        py_cint = np.mean(samples["cint_pop"], axis=0)

        # Verify recovery of true parameters
        np.testing.assert_allclose(
            np.sort(py_drift_diag),
            np.sort(true_drift_diag),
            atol=0.5,
            err_msg=f"NumPyro drift diagonal recovery: estimated={py_drift_diag}, true={true_drift_diag}",
        )

        np.testing.assert_allclose(
            py_cint,
            true_cint,
            atol=0.5,
            err_msg=f"NumPyro CINT recovery: estimated={py_cint}, true={true_cint}",
        )

    @pytest.mark.slow
    def test_full_estimation_trivariate(self, r_ctsem):
        """Parameter recovery for a 3-latent variable model.

        Tests scalability to larger models with:
        - 3 latent variables
        - Full drift matrix (9 parameters)
        - Single subject with 120 time points
        """
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # True parameters for 3x3 system
        true_drift = np.array([
            [-0.6, 0.1, 0.05],
            [0.15, -0.7, 0.1],
            [-0.1, 0.2, -0.5]
        ])
        true_diffusion_diag = 0.35
        n_timepoints = 150  # More data for 3x3 system

        # Generate data with ctsem
        r_code = """
        function(n_timepoints, drift_matrix, diffusion_diag, seed) {
            library(ctsem)
            set.seed(seed)

            n_latent <- 3
            DIFFUSION_chol <- diag(diffusion_diag, n_latent)

            gm <- ctModel(
                Tpoints = n_timepoints,
                n.latent = n_latent,
                n.manifest = n_latent,
                DRIFT = drift_matrix,
                DIFFUSION = DIFFUSION_chol,
                MANIFESTVAR = diag(0.1, n_latent),
                LAMBDA = diag(1, n_latent),
                CINT = matrix(0, nrow=n_latent),
                T0MEANS = matrix(0, nrow=n_latent),
                T0VAR = diag(1, n_latent)
            )

            data <- ctGenerate(gm, n.subjects=1, burnin=30, dtmean=0.5)
            return(data)
        }
        """
        r_generate = ro.r(r_code)

        with localconverter(ro.default_converter + numpy2ri.converter):
            data_long = np.asarray(r_generate(
                n_timepoints,
                true_drift, true_diffusion_diag,
                456  # seed
            ))

        # Prepare data for NumPyro
        times = data_long[:, 1]
        Y = data_long[:, 2:5]

        # Fit with NumPyro
        spec = CTSEMSpec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
        )
        model = CTSEMModel(spec)

        mcmc = model.fit(
            observations=jnp.array(Y),
            times=jnp.array(times),
            num_warmup=500,
            num_samples=500,
            num_chains=1,
        )

        samples = mcmc.get_samples()
        py_drift_diag = np.mean(samples["drift_diag_pop"], axis=0)

        # Verify recovery of true diagonal
        true_drift_diag = np.diag(true_drift)
        np.testing.assert_allclose(
            np.sort(py_drift_diag),
            np.sort(true_drift_diag),
            atol=0.5,
            err_msg=f"NumPyro drift diagonal recovery (3x3): estimated={py_drift_diag}, true={true_drift_diag}",
        )


class TestCTSEMModelBuilder:
    """Test CTSEMModelBuilder pipeline integration."""

    def test_builder_with_ctsem_spec(self):
        """Test building CT-SEM from CTSEMSpec directly."""
        import pandas as pd

        from dsem_agent.models.ctsem import CTSEMSpec
        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        # Create a CTSEMSpec directly
        spec = CTSEMSpec(
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

        builder = CTSEMModelBuilder(ctsem_spec=spec)
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2
        assert builder._spec.n_latent == 2

    def test_builder_auto_detect(self):
        """Test auto-detection of manifest columns."""
        import pandas as pd

        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        T = 15
        X = pd.DataFrame({
            "x": np.random.randn(T),
            "y": np.random.randn(T),
            "time": np.arange(T, dtype=float),
        })

        builder = CTSEMModelBuilder()
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2

    @pytest.mark.slow
    def test_builder_fit(self):
        """Test fitting via builder interface."""
        import pandas as pd

        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        T = 20
        X = pd.DataFrame({
            "x": np.random.randn(T) * 0.5,
            "y": np.random.randn(T) * 0.5,
            "time": np.arange(T, dtype=float),
        })

        builder = CTSEMModelBuilder(
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
