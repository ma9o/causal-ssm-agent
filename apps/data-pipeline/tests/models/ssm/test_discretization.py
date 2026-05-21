"""Tests for CT→DT discretization module.

Covers: Lyapunov solver, asymptotic diffusion, discrete diffusion,
        discrete intercept, system discretization, and batched discretization.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.discretization.exact import (
    _kron_lyapunov_solve,
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_cint_exact,
    compute_discrete_diffusion,
    compute_discrete_diffusion_van_loan,
    discretize_linear_system_exact,
    discretize_linear_system_exact_batched,
    discretize_system,
    discretize_system_batched,
    discretize_system_with_inputs_batched,
    solve_lyapunov,
)

# =============================================================================
# Kronecker Lyapunov solver
# =============================================================================


class TestKronLyapunovSolve:
    def test_1d_system(self):
        """For scalar: a*x + x*a = -q → x = -q / (2a)."""
        A = jnp.array([[-2.0]])
        Q = jnp.array([[4.0]])
        X = _kron_lyapunov_solve(A, Q)
        assert jnp.isclose(X[0, 0], 1.0)  # -4 / (2*(-2)) = 1

    def test_2d_diagonal_system(self):
        """Diagonal system has known solution."""
        A = jnp.diag(jnp.array([-1.0, -2.0]))
        Q = jnp.diag(jnp.array([2.0, 8.0]))
        X = _kron_lyapunov_solve(A, Q)
        # x_ii = -q_ii / (2 * a_ii)
        assert jnp.isclose(X[0, 0], 1.0, atol=1e-5)
        assert jnp.isclose(X[1, 1], 2.0, atol=1e-5)


# =============================================================================
# solve_lyapunov (Schur-based, custom VJP)
# =============================================================================


class TestSolveLyapunov:
    def test_satisfies_equation(self):
        """Verify A*X + X*A' = -Q for the Schur solver."""
        A = jnp.array([[-2.0, 0.5], [0.0, -3.0]])
        Q = jnp.array([[1.0, 0.2], [0.2, 2.0]])
        X = solve_lyapunov(A, Q)
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_symmetry(self):
        """Solution should be symmetric for symmetric Q."""
        A = jnp.array([[-1.0, 0.3], [-0.3, -2.0]])
        Q = jnp.eye(2)
        X = solve_lyapunov(A, Q)
        assert jnp.allclose(X, X.T, atol=1e-6)

    def test_is_differentiable(self):
        """VJP should produce finite gradients."""

        def loss(A_flat):
            A = A_flat.reshape(2, 2)
            Q = jnp.eye(2)
            X = solve_lyapunov(A, Q)
            return jnp.sum(X**2)

        A = jnp.array([-2.0, 0.5, 0.0, -3.0])
        grads = jax.grad(loss)(A)
        assert jnp.all(jnp.isfinite(grads))


# =============================================================================
# Asymptotic diffusion
# =============================================================================


class TestAsymptoticDiffusion:
    def test_ou_process(self):
        """OU process: A=-theta*I, G=sigma*I → Q_inf = sigma^2/(2*theta) * I."""
        theta = 2.0
        sigma = 1.0
        A = jnp.array([[-theta]])
        Q = jnp.array([[sigma**2]])
        Q_inf = compute_asymptotic_diffusion(A, Q)
        expected = sigma**2 / (2 * theta)
        assert jnp.isclose(Q_inf[0, 0], expected, atol=1e-5)


# =============================================================================
# Discrete diffusion
# =============================================================================


class TestDiscreteDiffusion:
    def test_positive_semidefinite(self):
        """Discrete diffusion should be PSD."""
        A = jnp.array([[-1.0, 0.2], [0.0, -2.0]])
        Q = jnp.eye(2) * 0.5
        dt = 0.1
        Q_dt = compute_discrete_diffusion(A, Q, dt)
        eigvals = jnp.linalg.eigvalsh(Q_dt)
        assert jnp.all(eigvals >= -1e-6)

    def test_symmetry(self):
        """Discrete diffusion should be symmetric."""
        A = jnp.array([[-1.0, 0.2], [0.0, -2.0]])
        Q = jnp.eye(2)
        dt = 0.5
        Q_dt = compute_discrete_diffusion(A, Q, dt)
        assert jnp.allclose(Q_dt, Q_dt.T, atol=1e-10)

    def test_grows_with_dt(self):
        """Larger dt should give more process noise."""
        A = jnp.array([[-1.0]])
        Q = jnp.array([[1.0]])
        Q_small = compute_discrete_diffusion(A, Q, dt=0.1)
        Q_large = compute_discrete_diffusion(A, Q, dt=1.0)
        assert Q_large[0, 0] > Q_small[0, 0]

    def test_reuses_precomputed_dynamics(self):
        """Passing discrete_dynamics should preserve results."""
        A = jnp.array([[-1.0, 0.2], [0.0, -2.0]])
        Q = jnp.eye(2)
        dt = 0.5
        Ad = jla.expm(A * dt)
        Q_dt_1 = compute_discrete_diffusion(A, Q, dt)
        Q_dt_2 = compute_discrete_diffusion(A, Q, dt, discrete_dynamics=Ad)
        assert jnp.allclose(Q_dt_1, Q_dt_2, atol=1e-10)

    def test_matches_lyapunov_reference(self):
        """Van Loan discretization should match the stationary-covariance identity."""
        A = jnp.array([[-1.0, 0.3], [0.2, -1.5]])
        Q = jnp.array([[1.0, 0.4], [0.4, 0.8]])
        dt = 0.75
        Ad = jla.expm(A * dt)
        Q_inf = solve_lyapunov(A, Q)
        expected = Q_inf - Ad @ Q_inf @ Ad.T
        actual = compute_discrete_diffusion(A, Q, dt)
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_van_loan_matches_stationary_identity_for_stable_system(self):
        """The exact Van Loan path should agree with the stable-system shortcut."""
        A = jnp.array([[-0.8, 0.2], [0.1, -1.4]])
        Q = jnp.array([[0.7, 0.1], [0.1, 0.5]])
        dt = 0.4
        exact = compute_discrete_diffusion_van_loan(A, Q, dt)
        stable = compute_discrete_diffusion(A, Q, dt)
        assert jnp.allclose(exact, stable, atol=1e-6)


# =============================================================================
# Discrete intercept
# =============================================================================


class TestDiscreteCint:
    def test_zero_dynamics_limit(self):
        """For nearly-zero dt, c_dt ≈ c * dt."""
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        c = jnp.array([1.0, 2.0])
        dt = 1e-4
        c_dt = compute_discrete_cint(A, c, dt)
        assert jnp.allclose(c_dt, c * dt, atol=1e-3)

    def test_shape(self):
        """Output shape should match input intercept."""
        A = jnp.array([[-1.0, 0.3], [0.0, -2.0]])
        c = jnp.array([1.0, 2.0])
        c_dt = compute_discrete_cint(A, c, dt=0.5)
        assert c_dt.shape == c.shape

    def test_exact_handles_singular_dynamics(self):
        """The exact block-exponential path should work for singular dynamics."""
        A = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        c = jnp.array([2.0, 0.0])
        dt = 0.5
        c_dt = compute_discrete_cint_exact(A, c, dt)
        expected = jnp.array([1.0, 0.25])
        assert jnp.allclose(c_dt, expected, atol=1e-6)


# =============================================================================
# Full system discretization
# =============================================================================


class TestDiscretizeSystem:
    def test_output_shapes(self):
        """All outputs should have correct shapes."""
        n = 3
        A = -jnp.eye(n)
        Q = jnp.eye(n) * 0.1
        c = jnp.ones(n)
        Ad, Qd, cd = discretize_system(A, Q, c, dt=0.5)
        assert Ad.shape == (n, n)
        assert Qd.shape == (n, n)
        assert cd is not None
        assert cd.shape == c.shape

    def test_no_cint_returns_none(self):
        """When cint=None, cd should be None."""
        n = 2
        A = -jnp.eye(n)
        Q = jnp.eye(n)
        Ad, Qd, cd = discretize_system(A, Q, None, dt=0.5)
        assert cd is None
        assert Ad.shape == (n, n)
        assert Qd.shape == (n, n)

    def test_dynamics_is_matrix_exponential(self):
        """Discrete dynamics should be exp(A*dt)."""
        A = jnp.array([[-1.0, 0.5], [0.0, -2.0]])
        Q = jnp.eye(2)
        dt = 0.5
        Ad, _, _ = discretize_system(A, Q, None, dt)
        expected = jla.expm(A * dt)
        assert jnp.allclose(Ad, expected, atol=1e-6)


# =============================================================================
# Batched discretization
# =============================================================================


class TestDiscretizeSystemBatched:
    def test_output_shapes(self):
        """Batched output should have (T, n, n) shapes."""
        n = 2
        T = 5
        A = -jnp.eye(n)
        Q = jnp.eye(n) * 0.1
        c = jnp.ones(n)
        dts = jnp.ones(T) * 0.5
        Ad, Qd, cd = discretize_system_batched(A, Q, c, dts)
        assert Ad.shape == (T, n, n)
        assert Qd.shape == (T, n, n)
        assert cd is not None
        assert cd.shape == (T, n)

    def test_known_inputs_add_discrete_forcing_offsets(self):
        """Known input forcing is integrated into the discrete intercept."""
        A = jnp.array([[-1.0]])
        Q = jnp.array([[0.1]])
        B = jnp.array([[2.0]])
        u = jnp.array([[1.0], [3.0]])
        dts = jnp.array([0.5, 2.0])

        _Ad, _Qd, cd = discretize_system_with_inputs_batched(A, Q, jnp.array([0.0]), B, u, dts)

        assert cd is not None
        assert cd.shape == (2, 1)
        expected = jnp.array(
            [
                2.0 * (1.0 - jnp.exp(-0.5)),
                6.0 * (1.0 - jnp.exp(-2.0)),
            ]
        )
        assert jnp.allclose(cd[:, 0], expected, atol=1e-6)

    def test_no_cint_batched(self):
        """Batched without cint should return None."""
        n = 2
        T = 3
        A = -jnp.eye(n)
        Q = jnp.eye(n)
        dts = jnp.ones(T) * 0.5
        Ad, _Qd, cd = discretize_system_batched(A, Q, None, dts)
        assert cd is None
        assert Ad.shape == (T, n, n)

    def test_varying_dt(self):
        """Different dt values should produce different discrete matrices."""
        n = 2
        A = -jnp.eye(n)
        Q = jnp.eye(n)
        dts = jnp.array([0.1, 0.5, 1.0])
        Ad, _Qd, _ = discretize_system_batched(A, Q, None, dts)
        # Each Ad should be different
        assert not jnp.allclose(Ad[0], Ad[1])
        assert not jnp.allclose(Ad[1], Ad[2])

    def test_consistent_with_single(self):
        """Batched result should match per-element discretize_system."""
        n = 2
        A = -jnp.eye(n) * 1.5
        Q = jnp.eye(n) * 0.3
        c = jnp.array([0.5, -0.5])
        dts = jnp.array([0.1, 0.5])
        Ad_b, Qd_b, _cd_b = discretize_system_batched(A, Q, c, dts)

        Ad_0, Qd_0, _cd_0 = discretize_system(A, Q, c, 0.1)
        Ad_1, Qd_1, _cd_1 = discretize_system(A, Q, c, 0.5)

        assert jnp.allclose(Ad_b[0], Ad_0, atol=1e-5)
        assert jnp.allclose(Ad_b[1], Ad_1, atol=1e-5)
        assert jnp.allclose(Qd_b[0], Qd_0, atol=1e-5)
        assert jnp.allclose(Qd_b[1], Qd_1, atol=1e-5)

    def test_constant_dt_matches_broadcast_single(self):
        """Uniform dt should match a single discretization broadcast over time."""
        A = jnp.array([[-1.2, 0.1], [0.0, -0.8]])
        Q = jnp.array([[0.4, 0.05], [0.05, 0.3]])
        c = jnp.array([0.25, -0.1])
        dts = jnp.full((4,), 0.25)

        Ad_b, Qd_b, cd_b = discretize_system_batched(A, Q, c, dts)
        Ad_single, Qd_single, cd_single = discretize_system(A, Q, c, 0.25)
        assert cd_b is not None
        assert cd_single is not None

        assert jnp.allclose(Ad_b, jnp.broadcast_to(Ad_single, Ad_b.shape), atol=1e-5)
        assert jnp.allclose(Qd_b, jnp.broadcast_to(Qd_single, Qd_b.shape), atol=1e-5)
        assert jnp.allclose(cd_b, jnp.broadcast_to(cd_single, cd_b.shape), atol=1e-5)

    def test_exact_general_batched_matches_single(self):
        """Exact batched discretization should match the single-step helper."""
        A = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        Q = jnp.array([[0.2, 0.0], [0.0, 0.0]])
        c = jnp.array([0.0, 1.0])
        dts = jnp.array([0.25, 0.75])

        Ad_b, Qd_b, cd_b = discretize_linear_system_exact_batched(A, Q, c, dts)
        Ad_0, Qd_0, cd_0 = discretize_linear_system_exact(A, Q, c, dts[0])
        Ad_1, Qd_1, cd_1 = discretize_linear_system_exact(A, Q, c, dts[1])

        assert cd_b is not None
        assert jnp.allclose(Ad_b[0], Ad_0, atol=1e-6)
        assert jnp.allclose(Qd_b[0], Qd_0, atol=1e-6)
        assert jnp.allclose(cd_b[0], cd_0, atol=1e-6)
        assert jnp.allclose(Ad_b[1], Ad_1, atol=1e-6)
        assert jnp.allclose(Qd_b[1], Qd_1, atol=1e-6)
        assert jnp.allclose(cd_b[1], cd_1, atol=1e-6)
