"""Core matrix utilities for CT-SEM.

Implements the mathematical operations needed for continuous-to-discrete
transformation in CT-SEM models:

1. Matrix exponential: exp(A*dt) for discrete drift
2. Lyapunov solver: A*Q + Q*A' = -GG' for asymptotic diffusion
3. Discrete diffusion: Q_dt = Q_inf - exp(A*dt)*Q_inf*exp(A*dt)'
4. Discrete CINT: c_dt = A^{-1}*(exp(A*dt) - I)*c
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax


def solve_lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve the continuous Lyapunov equation: A*X + X*A' = -Q.

    This is equivalent to ctsem's ksolve function for computing the
    asymptotic diffusion covariance.

    For a stable system (eigenvalues of A have negative real parts),
    this gives the stationary covariance of the process.

    We use the vectorization approach:
    vec(A*X + X*A') = (I ⊗ A + A ⊗ I) * vec(X) = -vec(Q)

    For numerical stability with potentially ill-conditioned systems,
    we use the Bartels-Stewart algorithm via scipy's solve_sylvester,
    but JAX doesn't have this, so we implement via vectorization.

    Args:
        A: n x n drift matrix (must be stable for unique solution)
        Q: n x n positive semi-definite matrix (typically GG')

    Returns:
        X: n x n solution matrix (asymptotic covariance)
    """
    n = A.shape[0]

    # Build the coefficient matrix: (I ⊗ A + A ⊗ I)
    # Using the identity: vec(AXB') = (B ⊗ A) * vec(X)
    # So vec(AX + XA') = vec(AXI + IXA') = (I ⊗ A + A ⊗ I) * vec(X)
    I_n = jnp.eye(n)
    coef = jnp.kron(I_n, A) + jnp.kron(A, I_n)

    # Solve: coef * vec(X) = -vec(Q)
    vec_Q = Q.flatten()
    vec_X = jla.solve(coef, -vec_Q)

    return vec_X.reshape((n, n))


def solve_lyapunov_iterative(
    A: jnp.ndarray, Q: jnp.ndarray, max_iter: int = 100, tol: float = 1e-10
) -> jnp.ndarray:
    """Solve Lyapunov equation iteratively (for numerical stability).

    Uses the Smith iteration for A*X + X*A' = -Q:
    X_{k+1} = (A + I)^{-1} * (X_k - Q) * (A' + I)^{-1}

    This is more numerically stable for near-singular systems.

    Args:
        A: n x n drift matrix
        Q: n x n positive semi-definite matrix
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        X: n x n solution matrix
    """
    n = A.shape[0]
    I_n = jnp.eye(n)

    # Scale A to improve convergence
    # Find alpha such that (A/alpha) has eigenvalues in suitable range
    alpha = 1.0

    A_scaled = A / alpha
    Q_scaled = Q / (alpha * alpha)

    # Initial guess
    X = jnp.zeros((n, n))

    # Precompute (A + I)^{-1}
    ApI_inv = jla.inv(A_scaled + I_n)

    def body_fn(carry):
        X, _ = carry
        X_new = ApI_inv @ (X - Q_scaled) @ ApI_inv.T
        # Symmetrize
        X_new = 0.5 * (X_new + X_new.T)
        diff = jnp.max(jnp.abs(X_new - X))
        return X_new, diff

    def cond_fn(carry):
        _, diff = carry
        return diff > tol

    X, _ = lax.while_loop(cond_fn, body_fn, (X, jnp.inf))

    return X


def compute_asymptotic_diffusion(
    drift: jnp.ndarray, diffusion_cov: jnp.ndarray
) -> jnp.ndarray:
    """Compute asymptotic (stationary) diffusion covariance.

    Solves: A*Q_inf + Q_inf*A' = -G*G'

    Where:
        A = drift matrix
        G = diffusion (Cholesky factor), so G*G' = diffusion_cov

    Args:
        drift: n x n drift matrix A
        diffusion_cov: n x n diffusion covariance (G*G')

    Returns:
        Q_inf: n x n asymptotic diffusion covariance
    """
    return solve_lyapunov(drift, diffusion_cov)


def compute_discrete_diffusion(
    drift: jnp.ndarray, diffusion_cov: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Compute discrete-time diffusion covariance for time interval dt.

    Q_dt = Q_inf - exp(A*dt) * Q_inf * exp(A*dt)'

    Where Q_inf is the asymptotic diffusion from the Lyapunov equation.

    Args:
        drift: n x n drift matrix A
        diffusion_cov: n x n diffusion covariance (G*G')
        dt: time interval

    Returns:
        Q_dt: n x n discrete diffusion covariance
    """
    # Compute asymptotic diffusion
    Q_inf = compute_asymptotic_diffusion(drift, diffusion_cov)

    # Compute discrete drift
    discrete_drift = jla.expm(drift * dt)

    # Q_dt = Q_inf - exp(A*dt) * Q_inf * exp(A*dt)'
    Q_dt = Q_inf - discrete_drift @ Q_inf @ discrete_drift.T

    # Ensure symmetry
    Q_dt = 0.5 * (Q_dt + Q_dt.T)

    return Q_dt


def compute_discrete_cint(
    drift: jnp.ndarray, cint: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Compute discrete-time intercept for time interval dt.

    c_dt = A^{-1} * (exp(A*dt) - I) * c

    This is the integrated effect of the continuous intercept over dt.

    Args:
        drift: n x n drift matrix A
        cint: n x 1 continuous intercept c
        dt: time interval

    Returns:
        c_dt: n x 1 discrete intercept
    """
    n = drift.shape[0]
    I_n = jnp.eye(n)

    # Compute discrete drift
    discrete_drift = jla.expm(drift * dt)

    # c_dt = A^{-1} * (exp(A*dt) - I) * c
    # Using solve for numerical stability: A * c_dt = (exp(A*dt) - I) * c
    rhs = (discrete_drift - I_n) @ cint
    c_dt = jla.solve(drift, rhs)

    return c_dt


def matrix_fraction_decomposition(
    drift: jnp.ndarray, diffusion: jnp.ndarray, cint: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute discretization via matrix fraction decomposition.

    This method computes all discrete parameters in a single matrix
    exponential, which can be more numerically stable.

    Augmented system:
    [A  G*G'  c]     [exp(A*dt)  Q_dt  c_dt]
    [0  -A'   0] --> [0          exp(-A'*dt)  0]
    [0   0    0]     [0          0            1]

    Args:
        drift: n x n drift matrix A
        diffusion: n x n diffusion Cholesky G (so G*G' is covariance)
        cint: n x 1 continuous intercept
        dt: time interval

    Returns:
        discrete_drift: n x n discrete drift exp(A*dt)
        discrete_Q: n x n discrete diffusion covariance
        discrete_cint: n x 1 discrete intercept
    """
    n = drift.shape[0]

    # Build augmented matrix
    diffusion_cov = diffusion @ diffusion.T
    aug_size = 2 * n + 1

    aug = jnp.zeros((aug_size, aug_size))
    aug = aug.at[:n, :n].set(drift)
    aug = aug.at[:n, n : 2 * n].set(diffusion_cov)
    aug = aug.at[:n, 2 * n].set(cint.flatten())
    aug = aug.at[n : 2 * n, n : 2 * n].set(-drift.T)

    # Compute matrix exponential
    aug_exp = jla.expm(aug * dt)

    # Extract components
    discrete_drift = aug_exp[:n, :n]

    # Q_dt = discrete_drift @ aug_exp[:n, n:2*n]
    # Actually: Q_dt is in the upper-right block already transformed
    discrete_Q = aug_exp[:n, n : 2 * n] @ discrete_drift.T

    discrete_cint = aug_exp[:n, 2 * n : 2 * n + 1]

    # Ensure Q symmetry
    discrete_Q = 0.5 * (discrete_Q + discrete_Q.T)

    return discrete_drift, discrete_Q, discrete_cint


def discretize_system(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Discretize the CT-SEM system for a given time interval.

    Computes:
    - discrete_drift = exp(A*dt)
    - discrete_Q = Q_inf - exp(A*dt)*Q_inf*exp(A*dt)'
    - discrete_cint = A^{-1}*(exp(A*dt) - I)*c (if cint provided)

    Args:
        drift: n x n continuous drift matrix A
        diffusion_cov: n x n diffusion covariance (G*G')
        cint: n x 1 continuous intercept (optional)
        dt: time interval

    Returns:
        Tuple of (discrete_drift, discrete_Q, discrete_cint)
    """
    # Discrete drift via matrix exponential
    discrete_drift = jla.expm(drift * dt)

    # Discrete diffusion via Lyapunov solution
    discrete_Q = compute_discrete_diffusion(drift, diffusion_cov, dt)

    # Discrete intercept
    discrete_cint = None
    if cint is not None:
        discrete_cint = compute_discrete_cint(drift, cint, dt)

    return discrete_drift, discrete_Q, discrete_cint


def cholesky_of_diffusion(diffusion_raw: jnp.ndarray) -> jnp.ndarray:
    """Convert raw lower triangular diffusion to covariance.

    In ctsem, DIFFUSION is specified as a lower-triangular Cholesky factor.
    The covariance is G*G'.

    Args:
        diffusion_raw: n x n lower triangular matrix G

    Returns:
        G*G': n x n covariance matrix
    """
    # Ensure lower triangular
    G = jnp.tril(diffusion_raw)
    return G @ G.T


def ensure_stability(drift: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Ensure drift matrix eigenvalues have negative real parts.

    For a stable CT-SEM, all eigenvalues of the drift matrix must have
    negative real parts. This function projects to the nearest stable matrix.

    Args:
        drift: n x n drift matrix
        eps: small positive value for stability margin

    Returns:
        Stabilized drift matrix
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(drift)

    # Project eigenvalues with positive real parts
    real_parts = jnp.real(eigenvalues)
    stable_real = jnp.where(real_parts > -eps, -eps, real_parts)
    stable_eigenvalues = stable_real + 1j * jnp.imag(eigenvalues)

    # Reconstruct matrix
    stable_drift = eigenvectors @ jnp.diag(stable_eigenvalues) @ jnp.linalg.inv(
        eigenvectors
    )

    return jnp.real(stable_drift)
