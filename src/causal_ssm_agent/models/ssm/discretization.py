"""CT→DT Discretization for continuous-time state-space models.

Implements the mathematical operations needed for continuous-to-discrete
transformation in state-space models:

1. Matrix exponential: exp(A*dt) for discrete drift
2. Lyapunov solver: A*Q + Q*A' = -GG' for asymptotic diffusion
3. Discrete diffusion: Q_dt = Q_inf - exp(A*dt)*Q_inf*exp(A*dt)'
4. Discrete CINT: c_dt = A^{-1}*(exp(A*dt) - I)*c

This module is decoupled from state marginalization (Kalman/UKF/Particle)
to support different inference strategies.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import vmap


def solve_lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve the continuous Lyapunov equation: A*X + X*A' = -Q.

    Computes the asymptotic diffusion covariance.

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


def compute_asymptotic_diffusion(drift: jnp.ndarray, diffusion_cov: jnp.ndarray) -> jnp.ndarray:
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


def compute_discrete_cint(drift: jnp.ndarray, cint: jnp.ndarray, dt: float) -> jnp.ndarray:
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


def discretize_system(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Discretize the continuous-time system for a given time interval.

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


def _discretize_system_with_cint(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discretize with cint always present (vmap-compatible).

    Unlike discretize_system, this always computes discrete_cint,
    making it safe for use with jax.vmap over the dt axis.
    """
    discrete_drift = jla.expm(drift * dt)
    discrete_Q = compute_discrete_diffusion(drift, diffusion_cov, dt)
    discrete_cint = compute_discrete_cint(drift, cint, dt)
    return discrete_drift, discrete_Q, discrete_cint


def _discretize_system_no_cint(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discretize without cint (vmap-compatible)."""
    discrete_drift = jla.expm(drift * dt)
    discrete_Q = compute_discrete_diffusion(drift, diffusion_cov, dt)
    return discrete_drift, discrete_Q


def discretize_system_batched(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    dt_array: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Batch-discretize CT system over an array of time intervals.

    Uses jax.vmap over the dt dimension. For T timesteps, produces
    (T, n, n) arrays for drift and Q, and (T, n) for cint.

    Args:
        drift: (n, n) continuous drift matrix A
        diffusion_cov: (n, n) diffusion covariance (G*G')
        cint: (n,) continuous intercept or None
        dt_array: (T,) array of time intervals

    Returns:
        Ad: (T, n, n) discrete drift matrices
        Qd: (T, n, n) discrete process noise covariances
        cd: (T, n) discrete intercepts, or None if cint is None
    """
    if cint is not None:
        Ad, Qd, cd = vmap(lambda dt: _discretize_system_with_cint(drift, diffusion_cov, cint, dt))(
            dt_array
        )
        # cd comes out as (T, n, 1) from compute_discrete_cint — squeeze
        if cd.ndim == 3:
            cd = cd.squeeze(-1)
        return Ad, Qd, cd
    else:
        Ad, Qd = vmap(lambda dt: _discretize_system_no_cint(drift, diffusion_cov, dt))(dt_array)
        return Ad, Qd, None
