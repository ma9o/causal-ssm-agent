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

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax, vmap


def _kron_lyapunov_solve(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve AX + XA' = -Q via Kronecker vectorization.

    (I ⊗ A + A ⊗ I) vec(X) = vec(-Q). O(n^6) but fully differentiable.
    """
    n = A.shape[0]
    I_n = jnp.eye(n)
    M = jnp.kron(I_n, A) + jnp.kron(A, I_n)
    X_vec = jla.solve(M, (-Q).reshape(-1))
    return X_vec.reshape(n, n)


@jax.custom_jvp
def solve_lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve the continuous Lyapunov equation: A*X + X*A' = -Q.

    Computes the asymptotic diffusion covariance.

    For a stable system (eigenvalues of A have negative real parts),
    this gives the stationary covariance of the process.

    Uses Kronecker vectorization: (I⊗A + A⊗I) vec(X) = vec(-Q).
    O(n^6) but GPU-compatible (jla.solve_sylvester uses Schur which has
    no CUDA XLA implementation). Backward pass uses implicit differentiation.

    Args:
        A: n x n drift matrix (must be stable for unique solution)
        Q: n x n positive semi-definite matrix (typically GG')

    Returns:
        X: n x n solution matrix (asymptotic covariance)
    """
    return _kron_lyapunov_solve(A, Q)


@solve_lyapunov.defjvp
def _solve_lyapunov_jvp(primals, tangents):
    """JVP via implicit differentiation of AX + XA' = -Q.

    Differentiating the Lyapunov equation gives:
    A dX + dX A' = -(dA X + X dA' + dQ)
    """
    A, Q = primals
    dA, dQ = tangents
    X = _kron_lyapunov_solve(A, Q)
    tangent_rhs = dA @ X + X @ dA.T + dQ
    dX = _kron_lyapunov_solve(A, tangent_rhs)
    return X, dX


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
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    dt: float | jax.Array,
    discrete_drift: jnp.ndarray | None = None,
    asymptotic_diffusion: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute discrete-time diffusion covariance for time interval dt.

    Q_dt = Q_inf - exp(A*dt) * Q_inf * exp(A*dt)'

    Where Q_inf is the asymptotic diffusion from the Lyapunov equation.

    Args:
        drift: n x n drift matrix A
        diffusion_cov: n x n diffusion covariance (G*G')
        dt: time interval
        discrete_drift: Pre-computed exp(A*dt), or None to compute internally.
        asymptotic_diffusion: Pre-computed stationary covariance Q_inf, or
            None to solve the Lyapunov equation internally.

    Returns:
        Q_dt: n x n discrete diffusion covariance
    """
    # Compute asymptotic diffusion
    Q_inf = (
        asymptotic_diffusion
        if asymptotic_diffusion is not None
        else compute_asymptotic_diffusion(drift, diffusion_cov)
    )

    # Compute discrete drift (reuse if provided)
    if discrete_drift is None:
        discrete_drift = jla.expm(drift * dt)

    # Q_dt = Q_inf - exp(A*dt) * Q_inf * exp(A*dt)'
    Q_dt = Q_inf - discrete_drift @ Q_inf @ discrete_drift.T

    # Ensure symmetry
    return 0.5 * (Q_dt + Q_dt.T)


def compute_discrete_cint(
    drift: jnp.ndarray,
    cint: jnp.ndarray,
    dt: float | jax.Array,
    discrete_drift: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute discrete-time intercept for time interval dt.

    c_dt = A^{-1} * (exp(A*dt) - I) * c

    This is the integrated effect of the continuous intercept over dt.

    Args:
        drift: n x n drift matrix A
        cint: n x 1 continuous intercept c
        dt: time interval
        discrete_drift: Pre-computed exp(A*dt), or None to compute internally.

    Returns:
        c_dt: n x 1 discrete intercept
    """
    n = drift.shape[0]
    I_n = jnp.eye(n)

    # Compute discrete drift (reuse if provided)
    if discrete_drift is None:
        discrete_drift = jla.expm(drift * dt)

    # c_dt = A^{-1} * (exp(A*dt) - I) * c
    # Using solve for numerical stability: A * c_dt = (exp(A*dt) - I) * c
    rhs = (discrete_drift - I_n) @ cint
    return jla.solve(drift, rhs)


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
    asymptotic_diffusion = compute_asymptotic_diffusion(drift, diffusion_cov)

    # Discrete drift via matrix exponential (computed once, shared)
    discrete_drift = jla.expm(drift * dt)

    # Discrete diffusion via Lyapunov solution
    discrete_Q = compute_discrete_diffusion(
        drift,
        diffusion_cov,
        dt,
        discrete_drift=discrete_drift,
        asymptotic_diffusion=asymptotic_diffusion,
    )

    # Discrete intercept
    discrete_cint = None
    if cint is not None:
        discrete_cint = compute_discrete_cint(drift, cint, dt, discrete_drift=discrete_drift)

    return discrete_drift, discrete_Q, discrete_cint


def _discretize_system_with_cint(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray,
    dt: float | jax.Array,
    asymptotic_diffusion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discretize with cint always present (vmap-compatible).

    Unlike discretize_system, this always computes discrete_cint,
    making it safe for use with jax.vmap over the dt axis.
    """
    discrete_drift = jla.expm(drift * dt)
    discrete_Q = compute_discrete_diffusion(
        drift,
        diffusion_cov,
        dt,
        discrete_drift=discrete_drift,
        asymptotic_diffusion=asymptotic_diffusion,
    )
    discrete_cint = compute_discrete_cint(drift, cint, dt, discrete_drift=discrete_drift)
    return discrete_drift, discrete_Q, discrete_cint


def _discretize_system_no_cint(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    dt: float | jax.Array,
    asymptotic_diffusion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discretize without cint (vmap-compatible)."""
    discrete_drift = jla.expm(drift * dt)
    discrete_Q = compute_discrete_diffusion(
        drift,
        diffusion_cov,
        dt,
        discrete_drift=discrete_drift,
        asymptotic_diffusion=asymptotic_diffusion,
    )
    return discrete_drift, discrete_Q


def _normalize_batched_cint(discrete_cint: jnp.ndarray) -> jnp.ndarray:
    """Drop the trailing singleton axis some solve paths emit for cint."""
    if discrete_cint.ndim > 0 and discrete_cint.shape[-1] == 1:
        return discrete_cint.squeeze(-1)
    return discrete_cint


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
    n_steps = dt_array.shape[0]
    n_latent = drift.shape[0]

    if n_steps == 0:
        Ad = jnp.empty((0, n_latent, n_latent), dtype=drift.dtype)
        Qd = jnp.empty((0, n_latent, n_latent), dtype=diffusion_cov.dtype)
        if cint is None:
            return Ad, Qd, None
        cint_arr = _normalize_batched_cint(jnp.asarray(cint))
        cd = jnp.empty((0, *cint_arr.shape), dtype=cint_arr.dtype)
        return Ad, Qd, cd

    asymptotic_diffusion = compute_asymptotic_diffusion(drift, diffusion_cov)
    same_dt = jnp.all(jnp.isclose(dt_array, dt_array[0]))

    if cint is not None:

        def _all_same_dt(_):
            Ad_single, Qd_single, cd_single = _discretize_system_with_cint(
                drift,
                diffusion_cov,
                cint,
                dt_array[0],
                asymptotic_diffusion,
            )
            cd_single = _normalize_batched_cint(cd_single)
            return (
                jnp.broadcast_to(Ad_single, (n_steps, *Ad_single.shape)),
                jnp.broadcast_to(Qd_single, (n_steps, *Qd_single.shape)),
                jnp.broadcast_to(cd_single, (n_steps, *cd_single.shape)),
            )

        def _varying_dt(_):
            Ad, Qd, cd = vmap(
                lambda dt: _discretize_system_with_cint(
                    drift,
                    diffusion_cov,
                    cint,
                    dt,
                    asymptotic_diffusion,
                )
            )(dt_array)
            return Ad, Qd, _normalize_batched_cint(cd)

        return lax.cond(same_dt, _all_same_dt, _varying_dt, operand=None)

    def _all_same_dt_no_cint(_):
        Ad_single, Qd_single = _discretize_system_no_cint(
            drift,
            diffusion_cov,
            dt_array[0],
            asymptotic_diffusion,
        )
        return (
            jnp.broadcast_to(Ad_single, (n_steps, *Ad_single.shape)),
            jnp.broadcast_to(Qd_single, (n_steps, *Qd_single.shape)),
        )

    def _varying_dt_no_cint(_):
        return vmap(
            lambda dt: _discretize_system_no_cint(
                drift,
                diffusion_cov,
                dt,
                asymptotic_diffusion,
            )
        )(dt_array)

    Ad, Qd = lax.cond(same_dt, _all_same_dt_no_cint, _varying_dt_no_cint, operand=None)
    return Ad, Qd, None
