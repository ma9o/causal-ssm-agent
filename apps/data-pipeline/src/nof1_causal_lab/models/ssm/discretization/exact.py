"""CT→DT Discretization for continuous-time state-space models.

Implements the mathematical operations needed for continuous-to-discrete
transformation in state-space models:

1. Matrix exponential: exp(A*dt) for discrete drift
2. Van Loan block exponential for discrete diffusion
3. Augmented matrix exponential for discrete CINT

This module is decoupled from state marginalization (Kalman/UKF/Particle)
to support different inference strategies.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax, vmap

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize
from nof1_causal_lab.models.ssm.shapes import Array, Float


def _kron_lyapunov_solve(A: Float[Array, "D D"], Q: Float[Array, "D D"]) -> Float[Array, "D D"]:
    """Solve AX + XA' = -Q via Kronecker vectorization.

    (I ⊗ A + A ⊗ I) vec(X) = vec(-Q). O(n^6) but fully differentiable.
    """
    n = A.shape[0]
    I_n = jnp.eye(n)
    M = jnp.kron(I_n, A) + jnp.kron(A, I_n)
    X_vec = jla.solve(M, (-Q).reshape(-1))
    return X_vec.reshape(n, n)


@jax.custom_jvp
def solve_lyapunov(A: Float[Array, "D D"], Q: Float[Array, "D D"]) -> Float[Array, "D D"]:
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


def compute_discrete_diffusion_van_loan(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    dt: float | jax.Array,
) -> Float[Array, "D D"]:
    """Compute discrete diffusion exactly with the Van Loan block exponential.

    Unlike the stationary-covariance identity ``Q_inf − e^{A·dt} Q_inf e^{A·dt}ᵀ``
    (valid only for stable ``drift``), the Van Loan block exponential remains
    valid when ``drift`` is singular or unstable. That matters for augmented
    systems with accumulator states whose drift has zero eigenvalues.
    """
    n = drift.shape[0]
    zero = jnp.zeros_like(drift)
    van_loan = jnp.block(
        [
            [drift, diffusion_cov],
            [zero, -drift.T],
        ]
    )
    van_loan_exp = jla.expm(van_loan * dt)
    discrete_drift = van_loan_exp[:n, :n]
    upper_right = van_loan_exp[:n, n:]
    return symmetrize(upper_right @ discrete_drift.T)


def compute_discrete_cint_exact(
    drift: Float[Array, "D D"],
    cint: Array,
    dt: float | jax.Array,
) -> Float[Array, " D"]:
    """Compute the exact discrete intercept without assuming invertible drift."""
    n = drift.shape[0]
    cint_vec = jnp.asarray(cint, dtype=drift.dtype).reshape(n)
    augmented = jnp.zeros((n + 1, n + 1), dtype=drift.dtype)
    augmented = augmented.at[:n, :n].set(drift)
    augmented = augmented.at[:n, n].set(cint_vec)
    augmented_exp = jla.expm(augmented * dt)
    return augmented_exp[:n, n]


def discretize_system(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    cint: Array | None,
    dt: float,
) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Array | None]:
    """Discretize the continuous-time system for a given time interval.

    Computes:
    - discrete_drift = exp(A*dt)
    - discrete_Q via the Van Loan block exponential
    - discrete_cint via an augmented matrix exponential when provided

    Args:
        drift: n x n continuous drift matrix A
        diffusion_cov: n x n diffusion covariance (G*G')
        cint: n x 1 continuous intercept (optional)
        dt: time interval

    Returns:
        Tuple of (discrete_drift, discrete_Q, discrete_cint)
    """
    return discretize_linear_system_exact(drift, diffusion_cov, cint, dt)


def _normalize_batched_cint(discrete_cint: jnp.ndarray) -> jnp.ndarray:
    """Drop the trailing singleton axis some solve paths emit for cint."""
    if discrete_cint.ndim > 0 and discrete_cint.shape[-1] == 1:
        return discrete_cint.squeeze(-1)
    return discrete_cint


def discretize_linear_system_exact(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    cint: Array | None,
    dt: float | jax.Array,
) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Array | None]:
    """Exact CT→DT discretization for general linear systems.

    This variant is valid for augmented systems with singular drift, such as
    accumulator states used for linear interval summaries.
    """
    discrete_drift = jla.expm(drift * dt)
    discrete_Q = compute_discrete_diffusion_van_loan(drift, diffusion_cov, dt)
    discrete_cint = None
    if cint is not None:
        discrete_cint = compute_discrete_cint_exact(drift, cint, dt)
    return discrete_drift, discrete_Q, discrete_cint


def discretize_linear_system_exact_batched(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    cint: Array | None,
    dt_array: Float[Array, " T"],
) -> tuple[Float[Array, "T D D"], Float[Array, "T D D"], Array | None]:
    """Batch exact discretization for general linear systems."""
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

    same_dt = jnp.all(jnp.isclose(dt_array, dt_array[0]))

    if cint is not None:

        def _all_same_dt(_):
            Ad_single, Qd_single, cd_single = discretize_linear_system_exact(
                drift,
                diffusion_cov,
                cint,
                dt_array[0],
            )
            cd_single = _normalize_batched_cint(jnp.asarray(cd_single))
            return (
                jnp.broadcast_to(Ad_single, (n_steps, *Ad_single.shape)),
                jnp.broadcast_to(Qd_single, (n_steps, *Qd_single.shape)),
                jnp.broadcast_to(cd_single, (n_steps, *cd_single.shape)),
            )

        def _varying_dt(_):
            Ad, Qd, cd = vmap(
                lambda dt: discretize_linear_system_exact(
                    drift,
                    diffusion_cov,
                    cint,
                    dt,
                )
            )(dt_array)
            return Ad, Qd, _normalize_batched_cint(jnp.asarray(cd))

        return lax.cond(same_dt, _all_same_dt, _varying_dt, operand=None)

    def _all_same_dt_no_cint(_):
        Ad_single, Qd_single, _ = discretize_linear_system_exact(
            drift,
            diffusion_cov,
            None,
            dt_array[0],
        )
        return (
            jnp.broadcast_to(Ad_single, (n_steps, *Ad_single.shape)),
            jnp.broadcast_to(Qd_single, (n_steps, *Qd_single.shape)),
        )

    def _varying_dt_no_cint(_):
        Ad, Qd, _ = vmap(
            lambda dt: discretize_linear_system_exact(
                drift,
                diffusion_cov,
                None,
                dt,
            )
        )(dt_array)
        return Ad, Qd

    Ad, Qd = lax.cond(same_dt, _all_same_dt_no_cint, _varying_dt_no_cint, operand=None)
    return Ad, Qd, None


def discretize_system_batched(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    cint: Array | None,
    dt_array: Float[Array, " T"],
) -> tuple[Float[Array, "T D D"], Float[Array, "T D D"], Array | None]:
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

    return discretize_linear_system_exact_batched(drift, diffusion_cov, cint, dt_array)


def discretize_system_with_inputs_batched(
    drift: Float[Array, "D D"],
    diffusion_cov: Float[Array, "D D"],
    cint: Array | None,
    input_effect: Array | None,
    transition_inputs: Array | None,
    dt_array: Float[Array, " T"],
) -> tuple[Float[Array, "T D D"], Float[Array, "T D D"], Array | None]:
    """Batch-discretize CT dynamics with piecewise-constant known inputs."""
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)
    if input_effect is None or input_effect.shape[1] == 0:
        return Ad, Qd, cd
    if transition_inputs is None:
        raise ValueError("SSM has known input effects but transition_inputs was not provided.")

    transition_inputs = jnp.asarray(transition_inputs, dtype=drift.dtype)
    if transition_inputs.shape != (dt_array.shape[0], input_effect.shape[1]):
        raise ValueError(
            "transition_inputs must have shape "
            f"({dt_array.shape[0]}, {input_effect.shape[1]}), got {transition_inputs.shape}"
        )

    continuous_forcing = transition_inputs @ jnp.asarray(input_effect, dtype=drift.dtype).T
    input_cd = vmap(lambda forcing, dt: compute_discrete_cint_exact(drift, forcing, dt))(
        continuous_forcing,
        dt_array,
    )
    if cd is None:
        return Ad, Qd, input_cd
    cd = jnp.asarray(cd, dtype=input_cd.dtype)
    if cd.ndim == 1:
        cd = cd[:, None]
    return Ad, Qd, cd + input_cd
