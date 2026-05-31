"""Parameter-space proposal helpers shared by particle MCMC methods."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random


def _preconditioner_chol(
    dim: int,
    *,
    dtype,
    parameter_preconditioner_chol: jnp.ndarray | None,
) -> jnp.ndarray:
    if parameter_preconditioner_chol is None:
        return jnp.eye(dim, dtype=dtype)
    return jnp.asarray(parameter_preconditioner_chol, dtype=dtype)


def preconditioned_random_walk_proposal(
    key: jnp.ndarray,
    position: jnp.ndarray,
    step_size: jnp.ndarray,
    *,
    parameter_preconditioner_chol: jnp.ndarray | None,
    variance_factor: float,
) -> jnp.ndarray:
    """Symmetric Gaussian random-walk proposal in unconstrained coordinates."""
    dim = int(position.shape[0])
    chol = _preconditioner_chol(
        dim,
        dtype=position.dtype,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
    )
    step = jnp.asarray(step_size, dtype=position.dtype)
    scale = jnp.sqrt(jnp.asarray(variance_factor, dtype=position.dtype) * step)
    noise = random.normal(key, position.shape, dtype=position.dtype) @ chol.T
    return position + scale * noise


def parameter_jump_rms(next_position: jnp.ndarray, current_position: jnp.ndarray) -> jnp.ndarray:
    jump = next_position - current_position
    return jnp.sqrt(jnp.mean(jump * jump))
