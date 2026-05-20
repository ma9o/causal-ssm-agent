"""Numerical steady-state finder via Optimistix.

Generalises the closed-form ``-A⁻¹c`` to any vector field by solving the
nonlinear root ``f(0, η*, args) = 0``. For linear vector fields the root
is unique and the solver converges in a couple of Newton steps from any
reasonable initial guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import optimistix as optx

from .vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from jax import Array

    from .intervention import Intervention
    from .vector_field import VectorField


def compute_steady_state(
    vector_field: VectorField,
    params: dict[str, Array],
    intervention: Intervention,
    initial_guess: Array | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 256,
) -> Array:
    """Find ``η*`` such that ``f(0, η*, args) = 0``.

    For ``LinearVectorField`` with a stable drift, this reproduces
    ``-A⁻¹c`` (up to solver tolerance) and naturally extends to
    interventions, which simply alter the equation system whose root we
    seek.
    """
    args = VectorFieldArgs(params=params, intervention=intervention)
    if initial_guess is None:
        initial_guess = jnp.zeros(vector_field.n_latent)

    initial_guess = vector_field.initial_condition(initial_guess, args)

    def residual(eta: Array, residual_args: VectorFieldArgs) -> Array:
        return vector_field.steady_state_residual(eta, residual_args)

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    solution = optx.root_find(
        residual,
        solver,
        initial_guess,
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    return solution.value
