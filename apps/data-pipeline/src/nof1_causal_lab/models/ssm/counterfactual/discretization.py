"""CT→DT discretization for arbitrary vector fields via local linearization.

The existing ``models/ssm/discretization.py`` handles the dense-matrix
linear case directly. This module extends that to non-linear vector
fields by:

1. Locally linearising ``f(t, x, args)`` around a chosen point ``x_lin``
   (typically the filter's current mean estimate).
2. Applying the existing exact (Van Loan) expm discretization to the
   linearised system.

For a pure ``DenseLinear`` component with no intervention this reduces
to the existing ``discretize_linear_system_exact(A, GG', c, dt)``
exactly. For Hill / Multiplicative / Effect-compartment compositions,
the local Jacobian falls out of ``jax.jacfwd`` and the same expm path
applies.

The Van Loan formulation is used rather than the stationary-covariance
identity because local linearisations far from equilibrium can be
unstable (eigenvalues with positive real part), which breaks the
``A·Q + Q·A' = -G·G'`` solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.discretization import discretize_linear_system_exact

if TYPE_CHECKING:
    from jax import Array

    from .vector_field import CompositeVectorField, VectorFieldArgs


def discretize_at_state(
    vector_field: CompositeVectorField,
    x_lin: Array,
    args: VectorFieldArgs,
    diffusion_cov: Array,
    dt: float | Array,
    t: float | Array = 0.0,
) -> tuple[Array, Array, Array]:
    """Locally linearise ``vector_field`` at ``x_lin`` and discretise over ``dt``.

    Returns ``(A_d, Q_d, b_d)`` such that the linearised conditional
    transition over the interval ``dt`` is::

        E[x_{t+dt} | x_t] ≈ A_d · x_t + b_d
        Cov[x_{t+dt} | x_t] ≈ Q_d

    Args:
        vector_field: The composite drift; ``linearize`` is called on it.
        x_lin: Linearisation point (``(n_latent,)``). For EKF-style use
            this is the current mean estimate at the start of the interval.
        args: ``VectorFieldArgs`` threaded to the field — its
            ``intervention`` is honoured by the linearisation (so
            counterfactual discretisation uses the intervened drift).
        diffusion_cov: ``G · G'`` for the assumed state-independent
            diffusion. Shape ``(n_latent, n_latent)``.
        dt: Length of the interval.
        t: Time at which to linearise (passes into the vector field).
            Defaults to ``0.0`` — irrelevant when the field is
            time-invariant, which is the case for every primitive
            currently in the library.

    Returns:
        ``(A_d, Q_d, b_d)`` discrete-time linearised system.
    """
    A_loc, b_loc = vector_field.linearize(x_lin, args, jnp.asarray(t))
    return discretize_linear_system_exact(A_loc, diffusion_cov, b_loc, dt)
