"""CT→DT discretization for arbitrary vector fields via local linearization.

The exact discretization module handles the dense-matrix
linear case directly. This module extends that to non-linear vector
fields by:

1. Locally linearising ``f(t, x, args)`` around a chosen point ``x_lin``
   (typically the filter's current mean estimate, or the current MCMC
   trajectory sample at step ``t-1``).
2. Applying the existing exact (Van Loan) expm discretization to the
   linearised system.

For a pure ``DenseLinear`` component with no intervention this reduces
to the existing ``discretize_linear_system_exact(A, GG', c, dt)``
exactly. For Hill / Multiplicative / effect-compartment compositions,
the local Jacobian falls out of ``jax.jacfwd`` and the same expm path
applies.

The Van Loan formulation is used rather than the stationary-covariance
identity because local linearisations far from equilibrium can be
unstable (eigenvalues with positive real part), which breaks the
``A·Q + Q·A' = -G·G'`` solution.

The relation to Corenflos & Särkkä (2025) §2.3 / Example 2.1: the per-
step linearisation here is exactly the ``F_{t-1} ≈ ∇f(x_{t-1})``,
``b_{t-1} ≈ f(x_{t-1}) - F_{t-1} x_{t-1}`` first-order approximation.
These primitives are consumed only by the IEKS/Laplace warmup backend
that *initialises* the particle samplers (positions, preconditioner,
reference path); no production estimate, diagnostic, or counterfactual
uses them.
"""

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.discretization.exact import discretize_linear_system_exact
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from nof1_causal_lab.models.ssm.shapes import Array, Float


def discretize_at_state(
    vector_field: VectorField,
    x_lin: Float[Array, " D"],
    args: VectorFieldArgs,
    diffusion_cov: Float[Array, "D D"],
    dt: float | Array,
    t: float | Array = 0.0,
) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Float[Array, " D"]]:
    """Locally linearise ``vector_field`` at ``x_lin`` and discretise over ``dt``.

    Returns ``(A_d, Q_d, b_d)`` such that the linearised conditional
    transition over the interval ``dt`` is::

        E[x_{t+dt} | x_t] ≈ A_d · x_t + b_d
        Cov[x_{t+dt} | x_t] ≈ Q_d

    Args:
        vector_field: The vector-field drift; ``linearize`` is called on it.
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
    # b_loc is always a concrete intercept (linearize never returns None), so the
    # exact discretizer's conditional `b_d: Array | None` is non-None here; ty
    # cannot narrow the input-conditioned None away.
    return discretize_linear_system_exact(A_loc, diffusion_cov, b_loc, dt)  # ty: ignore[invalid-return-type]


def discretize_at_states_batched(
    vector_field: VectorField,
    x_lin_batch: Float[Array, "T D"],
    args: VectorFieldArgs,
    diffusion_cov: Float[Array, "D D"],
    dt_batch: Float[Array, " T"],
) -> tuple[Float[Array, "T D D"], Float[Array, "T D D"], Float[Array, "T D"]]:
    """Per-step discretisation along a trajectory.

    ``vmap`` over per-step ``(x_lin_t, dt_t)`` so that an offline
    pre-discretisation can be computed when the linearisation trajectory
    is known (e.g., the Laplace mode-finding path, or the auxiliary
    Kalman sampler's current trajectory snapshot).

    Args:
        vector_field: As in ``discretize_at_state``.
        x_lin_batch: ``(T, n_latent)`` per-step linearisation points.
        args: Shared ``VectorFieldArgs`` (params + intervention).
        diffusion_cov: ``(n_latent, n_latent)``, shared across steps.
        dt_batch: ``(T,)`` per-step intervals.

    Returns:
        ``(A_d_batch, Q_d_batch, b_d_batch)`` with leading dim ``T``.
    """

    def _per_step(
        x_lin: Float[Array, " D"], dt: Array
    ) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Float[Array, " D"]]:
        return discretize_at_state(vector_field, x_lin, args, diffusion_cov, dt)

    return jax.vmap(_per_step, in_axes=(0, 0))(x_lin_batch, dt_batch)
