"""CT→DT discretization for arbitrary vector fields via local linearization.

The existing ``models/ssm/discretization.py`` handles the dense-matrix
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
``b_{t-1} ≈ f(x_{t-1}) - F_{t-1} x_{t-1}`` first-order approximation
that auxiliary Kalman samplers use to form proposals for non-linear
latent dynamics. The actual integration into ``auxiliary_kalman.py`` is
deferred (see ``scratchpad/TODO.md``); this module provides the
primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.discretization import discretize_linear_system_exact

from .intervention import Intervention
from .vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from .vector_field import CompositeVectorField


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


def discretize_at_states_batched(
    vector_field: CompositeVectorField,
    x_lin_batch: Array,
    args: VectorFieldArgs,
    diffusion_cov: Array,
    dt_batch: Array,
) -> tuple[Array, Array, Array]:
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

    def _per_step(x_lin: Array, dt: Array) -> tuple[Array, Array, Array]:
        return discretize_at_state(vector_field, x_lin, args, diffusion_cov, dt)

    return jax.vmap(_per_step, in_axes=(0, 0))(x_lin_batch, dt_batch)


def make_filter_dynamics_callback(
    vector_field: CompositeVectorField,
    vf_params: tuple[dict[str, Array], ...],
    intervention: Intervention | None = None,
    *,
    diffusion_cov: Array,
    jitter: float = 1e-6,
) -> Callable:
    """Build a cuthbert-compatible ``get_dynamics_params`` callback.

    The returned callable matches the signature
    ``(state, model_inputs) -> (dynamics_fn, mean)`` expected by
    ``cuthbert.gaussian.moments.build_filter``. Each call:

    1. Reads the current step's ``dt`` from ``model_inputs['dt']``.
    2. Linearises ``vector_field`` at ``state.mean`` (the filter's
       running estimate).
    3. Discretises to ``(A_d, b_d, chol(Q_d + jitter·I))`` via the Van
       Loan expm path.
    4. Returns a ``dynamics_fn(x) = (A_d·x + b_d, chol_Q)`` closure that
       is *linear* in ``x``; the per-step linearisation point is fixed
       per call (proper EKF semantics).

    ``vf_params`` and ``intervention`` are captured at construction
    time — within a single filter pass they don't vary. For inference
    use, build a fresh callback per parameter draw or per MCMC step
    (cheap; just creates a closure).

    Args:
        vector_field: Drift specification.
        vf_params: Per-component parameter tuple matching ``vector_field.components``.
        intervention: Optional intervention; defaults to ``Intervention.none()``.
            For inference, the natural-dynamics case, leave as ``None``.
        diffusion_cov: ``G·G'`` state-independent diffusion.
        jitter: Added to the diagonal of ``Q_d`` before Cholesky for
            numerical stability when the linearisation gives a
            near-singular noise covariance.

    Returns:
        A ``get_dynamics_params(state, model_inputs)`` callable.
    """
    if intervention is None:
        intervention = Intervention.none()
    args = VectorFieldArgs(params=vf_params, intervention=intervention)
    n = vector_field.n_latent

    def get_dynamics_params(state, model_inputs):
        dt = model_inputs["dt"]
        x_lin = state.mean
        A_d, Q_d, b_d = discretize_at_state(
            vector_field, x_lin, args, diffusion_cov, dt
        )
        chol_Q = jnp.linalg.cholesky(Q_d + jitter * jnp.eye(n, dtype=Q_d.dtype))

        def dynamics_fn(x: Array) -> tuple[Array, Array]:
            return A_d @ x + b_d, chol_Q

        return dynamics_fn, state.mean

    return get_dynamics_params
