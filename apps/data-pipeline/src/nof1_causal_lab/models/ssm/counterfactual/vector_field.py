"""Vector field protocol and current linear implementation.

The vector field abstracts the SSM drift ``f(t, η, args) -> dη/dt``. Both
the trajectory simulator (Diffrax) and the steady-state root-finder
(Optimistix) consume this protocol, so adding a non-linear edge primitive
library later is purely additive: a new ``VectorField`` implementation
slots in behind the same API with no caller changes.

``LinearVectorField`` is the current regime: ``f(t, η) = A·η + c``. Both
``A`` and ``c`` are read from a parameter pytree at call time, which keeps
the field itself stateless (and therefore safe inside ``vmap`` over
posterior draws). Interventions are applied through the field rather than
mutating the parameters, so a single set of posterior draws can be reused
for baseline and counterfactual paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .intervention import EdgeInputOverride, Intervention, VariableOverride


class VectorFieldArgs(eqx.Module):
    """Arguments threaded through Diffrax / Optimistix to the field.

    ``params`` carries traced posterior arrays (one pytree leaf per array).
    ``intervention`` is an ``eqx.Module`` pytree — its array fields are
    traced, its structure (override count, types, indices) lives in the
    treedef.
    """

    params: dict[str, Array]
    intervention: Intervention


@runtime_checkable
class VectorField(Protocol):
    """Stateless callable producing ``dη/dt`` at a state ``η`` and time ``t``.

    Implementations must also provide ``initial_condition``, which clamps the
    initial state to variable overrides at ``t0`` (and may enforce richer
    invariants for non-linear primitive vocabularies later).
    """

    n_latent: int

    def __call__(self, t: Array, eta: Array, args: VectorFieldArgs) -> Array: ...

    def initial_condition(self, eta0: Array, args: VectorFieldArgs) -> Array: ...

    def steady_state_residual(self, eta: Array, args: VectorFieldArgs) -> Array: ...


def apply_variable_overrides_to_state(
    eta: Array,
    t: Array,
    intervention: Intervention,
) -> Array:
    """Clamp eta[i] = u_i(t) for each VariableOverride."""
    for ov in intervention.variable_overrides():
        eta = eta.at[ov.index].set(ov.value_fn(t))
    return eta


@dataclass(frozen=True)
class LinearVectorField:
    """Linear drift ``f(t, η) = A·η + c``.

    Parameters are read from ``args.params``:
      - ``params['drift']``: ``(n, n)`` drift matrix A
      - ``params['cint']``: ``(n,)`` continuous intercept c

    Edge-input overrides substitute the per-target source value before the
    matrix-vector product. Variable overrides clamp the drift component to
    ``d(value_fn)/dt`` so that ``eta[index]`` tracks the value function under
    forward integration (the initial condition is set via
    ``initial_condition``).
    """

    n_latent: int

    def __call__(self, t: Array, eta: Array, args: VectorFieldArgs) -> Array:
        drift_matrix = args.params["drift"]
        cint = args.params.get("cint", jnp.zeros(self.n_latent, dtype=eta.dtype))

        eta_eff = jnp.broadcast_to(eta[None, :], (self.n_latent, self.n_latent))
        eta_eff = _apply_edge_input_overrides(eta_eff, t, args.intervention)

        d_eta = (drift_matrix * eta_eff).sum(axis=1) + cint
        return _apply_variable_overrides_to_drift(d_eta, t, args.intervention)

    def initial_condition(self, eta0: Array, args: VectorFieldArgs) -> Array:
        return apply_variable_overrides_to_state(eta0, jnp.asarray(0.0), args.intervention)

    def steady_state_residual(self, eta: Array, args: VectorFieldArgs) -> Array:
        """Equilibrium residual with variable overrides treated as constraints.

        For unconstrained latents the residual is ``A·η + c`` (zero at the
        natural equilibrium). For constrained latents the residual is
        ``eta[i] - u(0)`` so the root pins ``eta[i]`` exactly, instead of
        relying on the drift component which the simulator-side override
        forces to ``du/dt``.
        """
        drift_matrix = args.params["drift"]
        cint = args.params.get("cint", jnp.zeros(self.n_latent, dtype=eta.dtype))
        eta_eff = jnp.broadcast_to(eta[None, :], (self.n_latent, self.n_latent))
        eta_eff = _apply_edge_input_overrides(eta_eff, jnp.asarray(0.0), args.intervention)
        residual = (drift_matrix * eta_eff).sum(axis=1) + cint
        for ov in args.intervention.variable_overrides():
            target = ov.value_fn(jnp.asarray(0.0))
            residual = residual.at[ov.index].set(eta[ov.index] - target)
        return residual


def _apply_edge_input_overrides(
    eta_eff: Array,
    t: Array,
    intervention: Intervention,
) -> Array:
    """Replace ``eta_eff[target, source]`` with ``u(t)`` per edge override."""
    for ov in intervention.edge_input_overrides():
        if not isinstance(ov, EdgeInputOverride):
            continue
        eta_eff = eta_eff.at[ov.target, ov.source].set(ov.value_fn(t))
    return eta_eff


def _apply_variable_overrides_to_drift(
    d_eta: Array,
    t: Array,
    intervention: Intervention,
) -> Array:
    """Replace ``d_eta[index]`` with ``d(value_fn)/dt`` for each variable
    override so the integrated trajectory matches ``value_fn``."""
    for ov in intervention.variable_overrides():
        if not isinstance(ov, VariableOverride):
            continue
        du_dt = jax.grad(lambda tt, fn=ov.value_fn: jnp.sum(fn(tt)))(t)
        d_eta = d_eta.at[ov.index].set(du_dt)
    return d_eta
