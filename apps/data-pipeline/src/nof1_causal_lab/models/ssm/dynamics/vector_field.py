"""Vector field — single concrete implementation built from components.

``CompositeVectorField`` is the only vector field. It owns a tuple of
``VectorFieldComponent``s (see ``edges.py``); each component contributes to
the derivative vector. The dense linear case (the existing Stage 5b posterior
shape) is one component (``DenseLinear``); the non-linear pharmacology
case is many components (``DiagonalDecay`` + ``Intercept`` + per-edge
Linear / Hill / Multiplicative).

The vector field is responsible for:

- Building the ``(n_target, n_source)`` ``eta_per_edge`` matrix with
  edge-input overrides applied once.
- Iterating over components and accumulating their contributions into
  the derivative.
- Translating ``VariableOverride``s into the right semantics for the
  simulator (derivative component set to ``du/dt``) and the steady-state
  root finder (residual set to ``eta − u(0)`` so the root pins the
  intervened latent exactly).

``args.params`` is a tuple matching the components tuple by position;
each component reads its own slice and never sees others'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from .intervention import EdgeInputOverride, Intervention, VariableOverride

if TYPE_CHECKING:
    from jax import Array

    from .edges import VectorFieldComponent


class VectorFieldArgs(eqx.Module):
    """Arguments threaded through Diffrax / Optimistix to the field.

    ``params`` is a tuple of per-component pytrees (one slice per
    component, matched by position). ``intervention`` is the
    ``eqx.Module`` pytree carrying override structure.
    """

    params: tuple[dict[str, Array], ...]
    intervention: Intervention


@runtime_checkable
class VectorField(Protocol):
    """Vector-field callable plus simulator / root-finder companions.

    Kept as a Protocol so future alternative implementations (e.g., a
    JAX-jit-cached variant or a structured sparse path) can slot in
    without changing call sites.
    """

    n_latent: int

    def __call__(self, t: Array, eta: Array, args: VectorFieldArgs) -> Array: ...

    def initial_condition(self, eta0: Array, args: VectorFieldArgs) -> Array: ...

    def steady_state_residual(self, eta: Array, args: VectorFieldArgs) -> Array: ...

    def linearize(
        self,
        x_lin: Array,
        args: VectorFieldArgs,
        t: Array | None = None,
    ) -> tuple[Array, Array]: ...


def apply_variable_overrides_to_state(
    eta: Array,
    t: Array,
    intervention: Intervention,
) -> Array:
    """Clamp ``eta[i] = u_i(t)`` for each variable override."""
    for ov in intervention.variable_overrides():
        eta = eta.at[ov.index].set(ov.value_fn(t))
    return eta


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


def _apply_variable_overrides_to_derivative(
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


class CompositeVectorField(eqx.Module):
    """Vector field as a sum of ``VectorFieldComponent`` contributions.

    Equivalent dense-matrix dynamics: a single ``DenseLinear`` component
    with parameter slice ``{"drift": A, "cint": c}`` reproduces the
    classic ``f(t, η) = A·η + c`` form exactly (and uses one matmul, not
    n² scatter-adds).

    Composite primitive dynamics: typically one ``DiagonalDecay`` + one
    ``Intercept`` + per-edge ``LinearEdge`` / ``HillEdge`` /
    ``MultiplicativeEdge``. Each component reads its slice of
    ``args.params`` by position.
    """

    n_latent: int = eqx.field(static=True)
    components: tuple[VectorFieldComponent, ...]

    def __call__(self, t: Array, eta: Array, args: VectorFieldArgs) -> Array:
        d_eta = self._natural_derivative(t, eta, args)
        return _apply_variable_overrides_to_derivative(d_eta, t, args.intervention)

    def initial_condition(self, eta0: Array, args: VectorFieldArgs) -> Array:
        return apply_variable_overrides_to_state(eta0, jnp.asarray(0.0), args.intervention)

    def steady_state_residual(self, eta: Array, args: VectorFieldArgs) -> Array:
        residual = self._natural_derivative(jnp.asarray(0.0), eta, args)
        for ov in args.intervention.variable_overrides():
            target = ov.value_fn(jnp.asarray(0.0))
            residual = residual.at[ov.index].set(eta[ov.index] - target)
        return residual

    def _natural_derivative(self, t: Array, eta: Array, args: VectorFieldArgs) -> Array:
        eta_eff = jnp.broadcast_to(eta[None, :], (self.n_latent, self.n_latent))
        eta_eff = _apply_edge_input_overrides(eta_eff, t, args.intervention)

        accumulator = jnp.zeros(self.n_latent, dtype=eta.dtype)
        for component, slice_params in zip(self.components, args.params, strict=True):
            accumulator = component.contribute(accumulator, eta, eta_eff, t, slice_params)
        return accumulator

    def linearize(
        self,
        x_lin: Array,
        args: VectorFieldArgs,
        t: Array | None = None,
    ) -> tuple[Array, Array]:
        """Local affine approximation ``f(t, x, args) ≈ A · x + b`` near ``x_lin``.

        ``A`` is the Jacobian ``∂f/∂x`` evaluated at ``x_lin`` via
        ``jax.jacfwd``; ``b = f(x_lin) - A · x_lin`` is the implied
        intercept. For a single ``DenseLinear`` component without
        intervention, ``A`` equals ``params['drift']`` and ``b`` equals
        ``params['cint']`` exactly. For non-linear components (Hill,
        Multiplicative, ...) the Jacobian falls out of autodiff.

        This is the seam through which the existing CT→DT expm
        discretization extends to non-linear vector fields: discretize the
        locally-linearized system at the filter's current mean estimate.
        """
        if t is None:
            t = jnp.asarray(0.0)
        f_at_x = self(t, x_lin, args)
        jacobian = jax.jacfwd(lambda x: self(t, x, args))(x_lin)
        intercept = f_at_x - jacobian @ x_lin
        return jacobian, intercept
