"""Intervention DSL for counterfactual simulation.

Two override kinds, both targeted at a vector field:

- ``VariableOverride``: hard clamp ``eta[index] = value_fn(t)``. Sets the
  initial condition and the drift component so the latent tracks the
  callable. Equivalent to Pearl's ``do(η_index = u(t))``.

- ``EdgeInputOverride``: surgical replacement of the source value as seen
  by a specific target edge. Other consumers of the source are unchanged.
  This is the primitive that makes the non-linear edge vocabulary
  (Hill, multiplicative, effect-compartment) cleanly interveneable later.

Value functions are ``eqx.Module`` pytrees: array fields traced through
``vmap`` / ``jit``, structure carried in the treedef. ``ConstantValueFn``
covers the immediate ``set`` / ``shift`` cases; ``LinearRampValueFn``
covers piecewise-linear protocols (e.g., dose tapers). Adding a new value
family is a new ``eqx.Module`` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


class ConstantValueFn(eqx.Module):
    """Time-invariant value function ``u(t) = value``."""

    value: Array

    def __call__(self, _t: Array) -> Array:
        return self.value


class LinearRampValueFn(eqx.Module):
    """Piecewise-linear ramp between ``(t_start, value_start)`` and
    ``(t_end, value_end)``. Holds the endpoints outside the window. Useful
    for dose tapers and other protocol-shaped interventions."""

    t_start: Array
    t_end: Array
    value_start: Array
    value_end: Array

    def __call__(self, t: Array) -> Array:
        frac = jnp.clip(
            (t - self.t_start) / jnp.maximum(self.t_end - self.t_start, 1e-12), 0.0, 1.0
        )
        return self.value_start + frac * (self.value_end - self.value_start)


class PrecomputedValueFn(eqx.Module):
    """Piecewise-linear interpolation of ``values`` sampled at ``times``.

    Holds the endpoints outside ``[times[0], times[-1]]`` (``jnp.interp``
    semantics). Used for ``trajectory`` clamps where the caller supplies an
    explicit list of values across a window."""

    times: Array
    values: Array

    def __call__(self, t: Array) -> Array:
        return jnp.interp(t, self.times, self.values)


ValueFn = ConstantValueFn | LinearRampValueFn | PrecomputedValueFn


class VariableOverride(eqx.Module):
    """Hard clamp ``eta[index]`` to ``value_fn(t)`` for all simulated time."""

    index: int = eqx.field(static=True)
    value_fn: ValueFn


class EdgeInputOverride(eqx.Module):
    """Replace ``eta[source]`` with ``value_fn(t)`` only when computing the
    drift contribution to ``eta[target]``. Other edges from ``source`` see
    the natural state."""

    source: int = eqx.field(static=True)
    target: int = eqx.field(static=True)
    value_fn: ValueFn


Override = VariableOverride | EdgeInputOverride


class Intervention(eqx.Module):
    """Set of overrides active for the entire simulation horizon.

    Time-windowed activation is expressed inside ``value_fn`` rather than
    at the ``Intervention`` level (use ``LinearRampValueFn`` or compose
    new ``ValueFn`` modules).
    """

    overrides: tuple[Override, ...]

    @classmethod
    def none(cls) -> Intervention:
        return cls(overrides=())

    def variable_overrides(self) -> tuple[VariableOverride, ...]:
        return tuple(o for o in self.overrides if isinstance(o, VariableOverride))

    def edge_input_overrides(self) -> tuple[EdgeInputOverride, ...]:
        return tuple(o for o in self.overrides if isinstance(o, EdgeInputOverride))


def constant_value(value: Array) -> ConstantValueFn:
    """Factory for ``ConstantValueFn``."""
    return ConstantValueFn(value=value)


def linear_ramp(
    *,
    t_start: Array,
    t_end: Array,
    value_start: Array,
    value_end: Array,
) -> LinearRampValueFn:
    """Factory for ``LinearRampValueFn``."""
    return LinearRampValueFn(
        t_start=t_start,
        t_end=t_end,
        value_start=value_start,
        value_end=value_end,
    )


def precomputed_value(times: Array, values: Array) -> PrecomputedValueFn:
    """Factory for ``PrecomputedValueFn``."""
    return PrecomputedValueFn(times=times, values=values)
