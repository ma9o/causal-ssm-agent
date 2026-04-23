"""Composition combinator for bijectors."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .bijectors import Bijector

Array = jax.Array


@dataclass(frozen=True)
class Chain(Bijector):
    """Compose a sequence of bijectors; ``bijectors[0]`` is applied first in forward."""

    bijectors: tuple[Bijector, ...]

    def forward(self, u: Array) -> Array:
        x = u
        for b in self.bijectors:
            x = b.forward(x)
        return x

    def inverse(self, x: Array) -> Array:
        u = x
        for b in reversed(self.bijectors):
            u = b.inverse(u)
        return u

    def forward_log_det_jac(self, u: Array) -> Array:
        total = jnp.zeros(u.shape[:-1])
        x = u
        for b in self.bijectors:
            total = total + b.forward_log_det_jac(x)
            x = b.forward(x)
        return total

    def inverse_log_det_jac(self, x: Array) -> Array:
        total = jnp.zeros(x.shape[:-1])
        u = x
        for b in reversed(self.bijectors):
            total = total + b.inverse_log_det_jac(u)
            u = b.inverse(u)
        return total
