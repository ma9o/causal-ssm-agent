"""Target distributions built from bases, bijectors, and structural combinators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from .bases import Base
    from .bijectors import Bijector

Array = jax.Array
PRNGKey = jax.Array


class Target:
    """Density over R^D with ``log_prob``; may also support ``sample``.

    ``log_prob`` is exact (includes normalisation) for ``TransformedTarget``,
    ``Mixture``, and ``Mirror``. ``SoftConstraint`` returns an unnormalised
    log-density because its factor's partition function is generally intractable.
    """

    def log_prob(self, x: Array) -> Array:
        raise NotImplementedError

    def sample(self, key: PRNGKey, n: int) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__} does not support direct sampling; use MCMC on log_prob."
        )


@dataclass(frozen=True)
class TransformedTarget(Target):
    """Push-forward of ``base`` through ``bijector``.

    ``log p(x) = base.log_prob(bijector⁻¹(x)) + log|det J_{bijector⁻¹}(x)|``.
    """

    base: Base
    bijector: Bijector

    def log_prob(self, x: Array) -> Array:
        u = self.bijector.inverse(x)
        return self.base.log_prob(u) + self.bijector.inverse_log_det_jac(x)

    def sample(self, key: PRNGKey, n: int) -> Array:
        return self.bijector.forward(self.base.sample(key, n))


@dataclass(frozen=True)
class Mixture(Target):
    """Weighted mixture. Weights default to uniform; normalised internally."""

    components: tuple[Target, ...]
    weights: tuple[float, ...] = ()

    def _log_weights(self) -> Array:
        if self.weights:
            log_w = jnp.log(jnp.asarray(self.weights))
        else:
            log_w = jnp.zeros(len(self.components))
        return log_w - jax.scipy.special.logsumexp(log_w)

    def log_prob(self, x: Array) -> Array:
        lps = jnp.stack([c.log_prob(x) for c in self.components], axis=0)
        lw = self._log_weights().reshape((-1,) + (1,) * (lps.ndim - 1))
        return jax.scipy.special.logsumexp(lps + lw, axis=0)

    def sample(self, key: PRNGKey, n: int) -> Array:
        keys = jax.random.split(key, len(self.components) + 1)
        idx = jax.random.categorical(keys[0], self._log_weights(), shape=(n,))
        stacked = jnp.stack(
            [c.sample(k, n) for c, k in zip(self.components, keys[1:], strict=True)],
            axis=0,
        )
        return stacked[idx, jnp.arange(n)]


@dataclass(frozen=True)
class Mirror(Target):
    """Reflect ``base`` across the given axes; produces ``2^len(flip_axes)`` modes."""

    base: Target
    flip_axes: tuple[int, ...] = (0,)

    def _signs(self, dim: int) -> Array:
        k = len(self.flip_axes)
        patterns = jnp.stack(
            jnp.meshgrid(*([jnp.array([1.0, -1.0])] * k), indexing="ij"), axis=-1
        ).reshape(-1, k)
        return jnp.ones((patterns.shape[0], dim)).at[:, jnp.array(self.flip_axes)].set(patterns)

    def log_prob(self, x: Array) -> Array:
        signs = self._signs(x.shape[-1])
        lps = jnp.stack([self.base.log_prob(x * s) for s in signs], axis=0)
        return jax.scipy.special.logsumexp(lps, axis=0) - jnp.log(signs.shape[0])

    def sample(self, key: PRNGKey, n: int) -> Array:
        key_base, key_flip = jax.random.split(key)
        samples = self.base.sample(key_base, n)
        signs = self._signs(samples.shape[-1])
        idx = jax.random.randint(key_flip, (n,), 0, signs.shape[0])
        return samples * signs[idx]


@dataclass(frozen=True)
class SoftConstraint(Target):
    """Add a penalty factor to ``base.log_prob``; yields an **unnormalised** density.

    ``log p(x) = base.log_prob(x) - weight · penalty(x) + const``.
    """

    base: Target
    penalty: Callable[[Array], Array]
    weight: float = 1.0

    def log_prob(self, x: Array) -> Array:
        return self.base.log_prob(x) - self.weight * self.penalty(x)


def invariance(
    base: Target,
    phi: Callable[[Array], Array],
    target_value: float = 0.0,
    tol: float = 0.1,
) -> SoftConstraint:
    """Constrain the projection ``phi(x) ≈ target_value``; orthogonal direction stays prior-dominated.

    Quadratic penalty: ``((phi(x) - target_value) / tol)²``. Specialises to

    - ``phi(x) = a·x₀ + b·x₁`` → additive ridge
    - ``phi(x) = x₀·x₁`` → hyperbolic (multiplicative) ridge
    - ``phi(x) = x₀² + x₁²`` → rotational / ring
    - ``phi(x) = x₀ / x₁`` → projective (angle-only identified)
    """

    def penalty(x: Array) -> Array:
        return ((phi(x) - target_value) / tol) ** 2

    return SoftConstraint(base=base, penalty=penalty, weight=0.5)
