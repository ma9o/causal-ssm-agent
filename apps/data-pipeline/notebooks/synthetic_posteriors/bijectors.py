"""Bijectors: invertible differentiable maps R^D -> R^D with tractable log-det-jacobian.

Each primitive corresponds to one named geometric pathology. Compose via ``Chain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

Array = jax.Array


class Bijector:
    """Invertible map u → x with analytical log|det J| in at least one direction."""

    def forward(self, u: Array) -> Array:
        raise NotImplementedError

    def inverse(self, x: Array) -> Array:
        raise NotImplementedError

    def forward_log_det_jac(self, u: Array) -> Array:
        raise NotImplementedError

    def inverse_log_det_jac(self, x: Array) -> Array:
        return -self.forward_log_det_jac(self.inverse(x))


@dataclass(frozen=True)
class Identity(Bijector):
    def forward(self, u: Array) -> Array:
        return u

    def inverse(self, x: Array) -> Array:
        return x

    def forward_log_det_jac(self, u: Array) -> Array:
        return jnp.zeros(u.shape[:-1])


@dataclass(frozen=True)
class Shift(Bijector):
    """Constant translation: ``x = u + offset``. log|det J| = 0."""

    offset: tuple[float, ...] = (0.0, 0.0)

    def forward(self, u: Array) -> Array:
        return u + jnp.asarray(self.offset)

    def inverse(self, x: Array) -> Array:
        return x - jnp.asarray(self.offset)

    def forward_log_det_jac(self, u: Array) -> Array:
        return jnp.zeros(u.shape[:-1])


@dataclass(frozen=True)
class Shear(Bijector):
    """2D rotate-and-scale: ``x = R(theta) · diag(scale) · u``.

    Produces linear correlation / elongated ridges. Equal scales → pure rotation.
    Unequal scales → elliptical Gaussian. ``theta`` rotates the ridge direction.
    """

    theta: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)

    def _R(self) -> Array:
        c, s = jnp.cos(self.theta), jnp.sin(self.theta)
        return jnp.array([[c, -s], [s, c]])

    def forward(self, u: Array) -> Array:
        return (u * jnp.asarray(self.scale)) @ self._R().T

    def inverse(self, x: Array) -> Array:
        return (x @ self._R()) / jnp.asarray(self.scale)

    def forward_log_det_jac(self, u: Array) -> Array:
        log_det = jnp.log(jnp.asarray(self.scale)).sum()
        return jnp.broadcast_to(log_det, u.shape[:-1])


@dataclass(frozen=True)
class Bend(Bijector):
    """Banana warp: shift one axis by a nonlinear function of another.

    ``x[axis] = u[axis] + f(u[source])``, all other axes pass through.
    Jacobian is unit lower-triangular, so log|det J| = 0.
    """

    f: Callable[[Array], Array]
    axis: int = 1
    source: int = 0

    def forward(self, u: Array) -> Array:
        return u.at[..., self.axis].add(self.f(u[..., self.source]))

    def inverse(self, x: Array) -> Array:
        return x.at[..., self.axis].add(-self.f(x[..., self.source]))

    def forward_log_det_jac(self, u: Array) -> Array:
        return jnp.zeros(u.shape[:-1])


@dataclass(frozen=True)
class Funnel(Bijector):
    """Hierarchical scale warp (Neal's funnel): ``x[axis] = u[axis] · exp(g(u[source]))``.

    ``g`` is an arbitrary log-scale function; typical choice is ``g = λ · u[source]``.
    log|det J| = g(u[source]).
    """

    g: Callable[[Array], Array]
    axis: int = 1
    source: int = 0

    def forward(self, u: Array) -> Array:
        return u.at[..., self.axis].multiply(jnp.exp(self.g(u[..., self.source])))

    def inverse(self, x: Array) -> Array:
        return x.at[..., self.axis].multiply(jnp.exp(-self.g(x[..., self.source])))

    def forward_log_det_jac(self, u: Array) -> Array:
        return self.g(u[..., self.source])


@dataclass(frozen=True)
class Softplus(Bijector):
    """Elementwise softplus on selected axes; maps R -> (0, infinity).

    Use to induce positive-orthant support (boundary effects).
    """

    axes: tuple[int, ...] = (0,)

    def forward(self, u: Array) -> Array:
        axes = jnp.array(self.axes)
        updated = jax.nn.softplus(u[..., axes])
        return u.at[..., axes].set(updated)

    def inverse(self, x: Array) -> Array:
        axes = jnp.array(self.axes)
        v = x[..., axes]
        inv = jnp.log(jnp.expm1(v))
        return x.at[..., axes].set(inv)

    def forward_log_det_jac(self, u: Array) -> Array:
        axes = jnp.array(self.axes)
        return jax.nn.log_sigmoid(u[..., axes]).sum(axis=-1)


@dataclass(frozen=True)
class Logit(Bijector):
    """Sigmoid on selected axes; maps R -> (0, 1). Use for bounded support."""

    axes: tuple[int, ...] = (0,)

    def forward(self, u: Array) -> Array:
        axes = jnp.array(self.axes)
        return u.at[..., axes].set(jax.nn.sigmoid(u[..., axes]))

    def inverse(self, x: Array) -> Array:
        axes = jnp.array(self.axes)
        v = jnp.clip(x[..., axes], 1e-6, 1.0 - 1e-6)
        return x.at[..., axes].set(jnp.log(v) - jnp.log1p(-v))

    def forward_log_det_jac(self, u: Array) -> Array:
        axes = jnp.array(self.axes)
        v = u[..., axes]
        return (jax.nn.log_sigmoid(v) + jax.nn.log_sigmoid(-v)).sum(axis=-1)
