"""Base distributions: independent 1D components with known sample + log_prob."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
from jax.scipy.stats import cauchy, laplace, norm
from jax.scipy.stats import t as student_t

Array = jax.Array
PRNGKey = jax.Array


class Base:
    """Product distribution over R^D with independent, identically parametrised components."""

    dim: int

    def log_prob(self, x: Array) -> Array:
        raise NotImplementedError

    def sample(self, key: PRNGKey, n: int) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class Gaussian(Base):
    dim: int = 2
    loc: float = 0.0
    scale: float = 1.0

    @override
    def log_prob(self, x: Array) -> Array:
        return norm.logpdf(x, loc=self.loc, scale=self.scale).sum(axis=-1)

    @override
    def sample(self, key: PRNGKey, n: int) -> Array:
        return self.loc + self.scale * jax.random.normal(key, (n, self.dim))


@dataclass(frozen=True)
class StudentT(Base):
    dim: int = 2
    df: float = 3.0
    loc: float = 0.0
    scale: float = 1.0

    @override
    def log_prob(self, x: Array) -> Array:
        return student_t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale).sum(axis=-1)

    @override
    def sample(self, key: PRNGKey, n: int) -> Array:
        return self.loc + self.scale * jax.random.t(key, self.df, (n, self.dim))


@dataclass(frozen=True)
class Laplace(Base):
    dim: int = 2
    loc: float = 0.0
    scale: float = 1.0

    @override
    def log_prob(self, x: Array) -> Array:
        return laplace.logpdf(x, loc=self.loc, scale=self.scale).sum(axis=-1)

    @override
    def sample(self, key: PRNGKey, n: int) -> Array:
        return self.loc + self.scale * jax.random.laplace(key, (n, self.dim))


@dataclass(frozen=True)
class Cauchy(Base):
    dim: int = 2
    loc: float = 0.0
    scale: float = 1.0

    @override
    def log_prob(self, x: Array) -> Array:
        return cauchy.logpdf(x, loc=self.loc, scale=self.scale).sum(axis=-1)

    @override
    def sample(self, key: PRNGKey, n: int) -> Array:
        return self.loc + self.scale * jax.random.cauchy(key, (n, self.dim))
