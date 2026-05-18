"""Shared observation-kernel helpers used by family metadata and kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from .emissions import categorical_moments, ordered_logistic_moments

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_variance_poisson() -> Callable:
    """Poisson: Var(Y) = lambda = mean."""

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(jnp.maximum(mean, 1e-8))

    return variance_fn


def _make_variance_negative_binomial(r: float) -> Callable:
    """NegBin: Var(Y) = mu + mu^2/r."""

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu + mu**2 / (r + 1e-8))

    return variance_fn


def _make_variance_gamma(shape: float) -> Callable:
    """Gamma: Var(Y) = mean^2 / shape."""

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu**2 / (shape + 1e-8))

    return variance_fn


def _make_variance_bernoulli() -> Callable:
    """Bernoulli: Var(Y) = p(1-p)."""

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p))

    return variance_fn


def _make_variance_beta(concentration: float) -> Callable:
    """Beta: Var(Y) = p(1-p) / (phi + 1)."""

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p) / (concentration + 1.0))

    return variance_fn


def _make_variance_identity(manifest_cov: jnp.ndarray) -> Callable:
    """Gaussian/Student-t: pseudo-R = measurement covariance (constant)."""

    def variance_fn(_mean: jnp.ndarray) -> jnp.ndarray:
        return manifest_cov

    return variance_fn


def _make_discrete_response_ordered_logistic(
    cutpoints: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> Callable:
    def response_fn(eta: jnp.ndarray) -> jnp.ndarray:
        mean, _variance = ordered_logistic_moments(eta, cutpoints, level_counts)
        return mean

    return response_fn


def _make_discrete_response_categorical(
    intercepts: jnp.ndarray,
    slopes: jnp.ndarray,
    level_counts: jnp.ndarray,
) -> Callable:
    def response_fn(eta: jnp.ndarray) -> jnp.ndarray:
        mean, _variance = categorical_moments(eta, intercepts, slopes, level_counts)
        return mean

    return response_fn


def _make_discrete_variance_from_moments(moment_fn: Callable) -> Callable:
    def variance_fn(eta: jnp.ndarray) -> jnp.ndarray:
        _mean, variance = moment_fn(eta)
        return jnp.diag(jnp.maximum(variance, 1e-8))

    return variance_fn
