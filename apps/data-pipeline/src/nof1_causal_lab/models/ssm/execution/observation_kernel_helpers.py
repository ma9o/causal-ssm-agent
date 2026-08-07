"""Shared observation-family moment helpers."""

from collections.abc import Callable

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.shapes import Array, Float, Int

from .emissions import categorical_moments, ordered_logistic_moments

type VarianceFn = Callable[[Float[Array, " M"]], Float[Array, "M M"]]
type ResponseFn = Callable[[Float[Array, " M"]], Float[Array, " M"]]
type MomentFn = Callable[
    [Float[Array, " M"]],
    tuple[Float[Array, " M"], Float[Array, " M"]],
]


def _make_variance_poisson() -> VarianceFn:
    """Poisson: Var(Y) = lambda = mean."""

    def variance_fn(mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        return jnp.diag(jnp.maximum(mean, 1e-8))

    return variance_fn


def _make_variance_negative_binomial(r: float | Float[Array, ""]) -> VarianceFn:
    """NegBin: Var(Y) = mu + mu^2/r."""

    def variance_fn(mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu + mu**2 / (r + 1e-8))

    return variance_fn


def _make_variance_gamma(shape: float | Float[Array, ""]) -> VarianceFn:
    """Gamma: Var(Y) = mean^2 / shape."""

    def variance_fn(mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu**2 / (shape + 1e-8))

    return variance_fn


def _make_variance_bernoulli() -> VarianceFn:
    """Bernoulli: Var(Y) = p(1-p)."""

    def variance_fn(mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p))

    return variance_fn


def _make_variance_beta(concentration: float | Float[Array, ""]) -> VarianceFn:
    """Beta: Var(Y) = p(1-p) / (phi + 1)."""

    def variance_fn(mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p) / (concentration + 1.0))

    return variance_fn


def _make_variance_identity(manifest_cov: Float[Array, "M M"]) -> VarianceFn:
    """Gaussian/Student-t: pseudo-R = measurement covariance (constant)."""

    def variance_fn(_mean: Float[Array, " M"]) -> Float[Array, "M M"]:
        return manifest_cov

    return variance_fn


def _make_discrete_response_ordered_logistic(
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> ResponseFn:
    def response_fn(eta: Float[Array, " M"]) -> Float[Array, " M"]:
        mean, _variance = ordered_logistic_moments(eta, cutpoints, level_counts)
        return mean

    return response_fn


def _make_discrete_response_categorical(
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> ResponseFn:
    def response_fn(eta: Float[Array, " M"]) -> Float[Array, " M"]:
        mean, _variance = categorical_moments(eta, intercepts, slopes, level_counts)
        return mean

    return response_fn


def _make_discrete_variance_from_moments(moment_fn: MomentFn) -> VarianceFn:
    def variance_fn(eta: Float[Array, " M"]) -> Float[Array, "M M"]:
        _mean, variance = moment_fn(eta)
        return jnp.diag(jnp.maximum(variance, 1e-8))

    return variance_fn
