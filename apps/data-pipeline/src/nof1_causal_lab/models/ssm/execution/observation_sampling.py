"""Low-level random draws shared by predictor- and mean-space observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.execution.contracts import NUMERICAL_EPSILON


def sample_student_t_from_location(
    key: jax.Array,
    location: jnp.ndarray,
    scale: jnp.ndarray,
    df: jnp.ndarray | float,
) -> jnp.ndarray:
    """Draw an independent Student-t variate for every location entry."""
    key_num, key_den = jax.random.split(key)
    z = jax.random.normal(key_num, location.shape)
    chi2 = 2.0 * jax.random.gamma(key_den, df / 2.0, shape=location.shape)
    t_value = z * jnp.sqrt(df / jnp.maximum(chi2, NUMERICAL_EPSILON))
    return location + scale * t_value


def sample_poisson_from_mean(key: jax.Array, mean: jnp.ndarray) -> jnp.ndarray:
    """Draw Poisson observations from a valid mean parameter."""
    return jax.random.poisson(key, mean).astype(jnp.float32)


def sample_gamma_from_mean(
    key: jax.Array,
    mean: jnp.ndarray,
    shape_parameter: jnp.ndarray | float,
    *,
    denominator_floor: float = NUMERICAL_EPSILON,
    scale_floor: float | None = None,
) -> jnp.ndarray:
    """Draw Gamma observations parameterized by mean and shape."""
    scale = mean / jnp.maximum(shape_parameter, denominator_floor)
    if scale_floor is not None:
        scale = jnp.maximum(scale, scale_floor)
    return jax.random.gamma(key, shape_parameter, shape=mean.shape) * scale


def sample_bernoulli_from_mean(key: jax.Array, mean: jnp.ndarray) -> jnp.ndarray:
    """Draw Bernoulli observations from a valid probability."""
    return jax.random.bernoulli(key, mean).astype(jnp.float32)


def sample_negative_binomial_from_mean(
    key: jax.Array,
    mean: jnp.ndarray,
    dispersion: jnp.ndarray | float,
) -> jnp.ndarray:
    """Draw a mean/dispersion negative binomial via a Gamma-Poisson mixture."""
    key_gamma, key_poisson = jax.random.split(key)
    gamma_draw = (
        jax.random.gamma(key_gamma, dispersion, shape=mean.shape)
        * mean
        / jnp.maximum(dispersion, 1e-8)
    )
    poisson_rate = jnp.where(
        mean == 0.0,
        0.0,
        jnp.maximum(gamma_draw, NUMERICAL_EPSILON),
    )
    return jax.random.poisson(
        key_poisson,
        poisson_rate,
    ).astype(jnp.float32)


def sample_beta_from_mean(
    key: jax.Array,
    mean: jnp.ndarray,
    concentration: jnp.ndarray | float,
) -> jnp.ndarray:
    """Draw a Beta observation from its mean and concentration."""
    alpha = jnp.maximum(mean * concentration, 1e-4)
    beta_parameter = jnp.maximum((1.0 - mean) * concentration, 1e-4)
    key_alpha, key_beta = jax.random.split(key)
    gamma_alpha = jax.random.gamma(key_alpha, alpha)
    gamma_beta = jax.random.gamma(key_beta, beta_parameter)
    return gamma_alpha / jnp.maximum(gamma_alpha + gamma_beta, NUMERICAL_EPSILON)
