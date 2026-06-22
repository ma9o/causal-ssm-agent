"""Numerical helpers shared by the MPGibbs latent smoothers."""

# References: docs/reference/bibliography.md — Särkkä (2013), Bayesian Filtering and
# Smoothing, for the Gaussian log-density and resampling primitives used here.
#
# Instrumented for runtime shape checks, hence no ``from __future__ import
# annotations`` (eager annotations keep the jaxtyping imports resolvable for
# beartype). ``contexts`` / ``obs_increment_fn`` are pytree/closure boundaries and
# stay unannotated by design (see :mod:`nof1_causal_lab.models.ssm.shapes`).

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.bundle import (
    AUX_JITTER,
)
from nof1_causal_lab.models.ssm.shapes import Array, Float, FloatScalar, Int, PRNGKeyArray


def _normalize_log_probs(
    logits: Float[Array, "*shape"], *, axis: int = -1
) -> Float[Array, "*shape"]:
    return logits - jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)


def _particle_ess_from_log_weights(
    log_weights: Float[Array, "*batch P"],
) -> Float[Array, "*batch"]:
    probabilities = jnp.exp(log_weights)
    return 1.0 / jnp.sum(probabilities * probabilities, axis=-1)


def _log_weight_range(log_weights: Float[Array, "*batch P"]) -> Float[Array, "*batch"]:
    return jnp.max(log_weights, axis=-1) - jnp.min(log_weights, axis=-1)


def _log_weight_variance(log_weights: Float[Array, "*batch P"]) -> Float[Array, "*batch"]:
    return jnp.var(log_weights, axis=-1)


def _categorical_entropy_from_log_probs(
    log_probs: Float[Array, "*batch N"],
) -> Float[Array, "*batch"]:
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def _categorical_max_prob_from_log_probs(
    log_probs: Float[Array, "*batch N"],
) -> Float[Array, "*batch"]:
    return jnp.max(jnp.exp(log_probs), axis=-1)


def _categorical_rows(key: PRNGKeyArray, logits: Float[Array, "R C"]) -> Int[Array, " R"]:
    keys = random.split(key, int(logits.shape[0]))
    return jax.vmap(lambda row_key, row_logits: random.categorical(row_key, row_logits))(
        keys,
        logits,
    ).astype(jnp.int32)


def _sample_gaussian_from_chol(
    key: PRNGKeyArray,
    mean: Float[Array, "*batch D"],
    chol: Float[Array, "*batch D D"],
) -> Float[Array, "*batch D"]:
    eps = random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + jnp.einsum("...ij,...j->...i", chol, eps)


def _cholesky_batch(covariances: Float[Array, "K D D"]) -> Float[Array, "K D D"]:
    return jax.vmap(
        lambda cov: jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=AUX_JITTER))
    )(covariances)


def _gaussian_log_prob_shared_cholesky(
    value: Float[Array, "*batch D"],
    mean: Float[Array, " D"],
    cholesky: Float[Array, "D D"],
    logdet: FloatScalar,
) -> Float[Array, "*batch"]:
    diff = value - mean
    dim = diff.shape[-1]
    flat_diff = jnp.reshape(diff, (-1, dim))
    whitened = jla.solve_triangular(cholesky, flat_diff.T, lower=True).T
    quadratic = jnp.reshape(
        jnp.sum(whitened * whitened, axis=-1),
        diff.shape[:-1],
    )
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + quadratic)


def _observation_log_probs_by_param(
    contexts,
    particles_t: Float[Array, "P D"],
    time_idx: Int[Array, ""],
    runtime_observations: Float[Array, "T M"],
    obs_increment_fn,
) -> Float[Array, "P K"]:
    def _one_param(context):
        return jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                time_idx,
                runtime_observations,
            )
        )(particles_t)

    return jnp.swapaxes(jax.vmap(_one_param)(contexts), 0, 1)


def _single_observation_log_probs_by_param(
    contexts,
    particle_t: Float[Array, " D"],
    time_idx: Int[Array, ""],
    runtime_observations: Float[Array, "T M"],
    obs_increment_fn,
) -> Float[Array, " K"]:
    return _observation_log_probs_by_param(
        contexts,
        particle_t[None, :],
        time_idx,
        runtime_observations,
        obs_increment_fn,
    )[0]


def _select_pytree(ensemble, index: Int[Array, "..."]):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], ensemble)
