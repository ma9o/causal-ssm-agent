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
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.shapes import Array, Float, FloatScalar, Int

AUX_JITTER = 1e-6


def _normalize_log_probs(
    logits: Float[Array, "*shape"], *, axis: int = -1
) -> Float[Array, "*shape"]:
    return logits - jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)


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
