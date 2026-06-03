"""Euler-Maruyama transition target utilities for nonlinear vector fields."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField

_LOG_2PI = math.log(2.0 * math.pi)


def _forcing_for_time(context, time_idx: Array) -> Array:
    input_effect = context.input_effect
    if input_effect is None or int(input_effect.shape[1]) == 0:
        return jnp.zeros((context.init_mean.shape[0],), dtype=context.init_mean.dtype)
    if context.transition_inputs is None:
        raise ValueError("Euler-Maruyama transition target requires transition_inputs.")
    return input_effect @ context.transition_inputs[time_idx]


def transition_mean(
    vector_field: VectorField,
    context,
    previous_state: Array,
    time_idx: Array,
) -> Array:
    """Mean of one Euler-Maruyama step into ``time_idx``."""
    args = VectorFieldArgs(params=context.vf_params, intervention=Intervention.none())
    dt = context.time_intervals[time_idx]
    drift = vector_field(context.runtime_times[time_idx], previous_state, args)
    forcing = _forcing_for_time(context, time_idx)
    return previous_state + dt * (drift + forcing)


def transition_cov(context, time_idx: Array) -> Array:
    """Covariance of one Euler-Maruyama step into ``time_idx``."""
    dt = context.time_intervals[time_idx]
    return symmetrize_with_jitter(context.diffusion_cov * dt)


def _log_prob_with_chol(value: Array, mean: Array, chol: Array) -> Array:
    residual = value - mean
    whitened = jla.solve_triangular(chol, residual, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = value.shape[-1]
    return -0.5 * (dim * _LOG_2PI + logdet + jnp.sum(whitened * whitened))


def _diffusion_precision_logdet(context) -> tuple[Array, Array]:
    """Precision ``Q^-1`` and log-det of the state-independent diffusion covariance ``Q``.

    ``Q = context.diffusion_cov`` is time-invariant, so it is factored once per context
    (callers that scan / vmap over irregular time hoist this out). The per-step Mahalanobis
    term ``r^T (Q·dt)^-1 r = (r^T Q^-1 r)/dt`` then lowers to a small matmul (GEMM) rather
    than a triangular solve per step/seam, and ``logdet(Q·dt) = logdet(Q) + dim·log(dt)``.
    Jitter is applied to ``Q`` (a ``~1e-8`` difference) so the factor is step-independent.
    """
    q = symmetrize_with_jitter(context.diffusion_cov)
    chol = jnp.linalg.cholesky(q)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    precision = jla.cho_solve((chol, True), jnp.eye(q.shape[-1], dtype=q.dtype))
    return precision, logdet


def _scaled_transition_log_prob(
    value: Array, mean: Array, precision: Array, logdet_q: Array, dt: Array
) -> Array:
    """``log N(value; mean, Q·dt)`` from the precision ``Q^-1`` and scalar ``dt``.

    Mahalanobis term via a GEMM (``r^T Q^-1 r / dt``); log-det via ``logdet_q + dim·log(dt)``.
    """
    residual = value - mean
    dim = value.shape[-1]
    quadratic = (residual @ precision @ residual) / dt
    logdet = logdet_q + dim * jnp.log(dt)
    return -0.5 * (dim * _LOG_2PI + logdet + quadratic)


def initial_moments(vector_field: VectorField, context) -> tuple[Array, Array]:
    """Initial latent marginal under one Euler-Maruyama step from ``init_mean``."""
    mean = transition_mean(
        vector_field, context, context.init_mean, jnp.asarray(0, dtype=jnp.int32)
    )
    cov = symmetrize_with_jitter(
        context.init_cov + context.diffusion_cov * context.time_intervals[0]
    )
    return mean, cov


def initial_log_prob(vector_field: VectorField, context, particle0: Array) -> Array:
    mean, cov = initial_moments(vector_field, context)
    return _log_prob_with_chol(particle0, mean, jnp.linalg.cholesky(cov))


def predictive_latent_init(vector_field: VectorField, context) -> Array:
    """Deterministic Euler-Maruyama mean path used to initialize latent trajectories."""

    def _step(previous_state, time_idx):
        current_state = transition_mean(vector_field, context, previous_state, time_idx)
        return current_state, current_state

    _, trajectory = jax.lax.scan(
        _step,
        context.init_mean,
        jnp.arange(context.runtime_times.shape[0], dtype=jnp.int32),
    )
    return trajectory


def transition_log_prob(
    vector_field: VectorField,
    context,
    previous_state: Array,
    current_state: Array,
    time_idx: Array,
) -> Array:
    mean = transition_mean(vector_field, context, previous_state, time_idx)
    precision, logdet_q = _diffusion_precision_logdet(context)
    dt = context.time_intervals[time_idx]
    return _scaled_transition_log_prob(current_state, mean, precision, logdet_q, dt)


def transition_log_probs_for_pairs(
    vector_field: VectorField,
    context,
    previous_states: Array,
    current_states: Array,
    time_idx: Array,
) -> Array:
    """Log probabilities for aligned previous/current particle pairs."""
    return jax.vmap(
        lambda previous_state, current_state: transition_log_prob(
            vector_field,
            context,
            previous_state,
            current_state,
            time_idx,
        )
    )(previous_states, current_states)


def pairwise_transition_log_probs(
    vector_field: VectorField,
    context,
    previous_states: Array,
    current_states: Array,
    time_idx: Array,
) -> Array:
    """Return all pairwise transition scores with shape ``(n_prev, n_curr)``."""
    means = jax.vmap(
        lambda previous_state: transition_mean(vector_field, context, previous_state, time_idx)
    )(previous_states)
    precision, logdet_q = _diffusion_precision_logdet(context)
    dt = context.time_intervals[time_idx]
    diff = current_states[None, :, :] - means[:, None, :]
    # Mahalanobis over all (n_prev, n_curr) pairs as a batched GEMM, not a triangular solve.
    quadratic = jnp.sum((diff @ precision) * diff, axis=-1) / dt
    dim = previous_states.shape[-1]
    logdet = logdet_q + dim * jnp.log(dt)
    return -0.5 * (dim * _LOG_2PI + logdet + quadratic)


def trajectory_prior_log_prob(
    vector_field: VectorField, context, latent_trajectory: Array
) -> Array:
    """Euler-Maruyama latent path prior log-density at observation-grid states."""
    initial_lp = initial_log_prob(vector_field, context, latent_trajectory[0])
    if int(latent_trajectory.shape[0]) <= 1:
        return initial_lp

    # The transition terms are independent given a fixed trajectory, so evaluate them
    # vectorized over time (one batched kernel + a reduction) instead of a sequential
    # scan: fewer/larger kernels, and the parameter gradient backprops through a batch
    # rather than a T-step reverse scan. Q is factored once (loop-invariant).
    precision, logdet_q = _diffusion_precision_logdet(context)
    steps = jnp.arange(1, latent_trajectory.shape[0], dtype=jnp.int32)

    def _term(time_idx):
        mean = transition_mean(vector_field, context, latent_trajectory[time_idx - 1], time_idx)
        dt = context.time_intervals[time_idx]
        return _scaled_transition_log_prob(
            latent_trajectory[time_idx], mean, precision, logdet_q, dt
        )

    return initial_lp + jnp.sum(jax.vmap(_term)(steps))
