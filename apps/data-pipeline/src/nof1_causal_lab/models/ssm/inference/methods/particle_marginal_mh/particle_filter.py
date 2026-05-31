"""Bootstrap particle-filter likelihood estimator for PMMH."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.bundle import AUX_JITTER
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _cholesky_batch,
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _particle_ess_from_log_weights,
    _sample_gaussian_from_chol,
)


def _safe_normalize_log_weights(raw_log_weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_norm = jax.scipy.special.logsumexp(raw_log_weights)
    fallback = -jnp.log(jnp.asarray(raw_log_weights.shape[0], dtype=raw_log_weights.dtype))
    log_weights = jnp.where(
        jnp.isfinite(log_norm),
        _normalize_log_probs(raw_log_weights),
        jnp.full_like(raw_log_weights, fallback),
    )
    return log_weights, log_norm


def estimate_bootstrap_log_likelihood(
    key: jnp.ndarray,
    position: jnp.ndarray,
    *,
    bundle: dict[str, Any],
    num_particles: int,
    return_particle_diagnostics: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Estimate ``log p(y | theta)`` with a bootstrap particle filter.

    The returned log value is the log of an unbiased likelihood estimator for
    the discretized runtime state-space model. The log itself is biased, as in
    standard PMMH.
    """
    runtime_observations = bundle["observations"]
    runtime_times = bundle["times"]
    context = bundle["latent_context_runtime_fn"](position, runtime_times)
    latent_template = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_runtime_fn"](
        context,
        latent_template,
        runtime_observations,
    )
    obs_increment_fn = bundle["observation_increment_log_prob_conditioned_from_context_runtime_fn"]
    initial_moments_fn = bundle["initial_latent_moments_from_context_fn"]
    initial_mean, initial_cov = initial_moments_fn(context)
    initial_chol = jnp.linalg.cholesky(symmetrize_with_jitter(initial_cov, jitter=AUX_JITTER))
    transition_chols = _cholesky_batch(context.Qd)
    log_num_particles = jnp.log(jnp.asarray(num_particles, dtype=position.dtype))

    init_key, scan_key = random.split(key)
    init_mean_particles = jnp.broadcast_to(initial_mean, (num_particles, initial_mean.shape[0]))
    init_chol_particles = jnp.broadcast_to(
        initial_chol,
        (num_particles, initial_chol.shape[0], initial_chol.shape[1]),
    )
    particles0 = _sample_gaussian_from_chol(init_key, init_mean_particles, init_chol_particles)
    raw_log_weights0 = jax.vmap(
        lambda particle: obs_increment_fn(
            context,
            particle,
            observation_auxiliary,
            jnp.asarray(0, dtype=jnp.int32),
            runtime_observations,
        )
    )(particles0)
    log_weights0, log_norm0 = _safe_normalize_log_weights(raw_log_weights0)
    log_likelihood0 = log_norm0 - log_num_particles
    log_likelihood0 = jnp.where(jnp.isfinite(log_norm0), log_likelihood0, -jnp.inf)
    ess0 = _particle_ess_from_log_weights(log_weights0)
    range0 = _log_weight_range(log_weights0)
    variance0 = _log_weight_variance(log_weights0)

    step_keys = random.split(scan_key, max(int(runtime_observations.shape[0]) - 1, 1))

    def _scan_step(carry, inputs):
        particles_prev, log_weights_prev, log_likelihood_prev = carry
        time_idx, step_key = inputs
        resample_key, transition_key = random.split(step_key)
        ancestors = random.categorical(
            resample_key,
            log_weights_prev,
            shape=(num_particles,),
        ).astype(jnp.int32)
        ancestor_particles = jnp.take(particles_prev, ancestors, axis=0)
        means = ancestor_particles @ context.Ad[time_idx].T + context.cd[time_idx]
        chol_t = jnp.broadcast_to(
            transition_chols[time_idx],
            (num_particles, transition_chols.shape[-2], transition_chols.shape[-1]),
        )
        particles_t = _sample_gaussian_from_chol(transition_key, means, chol_t)
        raw_log_weights = jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                observation_auxiliary,
                time_idx,
                runtime_observations,
            )
        )(particles_t)
        log_weights, log_norm = _safe_normalize_log_weights(raw_log_weights)
        increment = log_norm - log_num_particles
        increment = jnp.where(jnp.isfinite(log_norm), increment, -jnp.inf)
        log_likelihood = log_likelihood_prev + increment
        diagnostics_t = (
            _particle_ess_from_log_weights(log_weights),
            _log_weight_range(log_weights),
            _log_weight_variance(log_weights),
            increment,
        )
        return (particles_t, log_weights, log_likelihood), diagnostics_t

    if int(runtime_observations.shape[0]) > 1:
        (_, _, log_likelihood), (ess_tail, range_tail, variance_tail, increment_tail) = (
            jax.lax.scan(
                _scan_step,
                (particles0, log_weights0, log_likelihood0),
                (
                    jnp.arange(1, runtime_observations.shape[0], dtype=jnp.int32),
                    step_keys[: int(runtime_observations.shape[0]) - 1],
                ),
            )
        )
        ess_by_t = jnp.concatenate([ess0[None], ess_tail], axis=0)
        range_by_t = jnp.concatenate([range0[None], range_tail], axis=0)
        variance_by_t = jnp.concatenate([variance0[None], variance_tail], axis=0)
        increments = jnp.concatenate([log_likelihood0[None], increment_tail], axis=0)
    else:
        log_likelihood = log_likelihood0
        ess_by_t = ess0[None]
        range_by_t = range0[None]
        variance_by_t = variance0[None]
        increments = log_likelihood0[None]

    diagnostics = {
        "pf_ess_min": jnp.min(ess_by_t),
        "pf_ess_mean": jnp.mean(ess_by_t),
        "pf_log_weight_range_max": jnp.max(range_by_t),
        "pf_log_weight_variance_mean": jnp.mean(variance_by_t),
        "pf_log_likelihood_increment_variance": jnp.var(increments),
    }
    if return_particle_diagnostics:
        diagnostics.update(
            {
                "pf_ess_by_t": ess_by_t,
                "pf_log_weight_range_by_t": range_by_t,
                "pf_log_weight_variance_by_t": variance_by_t,
                "pf_log_likelihood_increment_by_t": increments,
            }
        )
    return log_likelihood.astype(position.dtype), diagnostics
