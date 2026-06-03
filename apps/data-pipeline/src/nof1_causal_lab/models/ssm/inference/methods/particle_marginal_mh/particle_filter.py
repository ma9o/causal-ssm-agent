"""Bootstrap particle-filter likelihood estimator for PMMH."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _particle_ess_from_log_weights,
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
    trajectory_target = bundle["trajectory_target"]
    obs_increment_fn = bundle["observation_increment_log_prob_from_context_runtime_fn"]
    log_num_particles = jnp.log(jnp.asarray(num_particles, dtype=position.dtype))

    init_key, scan_key = random.split(key)
    particles0 = trajectory_target.sample_initial(
        init_key,
        context,
        sample_shape=(num_particles,),
    )
    raw_log_weights0 = jax.vmap(
        lambda particle: obs_increment_fn(
            context,
            particle,
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
        particles_t = trajectory_target.sample_transition(
            transition_key,
            context,
            ancestor_particles,
            time_idx,
        )
        raw_log_weights = jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
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
