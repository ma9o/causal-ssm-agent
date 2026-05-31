"""Builds the per-step :class:`SmootherContext` for MPGibbs latent smoothers."""

# References:
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf — Corenflos (2025), arXiv:2505.04611:
#     the per-step posterior mixture over the parameter ensemble that the smoothers
#     condition on.
#   docs/reference/bibliography.md — Särkkä (2013), Bayesian Filtering and Smoothing,
#     for the Gaussian transition / initial-state moments assembled here.

from __future__ import annotations

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsStatic,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _cholesky_batch,
    _gaussian_log_prob_shared_cholesky,
    _logdet_from_cholesky,
    _normalize_log_probs,
    _single_observation_log_probs_by_param,
    _transition_log_probs_by_param,
)


def build_smoother_context(
    static: MPGibbsStatic,
    state,
    parameter_particles: jnp.ndarray,
    label_correction: jnp.ndarray,
) -> SmootherContext:
    """Assemble per-step latent contexts and shared smoother helper closures."""
    latent_context_runtime_fn = static.latent_context_runtime_fn
    log_prior_unc_fn = static.log_prior_unc_fn
    initial_latent_moments_fn = static.initial_latent_moments_fn
    obs_increment_fn = static.obs_increment_fn
    trajectory_log_prob_fn = static.trajectory_log_prob_fn
    prior_terms_from_context_fn = static.prior_terms_from_context_fn
    initial_observation_auxiliary_fn = static.initial_observation_auxiliary_fn
    runtime_observations = static.runtime_observations
    runtime_times = static.runtime_times
    num_parameter_particles = static.num_parameter_particles

    x_ref = state.latent_trajectory
    latent_dtype = x_ref.dtype
    traj_dtype = state.trajectory_log_prob.dtype
    complete_dtype = state.complete_log_posterior.dtype
    num_steps = int(x_ref.shape[0])
    block_size = min(int(static.latent_block_size), num_steps)
    num_blocks = (num_steps + block_size - 1) // block_size
    num_free_particles = static.num_particles - 1

    with jax.named_scope("build_contexts"):
        contexts = jax.vmap(lambda z: latent_context_runtime_fn(z, runtime_times))(
            parameter_particles
        )
        parameter_log_probs = (
            jax.vmap(log_prior_unc_fn)(parameter_particles) + label_correction
        ).astype(traj_dtype)
        initial_label_log_probs = _normalize_log_probs(parameter_log_probs)

        init_means, init_covs = jax.vmap(initial_latent_moments_fn)(contexts)
        init_chols = _cholesky_batch(init_covs)
        init_logdets = _logdet_from_cholesky(init_chols)
        transition_chols = _cholesky_batch(contexts.Qd)
        transition_logdets = _logdet_from_cholesky(transition_chols)

    def _transition_log_probs_from_fixed_prev(
        prev_particle: jnp.ndarray,
        particles_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return _transition_log_probs_by_param(
            contexts,
            transition_chols,
            transition_logdets,
            jnp.broadcast_to(prev_particle, particles_t.shape),
            particles_t,
            time_idx,
        )

    def _initial_label_log_probs_for_particle(particle0: jnp.ndarray) -> jnp.ndarray:
        initial_prior_lp_by_param = jax.vmap(
            lambda mean, chol, logdet: _gaussian_log_prob_shared_cholesky(
                particle0[None, :],
                mean,
                chol,
                logdet,
            )[0]
        )(init_means, init_chols, init_logdets)
        initial_obs_lp_by_param = _single_observation_log_probs_by_param(
            contexts,
            particle0,
            state.observation_auxiliary,
            jnp.asarray(0, dtype=jnp.int32),
            runtime_observations,
            obs_increment_fn,
        )
        return _normalize_log_probs(
            initial_label_log_probs + initial_prior_lp_by_param + initial_obs_lp_by_param
        ).astype(traj_dtype)

    def _segment_terminal_label_log_probs(
        prefix_label_log_probs: jnp.ndarray,
        segment_path: jnp.ndarray,
        previous_particle: jnp.ndarray,
        block_start: int,
    ) -> jnp.ndarray:
        if block_start == 0:
            label_log_probs0 = _initial_label_log_probs_for_particle(segment_path[0])
        else:
            time0 = jnp.asarray(block_start, dtype=jnp.int32)
            transition_lp0 = _transition_log_probs_from_fixed_prev(
                previous_particle,
                segment_path[0][None, :],
                time0,
            )[0]
            obs_lp0 = _single_observation_log_probs_by_param(
                contexts,
                segment_path[0],
                state.observation_auxiliary,
                time0,
                runtime_observations,
                obs_increment_fn,
            )
            label_log_probs0 = _normalize_log_probs(
                prefix_label_log_probs + transition_lp0 + obs_lp0
            ).astype(traj_dtype)

        def _scan_segment(carry, offset):
            label_log_probs, prev_particle = carry
            time_idx = jnp.asarray(block_start, dtype=jnp.int32) + offset
            particle_t = segment_path[offset]
            transition_lp = _transition_log_probs_from_fixed_prev(
                prev_particle,
                particle_t[None, :],
                time_idx,
            )[0]
            obs_lp = _single_observation_log_probs_by_param(
                contexts,
                particle_t,
                state.observation_auxiliary,
                time_idx,
                runtime_observations,
                obs_increment_fn,
            )
            next_label_log_probs = _normalize_log_probs(
                label_log_probs + transition_lp + obs_lp
            ).astype(traj_dtype)
            return (next_label_log_probs, particle_t), None

        if int(segment_path.shape[0]) > 1:
            (label_log_probs, _), _ = jax.lax.scan(
                _scan_segment,
                (label_log_probs0, segment_path[0]),
                jnp.arange(1, segment_path.shape[0], dtype=jnp.int32),
            )
            return label_log_probs
        return label_log_probs0

    def _path_future_tail_log_probs(path: jnp.ndarray) -> jnp.ndarray:
        zeros = jnp.zeros((num_parameter_particles,), dtype=traj_dtype)

        def _scan_tail(tail_log_probs, time_idx):
            prev_particle = path[time_idx]
            next_particle = path[time_idx + 1]
            transition_lp = _transition_log_probs_from_fixed_prev(
                prev_particle,
                next_particle[None, :],
                time_idx + 1,
            )[0]
            obs_lp = _single_observation_log_probs_by_param(
                contexts,
                next_particle,
                state.observation_auxiliary,
                time_idx + 1,
                runtime_observations,
                obs_increment_fn,
            )
            next_tail = (transition_lp + obs_lp + tail_log_probs).astype(traj_dtype)
            return next_tail, next_tail

        if num_steps > 1:
            _, reversed_tail = jax.lax.scan(
                _scan_tail,
                zeros,
                jnp.arange(num_steps - 2, -1, -1, dtype=jnp.int32),
            )
            tail = jnp.flip(reversed_tail, axis=0)
            return jnp.concatenate([tail, zeros[None, :]], axis=0)
        return zeros[None, :]

    def _trajectory_label_log_probs(path: jnp.ndarray) -> jnp.ndarray:
        def _one_context(context):
            observation_auxiliary = initial_observation_auxiliary_fn(
                context,
                path,
                runtime_observations,
            )
            prior_terms = prior_terms_from_context_fn(context)
            return trajectory_log_prob_fn(
                context,
                path,
                observation_auxiliary,
                runtime_observations,
                prior_terms=prior_terms,
            )

        trajectory_log_probs = jax.vmap(_one_context)(contexts).astype(traj_dtype)
        return _normalize_log_probs(parameter_log_probs + trajectory_log_probs).astype(traj_dtype)

    return SmootherContext(
        contexts=contexts,
        parameter_particles=parameter_particles,
        parameter_log_probs=parameter_log_probs,
        initial_label_log_probs=initial_label_log_probs,
        init_means=init_means,
        init_chols=init_chols,
        init_logdets=init_logdets,
        transition_chols=transition_chols,
        transition_logdets=transition_logdets,
        num_steps=num_steps,
        num_free_particles=num_free_particles,
        num_parameter_particles=num_parameter_particles,
        block_size=block_size,
        num_blocks=num_blocks,
        latent_dtype=latent_dtype,
        traj_dtype=traj_dtype,
        complete_dtype=complete_dtype,
        state=state,
        obs_increment_fn=obs_increment_fn,
        runtime_observations=runtime_observations,
        initial_observation_auxiliary_fn=initial_observation_auxiliary_fn,
        trajectory_log_prob_fn=trajectory_log_prob_fn,
        prior_terms_from_context_fn=prior_terms_from_context_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        mgrad_latent_kernel=static.mgrad_latent_kernel,
        amala_delta=jnp.asarray(state.latent_delta, dtype=latent_dtype),
        amala_kappa=static.amala_kappa,
        amala_grad_clip=static.amala_grad_clip,
        dsmc_leaf_proposal=static.dsmc_leaf_proposal,
        diagnostic_metrics=static.diagnostic_metrics,
        transition_log_probs_from_fixed_prev=_transition_log_probs_from_fixed_prev,
        segment_terminal_label_log_probs=_segment_terminal_label_log_probs,
        path_future_tail_log_probs=_path_future_tail_log_probs,
        trajectory_label_log_probs=_trajectory_label_log_probs,
    )
