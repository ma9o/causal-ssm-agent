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

from nof1_causal_lab.models.ssm.covariance_utils import logdet_from_cholesky
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsStatic,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _cholesky_batch,
    _gaussian_log_prob_shared_cholesky,
    _normalize_log_probs,
    _single_observation_log_probs_by_param,
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
    runtime_observations = static.runtime_observations
    runtime_times = static.runtime_times
    num_parameter_particles = static.num_parameter_particles
    transition_initial_log_prob_fn = static.transition_initial_log_prob_fn
    transition_log_prob_fn = static.transition_log_prob_fn
    transition_log_probs_for_pairs_fn = static.transition_log_probs_for_pairs_fn
    transition_pairwise_log_probs_fn = static.transition_pairwise_log_probs_fn
    transition_sample_fn = static.transition_sample_fn

    x_ref = state.latent_trajectory
    latent_dtype = x_ref.dtype
    traj_dtype = state.trajectory_log_prob.dtype
    complete_dtype = state.complete_log_posterior.dtype
    num_steps = int(x_ref.shape[0])
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
        init_logdets = logdet_from_cholesky(init_chols)

    def _transition_log_probs_from_fixed_prev(
        prev_particle: jnp.ndarray,
        particles_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        previous_particles = jnp.broadcast_to(prev_particle, particles_t.shape)
        per_param = jax.vmap(
            lambda context: transition_log_probs_for_pairs_fn(
                context,
                previous_particles,
                particles_t,
                time_idx,
            )
        )(contexts)
        return jnp.swapaxes(per_param, 0, 1).astype(traj_dtype)

    def _transition_log_probs_by_param(
        prev_particles: jnp.ndarray,
        particles_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        per_param = jax.vmap(
            lambda context: transition_log_probs_for_pairs_fn(
                context,
                prev_particles,
                particles_t,
                time_idx,
            )
        )(contexts)
        return jnp.swapaxes(per_param, 0, 1).astype(traj_dtype)

    def _transition_log_probs_to_next_by_param(
        prev_particles: jnp.ndarray,
        next_particle: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        next_particles = jnp.broadcast_to(next_particle, prev_particles.shape)
        per_param = jax.vmap(
            lambda context: transition_log_probs_for_pairs_fn(
                context,
                prev_particles,
                next_particles,
                time_idx,
            )
        )(contexts)
        return jnp.swapaxes(per_param, 0, 1).astype(traj_dtype)

    def _sample_transition_by_label(
        key: jnp.ndarray,
        prev_particles: jnp.ndarray,
        labels: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        param_keys = jax.random.split(key, num_parameter_particles)
        per_param = jax.vmap(
            lambda context, param_key: transition_sample_fn(
                param_key,
                context,
                prev_particles,
                time_idx,
            )
        )(contexts, param_keys)
        return per_param[labels, jnp.arange(prev_particles.shape[0])].astype(prev_particles.dtype)

    def _initial_value_grad_by_param(particle0: jnp.ndarray):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_initial_log_prob_fn(context, particle)
            )(particle0)

        log_prob, grad = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad.astype(latent_dtype)

    def _transition_current_value_grad_by_param(
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_log_prob_fn(
                    context,
                    prev_particle,
                    particle,
                    time_idx,
                )
            )(particle_t)

        log_prob, grad_current = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad_current.astype(latent_dtype)

    def _transition_next_value_grad_by_param(
        particle_t: jnp.ndarray,
        next_particle: jnp.ndarray,
        next_time_idx: jnp.ndarray,
    ):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_log_prob_fn(
                    context,
                    particle,
                    next_particle,
                    next_time_idx,
                )
            )(particle_t)

        log_prob, grad_prev = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad_prev.astype(latent_dtype)

    def _selected_transition_log_probs(
        prev_particles: jnp.ndarray,
        next_particles: jnp.ndarray,
        seam: jnp.ndarray,
    ) -> jnp.ndarray:
        seam_clamped = jnp.minimum(seam, num_steps - 1)
        real_seam = seam < num_steps
        per_param = jax.vmap(
            lambda context: transition_log_probs_for_pairs_fn(
                context,
                prev_particles,
                next_particles,
                seam_clamped,
            )
        )(contexts)
        return jnp.where(real_seam, jnp.swapaxes(per_param, 0, 1), 0.0).astype(traj_dtype)

    def _pairwise_transition_log_probs(
        prev_particles: jnp.ndarray,
        next_particles: jnp.ndarray,
        seam: jnp.ndarray,
    ) -> jnp.ndarray:
        seam_clamped = jnp.minimum(seam, num_steps - 1)
        real_seam = seam < num_steps
        per_param = jax.vmap(
            lambda context: transition_pairwise_log_probs_fn(
                context,
                prev_particles,
                next_particles,
                seam_clamped,
            )
        )(contexts)
        transition_lp = jnp.moveaxis(per_param, 0, -1)
        return jnp.where(real_seam, transition_lp, 0.0).astype(traj_dtype)

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
            prior_terms = prior_terms_from_context_fn(context)
            return trajectory_log_prob_fn(
                context,
                path,
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
        num_steps=num_steps,
        num_free_particles=num_free_particles,
        num_parameter_particles=num_parameter_particles,
        latent_dtype=latent_dtype,
        traj_dtype=traj_dtype,
        complete_dtype=complete_dtype,
        obs_increment_fn=obs_increment_fn,
        runtime_observations=runtime_observations,
        trajectory_log_prob_fn=trajectory_log_prob_fn,
        prior_terms_from_context_fn=prior_terms_from_context_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        amala_delta=jnp.asarray(state.latent_delta, dtype=latent_dtype),
        amala_kappa=static.amala_kappa,
        amala_grad_clip=static.amala_grad_clip,
        dsmc_leaf_proposal=static.dsmc_leaf_proposal,
        latent_block_coords=static.latent_block_coords,
        paid_mix_z_weight=static.paid_mix_z_weight,
        paid_mix_pilot_weight=static.paid_mix_pilot_weight,
        pilot_means=(
            None
            if static.pilot_means is None
            else jnp.asarray(static.pilot_means, dtype=latent_dtype)
        ),
        pilot_vars=(
            None
            if static.pilot_vars is None
            else jnp.asarray(static.pilot_vars, dtype=latent_dtype)
        ),
        pilot_wide_vars=(
            None
            if static.pilot_wide_vars is None
            else jnp.asarray(static.pilot_wide_vars, dtype=latent_dtype)
        ),
        diagnostic_metrics=static.diagnostic_metrics,
        initial_value_grad_by_param=_initial_value_grad_by_param,
        transition_current_value_grad_by_param=_transition_current_value_grad_by_param,
        transition_next_value_grad_by_param=_transition_next_value_grad_by_param,
        selected_transition_log_probs=_selected_transition_log_probs,
        pairwise_transition_log_probs=_pairwise_transition_log_probs,
        transition_log_probs_from_fixed_prev=_transition_log_probs_from_fixed_prev,
        transition_log_probs_by_param=_transition_log_probs_by_param,
        transition_log_probs_to_next_by_param=_transition_log_probs_to_next_by_param,
        sample_transition_by_label=_sample_transition_by_label,
        segment_terminal_label_log_probs=_segment_terminal_label_log_probs,
        path_future_tail_log_probs=_path_future_tail_log_probs,
        trajectory_label_log_probs=_trajectory_label_log_probs,
    )
