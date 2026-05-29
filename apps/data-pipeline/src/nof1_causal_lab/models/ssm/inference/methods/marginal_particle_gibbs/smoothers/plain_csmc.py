"""Plain blocked posterior-mixture backward conditional-SMC latent smoother."""

# References:
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf — Corenflos (2025), arXiv:2505.04611:
#     conditional SMC against the posterior mixture over the parameter ensemble.
#   docs/reference/bibliography.md — Andrieu, Doucet & Holenstein (2010) for cSMC;
#     Lindsten, Jordan & Schön (2014) for backward / ancestor sampling.

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _categorical_entropy_from_log_probs,
    _categorical_max_prob_from_log_probs,
    _categorical_rows,
    _gaussian_log_prob_shared_cholesky,
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _observation_log_probs_by_param,
    _particle_ess_from_log_weights,
    _sample_gaussian_from_chol,
    _single_observation_log_probs_by_param,
    _transition_log_probs_by_param,
    _transition_log_probs_to_next_by_param,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
)


def smooth(ctx, key, x_ref):
    """Blocked backward conditional-SMC sweep over the posterior parameter mixture."""
    contexts = ctx.contexts
    transition_chols = ctx.transition_chols
    transition_logdets = ctx.transition_logdets
    init_means = ctx.init_means
    init_chols = ctx.init_chols
    init_logdets = ctx.init_logdets
    initial_label_log_probs = ctx.initial_label_log_probs
    state = ctx.state
    runtime_observations = ctx.runtime_observations
    obs_increment_fn = ctx.obs_increment_fn
    num_steps = ctx.num_steps
    num_free_particles = ctx.num_free_particles
    num_parameter_particles = ctx.num_parameter_particles
    block_size = ctx.block_size
    num_blocks = ctx.num_blocks
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    _transition_log_probs_from_fixed_prev = ctx.transition_log_probs_from_fixed_prev
    _segment_terminal_label_log_probs = ctx.segment_terminal_label_log_probs
    _path_future_tail_log_probs = ctx.path_future_tail_log_probs
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        latent_smoother="plain",
        diagnostic_metrics=ctx.diagnostic_metrics,
    )
    block_key = key

    def _backward_sample_block(
        block_key_t: jnp.ndarray,
        current_path: jnp.ndarray,
        block_start: int,
        block_end: int,
        prefix_label_log_probs: jnp.ndarray,
        future_tail_history: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        (
            init_component_key,
            init_sample_key,
            resample_key,
            component_key,
            transition_key,
            final_key,
            backward_key,
        ) = random.split(block_key_t, 7)
        block_len = block_end - block_start + 1

        if block_start == 0:
            free_init_labels = random.categorical(
                init_component_key,
                prefix_label_log_probs,
                shape=(num_free_particles,),
            ).astype(jnp.int32)
            init_free_particles = _sample_gaussian_from_chol(
                init_sample_key,
                init_means[free_init_labels],
                init_chols[free_init_labels],
            )
            particles0 = jnp.concatenate(
                [current_path[block_start][None, :], init_free_particles],
                axis=0,
            )
            init_prior_lp_by_param = jnp.swapaxes(
                jax.vmap(
                    lambda mean, chol, logdet: _gaussian_log_prob_shared_cholesky(
                        particles0,
                        mean,
                        chol,
                        logdet,
                    )
                )(init_means, init_chols, init_logdets),
                0,
                1,
            )
        else:
            prev_fixed = current_path[block_start - 1]
            free_init_labels = random.categorical(
                init_component_key,
                prefix_label_log_probs,
                shape=(num_free_particles,),
            ).astype(jnp.int32)
            Ad_free0 = contexts.Ad[free_init_labels, block_start]
            cd_free0 = contexts.cd[free_init_labels, block_start]
            free_means0 = jnp.einsum("j,nij->ni", prev_fixed, Ad_free0) + cd_free0
            free_chols0 = transition_chols[free_init_labels, block_start]
            init_free_particles = _sample_gaussian_from_chol(
                init_sample_key,
                free_means0,
                free_chols0,
            )
            particles0 = jnp.concatenate(
                [current_path[block_start][None, :], init_free_particles],
                axis=0,
            )
            init_prior_lp_by_param = _transition_log_probs_from_fixed_prev(
                prev_fixed,
                particles0,
                jnp.asarray(block_start, dtype=jnp.int32),
            )

        init_obs_lp_by_param = _observation_log_probs_by_param(
            contexts,
            particles0,
            state.observation_auxiliary,
            jnp.asarray(block_start, dtype=jnp.int32),
            runtime_observations,
            obs_increment_fn,
        )
        proposal_logits0 = prefix_label_log_probs[None, :] + init_prior_lp_by_param
        target_logits0 = proposal_logits0 + init_obs_lp_by_param
        raw_log_weights0 = jax.scipy.special.logsumexp(
            target_logits0,
            axis=1,
        ) - jax.scipy.special.logsumexp(proposal_logits0, axis=1)
        log_weights0 = _normalize_log_probs(raw_log_weights0).astype(traj_dtype)
        label_log_probs0 = _normalize_log_probs(target_logits0).astype(traj_dtype)

        resample_keys = random.split(resample_key, max(block_len - 1, 1))
        component_keys = random.split(component_key, max(block_len - 1, 1))
        transition_keys = random.split(transition_key, max(block_len - 1, 1))

        def _scan_step(carry, inputs):
            prev_log_weights, prev_label_log_probs, prev_particles = carry
            time_idx, resample_key_t, component_key_t, transition_key_t = inputs
            free_ancestors = random.categorical(
                resample_key_t,
                prev_log_weights,
                shape=(num_free_particles,),
            ).astype(jnp.int32)
            ancestors = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), free_ancestors],
                axis=0,
            )
            ancestor_particles = jnp.take(prev_particles, ancestors, axis=0)
            ancestor_label_log_probs = jnp.take(prev_label_log_probs, ancestors, axis=0)

            free_labels = _categorical_rows(
                component_key_t,
                ancestor_label_log_probs[1:],
            )
            free_prev = ancestor_particles[1:]
            Ad_free = contexts.Ad[free_labels, time_idx]
            cd_free = contexts.cd[free_labels, time_idx]
            free_means = jnp.einsum("nj,nij->ni", free_prev, Ad_free) + cd_free
            free_chols = transition_chols[free_labels, time_idx]
            free_particles = _sample_gaussian_from_chol(
                transition_key_t,
                free_means,
                free_chols,
            )
            particles_t = jnp.concatenate(
                [current_path[time_idx][None, :], free_particles],
                axis=0,
            )

            transition_lp_by_param = _transition_log_probs_by_param(
                contexts,
                transition_chols,
                transition_logdets,
                ancestor_particles,
                particles_t,
                time_idx,
            )
            obs_lp_by_param = _observation_log_probs_by_param(
                contexts,
                particles_t,
                state.observation_auxiliary,
                time_idx,
                runtime_observations,
                obs_increment_fn,
            )
            proposal_logits = ancestor_label_log_probs + transition_lp_by_param
            target_logits = proposal_logits + obs_lp_by_param
            raw_next_log_weights = jax.scipy.special.logsumexp(
                target_logits,
                axis=1,
            ) - jax.scipy.special.logsumexp(proposal_logits, axis=1)
            next_log_weights = _normalize_log_probs(raw_next_log_weights).astype(traj_dtype)
            next_label_log_probs = _normalize_log_probs(target_logits).astype(traj_dtype)
            return (
                next_log_weights,
                next_label_log_probs,
                particles_t,
            ), (next_log_weights, next_label_log_probs, particles_t)

        if block_len > 1:
            with jax.named_scope("forward_filter_scan"):
                (
                    (
                        log_weights,
                        label_log_probs,
                        _last_particles,
                    ),
                    (
                        tail_log_weights,
                        tail_label_log_probs,
                        tail_particles,
                    ),
                ) = jax.lax.scan(
                    _scan_step,
                    (log_weights0, label_log_probs0, particles0),
                    (
                        jnp.arange(block_start + 1, block_end + 1, dtype=jnp.int32),
                        resample_keys[: block_len - 1],
                        component_keys[: block_len - 1],
                        transition_keys[: block_len - 1],
                    ),
                )
            log_weights_history = jnp.concatenate(
                [log_weights0[None, :], tail_log_weights],
                axis=0,
            )
            label_log_probs_history = jnp.concatenate(
                [label_log_probs0[None, :, :], tail_label_log_probs],
                axis=0,
            )
            particles_history = jnp.concatenate(
                [particles0[None, :, :], tail_particles],
                axis=0,
            )
        else:
            log_weights = log_weights0
            label_log_probs = label_log_probs0
            log_weights_history = log_weights0[None, :]
            label_log_probs_history = label_log_probs0[None, :, :]
            particles_history = particles0[None, :, :]

        forward_particle_ess = _particle_ess_from_log_weights(log_weights_history).astype(
            traj_dtype
        )
        forward_log_weight_range = _log_weight_range(log_weights_history).astype(traj_dtype)
        forward_log_weight_variance = _log_weight_variance(log_weights_history).astype(traj_dtype)

        if block_end < num_steps - 1:
            next_fixed = current_path[block_end + 1]
            bridge_transition_lp = _transition_log_probs_to_next_by_param(
                contexts,
                transition_chols,
                transition_logdets,
                particles_history[-1],
                next_fixed,
                jnp.asarray(block_end + 1, dtype=jnp.int32),
            )
            bridge_obs_lp = _single_observation_log_probs_by_param(
                contexts,
                next_fixed,
                state.observation_auxiliary,
                jnp.asarray(block_end + 1, dtype=jnp.int32),
                runtime_observations,
                obs_increment_fn,
            )
            bridge_label_logits = (
                label_log_probs
                + bridge_transition_lp
                + bridge_obs_lp[None, :]
                + future_tail_history[block_end + 1][None, :]
            )
            terminal_log_weights = _normalize_log_probs(
                log_weights + jax.scipy.special.logsumexp(bridge_label_logits, axis=1)
            ).astype(traj_dtype)
        else:
            terminal_log_weights = log_weights

        terminal_particle_ess = _particle_ess_from_log_weights(terminal_log_weights).astype(
            traj_dtype
        )
        terminal_selection_entropy = _categorical_entropy_from_log_probs(
            terminal_log_weights
        ).astype(traj_dtype)
        terminal_selection_max_prob = _categorical_max_prob_from_log_probs(
            terminal_log_weights
        ).astype(traj_dtype)
        final_particle = random.categorical(final_key, terminal_log_weights).astype(jnp.int32)
        final_latent = particles_history[-1, final_particle]
        if block_end < num_steps - 1:
            next_fixed = current_path[block_end + 1]
            final_transition_lp = _transition_log_probs_from_fixed_prev(
                final_latent,
                next_fixed[None, :],
                jnp.asarray(block_end + 1, dtype=jnp.int32),
            )[0]
            final_obs_lp = _single_observation_log_probs_by_param(
                contexts,
                next_fixed,
                state.observation_auxiliary,
                jnp.asarray(block_end + 1, dtype=jnp.int32),
                runtime_observations,
                obs_increment_fn,
            )
            final_future_tail = (
                final_transition_lp + final_obs_lp + future_tail_history[block_end + 1]
            ).astype(traj_dtype)
        else:
            final_future_tail = jnp.zeros((num_parameter_particles,), dtype=traj_dtype)

        backward_keys = random.split(backward_key, max(block_len - 1, 1))

        def _backward_step(carry, inputs):
            next_particle, next_future_tail = carry
            local_time_idx, backward_key_t = inputs
            particles_t = particles_history[local_time_idx - block_start]
            log_weights_t = log_weights_history[local_time_idx - block_start]
            label_log_probs_t = label_log_probs_history[local_time_idx - block_start]
            transition_lp = _transition_log_probs_to_next_by_param(
                contexts,
                transition_chols,
                transition_logdets,
                particles_t,
                next_particle,
                local_time_idx + 1,
            )
            obs_lp = _single_observation_log_probs_by_param(
                contexts,
                next_particle,
                state.observation_auxiliary,
                local_time_idx + 1,
                runtime_observations,
                obs_increment_fn,
            )
            backward_logits = log_weights_t + jax.scipy.special.logsumexp(
                label_log_probs_t + transition_lp + obs_lp[None, :] + next_future_tail[None, :],
                axis=1,
            )
            backward_log_probs = _normalize_log_probs(backward_logits)
            selected_particle = random.categorical(backward_key_t, backward_log_probs).astype(
                jnp.int32
            )
            latent_t = particles_t[selected_particle]
            selected_transition_lp = _transition_log_probs_from_fixed_prev(
                latent_t,
                next_particle[None, :],
                local_time_idx + 1,
            )[0]
            selected_future_tail = (selected_transition_lp + obs_lp + next_future_tail).astype(
                traj_dtype
            )
            return (latent_t, selected_future_tail), (
                latent_t,
                selected_particle,
                _particle_ess_from_log_weights(backward_log_probs).astype(traj_dtype),
                _categorical_entropy_from_log_probs(backward_log_probs).astype(traj_dtype),
                _categorical_max_prob_from_log_probs(backward_log_probs).astype(traj_dtype),
            )

        if block_len > 1:
            with jax.named_scope("backward_sample_scan"):
                (
                    _,
                    (
                        reversed_latents,
                        reversed_indices,
                        reversed_backward_ess,
                        reversed_backward_entropy,
                        reversed_backward_max_prob,
                    ),
                ) = jax.lax.scan(
                    _backward_step,
                    (final_latent, final_future_tail),
                    (
                        jnp.arange(block_end - 1, block_start - 1, -1, dtype=jnp.int32),
                        backward_keys[: block_len - 1],
                    ),
                )
            block_path = jnp.concatenate(
                [jnp.flip(reversed_latents, axis=0), final_latent[None, :]],
                axis=0,
            )
            block_indices = jnp.concatenate(
                [jnp.flip(reversed_indices, axis=0), final_particle[None]],
                axis=0,
            )
            backward_particle_ess = jnp.concatenate(
                [jnp.flip(reversed_backward_ess, axis=0), terminal_particle_ess[None]],
                axis=0,
            )
            backward_selection_entropy = jnp.concatenate(
                [
                    jnp.flip(reversed_backward_entropy, axis=0),
                    terminal_selection_entropy[None],
                ],
                axis=0,
            )
            backward_selection_max_prob = jnp.concatenate(
                [
                    jnp.flip(reversed_backward_max_prob, axis=0),
                    terminal_selection_max_prob[None],
                ],
                axis=0,
            )
        else:
            block_path = final_latent[None, :]
            block_indices = final_particle[None]
            backward_particle_ess = terminal_particle_ess[None]
            backward_selection_entropy = terminal_selection_entropy[None]
            backward_selection_max_prob = terminal_selection_max_prob[None]

        return (
            block_path.astype(latent_dtype),
            block_indices,
            forward_particle_ess,
            forward_log_weight_range,
            forward_log_weight_variance,
            backward_particle_ess,
            backward_selection_entropy,
            backward_selection_max_prob,
        )

    with jax.named_scope("backward_sample_loop"):
        block_keys = random.split(block_key, num_blocks)
        latent_path = x_ref
        origin_path = jnp.zeros((num_steps,), dtype=jnp.int32)
        forward_particle_ess = jnp.zeros((num_steps,), dtype=traj_dtype)
        forward_log_weight_range = jnp.zeros((num_steps,), dtype=traj_dtype)
        forward_log_weight_variance = jnp.zeros((num_steps,), dtype=traj_dtype)
        backward_particle_ess = jnp.zeros((num_steps,), dtype=traj_dtype)
        backward_selection_entropy = jnp.zeros((num_steps,), dtype=traj_dtype)
        backward_selection_max_prob = jnp.zeros((num_steps,), dtype=traj_dtype)
        future_tail_history = _path_future_tail_log_probs(latent_path)
        prefix_label_log_probs = initial_label_log_probs
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, num_steps) - 1
            previous_particle = latent_path[block_start - 1] if block_start > 0 else latent_path[0]
            with jax.named_scope(f"backward_block_{block_idx}"):
                (
                    block_path,
                    block_indices,
                    block_forward_ess,
                    block_forward_range,
                    block_forward_variance,
                    block_backward_ess,
                    block_backward_entropy,
                    block_backward_max_prob,
                ) = _backward_sample_block(
                    block_keys[block_idx],
                    latent_path,
                    block_start,
                    block_end,
                    prefix_label_log_probs,
                    future_tail_history,
                )
            latent_path = latent_path.at[block_start : block_end + 1].set(block_path)
            origin_path = origin_path.at[block_start : block_end + 1].set(block_indices)
            forward_particle_ess = forward_particle_ess.at[block_start : block_end + 1].set(
                block_forward_ess
            )
            forward_log_weight_range = forward_log_weight_range.at[block_start : block_end + 1].set(
                block_forward_range
            )
            forward_log_weight_variance = forward_log_weight_variance.at[
                block_start : block_end + 1
            ].set(block_forward_variance)
            backward_particle_ess = backward_particle_ess.at[block_start : block_end + 1].set(
                block_backward_ess
            )
            backward_selection_entropy = backward_selection_entropy.at[
                block_start : block_end + 1
            ].set(block_backward_entropy)
            backward_selection_max_prob = backward_selection_max_prob.at[
                block_start : block_end + 1
            ].set(block_backward_max_prob)
            prefix_label_log_probs = _segment_terminal_label_log_probs(
                prefix_label_log_probs,
                block_path,
                previous_particle,
                block_start,
            )
        diagnostics = {}
        if diagnostic_flags.particle_filter:
            diagnostics.update(
                {
                    "forward_particle_ess_by_t": forward_particle_ess,
                    "forward_log_weight_range_by_t": forward_log_weight_range,
                    "forward_log_weight_variance_by_t": forward_log_weight_variance,
                }
            )
        if diagnostic_flags.backward_selection:
            diagnostics.update(
                {
                    "backward_selection_ess_by_t": backward_particle_ess,
                    "backward_selection_entropy_by_t": backward_selection_entropy,
                    "backward_selection_max_prob_by_t": backward_selection_max_prob,
                }
            )
        return MPGibbsLatentSmootherResult(
            latent_path=latent_path,
            final_label_log_probs=prefix_label_log_probs,
            origin_path=origin_path,
            diagnostics=diagnostics,
        )
