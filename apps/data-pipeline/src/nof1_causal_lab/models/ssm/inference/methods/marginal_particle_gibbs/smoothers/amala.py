"""Blocked Particle-aMALA conditional-SMC latent smoother.

This instantiates Particle-aMALA on the collapsed MPGibbs trajectory target,
not on a fixed-parameter smoothing target. Within each latent block, the CSMC
potential is the auxiliary-corrected MPG mixture increment

    G'_t(x_t) = eta_t(x_t) N(z_t; x_t + phi_t(x_t), sigma^2 I)
                         / N(z_t; x_t, sigma^2 I),

where eta_t is the marginalized MPG increment and phi_t is the local gradient
drift of log eta_t. Backward selection inside each block uses the augmented
Particle-aMALA Q' kernel, while block endpoints are scored against the exact
non-augmented MPG future tail outside the block.
"""

# References:
#   docs/papers/particle-mala-mgrad.pdf — Corenflos & Finke (2024), "Particle-MALA
#     and Particle-mGRAD" (arXiv:2401.14868), Algorithm 3.
#   docs/papers/auxiliary-kalman-samplers.pdf — Corenflos & Särkkä (arXiv:2303.00301).

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
    _gaussian_log_prob_shared_cholesky,
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _particle_ess_from_log_weights,
    _single_observation_log_probs_by_param,
    _transition_log_probs_by_param,
    _transition_log_probs_to_next_by_param,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
)


def smooth(ctx, key, x_ref):
    """Blocked particle-aMALA CSMC sweep over MPG's collapsed trajectory target."""
    contexts = ctx.contexts
    initial_label_log_probs = ctx.initial_label_log_probs
    init_means = ctx.init_means
    init_chols = ctx.init_chols
    init_logdets = ctx.init_logdets
    transition_chols = ctx.transition_chols
    transition_logdets = ctx.transition_logdets
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
    amala_delta = jnp.asarray(ctx.amala_delta, dtype=latent_dtype)
    amala_kappa = ctx.amala_kappa
    amala_grad_clip = ctx.amala_grad_clip
    _segment_terminal_label_log_probs = ctx.segment_terminal_label_log_probs
    _path_future_tail_log_probs = ctx.path_future_tail_log_probs
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        latent_smoother="amala",
        diagnostic_metrics=ctx.diagnostic_metrics,
    )

    proposal_kappa = jnp.asarray(amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(amala_grad_clip, dtype=latent_dtype)
    num_particles = num_free_particles + 1
    latent_dim = int(x_ref.shape[-1])

    def _log_isotropic_density(
        value: jnp.ndarray,
        mean: jnp.ndarray,
        proposal_var_t: jnp.ndarray,
    ) -> jnp.ndarray:
        diff = value - mean
        dim = int(diff.shape[-1])
        quadratic = jnp.sum(diff * diff, axis=-1) / proposal_var_t
        return -0.5 * (dim * jnp.log(2.0 * jnp.pi * proposal_var_t) + quadratic)

    def _clip_gradient(grad: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        norm = jnp.linalg.norm(grad)
        multiplier = jnp.minimum(
            jnp.asarray(1.0, dtype=latent_dtype),
            grad_clip / jnp.maximum(norm, jnp.asarray(1e-12, dtype=latent_dtype)),
        )
        return (grad * multiplier).astype(latent_dtype), norm.astype(traj_dtype)

    def _initial_label_logits(particle0: jnp.ndarray) -> jnp.ndarray:
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
        return initial_label_log_probs + initial_prior_lp_by_param + initial_obs_lp_by_param

    def _transition_label_logits(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        transition_lp = _transition_log_probs_by_param(
            contexts,
            transition_chols,
            transition_logdets,
            prev_particle[None, :],
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
        return prev_label_log_probs + transition_lp + obs_lp

    def _eta_label_drift_and_norm_from_logits(
        logits_fn,
        particle: jnp.ndarray,
        proposal_var_t: jnp.ndarray,
    ):
        def _value_and_label(z):
            logits = logits_fn(z)
            log_eta = jax.scipy.special.logsumexp(logits)
            return log_eta, _normalize_log_probs(logits).astype(traj_dtype)

        (log_eta, label_log_probs), grad = jax.value_and_grad(
            _value_and_label,
            has_aux=True,
        )(particle)
        clipped_grad, norm = _clip_gradient(grad)
        drift = (proposal_kappa * proposal_var_t * clipped_grad).astype(latent_dtype)
        return log_eta.astype(traj_dtype), label_log_probs, drift, norm

    def _sample_block(
        block_key_t: jnp.ndarray,
        current_path: jnp.ndarray,
        block_start: int,
        block_end: int,
        prefix_label_log_probs: jnp.ndarray,
        future_tail_history: jnp.ndarray,
    ):
        aux_key, proposal_key, resample_key, final_key, backward_key = random.split(block_key_t, 5)
        block_len = block_end - block_start + 1
        block_ref = current_path[block_start : block_end + 1]
        block_proposal_var = (
            jnp.asarray(0.5, dtype=latent_dtype) * amala_delta[block_start : block_end + 1]
        )
        block_proposal_scale = jnp.sqrt(block_proposal_var)
        previous_particle = current_path[block_start - 1] if block_start > 0 else current_path[0]

        def _first_eta_label_drift_and_norm(
            particle_t: jnp.ndarray,
            proposal_var_t: jnp.ndarray,
        ):
            if block_start == 0:
                return _eta_label_drift_and_norm_from_logits(
                    _initial_label_logits,
                    particle_t,
                    proposal_var_t,
                )
            time0 = jnp.asarray(block_start, dtype=jnp.int32)
            return _eta_label_drift_and_norm_from_logits(
                lambda z: _transition_label_logits(
                    prefix_label_log_probs,
                    previous_particle,
                    z,
                    time0,
                ),
                particle_t,
                proposal_var_t,
            )

        def _transition_eta_label_drift_and_norm(
            prev_label_log_probs: jnp.ndarray,
            prev_particle: jnp.ndarray,
            particle_t: jnp.ndarray,
            time_idx: jnp.ndarray,
            proposal_var_t: jnp.ndarray,
        ):
            return _eta_label_drift_and_norm_from_logits(
                lambda z: _transition_label_logits(
                    prev_label_log_probs,
                    prev_particle,
                    z,
                    time_idx,
                ),
                particle_t,
                proposal_var_t,
            )

        _, label_log_probs0_ref, drift0, norm0 = _first_eta_label_drift_and_norm(
            block_ref[0],
            block_proposal_var[0],
        )

        def _reference_scan(carry, inputs):
            prev_label_log_probs, prev_particle = carry
            offset, time_idx, proposal_var_t = inputs
            particle_t = block_ref[offset]
            _, next_label_log_probs, drift_t, norm_t = _transition_eta_label_drift_and_norm(
                prev_label_log_probs,
                prev_particle,
                particle_t,
                time_idx,
                proposal_var_t,
            )
            return (next_label_log_probs, particle_t), (drift_t, norm_t)

        if block_len > 1:
            _, (tail_drifts, tail_norms) = jax.lax.scan(
                _reference_scan,
                (label_log_probs0_ref, block_ref[0]),
                (
                    jnp.arange(1, block_len, dtype=jnp.int32),
                    jnp.arange(block_start + 1, block_end + 1, dtype=jnp.int32),
                    block_proposal_var[1:],
                ),
            )
            block_reference_drifts = jnp.concatenate([drift0[None, :], tail_drifts], axis=0)
            block_grad_norms = jnp.concatenate([norm0[None], tail_norms], axis=0)
        else:
            block_reference_drifts = drift0[None, :]
            block_grad_norms = norm0[None]

        auxiliary_block = (
            block_ref
            + block_reference_drifts
            + block_proposal_scale[:, None]
            * random.normal(
                aux_key,
                block_ref.shape,
                dtype=latent_dtype,
            )
        )
        free_particles_by_offset = auxiliary_block[:, None, :] + block_proposal_scale[
            :, None, None
        ] * random.normal(
            proposal_key,
            (block_len, num_free_particles, latent_dim),
            dtype=latent_dtype,
        )

        particles0 = jnp.concatenate(
            [block_ref[0][None, :], free_particles_by_offset[0]],
            axis=0,
        )
        log_eta0, label_log_probs0, drifts0, _ = jax.vmap(
            lambda particle: _first_eta_label_drift_and_norm(
                particle,
                block_proposal_var[0],
            )
        )(particles0)
        aux_num0 = _log_isotropic_density(
            auxiliary_block[0],
            particles0 + drifts0,
            block_proposal_var[0],
        )
        aux_den0 = _log_isotropic_density(
            auxiliary_block[0],
            particles0,
            block_proposal_var[0],
        )
        raw_log_weights0 = (log_eta0 + aux_num0 - aux_den0).astype(traj_dtype)
        aux_corrections0 = (raw_log_weights0 - log_eta0).astype(traj_dtype)
        log_weights0 = _normalize_log_probs(raw_log_weights0).astype(traj_dtype)

        resample_keys = random.split(resample_key, max(block_len - 1, 1))

        def _scan_step(carry, inputs):
            prev_log_weights, prev_label_log_probs, prev_particles = carry
            offset, time_idx, proposal_var_t, resample_key_t = inputs
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
            particles_t = jnp.concatenate(
                [block_ref[offset][None, :], free_particles_by_offset[offset]],
                axis=0,
            )
            log_eta_t, next_label_log_probs, drifts_t, _ = jax.vmap(
                lambda prev_label, prev_particle, particle: _transition_eta_label_drift_and_norm(
                    prev_label,
                    prev_particle,
                    particle,
                    time_idx,
                    proposal_var_t,
                )
            )(ancestor_label_log_probs, ancestor_particles, particles_t)
            aux_num_t = _log_isotropic_density(
                auxiliary_block[offset],
                particles_t + drifts_t,
                proposal_var_t,
            )
            aux_den_t = _log_isotropic_density(
                auxiliary_block[offset],
                particles_t,
                proposal_var_t,
            )
            raw_next_log_weights = (log_eta_t + aux_num_t - aux_den_t).astype(traj_dtype)
            aux_corrections_t = (raw_next_log_weights - log_eta_t).astype(traj_dtype)
            next_log_weights = _normalize_log_probs(raw_next_log_weights).astype(traj_dtype)
            return (
                next_log_weights,
                next_label_log_probs,
                particles_t,
            ), (
                next_log_weights,
                next_label_log_probs,
                particles_t,
                aux_corrections_t,
            )

        if block_len > 1:
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
                    tail_aux_corrections,
                ),
            ) = jax.lax.scan(
                _scan_step,
                (log_weights0, label_log_probs0, particles0),
                (
                    jnp.arange(1, block_len, dtype=jnp.int32),
                    jnp.arange(block_start + 1, block_end + 1, dtype=jnp.int32),
                    block_proposal_var[1:],
                    resample_keys[: block_len - 1],
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
            aux_corrections_history = jnp.concatenate(
                [aux_corrections0[None, :], tail_aux_corrections],
                axis=0,
            )
        else:
            log_weights = log_weights0
            label_log_probs = label_log_probs0
            log_weights_history = log_weights0[None, :]
            label_log_probs_history = label_log_probs0[None, :, :]
            particles_history = particles0[None, :, :]
            aux_corrections_history = aux_corrections0[None, :]

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
            final_transition_lp = _transition_log_probs_to_next_by_param(
                contexts,
                transition_chols,
                transition_logdets,
                final_latent[None, :],
                next_fixed,
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
            local_offset = local_time_idx - block_start
            particles_t = particles_history[local_offset]
            log_weights_t = log_weights_history[local_offset]
            label_log_probs_t = label_log_probs_history[local_offset]
            time_next = jnp.asarray(local_time_idx + 1, dtype=jnp.int32)
            transition_lp = _transition_log_probs_to_next_by_param(
                contexts,
                transition_chols,
                transition_logdets,
                particles_t,
                next_particle,
                time_next,
            )
            obs_lp = _single_observation_log_probs_by_param(
                contexts,
                next_particle,
                state.observation_auxiliary,
                time_next,
                runtime_observations,
                obs_increment_fn,
            )

            def _next_auxiliary_log_density(prev_label_log_probs, prev_particle):
                _, _, drift, _ = _transition_eta_label_drift_and_norm(
                    prev_label_log_probs,
                    prev_particle,
                    next_particle,
                    time_next,
                    block_proposal_var[local_offset + 1],
                )
                auxiliary_next = auxiliary_block[local_offset + 1]
                return _log_isotropic_density(
                    auxiliary_next,
                    next_particle + drift,
                    block_proposal_var[local_offset + 1],
                ).astype(traj_dtype)

            next_auxiliary_log_density = jax.vmap(_next_auxiliary_log_density)(
                label_log_probs_t,
                particles_t,
            )
            backward_logits = (
                log_weights_t
                + jax.scipy.special.logsumexp(
                    label_log_probs_t + transition_lp + obs_lp[None, :] + next_future_tail[None, :],
                    axis=1,
                )
                + next_auxiliary_log_density
            )
            backward_log_probs = _normalize_log_probs(backward_logits)
            selected_particle = random.categorical(backward_key_t, backward_log_probs).astype(
                jnp.int32
            )
            latent_t = particles_t[selected_particle]
            selected_future_tail = (
                transition_lp[selected_particle] + obs_lp + next_future_tail
            ).astype(traj_dtype)
            return (latent_t, selected_future_tail), (
                latent_t,
                selected_particle,
                _particle_ess_from_log_weights(backward_log_probs).astype(traj_dtype),
                _categorical_entropy_from_log_probs(backward_log_probs).astype(traj_dtype),
                _categorical_max_prob_from_log_probs(backward_log_probs).astype(traj_dtype),
            )

        if block_len > 1:
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
            selection_particle_ess = jnp.concatenate(
                [jnp.flip(reversed_backward_ess, axis=0), terminal_particle_ess[None]],
                axis=0,
            )
            selection_entropy = jnp.concatenate(
                [jnp.flip(reversed_backward_entropy, axis=0), terminal_selection_entropy[None]],
                axis=0,
            )
            selection_max_prob = jnp.concatenate(
                [jnp.flip(reversed_backward_max_prob, axis=0), terminal_selection_max_prob[None]],
                axis=0,
            )
        else:
            block_path = final_latent[None, :]
            block_indices = final_particle[None]
            selection_particle_ess = terminal_particle_ess[None]
            selection_entropy = terminal_selection_entropy[None]
            selection_max_prob = terminal_selection_max_prob[None]

        auxiliary_noise_norms = jnp.linalg.norm(
            auxiliary_block - block_ref - block_reference_drifts,
            axis=-1,
        ).astype(traj_dtype)
        proposal_displacement_norms = jnp.linalg.norm(
            free_particles_by_offset - block_ref[:, None, :],
            axis=-1,
        ).astype(traj_dtype)

        return (
            block_path.astype(latent_dtype),
            block_indices,
            forward_particle_ess,
            forward_log_weight_range,
            forward_log_weight_variance,
            selection_particle_ess,
            selection_entropy,
            selection_max_prob,
            block_reference_drifts,
            block_grad_norms,
            auxiliary_noise_norms,
            proposal_displacement_norms,
            aux_corrections_history,
        )

    with jax.named_scope("particle_amala_block_loop"):
        block_keys = random.split(key, num_blocks)
        latent_path = x_ref
        origin_path = jnp.zeros((num_steps,), dtype=jnp.int32)
        forward_particle_ess = jnp.zeros((num_steps,), dtype=traj_dtype)
        forward_log_weight_range = jnp.zeros((num_steps,), dtype=traj_dtype)
        forward_log_weight_variance = jnp.zeros((num_steps,), dtype=traj_dtype)
        selection_particle_ess = jnp.zeros((num_steps,), dtype=traj_dtype)
        selection_entropy = jnp.zeros((num_steps,), dtype=traj_dtype)
        selection_max_prob = jnp.zeros((num_steps,), dtype=traj_dtype)
        reference_drifts = jnp.zeros((num_steps, latent_dim), dtype=latent_dtype)
        grad_norms = jnp.zeros((num_steps,), dtype=traj_dtype)
        auxiliary_noise_norms = jnp.zeros((num_steps,), dtype=traj_dtype)
        proposal_displacement_norms = jnp.zeros(
            (num_steps, num_free_particles),
            dtype=traj_dtype,
        )
        aux_corrections_history = jnp.zeros((num_steps, num_particles), dtype=traj_dtype)
        future_tail_history = _path_future_tail_log_probs(latent_path)
        prefix_label_log_probs = initial_label_log_probs
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, num_steps) - 1
            previous_particle = latent_path[block_start - 1] if block_start > 0 else latent_path[0]
            with jax.named_scope(f"particle_amala_block_{block_idx}"):
                (
                    block_path,
                    block_indices,
                    block_forward_ess,
                    block_forward_range,
                    block_forward_variance,
                    block_selection_ess,
                    block_selection_entropy,
                    block_selection_max_prob,
                    block_reference_drifts,
                    block_grad_norms,
                    block_auxiliary_noise_norms,
                    block_proposal_displacement_norms,
                    block_aux_corrections,
                ) = _sample_block(
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
            selection_particle_ess = selection_particle_ess.at[block_start : block_end + 1].set(
                block_selection_ess
            )
            selection_entropy = selection_entropy.at[block_start : block_end + 1].set(
                block_selection_entropy
            )
            selection_max_prob = selection_max_prob.at[block_start : block_end + 1].set(
                block_selection_max_prob
            )
            reference_drifts = reference_drifts.at[block_start : block_end + 1].set(
                block_reference_drifts
            )
            grad_norms = grad_norms.at[block_start : block_end + 1].set(block_grad_norms)
            auxiliary_noise_norms = auxiliary_noise_norms.at[block_start : block_end + 1].set(
                block_auxiliary_noise_norms
            )
            proposal_displacement_norms = proposal_displacement_norms.at[
                block_start : block_end + 1
            ].set(block_proposal_displacement_norms)
            aux_corrections_history = aux_corrections_history.at[block_start : block_end + 1].set(
                block_aux_corrections
            )
            prefix_label_log_probs = _segment_terminal_label_log_probs(
                prefix_label_log_probs,
                block_path,
                previous_particle,
                block_start,
            )

    diagnostics = {
        "amala_grad_norm_mean": jnp.mean(grad_norms).astype(traj_dtype),
        "amala_grad_norm_max": jnp.max(grad_norms).astype(traj_dtype),
    }
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
                "backward_selection_ess_by_t": selection_particle_ess,
                "backward_selection_entropy_by_t": selection_entropy,
                "backward_selection_max_prob_by_t": selection_max_prob,
            }
        )
    if diagnostic_flags.amala_proposal:
        drift_norms = jnp.linalg.norm(reference_drifts, axis=-1).astype(traj_dtype)
        auxiliary_correction_abs = jnp.abs(aux_corrections_history)
        auxiliary_noise_norm_mean = jnp.mean(auxiliary_noise_norms)
        diagnostics.update(
            {
                "amala_grad_clip_fraction": jnp.mean(grad_norms > grad_clip).astype(traj_dtype),
                "amala_drift_norm_mean": jnp.mean(drift_norms).astype(traj_dtype),
                "amala_drift_norm_max": jnp.max(drift_norms).astype(traj_dtype),
                "amala_auxiliary_noise_norm_mean": auxiliary_noise_norm_mean.astype(traj_dtype),
                "amala_auxiliary_noise_norm_max": jnp.max(auxiliary_noise_norms).astype(traj_dtype),
                "amala_drift_to_auxiliary_noise_ratio_mean": (
                    jnp.mean(drift_norms) / jnp.maximum(auxiliary_noise_norm_mean, 1e-12)
                ).astype(traj_dtype),
                "amala_proposal_displacement_norm_mean": jnp.mean(
                    proposal_displacement_norms
                ).astype(traj_dtype),
                "amala_proposal_displacement_norm_max": jnp.max(proposal_displacement_norms).astype(
                    traj_dtype
                ),
                "amala_auxiliary_correction_variance": jnp.var(aux_corrections_history).astype(
                    traj_dtype
                ),
                "amala_auxiliary_correction_max_abs": jnp.max(auxiliary_correction_abs).astype(
                    traj_dtype
                ),
            }
        )
    return MPGibbsLatentSmootherResult(
        latent_path=latent_path.astype(latent_dtype),
        final_label_log_probs=prefix_label_log_probs.astype(traj_dtype),
        origin_path=origin_path,
        diagnostics=diagnostics,
    )
