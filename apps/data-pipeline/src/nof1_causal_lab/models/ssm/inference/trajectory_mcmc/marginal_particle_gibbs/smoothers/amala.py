"""Particle-aMALA sequential conditional-SMC latent smoother.

This instantiates Particle-aMALA on the collapsed MPGibbs trajectory target,
not on a fixed-parameter smoothing target. The CSMC potential is the
auxiliary-corrected MPG mixture increment

    G'_t(x_t) = eta_t(x_t) N(z_t; x_t + phi_t(x_t), sigma^2 I)
                         / N(z_t; x_t, sigma^2 I),

where eta_t is the marginalized MPG increment and phi_t is the local
gradient drift of log eta_t. Backward sampling scores future suffixes under
the augmented target ratio, avoiding the label-factorized shortcut used by
the plain posterior-mixture smoother.
"""

# References:
#   docs/papers/particle-mala-mgrad.pdf — Corenflos & Finke (2024), "Particle-MALA
#     and Particle-mGRAD" (arXiv:2401.14868), Algorithm 3.
#   docs/papers/auxiliary-kalman-samplers.pdf — Corenflos & Särkkä (arXiv:2303.00301).

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.marginal_particle_gibbs._math import (
    _categorical_entropy_from_log_probs,
    _categorical_max_prob_from_log_probs,
    _gaussian_log_prob_shared_cholesky,
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _observation_log_probs_by_param,
    _particle_ess_from_log_weights,
    _single_observation_log_probs_by_param,
    _transition_log_probs_by_param,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
)


def smooth(ctx, key, x_ref):
    """Sequential particle-aMALA CSMC sweep over MPG's collapsed trajectory target."""
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
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    amala_q_scale = ctx.amala_q_scale
    amala_kappa = ctx.amala_kappa
    amala_grad_clip = ctx.amala_grad_clip
    _segment_terminal_label_log_probs = ctx.segment_terminal_label_log_probs
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        latent_smoother="amala",
        diagnostic_metrics=ctx.diagnostic_metrics,
    )
    smoother_key = key
    current_path = x_ref

    proposal_scale = jnp.asarray(amala_q_scale, dtype=latent_dtype)
    proposal_var = proposal_scale * proposal_scale
    proposal_kappa = jnp.asarray(amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(amala_grad_clip, dtype=latent_dtype)

    def _log_isotropic_density(value: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
        diff = value - mean
        dim = int(diff.shape[-1])
        quadratic = jnp.sum(diff * diff, axis=-1) / proposal_var
        return -0.5 * (dim * jnp.log(2.0 * jnp.pi * proposal_var) + quadratic)

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

    def _initial_log_eta(particle0: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.special.logsumexp(_initial_label_logits(particle0))

    def _initial_label_log_probs(particle0: jnp.ndarray) -> jnp.ndarray:
        return _normalize_log_probs(_initial_label_logits(particle0)).astype(traj_dtype)

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

    def _transition_log_eta(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.scipy.special.logsumexp(
            _transition_label_logits(
                prev_label_log_probs,
                prev_particle,
                particle_t,
                time_idx,
            )
        )

    def _transition_label_log_probs(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return _normalize_log_probs(
            _transition_label_logits(
                prev_label_log_probs,
                prev_particle,
                particle_t,
                time_idx,
            )
        ).astype(traj_dtype)

    def _initial_drift_and_norm(particle0: jnp.ndarray):
        grad = jax.grad(_initial_log_eta)(particle0)
        clipped_grad, norm = _clip_gradient(grad)
        return (proposal_kappa * proposal_var * clipped_grad).astype(latent_dtype), norm

    def _transition_drift_and_norm(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ):
        grad = jax.grad(
            lambda z: _transition_log_eta(
                prev_label_log_probs,
                prev_particle,
                z,
                time_idx,
            )
        )(particle_t)
        clipped_grad, norm = _clip_gradient(grad)
        return (proposal_kappa * proposal_var * clipped_grad).astype(latent_dtype), norm

    (
        aux_key,
        proposal_key,
        resample_key,
        final_key,
        backward_key,
    ) = random.split(smoother_key, 5)
    latent_dim = int(current_path.shape[-1])
    drift0, norm0 = _initial_drift_and_norm(current_path[0])
    label_log_probs0_ref = _initial_label_log_probs(current_path[0])

    def _reference_scan(carry, time_idx):
        prev_label_log_probs, prev_particle = carry
        particle_t = current_path[time_idx]
        drift_t, norm_t = _transition_drift_and_norm(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        next_label_log_probs = _transition_label_log_probs(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        return (next_label_log_probs, particle_t), (drift_t, norm_t)

    if num_steps > 1:
        _, (tail_drifts, tail_norms) = jax.lax.scan(
            _reference_scan,
            (label_log_probs0_ref, current_path[0]),
            jnp.arange(1, num_steps, dtype=jnp.int32),
        )
        reference_drifts = jnp.concatenate([drift0[None, :], tail_drifts], axis=0)
        grad_norms = jnp.concatenate([norm0[None], tail_norms], axis=0)
    else:
        reference_drifts = drift0[None, :]
        grad_norms = norm0[None]

    auxiliary_path = (
        current_path
        + reference_drifts
        + proposal_scale
        * random.normal(
            aux_key,
            current_path.shape,
            dtype=latent_dtype,
        )
    )
    free_particles_by_time = auxiliary_path[:, None, :] + proposal_scale * random.normal(
        proposal_key,
        (num_steps, num_free_particles, latent_dim),
        dtype=latent_dtype,
    )

    def _initial_log_gprime(particle0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        drift, _norm = _initial_drift_and_norm(particle0)
        log_eta = _initial_log_eta(particle0)
        aux_num = _log_isotropic_density(auxiliary_path[0], particle0 + drift)
        aux_den = _log_isotropic_density(auxiliary_path[0], particle0)
        return (log_eta + aux_num - aux_den).astype(traj_dtype), log_eta.astype(traj_dtype)

    def _transition_log_gprime(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        drift, _norm = _transition_drift_and_norm(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        log_eta = _transition_log_eta(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        aux_num = _log_isotropic_density(auxiliary_path[time_idx], particle_t + drift)
        aux_den = _log_isotropic_density(auxiliary_path[time_idx], particle_t)
        return (log_eta + aux_num - aux_den).astype(traj_dtype), log_eta.astype(traj_dtype)

    def _transition_log_qprime(
        prev_label_log_probs: jnp.ndarray,
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        drift, _norm = _transition_drift_and_norm(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        log_eta = _transition_log_eta(
            prev_label_log_probs,
            prev_particle,
            particle_t,
            time_idx,
        )
        aux_num = _log_isotropic_density(auxiliary_path[time_idx], particle_t + drift)
        return (log_eta + aux_num).astype(traj_dtype)

    def _future_log_qprime(
        prefix_label_log_probs: jnp.ndarray,
        prefix_particle: jnp.ndarray,
        selected_path: jnp.ndarray,
        start_time: jnp.ndarray,
    ) -> jnp.ndarray:
        def _scan_future(carry, time_idx):
            total, prev_label_log_probs, prev_particle = carry
            particle_t = selected_path[time_idx]

            def _include(include_carry):
                include_total, include_label_log_probs, include_prev_particle = include_carry
                increment = _transition_log_qprime(
                    include_label_log_probs,
                    include_prev_particle,
                    particle_t,
                    time_idx,
                )
                next_label_log_probs = _transition_label_log_probs(
                    include_label_log_probs,
                    include_prev_particle,
                    particle_t,
                    time_idx,
                )
                return (
                    include_total + increment,
                    next_label_log_probs,
                    particle_t,
                )

            next_carry = jax.lax.cond(
                time_idx >= start_time,
                _include,
                lambda skip_carry: skip_carry,
                (total, prev_label_log_probs, prev_particle),
            )
            return next_carry, None

        if num_steps > 1:
            (future_total, _, _), _ = jax.lax.scan(
                _scan_future,
                (
                    jnp.asarray(0.0, dtype=traj_dtype),
                    prefix_label_log_probs,
                    prefix_particle,
                ),
                jnp.arange(1, num_steps, dtype=jnp.int32),
            )
            return future_total.astype(traj_dtype)
        return jnp.asarray(0.0, dtype=traj_dtype)

    particles0 = jnp.concatenate(
        [current_path[0][None, :], free_particles_by_time[0]],
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
    init_obs_lp_by_param = _observation_log_probs_by_param(
        contexts,
        particles0,
        state.observation_auxiliary,
        jnp.asarray(0, dtype=jnp.int32),
        runtime_observations,
        obs_increment_fn,
    )
    target_logits0 = (
        initial_label_log_probs[None, :] + init_prior_lp_by_param + init_obs_lp_by_param
    )
    raw_log_weights0, log_eta0 = jax.vmap(_initial_log_gprime)(particles0)
    aux_corrections0 = (raw_log_weights0 - log_eta0).astype(traj_dtype)
    log_weights0 = _normalize_log_probs(raw_log_weights0).astype(traj_dtype)
    label_log_probs0 = _normalize_log_probs(target_logits0).astype(traj_dtype)

    resample_keys = random.split(resample_key, max(num_steps - 1, 1))

    def _scan_step(carry, inputs):
        prev_log_weights, prev_label_log_probs, prev_particles = carry
        time_idx, resample_key_t = inputs
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
            [current_path[time_idx][None, :], free_particles_by_time[time_idx]],
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
        target_logits = ancestor_label_log_probs + transition_lp_by_param + obs_lp_by_param
        raw_next_log_weights, log_eta_t = jax.vmap(
            lambda prev_label, prev_particle, particle: _transition_log_gprime(
                prev_label,
                prev_particle,
                particle,
                time_idx,
            )
        )(ancestor_label_log_probs, ancestor_particles, particles_t)
        aux_corrections_t = (raw_next_log_weights - log_eta_t).astype(traj_dtype)
        next_log_weights = _normalize_log_probs(raw_next_log_weights).astype(traj_dtype)
        next_label_log_probs = _normalize_log_probs(target_logits).astype(traj_dtype)
        return (
            next_log_weights,
            next_label_log_probs,
            particles_t,
        ), (next_log_weights, next_label_log_probs, particles_t, aux_corrections_t)

    if num_steps > 1:
        with jax.named_scope("particle_amala_forward_filter_scan"):
            (
                (
                    log_weights,
                    _label_log_probs,
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
                    jnp.arange(1, num_steps, dtype=jnp.int32),
                    resample_keys[: num_steps - 1],
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
        _label_log_probs = label_log_probs0
        log_weights_history = log_weights0[None, :]
        label_log_probs_history = label_log_probs0[None, :, :]
        particles_history = particles0[None, :, :]
        aux_corrections_history = aux_corrections0[None, :]

    forward_particle_ess = _particle_ess_from_log_weights(log_weights_history).astype(traj_dtype)
    forward_log_weight_range = _log_weight_range(log_weights_history).astype(traj_dtype)
    forward_log_weight_variance = _log_weight_variance(log_weights_history).astype(traj_dtype)

    final_particle_ess = _particle_ess_from_log_weights(log_weights).astype(traj_dtype)
    final_selection_entropy = _categorical_entropy_from_log_probs(log_weights).astype(traj_dtype)
    final_selection_max_prob = _categorical_max_prob_from_log_probs(log_weights).astype(traj_dtype)
    final_particle = random.categorical(final_key, log_weights).astype(jnp.int32)
    final_latent = particles_history[-1, final_particle]
    backward_keys = random.split(backward_key, max(num_steps - 1, 1))

    def _backward_step(carry, inputs):
        selected_path, selected_indices = carry
        local_time_idx, backward_key_t = inputs
        particles_t = particles_history[local_time_idx]
        log_weights_t = log_weights_history[local_time_idx]
        label_log_probs_t = label_log_probs_history[local_time_idx]
        future_scores = jax.vmap(
            lambda label_log_probs, particle_t: _future_log_qprime(
                label_log_probs,
                particle_t,
                selected_path,
                local_time_idx + 1,
            )
        )(label_log_probs_t, particles_t)
        backward_logits = log_weights_t + future_scores
        backward_log_probs = _normalize_log_probs(backward_logits)
        selected_particle = random.categorical(backward_key_t, backward_log_probs).astype(jnp.int32)
        latent_t = particles_t[selected_particle]
        next_path = selected_path.at[local_time_idx].set(latent_t)
        next_indices = selected_indices.at[local_time_idx].set(selected_particle)
        return (next_path, next_indices), (
            latent_t,
            selected_particle,
            _particle_ess_from_log_weights(backward_log_probs).astype(traj_dtype),
            _categorical_entropy_from_log_probs(backward_log_probs).astype(traj_dtype),
            _categorical_max_prob_from_log_probs(backward_log_probs).astype(traj_dtype),
        )

    selected_path0 = current_path.at[num_steps - 1].set(final_latent)
    selected_indices0 = (
        jnp.zeros((num_steps,), dtype=jnp.int32).at[num_steps - 1].set(final_particle)
    )
    if num_steps > 1:
        with jax.named_scope("particle_amala_backward_sample_scan"):
            (
                (
                    latent_path,
                    origin_path,
                ),
                (
                    _reversed_latents,
                    _reversed_indices,
                    reversed_backward_ess,
                    reversed_backward_entropy,
                    reversed_backward_max_prob,
                ),
            ) = jax.lax.scan(
                _backward_step,
                (selected_path0, selected_indices0),
                (
                    jnp.arange(num_steps - 2, -1, -1, dtype=jnp.int32),
                    backward_keys[: num_steps - 1],
                ),
            )
        backward_particle_ess = jnp.concatenate(
            [jnp.flip(reversed_backward_ess, axis=0), final_particle_ess[None]],
            axis=0,
        )
        backward_selection_entropy = jnp.concatenate(
            [jnp.flip(reversed_backward_entropy, axis=0), final_selection_entropy[None]],
            axis=0,
        )
        backward_selection_max_prob = jnp.concatenate(
            [jnp.flip(reversed_backward_max_prob, axis=0), final_selection_max_prob[None]],
            axis=0,
        )
    else:
        latent_path = final_latent[None, :]
        origin_path = final_particle[None]
        backward_particle_ess = final_particle_ess[None]
        backward_selection_entropy = final_selection_entropy[None]
        backward_selection_max_prob = final_selection_max_prob[None]

    final_label_log_probs = _segment_terminal_label_log_probs(
        initial_label_log_probs,
        latent_path,
        latent_path[0],
        0,
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
                "backward_selection_ess_by_t": backward_particle_ess,
                "backward_selection_entropy_by_t": backward_selection_entropy,
                "backward_selection_max_prob_by_t": backward_selection_max_prob,
            }
        )
    if diagnostic_flags.amala_proposal:
        drift_norms = jnp.linalg.norm(reference_drifts, axis=-1).astype(traj_dtype)
        auxiliary_noise_norms = jnp.linalg.norm(
            auxiliary_path - current_path - reference_drifts,
            axis=-1,
        ).astype(traj_dtype)
        proposal_displacement_norms = jnp.linalg.norm(
            free_particles_by_time - current_path[:, None, :],
            axis=-1,
        ).astype(traj_dtype)
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
        final_label_log_probs=final_label_log_probs.astype(traj_dtype),
        origin_path=origin_path,
        diagnostics=diagnostics,
    )
