"""Blocked Particle-aMALA+ conditional-SMC latent smoother for MPGibbs."""

# References:
#   docs/papers/particle-mala-mgrad.pdf - Corenflos & Finke (2024), "Particle-MALA
#     and Particle-mGRAD" (arXiv:2401.14868), Particle-aMALA+ and the auxiliary
#     target construction.
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf - Corenflos (2025), arXiv:2505.04611:
#     MPGibbs's collapsed posterior mixture over the parameter ensemble.

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _categorical_entropy_from_log_probs,
    _categorical_max_prob_from_log_probs,
    _log_weight_range,
    _log_weight_variance,
    _normalize_log_probs,
    _particle_ess_from_log_weights,
    _single_observation_log_probs_by_param,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
)


def smooth(ctx, key, x_ref):
    """Run blocked Particle-aMALA+ over MPG's collapsed trajectory target."""
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
    block_size = ctx.block_size
    num_blocks = ctx.num_blocks
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    amala_delta = jnp.asarray(ctx.amala_delta, dtype=latent_dtype)
    proposal_kappa = jnp.asarray(ctx.amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(ctx.amala_grad_clip, dtype=latent_dtype)
    num_particles = num_free_particles + 1
    latent_dim = int(x_ref.shape[-1])
    num_parameter_particles = int(contexts.Ad.shape[0])
    zero_index = jnp.asarray(0, dtype=jnp.int32)
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        latent_smoother="amala_plus",
        diagnostic_metrics=ctx.diagnostic_metrics,
    )

    def _log_isotropic_density(
        value: jnp.ndarray,
        mean: jnp.ndarray,
        proposal_var_t: jnp.ndarray,
    ) -> jnp.ndarray:
        diff = value - mean
        dim = int(diff.shape[-1])
        quadratic = jnp.sum(diff * diff, axis=-1) / proposal_var_t
        return -0.5 * (dim * jnp.log(2.0 * jnp.pi * proposal_var_t) + quadratic)

    def _log_isotropic_density_by_t(
        value: jnp.ndarray,
        mean: jnp.ndarray,
        proposal_var_by_t: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(_log_isotropic_density)(value, mean, proposal_var_by_t)

    def _clip_gradients(grad_path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        norms = jnp.linalg.norm(grad_path, axis=-1)
        multipliers = jnp.minimum(
            jnp.asarray(1.0, dtype=latent_dtype),
            grad_clip / jnp.maximum(norms, jnp.asarray(1e-12, dtype=latent_dtype)),
        )
        return (grad_path * multipliers[:, None]).astype(latent_dtype), norms.astype(traj_dtype)

    def _obs_value_grad_by_param(
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: obs_increment_fn(
                    context,
                    particle,
                    state.observation_auxiliary,
                    time_idx,
                    runtime_observations,
                )
            )(particle_t)

        value, grad = jax.vmap(_one_context)(contexts)
        return value.astype(traj_dtype), grad.astype(latent_dtype)

    def _initial_prior_value_grad_by_param(
        particle0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def _one_param(mean, chol, logdet):
            residual = particle0 - mean
            whitened = jla.solve_triangular(chol, residual, lower=True)
            precision_residual = jla.solve_triangular(chol.T, whitened, lower=False)
            log_prob = -0.5 * (
                latent_dim * jnp.log(2.0 * jnp.pi) + logdet + jnp.sum(whitened * whitened)
            )
            return log_prob, -precision_residual

        log_prob, grad = jax.vmap(_one_param)(init_means, init_chols, init_logdets)
        return log_prob.astype(traj_dtype), grad.astype(latent_dtype)

    def _transition_value_grad_by_param(
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def _one_param(context, chol_by_time, logdet_by_time):
            Ad = context.Ad[time_idx]
            mean = prev_particle @ Ad.T + context.cd[time_idx]
            residual = particle_t - mean
            chol = chol_by_time[time_idx]
            whitened = jla.solve_triangular(chol, residual, lower=True)
            precision_residual = jla.solve_triangular(chol.T, whitened, lower=False)
            log_prob = -0.5 * (
                latent_dim * jnp.log(2.0 * jnp.pi)
                + logdet_by_time[time_idx]
                + jnp.sum(whitened * whitened)
            )
            grad_prev = Ad.T @ precision_residual
            grad_current = -precision_residual
            return log_prob, grad_prev, grad_current

        log_prob, grad_prev, grad_current = jax.vmap(_one_param)(
            contexts,
            transition_chols,
            transition_logdets,
        )
        return (
            log_prob.astype(traj_dtype),
            grad_prev.astype(latent_dtype),
            grad_current.astype(latent_dtype),
        )

    def _sample_block(
        block_key_t: jnp.ndarray,
        current_path: jnp.ndarray,
        block_start: jnp.ndarray,
        block_len: int,
        prefix_label_log_probs: jnp.ndarray,
        future_tail_history: jnp.ndarray,
        *,
        has_previous: bool,
        has_future: bool,
    ):
        aux_key, proposal_key, resample_key, final_key, backward_key = random.split(block_key_t, 5)
        block_start = jnp.asarray(block_start, dtype=jnp.int32)
        block_end = block_start + jnp.asarray(block_len - 1, dtype=jnp.int32)
        block_ref = jax.lax.dynamic_slice(
            current_path,
            (block_start, zero_index),
            (block_len, latent_dim),
        )
        block_delta = jax.lax.dynamic_slice(amala_delta, (block_start,), (block_len,))
        block_proposal_var = jnp.asarray(0.5, dtype=latent_dtype) * block_delta
        block_proposal_scale = jnp.sqrt(block_proposal_var)
        previous_particle = current_path[block_start - 1] if has_previous else current_path[0]

        def _initial_stats(
            particle0: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            component_grad = jnp.zeros(
                (num_parameter_particles, block_len, latent_dim),
                dtype=latent_dtype,
            )
            obs_lp, obs_grad = _obs_value_grad_by_param(
                particle0,
                jnp.asarray(block_start, dtype=jnp.int32),
            )
            if not has_previous:
                prior_lp, prior_grad = _initial_prior_value_grad_by_param(particle0)
                logits = initial_label_log_probs + prior_lp + obs_lp
                grad0 = prior_grad + obs_grad
            else:
                transition_lp, _grad_prev, grad_current = _transition_value_grad_by_param(
                    previous_particle,
                    particle0,
                    jnp.asarray(block_start, dtype=jnp.int32),
                )
                logits = prefix_label_log_probs + transition_lp + obs_lp
                grad0 = grad_current + obs_grad
            component_grad = component_grad.at[:, 0, :].set(grad0)
            return logits.astype(traj_dtype), component_grad

        def _extend_stats(
            prev_particle: jnp.ndarray,
            particle_t: jnp.ndarray,
            local_offset: int,
            logits: jnp.ndarray,
            component_grad: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            time_idx = jnp.asarray(block_start + local_offset, dtype=jnp.int32)
            transition_lp, grad_prev, grad_current = _transition_value_grad_by_param(
                prev_particle,
                particle_t,
                time_idx,
            )
            obs_lp, obs_grad = _obs_value_grad_by_param(particle_t, time_idx)
            next_logits = (logits + transition_lp + obs_lp).astype(traj_dtype)
            next_component_grad = component_grad.at[:, local_offset - 1, :].add(grad_prev)
            next_component_grad = next_component_grad.at[:, local_offset, :].add(
                grad_current + obs_grad
            )
            return next_logits, next_component_grad

        def _terminal_stats(
            block_path: jnp.ndarray,
            logits: jnp.ndarray,
            component_grad: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            if has_future:
                next_fixed = current_path[block_end + 1]
                time_next = jnp.asarray(block_end + 1, dtype=jnp.int32)
                transition_lp, bridge_grad, _grad_next = _transition_value_grad_by_param(
                    block_path[-1],
                    next_fixed,
                    time_next,
                )
                bridge_obs_lp = _single_observation_log_probs_by_param(
                    contexts,
                    next_fixed,
                    state.observation_auxiliary,
                    time_next,
                    runtime_observations,
                    obs_increment_fn,
                )
                terminal_logits = (
                    logits + transition_lp + bridge_obs_lp + future_tail_history[block_end + 1]
                ).astype(traj_dtype)
                terminal_grad = component_grad.at[:, block_len - 1, :].add(bridge_grad)
                return terminal_logits, terminal_grad
            return logits, component_grad

        def _gamma_plus_from_stats(
            logits: jnp.ndarray,
            component_grad: jnp.ndarray,
            block_path: jnp.ndarray,
            auxiliary_block: jnp.ndarray,
            local_end: int,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            log_gamma = jax.scipy.special.logsumexp(logits).astype(traj_dtype)
            label_log_probs = _normalize_log_probs(logits).astype(traj_dtype)
            label_probs = jnp.exp(label_log_probs).astype(latent_dtype)
            grad_path = jnp.einsum("m,mtd->td", label_probs, component_grad)
            clipped_grad, grad_norms = _clip_gradients(grad_path)
            drift_path = (proposal_kappa * block_proposal_var[:, None] * clipped_grad).astype(
                latent_dtype
            )
            auxiliary_log_density = _log_isotropic_density_by_t(
                auxiliary_block,
                block_path + drift_path,
                block_proposal_var,
            )
            active = jnp.arange(block_len, dtype=jnp.int32) <= jnp.asarray(
                local_end,
                dtype=jnp.int32,
            )
            log_gamma_plus = log_gamma + jnp.sum(
                jnp.where(active, auxiliary_log_density, 0.0)
            ).astype(traj_dtype)
            return log_gamma_plus.astype(traj_dtype), log_gamma, drift_path, grad_norms

        def _reference_terminal_stats(
            block_path: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            logits0, grad0 = _initial_stats(block_path[0])
            if block_len > 1:
                (logits_t, grad_t), _ = jax.lax.scan(
                    lambda carry, offset: (
                        _extend_stats(
                            block_path[offset - 1],
                            block_path[offset],
                            offset,
                            carry[0],
                            carry[1],
                        ),
                        None,
                    ),
                    (logits0, grad0),
                    jnp.arange(1, block_len, dtype=jnp.int32),
                )
            else:
                logits_t = logits0
                grad_t = grad0
            terminal_logits, terminal_grad = _terminal_stats(block_path, logits_t, grad_t)
            return terminal_logits, terminal_grad

        def _selected_block_label_log_probs(block_path: jnp.ndarray) -> jnp.ndarray:
            logits0, grad0 = _initial_stats(block_path[0])
            if block_len > 1:
                (logits_t, _), _ = jax.lax.scan(
                    lambda carry, offset: (
                        _extend_stats(
                            block_path[offset - 1],
                            block_path[offset],
                            offset,
                            carry[0],
                            carry[1],
                        ),
                        None,
                    ),
                    (logits0, grad0),
                    jnp.arange(1, block_len, dtype=jnp.int32),
                )
            else:
                logits_t = logits0
            return _normalize_log_probs(logits_t).astype(traj_dtype)

        terminal_reference_logits, terminal_reference_component_grad = _reference_terminal_stats(
            block_ref
        )
        _, _, terminal_reference_drifts, terminal_reference_grad_norms = _gamma_plus_from_stats(
            terminal_reference_logits,
            terminal_reference_component_grad,
            block_ref,
            block_ref,
            block_len - 1,
        )
        auxiliary_block = (
            block_ref
            + terminal_reference_drifts
            + block_proposal_scale[:, None]
            * random.normal(aux_key, block_ref.shape, dtype=latent_dtype)
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
        paths0 = jnp.zeros((num_particles, block_len, latent_dim), dtype=latent_dtype)
        paths0 = paths0.at[:, 0, :].set(particles0)
        logits0, component_grad0 = jax.vmap(_initial_stats)(particles0)
        log_gamma_plus0, base_log_gamma0, _drift0, _grad_norms0 = jax.vmap(
            lambda logits, component_grad, block_path: _gamma_plus_from_stats(
                logits,
                component_grad,
                block_path,
                auxiliary_block,
                0,
            )
        )(logits0, component_grad0, paths0)
        proposal_log_density0 = _log_isotropic_density(
            auxiliary_block[0],
            particles0,
            block_proposal_var[0],
        )
        raw_log_weights0 = (log_gamma_plus0 - proposal_log_density0).astype(traj_dtype)
        log_weights0 = _normalize_log_probs(raw_log_weights0).astype(traj_dtype)
        aux_corrections0 = (raw_log_weights0 - base_log_gamma0).astype(traj_dtype)

        resample_keys = random.split(resample_key, max(block_len - 1, 1))
        if block_len > 1:

            def _forward_step(carry, inputs):
                (
                    prev_log_weights,
                    prev_log_gamma_plus,
                    prev_base_log_gamma,
                    prev_paths,
                    prev_logits,
                    prev_component_grad,
                ) = carry
                local_offset, resample_key_t = inputs
                free_ancestors = random.categorical(
                    resample_key_t,
                    prev_log_weights,
                    shape=(num_free_particles,),
                ).astype(jnp.int32)
                ancestors = jnp.concatenate(
                    [jnp.zeros((1,), dtype=jnp.int32), free_ancestors],
                    axis=0,
                )
                ancestor_paths = jnp.take(prev_paths, ancestors, axis=0)
                ancestor_logits = jnp.take(prev_logits, ancestors, axis=0)
                ancestor_component_grad = jnp.take(prev_component_grad, ancestors, axis=0)
                ancestor_log_gamma_plus = jnp.take(prev_log_gamma_plus, ancestors, axis=0)
                ancestor_base_log_gamma = jnp.take(prev_base_log_gamma, ancestors, axis=0)
                particles_t = jnp.concatenate(
                    [block_ref[local_offset][None, :], free_particles_by_offset[local_offset]],
                    axis=0,
                )
                paths_t = ancestor_paths.at[:, local_offset, :].set(particles_t)
                logits_t, component_grad_t = jax.vmap(
                    lambda prev_particle, particle_t, logits, component_grad: _extend_stats(
                        prev_particle,
                        particle_t,
                        local_offset,
                        logits,
                        component_grad,
                    )
                )(
                    ancestor_paths[:, local_offset - 1, :],
                    particles_t,
                    ancestor_logits,
                    ancestor_component_grad,
                )
                log_gamma_plus_t, base_log_gamma_t, _drift_t, _grad_norms_t = jax.vmap(
                    lambda logits, component_grad, block_path: _gamma_plus_from_stats(
                        logits,
                        component_grad,
                        block_path,
                        auxiliary_block,
                        local_offset,
                    )
                )(logits_t, component_grad_t, paths_t)
                proposal_log_density_t = _log_isotropic_density(
                    auxiliary_block[local_offset],
                    particles_t,
                    block_proposal_var[local_offset],
                )
                raw_log_weights_t = (
                    log_gamma_plus_t - ancestor_log_gamma_plus - proposal_log_density_t
                ).astype(traj_dtype)
                log_weights_t = _normalize_log_probs(raw_log_weights_t).astype(traj_dtype)
                base_increment_t = (base_log_gamma_t - ancestor_base_log_gamma).astype(traj_dtype)
                aux_corrections_t = (raw_log_weights_t - base_increment_t).astype(traj_dtype)
                return (
                    log_weights_t,
                    log_gamma_plus_t,
                    base_log_gamma_t,
                    paths_t,
                    logits_t,
                    component_grad_t,
                ), (
                    particles_t,
                    paths_t,
                    logits_t,
                    component_grad_t,
                    log_weights_t,
                    log_gamma_plus_t,
                    aux_corrections_t,
                )

            (
                _,
                (
                    tail_particles,
                    tail_paths,
                    tail_logits,
                    tail_component_grad,
                    tail_log_weights,
                    tail_log_gamma_plus,
                    tail_aux_corrections,
                ),
            ) = jax.lax.scan(
                _forward_step,
                (
                    log_weights0,
                    log_gamma_plus0,
                    base_log_gamma0,
                    paths0,
                    logits0,
                    component_grad0,
                ),
                (
                    jnp.arange(1, block_len, dtype=jnp.int32),
                    resample_keys[: block_len - 1],
                ),
            )
            particles_history = jnp.concatenate([particles0[None, :], tail_particles], axis=0)
            paths_history = jnp.concatenate([paths0[None, :], tail_paths], axis=0)
            logits_history = jnp.concatenate([logits0[None, :], tail_logits], axis=0)
            component_grad_history = jnp.concatenate(
                [component_grad0[None, :], tail_component_grad],
                axis=0,
            )
            log_weights_history = jnp.concatenate(
                [log_weights0[None, :], tail_log_weights],
                axis=0,
            )
            log_gamma_plus_history = jnp.concatenate(
                [log_gamma_plus0[None, :], tail_log_gamma_plus],
                axis=0,
            )
            aux_corrections_history = jnp.concatenate(
                [aux_corrections0[None, :], tail_aux_corrections],
                axis=0,
            )
        else:
            particles_history = particles0[None, :]
            paths_history = paths0[None, :]
            logits_history = logits0[None, :]
            component_grad_history = component_grad0[None, :]
            log_weights_history = log_weights0[None, :]
            log_gamma_plus_history = log_gamma_plus0[None, :]
            aux_corrections_history = aux_corrections0[None, :]

        forward_particle_ess = _particle_ess_from_log_weights(log_weights_history).astype(
            traj_dtype
        )
        forward_log_weight_range = _log_weight_range(log_weights_history).astype(traj_dtype)
        forward_log_weight_variance = _log_weight_variance(log_weights_history).astype(traj_dtype)

        terminal_logits, terminal_component_grad = jax.vmap(_terminal_stats)(
            paths_history[-1],
            logits_history[-1],
            component_grad_history[-1],
        )
        terminal_log_gamma_plus, _terminal_base_log_gamma, _terminal_drift, _terminal_grad_norms = (
            jax.vmap(
                lambda logits, component_grad, block_path: _gamma_plus_from_stats(
                    logits,
                    component_grad,
                    block_path,
                    auxiliary_block,
                    block_len - 1,
                )
            )(terminal_logits, terminal_component_grad, paths_history[-1])
        )
        terminal_log_weights = _normalize_log_probs(
            log_weights_history[-1] + terminal_log_gamma_plus - log_gamma_plus_history[-1]
        ).astype(traj_dtype)
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
        block_path = paths_history[-1, final_particle]
        block_indices = jnp.zeros((block_len,), dtype=jnp.int32).at[-1].set(final_particle)
        selection_particle_ess = (
            jnp.zeros((block_len,), dtype=traj_dtype).at[-1].set(terminal_particle_ess)
        )
        selection_entropy = (
            jnp.zeros((block_len,), dtype=traj_dtype).at[-1].set(terminal_selection_entropy)
        )
        selection_max_prob = (
            jnp.zeros((block_len,), dtype=traj_dtype).at[-1].set(terminal_selection_max_prob)
        )

        backward_keys = random.split(backward_key, max(block_len - 1, 1))

        def _candidate_terminal_gamma(
            candidate_prefix_path: jnp.ndarray,
            candidate_logits: jnp.ndarray,
            candidate_component_grad: jnp.ndarray,
            local_offset: int,
            selected_suffix: jnp.ndarray,
        ) -> jnp.ndarray:
            use_suffix = jnp.arange(block_len, dtype=jnp.int32) > local_offset
            candidate_path = jnp.where(
                use_suffix[:, None],
                selected_suffix,
                candidate_prefix_path,
            )
            if block_len > 1:

                def _suffix_step(carry, suffix_offset):
                    logits_t, component_grad_t, prev_particle = carry
                    particle_t = selected_suffix[suffix_offset]
                    extended_logits, extended_component_grad = _extend_stats(
                        prev_particle,
                        particle_t,
                        suffix_offset,
                        logits_t,
                        component_grad_t,
                    )
                    use_extended = suffix_offset > local_offset
                    return (
                        jnp.where(use_extended, extended_logits, logits_t),
                        jnp.where(use_extended, extended_component_grad, component_grad_t),
                        jnp.where(use_extended, particle_t, prev_particle),
                    ), None

                (logits_t, component_grad_t, _), _ = jax.lax.scan(
                    _suffix_step,
                    (
                        candidate_logits,
                        candidate_component_grad,
                        candidate_prefix_path[local_offset],
                    ),
                    jnp.arange(1, block_len, dtype=jnp.int32),
                )
            else:
                logits_t = candidate_logits
                component_grad_t = candidate_component_grad
            terminal_logits_t, terminal_component_grad_t = _terminal_stats(
                candidate_path,
                logits_t,
                component_grad_t,
            )
            terminal_log_gamma_plus_t, _base, _drift, _grad_norms = _gamma_plus_from_stats(
                terminal_logits_t,
                terminal_component_grad_t,
                candidate_path,
                auxiliary_block,
                block_len - 1,
            )
            return terminal_log_gamma_plus_t

        if block_len > 1:

            def _backward_step(selected_suffix, inputs):
                local_offset, backward_key_t = inputs
                candidate_terminal_log_gamma_plus = jax.vmap(
                    lambda path, logits, component_grad: _candidate_terminal_gamma(
                        path,
                        logits,
                        component_grad,
                        local_offset,
                        selected_suffix,
                    )
                )(
                    paths_history[local_offset],
                    logits_history[local_offset],
                    component_grad_history[local_offset],
                )
                backward_log_probs = _normalize_log_probs(
                    log_weights_history[local_offset]
                    + candidate_terminal_log_gamma_plus
                    - log_gamma_plus_history[local_offset]
                ).astype(traj_dtype)
                selected_particle = random.categorical(
                    backward_key_t,
                    backward_log_probs,
                ).astype(jnp.int32)
                next_suffix = selected_suffix.at[local_offset].set(
                    particles_history[local_offset, selected_particle]
                )
                return next_suffix, (
                    selected_particle,
                    _particle_ess_from_log_weights(backward_log_probs).astype(traj_dtype),
                    _categorical_entropy_from_log_probs(backward_log_probs).astype(traj_dtype),
                    _categorical_max_prob_from_log_probs(backward_log_probs).astype(traj_dtype),
                )

            (
                block_path,
                (
                    reversed_indices,
                    reversed_selection_ess,
                    reversed_selection_entropy,
                    reversed_selection_max_prob,
                ),
            ) = jax.lax.scan(
                _backward_step,
                block_path,
                (
                    jnp.arange(block_len - 2, -1, -1, dtype=jnp.int32),
                    backward_keys[: block_len - 1],
                ),
            )
            block_indices = jnp.concatenate(
                [jnp.flip(reversed_indices, axis=0), final_particle[None]],
                axis=0,
            )
            selection_particle_ess = jnp.concatenate(
                [jnp.flip(reversed_selection_ess, axis=0), terminal_particle_ess[None]],
                axis=0,
            )
            selection_entropy = jnp.concatenate(
                [
                    jnp.flip(reversed_selection_entropy, axis=0),
                    terminal_selection_entropy[None],
                ],
                axis=0,
            )
            selection_max_prob = jnp.concatenate(
                [
                    jnp.flip(reversed_selection_max_prob, axis=0),
                    terminal_selection_max_prob[None],
                ],
                axis=0,
            )

        auxiliary_noise_norms = jnp.linalg.norm(
            auxiliary_block - block_ref - terminal_reference_drifts,
            axis=-1,
        ).astype(traj_dtype)
        proposal_displacement_norms = jnp.linalg.norm(
            free_particles_by_offset - block_ref[:, None, :],
            axis=-1,
        ).astype(traj_dtype)
        block_final_label_log_probs = _selected_block_label_log_probs(block_path)

        return (
            block_path.astype(latent_dtype),
            block_indices,
            forward_particle_ess,
            forward_log_weight_range,
            forward_log_weight_variance,
            selection_particle_ess,
            selection_entropy,
            selection_max_prob,
            terminal_reference_drifts,
            terminal_reference_grad_norms,
            auxiliary_noise_norms,
            proposal_displacement_norms,
            aux_corrections_history,
            block_final_label_log_probs,
        )

    with jax.named_scope("particle_amala_plus_block_loop"):
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
        future_tail_history = ctx.path_future_tail_log_probs(latent_path)
        prefix_label_log_probs = initial_label_log_probs

        def _store_block(carry, block_start: jnp.ndarray, block_outputs):
            (
                latent_path,
                _prefix_label_log_probs,
                origin_path,
                forward_particle_ess,
                forward_log_weight_range,
                forward_log_weight_variance,
                selection_particle_ess,
                selection_entropy,
                selection_max_prob,
                reference_drifts,
                grad_norms,
                auxiliary_noise_norms,
                proposal_displacement_norms,
                aux_corrections_history,
            ) = carry
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
                block_final_label_log_probs,
            ) = block_outputs
            block_start = jnp.asarray(block_start, dtype=jnp.int32)
            latent_path = jax.lax.dynamic_update_slice(
                latent_path,
                block_path,
                (block_start, zero_index),
            )
            origin_path = jax.lax.dynamic_update_slice(
                origin_path,
                block_indices,
                (block_start,),
            )
            forward_particle_ess = jax.lax.dynamic_update_slice(
                forward_particle_ess,
                block_forward_ess,
                (block_start,),
            )
            forward_log_weight_range = jax.lax.dynamic_update_slice(
                forward_log_weight_range,
                block_forward_range,
                (block_start,),
            )
            forward_log_weight_variance = jax.lax.dynamic_update_slice(
                forward_log_weight_variance,
                block_forward_variance,
                (block_start,),
            )
            selection_particle_ess = jax.lax.dynamic_update_slice(
                selection_particle_ess,
                block_selection_ess,
                (block_start,),
            )
            selection_entropy = jax.lax.dynamic_update_slice(
                selection_entropy,
                block_selection_entropy,
                (block_start,),
            )
            selection_max_prob = jax.lax.dynamic_update_slice(
                selection_max_prob,
                block_selection_max_prob,
                (block_start,),
            )
            reference_drifts = jax.lax.dynamic_update_slice(
                reference_drifts,
                block_reference_drifts,
                (block_start, zero_index),
            )
            grad_norms = jax.lax.dynamic_update_slice(
                grad_norms,
                block_grad_norms,
                (block_start,),
            )
            auxiliary_noise_norms = jax.lax.dynamic_update_slice(
                auxiliary_noise_norms,
                block_auxiliary_noise_norms,
                (block_start,),
            )
            proposal_displacement_norms = jax.lax.dynamic_update_slice(
                proposal_displacement_norms,
                block_proposal_displacement_norms,
                (block_start, zero_index),
            )
            aux_corrections_history = jax.lax.dynamic_update_slice(
                aux_corrections_history,
                block_aux_corrections,
                (block_start, zero_index),
            )
            return (
                latent_path,
                block_final_label_log_probs,
                origin_path,
                forward_particle_ess,
                forward_log_weight_range,
                forward_log_weight_variance,
                selection_particle_ess,
                selection_entropy,
                selection_max_prob,
                reference_drifts,
                grad_norms,
                auxiliary_noise_norms,
                proposal_displacement_norms,
                aux_corrections_history,
            )

        carry = (
            latent_path,
            prefix_label_log_probs,
            origin_path,
            forward_particle_ess,
            forward_log_weight_range,
            forward_log_weight_variance,
            selection_particle_ess,
            selection_entropy,
            selection_max_prob,
            reference_drifts,
            grad_norms,
            auxiliary_noise_norms,
            proposal_displacement_norms,
            aux_corrections_history,
        )

        first_block_len = min(block_size, num_steps)
        first_outputs = _sample_block(
            block_keys[0],
            latent_path,
            jnp.asarray(0, dtype=jnp.int32),
            first_block_len,
            prefix_label_log_probs,
            future_tail_history,
            has_previous=False,
            has_future=num_blocks > 1,
        )
        carry = _store_block(carry, jnp.asarray(0, dtype=jnp.int32), first_outputs)

        if num_blocks > 2:

            def _middle_block_step(carry, inputs):
                block_idx, block_key_t = inputs
                latent_path = carry[0]
                prefix_label_log_probs = carry[1]
                block_start = block_idx * jnp.asarray(block_size, dtype=jnp.int32)
                block_outputs = _sample_block(
                    block_key_t,
                    latent_path,
                    block_start,
                    block_size,
                    prefix_label_log_probs,
                    future_tail_history,
                    has_previous=True,
                    has_future=True,
                )
                return _store_block(carry, block_start, block_outputs), None

            carry, _ = jax.lax.scan(
                _middle_block_step,
                carry,
                (
                    jnp.arange(1, num_blocks - 1, dtype=jnp.int32),
                    block_keys[1 : num_blocks - 1],
                ),
            )

        if num_blocks > 1:
            (
                latent_path,
                prefix_label_log_probs,
                origin_path,
                forward_particle_ess,
                forward_log_weight_range,
                forward_log_weight_variance,
                selection_particle_ess,
                selection_entropy,
                selection_max_prob,
                reference_drifts,
                grad_norms,
                auxiliary_noise_norms,
                proposal_displacement_norms,
                aux_corrections_history,
            ) = carry
            final_block_idx = num_blocks - 1
            final_block_start = final_block_idx * block_size
            final_block_len = num_steps - final_block_start
            final_outputs = _sample_block(
                block_keys[final_block_idx],
                latent_path,
                jnp.asarray(final_block_start, dtype=jnp.int32),
                final_block_len,
                prefix_label_log_probs,
                future_tail_history,
                has_previous=True,
                has_future=False,
            )
            carry = _store_block(
                (
                    latent_path,
                    prefix_label_log_probs,
                    origin_path,
                    forward_particle_ess,
                    forward_log_weight_range,
                    forward_log_weight_variance,
                    selection_particle_ess,
                    selection_entropy,
                    selection_max_prob,
                    reference_drifts,
                    grad_norms,
                    auxiliary_noise_norms,
                    proposal_displacement_norms,
                    aux_corrections_history,
                ),
                jnp.asarray(final_block_start, dtype=jnp.int32),
                final_outputs,
            )

        (
            latent_path,
            prefix_label_log_probs,
            origin_path,
            forward_particle_ess,
            forward_log_weight_range,
            forward_log_weight_variance,
            selection_particle_ess,
            selection_entropy,
            selection_max_prob,
            reference_drifts,
            grad_norms,
            auxiliary_noise_norms,
            proposal_displacement_norms,
            aux_corrections_history,
        ) = carry

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
