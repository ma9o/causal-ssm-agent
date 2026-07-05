"""De-sequentialized conditional-SMC latent smoother (parallel-in-time)."""

# References:
#   docs/papers/desequentialized-smc.pdf — Corenflos, Chopin & Särkkä (2022),
#     arXiv:2202.02264: de-Sequentialized Monte Carlo. The smoothing distribution
#     is built by a divide-and-conquer binary tree over time (Prop. 2.2 stitching,
#     eq. 10-11), instead of a sequential forward filter + backward sampling.
#     We implement the conditional formulation c-dSMC (Section 3,
#     Algorithm 4): the reference trajectory is preserved through every block
#     combination, making this a valid parallel-in-time conditional-SMC Particle
#     Gibbs kernel. No backward-sampling step is needed — the balanced tree resamples
#     every time step at most O(log T) times, so there is no genealogy degeneracy.
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf — Corenflos (2025), arXiv:2505.04611:
#     conditional SMC against the posterior mixture over the parameter ensemble. The
#     seam weights and trajectory evidence are marginalized over the parameter labels
#     exactly as in the sibling smoothers (logsumexp over the parameter axis).
#
# Leaf proposals are gradient-informed auxiliary ("amala_exact") proposals — or the
# paid mixture ("paid_mix") that wraps the same z-anchored component together with a
# fixed IEKS-pilot component and a wide tail in one paid density. Both are exactly
# corrected: the auxiliary pseudo-observation potential restores invariance for the
# reference-dependent component, and the mixture density is subtracted in full.
#
# Implementation notes (math-preserving):
#   * The tree is evaluated level-by-level — the time axis is padded to a power of two
#     with evidence-free phantom leaves (their seams are masked out) so every depth is
#     a single batched combine, giving O(log T) sequential depth.
#   * Prior-predictive marginals come from a parallel-prefix (associative) scan over
#     the affine-Gaussian transition operator, not a sequential scan.
#   * The tree carries only segment endpoints, per-label evidence, and integer
#     ancestry. The selected latent path is reconstructed from the leaf particle
#     bank after the root draw, so intermediate levels do not concatenate full
#     trajectory tensors.
#   * Pairwise seam Mahalanobis distances use a whiten-then-GEMM expansion rather than
#     an N x N batch of triangular solves.
#   * Resampling is inverse-CDF multinomial (a single cumulative sum + searchsorted),
#     avoiding the O(N^3) Gumbel-noise materialization of a batched categorical draw.

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _DSMC_LEAF_PROPOSAL_PAID_MIX,
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _gaussian_log_prob_shared_cholesky,
    _normalize_log_probs,
    _observation_log_probs_by_param,
)


def smooth(ctx, key, x_ref):
    """Conditional de-sequentialized SMC sweep over the posterior parameter mixture."""
    contexts = ctx.contexts
    init_means = ctx.init_means
    init_chols = ctx.init_chols
    init_logdets = ctx.init_logdets
    logpi = ctx.initial_label_log_probs
    runtime_observations = ctx.runtime_observations
    obs_increment_fn = ctx.obs_increment_fn
    trajectory_label_log_probs = ctx.trajectory_label_log_probs
    num_steps = ctx.num_steps
    num_parameter_particles = ctx.num_parameter_particles
    num_particles = ctx.num_free_particles + 1
    num_free_particles = ctx.num_free_particles
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    latent_dim = int(init_means.shape[-1])
    dsmc_leaf_proposal = ctx.dsmc_leaf_proposal
    # Both leaves (amala_exact, paid_mix) draw the auxiliary trajectory z and pay its
    # pseudo-observation potential; paid_mix additionally mixes in the fixed pilot
    # and wide components (whose reference-independence needs no payment).
    is_paid_mix = dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PAID_MIX
    amala_delta = jnp.asarray(ctx.amala_delta, dtype=latent_dtype)
    proposal_var_by_t = jnp.asarray(0.5, dtype=latent_dtype) * amala_delta
    proposal_scale_by_t = jnp.sqrt(proposal_var_by_t)
    proposal_kappa = jnp.asarray(ctx.amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(ctx.amala_grad_clip, dtype=latent_dtype)

    # Coordinate-block restriction: propose only `latent_block_coords` randomly
    # chosen coordinates this sweep (the same block at every time step — a whole-path
    # coordinate-conditional update); all other coordinates are copied from the
    # reference into every particle. The seams still evaluate the full transition
    # density, so cross-coordinate coupling is priced exactly and the sweep is
    # conditional SMC on the coordinate-block conditional of the same target.
    block_coords = ctx.latent_block_coords
    if block_coords is not None and block_coords < latent_dim:
        mask_key = random.fold_in(key, 1)
        chosen_coords = random.permutation(mask_key, latent_dim)[:block_coords]
        coord_mask = jnp.zeros((latent_dim,), dtype=bool).at[chosen_coords].set(True)
        proposed_dim = int(block_coords)
    else:
        coord_mask = None
        proposed_dim = latent_dim

    def _per_param_gaussian_log_probs(values, means, chols, logdets):
        per_param = jax.vmap(
            lambda mean, chol, logdet: _gaussian_log_prob_shared_cholesky(
                values, mean, chol, logdet
            )
        )(means, chols, logdets)
        return jnp.swapaxes(per_param, 0, 1)

    def _log_isotropic_density(
        values: jnp.ndarray,
        mean: jnp.ndarray,
        proposal_var_t: jnp.ndarray,
    ) -> jnp.ndarray:
        diff = values - mean
        squared = diff * diff
        if coord_mask is not None:
            squared = jnp.where(coord_mask, squared, jnp.zeros_like(squared))
        quadratic = jnp.sum(squared, axis=-1) / proposal_var_t
        return -0.5 * (proposed_dim * jnp.log(2.0 * jnp.pi * proposal_var_t) + quadratic)

    def _log_diagonal_density(
        values: jnp.ndarray,
        mean_t: jnp.ndarray,
        var_t: jnp.ndarray,
    ) -> jnp.ndarray:
        diff = values - mean_t
        per_coord = -0.5 * (jnp.log(2.0 * jnp.pi * var_t) + diff * diff / var_t)
        if coord_mask is not None:
            per_coord = jnp.where(coord_mask, per_coord, jnp.zeros_like(per_coord))
        return jnp.sum(per_coord, axis=-1)

    def _clip_gradient(grad: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        norm = jnp.linalg.norm(grad)
        multiplier = jnp.minimum(
            jnp.asarray(1.0, dtype=latent_dtype),
            grad_clip / jnp.maximum(norm, jnp.asarray(1e-12, dtype=latent_dtype)),
        )
        return (grad * multiplier).astype(latent_dtype), norm.astype(traj_dtype)

    def _obs_value_grad_by_param(particle_t: jnp.ndarray, time_idx: jnp.ndarray):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: obs_increment_fn(
                    context,
                    particle,
                    time_idx,
                    runtime_observations,
                )
            )(particle_t)

        value, grad = jax.vmap(_one_context)(contexts)
        return value.astype(traj_dtype), grad.astype(latent_dtype)

    _initial_prior_value_grad_by_param = ctx.initial_value_grad_by_param
    _transition_current_value_grad_by_param = ctx.transition_current_value_grad_by_param
    _transition_next_value_grad_by_param = ctx.transition_next_value_grad_by_param

    # Linearisation points for the gradient leaf: draw an auxiliary trajectory
    # z ~ N(x_ref, (delta/2) I) and linearise there; the matching N(z_t; x_t,
    # (delta/2) I) pseudo-observation potential is added to the leaf weight in
    # _leaf, recovering the exact gradient-informed cSMC proposal of section
    # 3.3.1 in docs/papers/auxiliary-kalman-samplers.pdf. Centring on x_ref
    # directly (the historical amala/amala_plus variants) is reference-dependent
    # without an auxiliary correction and does not leave the target invariant.
    aux_key = random.fold_in(key, 0)
    lin_pts = x_ref + proposal_scale_by_t[:, None] * random.normal(
        aux_key, x_ref.shape, dtype=latent_dtype
    )

    if is_paid_mix:
        pilot_means = ctx.pilot_means
        pilot_vars = ctx.pilot_vars
        pilot_wide_vars = ctx.pilot_wide_vars
        pilot_scales = jnp.sqrt(pilot_vars)
        pilot_wide_scales = jnp.sqrt(pilot_wide_vars)
        mix_weights = jnp.asarray(
            [
                ctx.paid_mix_z_weight,
                ctx.paid_mix_pilot_weight,
                1.0 - ctx.paid_mix_z_weight - ctx.paid_mix_pilot_weight,
            ],
            dtype=latent_dtype,
        )
        mix_log_weights = jnp.log(mix_weights)

    def _gradient_leaf_proposal_stats(time_idx: jnp.ndarray):
        particle_t = lin_pts[time_idx]
        obs_lp, obs_grad = _obs_value_grad_by_param(particle_t, time_idx)
        init_prior_lp, init_prior_grad = _initial_prior_value_grad_by_param(particle_t)
        prev_idx = jnp.maximum(time_idx - 1, 0)
        transition_lp, transition_grad = _transition_current_value_grad_by_param(
            lin_pts[prev_idx],
            particle_t,
            time_idx,
        )
        is_initial = time_idx == 0
        logits = jnp.where(
            is_initial,
            logpi + init_prior_lp + obs_lp,
            logpi + transition_lp + obs_lp,
        )
        component_grad = jnp.where(
            is_initial,
            init_prior_grad + obs_grad,
            transition_grad + obs_grad,
        )
        next_idx = jnp.minimum(time_idx + 1, num_steps - 1)
        future_lp, future_grad = _transition_next_value_grad_by_param(
            particle_t,
            lin_pts[next_idx],
            next_idx,
        )
        has_future = time_idx < (num_steps - 1)
        logits = jnp.where(has_future, logits + future_lp, logits)
        component_grad = jnp.where(
            has_future,
            component_grad + future_grad,
            component_grad,
        )
        label_log_probs = _normalize_log_probs(logits)
        label_probs = jnp.exp(label_log_probs).astype(latent_dtype)
        grad = jnp.einsum("p,pd->d", label_probs, component_grad)
        clipped_grad, grad_norm = _clip_gradient(grad)
        drift = (proposal_kappa * proposal_var_by_t[time_idx] * clipped_grad).astype(latent_dtype)
        return (particle_t + drift).astype(latent_dtype), grad_norm

    proposal_centers, proposal_grad_norms = jax.vmap(_gradient_leaf_proposal_stats)(
        jnp.arange(num_steps, dtype=jnp.int32)
    )

    def _leaf(time_idx, leaf_key):
        component_key, sample_key = random.split(leaf_key, 2)
        noise = random.normal(
            sample_key,
            (num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        z_component = proposal_centers[time_idx] + proposal_scale_by_t[time_idx] * noise
        if is_paid_mix:
            component = random.categorical(
                component_key, mix_log_weights, shape=(num_free_particles,)
            )
            pilot_component = pilot_means[time_idx] + pilot_scales[time_idx] * noise
            wide_component = pilot_means[time_idx] + pilot_wide_scales[time_idx] * noise
            free_particles = jnp.where(
                (component == 0)[:, None],
                z_component,
                jnp.where((component == 1)[:, None], pilot_component, wide_component),
            )
        else:
            free_particles = z_component
        if coord_mask is not None:
            free_particles = jnp.where(
                coord_mask[None, :], free_particles, x_ref[time_idx][None, :]
            )
        particles = jnp.concatenate([x_ref[time_idx][None, :], free_particles], axis=0)
        obs_lp = _observation_log_probs_by_param(
            contexts,
            particles,
            jnp.asarray(time_idx, dtype=jnp.int32),
            runtime_observations,
            obs_increment_fn,
        )
        z_proposal_lp = _log_isotropic_density(
            particles,
            proposal_centers[time_idx],
            proposal_var_by_t[time_idx],
        )
        if is_paid_mix:
            pilot_lp = _log_diagonal_density(particles, pilot_means[time_idx], pilot_vars[time_idx])
            wide_lp = _log_diagonal_density(
                particles, pilot_means[time_idx], pilot_wide_vars[time_idx]
            )
            proposal_lp = jax.scipy.special.logsumexp(
                jnp.stack(
                    [
                        mix_log_weights[0] + z_proposal_lp,
                        mix_log_weights[1] + pilot_lp,
                        mix_log_weights[2] + wide_lp,
                    ]
                ),
                axis=0,
            )
        else:
            proposal_lp = z_proposal_lp
        init_prior_lp = _per_param_gaussian_log_probs(
            particles, init_means, init_chols, init_logdets
        )
        # Pseudo-observation potential N(z_t; x_t, (delta/2) I) of the auxiliary
        # extended target (label-independent, so it factors through the parameter
        # logsumexp). With the q(x_t | z_t) proposal density already subtracted
        # above, this is the exact target/proposal leaf weight; the pilot/wide
        # mixture components are reference-independent so the same payment covers
        # paid_mix.
        aux_lp = _log_isotropic_density(particles, lin_pts[time_idx], proposal_var_by_t[time_idx])
        tail_psi = obs_lp - proposal_lp[:, None] + aux_lp[:, None]
        initial_psi = init_prior_lp + tail_psi
        psi = jnp.where(time_idx == 0, initial_psi, tail_psi).astype(traj_dtype)
        evidence = jax.scipy.special.logsumexp(logpi[None, :] + psi, axis=1)
        log_weights = _normalize_log_probs(evidence).astype(traj_dtype)
        origin = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32)[:, None], (num_particles, 1)
        )
        return particles.astype(latent_dtype), psi, origin, log_weights

    def _phantom_leaf():
        """Evidence-free padding leaf: psi = 0, uniform weights, sliced off the output."""
        particles = jnp.zeros((num_particles, latent_dim), dtype=latent_dtype)
        psi = jnp.zeros((num_particles, num_parameter_particles), dtype=traj_dtype)
        origin = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32)[:, None], (num_particles, 1)
        )
        log_weights = jnp.full((num_particles,), -math.log(num_particles), dtype=traj_dtype)
        return particles, psi, origin, log_weights

    def _multinomial(draw_key, logits, num_draws):
        probabilities = jax.nn.softmax(logits)
        cumulative = jnp.cumsum(probabilities)
        uniforms = random.uniform(draw_key, (num_draws,), dtype=cumulative.dtype)
        indices = jnp.searchsorted(cumulative, uniforms, side="right")
        return jnp.minimum(indices, logits.shape[0] - 1).astype(jnp.int32)

    def _selected_transition_log_probs(prev_particles, next_particles, seam):
        return ctx.selected_transition_log_probs(prev_particles, next_particles, seam)

    def _stitch_logits(left, right, seam):
        _, left_last, left_psi, _, left_weights = left
        right_first, _, right_psi, _, right_weights = right
        transition_lp = ctx.pairwise_transition_log_probs(left_last, right_first, seam)
        log_joint = jax.scipy.special.logsumexp(
            logpi[None, None, :] + left_psi[:, None, :] + right_psi[None, :, :] + transition_lp,
            axis=-1,
        )
        log_left = jax.scipy.special.logsumexp(logpi[None, :] + left_psi, axis=1)
        log_right = jax.scipy.special.logsumexp(logpi[None, :] + right_psi, axis=1)
        seam_coupling = log_joint - log_left[:, None] - log_right[None, :]
        pair_logits = left_weights[:, None] + right_weights[None, :] + seam_coupling
        return pair_logits.astype(traj_dtype)

    def _combine(left, right, seam, combine_key):
        pair_logits = _stitch_logits(left, right, seam)
        free_pairs = _multinomial(combine_key, pair_logits.reshape(-1), num_free_particles)
        selected = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), free_pairs], axis=0)
        left_idx = selected // num_particles
        right_idx = selected % num_particles
        left_first, left_last, left_psi, left_origin, _ = left
        right_first, right_last, right_psi, right_origin, _ = right
        first = left_first[left_idx]
        last = right_last[right_idx]
        origin = jnp.concatenate([left_origin[left_idx], right_origin[right_idx]], axis=1)
        transition_lp = _selected_transition_log_probs(
            left_last[left_idx],
            right_first[right_idx],
            seam,
        )
        psi = (left_psi[left_idx] + right_psi[right_idx] + transition_lp).astype(traj_dtype)
        log_weights = jnp.full((num_particles,), -math.log(num_particles), dtype=traj_dtype)
        return first, last, psi, origin, log_weights

    with jax.named_scope("dsmc_tree"):
        depth = max((num_steps - 1).bit_length(), 0)
        padded_steps = 1 << depth
        key_leaves, key_tree, key_root = random.split(key, 3)
        leaf_keys = random.split(key_leaves, num_steps)
        with jax.named_scope("dsmc_leaves"):
            leaf_particles, leaf_psi, leaf_origin, leaf_weights = jax.vmap(_leaf)(
                jnp.arange(num_steps, dtype=jnp.int32),
                leaf_keys,
            )
        if num_steps == 1:
            evidence = jax.scipy.special.logsumexp(logpi[None, :] + leaf_psi[0], axis=1)
            chosen = _multinomial(key_root, evidence, 1)[0]
            latent_path = leaf_particles[0, chosen][None, :]
            origin_path = leaf_origin[0, chosen]
        else:
            phantom_particles, phantom_psi, phantom_origin, phantom_weights = _phantom_leaf()
            num_phantom = padded_steps - num_steps
            phantom_particles = jnp.broadcast_to(
                phantom_particles,
                (num_phantom, num_particles, latent_dim),
            )
            phantom_psi = jnp.broadcast_to(
                phantom_psi,
                (num_phantom, num_particles, num_parameter_particles),
            )
            phantom_origin = jnp.broadcast_to(
                phantom_origin,
                (num_phantom, num_particles, 1),
            )
            phantom_weights = jnp.broadcast_to(
                phantom_weights,
                (num_phantom, num_particles),
            )
            first = jnp.concatenate([leaf_particles, phantom_particles], axis=0)
            last = first
            psi = jnp.concatenate([leaf_psi, phantom_psi], axis=0)
            origin = jnp.concatenate([leaf_origin, phantom_origin], axis=0)
            weights = jnp.concatenate([leaf_weights, phantom_weights], axis=0)
            level_keys = random.split(key_tree, max(depth - 1, 1))
            segments = padded_steps
            for level in range(depth - 1):
                num_pairs = segments // 2
                seams = (1 << level) + jnp.arange(num_pairs, dtype=jnp.int32) * (1 << (level + 1))
                pair_keys = random.split(level_keys[level], num_pairs)
                left = (
                    first[0::2],
                    last[0::2],
                    psi[0::2],
                    origin[0::2],
                    weights[0::2],
                )
                right = (
                    first[1::2],
                    last[1::2],
                    psi[1::2],
                    origin[1::2],
                    weights[1::2],
                )
                with jax.named_scope("dsmc_combine"):
                    first, last, psi, origin, weights = jax.vmap(_combine, in_axes=(0, 0, 0, 0))(
                        left, right, seams, pair_keys
                    )
                segments = num_pairs
            left_root = (first[0], last[0], psi[0], origin[0], weights[0])
            right_root = (first[1], last[1], psi[1], origin[1], weights[1])
            with jax.named_scope("dsmc_combine"):
                pair_logits = _stitch_logits(left_root, right_root, padded_steps // 2)
            chosen = _multinomial(key_root, pair_logits.reshape(-1), 1)[0]
            left_idx = chosen // num_particles
            right_idx = chosen % num_particles
            origin_path = jnp.concatenate([origin[0][left_idx], origin[1][right_idx]], axis=0)[
                :num_steps
            ]
            latent_path = leaf_particles[jnp.arange(num_steps, dtype=jnp.int32), origin_path]

    latent_path = latent_path.astype(latent_dtype)
    final_label_log_probs = trajectory_label_log_probs(latent_path)
    diagnostics = {
        "amala_grad_norm_mean": jnp.mean(proposal_grad_norms).astype(traj_dtype),
        "amala_grad_norm_max": jnp.max(proposal_grad_norms).astype(traj_dtype),
    }
    return MPGibbsLatentSmootherResult(
        latent_path=latent_path,
        final_label_log_probs=final_label_log_probs,
        origin_path=origin_path.astype(jnp.int32),
        diagnostics=diagnostics,
    )
