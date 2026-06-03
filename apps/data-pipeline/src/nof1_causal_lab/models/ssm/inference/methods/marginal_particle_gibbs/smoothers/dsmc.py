"""De-sequentialized conditional-SMC latent smoother (parallel-in-time)."""

# References:
#   docs/papers/desequentialized-smc.pdf — Corenflos, Chopin & Särkkä (2022),
#     arXiv:2202.02264: de-Sequentialized Monte Carlo. The smoothing distribution
#     is built by a divide-and-conquer binary tree over time (Prop. 2.2 stitching,
#     eq. 10-11), instead of the sequential forward filter + backward sampling used
#     by plain_csmc. We implement the conditional formulation c-dSMC (Section 3,
#     Algorithm 4): the reference trajectory is preserved through every block
#     combination, making this a valid parallel-in-time conditional-SMC Particle
#     Gibbs kernel. No backward-sampling step is needed — the balanced tree resamples
#     every time step at most O(log T) times, so there is no genealogy degeneracy.
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf — Corenflos (2025), arXiv:2505.04611:
#     conditional SMC against the posterior mixture over the parameter ensemble. The
#     seam weights and trajectory evidence are marginalized over the parameter labels
#     exactly as in the sibling smoothers (logsumexp over the parameter axis).
#
# Leaf proposals are the per-parameter prior-predictive marginals (a Gaussian mixture
# weighted by the parameter posterior), assembled here from the existing discrete
# dynamics in the context. This deliberately avoids the parallel-in-time Gaussian
# approximation of Algorithm 5 (Section 4.1): independent proposals only, no extra
# context machinery.
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
    _DSMC_LEAF_PROPOSAL_AMALA,
    _DSMC_LEAF_PROPOSAL_AMALA_PLUS,
    _DSMC_LEAF_PROPOSAL_PRIOR_PREDICTIVE,
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _cholesky_batch,
    _gaussian_log_prob_shared_cholesky,
    _logdet_from_cholesky,
    _normalize_log_probs,
    _observation_log_probs_by_param,
    _sample_gaussian_from_chol,
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
    amala_delta = jnp.asarray(ctx.amala_delta, dtype=latent_dtype)
    proposal_var_by_t = jnp.asarray(0.5, dtype=latent_dtype) * amala_delta
    proposal_scale_by_t = jnp.sqrt(proposal_var_by_t)
    proposal_kappa = jnp.asarray(ctx.amala_kappa, dtype=latent_dtype)
    grad_clip = jnp.asarray(ctx.amala_grad_clip, dtype=latent_dtype)

    def _prior_predictive_marginals():
        """Roll the discrete dynamics forward without observations, per parameter.

        Computed by a parallel-prefix scan over the affine-Gaussian transition
        operator ``(A, c, Q)``: composing the map ``x -> A x + c`` (noise cov ``Q``)
        is associative, so ``jax.lax.associative_scan`` yields the same marginals as
        a sequential roll-forward in O(log T) depth.
        """
        init_covs = jax.vmap(lambda chol: chol @ chol.T)(init_chols)
        identity = jnp.broadcast_to(
            jnp.eye(latent_dim, dtype=init_means.dtype),
            (num_parameter_particles, latent_dim, latent_dim),
        )
        zero_shift = jnp.zeros((num_parameter_particles, latent_dim), dtype=init_means.dtype)
        zero_noise = jnp.zeros(
            (num_parameter_particles, latent_dim, latent_dim), dtype=init_means.dtype
        )
        # Element t is the transition into time t; element 0 is the identity so the
        # inclusive prefix at t reproduces x_0 -> x_t. Ad[:, 0] is never a real
        # transition, so it is overwritten with the identity.
        drift = jnp.swapaxes(contexts.Ad, 0, 1).at[0].set(identity)
        shift = jnp.swapaxes(contexts.cd, 0, 1).at[0].set(zero_shift)
        noise = jnp.swapaxes(contexts.Qd, 0, 1).at[0].set(zero_noise)

        def _compose(earlier, later):
            drift_a, shift_a, noise_a = earlier
            drift_b, shift_b, noise_b = later
            composed_drift = jnp.einsum("...ij,...jk->...ik", drift_b, drift_a)
            composed_shift = jnp.einsum("...ij,...j->...i", drift_b, shift_a) + shift_b
            composed_noise = (
                jnp.einsum("...ij,...jk,...lk->...il", drift_b, noise_a, drift_b) + noise_b
            )
            return composed_drift, composed_shift, composed_noise

        drift_cum, shift_cum, noise_cum = jax.lax.associative_scan(
            _compose, (drift, shift, noise), axis=0
        )
        means = jnp.einsum("tpij,pj->tpi", drift_cum, init_means) + shift_cum
        covs = jnp.einsum("tpij,pjk,tplk->tpil", drift_cum, init_covs, drift_cum) + noise_cum
        chols = _cholesky_batch(
            covs.reshape(num_steps * num_parameter_particles, latent_dim, latent_dim)
        ).reshape(num_steps, num_parameter_particles, latent_dim, latent_dim)
        logdets = _logdet_from_cholesky(chols)
        return means, chols, logdets

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
        quadratic = jnp.sum(diff * diff, axis=-1) / proposal_var_t
        return -0.5 * (latent_dim * jnp.log(2.0 * jnp.pi * proposal_var_t) + quadratic)

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

    if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PRIOR_PREDICTIVE:
        prior_proposal_means, prior_proposal_chols, prior_proposal_logdets = (
            _prior_predictive_marginals()
        )

        def _proposal_log_probs(values, time_idx):
            per_param = _per_param_gaussian_log_probs(
                values,
                prior_proposal_means[time_idx],
                prior_proposal_chols[time_idx],
                prior_proposal_logdets[time_idx],
            )
            return jax.scipy.special.logsumexp(logpi[None, :] + per_param, axis=1)

    else:

        def _gradient_leaf_proposal_stats(time_idx: jnp.ndarray):
            particle_t = x_ref[time_idx]
            obs_lp, obs_grad = _obs_value_grad_by_param(particle_t, time_idx)
            init_prior_lp, init_prior_grad = _initial_prior_value_grad_by_param(particle_t)
            prev_idx = jnp.maximum(time_idx - 1, 0)
            transition_lp, transition_grad = _transition_current_value_grad_by_param(
                x_ref[prev_idx],
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
            if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_AMALA_PLUS:
                next_idx = jnp.minimum(time_idx + 1, num_steps - 1)
                future_lp, future_grad = _transition_next_value_grad_by_param(
                    particle_t,
                    x_ref[next_idx],
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
            drift = (proposal_kappa * proposal_var_by_t[time_idx] * clipped_grad).astype(
                latent_dtype
            )
            return (particle_t + drift).astype(latent_dtype), grad_norm

        proposal_centers, proposal_grad_norms = jax.vmap(_gradient_leaf_proposal_stats)(
            jnp.arange(num_steps, dtype=jnp.int32)
        )

    def _leaf(time_idx, leaf_key):
        component_key, sample_key = random.split(leaf_key, 2)
        if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PRIOR_PREDICTIVE:
            free_components = random.categorical(
                component_key, logpi, shape=(num_free_particles,)
            ).astype(jnp.int32)
            free_particles = _sample_gaussian_from_chol(
                sample_key,
                prior_proposal_means[time_idx][free_components],
                prior_proposal_chols[time_idx][free_components],
            )
        else:
            del component_key
            free_particles = proposal_centers[time_idx] + proposal_scale_by_t[
                time_idx
            ] * random.normal(
                sample_key,
                (num_free_particles, latent_dim),
                dtype=latent_dtype,
            )
        particles = jnp.concatenate([x_ref[time_idx][None, :], free_particles], axis=0)
        obs_lp = _observation_log_probs_by_param(
            contexts,
            particles,
            jnp.asarray(time_idx, dtype=jnp.int32),
            runtime_observations,
            obs_increment_fn,
        )
        if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PRIOR_PREDICTIVE:
            proposal_lp = _proposal_log_probs(particles, time_idx)
        else:
            proposal_lp = _log_isotropic_density(
                particles,
                proposal_centers[time_idx],
                proposal_var_by_t[time_idx],
            )
        init_prior_lp = _per_param_gaussian_log_probs(
            particles, init_means, init_chols, init_logdets
        )
        tail_psi = obs_lp - proposal_lp[:, None]
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
    if dsmc_leaf_proposal in {
        _DSMC_LEAF_PROPOSAL_AMALA,
        _DSMC_LEAF_PROPOSAL_AMALA_PLUS,
    }:
        diagnostics = {
            "amala_grad_norm_mean": jnp.mean(proposal_grad_norms).astype(traj_dtype),
            "amala_grad_norm_max": jnp.max(proposal_grad_norms).astype(traj_dtype),
        }
    else:
        diagnostics = {}
    return MPGibbsLatentSmootherResult(
        latent_path=latent_path,
        final_label_log_probs=final_label_log_probs,
        origin_path=origin_path.astype(jnp.int32),
        diagnostics=diagnostics,
    )
