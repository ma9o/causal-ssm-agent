"""Particle latent Gibbs kernels for complete-data SSM inference."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.auxiliary_kalman import (
    _AUX_JITTER,
    _gaussian_log_prob_isotropic,
    _initial_latent_moments,
    _tame_gradient_tulac,
)


class _DSMCState(NamedTuple):
    trajectories: jnp.ndarray
    log_weights: jnp.ndarray
    origins: jnp.ndarray
    keys: jnp.ndarray
    time_indices: jnp.ndarray


def _gaussian_log_prob_full(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
) -> jnp.ndarray:
    covariance = symmetrize_with_jitter(covariance, jitter=_AUX_JITTER)
    chol = jnp.linalg.cholesky(covariance)
    diff = value - mean
    whitened = jla.solve_triangular(chol, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = diff.shape[-1]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)


def _next_power_of_2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _pad_dsmc_leaf(leaf: jnp.ndarray, *, padded_length: int, original_length: int) -> jnp.ndarray:
    if padded_length == original_length:
        return leaf
    constant = 0 if jnp.issubdtype(leaf.dtype, jnp.integer) else jnp.nan
    pad_width = [(0, padded_length - original_length)] + [(0, 0)] * (leaf.ndim - 1)
    return jnp.pad(leaf, pad_width, constant_values=constant)


def _reshape_dsmc_leaf(
    leaf: jnp.ndarray,
    *,
    segment_length: int,
    original_shape: tuple[int, ...],
) -> jnp.ndarray:
    return jnp.reshape(leaf, (-1, segment_length, *original_shape[1:]))


def _concat_dsmc_segments(left: _DSMCState, right: _DSMCState) -> _DSMCState:
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), left, right)


def _dsmc_passthrough_batch(left: _DSMCState, right: _DSMCState) -> _DSMCState:
    return jax.vmap(_concat_dsmc_segments)(left, right)


def _conditional_flat_multinomial(
    key: jnp.ndarray,
    log_weights: jnp.ndarray,
    *,
    num_particles: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    free_idx = random.categorical(
        key,
        jnp.reshape(log_weights, (-1,)),
        shape=(num_particles - 1,),
    ).astype(jnp.int32)
    idx = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), free_idx], axis=0)
    left_idx = idx // num_particles
    right_idx = idx - num_particles * left_idx
    return left_idx.astype(jnp.int32), right_idx.astype(jnp.int32)


def _dsmc_tree_reduce(
    initial_state: _DSMCState,
    combine_operator,
    *,
    num_steps: int,
) -> _DSMCState:
    padded_length = _next_power_of_2(num_steps)
    trajectory_shape = tuple(initial_state.trajectories.shape)
    log_weight_shape = tuple(initial_state.log_weights.shape)
    origin_shape = tuple(initial_state.origins.shape)
    key_shape = tuple(initial_state.keys.shape)
    time_index_shape = tuple(initial_state.time_indices.shape)
    state = jax.tree_util.tree_map(
        lambda leaf: _pad_dsmc_leaf(
            leaf,
            padded_length=padded_length,
            original_length=num_steps,
        ),
        initial_state,
    )

    indices = np.arange(padded_length)
    num_levels = int(math.log2(padded_length))
    for level in range(num_levels):
        segment_length = 2**level
        state = _DSMCState(
            trajectories=_reshape_dsmc_leaf(
                state.trajectories,
                segment_length=segment_length,
                original_shape=trajectory_shape,
            ),
            log_weights=_reshape_dsmc_leaf(
                state.log_weights,
                segment_length=segment_length,
                original_shape=log_weight_shape,
            ),
            origins=_reshape_dsmc_leaf(
                state.origins,
                segment_length=segment_length,
                original_shape=origin_shape,
            ),
            keys=_reshape_dsmc_leaf(
                state.keys,
                segment_length=segment_length,
                original_shape=key_shape,
            ),
            time_indices=_reshape_dsmc_leaf(
                state.time_indices,
                segment_length=segment_length,
                original_shape=time_index_shape,
            ),
        )
        indices = np.reshape(indices, (-1, segment_length))
        even_state = jax.tree_util.tree_map(lambda leaf: leaf[::2], state)
        odd_state = jax.tree_util.tree_map(lambda leaf: leaf[1::2], state)
        even_indices = indices[::2]
        odd_indices = indices[1::2]
        valid_mask = np.logical_and(
            even_indices[:, -1] < num_steps,
            odd_indices[:, 0] < num_steps,
        )

        invalid_mask = ~valid_mask
        valid_left = jax.tree_util.tree_map(
            lambda leaf, mask=valid_mask: leaf[mask],
            even_state,
        )
        valid_right = jax.tree_util.tree_map(
            lambda leaf, mask=valid_mask: leaf[mask],
            odd_state,
        )
        invalid_left = jax.tree_util.tree_map(
            lambda leaf, mask=invalid_mask: leaf[mask],
            even_state,
        )
        invalid_right = jax.tree_util.tree_map(
            lambda leaf, mask=invalid_mask: leaf[mask],
            odd_state,
        )

        combined = jax.vmap(combine_operator)(valid_left, valid_right)
        passthrough = _dsmc_passthrough_batch(invalid_left, invalid_right)
        state = jax.tree_util.tree_map(
            lambda left, right: jnp.concatenate([left, right], axis=0),
            combined,
            passthrough,
        )

    return _DSMCState(
        trajectories=state.trajectories[0, :num_steps, ...].reshape(trajectory_shape),
        log_weights=state.log_weights[0, :num_steps, ...].reshape(log_weight_shape),
        origins=state.origins[0, :num_steps, ...].reshape(origin_shape),
        keys=state.keys[0, :num_steps, ...].reshape(key_shape),
        time_indices=state.time_indices[0, :num_steps, ...].reshape(time_index_shape),
    )


def build_pit_particle_mgrad_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
    num_particles: int,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> dict[str, Any]:
    """Build the PIT Particle-mGRAD latent update from Corenflos & Finke (2024).

    This uses the Corenflos-Chopin-Särkkä de-sequentialized conditional particle
    smoother inside the PIT Particle-mGRAD latent block:

    1. draw pseudo-observations ``u_t ~ N(x_t + (δ_t/2) ∇ log G_t(x_t), (δ_t/2) I)``
       from the current reference trajectory,
    2. sample independent per-time particles from ``q_t(x_t) = N(u_t, (δ_t/2)I)``,
    3. stitch partial trajectories with a divide-and-conquer conditional dSMC
       operator whose boundary weight is ``p(x_t | x_{t-1}) G_t(x_t) / q_t(x_t)``.

    The reference trajectory is preserved as particle 0 throughout the tree,
    yielding a particle-Gibbs latent transition with O(log T) parallel depth.
    """
    if num_particles < 2:
        raise ValueError("pit_particle_mgrad requires num_particles >= 2.")

    obs_increment_fn = bundle["observation_increment_log_prob_from_context_fn"]
    _raw_obs_increment_grad_fn = jax.grad(obs_increment_fn, argnums=1)

    def obs_increment_grad_fn(context, latent_t, time_idx):
        """TULAc-tamed observation log-prob gradient (fixed h, see _TULAC_H).

        Same fix as `_tame_gradient_tulac` used in aux_kalman_mcmc: bounds the
        gradient-augmented pseudo-observation perturbation so it can shrink
        with adaptation. Applied identically to every grad call so the MH
        ratio remains valid (proposal kernel just becomes a different but
        still-valid kernel).
        """
        return _tame_gradient_tulac(_raw_obs_increment_grad_fn(context, latent_t, time_idx))

    ref_particle_index = 0
    num_free_particles = num_particles - 1

    def _latent_pit_particle_mgrad_step(state, key: jnp.ndarray):
        aux_key, proposal_key, combine_key, select_key = random.split(key, 4)
        x_ref = state.latent_trajectory
        context = state.latent_context
        latent_dtype = x_ref.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        latent_dim = int(x_ref.shape[-1])
        num_steps = int(x_ref.shape[0])
        delta_per_t = jnp.broadcast_to(
            jnp.asarray(state.latent_delta, dtype=latent_dtype),
            (num_steps,),
        )
        auxiliary_var = 0.5 * delta_per_t
        auxiliary_std = jnp.sqrt(auxiliary_var)[:, None]
        time_indices = jnp.arange(num_steps, dtype=jnp.int32)
        ref_grads = jax.vmap(
            lambda latent_t, time_idx: jnp.asarray(
                obs_increment_grad_fn(context, latent_t, time_idx),
                dtype=latent_dtype,
            )
        )(x_ref, time_indices)
        u = x_ref + auxiliary_var[:, None] * ref_grads
        u = u + auxiliary_std * random.normal(aux_key, x_ref.shape, dtype=latent_dtype)

        proposal_eps = random.normal(
            proposal_key,
            (num_steps, num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        free_particles = u[:, None, :] + auxiliary_std[:, None, :] * proposal_eps
        x_particles = jnp.concatenate([x_ref[:, None, :], free_particles], axis=1)

        proposal_log_probs = jax.vmap(
            lambda particles_t, u_t, var_t: jax.vmap(
                lambda particle: _gaussian_log_prob_isotropic(particle, u_t, var_t)
            )(particles_t)
        )(x_particles, u, auxiliary_var)

        init_mean, init_cov = _initial_latent_moments(context)
        init_prior_lp = jax.vmap(
            lambda particle: _gaussian_log_prob_full(particle, init_mean, init_cov)
        )(x_particles[0])
        init_obs_lp = jax.vmap(
            lambda particle: obs_increment_fn(context, particle, jnp.asarray(0, dtype=jnp.int32))
        )(x_particles[0])
        log_weights0 = init_prior_lp + init_obs_lp - proposal_log_probs[0]
        log_weights_rest = jnp.zeros((num_steps - 1, num_particles), dtype=log_weights0.dtype)
        log_weights = jnp.concatenate(
            [log_weights0[None, :], log_weights_rest],
            axis=0,
        )

        normalizers = jax.scipy.special.logsumexp(log_weights, axis=1)
        log_weights = log_weights - normalizers[:, None]
        origins = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32)[None, :],
            (num_steps, num_particles),
        )
        combine_keys = random.split(combine_key, num_steps)
        initial_state = _DSMCState(
            trajectories=x_particles,
            log_weights=log_weights.astype(traj_dtype),
            origins=origins,
            keys=combine_keys,
            time_indices=time_indices,
        )

        def _combine_segments(left: _DSMCState, right: _DSMCState) -> _DSMCState:
            stitch_key = right.keys[0]
            time_idx = right.time_indices[0]
            left_particles = left.trajectories[-1]
            right_particles = right.trajectories[0]
            left_log_weights = left.log_weights[-1]
            right_log_weights = right.log_weights[0]
            pred_means = left_particles @ context.Ad[time_idx].T + context.cd[time_idx]
            transition_lp = jax.vmap(
                lambda mean_left: jax.vmap(
                    lambda particle_right: _gaussian_log_prob_full(
                        particle_right,
                        mean_left,
                        context.Qd[time_idx],
                    )
                )(right_particles)
            )(pred_means)
            obs_lp = jax.vmap(lambda particle: obs_increment_fn(context, particle, time_idx))(
                right_particles
            )
            proposal_lp = jax.vmap(
                lambda particle: _gaussian_log_prob_isotropic(
                    particle,
                    u[time_idx],
                    auxiliary_var[time_idx],
                )
            )(right_particles)
            log_pair_weights = (
                left_log_weights[:, None]
                + right_log_weights[None, :]
                + transition_lp
                + obs_lp[None, :]
                - proposal_lp[None, :]
            )
            left_idx, right_idx = _conditional_flat_multinomial(
                stitch_key,
                log_pair_weights,
                num_particles=num_particles,
            )
            left_trajectories = jnp.take(left.trajectories, left_idx, axis=1)
            right_trajectories = jnp.take(right.trajectories, right_idx, axis=1)
            left_origins = jnp.take(left.origins, left_idx, axis=1)
            right_origins = jnp.take(right.origins, right_idx, axis=1)
            merged_length = left.log_weights.shape[0] + right.log_weights.shape[0]
            merged_log_weights = jnp.full(
                (merged_length, num_particles),
                -jnp.log(jnp.asarray(num_particles, dtype=traj_dtype)),
                dtype=traj_dtype,
            )
            return _DSMCState(
                trajectories=jnp.concatenate([left_trajectories, right_trajectories], axis=0),
                log_weights=merged_log_weights,
                origins=jnp.concatenate([left_origins, right_origins], axis=0),
                keys=jnp.concatenate([left.keys, right.keys], axis=0),
                time_indices=jnp.concatenate([left.time_indices, right.time_indices], axis=0),
            )

        final_state = _dsmc_tree_reduce(
            initial_state,
            _combine_segments,
            num_steps=num_steps,
        )
        final_idx = random.categorical(select_key, final_state.log_weights[-1]).astype(jnp.int32)
        latent_path = final_state.trajectories[:, final_idx, :]
        origin_path = final_state.origins[:, final_idx]

        prior_terms = bundle["prior_terms_from_context_fn"](context)
        next_traj_lp = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](context, latent_path, prior_terms),
            dtype=traj_dtype,
        )
        log_prior_z = jnp.asarray(
            bundle["log_prior_unc_fn"](state.position),
            dtype=complete_dtype,
        )
        next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
        updated_mask = (origin_path != ref_particle_index).astype(state.position.dtype)
        latent_move = latent_path - x_ref
        latent_move_rms_per_t = jnp.sqrt(jnp.mean(latent_move * latent_move, axis=-1))
        latent_move_rms = jnp.sqrt(jnp.mean(latent_move * latent_move))
        latent_move_max_abs = jnp.max(jnp.abs(latent_move))

        next_state = state._replace(
            latent_trajectory=latent_path,
            trajectory_log_prob=next_traj_lp,
            complete_log_posterior=next_complete,
        )
        return next_state, {
            "accepted": updated_mask,
            "latent_move_rms": latent_move_rms,
            "latent_move_max_abs": latent_move_max_abs,
            "latent_move_rms_per_t": latent_move_rms_per_t,
        }

    return {
        "name": "pit_particle_mgrad",
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "initial_scale_value": delta,
        "initial_scale_mode": "per_time_constant",
        "min_scale": min_scale,
        "max_scale": max_scale,
        "target_accept": target_accept,
        "step_fn": _latent_pit_particle_mgrad_step,
    }
