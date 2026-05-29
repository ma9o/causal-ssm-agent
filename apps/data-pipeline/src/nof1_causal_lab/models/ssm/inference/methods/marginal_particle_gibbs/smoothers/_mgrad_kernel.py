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
from nof1_causal_lab.models.ssm.inference.bundle import (
    AUX_JITTER,
    _initial_latent_moments,
    gaussian_log_prob_isotropic,
    tame_gradient_tulac,
)


class _DSMCState(NamedTuple):
    trajectories: jnp.ndarray
    log_weights: jnp.ndarray
    origins: jnp.ndarray
    keys: jnp.ndarray
    time_indices: jnp.ndarray
    resampling_entropy: jnp.ndarray
    resampling_ess: jnp.ndarray
    resampling_log_normalizer: jnp.ndarray


def _gaussian_log_prob_full(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
) -> jnp.ndarray:
    covariance = symmetrize_with_jitter(covariance, jitter=AUX_JITTER)
    chol = jnp.linalg.cholesky(covariance)
    diff = value - mean
    whitened = jla.solve_triangular(chol, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = diff.shape[-1]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)


def _gaussian_log_prob_shared_covariance(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
) -> jnp.ndarray:
    covariance = symmetrize_with_jitter(covariance, jitter=AUX_JITTER)
    chol = jnp.linalg.cholesky(covariance)
    diff = value - mean
    dim = diff.shape[-1]
    flat_diff = jnp.reshape(diff, (-1, dim))
    whitened = jla.solve_triangular(chol, flat_diff.T, lower=True).T
    quadratic = jnp.reshape(
        jnp.sum(whitened * whitened, axis=-1),
        diff.shape[:-1],
    )
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + quadratic)


def _normalize_log_weights_with_summary(
    log_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_normalizer = jax.scipy.special.logsumexp(log_weights)
    normalized = log_weights - log_normalizer
    probabilities = jnp.exp(normalized)
    entropy = -jnp.sum(probabilities * normalized)
    ess = 1.0 / jnp.sum(probabilities * probabilities)
    return normalized, entropy, ess, log_normalizer


def _pit_debug_extras(
    *,
    updated_mask: jnp.ndarray,
    origin_path: jnp.ndarray,
    proposal_grads: jnp.ndarray,
    auxiliary_var: jnp.ndarray,
    proposal_mean: jnp.ndarray,
    u: jnp.ndarray,
    x_ref: jnp.ndarray,
    resampling_entropy: jnp.ndarray,
    resampling_ess: jnp.ndarray,
    resampling_log_normalizer: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    proposal_shift = proposal_mean - u
    auxiliary_noise = u - x_ref
    return {
        "pit_updated_mask_per_t": updated_mask,
        "pit_selected_origin_per_t": origin_path.astype(jnp.int32),
        "pit_proposal_grad_norm_per_t": jnp.linalg.norm(proposal_grads, axis=-1),
        "pit_proposal_shift_norm_per_t": jnp.linalg.norm(proposal_shift, axis=-1),
        "pit_auxiliary_noise_rms_per_t": jnp.sqrt(
            jnp.mean(auxiliary_noise * auxiliary_noise, axis=-1)
        ),
        "pit_delta_per_t": 2.0 * auxiliary_var,
        "pit_resampling_entropy_per_t": resampling_entropy,
        "pit_resampling_ess_per_t": resampling_ess,
        "pit_resampling_log_normalizer_per_t": resampling_log_normalizer,
    }


def _particle_mgrad_debug_extras(
    *,
    updated_mask: jnp.ndarray,
    origin_path: jnp.ndarray,
    ref_grads: jnp.ndarray,
    auxiliary_var: jnp.ndarray,
    u: jnp.ndarray,
    x_ref: jnp.ndarray,
    resampling_entropy: jnp.ndarray,
    resampling_ess: jnp.ndarray,
    resampling_log_normalizer: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    auxiliary_shift = auxiliary_var[:, None] * ref_grads
    auxiliary_noise = u - x_ref - auxiliary_shift
    return {
        "particle_mgrad_updated_mask_per_t": updated_mask,
        "particle_mgrad_selected_origin_per_t": origin_path.astype(jnp.int32),
        "particle_mgrad_ref_grad_norm_per_t": jnp.linalg.norm(ref_grads, axis=-1),
        "particle_mgrad_auxiliary_shift_norm_per_t": jnp.linalg.norm(auxiliary_shift, axis=-1),
        "particle_mgrad_auxiliary_noise_rms_per_t": jnp.sqrt(
            jnp.mean(auxiliary_noise * auxiliary_noise, axis=-1)
        ),
        "particle_mgrad_delta_per_t": 2.0 * auxiliary_var,
        "particle_mgrad_resampling_entropy_per_t": resampling_entropy,
        "particle_mgrad_resampling_ess_per_t": resampling_ess,
        "particle_mgrad_resampling_log_normalizer_per_t": resampling_log_normalizer,
    }


def _mgrad_gain_and_covariance(
    prior_covariance: jnp.ndarray,
    auxiliary_var: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    covariance = symmetrize_with_jitter(prior_covariance, jitter=AUX_JITTER)
    dim = covariance.shape[-1]
    eye = jnp.eye(dim, dtype=covariance.dtype)
    gain = jla.solve(covariance + auxiliary_var * eye, covariance, assume_a="pos")
    proposal_covariance = symmetrize_with_jitter(auxiliary_var * gain, jitter=AUX_JITTER)
    return gain, proposal_covariance


def _mgrad_log_q_minus(
    particles: jnp.ndarray,
    prior_means: jnp.ndarray,
    candidate_shifts: jnp.ndarray,
    gain: jnp.ndarray,
    proposal_covariance: jnp.ndarray,
    auxiliary_var: jnp.ndarray,
) -> jnp.ndarray:
    """Relative log q_{-n}(x^{-n} | x^n) for marginal Particle-mGRAD weights."""
    dim = particles.shape[-1]
    num_particles = particles.shape[0]
    proposal_covariance = symmetrize_with_jitter(proposal_covariance, jitter=AUX_JITTER)
    proposal_chol = jnp.linalg.cholesky(proposal_covariance)
    identity = jnp.eye(dim, dtype=particles.dtype)
    v_terms = prior_means @ (identity - gain).T
    residuals = particles - v_terms
    proposal_precision_residuals = jla.cho_solve((proposal_chol, True), residuals.T).T
    proposal_precision_gain = jla.cho_solve((proposal_chol, True), gain)
    residual_precision_sum = jnp.sum(proposal_precision_residuals, axis=0)
    residual_quadratic = jnp.sum(residuals * proposal_precision_residuals, axis=-1)
    residual_quadratic_sum = jnp.sum(residual_quadratic)
    other_count = jnp.asarray(num_particles - 1, dtype=particles.dtype)
    marginal_precision = identity / auxiliary_var + other_count * (gain.T @ proposal_precision_gain)
    marginal_precision = symmetrize_with_jitter(marginal_precision, jitter=AUX_JITTER)
    marginal_chol = jnp.linalg.cholesky(marginal_precision)

    other_precision_sums = residual_precision_sum[None, :] - proposal_precision_residuals
    b_matrix = candidate_shifts / auxiliary_var + other_precision_sums @ gain
    c_values = (
        jnp.sum(candidate_shifts * candidate_shifts, axis=-1) / auxiliary_var
        + residual_quadratic_sum
        - residual_quadratic
    )
    solved_b_matrix = jla.cho_solve((marginal_chol, True), b_matrix.T).T
    return 0.5 * jnp.sum(b_matrix * solved_b_matrix, axis=-1) - 0.5 * c_values


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


def _forced_move_final_index(key: jnp.ndarray, log_weights: jnp.ndarray) -> jnp.ndarray:
    """Forced-move final selection for a conditional particle system."""
    candidate_key, accept_key = random.split(key)
    candidate_idx = random.categorical(candidate_key, log_weights[1:]).astype(jnp.int32) + 1
    weights = jnp.exp(log_weights)
    ref_weight = weights[0]
    candidate_weight = weights[candidate_idx]
    one = jnp.asarray(1.0, dtype=weights.dtype)
    accept_prob = jnp.minimum(one, (one - ref_weight) / (one - candidate_weight))
    return jnp.where(
        random.uniform(accept_key, (), dtype=weights.dtype) < accept_prob,
        candidate_idx,
        jnp.asarray(0, dtype=jnp.int32),
    )


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
    resampling_entropy_shape = tuple(initial_state.resampling_entropy.shape)
    resampling_ess_shape = tuple(initial_state.resampling_ess.shape)
    resampling_log_normalizer_shape = tuple(initial_state.resampling_log_normalizer.shape)
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
            resampling_entropy=_reshape_dsmc_leaf(
                state.resampling_entropy,
                segment_length=segment_length,
                original_shape=resampling_entropy_shape,
            ),
            resampling_ess=_reshape_dsmc_leaf(
                state.resampling_ess,
                segment_length=segment_length,
                original_shape=resampling_ess_shape,
            ),
            resampling_log_normalizer=_reshape_dsmc_leaf(
                state.resampling_log_normalizer,
                segment_length=segment_length,
                original_shape=resampling_log_normalizer_shape,
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
        resampling_entropy=state.resampling_entropy[0, :num_steps, ...].reshape(
            resampling_entropy_shape
        ),
        resampling_ess=state.resampling_ess[0, :num_steps, ...].reshape(resampling_ess_shape),
        resampling_log_normalizer=state.resampling_log_normalizer[0, :num_steps, ...].reshape(
            resampling_log_normalizer_shape
        ),
    )


def build_pit_particle_mgrad_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
    num_particles: int,
    min_scale: float | None = None,
    max_scale: float | None = None,
    debug_particle_trace: bool = False,
    latent_kernel_algorithm: str = "particle_mgrad",
) -> dict[str, Any]:
    """Build the particle latent update.

    ``latent_kernel_algorithm="particle_mgrad"`` uses the prior-informed
    marginal Particle-mGRAD kernel from Corenflos & Finke (2024). Its proposal
    is ancestor-dependent, so it is sequential in time.

    ``latent_kernel_algorithm="pit_aux_csmc"`` uses the separable
    parallel-in-time auxiliary particle Gibbs family from Corenflos & Särkkä
    (2025):

    1. draw pseudo-observations ``u_t ~ N(x_t, (δ_t/2) I)`` from the current
       reference trajectory,
    2. sample independent per-time particles from the locally adapted proposal
       ``q_t(x_t) = N(u_t + (δ_t/2) ∇ log G_t(u_t), (δ_t/2)I)``,
    3. stitch partial trajectories with a divide-and-conquer conditional dSMC
       operator whose boundary weight is
       ``p(x_t | x_{t-1}) G_t(x_t) N(u_t; x_t, δ_t/2 I) / q_t(x_t)``.

    The reference trajectory is preserved as particle 0 throughout the tree,
    yielding a particle-Gibbs latent transition with O(log T) parallel depth.
    """
    if num_particles < 2:
        raise ValueError("pit_particle_mgrad requires num_particles >= 2.")
    if latent_kernel_algorithm not in {"particle_mgrad", "pit_aux_csmc"}:
        raise ValueError(
            "Unknown latent_kernel_algorithm "
            f"{latent_kernel_algorithm!r}; expected 'particle_mgrad' or 'pit_aux_csmc'."
        )

    obs_increment_fn = bundle["observation_increment_log_prob_conditioned_from_context_fn"]
    obs_full_grad_fn = bundle["observation_grad_conditioned_from_context_fn"]
    rbpf_structure = str(bundle.get("rbpf_structure", "none"))
    if latent_kernel_algorithm == "particle_mgrad" and rbpf_structure == "conditional":
        raise ValueError(
            "particle_mgrad does not support conditional RBPF because the marginalized "
            "filter state is path-dependent; set latent_kernel_algorithm='pit_aux_csmc' "
            "for conditional RBPF."
        )
    rbpf_initial_filter_update_fn = bundle["rbpf_initial_filter_update_fn"]
    rbpf_step_filter_update_fn = bundle["rbpf_step_filter_update_fn"]
    runtime_observations = bundle["observations"]
    _raw_obs_increment_grad_fn = jax.grad(obs_increment_fn, argnums=1)

    def obs_increment_grad_fn(context, latent_t, observation_auxiliary, time_idx):
        """TULAc-tamed observation log-prob gradient (fixed h, see TULAC_H).

        Same fix as `tame_gradient_tulac` used in aux_kalman_mcmc: bounds the
        gradient-augmented pseudo-observation perturbation so it can shrink
        with adaptation. Applied identically to every grad call so the proposal
        density used in the particle weights matches the generated particles.
        """
        return tame_gradient_tulac(
            _raw_obs_increment_grad_fn(context, latent_t, observation_auxiliary, time_idx)
        )

    ref_particle_index = 0
    num_free_particles = num_particles - 1

    def _sequential_particle_mgrad_step(state, key: jnp.ndarray):
        aux_key, proposal_key, resample_key, final_key, backward_key = random.split(key, 5)
        x_ref = state.latent_trajectory
        context = state.latent_context
        observation_auxiliary = state.observation_auxiliary
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
                obs_increment_grad_fn(context, latent_t, observation_auxiliary, time_idx),
                dtype=latent_dtype,
            )
        )(x_ref, time_indices)
        u = x_ref + auxiliary_var[:, None] * ref_grads
        u = u + auxiliary_std * random.normal(aux_key, x_ref.shape, dtype=latent_dtype)

        proposal_keys = random.split(proposal_key, num_steps)
        init_mean, init_cov = _initial_latent_moments(context)
        init_gain, init_proposal_cov = _mgrad_gain_and_covariance(init_cov, auxiliary_var[0])
        init_proposal_chol = jnp.linalg.cholesky(init_proposal_cov)
        init_free_mean = init_mean + (u[0] - init_mean) @ init_gain.T
        init_free_eps = random.normal(
            proposal_keys[0],
            (num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        init_free_particles = init_free_mean[None, :] + init_free_eps @ init_proposal_chol.T
        x_particles0 = jnp.concatenate([x_ref[0][None, :], init_free_particles], axis=0)
        init_prior_means = jnp.broadcast_to(init_mean, (num_particles, latent_dim))
        init_prior_lp = _gaussian_log_prob_shared_covariance(
            x_particles0,
            init_mean,
            init_cov,
        )
        init_obs_lp = jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                observation_auxiliary,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(x_particles0)
        init_candidate_grads = jax.vmap(
            lambda particle: obs_increment_grad_fn(
                context,
                particle,
                observation_auxiliary,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(x_particles0)
        init_candidate_shifts = x_particles0 + auxiliary_var[0] * init_candidate_grads
        init_log_q_minus = _mgrad_log_q_minus(
            x_particles0,
            init_prior_means,
            init_candidate_shifts,
            init_gain,
            init_proposal_cov,
            auxiliary_var[0],
        )
        raw_log_weights0 = init_prior_lp + init_obs_lp + init_log_q_minus
        if debug_particle_trace:
            log_weights0, entropy0, ess0, log_normalizer0 = _normalize_log_weights_with_summary(
                raw_log_weights0
            )
        else:
            log_weights0 = raw_log_weights0 - jax.scipy.special.logsumexp(raw_log_weights0)

        resample_keys = random.split(resample_key, max(num_steps - 1, 1))

        def _step(carry, inputs):
            prev_log_weights, prev_particles_bank = carry
            u_t, auxiliary_var_t, time_idx, proposal_key_t, resample_key_t = inputs
            free_ancestors = random.categorical(
                resample_key_t,
                prev_log_weights,
                shape=(num_free_particles,),
            ).astype(jnp.int32)
            ancestors = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), free_ancestors],
                axis=0,
            )
            prev_particles = jnp.take(prev_particles_bank, ancestors, axis=0)
            prior_means = prev_particles @ context.Ad[time_idx].T + context.cd[time_idx]
            gain_t, proposal_cov_t = _mgrad_gain_and_covariance(
                context.Qd[time_idx],
                auxiliary_var_t,
            )
            proposal_chol_t = jnp.linalg.cholesky(proposal_cov_t)
            free_means = prior_means[1:] + (u_t[None, :] - prior_means[1:]) @ gain_t.T
            free_eps = random.normal(
                proposal_key_t,
                (num_free_particles, latent_dim),
                dtype=latent_dtype,
            )
            free_particles = free_means + free_eps @ proposal_chol_t.T
            particles_t = jnp.concatenate([x_ref[time_idx][None, :], free_particles], axis=0)
            transition_lp = _gaussian_log_prob_shared_covariance(
                particles_t,
                prior_means,
                context.Qd[time_idx],
            )
            obs_lp = jax.vmap(
                lambda particle: obs_increment_fn(
                    context,
                    particle,
                    observation_auxiliary,
                    time_idx,
                )
            )(particles_t)
            candidate_grads = jax.vmap(
                lambda particle: obs_increment_grad_fn(
                    context,
                    particle,
                    observation_auxiliary,
                    time_idx,
                )
            )(particles_t)
            candidate_shifts = particles_t + auxiliary_var_t * candidate_grads
            log_q_minus = _mgrad_log_q_minus(
                particles_t,
                prior_means,
                candidate_shifts,
                gain_t,
                proposal_cov_t,
                auxiliary_var_t,
            )
            raw_next_log_weights = transition_lp + obs_lp + log_q_minus
            if debug_particle_trace:
                next_log_weights, entropy_t, ess_t, log_normalizer_t = (
                    _normalize_log_weights_with_summary(raw_next_log_weights)
                )
                step_debug = (entropy_t, ess_t, log_normalizer_t)
            else:
                next_log_weights = raw_next_log_weights - jax.scipy.special.logsumexp(
                    raw_next_log_weights
                )
                step_debug = None
            return (
                next_log_weights.astype(traj_dtype),
                particles_t,
            ), (particles_t, next_log_weights.astype(traj_dtype), step_debug)

        if num_steps > 1:
            (log_weights, _last_particles), scan_output = jax.lax.scan(
                _step,
                (log_weights0.astype(traj_dtype), x_particles0),
                (
                    u[1:],
                    auxiliary_var[1:],
                    jnp.arange(1, num_steps, dtype=jnp.int32),
                    proposal_keys[1:],
                    resample_keys[: num_steps - 1],
                ),
            )
            particle_tail, log_weight_tail, scan_debug = scan_output
            particle_history = jnp.concatenate([x_particles0[None, :, :], particle_tail], axis=0)
            log_weight_history = jnp.concatenate(
                [log_weights0.astype(traj_dtype)[None, :], log_weight_tail],
                axis=0,
            )
        else:
            log_weights = log_weights0.astype(traj_dtype)
            particle_history = x_particles0[None, :, :]
            log_weight_history = log_weights[None, :]
            if debug_particle_trace:
                scan_debug = (
                    jnp.zeros((0,), dtype=traj_dtype),
                    jnp.zeros((0,), dtype=traj_dtype),
                    jnp.zeros((0,), dtype=traj_dtype),
                )

        final_idx = _forced_move_final_index(final_key, log_weights)
        if num_steps > 1:
            backward_keys = random.split(backward_key, num_steps - 1)

            def _backward_step(next_idx, inputs):
                time_idx, backward_key_t = inputs
                next_particle = particle_history[time_idx + 1, next_idx, :]
                candidate_particles = particle_history[time_idx]
                pred_means = (
                    candidate_particles @ context.Ad[time_idx + 1].T + context.cd[time_idx + 1]
                )
                transition_lp = _gaussian_log_prob_shared_covariance(
                    next_particle,
                    pred_means,
                    context.Qd[time_idx + 1],
                )
                backward_logits = log_weight_history[time_idx] + transition_lp
                selected_idx = random.categorical(backward_key_t, backward_logits).astype(jnp.int32)
                return selected_idx, selected_idx

            _first_idx, reverse_indices = jax.lax.scan(
                _backward_step,
                final_idx,
                (
                    jnp.arange(num_steps - 2, -1, -1, dtype=jnp.int32),
                    backward_keys,
                ),
            )
            origin_path = jnp.concatenate(
                [jnp.flip(reverse_indices, axis=0), jnp.reshape(final_idx, (1,))],
                axis=0,
            )
        else:
            origin_path = jnp.reshape(final_idx, (1,))
        latent_path = jnp.take_along_axis(
            particle_history,
            origin_path[:, None, None],
            axis=1,
        )[:, 0, :]
        prior_terms = bundle["prior_terms_from_context_fn"](context)
        next_traj_lp = jnp.asarray(
            bundle["trajectory_log_prob_conditioned_from_context_fn"](
                context,
                latent_path,
                observation_auxiliary,
                prior_terms,
            ),
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
        extras = {
            "accepted": updated_mask,
            "latent_move_rms": latent_move_rms,
            "latent_move_max_abs": latent_move_max_abs,
            "latent_move_rms_per_t": latent_move_rms_per_t,
        }
        if debug_particle_trace:
            tail_entropy, tail_ess, tail_log_normalizer = scan_debug
            extras.update(
                _particle_mgrad_debug_extras(
                    updated_mask=updated_mask,
                    origin_path=origin_path,
                    ref_grads=ref_grads,
                    auxiliary_var=auxiliary_var,
                    u=u,
                    x_ref=x_ref,
                    resampling_entropy=jnp.concatenate(
                        [jnp.reshape(entropy0, (1,)), tail_entropy]
                    ).astype(traj_dtype),
                    resampling_ess=jnp.concatenate([jnp.reshape(ess0, (1,)), tail_ess]).astype(
                        traj_dtype
                    ),
                    resampling_log_normalizer=jnp.concatenate(
                        [jnp.reshape(log_normalizer0, (1,)), tail_log_normalizer]
                    ).astype(traj_dtype),
                )
            )
        return next_state, extras

    def _conditional_rbpf_particle_step(state, key: jnp.ndarray):
        aux_key, proposal_key, resample_key, select_key = random.split(key, 4)
        x_ref = state.latent_trajectory
        context = state.latent_context
        observation_auxiliary = state.observation_auxiliary
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
        u = x_ref + auxiliary_std * random.normal(aux_key, x_ref.shape, dtype=latent_dtype)
        proposal_grads = jnp.asarray(
            tame_gradient_tulac(obs_full_grad_fn(context, u, observation_auxiliary)),
            dtype=latent_dtype,
        )
        proposal_mean = u + auxiliary_var[:, None] * proposal_grads

        proposal_eps = random.normal(
            proposal_key,
            (num_steps, num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        free_particles = proposal_mean[:, None, :] + auxiliary_std[:, None, :] * proposal_eps
        x_particles = jnp.concatenate([x_ref[:, None, :], free_particles], axis=1)

        proposal_log_probs = jax.vmap(
            lambda particles_t, mean_t, var_t: jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(particle, mean_t, var_t)
            )(particles_t)
        )(x_particles, proposal_mean, auxiliary_var)
        auxiliary_log_probs = jax.vmap(
            lambda particles_t, u_t, var_t: jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(particle, u_t, var_t)
            )(particles_t)
        )(x_particles, u, auxiliary_var)

        init_mean, init_cov = _initial_latent_moments(context)
        init_prior_lp = _gaussian_log_prob_shared_covariance(
            x_particles[0],
            init_mean,
            init_cov,
        )
        init_obs_lp = jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                observation_auxiliary,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(x_particles[0])
        init_filter_state, init_rbpf_lp = jax.vmap(
            lambda particle: rbpf_initial_filter_update_fn(
                context.rbpf_marginal_context,
                runtime_observations,
                observation_auxiliary,
                particle,
                jitter=AUX_JITTER,
            )
        )(x_particles[0])
        init_log_weights = (
            init_prior_lp
            + init_obs_lp
            + init_rbpf_lp
            + auxiliary_log_probs[0]
            - proposal_log_probs[0]
        )
        if debug_particle_trace:
            log_weights, entropy0, ess0, log_normalizer0 = _normalize_log_weights_with_summary(
                init_log_weights
            )
        else:
            log_weights = init_log_weights - jax.scipy.special.logsumexp(init_log_weights)
        paths = jnp.zeros((num_particles, num_steps, latent_dim), dtype=latent_dtype)
        paths = paths.at[:, 0, :].set(x_particles[0])
        origins = jnp.zeros((num_particles, num_steps), dtype=jnp.int32)
        origins = origins.at[:, 0].set(jnp.arange(num_particles, dtype=jnp.int32))

        resample_keys = random.split(resample_key, max(num_steps - 1, 1))

        def _step(carry, inputs):
            prev_log_weights, prev_filter_state, prev_paths, prev_origins = carry
            particles_t, proposal_lp_t, auxiliary_lp_t, time_idx, resample_key_t = inputs
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
            ancestor_origins = jnp.take(prev_origins, ancestors, axis=0)
            prev_particles = ancestor_paths[:, time_idx - 1, :]
            ancestor_filter_state = jax.tree_util.tree_map(
                lambda leaf: jnp.take(leaf, ancestors, axis=0),
                prev_filter_state,
            )

            pred_means = prev_particles @ context.Ad[time_idx].T + context.cd[time_idx]
            transition_lp = _gaussian_log_prob_shared_covariance(
                particles_t,
                pred_means,
                context.Qd[time_idx],
            )
            obs_lp = jax.vmap(
                lambda particle: obs_increment_fn(
                    context,
                    particle,
                    observation_auxiliary,
                    time_idx,
                )
            )(particles_t)
            next_filter_state, rbpf_lp = jax.vmap(
                lambda filter_state, carried_prev, carried_t: rbpf_step_filter_update_fn(
                    context.rbpf_marginal_context,
                    filter_state,
                    runtime_observations,
                    observation_auxiliary,
                    carried_prev,
                    carried_t,
                    time_idx,
                    jitter=AUX_JITTER,
                )
            )(ancestor_filter_state, prev_particles, particles_t)
            raw_next_log_weights = transition_lp + obs_lp + rbpf_lp + auxiliary_lp_t - proposal_lp_t
            if debug_particle_trace:
                next_log_weights, entropy_t, ess_t, log_normalizer_t = (
                    _normalize_log_weights_with_summary(raw_next_log_weights)
                )
                step_debug = (entropy_t, ess_t, log_normalizer_t)
            else:
                next_log_weights = raw_next_log_weights - jax.scipy.special.logsumexp(
                    raw_next_log_weights
                )
                step_debug = None
            next_paths = ancestor_paths.at[:, time_idx, :].set(particles_t)
            next_origins = ancestor_origins.at[:, time_idx].set(
                jnp.arange(num_particles, dtype=jnp.int32)
            )
            return (
                next_log_weights.astype(traj_dtype),
                next_filter_state,
                next_paths,
                next_origins,
            ), step_debug

        if num_steps > 1:
            (log_weights, _filter_state, paths, origins), scan_debug = jax.lax.scan(
                _step,
                (log_weights.astype(traj_dtype), init_filter_state, paths, origins),
                (
                    x_particles[1:],
                    proposal_log_probs[1:],
                    auxiliary_log_probs[1:],
                    jnp.arange(1, num_steps, dtype=jnp.int32),
                    resample_keys[: num_steps - 1],
                ),
            )
        elif debug_particle_trace:
            scan_debug = (
                jnp.zeros((0,), dtype=traj_dtype),
                jnp.zeros((0,), dtype=traj_dtype),
                jnp.zeros((0,), dtype=traj_dtype),
            )

        final_idx = random.categorical(select_key, log_weights).astype(jnp.int32)
        latent_path = paths[final_idx]
        origin_path = origins[final_idx]
        prior_terms = bundle["prior_terms_from_context_fn"](context)
        next_traj_lp = jnp.asarray(
            bundle["trajectory_log_prob_conditioned_from_context_fn"](
                context,
                latent_path,
                observation_auxiliary,
                prior_terms,
            ),
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
        extras = {
            "accepted": updated_mask,
            "latent_move_rms": latent_move_rms,
            "latent_move_max_abs": latent_move_max_abs,
            "latent_move_rms_per_t": latent_move_rms_per_t,
        }
        if debug_particle_trace:
            tail_entropy, tail_ess, tail_log_normalizer = scan_debug
            extras.update(
                _pit_debug_extras(
                    updated_mask=updated_mask,
                    origin_path=origin_path,
                    proposal_grads=proposal_grads,
                    auxiliary_var=auxiliary_var,
                    proposal_mean=proposal_mean,
                    u=u,
                    x_ref=x_ref,
                    resampling_entropy=jnp.concatenate(
                        [jnp.reshape(entropy0, (1,)), tail_entropy]
                    ).astype(traj_dtype),
                    resampling_ess=jnp.concatenate([jnp.reshape(ess0, (1,)), tail_ess]).astype(
                        traj_dtype
                    ),
                    resampling_log_normalizer=jnp.concatenate(
                        [jnp.reshape(log_normalizer0, (1,)), tail_log_normalizer]
                    ).astype(traj_dtype),
                )
            )
        return next_state, extras

    def _latent_pit_particle_mgrad_step(state, key: jnp.ndarray):
        if latent_kernel_algorithm == "particle_mgrad":
            return _sequential_particle_mgrad_step(state, key)
        if rbpf_structure == "conditional":
            return _conditional_rbpf_particle_step(state, key)

        aux_key, proposal_key, combine_key, select_key = random.split(key, 4)
        x_ref = state.latent_trajectory
        context = state.latent_context
        observation_auxiliary = state.observation_auxiliary
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
        u = x_ref + auxiliary_std * random.normal(aux_key, x_ref.shape, dtype=latent_dtype)
        proposal_grads = jax.vmap(
            lambda latent_t, time_idx: jnp.asarray(
                obs_increment_grad_fn(context, latent_t, observation_auxiliary, time_idx),
                dtype=latent_dtype,
            )
        )(u, time_indices)
        proposal_mean = u + auxiliary_var[:, None] * proposal_grads

        proposal_eps = random.normal(
            proposal_key,
            (num_steps, num_free_particles, latent_dim),
            dtype=latent_dtype,
        )
        free_particles = proposal_mean[:, None, :] + auxiliary_std[:, None, :] * proposal_eps
        x_particles = jnp.concatenate([x_ref[:, None, :], free_particles], axis=1)

        proposal_log_probs = jax.vmap(
            lambda particles_t, mean_t, var_t: jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(particle, mean_t, var_t)
            )(particles_t)
        )(x_particles, proposal_mean, auxiliary_var)
        auxiliary_log_probs = jax.vmap(
            lambda particles_t, u_t, var_t: jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(particle, u_t, var_t)
            )(particles_t)
        )(x_particles, u, auxiliary_var)

        init_mean, init_cov = _initial_latent_moments(context)
        init_prior_lp = _gaussian_log_prob_shared_covariance(
            x_particles[0],
            init_mean,
            init_cov,
        )
        init_obs_lp = jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                observation_auxiliary,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(x_particles[0])
        raw_log_weights0 = (
            init_prior_lp + init_obs_lp + auxiliary_log_probs[0] - proposal_log_probs[0]
        )
        if debug_particle_trace:
            log_weights0, entropy0, ess0, log_normalizer0 = _normalize_log_weights_with_summary(
                raw_log_weights0
            )
        else:
            log_weights0 = raw_log_weights0 - jax.scipy.special.logsumexp(raw_log_weights0)
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
        resampling_entropy = jnp.full((num_steps,), jnp.nan, dtype=traj_dtype)
        resampling_ess = jnp.full((num_steps,), jnp.nan, dtype=traj_dtype)
        resampling_log_normalizer = jnp.full((num_steps,), jnp.nan, dtype=traj_dtype)
        if debug_particle_trace:
            resampling_entropy = resampling_entropy.at[0].set(entropy0.astype(traj_dtype))
            resampling_ess = resampling_ess.at[0].set(ess0.astype(traj_dtype))
            resampling_log_normalizer = resampling_log_normalizer.at[0].set(
                log_normalizer0.astype(traj_dtype)
            )
        initial_state = _DSMCState(
            trajectories=x_particles,
            log_weights=log_weights.astype(traj_dtype),
            origins=origins,
            keys=combine_keys,
            time_indices=time_indices,
            resampling_entropy=resampling_entropy,
            resampling_ess=resampling_ess,
            resampling_log_normalizer=resampling_log_normalizer,
        )

        def _combine_segments(left: _DSMCState, right: _DSMCState) -> _DSMCState:
            stitch_key = right.keys[0]
            time_idx = right.time_indices[0]
            left_particles = left.trajectories[-1]
            right_particles = right.trajectories[0]
            left_log_weights = left.log_weights[-1]
            right_log_weights = right.log_weights[0]
            pred_means = left_particles @ context.Ad[time_idx].T + context.cd[time_idx]
            transition_lp = _gaussian_log_prob_shared_covariance(
                right_particles[None, :, :],
                pred_means[:, None, :],
                context.Qd[time_idx],
            )
            obs_lp = jax.vmap(
                lambda particle: obs_increment_fn(
                    context,
                    particle,
                    observation_auxiliary,
                    time_idx,
                )
            )(right_particles)
            proposal_lp = jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(
                    particle,
                    proposal_mean[time_idx],
                    auxiliary_var[time_idx],
                )
            )(right_particles)
            auxiliary_lp = jax.vmap(
                lambda particle: gaussian_log_prob_isotropic(
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
                + auxiliary_lp[None, :]
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
            right_resampling_entropy = right.resampling_entropy
            right_resampling_ess = right.resampling_ess
            right_resampling_log_normalizer = right.resampling_log_normalizer
            if debug_particle_trace:
                _, entropy_t, ess_t, log_normalizer_t = _normalize_log_weights_with_summary(
                    log_pair_weights
                )
                right_resampling_entropy = right_resampling_entropy.at[0].set(
                    entropy_t.astype(traj_dtype)
                )
                right_resampling_ess = right_resampling_ess.at[0].set(ess_t.astype(traj_dtype))
                right_resampling_log_normalizer = right_resampling_log_normalizer.at[0].set(
                    log_normalizer_t.astype(traj_dtype)
                )
            return _DSMCState(
                trajectories=jnp.concatenate([left_trajectories, right_trajectories], axis=0),
                log_weights=merged_log_weights,
                origins=jnp.concatenate([left_origins, right_origins], axis=0),
                keys=jnp.concatenate([left.keys, right.keys], axis=0),
                time_indices=jnp.concatenate([left.time_indices, right.time_indices], axis=0),
                resampling_entropy=jnp.concatenate(
                    [left.resampling_entropy, right_resampling_entropy], axis=0
                ),
                resampling_ess=jnp.concatenate([left.resampling_ess, right_resampling_ess], axis=0),
                resampling_log_normalizer=jnp.concatenate(
                    [left.resampling_log_normalizer, right_resampling_log_normalizer], axis=0
                ),
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
            bundle["trajectory_log_prob_conditioned_from_context_fn"](
                context,
                latent_path,
                observation_auxiliary,
                prior_terms,
            ),
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
        extras = {
            "accepted": updated_mask,
            "latent_move_rms": latent_move_rms,
            "latent_move_max_abs": latent_move_max_abs,
            "latent_move_rms_per_t": latent_move_rms_per_t,
        }
        if debug_particle_trace:
            extras.update(
                _pit_debug_extras(
                    updated_mask=updated_mask,
                    origin_path=origin_path,
                    proposal_grads=proposal_grads,
                    auxiliary_var=auxiliary_var,
                    proposal_mean=proposal_mean,
                    u=u,
                    x_ref=x_ref,
                    resampling_entropy=final_state.resampling_entropy.astype(traj_dtype),
                    resampling_ess=final_state.resampling_ess.astype(traj_dtype),
                    resampling_log_normalizer=final_state.resampling_log_normalizer.astype(
                        traj_dtype
                    ),
                )
            )
        return next_state, extras

    return {
        "name": "pit_particle_mgrad",
        "algorithm": (
            "conditional_aux_rbpf_csmc"
            if rbpf_structure == "conditional"
            else (
                "sequential_particle_mgrad"
                if latent_kernel_algorithm == "particle_mgrad"
                else "pit_aux_csmc"
            )
        ),
        "family": (
            "particle_mgrad" if latent_kernel_algorithm == "particle_mgrad" else "auxiliary_csmc"
        ),
        "selection": (
            "backward_sampling_forced_move"
            if latent_kernel_algorithm == "particle_mgrad"
            else "divide_and_conquer_particle_selection"
        ),
        "parallel": latent_kernel_algorithm == "pit_aux_csmc" and rbpf_structure != "conditional",
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "initial_scale_value": delta,
        "initial_scale_mode": "per_time_constant",
        "min_scale": min_scale,
        "max_scale": max_scale,
        "target_accept": target_accept,
        "step_fn": _latent_pit_particle_mgrad_step,
    }
