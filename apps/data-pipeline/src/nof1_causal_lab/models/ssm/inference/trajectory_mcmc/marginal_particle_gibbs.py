"""Marginalized Particle Gibbs joint parameter/trajectory kernel.

Implements the M=2-by-default collapsed Particle Gibbs construction from
Corenflos (2025), "Particle Gibbs without the Gibbs bit", for directly
evaluable SSM potentials. The parameter proposal is formed in unconstrained
space via the symmetric auxiliary decomposition

    u | theta ~ N(theta, 2 delta Sigma)
    theta' | u ~ N(u, 2 delta Sigma)

and the latent trajectory is updated by conditional SMC against the posterior
mixture over the current/proposed parameter ensemble.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.auxiliary_kalman import (
    AUX_JITTER,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.gibbs import (
    AuxKalmanMCMCResult,
    AuxKalmanMCMCState,
    _adapt_scale,
    _clip_scale,
    _latent_summary_from_chain_moments,
    _stack_chain_states,
    _stack_sample_history,
)

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3


@dataclass(frozen=True)
class MarginalParticleGibbsKernel:
    """Callable joint kernel and static metadata."""

    step_fn: Any
    num_particles: int
    num_parameter_particles: int
    initial_param_step_size: float
    target_accept: float
    min_scale: float
    max_scale: float
    preconditioned: bool
    latent_block_size: int


def _normalize_log_probs(logits: jnp.ndarray, *, axis: int = -1) -> jnp.ndarray:
    return logits - jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)


def _categorical_rows(key: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    keys = random.split(key, int(logits.shape[0]))
    return jax.vmap(lambda row_key, row_logits: random.categorical(row_key, row_logits))(
        keys,
        logits,
    ).astype(jnp.int32)


def _sample_gaussian_from_chol(
    key: jnp.ndarray,
    mean: jnp.ndarray,
    chol: jnp.ndarray,
) -> jnp.ndarray:
    eps = random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + jnp.einsum("...ij,...j->...i", chol, eps)


def _cholesky_batch(covariances: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(
        lambda cov: jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=AUX_JITTER))
    )(covariances)


def _logdet_from_cholesky(cholesky: jnp.ndarray) -> jnp.ndarray:
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    return 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)


def _gaussian_log_prob_shared_cholesky(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    cholesky: jnp.ndarray,
    logdet: jnp.ndarray,
) -> jnp.ndarray:
    diff = value - mean
    dim = diff.shape[-1]
    flat_diff = jnp.reshape(diff, (-1, dim))
    whitened = jla.solve_triangular(cholesky, flat_diff.T, lower=True).T
    quadratic = jnp.reshape(
        jnp.sum(whitened * whitened, axis=-1),
        diff.shape[:-1],
    )
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + quadratic)


def _transition_log_probs_by_param(
    contexts,
    transition_cholesky: jnp.ndarray,
    transition_logdet: jnp.ndarray,
    prev_particles: jnp.ndarray,
    particles_t: jnp.ndarray,
    time_idx: jnp.ndarray,
) -> jnp.ndarray:
    def _one_param(context, cholesky_by_time, logdet_by_time):
        means = prev_particles @ context.Ad[time_idx].T + context.cd[time_idx]
        return _gaussian_log_prob_shared_cholesky(
            particles_t,
            means,
            cholesky_by_time[time_idx],
            logdet_by_time[time_idx],
        )

    return jnp.swapaxes(
        jax.vmap(_one_param)(contexts, transition_cholesky, transition_logdet),
        0,
        1,
    )


def _observation_log_probs_by_param(
    contexts,
    particles_t: jnp.ndarray,
    observation_auxiliary,
    time_idx: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    obs_increment_fn,
) -> jnp.ndarray:
    def _one_param(context):
        return jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                observation_auxiliary,
                time_idx,
                runtime_observations,
            )
        )(particles_t)

    return jnp.swapaxes(jax.vmap(_one_param)(contexts), 0, 1)


def _transition_log_probs_to_next_by_param(
    contexts,
    transition_cholesky: jnp.ndarray,
    transition_logdet: jnp.ndarray,
    prev_particles: jnp.ndarray,
    next_particle: jnp.ndarray,
    time_idx: jnp.ndarray,
) -> jnp.ndarray:
    def _one_param(context, cholesky_by_time, logdet_by_time):
        means = prev_particles @ context.Ad[time_idx].T + context.cd[time_idx]
        next_particles = jnp.broadcast_to(next_particle, means.shape)
        return _gaussian_log_prob_shared_cholesky(
            next_particles,
            means,
            cholesky_by_time[time_idx],
            logdet_by_time[time_idx],
        )

    return jnp.swapaxes(
        jax.vmap(_one_param)(contexts, transition_cholesky, transition_logdet),
        0,
        1,
    )


def _single_observation_log_probs_by_param(
    contexts,
    particle_t: jnp.ndarray,
    observation_auxiliary,
    time_idx: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    obs_increment_fn,
) -> jnp.ndarray:
    return _observation_log_probs_by_param(
        contexts,
        particle_t[None, :],
        observation_auxiliary,
        time_idx,
        runtime_observations,
        obs_increment_fn,
    )[0]


def _select_pytree(ensemble, index: jnp.ndarray):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], ensemble)


def build_marginal_particle_gibbs_kernel(
    bundle: dict[str, Any],
    *,
    num_particles: int,
    num_parameter_particles: int,
    param_step_size: float,
    target_accept: float,
    min_scale: float = _DEFAULT_MIN_SCALE,
    max_scale: float = _DEFAULT_MAX_SCALE,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    latent_block_size: int = 256,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    if num_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_particles >= 2.")
    if num_parameter_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_parameter_particles >= 2.")
    if min_scale > max_scale:
        raise ValueError(
            "marginal_particle_gibbs min_scale must be <= max_scale; "
            f"got {min_scale} > {max_scale}."
        )
    if latent_block_size < 1:
        raise ValueError(
            f"marginal_particle_gibbs latent_block_size must be positive; got {latent_block_size}."
        )

    latent_context_runtime_fn = bundle["latent_context_runtime_fn"]
    log_prior_unc_fn = bundle["log_prior_unc_fn"]
    initial_latent_moments_fn = bundle["initial_latent_moments_from_context_fn"]
    obs_increment_fn = bundle["observation_increment_log_prob_conditioned_from_context_runtime_fn"]
    trajectory_log_prob_fn = bundle["trajectory_log_prob_conditioned_from_context_runtime_fn"]
    prior_terms_from_context_fn = bundle["prior_terms_from_context_fn"]
    initial_observation_auxiliary_fn = bundle[
        "initial_observation_auxiliary_from_context_runtime_fn"
    ]
    runtime_observations = bundle["observations"]
    runtime_times = bundle["times"]

    preconditioner = (
        None
        if parameter_preconditioner_chol is None
        else jnp.asarray(
            parameter_preconditioner_chol,
            dtype=bundle["flat_example"].dtype,
        )
    )

    def _propose_parameter_ensemble(
        current_position: jnp.ndarray,
        key: jnp.ndarray,
        step_size: jnp.ndarray,
    ) -> jnp.ndarray:
        dim = int(current_position.shape[0])
        proposal_scale = jnp.sqrt(
            jnp.asarray(2.0, dtype=current_position.dtype)
            * jnp.asarray(step_size, dtype=current_position.dtype)
        )
        if preconditioner is None:
            chol = jnp.eye(dim, dtype=current_position.dtype)
        else:
            chol = jnp.asarray(preconditioner, dtype=current_position.dtype)
        aux_key, proposal_key = random.split(key)
        u = current_position + proposal_scale * (
            random.normal(
                aux_key,
                current_position.shape,
                dtype=current_position.dtype,
            )
            @ chol.T
        )
        proposal_eps = random.normal(
            proposal_key,
            (num_parameter_particles - 1, dim),
            dtype=current_position.dtype,
        )
        proposed = u[None, :] + proposal_scale * (proposal_eps @ chol.T)
        return jnp.concatenate([current_position[None, :], proposed], axis=0)

    def _step_fn(state: AuxKalmanMCMCState, key: jnp.ndarray):
        param_key, block_key, label_key = random.split(key, 3)
        x_ref = state.latent_trajectory
        latent_dtype = x_ref.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype
        num_steps = int(x_ref.shape[0])
        block_size = min(int(latent_block_size), num_steps)
        num_blocks = (num_steps + block_size - 1) // block_size
        num_free_particles = num_particles - 1

        parameter_particles = _propose_parameter_ensemble(
            state.position,
            param_key,
            _clip_scale(
                state.param_step_size,
                min_scale=min_scale,
                max_scale=max_scale,
            ),
        )
        contexts = jax.vmap(lambda z: latent_context_runtime_fn(z, runtime_times))(
            parameter_particles
        )
        parameter_log_probs = jax.vmap(log_prior_unc_fn)(parameter_particles).astype(traj_dtype)
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
                selected_particle = random.categorical(
                    backward_key_t,
                    _normalize_log_probs(backward_logits),
                ).astype(jnp.int32)
                latent_t = particles_t[selected_particle]
                selected_transition_lp = _transition_log_probs_from_fixed_prev(
                    latent_t,
                    next_particle[None, :],
                    local_time_idx + 1,
                )[0]
                selected_future_tail = (selected_transition_lp + obs_lp + next_future_tail).astype(
                    traj_dtype
                )
                return (latent_t, selected_future_tail), (latent_t, selected_particle)

            if block_len > 1:
                _, (reversed_latents, reversed_indices) = jax.lax.scan(
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
            else:
                block_path = final_latent[None, :]
                block_indices = final_particle[None]

            return block_path.astype(latent_dtype), block_indices

        block_keys = random.split(block_key, num_blocks)
        latent_path = x_ref
        origin_path = jnp.zeros((num_steps,), dtype=jnp.int32)
        future_tail_history = _path_future_tail_log_probs(latent_path)
        prefix_label_log_probs = initial_label_log_probs
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, num_steps) - 1
            previous_particle = latent_path[block_start - 1] if block_start > 0 else latent_path[0]
            block_path, block_indices = _backward_sample_block(
                block_keys[block_idx],
                latent_path,
                block_start,
                block_end,
                prefix_label_log_probs,
                future_tail_history,
            )
            latent_path = latent_path.at[block_start : block_end + 1].set(block_path)
            origin_path = origin_path.at[block_start : block_end + 1].set(block_indices)
            prefix_label_log_probs = _segment_terminal_label_log_probs(
                prefix_label_log_probs,
                block_path,
                previous_particle,
                block_start,
            )

        final_label_log_probs = prefix_label_log_probs
        selected_label = random.categorical(label_key, final_label_log_probs).astype(jnp.int32)
        next_position = parameter_particles[selected_label]
        next_context = _select_pytree(contexts, selected_label)
        next_observation_auxiliary = initial_observation_auxiliary_fn(
            next_context,
            latent_path,
            runtime_observations,
        )
        prior_terms = prior_terms_from_context_fn(next_context)
        next_traj_lp = jnp.asarray(
            trajectory_log_prob_fn(
                next_context,
                latent_path,
                next_observation_auxiliary,
                runtime_observations,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )
        next_complete = jnp.asarray(log_prior_unc_fn(next_position), dtype=complete_dtype)
        next_complete = next_complete + next_traj_lp.astype(complete_dtype)
        latent_move = latent_path - x_ref
        latent_move_rms_per_t = jnp.sqrt(jnp.mean(latent_move * latent_move, axis=-1))
        latent_move_rms = jnp.sqrt(jnp.mean(latent_move * latent_move))
        latent_move_max_abs = jnp.max(jnp.abs(latent_move))
        parameter_accepted = (selected_label != 0).astype(state.position.dtype)
        latent_updated = (origin_path != 0).astype(state.position.dtype)

        return (
            state._replace(
                position=next_position,
                latent_context=next_context,
                latent_trajectory=latent_path,
                observation_auxiliary=next_observation_auxiliary,
                trajectory_log_prob=next_traj_lp,
                complete_log_posterior=next_complete,
            ),
            {
                "parameter_accepted": parameter_accepted,
                "latent_accepted": latent_updated,
                "selected_label": selected_label.astype(jnp.float32),
                "final_particle": origin_path[-1].astype(jnp.float32),
                "latent_move_rms": latent_move_rms,
                "latent_move_max_abs": latent_move_max_abs,
                "latent_move_rms_per_t": latent_move_rms_per_t,
            },
        )

    return MarginalParticleGibbsKernel(
        step_fn=_step_fn,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        initial_param_step_size=param_step_size,
        target_accept=target_accept,
        min_scale=min_scale,
        max_scale=max_scale,
        preconditioned=parameter_preconditioner_chol is not None,
        latent_block_size=latent_block_size,
    )


def _initialize_chain_state(
    init_position: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    bundle: dict[str, Any],
    latent_delta: float,
    param_step_size: float,
    param_min_scale: float,
    param_max_scale: float,
    param_target_accept: float,
    initial_latent_trajectory: jnp.ndarray | None,
) -> AuxKalmanMCMCState:
    context = bundle["latent_context_runtime_fn"](init_position, times)
    predictive_latent = bundle["initial_latent_from_context_fn"](context)
    latent_trajectory = (
        predictive_latent
        if initial_latent_trajectory is None
        else jnp.asarray(initial_latent_trajectory, dtype=predictive_latent.dtype)
    )
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_runtime_fn"](
        context,
        latent_trajectory,
        observations,
    )
    complete_lp, trajectory_lp = bundle[
        "complete_log_posterior_conditioned_from_context_runtime_fn"
    ](
        init_position,
        context,
        latent_trajectory,
        observation_auxiliary,
        observations,
    )
    latent_delta_value = jnp.asarray(latent_delta, dtype=latent_trajectory.dtype)
    param_step_value = _clip_scale(
        jnp.asarray(param_step_size, dtype=latent_trajectory.dtype),
        min_scale=param_min_scale,
        max_scale=param_max_scale,
    )
    da_init, _da_update, _ = dual_averaging_adaptation(target=float(param_target_accept))
    return AuxKalmanMCMCState(
        position=init_position,
        latent_context=context,
        latent_trajectory=latent_trajectory,
        observation_auxiliary=observation_auxiliary,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
        latent_delta=latent_delta_value,
        param_step_size=param_step_value,
        latent_da=da_init(latent_delta_value),
        param_da=da_init(param_step_value),
    )


@functools.partial(jax.jit, static_argnames=("step_fn",))
def _run_batched_step(
    states: AuxKalmanMCMCState,
    step_keys: jnp.ndarray,
    *,
    step_fn,
) -> tuple[AuxKalmanMCMCState, dict[str, jnp.ndarray]]:
    return jax.vmap(step_fn)(states, step_keys)


@functools.partial(jax.jit, static_argnames=("public_latent_fn",))
def _sample_public_latent_batch(
    states: AuxKalmanMCMCState,
    keys: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    public_latent_fn,
) -> jnp.ndarray:
    return jax.vmap(
        lambda state, key: public_latent_fn(
            state.latent_context,
            state.latent_trajectory,
            state.observation_auxiliary,
            observations,
            key,
        )
    )(states, keys)


def run_marginal_particle_gibbs(
    bundle: dict[str, Any],
    *,
    kernel: MarginalParticleGibbsKernel,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    adaptation_rate: float,
    init_scale: float,
    latent_delta: float,
    retain_latent_paths: bool,
    init_positions: jnp.ndarray | None = None,
    initial_latent_trajectories: jnp.ndarray | None = None,
    compute_latent_posterior_summary: bool = True,
) -> dict[str, Any]:
    """Run marginalized Particle Gibbs chains."""
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("marginal_particle_gibbs requires at least one MCMC step.")
    observations = bundle["observations"]
    times = bundle["times"]
    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    dim = int(bundle["flat_example"].shape[0])
    if init_positions is None:
        init_keys = random.split(init_key, num_chains)
        init_noise = jax.vmap(
            lambda key: random.normal(
                key,
                bundle["flat_example"].shape,
                dtype=bundle["flat_example"].dtype,
            )
        )(init_keys)
        chain_init_positions = bundle["flat_example"][None, :] + init_scale * init_noise
    else:
        chain_init_positions = jnp.asarray(init_positions, dtype=bundle["flat_example"].dtype)
        if chain_init_positions.shape != (num_chains, dim):
            raise ValueError(
                "init_positions must have shape (num_chains, dim); got "
                f"{chain_init_positions.shape} with num_chains={num_chains} and dim={dim}."
            )

    if initial_latent_trajectories is not None:
        chain_initial_latents = jnp.asarray(initial_latent_trajectories, dtype=observations.dtype)
        if chain_initial_latents.shape[0] != num_chains:
            raise ValueError(
                "initial_latent_trajectories must have leading dimension num_chains; got "
                f"{chain_initial_latents.shape[0]} with num_chains={num_chains}."
            )
    else:
        chain_initial_latents = None

    states = _stack_chain_states(
        [
            _initialize_chain_state(
                chain_init_positions[chain_idx],
                observations=observations,
                times=times,
                bundle=bundle,
                latent_delta=latent_delta,
                param_step_size=kernel.initial_param_step_size,
                param_min_scale=kernel.min_scale,
                param_max_scale=kernel.max_scale,
                param_target_accept=kernel.target_accept,
                initial_latent_trajectory=(
                    None if chain_initial_latents is None else chain_initial_latents[chain_idx]
                ),
            )
            for chain_idx in range(num_chains)
        ]
    )
    initial_param_step_size = states.param_step_size
    step_keys = random.split(chain_key, total_steps * num_chains * 2).reshape(
        total_steps,
        num_chains,
        2,
        2,
    )
    need_public_latent = compute_latent_posterior_summary or retain_latent_paths
    public_latent_fn = bundle["public_latent_trajectory_runtime_fn"]
    if compute_latent_posterior_summary:
        public_example = _sample_public_latent_batch(
            states,
            step_keys[0, :, 1, :],
            observations,
            public_latent_fn=public_latent_fn,
        )
        latent_sum = jnp.zeros_like(public_example)
        latent_sumsq = jnp.zeros_like(public_example)
        sample_count = jnp.asarray(0, dtype=jnp.int32)

    position_history: list[jnp.ndarray] = []
    parameter_accept_history: list[jnp.ndarray] = []
    latent_accept_history: list[jnp.ndarray] = []
    complete_lp_history: list[jnp.ndarray] = []
    latent_paths_history: list[jnp.ndarray] = []
    selected_label_history: list[jnp.ndarray] = []
    final_particle_history: list[jnp.ndarray] = []
    latent_move_rms_history: list[jnp.ndarray] = []
    latent_move_max_abs_history: list[jnp.ndarray] = []
    latent_move_rms_per_t_history: list[jnp.ndarray] = []

    progress_started = time.monotonic()
    progress_every = max(1, min(250, total_steps // 20))
    print(
        "marginal_particle_gibbs progress: "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"total_steps={total_steps} n_particles={kernel.num_particles} "
        f"n_parameter_particles={kernel.num_parameter_particles} "
        f"latent_block_size={kernel.latent_block_size} progress_every={progress_every}",
        flush=True,
    )

    sampling_loop_started = time.monotonic()
    first_step_seconds: float | None = None
    for step_idx in range(total_steps):
        step_started = time.monotonic()
        states, step_info = _run_batched_step(
            states,
            step_keys[step_idx, :, 0, :],
            step_fn=kernel.step_fn,
        )
        if (
            step_idx == 0
            or (step_idx + 1) % progress_every == 0
            or step_idx + 1 == num_warmup
            or step_idx + 1 == total_steps
        ):
            param_accept_now = jax.device_get(jnp.mean(step_info["parameter_accepted"]))
            latent_accept_now = jax.device_get(jnp.mean(step_info["latent_accepted"]))
            param_step_now = jax.device_get(states.param_step_size)
            complete_lp_now = jax.device_get(states.complete_log_posterior)
            phase = "warmup" if step_idx < num_warmup else "sample"
            elapsed = time.monotonic() - progress_started
            print(
                "marginal_particle_gibbs progress: "
                f"step={step_idx + 1}/{total_steps} phase={phase} elapsed={elapsed:.1f}s "
                f"parameter_accept_now={float(param_accept_now):.3f} "
                f"latent_update_now={float(latent_accept_now):.3f} "
                f"param_step_range=[{float(jnp.min(param_step_now)):.3g},"
                f"{float(jnp.max(param_step_now)):.3g}] "
                f"complete_lp_range=[{float(jnp.min(complete_lp_now)):.3g},"
                f"{float(jnp.max(complete_lp_now)):.3g}]",
                flush=True,
            )

        if step_idx == 0:
            states.complete_log_posterior.block_until_ready()
            first_step_seconds = time.monotonic() - step_started

        if step_idx < num_warmup:
            states = states._replace(
                param_step_size=_adapt_scale(
                    states.param_step_size,
                    accepted=step_info["parameter_accepted"],
                    target_accept=kernel.target_accept,
                    adaptation_rate=adaptation_rate,
                    min_scale=kernel.min_scale,
                    max_scale=kernel.max_scale,
                )
            )
            continue

        position_history.append(states.position)
        parameter_accept_history.append(step_info["parameter_accepted"])
        latent_accept_history.append(jnp.mean(step_info["latent_accepted"], axis=-1))
        complete_lp_history.append(states.complete_log_posterior)
        selected_label_history.append(step_info["selected_label"])
        final_particle_history.append(step_info["final_particle"])
        latent_move_rms_history.append(step_info["latent_move_rms"])
        latent_move_max_abs_history.append(step_info["latent_move_max_abs"])
        latent_move_rms_per_t_history.append(step_info["latent_move_rms_per_t"])

        if need_public_latent:
            public_latent = _sample_public_latent_batch(
                states,
                step_keys[step_idx, :, 1, :],
                observations,
                public_latent_fn=public_latent_fn,
            )
            if compute_latent_posterior_summary:
                latent_sum = latent_sum + public_latent
                latent_sumsq = latent_sumsq + public_latent * public_latent
                sample_count = sample_count + 1
            if retain_latent_paths:
                latent_paths_history.append(public_latent)

    states.complete_log_posterior.block_until_ready()
    sampling_loop_seconds = time.monotonic() - sampling_loop_started

    grouped_positions = _stack_sample_history(
        position_history,
        num_chains=num_chains,
        trailing_shape=(dim,),
        dtype=chain_init_positions.dtype,
    )
    chain_extra_fields = {
        "parameter_accept_prob": _stack_sample_history(
            parameter_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_accept_prob": _stack_sample_history(
            latent_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "selected_parameter_label": _stack_sample_history(
            selected_label_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "selected_particle": _stack_sample_history(
            final_particle_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_rms": _stack_sample_history(
            latent_move_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_max_abs": _stack_sample_history(
            latent_move_max_abs_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_move_rms_per_t": _stack_sample_history(
            latent_move_rms_per_t_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(latent_move_rms_per_t_history[0].shape[1:])
                if latent_move_rms_per_t_history
                else (int(states.latent_trajectory.shape[0]),)
            ),
            dtype=states.latent_trajectory.dtype,
        ),
    }
    complete_log_posterior_history = _stack_sample_history(
        complete_lp_history,
        num_chains=num_chains,
        trailing_shape=(),
        dtype=states.complete_log_posterior.dtype,
    )

    latent_summary = None
    if compute_latent_posterior_summary:
        denom = jnp.maximum(sample_count, 1).astype(latent_sum.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        latent_summary = _latent_summary_from_chain_moments(chain_mean, jnp.sqrt(chain_var))

    latent_paths = None
    if retain_latent_paths:
        latent_trailing_shape = (
            tuple(latent_paths_history[0].shape[1:])
            if latent_paths_history
            else tuple(
                _sample_public_latent_batch(
                    states,
                    step_keys[0, :, 1, :],
                    observations,
                    public_latent_fn=public_latent_fn,
                ).shape[1:]
            )
        )
        latent_paths = _stack_sample_history(
            latent_paths_history,
            num_chains=num_chains,
            trailing_shape=latent_trailing_shape,
            dtype=states.latent_trajectory.dtype,
        )

    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "complete_log_posterior_history": complete_log_posterior_history,
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
        "initial_param_step_size": initial_param_step_size,
        "final_param_step_size": states.param_step_size,
        "first_step_seconds": 0.0 if first_step_seconds is None else first_step_seconds,
        "sampling_loop_seconds": sampling_loop_seconds,
        "post_warmup_complete_log_posterior_mean": jnp.mean(
            complete_log_posterior_history,
            axis=1,
        ),
    }


def build_marginal_particle_gibbs_mcmc_result(
    *,
    chain_samples: dict[str, jnp.ndarray],
    chain_extra_fields: dict[str, jnp.ndarray],
    num_chains: int,
    num_samples: int,
) -> AuxKalmanMCMCResult:
    return AuxKalmanMCMCResult(
        chain_samples=chain_samples,
        chain_extra_fields=chain_extra_fields,
        num_chains=num_chains,
        num_samples=num_samples,
        backend="marginal_particle_gibbs",
    )
