"""Auxiliary cSMC latent Gibbs kernel for complete-data SSM inference."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.ssm.covariance_utils import symmetrize_with_jitter
from causal_ssm_agent.models.ssm.inference.trajectory_mcmc.auxiliary_kalman import (
    _AUX_JITTER,
    _initial_latent_moments,
)


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


def _condition_gaussian_on_auxiliary(
    prior_cov: jnp.ndarray,
    auxiliary_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    innovation_cov = symmetrize_with_jitter(prior_cov + auxiliary_cov, jitter=_AUX_JITTER)
    chol_innovation = jnp.linalg.cholesky(innovation_cov)
    gain = jla.cho_solve((chol_innovation, True), prior_cov).T
    conditioned_cov = symmetrize_with_jitter(
        prior_cov - gain @ innovation_cov @ gain.T,
        jitter=_AUX_JITTER,
    )
    chol_conditioned = jnp.linalg.cholesky(conditioned_cov)
    return gain, chol_conditioned, innovation_cov


def build_auxiliary_csmc_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
    num_particles: int,
    backward_sampling: bool,
) -> dict[str, Any]:
    """Build the auxiliary cSMC latent update from Corenflos & Sarkka (2025).

    The kernel samples auxiliary observations ``u_t ~ N(x_t, delta/2 * M)`` from
    the current trajectory and then applies a conditional SMC sweep targeting the
    conditional auxiliary model using the Gaussian-guided proposal of eqs. (38-40).
    The blocked Gibbs update is exact, so the latent state is always refreshed;
    ``accepted`` reports the fraction of time indices whose selected particle is
    not the conditioned reference path. That signal is what ``run_aux_gibbs``
    adapts against for the latent step size.
    """
    if num_particles < 2:
        raise ValueError("aux_csmc requires num_particles >= 2.")

    obs_increment_fn = bundle["observation_increment_log_prob_from_context_fn"]
    ref_particle_index = num_particles - 1

    def _latent_csmc_step(state, key: jnp.ndarray):
        aux_key, init_key, forward_key, final_key, backward_master_key = random.split(key, 5)
        x_ref = state.latent_trajectory
        context = state.latent_context
        latent_dtype = x_ref.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        latent_dim = int(x_ref.shape[-1])
        num_steps = int(x_ref.shape[0])
        num_free_particles = num_particles - 1
        delta_per_t = jnp.broadcast_to(
            jnp.asarray(state.latent_delta, dtype=latent_dtype),
            (num_steps,),
        )
        auxiliary_var = jnp.broadcast_to(
            0.5 * delta_per_t[:, None],
            (num_steps, latent_dim),
        )
        auxiliary_chol = jnp.sqrt(auxiliary_var)

        u = x_ref + auxiliary_chol * random.normal(aux_key, x_ref.shape, dtype=latent_dtype)

        init_mean, init_cov = _initial_latent_moments(context)
        gain0, chol0, innovation_cov0 = _condition_gaussian_on_auxiliary(
            init_cov,
            jnp.diag(auxiliary_var[0]),
        )
        init_guided_mean = init_mean + (u[0] - init_mean) @ gain0.T
        init_eps = random.normal(init_key, (num_free_particles, latent_dim), dtype=latent_dtype)
        init_free = init_guided_mean[None, :] + init_eps @ chol0.T
        x0_particles = jnp.concatenate([init_free, x_ref[0][None, :]], axis=0)
        log_w0 = jax.vmap(
            lambda particle: obs_increment_fn(context, particle, jnp.asarray(0, dtype=jnp.int32))
        )(x0_particles)
        log_w0 = log_w0 + _gaussian_log_prob_full(u[0], init_mean, innovation_cov0)

        if num_steps == 1:
            xs_history = x0_particles[None, ...]
            log_w_history = log_w0[None, ...]
            ancestor_history = jnp.zeros((0, num_particles), dtype=jnp.int32)
        else:
            forward_keys = random.split(forward_key, num_steps - 1)

            def _forward_step(carry, inputs):
                x_prev_particles, log_w_prev = carry
                time_idx, step_key = inputs
                resample_key, propagate_key = random.split(step_key)
                ancestor_free = random.categorical(
                    resample_key,
                    log_w_prev,
                    shape=(num_free_particles,),
                ).astype(jnp.int32)
                ancestors = jnp.concatenate(
                    [
                        ancestor_free,
                        jnp.asarray([ref_particle_index], dtype=jnp.int32),
                    ],
                    axis=0,
                )

                parent_free = x_prev_particles[ancestor_free]
                pred_free = parent_free @ context.Ad[time_idx].T + context.cd[time_idx]
                gain_t, chol_t, innovation_cov_t = _condition_gaussian_on_auxiliary(
                    context.Qd[time_idx],
                    jnp.diag(auxiliary_var[time_idx]),
                )
                prop_mean_free = pred_free + (u[time_idx] - pred_free) @ gain_t.T
                eps_t = random.normal(
                    propagate_key,
                    (num_free_particles, latent_dim),
                    dtype=latent_dtype,
                )
                x_free = prop_mean_free + eps_t @ chol_t.T
                x_particles = jnp.concatenate([x_free, x_ref[time_idx][None, :]], axis=0)

                parent_all = x_prev_particles[ancestors]
                pred_all = parent_all @ context.Ad[time_idx].T + context.cd[time_idx]
                aux_norm = jax.vmap(
                    lambda mean: _gaussian_log_prob_full(u[time_idx], mean, innovation_cov_t)
                )(pred_all)
                obs_lp = jax.vmap(lambda particle: obs_increment_fn(context, particle, time_idx))(
                    x_particles
                )
                log_w = obs_lp + aux_norm
                return (x_particles, log_w), (x_particles, log_w, ancestors)

            (_, _), (xs_tail, log_w_tail, ancestor_history) = jax.lax.scan(
                _forward_step,
                (x0_particles, log_w0),
                (
                    jnp.arange(1, num_steps, dtype=jnp.int32),
                    forward_keys,
                ),
            )
            xs_history = jnp.concatenate([x0_particles[None, ...], xs_tail], axis=0)
            log_w_history = jnp.concatenate([log_w0[None, ...], log_w_tail], axis=0)

        final_idx = random.categorical(final_key, log_w_history[-1]).astype(jnp.int32)
        final_x = xs_history[-1, final_idx]

        if num_steps == 1:
            index_path = final_idx[None]
            latent_path = final_x[None, :]
        elif backward_sampling:
            backward_keys = random.split(backward_master_key, num_steps - 1)

            def _backward_step(next_latent, inputs):
                xs_t, log_w_t, Ad_next, Qd_next, cd_next, step_key = inputs
                pred_means = xs_t @ Ad_next.T + cd_next
                transition_log_probs = jax.vmap(
                    lambda mean: _gaussian_log_prob_full(next_latent, mean, Qd_next)
                )(pred_means)
                idx_t = random.categorical(step_key, log_w_t + transition_log_probs).astype(
                    jnp.int32
                )
                latent_t = xs_t[idx_t]
                return latent_t, (latent_t, idx_t)

            _, (latent_rev, index_rev) = jax.lax.scan(
                _backward_step,
                final_x,
                (
                    xs_history[:-1][::-1],
                    log_w_history[:-1][::-1],
                    context.Ad[1:][::-1],
                    context.Qd[1:][::-1],
                    context.cd[1:][::-1],
                    backward_keys,
                ),
            )
            latent_path = jnp.concatenate([latent_rev[::-1], final_x[None, :]], axis=0)
            index_path = jnp.concatenate([index_rev[::-1], final_idx[None]], axis=0)
        else:

            def _trace_step(next_idx, inputs):
                xs_t, ancestors_next = inputs
                idx_t = ancestors_next[next_idx]
                latent_t = xs_t[idx_t]
                return idx_t, (latent_t, idx_t)

            _, (latent_rev, index_rev) = jax.lax.scan(
                _trace_step,
                final_idx,
                (
                    xs_history[:-1][::-1],
                    ancestor_history[::-1],
                ),
            )
            latent_path = jnp.concatenate([latent_rev[::-1], final_x[None, :]], axis=0)
            index_path = jnp.concatenate([index_rev[::-1], final_idx[None]], axis=0)

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
        updated_mask = (index_path != ref_particle_index).astype(state.position.dtype)

        next_state = state._replace(
            latent_trajectory=latent_path,
            trajectory_log_prob=next_traj_lp,
            complete_log_posterior=next_complete,
        )
        return next_state, {"accepted": updated_mask}

    return {
        "name": "csmc",
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "initial_scale_from_latent_fn": (
            lambda latent_trajectory, dtype: jnp.full(
                (latent_trajectory.shape[0],),
                jnp.asarray(delta, dtype=dtype),
                dtype=dtype,
            )
        ),
        "target_accept": target_accept,
        "step_fn": _latent_csmc_step,
    }
