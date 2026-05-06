"""Particle latent Gibbs kernels for complete-data SSM inference."""

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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    innovation_cov = symmetrize_with_jitter(prior_cov + auxiliary_cov, jitter=_AUX_JITTER)
    chol_innovation = jnp.linalg.cholesky(innovation_cov)
    gain = jla.cho_solve((chol_innovation, True), prior_cov).T
    conditioned_cov = symmetrize_with_jitter(
        prior_cov - gain @ innovation_cov @ gain.T,
        jitter=_AUX_JITTER,
    )
    chol_conditioned = jnp.linalg.cholesky(conditioned_cov)
    return gain, conditioned_cov, chol_conditioned, innovation_cov


def _marginal_log_h(
    x: jnp.ndarray,
    v: jnp.ndarray,
    x_bar: jnp.ndarray,
    v_bar: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    H: jnp.ndarray,
    D: jnp.ndarray,
    auxiliary_variance: jnp.ndarray,
    num_free_particles: int,
) -> jnp.ndarray:
    """Evaluate the generic marginal-algorithm weight term from Corenflos et al.

    This is Lemma 4 / Algorithm 15 instantiated with:

    * ``E = auxiliary_variance * I``
    * ``H = A_t`` (the fully adapted Gaussian conditioning gain)
    * ``D = C'_t`` (the conditioned covariance)
    """
    dtype = x.dtype
    n_free = jnp.asarray(num_free_particles, dtype=dtype)
    D = symmetrize_with_jitter(D, jitter=_AUX_JITTER)
    chol_D = jnp.linalg.cholesky(D)
    eye = jnp.eye(D.shape[0], dtype=dtype)
    D_inv = jla.cho_solve((chol_D, True), eye)
    heht = auxiliary_variance * (H @ H.T)
    g_system = symmetrize_with_jitter(D + n_free * heht, jitter=_AUX_JITTER)
    chol_g_system = jnp.linalg.cholesky(g_system)
    G = jla.cho_solve((chol_g_system, True), heht @ D_inv)
    G = 0.5 * (G + G.T)
    cross = (D_inv - n_free * G) @ H
    quad = 0.5 * (H.T @ cross + cross.T @ H)

    diff = x - v
    x_phi = x + phi
    mean_diff = x_bar - v_bar

    return (
        0.5 * diff @ (D_inv + G) @ diff
        - 0.5 * n_free * (x_phi @ quad @ x_phi)
        - (n_free + 1.0) * (mean_diff @ (G @ diff - cross @ x_phi))
        - diff @ cross @ x_phi
    )


def build_particle_mgrad_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
    num_particles: int,
    backward_sampling: bool,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> dict[str, Any]:
    """Build the Particle-mGRAD latent update from Corenflos & Finke (2024).

    This keeps the Particle-aGRAD proposal construction:

    1. draw pseudo-observations ``u_t ~ N(x_t + (δ_t/2) ∇ log G_t(x_t), (δ_t/2) I)``
       from the current reference trajectory,
    2. propose free particles from the fully adapted Gaussian conditional
       ``p(x_t | x_{t-1}, u_t)``,

    but uses the marginal Particle-mGRAD weights from Algorithm 7 / Lemma 4,
    i.e. with the auxiliary variables integrated out of the weight calculation.
    """
    if num_particles < 2:
        raise ValueError("particle_mgrad requires num_particles >= 2.")

    obs_increment_fn = bundle["observation_increment_log_prob_from_context_fn"]
    obs_increment_grad_fn = jax.grad(obs_increment_fn, argnums=1)
    ref_particle_index = num_particles - 1
    num_free_particles = num_particles - 1

    def _latent_particle_mgrad_step(state, key: jnp.ndarray):
        aux_key, init_key, forward_key, final_key, backward_master_key = random.split(key, 5)
        x_ref = state.latent_trajectory
        context = state.latent_context
        latent_dtype = x_ref.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        latent_dim = int(x_ref.shape[-1])
        num_steps = int(x_ref.shape[0])
        eye = jnp.eye(latent_dim, dtype=latent_dtype)
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

        init_mean, init_cov = _initial_latent_moments(context)
        gain0, conditioned_cov0, chol0, _innovation_cov0 = _condition_gaussian_on_auxiliary(
            init_cov,
            auxiliary_var[0] * eye,
        )
        init_guided_mean = init_mean + (u[0] - init_mean) @ gain0.T
        init_eps = random.normal(init_key, (num_free_particles, latent_dim), dtype=latent_dtype)
        init_free = init_guided_mean[None, :] + init_eps @ chol0.T
        x0_particles = jnp.concatenate([init_free, x_ref[0][None, :]], axis=0)
        init_mean_particles = jnp.broadcast_to(init_mean, x0_particles.shape)
        v0_particles = init_mean_particles - init_mean_particles @ gain0.T
        x0_bar = jnp.mean(x0_particles, axis=0)
        v0_bar = jnp.mean(v0_particles, axis=0)
        phi0_particles = auxiliary_var[0] * jax.vmap(
            lambda particle: jnp.asarray(
                obs_increment_grad_fn(context, particle, jnp.asarray(0, dtype=jnp.int32)),
                dtype=latent_dtype,
            )
        )(x0_particles)
        prior_lp0 = jax.vmap(
            lambda particle: _gaussian_log_prob_full(particle, init_mean, init_cov)
        )(x0_particles)
        obs_lp0 = jax.vmap(
            lambda particle: obs_increment_fn(context, particle, jnp.asarray(0, dtype=jnp.int32))
        )(x0_particles)
        log_h0 = jax.vmap(
            lambda particle, v_particle, phi_particle: _marginal_log_h(
                particle,
                v_particle,
                x0_bar,
                v0_bar,
                phi_particle,
                H=gain0,
                D=conditioned_cov0,
                auxiliary_variance=auxiliary_var[0],
                num_free_particles=num_free_particles,
            )
        )(x0_particles, v0_particles, phi0_particles)
        log_w0 = prior_lp0 + obs_lp0 + log_h0

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

                parent_all = x_prev_particles[ancestors]
                pred_all = parent_all @ context.Ad[time_idx].T + context.cd[time_idx]
                gain_t, conditioned_cov_t, chol_t, _innovation_cov_t = (
                    _condition_gaussian_on_auxiliary(
                        context.Qd[time_idx],
                        auxiliary_var[time_idx] * eye,
                    )
                )
                pred_free = pred_all[:num_free_particles]
                prop_mean_free = pred_free + (u[time_idx] - pred_free) @ gain_t.T
                eps_t = random.normal(
                    propagate_key,
                    (num_free_particles, latent_dim),
                    dtype=latent_dtype,
                )
                x_free = prop_mean_free + eps_t @ chol_t.T
                x_particles = jnp.concatenate([x_free, x_ref[time_idx][None, :]], axis=0)

                v_particles = pred_all - pred_all @ gain_t.T
                x_bar = jnp.mean(x_particles, axis=0)
                v_bar = jnp.mean(v_particles, axis=0)
                phi_particles = auxiliary_var[time_idx] * jax.vmap(
                    lambda particle: jnp.asarray(
                        obs_increment_grad_fn(context, particle, time_idx),
                        dtype=latent_dtype,
                    )
                )(x_particles)
                prior_lp = jax.vmap(
                    lambda particle, mean: _gaussian_log_prob_full(
                        particle,
                        mean,
                        context.Qd[time_idx],
                    )
                )(x_particles, pred_all)
                obs_lp = jax.vmap(lambda particle: obs_increment_fn(context, particle, time_idx))(
                    x_particles
                )
                log_h = jax.vmap(
                    lambda particle, v_particle, phi_particle: _marginal_log_h(
                        particle,
                        v_particle,
                        x_bar,
                        v_bar,
                        phi_particle,
                        H=gain_t,
                        D=conditioned_cov_t,
                        auxiliary_variance=auxiliary_var[time_idx],
                        num_free_particles=num_free_particles,
                    )
                )(x_particles, v_particles, phi_particles)
                log_w = prior_lp + obs_lp + log_h
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
        "name": "particle_mgrad",
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "initial_scale_value": delta,
        "initial_scale_mode": "per_time_constant",
        "min_scale": min_scale,
        "max_scale": max_scale,
        "target_accept": target_accept,
        "step_fn": _latent_particle_mgrad_step,
    }
