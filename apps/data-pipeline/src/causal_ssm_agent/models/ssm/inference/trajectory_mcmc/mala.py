"""Parameter-kernel builders for complete-data trajectory MCMC."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random


def _gaussian_log_prob_isotropic(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    diff = jnp.reshape(value - mean, (-1,))
    dim = diff.shape[0]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi * variance) + jnp.sum(diff * diff) / variance)


def build_mala_parameter_kernel(
    bundle: dict[str, Any],
    *,
    step_size: float,
    target_accept: float,
) -> dict[str, Any]:
    """Build a parameter MALA kernel for the complete-data target."""
    complete_value_and_grad = jax.value_and_grad(
        bundle["complete_log_posterior_with_aux_fn"],
        argnums=0,
        has_aux=True,
    )

    def _parameter_mala_step(state, key: jnp.ndarray):
        if bundle["dim"] == 0:
            return state, {"accepted": jnp.asarray(1.0, dtype=state.latent_trajectory.dtype)}

        proposal_key, accept_key = random.split(key)
        (complete_curr, _traj_curr), grad_curr = complete_value_and_grad(
            state.position,
            state.latent_trajectory,
        )
        mean_fwd = state.position + 0.5 * (state.param_step_size**2) * grad_curr
        proposal = mean_fwd + state.param_step_size * random.normal(
            proposal_key,
            state.position.shape,
            dtype=state.position.dtype,
        )
        (complete_prop, traj_prop), grad_prop = complete_value_and_grad(
            proposal,
            state.latent_trajectory,
        )
        mean_rev = proposal + 0.5 * (state.param_step_size**2) * grad_prop
        log_alpha = complete_prop - complete_curr
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(
            state.position,
            mean_rev,
            state.param_step_size**2,
        )
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(
            proposal,
            mean_fwd,
            state.param_step_size**2,
        )
        finite = jnp.isfinite(log_alpha) & jnp.isfinite(complete_prop)
        accept_prob = jnp.where(finite, jnp.exp(jnp.minimum(log_alpha, 0.0)), 0.0)
        accept = random.bernoulli(accept_key, accept_prob)
        next_state = state._replace(
            position=jnp.where(accept, proposal, state.position),
            trajectory_log_prob=jnp.where(accept, traj_prop, state.trajectory_log_prob),
            complete_log_posterior=jnp.where(accept, complete_prop, state.complete_log_posterior),
        )
        return next_state, {"accepted": accept.astype(state.position.dtype)}

    return {
        "name": "mala",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_mala_step,
    }
