"""Particle-mGRAD latent smoother bridging to the PIT particle-mGRAD kernel."""

# References:
#   docs/papers/particle-mala-mgrad.pdf — Corenflos & Finke (2024), "Particle-MALA
#     and Particle-mGRAD" (arXiv:2401.14868): this module evaluates the Particle-mGRAD
#     proposal (built in pit_particle_mgrad.py) against the parameter mixture.

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsLatentSmootherResult,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _select_pytree,
)


def smooth(ctx, key, x_ref):
    """Forced-move particle-mGRAD proposal evaluated against the parameter mixture."""
    contexts = ctx.contexts
    parameter_particles = ctx.parameter_particles
    initial_label_log_probs = ctx.initial_label_log_probs
    state = ctx.state
    runtime_observations = ctx.runtime_observations
    initial_observation_auxiliary_fn = ctx.initial_observation_auxiliary_fn
    trajectory_log_prob_fn = ctx.trajectory_log_prob_fn
    prior_terms_from_context_fn = ctx.prior_terms_from_context_fn
    log_prior_unc_fn = ctx.log_prior_unc_fn
    mgrad_latent_kernel = ctx.mgrad_latent_kernel
    latent_dtype = ctx.latent_dtype
    traj_dtype = ctx.traj_dtype
    complete_dtype = ctx.complete_dtype
    _trajectory_label_log_probs = ctx.trajectory_label_log_probs
    smoother_key = key
    current_path = x_ref

    proposal_label_key, mgrad_key = random.split(smoother_key)
    proposal_label = random.categorical(
        proposal_label_key,
        initial_label_log_probs,
    ).astype(jnp.int32)
    proposal_position = parameter_particles[proposal_label]
    proposal_context = _select_pytree(contexts, proposal_label)
    proposal_observation_auxiliary = initial_observation_auxiliary_fn(
        proposal_context,
        current_path,
        runtime_observations,
    )
    proposal_prior_terms = prior_terms_from_context_fn(proposal_context)
    proposal_traj_lp = jnp.asarray(
        trajectory_log_prob_fn(
            proposal_context,
            current_path,
            proposal_observation_auxiliary,
            runtime_observations,
            prior_terms=proposal_prior_terms,
        ),
        dtype=traj_dtype,
    )
    proposal_complete = jnp.asarray(
        log_prior_unc_fn(proposal_position),
        dtype=complete_dtype,
    )
    proposal_complete = proposal_complete + proposal_traj_lp.astype(complete_dtype)
    proposal_state = state._replace(
        position=proposal_position,
        latent_context=proposal_context,
        latent_trajectory=current_path,
        observation_auxiliary=proposal_observation_auxiliary,
        trajectory_log_prob=proposal_traj_lp,
        complete_log_posterior=proposal_complete,
        latent_delta=jnp.asarray(state.latent_delta, dtype=latent_dtype),
    )
    next_state, mgrad_info = mgrad_latent_kernel["step_fn"](proposal_state, mgrad_key)
    latent_path = next_state.latent_trajectory.astype(latent_dtype)
    origin_path = mgrad_info["accepted"].astype(jnp.int32)
    final_label_log_probs = _trajectory_label_log_probs(latent_path)
    return MPGibbsLatentSmootherResult(
        latent_path=latent_path,
        final_label_log_probs=final_label_log_probs,
        origin_path=origin_path,
        diagnostics={},
    )
