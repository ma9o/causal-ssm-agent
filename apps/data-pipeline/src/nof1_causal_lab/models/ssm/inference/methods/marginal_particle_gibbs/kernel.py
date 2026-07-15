"""Marginalized Particle Gibbs joint parameter/trajectory kernel.

Implements the M=2-by-default collapsed Particle Gibbs construction from
Corenflos (2025), "Particle Gibbs without the Gibbs bit", for directly
evaluable SSM potentials. The parameter proposal is formed in unconstrained
space via the auxiliary decomposition

    u | theta  ~ N(theta + 2 delta Sigma g(theta), 2 delta Sigma)
    theta' | u ~ N(u, 2 delta Sigma)

where ``g`` is a conditional parameter-gradient oracle. ``parameter_proposal``
selects the drift: ``random_walk`` sets ``g = 0`` (the symmetric special case),
while the default ``pseudo_langevin`` (Corenflos 2025, §3.1) drifts the
theta->u half by ``g`` so the two halves reproduce preconditioned MALA. The
resulting asymmetry is corrected exactly in the Barker label-selection weights
(identically zero in the random-walk case). The latent trajectory is updated by
conditional SMC against the posterior mixture over the parameter ensemble.

Step-size convention. ``delta`` here (``param_step_size``) is the variance
coefficient of each half, not the conventional MALA step. The combined kernel
has variance ``4 delta Sigma`` (and drift ``2 delta Sigma g`` for pseudo-Langevin),
so ``param_step_size`` maps to MALA's ``h`` and to the RW variance coefficient
via ``h = 4 * param_step_size`` in both branches. Tune accordingly: a familiar
MALA step of ``h = 0.1`` corresponds to ``param_step_size = 0.025`` here.
"""

# References:
#   docs/papers/particle-gibbs-no-gibbs-bit.pdf — Corenflos (2025), "Particle Gibbs
#     without the Gibbs bit" (arXiv:2505.04611): the collapsed M-ensemble parameter
#     proposal (§3.1 pseudo-Langevin) and Barker label-selection weights live here.

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.models.ssm.inference import _profiling
from nof1_causal_lab.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    _latent_summary_from_chain_moments,
    _stack_chain_states,
    _stack_sample_history,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._context import (
    build_smoother_context,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _DSMC_LEAF_PROPOSAL_PAID_MIX,
    _DSMC_LEAF_PROPOSALS,
    _LATENT_SMOOTHER_DSMC,
    DSMCLeafProposal,
    MPGibbsLatentSmoother,
    MPGibbsStatic,
    _resolve_latent_smoother,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _select_pytree,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
    resolve_mpgibbs_diagnostic_metrics,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.smoothers import (
    SMOOTHERS,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.bundle import ParticleRuntimeBundle

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
_DEFAULT_AMALA_DELTA_INIT = 1e-2
_DEFAULT_AMALA_DELTA_MIN = 1e-5
_DEFAULT_AMALA_DELTA_MAX = 1e1
_DEFAULT_AMALA_TARGET_ACCEPT = 0.75
_DEFAULT_AMALA_ADAPTATION_WINDOW = 100
_DEFAULT_AMALA_ADAPTATION_TOLERANCE = 0.05
_DEFAULT_AMALA_ADAPTATION_RHO = 0.5
_DEFAULT_AMALA_ADAPTATION_RHO_MIN = 1e-3
_DEFAULT_AMALA_ADAPTATION_GAMMA = -0.5
_DEFAULT_AMALA_GRAD_CLIP = float("inf")
# Target acceptance for the M=2 ensemble Barker selection. Its move-rate is
# bounded above by ~0.5 -- the step->0 coin-flip limit, (M-1)/M for M ensemble
# candidates -- so MALA-style optima (~0.574) do NOT transfer: a target above
# that ceiling makes dual averaging chase an unreachable rate and collapse the
# step size to the floor (observed empirically at 0.57). 0.35 sits safely below
# the ceiling, is reachable under dual averaging, and matches the long-standing
# baseline. The ceiling is a property of the M=2 selection, not of the proposal
# drift, so this default is shared by random_walk and pseudo_langevin.
_DEFAULT_PARAM_TARGET_ACCEPT = 0.35


def _uses_amala_delta(latent_smoother: MPGibbsLatentSmoother) -> bool:
    # Every dsmc leaf is z-anchored on the reference with scale delta/2, so the
    # per-time delta adaptation always applies.
    return latent_smoother.name == _LATENT_SMOOTHER_DSMC


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
    latent_smoother: MPGibbsLatentSmoother
    latent_delta: float
    amala_delta_init: float
    amala_delta_min: float
    amala_delta_max: float
    amala_target_accept: float
    amala_adaptation_window: int
    amala_adaptation_tolerance: float
    amala_adaptation_rho: float
    amala_adaptation_rho_min: float
    amala_adaptation_gamma: float
    adapt_amala_delta: bool
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: DSMCLeafProposal
    latent_block_coords: int | None
    diagnostic_metrics: frozenset[str]


def build_marginal_particle_gibbs_kernel(
    bundle: ParticleRuntimeBundle,
    *,
    num_particles: int,
    num_parameter_particles: int,
    param_step_size: float,
    target_accept: float | None = None,
    min_scale: float = _DEFAULT_MIN_SCALE,
    max_scale: float = _DEFAULT_MAX_SCALE,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_proposal: str = "pseudo_langevin",
    latent_smoother: str = _LATENT_SMOOTHER_DSMC,
    latent_delta: float = 0.2,
    amala_delta_init: float = _DEFAULT_AMALA_DELTA_INIT,
    amala_delta_min: float = _DEFAULT_AMALA_DELTA_MIN,
    amala_delta_max: float = _DEFAULT_AMALA_DELTA_MAX,
    amala_target_accept: float = _DEFAULT_AMALA_TARGET_ACCEPT,
    amala_adaptation_window: int = _DEFAULT_AMALA_ADAPTATION_WINDOW,
    amala_adaptation_tolerance: float = _DEFAULT_AMALA_ADAPTATION_TOLERANCE,
    amala_adaptation_rho: float = _DEFAULT_AMALA_ADAPTATION_RHO,
    amala_adaptation_rho_min: float = _DEFAULT_AMALA_ADAPTATION_RHO_MIN,
    amala_adaptation_gamma: float = _DEFAULT_AMALA_ADAPTATION_GAMMA,
    amala_kappa: float = 0.75,
    amala_grad_clip: float = _DEFAULT_AMALA_GRAD_CLIP,
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_exact",
    latent_block_coords: int | None = None,
    paid_mix_z_weight: float = 0.85,
    paid_mix_pilot_weight: float = 0.10,
    pilot_means: jnp.ndarray | None = None,
    pilot_vars: jnp.ndarray | None = None,
    pilot_wide_vars: jnp.ndarray | None = None,
    sign_flip_spec: Any | None = None,
    diagnostic_metrics_all: bool = False,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = None,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    latent_smoother_spec = _resolve_latent_smoother(latent_smoother)
    resolved_diagnostic_metrics = resolve_mpgibbs_diagnostic_metrics(
        diagnostic_metrics_all=diagnostic_metrics_all,
        diagnostic_metrics=diagnostic_metrics,
    )
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        diagnostic_metrics=resolved_diagnostic_metrics,
    )
    if num_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_particles >= 2.")
    if num_parameter_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_parameter_particles >= 2.")
    if min_scale > max_scale:
        raise ValueError(
            "marginal_particle_gibbs min_scale must be <= max_scale; "
            f"got {min_scale} > {max_scale}."
        )
    if parameter_proposal not in ("random_walk", "pseudo_langevin"):
        raise ValueError(
            "marginal_particle_gibbs parameter_proposal must be 'random_walk' or "
            f"'pseudo_langevin'; got {parameter_proposal!r}."
        )
    if amala_delta_init <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_delta_init must be positive; got {amala_delta_init}."
        )
    if amala_delta_min <= 0.0 or amala_delta_max <= 0.0 or amala_delta_min > amala_delta_max:
        raise ValueError(
            "marginal_particle_gibbs requires 0 < amala_delta_min <= amala_delta_max; "
            f"got {amala_delta_min} and {amala_delta_max}."
        )
    if not (amala_delta_min <= amala_delta_init <= amala_delta_max):
        raise ValueError(
            "marginal_particle_gibbs requires amala_delta_init inside "
            f"[amala_delta_min, amala_delta_max]; got {amala_delta_init}."
        )
    if not (0.0 < amala_target_accept < 1.0):
        raise ValueError(
            "marginal_particle_gibbs amala_target_accept must be in (0, 1); "
            f"got {amala_target_accept}."
        )
    if amala_adaptation_window < 1:
        raise ValueError(
            "marginal_particle_gibbs amala_adaptation_window must be positive; "
            f"got {amala_adaptation_window}."
        )
    if amala_adaptation_tolerance < 0.0:
        raise ValueError(
            "marginal_particle_gibbs amala_adaptation_tolerance must be non-negative; "
            f"got {amala_adaptation_tolerance}."
        )
    if amala_adaptation_rho <= 0.0 or amala_adaptation_rho_min <= 0.0:
        raise ValueError(
            "marginal_particle_gibbs amala adaptation rates must be positive; "
            f"got rho={amala_adaptation_rho}, rho_min={amala_adaptation_rho_min}."
        )
    if amala_kappa < 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_kappa must be non-negative; got {amala_kappa}."
        )
    if amala_grad_clip <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_grad_clip must be positive; got {amala_grad_clip}."
        )
    if dsmc_leaf_proposal not in _DSMC_LEAF_PROPOSALS:
        allowed = ", ".join(repr(candidate) for candidate in _DSMC_LEAF_PROPOSALS)
        raise ValueError(
            "marginal_particle_gibbs dsmc_leaf_proposal must be one of "
            f"{allowed}; got {dsmc_leaf_proposal!r}."
        )
    if latent_block_coords is not None and latent_block_coords < 1:
        raise ValueError(
            "marginal_particle_gibbs latent_block_coords must be a positive coordinate "
            f"count or None (all coordinates); got {latent_block_coords}."
        )
    if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PAID_MIX:
        if not (0.0 < paid_mix_z_weight < 1.0) or not (0.0 < paid_mix_pilot_weight < 1.0):
            raise ValueError(
                "marginal_particle_gibbs paid_mix weights must lie in (0, 1); got "
                f"z={paid_mix_z_weight}, pilot={paid_mix_pilot_weight}."
            )
        if paid_mix_z_weight + paid_mix_pilot_weight >= 1.0:
            raise ValueError(
                "marginal_particle_gibbs paid_mix weights must leave a positive wide-tail "
                f"share; got z + pilot = {paid_mix_z_weight + paid_mix_pilot_weight}."
            )
        if pilot_means is None or pilot_vars is None or pilot_wide_vars is None:
            raise ValueError(
                "marginal_particle_gibbs dsmc_leaf_proposal='paid_mix' requires pilot "
                "moments (pilot_means, pilot_vars, pilot_wide_vars) from the IEKS warmup."
            )
    use_gradient_drift = parameter_proposal == "pseudo_langevin"
    if target_accept is None:
        target_accept = _DEFAULT_PARAM_TARGET_ACCEPT

    latent_context_runtime_fn = bundle.cached.latent_context_runtime_fn
    log_prior_unc_fn = bundle.cached.log_prior_unc_fn
    initial_latent_moments_fn = bundle.cached.initial_latent_moments_from_context_fn
    obs_increment_fn = bundle.cached.observation_increment_log_prob_from_context_runtime_fn
    trajectory_log_prob_fn = bundle.cached.trajectory_log_prob_from_context_runtime_fn
    prior_terms_from_context_fn = bundle.cached.prior_terms_from_context_fn
    runtime_observations = bundle.observations
    runtime_times = bundle.times
    complete_log_posterior_runtime_fn = bundle.cached.complete_log_posterior_runtime_fn

    def _theta_logpost_grad(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        # Conditional parameter-gradient oracle g(z) ≈ ∇_z log π(z): the gradient of the
        # complete log-posterior at the current latent trajectory. Used only to drift the
        # pseudo-Langevin proposal; exactness is preserved by the Barker weight correction
        # regardless of this oracle's accuracy (a single reverse pass, not second-order).
        return jax.grad(
            lambda zz: complete_log_posterior_runtime_fn(
                zz, latent_trajectory, runtime_observations, runtime_times
            )
        )(z)

    preconditioner = (
        None
        if parameter_preconditioner_chol is None
        else jnp.asarray(
            parameter_preconditioner_chol,
            dtype=bundle.cached.flat_example.dtype,
        )
    )

    def _propose_parameter_ensemble(
        current_position: jnp.ndarray,
        key: jnp.ndarray,
        step_size: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dim = int(current_position.shape[0])
        step = jnp.asarray(step_size, dtype=current_position.dtype)
        proposal_scale = jnp.sqrt(jnp.asarray(2.0, dtype=current_position.dtype) * step)
        if preconditioner is None:
            chol = jnp.eye(dim, dtype=current_position.dtype)
        else:
            chol = jnp.asarray(preconditioner, dtype=current_position.dtype)
        aux_key, proposal_key = random.split(key)

        # Pseudo-Langevin (Corenflos 2025, §3.1): drift the θ→u half by 2·step·Σ·g(θ),
        # with Σ = chol·cholᵀ and g the conditional parameter-gradient oracle. The θ'←u
        # half stays a plain Gaussian, so only u carries the drift. Combined, the two
        # halves reproduce preconditioned MALA; random-walk is the zero-drift case.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_drift"):
                grad_ref = _theta_logpost_grad(current_position, latent_trajectory)
                drift = (jnp.asarray(2.0, dtype=step.dtype) * step) * (chol @ (chol.T @ grad_ref))
        else:
            drift = jnp.zeros_like(current_position)

        u = (
            current_position
            + drift
            + proposal_scale
            * (
                random.normal(aux_key, current_position.shape, dtype=current_position.dtype)
                @ chol.T
            )
        )
        proposal_eps = random.normal(
            proposal_key,
            (num_parameter_particles - 1, dim),
            dtype=current_position.dtype,
        )
        proposed = u[None, :] + proposal_scale * (proposal_eps @ chol.T)
        ensemble = jnp.concatenate([current_position[None, :], proposed], axis=0)

        # Barker label-prior correction Δ_l = log q(u|θˡ) − log q(θˡ|u). With the drift
        # only in q(u|·), Δ_l = g(θˡ)·(u−θˡ) − step·‖cholᵀ g(θˡ)‖². It is identically 0
        # in the random-walk (symmetric) case, recovering eq. (14). Added to the label
        # prior so the marginalized-PGibbs selection stays exact under the asymmetry.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_barker"):
                grads = jnp.concatenate(
                    [
                        grad_ref[None, :],
                        jax.vmap(lambda th: _theta_logpost_grad(th, latent_trajectory))(proposed),
                    ],
                    axis=0,
                )
                whitened_sq = jnp.sum((grads @ chol) ** 2, axis=1)
                drift_dot = jnp.sum(grads * (u[None, :] - ensemble), axis=1)
                label_correction = drift_dot - step * whitened_sq
        else:
            label_correction = jnp.zeros((num_parameter_particles,), dtype=current_position.dtype)

        return ensemble, label_correction

    static = MPGibbsStatic(
        latent_context_runtime_fn=latent_context_runtime_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        initial_latent_moments_fn=initial_latent_moments_fn,
        obs_increment_fn=obs_increment_fn,
        trajectory_log_prob_fn=trajectory_log_prob_fn,
        prior_terms_from_context_fn=prior_terms_from_context_fn,
        runtime_observations=runtime_observations,
        runtime_times=runtime_times,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        latent_delta=latent_delta,
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        dsmc_leaf_proposal=dsmc_leaf_proposal,
        latent_block_coords=latent_block_coords,
        paid_mix_z_weight=paid_mix_z_weight,
        paid_mix_pilot_weight=paid_mix_pilot_weight,
        pilot_means=pilot_means,
        pilot_vars=pilot_vars,
        pilot_wide_vars=pilot_wide_vars,
        transition_initial_log_prob_fn=bundle.cached.transition_initial_log_prob_from_context_fn,
        transition_log_prob_fn=bundle.cached.transition_log_prob_from_context_fn,
        transition_log_probs_for_pairs_fn=bundle.cached.transition_log_probs_for_pairs_from_context_fn,
        transition_pairwise_log_probs_fn=bundle.cached.transition_pairwise_log_probs_from_context_fn,
        transition_sample_fn=bundle.cached.transition_sample_from_context_fn,
        diagnostic_metrics=resolved_diagnostic_metrics,
    )

    def _step_fn(state: TrajectoryMCMCState, key: jnp.ndarray):
        if sign_flip_spec is not None:
            param_key, block_key, label_key, flip_choice_key, flip_accept_key = random.split(key, 5)
        else:
            param_key, block_key, label_key = random.split(key, 3)
        x_ref = state.latent_trajectory
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        with jax.named_scope("propose_parameters"):
            parameter_particles, label_correction = _propose_parameter_ensemble(
                state.position,
                param_key,
                _clip_scale(
                    state.param_step_size,
                    min_scale=min_scale,
                    max_scale=max_scale,
                ),
                x_ref,
            )

        with jax.named_scope("build_context"):
            ctx = build_smoother_context(static, state, parameter_particles, label_correction)
        contexts = ctx.contexts
        with jax.named_scope(latent_smoother_spec.name + "_smoother_full"):
            smoother_result = SMOOTHERS[latent_smoother_spec.name](ctx, block_key, x_ref)

        with jax.named_scope("postprocess"):
            latent_path = smoother_result.latent_path
            final_label_log_probs = smoother_result.final_label_log_probs
            origin_path = smoother_result.origin_path
            zero_scalar = jnp.asarray(0.0, dtype=traj_dtype)
            amala_grad_norm_mean = smoother_result.diagnostics.get(
                "amala_grad_norm_mean",
                zero_scalar,
            )
            amala_grad_norm_max = smoother_result.diagnostics.get(
                "amala_grad_norm_max",
                zero_scalar,
            )
            selected_label = random.categorical(label_key, final_label_log_probs).astype(jnp.int32)
            next_position = parameter_particles[selected_label]
            next_context = _select_pytree(contexts, selected_label)
            prior_terms = prior_terms_from_context_fn(next_context)
            next_traj_lp = jnp.asarray(
                trajectory_log_prob_fn(
                    next_context,
                    latent_path,
                    runtime_observations,
                    prior_terms=prior_terms,
                ),
                dtype=traj_dtype,
            )
            next_complete = jnp.asarray(log_prior_unc_fn(next_position), dtype=complete_dtype)
            next_complete = next_complete + next_traj_lp.astype(complete_dtype)

            if sign_flip_spec is not None:
                # Joint (latent coordinate, loading column) sign-flip MH move composed
                # after the smoother sweep: the flip is a state-independent involution,
                # so the acceptance is the exact joint posterior ratio — which prices
                # the sign-asymmetric loading prior and any drift coupling. This is
                # the escape route between the factor-sign mirror basins that the
                # alternating conditionals cannot cross on their own.
                n_flippable = int(sign_flip_spec.coords.shape[0])
                flip_idx = random.randint(flip_choice_key, (), 0, n_flippable)
                flip_coord = sign_flip_spec.coords[flip_idx]
                position_mask = sign_flip_spec.masks[flip_idx]
                flipped_position = jnp.where(position_mask, -next_position, next_position)
                latent_dim = int(latent_path.shape[1])
                coord_sign = jnp.where(
                    jnp.arange(latent_dim, dtype=jnp.int32) == flip_coord,
                    -jnp.ones((latent_dim,), dtype=latent_path.dtype),
                    jnp.ones((latent_dim,), dtype=latent_path.dtype),
                )
                flipped_latent = latent_path * coord_sign[None, :]
                flipped_context = latent_context_runtime_fn(flipped_position, runtime_times)
                flipped_complete, flipped_traj_lp = (
                    bundle.cached.complete_log_posterior_from_context_runtime_fn(
                        flipped_position,
                        flipped_context,
                        flipped_latent,
                        runtime_observations,
                    )
                )
                flip_delta = flipped_complete.astype(complete_dtype) - next_complete
                flip_accepted = (
                    jnp.log(random.uniform(flip_accept_key, dtype=flip_delta.dtype)) < flip_delta
                )
                next_position = jnp.where(flip_accepted, flipped_position, next_position)
                latent_path = jnp.where(flip_accepted, flipped_latent, latent_path)
                next_traj_lp = jnp.where(
                    flip_accepted,
                    jnp.asarray(flipped_traj_lp, dtype=traj_dtype),
                    next_traj_lp,
                )
                next_complete = jnp.where(
                    flip_accepted, flipped_complete.astype(complete_dtype), next_complete
                )
                next_context = jax.tree_util.tree_map(
                    lambda flipped, kept: jnp.where(flip_accepted, flipped, kept),
                    flipped_context,
                    next_context,
                )

            latent_move = latent_path - x_ref
            latent_move_rms_per_t = jnp.sqrt(jnp.mean(latent_move * latent_move, axis=-1))
            latent_move_rms = jnp.sqrt(jnp.mean(latent_move * latent_move))
            latent_move_max_abs = jnp.max(jnp.abs(latent_move))
            parameter_accepted = (selected_label != 0).astype(state.position.dtype)
            latent_updated = (origin_path != 0).astype(state.position.dtype)
            # Per-(t, d) exact-equality freeze indicator. A completely stuck chain has
            # zero autocovariance at every lag and therefore reports PERFECT ESS under
            # initial-positive-sequence estimators — this counter is the direct gauge
            # standard diagnostics structurally cannot provide. With coordinate-block
            # proposals it also resolves per-coordinate freezing that the per-t
            # `latent_accepted` trace cannot see.
            latent_frozen = (latent_path == x_ref).astype(state.position.dtype)
            latent_frozen_frac = jnp.mean(latent_frozen)
            latent_frozen_frac_by_d = jnp.mean(latent_frozen, axis=0)

            step_info = {
                "parameter_accepted": parameter_accepted,
                "latent_accepted": latent_updated,
                "latent_frozen_frac": latent_frozen_frac,
                "latent_frozen_frac_by_d": latent_frozen_frac_by_d,
                "selected_label": selected_label.astype(jnp.float32),
                "final_particle": origin_path[-1].astype(jnp.float32),
                "latent_move_rms": latent_move_rms,
                "latent_move_max_abs": latent_move_max_abs,
                "latent_move_rms_per_t": latent_move_rms_per_t,
                "final_label_log_probs": final_label_log_probs.astype(jnp.float32),
                "amala_grad_norm_mean": amala_grad_norm_mean.astype(jnp.float32),
                "amala_grad_norm_max": amala_grad_norm_max.astype(jnp.float32),
            }
            if sign_flip_spec is not None:
                step_info["sign_flip_accepted"] = flip_accepted.astype(state.position.dtype)
                step_info["sign_flip_delta"] = flip_delta.astype(jnp.float32)
            if diagnostic_flags.parameter_movement:
                parameter_jump = next_position - state.position
                step_info["parameter_jump_rms"] = jnp.sqrt(
                    jnp.mean(parameter_jump * parameter_jump)
                )
            if diagnostic_flags.particle_identity:
                reference_path_hit_rate = jnp.mean((origin_path == 0).astype(state.position.dtype))
                particle_ids = jnp.arange(num_particles, dtype=origin_path.dtype)
                selected_particle_unique_count = jnp.sum(
                    jnp.any(origin_path[:, None] == particle_ids[None, :], axis=0)
                ).astype(traj_dtype)
                step_info.update(
                    {
                        "selected_particle_per_t": origin_path.astype(jnp.float32),
                        "reference_path_hit_rate": reference_path_hit_rate.astype(jnp.float32),
                        "selected_particle_unique_count": (
                            selected_particle_unique_count.astype(jnp.float32)
                        ),
                    }
                )

        return (
            state._replace(
                position=next_position,
                latent_context=next_context,
                latent_trajectory=latent_path,
                trajectory_log_prob=next_traj_lp,
                complete_log_posterior=next_complete,
            ),
            step_info,
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
        latent_smoother=latent_smoother_spec,
        latent_delta=latent_delta,
        amala_delta_init=amala_delta_init,
        amala_delta_min=amala_delta_min,
        amala_delta_max=amala_delta_max,
        amala_target_accept=amala_target_accept,
        amala_adaptation_window=amala_adaptation_window,
        amala_adaptation_tolerance=amala_adaptation_tolerance,
        amala_adaptation_rho=amala_adaptation_rho,
        amala_adaptation_rho_min=amala_adaptation_rho_min,
        amala_adaptation_gamma=amala_adaptation_gamma,
        adapt_amala_delta=_uses_amala_delta(latent_smoother_spec),
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        dsmc_leaf_proposal=dsmc_leaf_proposal,
        latent_block_coords=latent_block_coords,
        diagnostic_metrics=resolved_diagnostic_metrics,
    )


def _initialize_chain_state(
    init_position: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    bundle: ParticleRuntimeBundle,
    initial_latent_delta: jnp.ndarray,
    param_step_size: float,
    param_min_scale: float,
    param_max_scale: float,
    param_target_accept: float,
    initial_latent_trajectory: jnp.ndarray | None,
) -> TrajectoryMCMCState:
    context = bundle.cached.latent_context_runtime_fn(init_position, times)
    predictive_latent = bundle.cached.initial_latent_from_context_fn(context)
    latent_trajectory = (
        predictive_latent
        if initial_latent_trajectory is None
        else jnp.asarray(initial_latent_trajectory, dtype=predictive_latent.dtype)
    )
    complete_lp, trajectory_lp = bundle.cached.complete_log_posterior_from_context_runtime_fn(
        init_position,
        context,
        latent_trajectory,
        observations,
    )
    latent_delta_value = jnp.asarray(initial_latent_delta, dtype=latent_trajectory.dtype)
    param_step_value = _clip_scale(
        jnp.asarray(param_step_size, dtype=latent_trajectory.dtype),
        min_scale=param_min_scale,
        max_scale=param_max_scale,
    )
    da_init, _da_update, _ = dual_averaging_adaptation(target=float(param_target_accept))
    return TrajectoryMCMCState(
        position=init_position,
        latent_context=context,
        latent_trajectory=latent_trajectory,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
        latent_delta=latent_delta_value,
        param_step_size=param_step_value,
        latent_da=da_init(jnp.mean(latent_delta_value)),
        param_da=da_init(param_step_value),
    )


@functools.partial(jax.jit, static_argnames=("step_fn",))
def _run_batched_step(
    states: TrajectoryMCMCState,
    step_keys: jnp.ndarray,
    *,
    step_fn,
) -> tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]:
    return jax.vmap(lambda state, key: step_fn(state, key))(states, step_keys)


@functools.partial(jax.jit, static_argnames=("public_latent_fn",))
def _sample_public_latent_batch(
    states: TrajectoryMCMCState,
    keys: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    public_latent_fn,
) -> jnp.ndarray:
    return jax.vmap(
        lambda state, key: public_latent_fn(
            state.latent_context,
            state.latent_trajectory,
            observations,
            key,
        )
    )(states, keys)


def run_marginal_particle_gibbs(
    bundle: ParticleRuntimeBundle,
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
    # Default "simple": m-PGibbs's parameter acceptance is a noisy near-binary
    # M=2 ensemble move-rate, and dual_averaging's sqrt(t) adjustments amplify
    # that into multi-order-of-magnitude per-chain step scatter (and outright
    # collapse for targets above the ~0.5 ensemble ceiling). The bounded
    # constant-rate scheme is robust to it. dual_averaging stays selectable and
    # is the right choice for aux_kalman_mcmc, whose acceptance is a smooth MALA
    # accept-probability.
    adaptation_scheme: str = "simple",
    profile_dir: str | None = None,
    profile_compile_analysis: bool = True,
    profile_runtime_trace: bool = True,
    profile_trace_start_step: int = 0,
    profile_trace_steps: int = 3,
) -> dict[str, Any]:
    """Run marginalized Particle Gibbs chains."""
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            f"Unknown adaptation_scheme {adaptation_scheme!r}; expected 'simple' or 'dual_averaging'."
        )
    if profile_trace_start_step < 0:
        raise ValueError("profile_trace_start_step must be non-negative.")
    if profile_trace_steps <= 0:
        raise ValueError("profile_trace_steps must be positive.")
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    da_param_update = (
        dual_averaging_adaptation(target=float(kernel.target_accept))[1]
        if use_dual_averaging
        else None
    )
    total_steps = num_warmup + num_samples
    if profile_runtime_trace and profile_trace_start_step >= total_steps:
        raise ValueError("profile_trace_start_step must be less than the total step count.")
    if total_steps <= 0:
        raise ValueError("marginal_particle_gibbs requires at least one MCMC step.")
    observations = bundle.observations
    times = bundle.times
    num_steps = int(observations.shape[0])
    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    dim = int(bundle.cached.flat_example.shape[0])
    if init_positions is None:
        init_keys = random.split(init_key, num_chains)
        init_noise = jax.vmap(
            lambda key: random.normal(
                key,
                bundle.cached.flat_example.shape,
                dtype=bundle.cached.flat_example.dtype,
            )
        )(init_keys)
        chain_init_positions = bundle.cached.flat_example[None, :] + init_scale * init_noise
    else:
        chain_init_positions = jnp.asarray(init_positions, dtype=bundle.cached.flat_example.dtype)
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

    initial_latent_delta_value = (
        kernel.amala_delta_init if kernel.adapt_amala_delta else latent_delta
    )
    initial_latent_delta = jnp.full(
        (num_steps,),
        jnp.asarray(initial_latent_delta_value, dtype=observations.dtype),
    )
    states = _stack_chain_states(
        [
            _initialize_chain_state(
                chain_init_positions[chain_idx],
                observations=observations,
                times=times,
                bundle=bundle,
                initial_latent_delta=initial_latent_delta,
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
    nonfinite_chains = [
        chain_idx
        for chain_idx in range(num_chains)
        if not bool(jnp.isfinite(states.complete_log_posterior[chain_idx]))
    ]
    if nonfinite_chains:
        raise ValueError(
            "Initial complete log-posterior is non-finite for chain(s) "
            f"{nonfinite_chains}. Every move would be rejected against a non-finite "
            "reference, so the sampler cannot recover from this state. The predictive "
            "latent init likely diverged: nonlinear vector fields can be explosive "
            "under unconditional forward simulation at data-informed initial "
            "parameters even when their posterior density is finite. Supply "
            "different init_positions or a data-conditioned "
            "initial_latent_trajectories."
        )
    initial_param_step_size = states.param_step_size
    initial_latent_delta = states.latent_delta
    latent_acceptance_window = jnp.zeros(
        (
            num_chains,
            int(kernel.amala_adaptation_window),
            num_steps,
        ),
        dtype=states.latent_delta.dtype,
    )
    latent_acceptance_window_count = 0
    step_keys = random.split(chain_key, total_steps * num_chains * 2).reshape(
        total_steps,
        num_chains,
        2,
        2,
    )
    need_public_latent = compute_latent_posterior_summary or retain_latent_paths
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        diagnostic_metrics=kernel.diagnostic_metrics,
    )
    public_latent_fn = bundle.cached.public_latent_trajectory_runtime_fn
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
    selected_particle_per_t_history: list[jnp.ndarray] = []
    reference_path_hit_rate_history: list[jnp.ndarray] = []
    selected_particle_unique_count_history: list[jnp.ndarray] = []
    latent_move_rms_history: list[jnp.ndarray] = []
    latent_move_max_abs_history: list[jnp.ndarray] = []
    latent_move_rms_per_t_history: list[jnp.ndarray] = []
    latent_frozen_frac_history: list[jnp.ndarray] = []
    latent_frozen_frac_by_d_history: list[jnp.ndarray] = []
    sign_flip_accept_history: list[jnp.ndarray] = []
    parameter_jump_rms_history: list[jnp.ndarray] = []
    final_label_log_probs_history: list[jnp.ndarray] = []
    amala_grad_norm_mean_history: list[jnp.ndarray] = []
    amala_grad_norm_max_history: list[jnp.ndarray] = []

    progress_started = time.monotonic()
    progress_every = max(1, min(250, total_steps // 20))
    print(
        "marginal_particle_gibbs progress: "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"total_steps={total_steps} n_particles={kernel.num_particles} "
        f"n_parameter_particles={kernel.num_parameter_particles} "
        f"latent_smoother={kernel.latent_smoother.name} "
        f"dsmc_leaf_proposal={kernel.dsmc_leaf_proposal} "
        f"latent_block_coords={kernel.latent_block_coords} progress_every={progress_every}",
        flush=True,
    )

    resolved_profile_dir = _profiling.resolve_profile_dir(profile_dir)
    if profile_compile_analysis:
        _profiling.dump_compiled_analysis(
            _run_batched_step,
            states,
            step_keys[0, :, 0, :],
            step_fn=kernel.step_fn,
            profile_dir=resolved_profile_dir,
            label="run_batched_step",
        )

    sampling_loop_started = time.monotonic()
    first_step_seconds: float | None = None
    trace_active = False
    trace_stop_step = profile_trace_start_step + profile_trace_steps
    try:
        for step_idx in range(total_steps):
            step_started = time.monotonic()
            if profile_runtime_trace and step_idx == profile_trace_start_step:
                _profiling.start_trace(resolved_profile_dir, label="run_loop")
                trace_active = resolved_profile_dir is not None
            if step_idx == 0:
                print(
                    "marginal_particle_gibbs progress: first step compile/run start",
                    flush=True,
                )
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
                latent_delta_now = jax.device_get(states.latent_delta)
                complete_lp_now = jax.device_get(states.complete_log_posterior)
                phase = "warmup" if step_idx < num_warmup else "sample"
                elapsed = time.monotonic() - progress_started
                latent_delta_status = (
                    f"amala_delta_range=[{float(jnp.min(latent_delta_now)):.3g},"
                    f"{float(jnp.max(latent_delta_now)):.3g}] "
                    if kernel.adapt_amala_delta
                    else ""
                )
                print(
                    "marginal_particle_gibbs progress: "
                    f"step={step_idx + 1}/{total_steps} phase={phase} elapsed={elapsed:.1f}s "
                    f"parameter_accept_now={float(param_accept_now):.3f} "
                    f"latent_update_now={float(latent_accept_now):.3f} "
                    f"param_step_range=[{float(jnp.min(param_step_now)):.3g},"
                    f"{float(jnp.max(param_step_now)):.3g}] "
                    f"{latent_delta_status}"
                    f"complete_lp_range=[{float(jnp.min(complete_lp_now)):.3g},"
                    f"{float(jnp.max(complete_lp_now)):.3g}]",
                    flush=True,
                )

            if step_idx == 0:
                states.complete_log_posterior.block_until_ready()
                first_step_seconds = time.monotonic() - step_started
                print(
                    "marginal_particle_gibbs progress: "
                    f"first step compile/run complete elapsed={first_step_seconds:.1f}s",
                    flush=True,
                )

            position_history.append(states.position)
            parameter_accept_history.append(step_info["parameter_accepted"])
            latent_accept_history.append(jnp.mean(step_info["latent_accepted"], axis=-1))
            complete_lp_history.append(states.complete_log_posterior)
            selected_label_history.append(step_info["selected_label"])
            final_particle_history.append(step_info["final_particle"])
            latent_move_rms_history.append(step_info["latent_move_rms"])
            latent_move_max_abs_history.append(step_info["latent_move_max_abs"])
            latent_move_rms_per_t_history.append(step_info["latent_move_rms_per_t"])
            latent_frozen_frac_history.append(step_info["latent_frozen_frac"])
            latent_frozen_frac_by_d_history.append(step_info["latent_frozen_frac_by_d"])
            if "sign_flip_accepted" in step_info:
                sign_flip_accept_history.append(step_info["sign_flip_accepted"])
            final_label_log_probs_history.append(step_info["final_label_log_probs"])
            amala_grad_norm_mean_history.append(step_info["amala_grad_norm_mean"])
            amala_grad_norm_max_history.append(step_info["amala_grad_norm_max"])
            if diagnostic_flags.particle_identity:
                selected_particle_per_t_history.append(step_info["selected_particle_per_t"])
                reference_path_hit_rate_history.append(step_info["reference_path_hit_rate"])
                selected_particle_unique_count_history.append(
                    step_info["selected_particle_unique_count"]
                )
            if diagnostic_flags.parameter_movement:
                parameter_jump_rms_history.append(step_info["parameter_jump_rms"])

            if need_public_latent:
                public_latent = _sample_public_latent_batch(
                    states,
                    step_keys[step_idx, :, 1, :],
                    observations,
                    public_latent_fn=public_latent_fn,
                )
                if step_idx >= num_warmup and compute_latent_posterior_summary:
                    latent_sum = latent_sum + public_latent
                    latent_sumsq = latent_sumsq + public_latent * public_latent
                    sample_count = sample_count + 1
                if retain_latent_paths:
                    latent_paths_history.append(public_latent)

            if trace_active and step_idx + 1 >= trace_stop_step:
                states.complete_log_posterior.block_until_ready()
                _profiling.stop_trace(resolved_profile_dir)
                trace_active = False

            if step_idx < num_warmup:
                if kernel.adapt_amala_delta:
                    window_slot = step_idx % int(kernel.amala_adaptation_window)
                    latent_acceptance_window = latent_acceptance_window.at[:, window_slot, :].set(
                        step_info["latent_accepted"].astype(latent_acceptance_window.dtype)
                    )
                    latent_acceptance_window_count = min(
                        latent_acceptance_window_count + 1,
                        int(kernel.amala_adaptation_window),
                    )
                    latent_acceptance_rate = jnp.sum(
                        latent_acceptance_window, axis=1
                    ) / jnp.asarray(
                        latent_acceptance_window_count,
                        dtype=latent_acceptance_window.dtype,
                    )
                    target_accept = jnp.asarray(
                        kernel.amala_target_accept,
                        dtype=states.latent_delta.dtype,
                    )
                    learning_rate = jnp.maximum(
                        jnp.asarray(step_idx + 1, dtype=states.latent_delta.dtype)
                        ** jnp.asarray(
                            kernel.amala_adaptation_gamma,
                            dtype=states.latent_delta.dtype,
                        )
                        * jnp.asarray(kernel.amala_adaptation_rho, dtype=states.latent_delta.dtype),
                        jnp.asarray(
                            kernel.amala_adaptation_rho_min,
                            dtype=states.latent_delta.dtype,
                        ),
                    )
                    delta_update = (
                        learning_rate
                        * states.latent_delta
                        * (latent_acceptance_rate - target_accept)
                        / target_accept
                    )
                    should_adapt_latent_delta = jnp.abs(
                        latent_acceptance_rate - target_accept
                    ) >= jnp.asarray(
                        kernel.amala_adaptation_tolerance,
                        dtype=states.latent_delta.dtype,
                    )
                    should_adapt_latent_delta = should_adapt_latent_delta & (
                        (step_idx + 1) > int(kernel.amala_adaptation_window)
                    )
                    next_latent_delta = jnp.where(
                        should_adapt_latent_delta,
                        states.latent_delta + delta_update,
                        states.latent_delta,
                    )
                    states = states._replace(
                        latent_delta=_clip_scale(
                            next_latent_delta,
                            min_scale=kernel.amala_delta_min,
                            max_scale=kernel.amala_delta_max,
                        )
                    )
                if da_param_update is not None:
                    # Dual averaging converges (unlike the constant-rate scheme), and we
                    # freeze to the Polyak-averaged step at the final warmup step rather
                    # than keeping a noisy live value — so per-chain steps no longer
                    # scatter across orders of magnitude.
                    updated_param_da = jax.vmap(
                        lambda da_state, accepted: _clip_dual_averaging_state(
                            da_param_update(da_state, accepted),
                            min_scale=kernel.min_scale,
                            max_scale=kernel.max_scale,
                        )
                    )(states.param_da, step_info["parameter_accepted"])
                    scale_dtype = states.param_step_size.dtype
                    if step_idx == num_warmup - 1:
                        next_param_step = jnp.exp(updated_param_da.log_step_size_avg)
                    else:
                        next_param_step = jnp.exp(updated_param_da.log_step_size)
                    states = states._replace(
                        param_step_size=_clip_scale(
                            next_param_step.astype(scale_dtype),
                            min_scale=kernel.min_scale,
                            max_scale=kernel.max_scale,
                        ),
                        param_da=updated_param_da,
                    )
                else:
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
    finally:
        if trace_active:
            states.complete_log_posterior.block_until_ready()
            _profiling.stop_trace(resolved_profile_dir)

    states.complete_log_posterior.block_until_ready()
    sampling_loop_seconds = time.monotonic() - sampling_loop_started

    all_grouped_positions = _stack_sample_history(
        position_history,
        num_chains=num_chains,
        trailing_shape=(dim,),
        dtype=chain_init_positions.dtype,
    )
    grouped_positions = all_grouped_positions[:, num_warmup:]
    all_chain_extra_fields = {
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
        "latent_frozen_frac": _stack_sample_history(
            latent_frozen_frac_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "latent_frozen_frac_by_d": _stack_sample_history(
            latent_frozen_frac_by_d_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(latent_frozen_frac_by_d_history[0].shape[1:])
                if latent_frozen_frac_by_d_history
                else (int(states.latent_trajectory.shape[1]),)
            ),
            dtype=states.latent_trajectory.dtype,
        ),
        "final_label_log_probs": _stack_sample_history(
            final_label_log_probs_history,
            num_chains=num_chains,
            trailing_shape=(
                tuple(final_label_log_probs_history[0].shape[1:])
                if final_label_log_probs_history
                else (int(kernel.num_parameter_particles),)
            ),
            dtype=chain_init_positions.dtype,
        ),
        "amala_grad_norm_mean": _stack_sample_history(
            amala_grad_norm_mean_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "amala_grad_norm_max": _stack_sample_history(
            amala_grad_norm_max_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
    }
    if sign_flip_accept_history:
        all_chain_extra_fields["sign_flip_accept_prob"] = _stack_sample_history(
            sign_flip_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        )
    if diagnostic_flags.particle_identity:
        all_chain_extra_fields.update(
            {
                "selected_particle_per_t": _stack_sample_history(
                    selected_particle_per_t_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(selected_particle_per_t_history[0].shape[1:]),
                    dtype=states.latent_trajectory.dtype,
                ),
                "reference_path_hit_rate": _stack_sample_history(
                    reference_path_hit_rate_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "selected_particle_unique_count": _stack_sample_history(
                    selected_particle_unique_count_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
    if diagnostic_flags.parameter_movement:
        all_chain_extra_fields["parameter_jump_rms"] = _stack_sample_history(
            parameter_jump_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        )
    chain_extra_fields = {
        name: values[:, num_warmup:] for name, values in all_chain_extra_fields.items()
    }
    warmup_chain_extra_fields = {
        name: values[:, :num_warmup] for name, values in all_chain_extra_fields.items()
    }
    all_complete_log_posterior_history = _stack_sample_history(
        complete_lp_history,
        num_chains=num_chains,
        trailing_shape=(),
        dtype=states.complete_log_posterior.dtype,
    )
    complete_log_posterior_history = all_complete_log_posterior_history[:, num_warmup:]
    warmup_complete_log_posterior_history = all_complete_log_posterior_history[:, :num_warmup]

    latent_summary = None
    if compute_latent_posterior_summary:
        denom = jnp.maximum(sample_count, 1).astype(latent_sum.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        latent_summary = _latent_summary_from_chain_moments(chain_mean, jnp.sqrt(chain_var))

    latent_paths = None
    all_latent_paths = None
    warmup_latent_paths = None
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
        all_latent_paths = _stack_sample_history(
            latent_paths_history,
            num_chains=num_chains,
            trailing_shape=latent_trailing_shape,
            dtype=states.latent_trajectory.dtype,
        )
        latent_paths = all_latent_paths[:, num_warmup:]
        warmup_latent_paths = all_latent_paths[:, :num_warmup]

    post_warmup_complete_log_posterior_mean = (
        jnp.mean(complete_log_posterior_history, axis=1)
        if num_samples > 0
        else jnp.full((num_chains,), jnp.nan, dtype=states.complete_log_posterior.dtype)
    )

    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "warmup_chain_extra_fields": warmup_chain_extra_fields,
        "all_chain_extra_fields": all_chain_extra_fields,
        "complete_log_posterior_history": complete_log_posterior_history,
        "warmup_complete_log_posterior_history": warmup_complete_log_posterior_history,
        "all_complete_log_posterior_history": all_complete_log_posterior_history,
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
        "warmup_latent_paths": warmup_latent_paths,
        "all_latent_paths": all_latent_paths,
        "initial_param_step_size": initial_param_step_size,
        "final_param_step_size": states.param_step_size,
        "initial_latent_delta": initial_latent_delta,
        "final_latent_delta": states.latent_delta,
        "first_step_seconds": 0.0 if first_step_seconds is None else first_step_seconds,
        "sampling_loop_seconds": sampling_loop_seconds,
        "post_warmup_complete_log_posterior_mean": post_warmup_complete_log_posterior_mean,
    }
