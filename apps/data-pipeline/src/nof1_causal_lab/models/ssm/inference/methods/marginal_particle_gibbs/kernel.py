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
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCResult,
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
    _LATENT_SMOOTHER_MGRAD,
    _LATENT_SMOOTHER_PLAIN,
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
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.smoothers._mgrad_kernel import (
    build_pit_particle_mgrad_latent_kernel,
)

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
# Target acceptance for the M=2 ensemble Barker selection. Its move-rate is
# bounded above by ~0.5 -- the step->0 coin-flip limit, (M-1)/M for M ensemble
# candidates -- so MALA-style optima (~0.574) do NOT transfer: a target above
# that ceiling makes dual averaging chase an unreachable rate and collapse the
# step size to the floor (observed empirically at 0.57). 0.35 sits safely below
# the ceiling, is reachable under dual averaging, and matches the long-standing
# baseline. The ceiling is a property of the M=2 selection, not of the proposal
# drift, so this default is shared by random_walk and pseudo_langevin.
_DEFAULT_PARAM_TARGET_ACCEPT = 0.35


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
    latent_smoother: MPGibbsLatentSmoother
    latent_delta: float
    amala_q_scale: float
    amala_kappa: float
    amala_grad_clip: float
    diagnostic_metrics: frozenset[str]


def build_marginal_particle_gibbs_kernel(
    bundle: dict[str, Any],
    *,
    num_particles: int,
    num_parameter_particles: int,
    param_step_size: float,
    target_accept: float | None = None,
    min_scale: float = _DEFAULT_MIN_SCALE,
    max_scale: float = _DEFAULT_MAX_SCALE,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    latent_block_size: int = 256,
    parameter_proposal: str = "pseudo_langevin",
    latent_smoother: str = _LATENT_SMOOTHER_PLAIN,
    latent_delta: float = 0.2,
    amala_q_scale: float = 1.0,
    amala_kappa: float = 0.5,
    amala_grad_clip: float = 1000.0,
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
        latent_smoother=latent_smoother_spec.name,
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
    if latent_block_size < 1:
        raise ValueError(
            f"marginal_particle_gibbs latent_block_size must be positive; got {latent_block_size}."
        )
    if parameter_proposal not in ("random_walk", "pseudo_langevin"):
        raise ValueError(
            "marginal_particle_gibbs parameter_proposal must be 'random_walk' or "
            f"'pseudo_langevin'; got {parameter_proposal!r}."
        )
    if latent_smoother_spec.name == _LATENT_SMOOTHER_MGRAD and latent_delta <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs latent_delta must be positive for mgrad; got {latent_delta}."
        )
    if amala_q_scale <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_q_scale must be positive; got {amala_q_scale}."
        )
    if amala_kappa < 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_kappa must be non-negative; got {amala_kappa}."
        )
    if amala_grad_clip <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_grad_clip must be positive; got {amala_grad_clip}."
        )
    use_gradient_drift = parameter_proposal == "pseudo_langevin"
    if target_accept is None:
        target_accept = _DEFAULT_PARAM_TARGET_ACCEPT

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
    complete_log_posterior_runtime_fn = bundle["complete_log_posterior_runtime_fn"]
    mgrad_latent_kernel = (
        build_pit_particle_mgrad_latent_kernel(
            bundle,
            delta=latent_delta,
            target_accept=0.75,
            num_particles=num_particles,
            latent_kernel_algorithm="particle_mgrad",
        )
        if latent_smoother_spec.name == _LATENT_SMOOTHER_MGRAD
        else None
    )

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
            dtype=bundle["flat_example"].dtype,
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
        initial_observation_auxiliary_fn=initial_observation_auxiliary_fn,
        runtime_observations=runtime_observations,
        runtime_times=runtime_times,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        latent_block_size=latent_block_size,
        latent_delta=latent_delta,
        amala_q_scale=amala_q_scale,
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        mgrad_latent_kernel=mgrad_latent_kernel,
        diagnostic_metrics=resolved_diagnostic_metrics,
    )

    def _step_fn(state: TrajectoryMCMCState, key: jnp.ndarray):
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

            step_info = {
                "parameter_accepted": parameter_accepted,
                "latent_accepted": latent_updated,
                "selected_label": selected_label.astype(jnp.float32),
                "final_particle": origin_path[-1].astype(jnp.float32),
                "latent_move_rms": latent_move_rms,
                "latent_move_max_abs": latent_move_max_abs,
                "latent_move_rms_per_t": latent_move_rms_per_t,
                "final_label_log_probs": final_label_log_probs.astype(jnp.float32),
                "amala_grad_norm_mean": amala_grad_norm_mean.astype(jnp.float32),
                "amala_grad_norm_max": amala_grad_norm_max.astype(jnp.float32),
            }
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
            if diagnostic_flags.particle_filter:
                step_info.update(
                    {
                        "forward_particle_ess_by_t": smoother_result.diagnostics[
                            "forward_particle_ess_by_t"
                        ].astype(jnp.float32),
                        "forward_log_weight_range_by_t": smoother_result.diagnostics[
                            "forward_log_weight_range_by_t"
                        ].astype(jnp.float32),
                        "forward_log_weight_variance_by_t": smoother_result.diagnostics[
                            "forward_log_weight_variance_by_t"
                        ].astype(jnp.float32),
                    }
                )
            if diagnostic_flags.backward_selection:
                step_info.update(
                    {
                        "backward_selection_ess_by_t": smoother_result.diagnostics[
                            "backward_selection_ess_by_t"
                        ].astype(jnp.float32),
                        "backward_selection_entropy_by_t": smoother_result.diagnostics[
                            "backward_selection_entropy_by_t"
                        ].astype(jnp.float32),
                        "backward_selection_max_prob_by_t": smoother_result.diagnostics[
                            "backward_selection_max_prob_by_t"
                        ].astype(jnp.float32),
                    }
                )
            if diagnostic_flags.amala_proposal:
                step_info.update(
                    {
                        "amala_grad_clip_fraction": smoother_result.diagnostics[
                            "amala_grad_clip_fraction"
                        ].astype(jnp.float32),
                        "amala_drift_norm_mean": smoother_result.diagnostics[
                            "amala_drift_norm_mean"
                        ].astype(jnp.float32),
                        "amala_drift_norm_max": smoother_result.diagnostics[
                            "amala_drift_norm_max"
                        ].astype(jnp.float32),
                        "amala_auxiliary_noise_norm_mean": smoother_result.diagnostics[
                            "amala_auxiliary_noise_norm_mean"
                        ].astype(jnp.float32),
                        "amala_auxiliary_noise_norm_max": smoother_result.diagnostics[
                            "amala_auxiliary_noise_norm_max"
                        ].astype(jnp.float32),
                        "amala_drift_to_auxiliary_noise_ratio_mean": (
                            smoother_result.diagnostics[
                                "amala_drift_to_auxiliary_noise_ratio_mean"
                            ].astype(jnp.float32)
                        ),
                        "amala_proposal_displacement_norm_mean": (
                            smoother_result.diagnostics[
                                "amala_proposal_displacement_norm_mean"
                            ].astype(jnp.float32)
                        ),
                        "amala_proposal_displacement_norm_max": (
                            smoother_result.diagnostics[
                                "amala_proposal_displacement_norm_max"
                            ].astype(jnp.float32)
                        ),
                        "amala_auxiliary_correction_variance": smoother_result.diagnostics[
                            "amala_auxiliary_correction_variance"
                        ].astype(jnp.float32),
                        "amala_auxiliary_correction_max_abs": smoother_result.diagnostics[
                            "amala_auxiliary_correction_max_abs"
                        ].astype(jnp.float32),
                    }
                )

        return (
            state._replace(
                position=next_position,
                latent_context=next_context,
                latent_trajectory=latent_path,
                observation_auxiliary=next_observation_auxiliary,
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
        latent_block_size=latent_block_size,
        latent_smoother=latent_smoother_spec,
        latent_delta=latent_delta,
        amala_q_scale=amala_q_scale,
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        diagnostic_metrics=resolved_diagnostic_metrics,
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
) -> TrajectoryMCMCState:
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
    return TrajectoryMCMCState(
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
    # Default "simple": m-PGibbs's parameter acceptance is a noisy near-binary
    # M=2 ensemble move-rate, and dual_averaging's sqrt(t) adjustments amplify
    # that into multi-order-of-magnitude per-chain step scatter (and outright
    # collapse for targets above the ~0.5 ensemble ceiling). The bounded
    # constant-rate scheme is robust to it. dual_averaging stays selectable and
    # is the right choice for aux_kalman_mcmc, whose acceptance is a smooth MALA
    # accept-probability.
    adaptation_scheme: str = "simple",
) -> dict[str, Any]:
    """Run marginalized Particle Gibbs chains."""
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            f"Unknown adaptation_scheme {adaptation_scheme!r}; expected 'simple' or 'dual_averaging'."
        )
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    da_param_update = (
        dual_averaging_adaptation(target=float(kernel.target_accept))[1]
        if use_dual_averaging
        else None
    )
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
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        latent_smoother=kernel.latent_smoother.name,
        diagnostic_metrics=kernel.diagnostic_metrics,
    )
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
    selected_particle_per_t_history: list[jnp.ndarray] = []
    reference_path_hit_rate_history: list[jnp.ndarray] = []
    selected_particle_unique_count_history: list[jnp.ndarray] = []
    latent_move_rms_history: list[jnp.ndarray] = []
    latent_move_max_abs_history: list[jnp.ndarray] = []
    latent_move_rms_per_t_history: list[jnp.ndarray] = []
    parameter_jump_rms_history: list[jnp.ndarray] = []
    final_label_log_probs_history: list[jnp.ndarray] = []
    forward_particle_ess_history: list[jnp.ndarray] = []
    forward_log_weight_range_history: list[jnp.ndarray] = []
    forward_log_weight_variance_history: list[jnp.ndarray] = []
    backward_selection_ess_history: list[jnp.ndarray] = []
    backward_selection_entropy_history: list[jnp.ndarray] = []
    backward_selection_max_prob_history: list[jnp.ndarray] = []
    amala_grad_norm_mean_history: list[jnp.ndarray] = []
    amala_grad_norm_max_history: list[jnp.ndarray] = []
    amala_grad_clip_fraction_history: list[jnp.ndarray] = []
    amala_drift_norm_mean_history: list[jnp.ndarray] = []
    amala_drift_norm_max_history: list[jnp.ndarray] = []
    amala_auxiliary_noise_norm_mean_history: list[jnp.ndarray] = []
    amala_auxiliary_noise_norm_max_history: list[jnp.ndarray] = []
    amala_drift_to_auxiliary_noise_ratio_mean_history: list[jnp.ndarray] = []
    amala_proposal_displacement_norm_mean_history: list[jnp.ndarray] = []
    amala_proposal_displacement_norm_max_history: list[jnp.ndarray] = []
    amala_auxiliary_correction_variance_history: list[jnp.ndarray] = []
    amala_auxiliary_correction_max_abs_history: list[jnp.ndarray] = []

    progress_started = time.monotonic()
    progress_every = max(1, min(250, total_steps // 20))
    print(
        "marginal_particle_gibbs progress: "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"total_steps={total_steps} n_particles={kernel.num_particles} "
        f"n_parameter_particles={kernel.num_parameter_particles} "
        f"latent_smoother={kernel.latent_smoother.name} "
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
            if use_dual_averaging:
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

        position_history.append(states.position)
        parameter_accept_history.append(step_info["parameter_accepted"])
        latent_accept_history.append(jnp.mean(step_info["latent_accepted"], axis=-1))
        complete_lp_history.append(states.complete_log_posterior)
        selected_label_history.append(step_info["selected_label"])
        final_particle_history.append(step_info["final_particle"])
        latent_move_rms_history.append(step_info["latent_move_rms"])
        latent_move_max_abs_history.append(step_info["latent_move_max_abs"])
        latent_move_rms_per_t_history.append(step_info["latent_move_rms_per_t"])
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
        if diagnostic_flags.particle_filter:
            forward_particle_ess_history.append(step_info["forward_particle_ess_by_t"])
            forward_log_weight_range_history.append(step_info["forward_log_weight_range_by_t"])
            forward_log_weight_variance_history.append(
                step_info["forward_log_weight_variance_by_t"]
            )
        if diagnostic_flags.backward_selection:
            backward_selection_ess_history.append(step_info["backward_selection_ess_by_t"])
            backward_selection_entropy_history.append(step_info["backward_selection_entropy_by_t"])
            backward_selection_max_prob_history.append(
                step_info["backward_selection_max_prob_by_t"]
            )
        if diagnostic_flags.amala_proposal:
            amala_grad_clip_fraction_history.append(step_info["amala_grad_clip_fraction"])
            amala_drift_norm_mean_history.append(step_info["amala_drift_norm_mean"])
            amala_drift_norm_max_history.append(step_info["amala_drift_norm_max"])
            amala_auxiliary_noise_norm_mean_history.append(
                step_info["amala_auxiliary_noise_norm_mean"]
            )
            amala_auxiliary_noise_norm_max_history.append(
                step_info["amala_auxiliary_noise_norm_max"]
            )
            amala_drift_to_auxiliary_noise_ratio_mean_history.append(
                step_info["amala_drift_to_auxiliary_noise_ratio_mean"]
            )
            amala_proposal_displacement_norm_mean_history.append(
                step_info["amala_proposal_displacement_norm_mean"]
            )
            amala_proposal_displacement_norm_max_history.append(
                step_info["amala_proposal_displacement_norm_max"]
            )
            amala_auxiliary_correction_variance_history.append(
                step_info["amala_auxiliary_correction_variance"]
            )
            amala_auxiliary_correction_max_abs_history.append(
                step_info["amala_auxiliary_correction_max_abs"]
            )

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
    if diagnostic_flags.particle_identity:
        chain_extra_fields.update(
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
        chain_extra_fields["parameter_jump_rms"] = _stack_sample_history(
            parameter_jump_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        )
    if diagnostic_flags.particle_filter:
        chain_extra_fields.update(
            {
                "forward_particle_ess_by_t": _stack_sample_history(
                    forward_particle_ess_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(forward_particle_ess_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
                "forward_log_weight_range_by_t": _stack_sample_history(
                    forward_log_weight_range_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(forward_log_weight_range_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
                "forward_log_weight_variance_by_t": _stack_sample_history(
                    forward_log_weight_variance_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(forward_log_weight_variance_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
    if diagnostic_flags.backward_selection:
        chain_extra_fields.update(
            {
                "backward_selection_ess_by_t": _stack_sample_history(
                    backward_selection_ess_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(backward_selection_ess_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
                "backward_selection_entropy_by_t": _stack_sample_history(
                    backward_selection_entropy_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(backward_selection_entropy_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
                "backward_selection_max_prob_by_t": _stack_sample_history(
                    backward_selection_max_prob_history,
                    num_chains=num_chains,
                    trailing_shape=tuple(backward_selection_max_prob_history[0].shape[1:]),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
    if diagnostic_flags.amala_proposal:
        chain_extra_fields.update(
            {
                "amala_grad_clip_fraction": _stack_sample_history(
                    amala_grad_clip_fraction_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_drift_norm_mean": _stack_sample_history(
                    amala_drift_norm_mean_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_drift_norm_max": _stack_sample_history(
                    amala_drift_norm_max_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_auxiliary_noise_norm_mean": _stack_sample_history(
                    amala_auxiliary_noise_norm_mean_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_auxiliary_noise_norm_max": _stack_sample_history(
                    amala_auxiliary_noise_norm_max_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_drift_to_auxiliary_noise_ratio_mean": _stack_sample_history(
                    amala_drift_to_auxiliary_noise_ratio_mean_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_proposal_displacement_norm_mean": _stack_sample_history(
                    amala_proposal_displacement_norm_mean_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_proposal_displacement_norm_max": _stack_sample_history(
                    amala_proposal_displacement_norm_max_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_auxiliary_correction_variance": _stack_sample_history(
                    amala_auxiliary_correction_variance_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
                "amala_auxiliary_correction_max_abs": _stack_sample_history(
                    amala_auxiliary_correction_max_abs_history,
                    num_chains=num_chains,
                    trailing_shape=(),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
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
) -> TrajectoryMCMCResult:
    return TrajectoryMCMCResult(
        chain_samples=chain_samples,
        chain_extra_fields=chain_extra_fields,
        num_chains=num_chains,
        num_samples=num_samples,
        backend="marginal_particle_gibbs",
    )
