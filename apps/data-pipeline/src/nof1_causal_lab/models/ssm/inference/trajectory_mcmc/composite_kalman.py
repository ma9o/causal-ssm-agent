"""Composite-vector-field bridge to the Corenflos auxiliary Kalman MH.

Two pieces here:

1. ``composite_latent_context_at_trajectory`` — builds the
   ``LatentContext`` shape (per-step ``Ad, Qd, cd``) by linearising the
   ``CompositeVectorField`` at each transition's source state along the
   current trajectory. For a pure ``DenseLinear`` field this matches
   the existing dense-path discretization exactly.

2. ``composite_latent_mh_step_eq10_11`` — the eq10_11 latent MH step
   from Corenflos & Särkkä (2025) §2.3 / Algorithm 2 generalised to
   non-linear dynamics. Builds two contexts per step: the forward
   proposal linearises at ``x_curr``; the reverse proposal at
   ``x_prop``. The acceptance ratio combines log-priors and proposal
   densities computed under the *appropriate* linearization on each
   side, so detailed balance holds w.r.t. the EKF-approximated target
   posterior.

The two-context structure is the key generalisation over the existing
``_latent_mh_step_eq10_11_runtime`` in ``auxiliary_kalman.py`` (which
caches a single context computed at θ-sample time, valid only when the
linearization is trajectory-independent — i.e., linear dynamics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VectorFieldArgs,
)
from nof1_causal_lab.models.ssm.inference.parallel_kalman import (
    aux_filter_lgssm_lightweight,
    sample_lgssm_trajectory,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    build_gaussian_trajectory_prior_terms,
    trajectory_prior_log_prob_from_terms,
)

from .auxiliary_kalman import (
    AUX_JITTER,
    LatentContext,
    gaussian_log_prob_isotropic,
    tame_gradient_tulac,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics import CompositeVectorField


def composite_latent_context_at_trajectory(
    *,
    vector_field: CompositeVectorField,
    vf_params: tuple[dict[str, Array], ...],
    x_traj: Array,
    init_mean: Array,
    init_cov: Array,
    diffusion_cov: Array,
    runtime_times: Array,
    H: Array,
    d_meas: Array,
    R: Array,
    transition_inputs: Array | None = None,
    input_effect: Array | None = None,
    extra_params: dict[str, Array] | None = None,
    H_rows: Array | None = None,
    d_rows: Array | None = None,
) -> LatentContext:
    """Build a per-step-linearized ``LatentContext``.

    For each transition ``t``, the linearization point is the trajectory
    value at ``t-1`` — i.e., the *source* state of the transition. The
    transition at ``t=0`` (from the initial distribution to the first
    state) uses the initial mean as the linearization point.

    The shape of the returned context matches the existing
    ``LatentContext`` from ``auxiliary_kalman.py`` so the downstream
    auxiliary LGSSM filter, sampler, and (eventually) the eq10_11 MH
    step can consume it without modification.

    Args:
        vector_field: The composite drift.
        vf_params: Per-component parameter tuple matching ``vector_field.components``.
        x_traj: ``(T, n_latent)`` current trajectory snapshot. Linearization
            for transition ``t`` happens at ``x_traj[t-1]``; for ``t=0`` at
            ``init_mean``.
        init_mean: ``(n_latent,)`` initial distribution mean.
        init_cov: ``(n_latent, n_latent)`` initial covariance.
        diffusion_cov: ``(n_latent, n_latent)`` SDE diffusion ``G·G'``.
        runtime_times: ``(T,)`` observation times. The synthetic first
            interval is set to ``MIN_DT`` to match the existing dense path
            convention in ``auxiliary_kalman.py``.
        H, d_meas, R: Observation model matrices, passed through unchanged.
        extra_params, H_rows, d_rows: Optional fields carried through for
            specialized observation paths.

    Returns:
        ``LatentContext`` with per-step ``(Ad, Qd, cd)`` reflecting the
        local linearization at ``x_traj``.
    """
    args = VectorFieldArgs(params=vf_params, intervention=Intervention.none())

    time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)

    # Each transition t maps the state at time t-1 to time t.
    # Linearization point for transition t is x_{t-1}.
    # For t=0 the "previous state" is the initial mean.
    n_transitions = time_intervals.shape[0]
    if x_traj.shape[0] >= n_transitions:
        x_lin_tail = x_traj[: n_transitions - 1]
    else:
        # Defensive: pad with the last available value if trajectory is too short
        x_lin_tail = jnp.broadcast_to(x_traj[-1], (n_transitions - 1, x_traj.shape[-1]))
    x_lin_batch = jnp.concatenate([init_mean[None, :], x_lin_tail], axis=0)

    # Per-step covariate forcing: input_effect @ transition_inputs[t]. Added
    # to b_local before discretization so the Van Loan formula integrates
    # the constant forcing over the interval (matches the dense path's
    # discretize_system_with_inputs_batched behaviour).
    n_latent = vector_field.n_latent
    if transition_inputs is not None and input_effect is not None:
        forcing_batch = jnp.asarray(transition_inputs) @ jnp.asarray(input_effect).T
    else:
        forcing_batch = jnp.zeros((n_transitions, n_latent), dtype=x_lin_batch.dtype)

    def _per_step(x_lin: Array, dt: Array, forcing: Array) -> tuple[Array, Array, Array]:
        from nof1_causal_lab.models.ssm.discretization import (
            discretize_linear_system_exact,
        )

        A_loc, b_loc = vector_field.linearize(x_lin, args)
        return discretize_linear_system_exact(A_loc, diffusion_cov, b_loc + forcing, dt)

    Ad_batch, Qd_batch, bd_batch = jax.vmap(_per_step, in_axes=(0, 0, 0))(
        x_lin_batch, time_intervals, forcing_batch
    )

    return LatentContext(
        Ad=Ad_batch,
        Qd=Qd_batch,
        cd=bd_batch,
        init_mean=init_mean,
        init_cov=init_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        extra_params=extra_params,
        H_rows=H_rows,
        d_rows=d_rows,
    )


# ---------------------------------------------------------------------------
# Two-context eq10_11 MH step for non-linear dynamics
# ---------------------------------------------------------------------------


class CompositeLatentMHState(NamedTuple):
    """Minimal state for a standalone composite trajectory MH step.

    Mirrors the relevant subset of the auxiliary-Kalman driver state.
    Unlike that one, the composite path does *not* cache
    ``latent_context`` in state — it's rebuilt each step from the
    current trajectory because the linearization is trajectory-dependent.
    """

    position: Array
    latent_trajectory: Array
    latent_delta: Array
    trajectory_log_prob: Array
    complete_log_posterior: Array


def composite_latent_mh_step_eq10_11(
    state: CompositeLatentMHState,
    key: Array,
    runtime_observations: Array,
    *,
    context_builder: Callable[[Array], LatentContext],
    log_prior_unc_fn: Callable[[Array], Array],
    observation_log_prob_and_grad_fn: Callable[
        [LatentContext, Array, Array], tuple[Array, Array]
    ],
    parallel: bool = False,
) -> tuple[CompositeLatentMHState, dict[str, Any]]:
    """One eq10_11 latent-MH step with per-step linearisation rebuilt at
    both ``x_curr`` and ``x_prop``.

    Implements Corenflos & Särkkä (2025) Algorithm 2 generalised to
    non-linear dynamics. The forward proposal linearises at the current
    trajectory (``x_curr``) and the reverse at the proposed trajectory
    (``x_prop``); the MH acceptance ratio combines log-priors and
    proposal densities under the matching linearization on each side.

    Detailed balance is preserved w.r.t. the EKF-approximated posterior
    (the target after the per-step linearization is folded into the
    transition density). This is the same target the paper's Algorithm 2
    samples from when ``p_t(z_t | z_{t-1})`` is replaced by the local
    Gaussian approximation around the current trajectory.

    Args:
        state: ``CompositeLatentMHState`` carrying current trajectory,
            position (unconstrained parameters), step size ``δ``, and
            cached log-posteriors.
        key: JAX PRNG key for the auxiliary observations, the trajectory
            sample, and the accept/reject draw.
        runtime_observations: ``(T, n_m)`` observation matrix.
        context_builder: ``x_traj → LatentContext``. Typically a closure
            over the vector field, parameters, init mean/cov, diffusion
            covariance, ``H, d, R``, and ``runtime_times``. Wrap
            ``composite_latent_context_at_trajectory``.
        log_prior_unc_fn: ``z → scalar`` log-prior on unconstrained
            parameter vector. Cancels out of the trajectory-MH ratio but
            is propagated into ``complete_log_posterior``.
        observation_log_prob_and_grad_fn: ``(context, x, obs) → (log_p,
            grad_x)`` where ``grad_x`` has shape ``(T, n_latent)``.
            Implementations can autodiff a simple obs log-prob or call
            into the existing trajectory_observations machinery.
        parallel: parallel scan flag for the auxiliary filter.

    Returns:
        ``(next_state, extras)`` where ``extras`` contains ``accepted``,
        ``log_alpha``, and the two filter log-evidences.
    """
    aux_key, sample_key, accept_key = random.split(key, 3)

    x_curr = state.latent_trajectory
    delta_val = state.latent_delta
    if delta_val.ndim == 0:
        half_delta_bcast = 0.5 * delta_val
    else:
        half_delta_bcast = 0.5 * delta_val[:, None]
    half_delta_variance = 0.5 * delta_val

    # 1. Forward linearization at x_curr.
    context_curr = context_builder(x_curr)
    prior_terms_curr = build_gaussian_trajectory_prior_terms(
        context_curr.Ad,
        context_curr.Qd,
        context_curr.cd,
        context_curr.init_mean,
        context_curr.init_cov,
        jitter=AUX_JITTER,
    )

    # 2. Current log posterior under x_curr's own linearization (the
    #    EKF-approximated target evaluated at x_curr).
    obs_curr, grad_curr = observation_log_prob_and_grad_fn(
        context_curr, x_curr, runtime_observations
    )
    prior_curr = trajectory_prior_log_prob_from_terms(
        x_curr, context_curr.Ad, context_curr.cd, prior_terms_curr
    )
    traj_curr = prior_curr + obs_curr
    log_prior_z = log_prior_unc_fn(state.position)

    # 3. Auxiliary observations + forward proposal.
    u = x_curr + jnp.sqrt(half_delta_bcast) * random.normal(
        aux_key, x_curr.shape, dtype=x_curr.dtype
    )
    grad_curr_tamed = tame_gradient_tulac(grad_curr)
    pseudo_obs_fwd = u + half_delta_bcast * grad_curr_tamed

    aux_state_fwd = aux_filter_lgssm_lightweight(
        init_mean=context_curr.init_mean,
        init_cov=context_curr.init_cov,
        Fs=context_curr.Ad,
        Qs=context_curr.Qd,
        bs=context_curr.cd,
        pseudo_observations=pseudo_obs_fwd,
        aux_variance=0.5 * delta_val,
        jitter=AUX_JITTER,
        parallel=parallel,
    )
    x_prop = jnp.asarray(
        sample_lgssm_trajectory(
            sample_key,
            aux_state_fwd.filt_mean,
            aux_state_fwd.filt_cov,
            Fs=context_curr.Ad[1:],
            Qs=context_curr.Qd[1:],
            bs=context_curr.cd[1:],
            jitter=AUX_JITTER,
            parallel=parallel,
        ),
        dtype=x_curr.dtype,
    )

    # 4. Reverse linearization at x_prop.
    context_prop = context_builder(x_prop)
    prior_terms_prop = build_gaussian_trajectory_prior_terms(
        context_prop.Ad,
        context_prop.Qd,
        context_prop.cd,
        context_prop.init_mean,
        context_prop.init_cov,
        jitter=AUX_JITTER,
    )

    # 5. Proposed log posterior under x_prop's own linearization.
    obs_prop, grad_prop = observation_log_prob_and_grad_fn(
        context_prop, x_prop, runtime_observations
    )
    prior_prop = trajectory_prior_log_prob_from_terms(
        x_prop, context_prop.Ad, context_prop.cd, prior_terms_prop
    )
    traj_prop = prior_prop + obs_prop

    # 6. Reverse-proposal pseudo-observations + filter via context_prop.
    grad_prop_tamed = tame_gradient_tulac(grad_prop)
    pseudo_obs_rev = u + half_delta_bcast * grad_prop_tamed
    aux_state_rev = aux_filter_lgssm_lightweight(
        init_mean=context_prop.init_mean,
        init_cov=context_prop.init_cov,
        Fs=context_prop.Ad,
        Qs=context_prop.Qd,
        bs=context_prop.cd,
        pseudo_observations=pseudo_obs_rev,
        aux_variance=0.5 * delta_val,
        jitter=AUX_JITTER,
        parallel=parallel,
    )

    # 7. Auxiliary posterior log-probs.
    log_evidence_fwd = jnp.sum(aux_state_fwd.loglik)
    log_evidence_rev = jnp.sum(aux_state_rev.loglik)
    # Forward proposal density at x_prop uses context_curr's prior
    # (the proposal LGSSM linearises at x_curr).
    q_fwd = (
        trajectory_prior_log_prob_from_terms(
            x_prop, context_curr.Ad, context_curr.cd, prior_terms_curr
        )
        + gaussian_log_prob_isotropic(pseudo_obs_fwd, x_prop, half_delta_variance)
        - log_evidence_fwd
    )
    # Reverse proposal density at x_curr uses context_prop's prior.
    q_rev = (
        trajectory_prior_log_prob_from_terms(
            x_curr, context_prop.Ad, context_prop.cd, prior_terms_prop
        )
        + gaussian_log_prob_isotropic(pseudo_obs_rev, x_curr, half_delta_variance)
        - log_evidence_rev
    )

    # 8. MH ratio.
    log_alpha = traj_prop - traj_curr
    log_alpha = log_alpha + gaussian_log_prob_isotropic(u, x_prop, half_delta_variance)
    log_alpha = log_alpha - gaussian_log_prob_isotropic(u, x_curr, half_delta_variance)
    log_alpha = log_alpha + q_rev - q_fwd

    accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    accept = random.bernoulli(accept_key, accept_prob)
    next_traj = jnp.where(accept, x_prop, x_curr)
    next_traj_lp = jnp.where(accept, traj_prop, traj_curr)
    next_complete = log_prior_z + next_traj_lp

    next_state = state._replace(
        latent_trajectory=next_traj,
        trajectory_log_prob=next_traj_lp,
        complete_log_posterior=next_complete,
    )
    extras: dict[str, Any] = {
        "accepted": accept.astype(x_curr.dtype),
        "log_alpha": log_alpha,
        "log_evidence_fwd": log_evidence_fwd,
        "log_evidence_rev": log_evidence_rev,
    }
    return next_state, extras
