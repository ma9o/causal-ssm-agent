"""Support-aware Laplace solvers for interval-summary observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from causal_ssm_agent.models.ssm.inference.targets.base import (
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    build_likelihood_eval_aux,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    accumulate_support_statistics,
    expected_observation_mean,
    get_point_like_mask,
    get_support_kind_codes,
    trajectory_observation_log_prob,
)

from .shared import (
    _SUPPORT_AWARE_IEKS_CONVERGENCE_RTOL,
    _SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS,
    _SUPPORT_AWARE_LM_DAMPING,
    _SUPPORT_AWARE_LM_DAMPING_GROWTH,
    _SUPPORT_AWARE_LM_DAMPING_MAX,
    _SUPPORT_AWARE_LM_DAMPING_MIN,
    _SUPPORT_AWARE_LM_DAMPING_SHRINK,
    GaussianTrajectoryPriorTerms,
    SupportObservationWindowBatch,
    _block_banded_logdet,
    _build_gaussian_trajectory_prior_terms,
    _build_prior_banded_system,
    _factor_block_banded_cholesky,
    _factor_block_profile_cholesky,
    _predictive_latent_init,
    _solve_block_profile_from_cholesky,
    _step_halving_search,
    _trajectory_prior_log_prob_from_terms,
    block_profile_logdet_packed_cotangent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime


def _assemble_support_aware_observation_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble exact Newton observation terms in block-banded form."""
    T, D = z_est.shape
    diag = jnp.zeros((T, D, D), dtype=z_est.dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=z_est.dtype)
    rhs = jnp.zeros((T, D), dtype=z_est.dtype)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    point_mask = obs_mask.astype(z_est.dtype) * point_like_mask[None, :]

    local_grads, local_hess = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
    )(clean_obs, z_est, point_mask)
    diag = diag + local_hess
    rhs = rhs + jax.vmap(lambda j_t, z_t, g_t: j_t @ z_t + g_t)(local_hess, z_est, local_grads)

    if len(support_window_batches) == 0:
        return diag, upper, rhs

    max_support_state_len = max(batch.max_state_len for batch in support_window_batches)
    padded_z = jnp.pad(z_est, ((0, max_support_state_len - 1), (0, 0)))

    for support_windows, batch_window_derivatives in zip(
        support_window_batches,
        window_derivatives,
        strict=True,
    ):
        segment_states = padded_z[support_windows.padded_state_indices]
        segment_flat = segment_states.reshape(segment_states.shape[0], -1)
        anchor_obs = clean_obs[support_windows.anchor_indices]
        grad_blocks, jac_blocks, mean_info = batch_window_derivatives(
            segment_flat,
            support_windows.state_lens,
            support_windows.mask_full.astype(z_est.dtype),
            support_windows.prev_coeffs.astype(z_est.dtype),
            support_windows.curr_coeffs.astype(z_est.dtype),
            support_windows.weights.astype(z_est.dtype),
            anchor_obs,
            H,
            d,
            R,
        )
        diag_updates = jnp.einsum("gmid,gmn,gnie->gide", jac_blocks, mean_info, jac_blocks)
        taylor_rhs = grad_blocks + jnp.einsum("gide,gie->gid", diag_updates, segment_states)

        diag = diag.at[support_windows.time_indices.reshape(-1)].add(
            (
                diag_updates * support_windows.valid_diag.astype(z_est.dtype)[..., None, None]
            ).reshape(-1, D, D)
        )

        batch_bandwidth = support_windows.cross_time_indices.shape[0]
        for offset in range(1, batch_bandwidth + 1):
            cross_len = support_windows.max_state_len - offset
            left_jac = jac_blocks[:, :, :cross_len, :]
            right_jac = jac_blocks[:, :, offset:, :]
            cross_updates = jnp.einsum("gmid,gmn,gnie->gide", left_jac, mean_info, right_jac)

            left_states = segment_states[:, :cross_len, :]
            right_states = segment_states[:, offset:, :]
            taylor_rhs = taylor_rhs.at[:, :cross_len, :].add(
                jnp.einsum("gide,gie->gid", cross_updates, right_states)
            )
            taylor_rhs = taylor_rhs.at[:, offset:, :].add(
                jnp.einsum("gide,gid->gie", cross_updates, left_states)
            )

            valid_cross = support_windows.valid_cross[offset - 1, :, :cross_len]
            upper_times = support_windows.cross_time_indices[offset - 1, :, :cross_len]
            upper = upper.at[offset - 1, upper_times.reshape(-1)].add(
                (cross_updates * valid_cross.astype(z_est.dtype)[..., None, None]).reshape(-1, D, D)
            )

        rhs = rhs.at[support_windows.time_indices.reshape(-1)].add(
            (taylor_rhs * support_windows.valid_diag.astype(z_est.dtype)[..., None]).reshape(
                -1,
                D,
            )
        )

    return diag, upper, rhs


def _make_support_window_derivatives(
    *,
    max_state_len: int,
    n_latent: int,
    n_manifest: int,
    summary_operator_codes: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
):
    """Build support-window derivatives with Gauss-Newton curvature in mean space."""

    def _window_expected_mean_single(
        segment_flat_single: jnp.ndarray,
        state_len_single: jnp.ndarray,
        mask_full_single: jnp.ndarray,
        prev_coeffs_single: jnp.ndarray,
        curr_coeffs_single: jnp.ndarray,
        weights_single: jnp.ndarray,
        anchor_obs_single: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
    ) -> jnp.ndarray:
        states = segment_flat_single.reshape(max_state_len, n_latent)
        responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d))(states)
        last_response = jax.lax.dynamic_index_in_dim(
            responses,
            jnp.maximum(state_len_single - 1, 0),
            axis=0,
            keepdims=False,
        )

        def _single_step_window(_):
            return last_response, last_response**2, mask_full_single

        def _multi_step_window(_):
            zeros = jnp.zeros((n_manifest, 1), dtype=responses.dtype)

            def _scan_step(carry, inputs):
                response_prev, accum_sum, accum_sumsq, accum_weight = carry
                response_t, prev_coeff_t, curr_coeff_t, weight_t = inputs
                obs_sum, obs_sumsq, obs_weight = accumulate_support_statistics(
                    response_prev,
                    accum_sum,
                    accum_sumsq,
                    accum_weight,
                    response_t,
                    prev_coeff_t,
                    curr_coeff_t,
                    weight_t,
                )
                return (response_t, obs_sum, obs_sumsq, obs_weight), None

            final_carry, _ = jax.lax.scan(
                _scan_step,
                (responses[0], zeros, zeros, zeros),
                (
                    responses[1:],
                    prev_coeffs_single[..., None],
                    curr_coeffs_single[..., None],
                    weights_single[..., None],
                ),
            )
            _response_last, obs_sum, obs_sumsq, obs_weight = final_carry
            return obs_sum.squeeze(-1), obs_sumsq.squeeze(-1), obs_weight.squeeze(-1)

        obs_sum, obs_sumsq, obs_weight = jax.lax.cond(
            state_len_single == 1,
            _single_step_window,
            _multi_step_window,
            operand=None,
        )

        expected_mean = expected_observation_mean(
            last_response,
            obs_sum,
            obs_sumsq,
            obs_weight,
            summary_operator_codes,
        )
        del anchor_obs_single, R
        return expected_mean

    def _window_mean_log_prob_single(
        expected_mean_single: jnp.ndarray,
        anchor_obs_single: jnp.ndarray,
        mask_full_single: jnp.ndarray,
        R: jnp.ndarray,
    ) -> jnp.ndarray:
        return mean_log_prob_fn(anchor_obs_single, expected_mean_single, R, mask_full_single)

    window_expected_mean_jacobian = jax.jacrev(_window_expected_mean_single)
    mean_log_prob_grad = jax.grad(_window_mean_log_prob_single)
    mean_log_prob_hessian = jax.hessian(_window_mean_log_prob_single)

    def _batched_window_derivatives(
        segment_flat: jnp.ndarray,
        state_lens: jnp.ndarray,
        mask_full: jnp.ndarray,
        prev_coeffs: jnp.ndarray,
        curr_coeffs: jnp.ndarray,
        weights: jnp.ndarray,
        anchor_obs: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        in_axes = (0, 0, 0, 0, 0, 0, 0, None, None, None)
        expected_mean = jax.vmap(_window_expected_mean_single, in_axes=in_axes)(
            segment_flat,
            state_lens,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
            H,
            d,
            R,
        )
        mean_grad = jax.vmap(mean_log_prob_grad, in_axes=(0, 0, 0, None))(
            expected_mean,
            anchor_obs,
            mask_full,
            R,
        )
        mean_hessian = jax.vmap(mean_log_prob_hessian, in_axes=(0, 0, 0, None))(
            expected_mean,
            anchor_obs,
            mask_full,
            R,
        )
        mean_info = -0.5 * (mean_hessian + jnp.swapaxes(mean_hessian, -1, -2))
        jac_flat = jax.vmap(window_expected_mean_jacobian, in_axes=in_axes)(
            segment_flat,
            state_lens,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
            H,
            d,
            R,
        )
        jac_blocks = jac_flat.reshape(-1, n_manifest, max_state_len, n_latent)
        grad_blocks = jnp.einsum("gmid,gm->gid", jac_blocks, mean_grad)
        return grad_blocks, jac_blocks, mean_info

    return _batched_window_derivatives


def _support_aware_joint_log_prob(
    z_est: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
) -> jnp.ndarray:
    """Exact latent joint log-density used for support-aware step acceptance."""
    return _trajectory_prior_log_prob_from_terms(z_est, Ad, cd, prior_terms) + (
        trajectory_observation_log_prob(
            z_est,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
    )


def _support_aware_step_halving_search(
    z_curr: jnp.ndarray,
    step_direction: jnp.ndarray,
    current_log_joint: jnp.ndarray,
    objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    max_halvings: int = _SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backtracking step-halving line search for the support-aware latent mode."""
    return _step_halving_search(
        z_curr,
        step_direction,
        current_log_joint,
        objective_fn,
        max_halvings=max_halvings,
    )


def _block_banded_matvec(
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a symmetric block-banded matrix to a trajectory-shaped vector."""
    result = jax.vmap(lambda diag_t, x_t: diag_t @ x_t)(diag, x)
    bandwidth = upper.shape[0]
    T = x.shape[0]

    for offset in range(1, bandwidth + 1):
        if offset >= T:
            break
        upper_blocks = upper[offset - 1, : T - offset]
        result = result.at[: T - offset].add(
            jax.vmap(lambda upper_t, x_next: upper_t @ x_next)(upper_blocks, x[offset:])
        )
        result = result.at[offset:].add(
            jax.vmap(lambda upper_t, x_prev: upper_t.T @ x_prev)(upper_blocks, x[: T - offset])
        )

    return result


def _negate_cotangent_tree(tree):
    """Negate a cotangent pytree while preserving `None` and `float0` leaves."""

    def _negate_leaf(leaf):
        if leaf is None or getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return -leaf

    return jax.tree_util.tree_map(_negate_leaf, tree)


def _add_cotangent_trees(lhs, rhs):
    """Add cotangent pytrees while preserving `None` and `float0` leaves."""

    def _add_leaves(left, right):
        if left is None:
            return right
        if right is None:
            return left
        if getattr(left, "dtype", None) == jax.dtypes.float0:
            return right
        if getattr(right, "dtype", None) == jax.dtypes.float0:
            return left
        return left + right

    return jax.tree_util.tree_map(_add_leaves, lhs, rhs)


def _support_aware_posterior_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the exact support-aware posterior Newton system at `z_est`."""
    prior_diag, prior_upper, prior_rhs = _build_prior_banded_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
        bandwidth,
    )
    obs_diag, obs_upper, obs_rhs = _assemble_support_aware_observation_system(
        z_est,
        observations,
        obs_mask,
        H,
        d,
        R,
        obs_kernel,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
    )
    return prior_diag + obs_diag, prior_upper + obs_upper, prior_rhs + obs_rhs


def _support_aware_mode_optimality(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    bandwidth: int,
) -> jnp.ndarray:
    """Return the latent-mode optimality residual F(z, theta) = 0."""
    system_diag, system_upper, system_rhs = _support_aware_posterior_system(
        z_est,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
    )
    return _block_banded_matvec(system_diag, system_upper, z_est) - system_rhs


def _support_aware_ieks_mode(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
    iterate_to_convergence: bool = False,
    factor_block_cholesky_fn=_factor_block_profile_cholesky,
    solve_block_from_cholesky_fn=_solve_block_profile_from_cholesky,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
]:
    """Run the support-aware IEKS Newton iterations and return the latent mode."""
    _T, D = observations.shape[0], init_mean.shape[0]
    prior_diag, prior_upper, prior_rhs = _build_prior_banded_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
        bandwidth,
    )
    prior_terms = _build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
    )
    point_like_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )

    def _support_log_joint(latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        return _support_aware_joint_log_prob(
            latent_trajectory,
            observations=observations,
            obs_mask=obs_mask,
            Ad=Ad,
            cd=cd,
            prior_terms=prior_terms,
            H=H,
            d=d,
            R=R,
            obs_kernel=obs_kernel,
            mean_log_prob_fn=mean_log_prob_fn,
            observation_support=observation_support,
        )

    system_dtype = jnp.result_type(
        prior_diag.dtype,
        observations.dtype,
        H.dtype,
        d.dtype,
        R.dtype,
    )
    if z_init is None:
        z_est = _predictive_latent_init(Ad, cd, init_mean)
    else:
        z_est = jnp.asarray(z_init)
    z_est = z_est.astype(system_dtype)

    log_joint_curr = _support_log_joint(z_est)
    init_log_joint = log_joint_curr

    def _newton_step(carry):
        (
            z_curr,
            log_joint_prev,
            damping,
            _active,
            n_iterations,
            n_accepted_steps,
            _last_rel_change,
            _last_alpha,
            _last_step_norm,
        ) = carry
        damping_curr = damping
        with jax.named_scope("map/support_aware_observation_system"):
            obs_diag, obs_upper, obs_rhs = _assemble_support_aware_observation_system(
                z_curr,
                observations,
                obs_mask,
                H,
                d,
                R,
                obs_kernel,
                support_window_batches,
                point_like_mask,
                window_derivatives,
                bandwidth,
            )
        system_diag = (
            prior_diag + obs_diag + damping_curr * jnp.eye(D, dtype=z_curr.dtype)[None, :, :]
        )
        system_upper = prior_upper + obs_upper
        system_rhs = prior_rhs + obs_rhs
        system_diag = system_diag.astype(system_dtype)
        system_upper = system_upper.astype(system_dtype)
        system_rhs = system_rhs.astype(system_dtype)
        with jax.named_scope("map/support_aware_solve"):
            chol_diag, lower = factor_block_cholesky_fn(
                system_diag,
                system_upper,
                row_upper_bandwidths,
                row_lower_bandwidths,
            )
            z_newton = solve_block_from_cholesky_fn(
                chol_diag,
                lower,
                system_rhs,
                row_upper_bandwidths,
                row_lower_bandwidths,
            )

        step_direction = z_newton - z_curr
        step_norm = jnp.linalg.norm(step_direction)
        z_next, log_joint_next, accepted, accepted_alpha = _support_aware_step_halving_search(
            z_curr,
            step_direction,
            log_joint_prev,
            _support_log_joint,
        )

        rel_change = jnp.linalg.norm(z_next - z_curr) / (1.0 + jnp.linalg.norm(z_curr))
        accepted_full_step = accepted & (accepted_alpha > 0.999)

        damping_next = jax.lax.cond(
            accepted_full_step,
            lambda _: jnp.maximum(
                damping * jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_SHRINK, dtype=z_curr.dtype),
                jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_MIN, dtype=z_curr.dtype),
            ),
            lambda _: jax.lax.cond(
                accepted,
                lambda __: damping_curr,
                lambda __: jnp.minimum(
                    damping_curr
                    * jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_GROWTH, dtype=z_curr.dtype),
                    jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_MAX, dtype=z_curr.dtype),
                ),
                operand=None,
            ),
            operand=None,
        )
        next_active = jax.lax.cond(
            accepted,
            lambda _: rel_change > _SUPPORT_AWARE_IEKS_CONVERGENCE_RTOL,
            lambda _: damping_next < jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_MAX, dtype=z_curr.dtype),
            operand=None,
        )
        return (
            z_next,
            log_joint_next,
            damping_next,
            next_active,
            n_iterations + jnp.asarray(1, dtype=jnp.int32),
            n_accepted_steps + accepted.astype(jnp.int32),
            rel_change,
            accepted_alpha,
            step_norm,
        )

    init_carry = (
        z_est,
        log_joint_curr,
        jnp.asarray(_SUPPORT_AWARE_LM_DAMPING, dtype=z_est.dtype),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.nan, dtype=z_est.dtype),
        jnp.asarray(jnp.nan, dtype=z_est.dtype),
        jnp.asarray(jnp.nan, dtype=z_est.dtype),
    )
    max_iters = jnp.asarray(max(n_ieks_iters, 1), dtype=jnp.int32)

    with jax.named_scope("map/support_aware_newton"):
        if iterate_to_convergence:

            def _continue(carry):
                return carry[3] & (carry[4] < max_iters)

            (
                z_est,
                _mode_log_joint,
                final_damping,
                _active,
                n_iterations,
                n_accepted_steps,
                final_rel_change,
                final_step_alpha,
                final_step_norm,
            ) = jax.lax.while_loop(
                _continue,
                _newton_step,
                init_carry,
            )
        else:

            def _scan_step(carry, _idx):
                carry_cast = carry
                return jax.lax.cond(
                    carry[3],
                    _newton_step,
                    lambda _: carry_cast,
                    operand=carry,
                ), None

            (
                (
                    z_est,
                    _mode_log_joint,
                    final_damping,
                    _active,
                    n_iterations,
                    n_accepted_steps,
                    final_rel_change,
                    final_step_alpha,
                    final_step_norm,
                ),
                _,
            ) = jax.lax.scan(
                _scan_step,
                init_carry,
                xs=jnp.arange(max(n_ieks_iters, 1)),
            )

    return z_est, (
        init_log_joint,
        n_iterations,
        n_accepted_steps,
        final_rel_change,
        final_damping,
        final_step_alpha,
        final_step_norm,
    )


def _support_aware_laplace_from_mode(
    z_mode: jnp.ndarray,
    mode_aux: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Evaluate the Laplace correction at an already-solved support-aware mode."""
    (
        init_log_joint,
        n_iterations,
        n_accepted_steps,
        final_rel_change,
        final_damping,
        final_step_alpha,
        final_step_norm,
    ) = mode_aux
    log_lik, mode_log_joint, laplace_logdet, min_chol_diag = _support_aware_laplace_terms_from_mode(
        z_mode,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        mean_log_prob_fn,
        observation_support,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        factor_block_cholesky_fn=factor_block_cholesky_fn,
    )
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
        n_iterations=n_iterations,
        n_accepted_steps=n_accepted_steps,
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        final_rel_change=final_rel_change,
        final_damping=final_damping,
        final_step_alpha=final_step_alpha,
        final_step_norm=final_step_norm,
        laplace_logdet=laplace_logdet,
        min_chol_diag=min_chol_diag,
    )
    inner_eval_aux["latent_mode"] = z_mode
    return log_lik, inner_eval_aux


def _support_aware_laplace_terms_from_mode(
    z_mode: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the Laplace log-likelihood terms at a fixed latent mode."""
    prior_terms = _build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
    )
    mode_log_joint = _support_aware_joint_log_prob(
        z_mode,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad,
        cd=cd,
        prior_terms=prior_terms,
        H=H,
        d=d,
        R=R,
        obs_kernel=obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_support=observation_support,
    )
    system_diag, system_upper, _system_rhs = _support_aware_posterior_system(
        z_mode,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
    )
    with jax.named_scope("map/support_aware_final_hessian"):
        chol_diag, _lower = factor_block_cholesky_fn(
            system_diag,
            system_upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

    flat_dim = observations.shape[0] * init_mean.shape[0]
    laplace_logdet = _block_banded_logdet(chol_diag)
    min_chol_diag = jnp.min(jnp.diagonal(chol_diag, axis1=1, axis2=2))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * laplace_logdet
    return log_lik, mode_log_joint, laplace_logdet, min_chol_diag


def _support_aware_ieks_laplace_core(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    build_measurement_objects: Callable[[jnp.ndarray, dict | None], tuple[Any, tuple[Any, ...]]],
    extra_params: dict | None,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
    final_factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Support-aware IEKS solve plus Laplace log-likelihood."""
    del obs_kernel, mean_log_prob_fn, window_derivatives
    point_like_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )

    def _mode_core(mode_params):
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            extra_params_curr,
        ) = mode_params
        measurement_semantics_curr, window_derivatives_curr = build_measurement_objects(
            R_curr,
            extra_params_curr,
        )
        return _support_aware_ieks_mode(
            observations=observations,
            obs_mask=obs_mask,
            Ad=Ad_curr,
            Qd=Qd_curr,
            cd=cd_curr,
            H=H_curr,
            d=d_curr,
            R=R_curr,
            init_mean=init_mean_curr,
            init_cov=init_cov_curr,
            obs_kernel=measurement_semantics_curr.obs_kernel,
            mean_log_prob_fn=measurement_semantics_curr.mean_log_prob_fn,
            observation_support=observation_support,
            support_window_batches=support_window_batches,
            bandwidth=bandwidth,
            row_upper_bandwidths=row_upper_bandwidths,
            row_lower_bandwidths=row_lower_bandwidths,
            window_derivatives=window_derivatives_curr,
            n_ieks_iters=n_ieks_iters,
            z_init=z_init,
            iterate_to_convergence=True,
        )

    @jax.custom_vjp
    def _implicit_mode_solve(mode_params):
        return _mode_core(mode_params)

    def _implicit_mode_solve_fwd(mode_params):
        z_mode, mode_aux = _mode_core(mode_params)
        return (z_mode, mode_aux), (mode_params, z_mode)

    def _implicit_mode_solve_bwd(res, output_ct):
        mode_params, z_mode = res
        z_mode_bar, _mode_aux_bar = output_ct
        del _mode_aux_bar
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            extra_params_curr,
        ) = mode_params
        measurement_semantics_curr, window_derivatives_curr = build_measurement_objects(
            R_curr,
            extra_params_curr,
        )
        system_diag, system_upper, _system_rhs = _support_aware_posterior_system(
            z_mode,
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            measurement_semantics_curr.obs_kernel,
            support_window_batches,
            point_like_mask,
            window_derivatives_curr,
            bandwidth,
        )
        chol_diag, lower = _factor_block_profile_cholesky(
            system_diag,
            system_upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        lambda_mode = _solve_block_profile_from_cholesky(
            chol_diag,
            lower,
            z_mode_bar,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

        def _optimality(mode_params_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_inner,
                d_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                extra_params_inner,
            ) = mode_params_inner
            measurement_semantics_inner, window_derivatives_inner = build_measurement_objects(
                R_inner,
                extra_params_inner,
            )
            return _support_aware_mode_optimality(
                z_mode,
                observations,
                obs_mask,
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_inner,
                d_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                measurement_semantics_inner.obs_kernel,
                support_window_batches,
                point_like_mask,
                window_derivatives_inner,
                bandwidth,
            )

        _, vjp_fn = jax.vjp(_optimality, mode_params)
        (mode_params_bar,) = vjp_fn(lambda_mode)
        return (_negate_cotangent_tree(mode_params_bar),)

    _implicit_mode_solve.defvjp(_implicit_mode_solve_fwd, _implicit_mode_solve_bwd)

    def _laplace_from_mode_core(mode_params, z_mode, *, factor_block_cholesky_fn):
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            extra_params_curr,
        ) = mode_params
        measurement_semantics_curr, window_derivatives_curr = build_measurement_objects(
            R_curr,
            extra_params_curr,
        )
        log_lik, mode_log_joint, laplace_logdet, min_chol_diag = (
            _support_aware_laplace_terms_from_mode(
                z_mode,
                observations,
                obs_mask,
                Ad_curr,
                Qd_curr,
                cd_curr,
                H_curr,
                d_curr,
                R_curr,
                init_mean_curr,
                init_cov_curr,
                measurement_semantics_curr.obs_kernel,
                measurement_semantics_curr.mean_log_prob_fn,
                observation_support,
                support_window_batches,
                point_like_mask,
                window_derivatives_curr,
                bandwidth,
                row_upper_bandwidths,
                row_lower_bandwidths,
                factor_block_cholesky_fn=factor_block_cholesky_fn,
            )
        )
        laplace_aux = {
            "mode_log_joint": mode_log_joint,
            "laplace_logdet": laplace_logdet,
            "min_chol_diag": min_chol_diag,
        }
        return log_lik, laplace_aux

    def _laplace_from_mode_rev_safe_loglik(mode_params, z_mode):
        log_lik, _laplace_aux = _laplace_from_mode_core(
            mode_params,
            z_mode,
            factor_block_cholesky_fn=_factor_block_banded_cholesky,
        )
        return log_lik

    @jax.custom_vjp
    def _laplace_from_mode_eval(mode_params, z_mode):
        return _laplace_from_mode_core(
            mode_params,
            z_mode,
            factor_block_cholesky_fn=final_factor_block_cholesky_fn,
        )

    def _laplace_from_mode_eval_fwd(mode_params, z_mode):
        outputs = _laplace_from_mode_core(
            mode_params,
            z_mode,
            factor_block_cholesky_fn=final_factor_block_cholesky_fn,
        )
        return outputs, (mode_params, z_mode)

    def _laplace_from_mode_eval_bwd(res, output_ct):
        mode_params, z_mode = res
        log_lik_bar, _laplace_aux_bar = output_ct
        del _laplace_aux_bar
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            extra_params_curr,
        ) = mode_params
        measurement_semantics_curr, window_derivatives_curr = build_measurement_objects(
            R_curr,
            extra_params_curr,
        )

        def _mode_log_joint_eval(mode_params_inner, z_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_inner,
                d_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                extra_params_inner,
            ) = mode_params_inner
            measurement_semantics_inner, _window_derivatives_inner = build_measurement_objects(
                R_inner,
                extra_params_inner,
            )
            prior_terms_inner = _build_gaussian_trajectory_prior_terms(
                Ad_inner,
                Qd_inner,
                cd_inner,
                init_mean_inner,
                init_cov_inner,
            )
            return _support_aware_joint_log_prob(
                z_inner,
                observations=observations,
                obs_mask=obs_mask,
                Ad=Ad_inner,
                cd=cd_inner,
                prior_terms=prior_terms_inner,
                H=H_inner,
                d=d_inner,
                R=R_inner,
                obs_kernel=measurement_semantics_inner.obs_kernel,
                mean_log_prob_fn=measurement_semantics_inner.mean_log_prob_fn,
                observation_support=observation_support,
            )

        _, mode_log_joint_vjp = jax.vjp(_mode_log_joint_eval, mode_params, z_mode)
        mode_params_joint_bar, z_mode_joint_bar = mode_log_joint_vjp(log_lik_bar)

        system_diag, system_upper, _system_rhs = _support_aware_posterior_system(
            z_mode,
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_curr,
            d_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            measurement_semantics_curr.obs_kernel,
            support_window_batches,
            point_like_mask,
            window_derivatives_curr,
            bandwidth,
        )
        chol_diag, lower = _factor_block_profile_cholesky(
            system_diag,
            system_upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        system_diag_bar, system_upper_bar = block_profile_logdet_packed_cotangent(
            chol_diag,
            lower,
            row_upper_bandwidths,
            row_lower_bandwidths,
            scale=-0.5 * log_lik_bar,
        )

        def _posterior_system_eval(mode_params_inner, z_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_inner,
                d_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                extra_params_inner,
            ) = mode_params_inner
            measurement_semantics_inner, window_derivatives_inner = build_measurement_objects(
                R_inner,
                extra_params_inner,
            )
            return _support_aware_posterior_system(
                z_inner,
                observations,
                obs_mask,
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_inner,
                d_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                measurement_semantics_inner.obs_kernel,
                support_window_batches,
                point_like_mask,
                window_derivatives_inner,
                bandwidth,
            )

        _, posterior_system_vjp = jax.vjp(_posterior_system_eval, mode_params, z_mode)
        mode_params_logdet_bar, z_mode_logdet_bar = posterior_system_vjp(
            (
                system_diag_bar,
                system_upper_bar,
                jnp.zeros_like(z_mode),
            )
        )
        mode_params_bar = _add_cotangent_trees(mode_params_joint_bar, mode_params_logdet_bar)
        z_mode_bar = z_mode_joint_bar + z_mode_logdet_bar
        return mode_params_bar, z_mode_bar

    _laplace_from_mode_eval.defvjp(_laplace_from_mode_eval_fwd, _laplace_from_mode_eval_bwd)

    mode_params = (
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        extra_params,
    )
    z_est, mode_aux = _implicit_mode_solve(mode_params)
    (
        init_log_joint,
        n_iterations,
        n_accepted_steps,
        final_rel_change,
        final_damping,
        final_step_alpha,
        final_step_norm,
    ) = mode_aux
    log_lik, laplace_aux = _laplace_from_mode_eval(mode_params, z_est)
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
        n_iterations=n_iterations,
        n_accepted_steps=n_accepted_steps,
        init_log_joint=init_log_joint,
        final_log_joint=laplace_aux["mode_log_joint"],
        final_rel_change=final_rel_change,
        final_damping=final_damping,
        final_step_alpha=final_step_alpha,
        final_step_norm=final_step_norm,
        laplace_logdet=laplace_aux["laplace_logdet"],
        min_chol_diag=laplace_aux["min_chol_diag"],
    )
    inner_eval_aux["latent_mode"] = z_est
    return log_lik, z_est, inner_eval_aux


def _support_aware_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[Any, ...],
    build_measurement_objects: Callable[[jnp.ndarray, dict | None], tuple[Any, tuple[Any, ...]]],
    extra_params: dict | None,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Support-aware IEKS solve plus Laplace log-likelihood."""
    return _support_aware_ieks_laplace_core(
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        mean_log_prob_fn,
        observation_support,
        support_window_batches,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        window_derivatives,
        build_measurement_objects,
        extra_params,
        n_ieks_iters,
        z_init=z_init,
        final_factor_block_cholesky_fn=_factor_block_profile_cholesky,
    )
