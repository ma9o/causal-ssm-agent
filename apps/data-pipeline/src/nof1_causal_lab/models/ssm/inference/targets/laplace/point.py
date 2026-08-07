"""Point-observation and linear-summary Laplace solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from nof1_causal_lab.models.ssm.execution.contracts import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    LikelihoodExtraParams,
    build_likelihood_eval_aux,
)
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    row_observation_log_prob as _row_observation_log_prob,
)
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    trajectory_observation_log_prob,
)
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import RuntimeDynamics

from .shared import (
    _POINT_IEKS_CONVERGENCE_RTOL,
    _POINT_LINE_SEARCH_MAX_HALVINGS,
    _POINT_LM_DAMPING,
    _POINT_LM_DAMPING_GROWTH,
    _POINT_LM_DAMPING_MAX,
    _POINT_LM_DAMPING_MIN,
    _POINT_LM_DAMPING_SHRINK,
    GaussianTrajectoryPriorTerms,
    _block_banded_logdet,
    _build_ieks_system_from_prior,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_banded_cholesky,
    _predictive_latent_init,
    _solve_block_banded_from_cholesky,
    _solve_block_tridiagonal,
    _step_halving_search,
    block_profile_logdet_packed_cotangent,
    build_gaussian_trajectory_prior_terms,
    trajectory_prior_log_prob_from_terms,
)


def _row_joint_log_prob(
    latent_trajectory: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Exact latent joint log-density for per-row observation operators."""
    return trajectory_prior_log_prob_from_terms(latent_trajectory, Ad, cd, prior_terms) + (
        _row_observation_log_prob(
            latent_trajectory,
            observations,
            obs_mask,
            H_rows,
            d_rows,
            R,
            obs_kernel,
        )
    )


def _negate_point_cotangent_tree(tree):
    """Negate every leaf in a cotangent pytree, preserving Nones."""
    return jax.tree_util.tree_map(lambda leaf: None if leaf is None else -leaf, tree)


def _add_point_cotangent_trees(lhs, rhs):
    """Add cotangent pytrees, treating missing leaves as additive identities."""
    leaves_lhs = jax.tree_util.tree_leaves(lhs, is_leaf=lambda leaf: leaf is None)
    leaves_rhs = jax.tree_util.tree_leaves(rhs, is_leaf=lambda leaf: leaf is None)
    if not leaves_lhs:
        return rhs
    if not leaves_rhs:
        return lhs
    return jax.tree_util.tree_map(
        lambda left, right: right if left is None else left if right is None else left + right,
        lhs,
        rhs,
        is_leaf=lambda leaf: leaf is None,
    )


def _point_profile_bandwidths(n_time: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return profile bandwidth vectors for a block-tridiagonal system."""
    row_upper = np.zeros((n_time,), dtype=np.int32)
    if n_time > 1:
        row_upper[:-1] = 1
    row_lower = _compute_profile_lower_bandwidths(row_upper.astype(np.int64)).astype(np.int32)
    return jnp.asarray(row_upper, dtype=jnp.int32), jnp.asarray(row_lower, dtype=jnp.int32)


def _block_tridiagonal_matvec(
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    latent_trajectory: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a symmetric block-tridiagonal matrix to a latent trajectory."""
    result = jax.vmap(lambda diag_t, z_t: diag_t @ z_t)(diag, latent_trajectory)
    if latent_trajectory.shape[0] <= 1:
        return result
    result = result.at[:-1].add(
        jax.vmap(lambda upper_t, z_next: upper_t @ z_next)(
            upper[:-1],
            latent_trajectory[1:],
        )
    )
    return result.at[1:].add(
        jax.vmap(lambda upper_t, z_prev: upper_t.T @ z_prev)(
            upper[:-1],
            latent_trajectory[:-1],
        )
    )


def _point_linearize(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
    z_estimate: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-time emission gradients and negative Hessians."""
    obs_mask_float = obs_mask.astype(observations.dtype)
    grads_and_hess = jax.vmap(
        lambda y_t, z_t, mask_t, H_t, d_t: obs_kernel.latent_grad_hess_fn(
            y_t,
            z_t,
            H_t,
            d_t,
            R,
            mask_t,
        )
    )(
        observations,
        z_estimate,
        obs_mask_float,
        H_rows,
        d_rows,
    )
    return grads_and_hess[0], grads_and_hess[1]


def _point_posterior_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the point-observation IEKS linear system at a latent iterate."""
    T, D = z_est.shape
    cd_scan = cd if cd is not None else jnp.zeros((T, D), dtype=z_est.dtype)
    prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )
    grads, J_t = _point_linearize(
        observations,
        obs_mask,
        H_rows,
        d_rows,
        R,
        obs_kernel,
        z_est,
    )
    tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)
    _lower, diag, upper, rhs = _build_ieks_system_from_prior(
        prior_lower,
        prior_diag,
        prior_upper,
        prior_rhs,
        J_t,
        tilde_y,
    )
    return diag, upper, rhs


def _point_mode_optimality(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Return the point-observation latent-mode optimality residual F(z, theta) = 0."""
    system_diag, system_upper, system_rhs = _point_posterior_system(
        z_est,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
    )
    return _block_tridiagonal_matvec(system_diag, system_upper, z_est) - system_rhs


def _point_ieks_mode(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    *,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
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
    """Run the point-observation IEKS solve to convergence or max-iteration cap."""
    T = observations.shape[0]
    D = init_mean.shape[0]
    cd_scan = cd if cd is not None else jnp.zeros((T, D))
    prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )
    prior_terms = build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )

    def _row_log_joint(latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        return _row_joint_log_prob(
            latent_trajectory,
            observations=observations,
            obs_mask=obs_mask,
            Ad=Ad,
            cd=cd_scan,
            prior_terms=prior_terms,
            H_rows=H_rows,
            d_rows=d_rows,
            R=R,
            obs_kernel=obs_kernel,
        )

    if z_init is None:
        z_est = _predictive_latent_init(Ad, cd_scan, init_mean)
    else:
        z_est = jnp.asarray(z_init, dtype=observations.dtype)

    log_joint_curr = _row_log_joint(z_est)
    init_log_joint = log_joint_curr
    max_iters = jnp.asarray(max(n_ieks_iters, 1), dtype=jnp.int32)

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
        grads, J_t = _point_linearize(
            observations,
            obs_mask,
            H_rows,
            d_rows,
            R,
            obs_kernel,
            z_curr,
        )
        tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_curr, grads)
        lower, diag, upper, rhs = _build_ieks_system_from_prior(
            prior_lower,
            prior_diag,
            prior_upper,
            prior_rhs,
            J_t,
            tilde_y,
        )
        diag = diag + damping * jnp.eye(D, dtype=z_curr.dtype)[None, :, :]

        with jax.named_scope("map/ieks_solve_system"):
            z_newton = jnp.asarray(
                _solve_block_tridiagonal(lower, diag, upper, rhs),
                dtype=z_curr.dtype,
            )

        step_direction = jnp.asarray(z_newton - z_curr, dtype=z_curr.dtype)
        step_norm = jnp.asarray(jnp.linalg.norm(step_direction), dtype=z_curr.dtype)
        z_next, log_joint_next, accepted, accepted_alpha = _step_halving_search(
            z_curr,
            step_direction,
            log_joint_prev,
            _row_log_joint,
            max_halvings=_POINT_LINE_SEARCH_MAX_HALVINGS,
        )

        rel_change = jnp.asarray(
            jnp.linalg.norm(z_next - z_curr) / (1.0 + jnp.linalg.norm(z_curr)),
            dtype=z_curr.dtype,
        )
        accepted_full_step = accepted & (accepted_alpha > 0.999)
        damping_next = jax.lax.cond(
            accepted_full_step,
            lambda _: jnp.maximum(
                damping * jnp.asarray(_POINT_LM_DAMPING_SHRINK, dtype=z_curr.dtype),
                jnp.asarray(_POINT_LM_DAMPING_MIN, dtype=z_curr.dtype),
            ),
            lambda _: jax.lax.cond(
                accepted,
                lambda __: damping,
                lambda __: jnp.minimum(
                    damping * jnp.asarray(_POINT_LM_DAMPING_GROWTH, dtype=z_curr.dtype),
                    jnp.asarray(_POINT_LM_DAMPING_MAX, dtype=z_curr.dtype),
                ),
                operand=None,
            ),
            operand=None,
        )
        next_active = jax.lax.cond(
            accepted,
            lambda _: rel_change > _POINT_IEKS_CONVERGENCE_RTOL,
            lambda _: damping_next < jnp.asarray(_POINT_LM_DAMPING_MAX, dtype=z_curr.dtype),
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

    def _continue(carry):
        return carry[3] & (carry[4] < max_iters)

    with jax.named_scope("map/ieks_iterations"):
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
            (
                z_est,
                log_joint_curr,
                jnp.asarray(_POINT_LM_DAMPING, dtype=z_est.dtype),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
            ),
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


def _point_laplace_terms_from_mode(
    z_mode: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    *,
    factor_block_cholesky_fn=_factor_block_banded_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the point-observation Laplace terms at a fixed latent mode."""
    T, D = z_mode.shape
    cd_scan = cd if cd is not None else jnp.zeros((T, D), dtype=z_mode.dtype)
    prior_terms = build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )
    mode_log_joint = _row_joint_log_prob(
        z_mode,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad,
        cd=cd_scan,
        prior_terms=prior_terms,
        H_rows=H_rows,
        d_rows=d_rows,
        R=R,
        obs_kernel=obs_kernel,
    )
    system_diag, system_upper, _system_rhs = _point_posterior_system(
        z_mode,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd_scan,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
    )
    row_upper_bandwidths, row_lower_bandwidths = _point_profile_bandwidths(T)
    chol_diag, _lower = factor_block_cholesky_fn(
        system_diag,
        system_upper[None, ...],
        row_upper_bandwidths,
        row_lower_bandwidths,
    )
    flat_dim = T * D
    laplace_logdet = _block_banded_logdet(chol_diag)
    min_chol_diag = jnp.min(jnp.diagonal(chol_diag, axis1=1, axis2=2))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * laplace_logdet
    return log_lik, mode_log_joint, laplace_logdet, min_chol_diag


def _point_ieks_laplace_core(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    *,
    build_measurement_objects=None,
    extra_params: LikelihoodExtraParams | None = None,
    solver_kind: int,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Point-observation IEKS solve plus Laplace likelihood with implicit gradients."""
    row_upper_bandwidths, row_lower_bandwidths = _point_profile_bandwidths(observations.shape[0])

    def _unpack_mode_params(mode_params):
        if build_measurement_objects is None:
            (
                Ad_curr,
                Qd_curr,
                cd_curr,
                H_rows_curr,
                d_rows_curr,
                R_curr,
                init_mean_curr,
                init_cov_curr,
            ) = mode_params
            return (
                Ad_curr,
                Qd_curr,
                cd_curr,
                H_rows_curr,
                d_rows_curr,
                R_curr,
                init_mean_curr,
                init_cov_curr,
                obs_kernel,
            )
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            extra_params_curr,
        ) = mode_params
        measurement_semantics_curr = build_measurement_objects(R_curr, extra_params_curr)
        return (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            measurement_semantics_curr.kernel,
        )

    def _mode_core(mode_params):
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
        ) = _unpack_mode_params(mode_params)
        return _point_ieks_mode(
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
            n_ieks_iters=n_ieks_iters,
            z_init=z_init,
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
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
        ) = _unpack_mode_params(mode_params)
        system_diag, system_upper, system_rhs = _point_posterior_system(
            z_mode,
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
        )
        chol_diag, lower = _factor_block_banded_cholesky(
            system_diag,
            jnp.asarray(system_upper[None, ...], dtype=system_diag.dtype),
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        lambda_mode = _solve_block_banded_from_cholesky(
            chol_diag,
            lower,
            jnp.asarray(z_mode_bar, dtype=system_rhs.dtype),
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

        def _optimality(mode_params_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_rows_inner,
                d_rows_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                obs_kernel_inner,
            ) = _unpack_mode_params(mode_params_inner)
            return _point_mode_optimality(
                z_mode,
                observations,
                obs_mask,
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_rows_inner,
                d_rows_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                obs_kernel_inner,
            )

        _, vjp_fn = jax.vjp(_optimality, mode_params)
        (mode_params_bar,) = vjp_fn(lambda_mode)
        return (_negate_point_cotangent_tree(mode_params_bar),)

    _implicit_mode_solve.defvjp(_implicit_mode_solve_fwd, _implicit_mode_solve_bwd)

    def _laplace_from_mode_core(mode_params, z_mode):
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
        ) = _unpack_mode_params(mode_params)
        log_lik, mode_log_joint, laplace_logdet, min_chol_diag = _point_laplace_terms_from_mode(
            z_mode,
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            obs_kernel_curr,
        )
        return log_lik, {
            "mode_log_joint": mode_log_joint,
            "laplace_logdet": laplace_logdet,
            "min_chol_diag": min_chol_diag,
        }

    @jax.custom_vjp
    def _laplace_from_mode_eval(mode_params, z_mode):
        return _laplace_from_mode_core(mode_params, z_mode)

    def _laplace_from_mode_eval_fwd(mode_params, z_mode):
        outputs = _laplace_from_mode_core(mode_params, z_mode)
        return outputs, (mode_params, z_mode)

    def _laplace_from_mode_eval_bwd(res, output_ct):
        mode_params, z_mode = res
        log_lik_bar, _laplace_aux_bar = output_ct
        del _laplace_aux_bar
        (
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            _obs_kernel_curr,
        ) = _unpack_mode_params(mode_params)

        def _mode_log_joint_eval(mode_params_inner, z_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_rows_inner,
                d_rows_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                obs_kernel_inner,
            ) = _unpack_mode_params(mode_params_inner)
            cd_scan_inner = (
                cd_inner
                if cd_inner is not None
                else jnp.zeros((z_inner.shape[0], z_inner.shape[1]), dtype=z_inner.dtype)
            )
            prior_terms_inner = build_gaussian_trajectory_prior_terms(
                Ad_inner,
                Qd_inner,
                cd_scan_inner,
                init_mean_inner,
                init_cov_inner,
            )
            return _row_joint_log_prob(
                z_inner,
                observations=observations,
                obs_mask=obs_mask,
                Ad=Ad_inner,
                cd=cd_scan_inner,
                prior_terms=prior_terms_inner,
                H_rows=H_rows_inner,
                d_rows=d_rows_inner,
                R=R_inner,
                obs_kernel=obs_kernel_inner,
            )

        _, mode_log_joint_vjp = jax.vjp(_mode_log_joint_eval, mode_params, z_mode)
        mode_params_joint_bar, z_mode_joint_bar = mode_log_joint_vjp(log_lik_bar)

        system_diag, system_upper, system_rhs = _point_posterior_system(
            z_mode,
            observations,
            obs_mask,
            Ad_curr,
            Qd_curr,
            cd_curr,
            H_rows_curr,
            d_rows_curr,
            R_curr,
            init_mean_curr,
            init_cov_curr,
            _obs_kernel_curr,
        )
        chol_diag, lower = _factor_block_banded_cholesky(
            system_diag,
            jnp.asarray(system_upper[None, ...], dtype=system_diag.dtype),
            row_upper_bandwidths,
            row_lower_bandwidths,
        )
        system_diag_bar, system_upper_bar_packed = block_profile_logdet_packed_cotangent(
            chol_diag,
            lower,
            row_upper_bandwidths,
            row_lower_bandwidths,
            scale=jnp.asarray(-0.5 * log_lik_bar, dtype=system_diag.dtype),
        )
        system_diag_bar = jnp.asarray(system_diag_bar, dtype=system_diag.dtype)
        system_upper_bar = jnp.asarray(system_upper_bar_packed[0], dtype=system_upper.dtype)

        def _posterior_system_eval(mode_params_inner, z_inner):
            (
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_rows_inner,
                d_rows_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                obs_kernel_inner,
            ) = _unpack_mode_params(mode_params_inner)
            return _point_posterior_system(
                z_inner,
                observations,
                obs_mask,
                Ad_inner,
                Qd_inner,
                cd_inner,
                H_rows_inner,
                d_rows_inner,
                R_inner,
                init_mean_inner,
                init_cov_inner,
                obs_kernel_inner,
            )

        _, posterior_system_vjp = jax.vjp(_posterior_system_eval, mode_params, z_mode)
        mode_params_logdet_bar, z_mode_logdet_bar = posterior_system_vjp(
            (
                jnp.asarray(system_diag_bar, dtype=system_diag.dtype),
                jnp.asarray(system_upper_bar, dtype=system_upper.dtype),
                jnp.zeros_like(system_rhs),
            )
        )
        mode_params_bar = _add_point_cotangent_trees(
            mode_params_joint_bar,
            mode_params_logdet_bar,
        )
        z_mode_bar = z_mode_joint_bar + z_mode_logdet_bar
        return mode_params_bar, z_mode_bar

    _laplace_from_mode_eval.defvjp(_laplace_from_mode_eval_fwd, _laplace_from_mode_eval_bwd)

    mode_params = (
        (
            Ad,
            Qd,
            cd,
            H_rows,
            d_rows,
            R,
            init_mean,
            init_cov,
        )
        if build_measurement_objects is None
        else (
            Ad,
            Qd,
            cd,
            H_rows,
            d_rows,
            R,
            init_mean,
            init_cov,
            extra_params,
        )
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
        solver_kind=solver_kind,
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
    return z_est, log_lik, inner_eval_aux


def _transition_start_linearization_states(
    latent_trajectory: jnp.ndarray,
    init_mean: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-transition start states for local dynamics linearization."""
    return jnp.concatenate((init_mean[None, :], latent_trajectory[:-1]), axis=0)


def _point_dynamic_transition_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    dynamics: RuntimeDynamics,
    time_intervals: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    *,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    solver_kind: int = LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Point IEKS/Laplace path with per-iteration local dynamics linearization."""
    T = observations.shape[0]
    D = init_mean.shape[0]

    def _transitions_at(z_path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        transitions = build_discrete_transitions(
            dynamics,
            time_intervals,
            linearization_states=_transition_start_linearization_states(z_path, init_mean),
            transition_inputs=transition_inputs,
        )
        cd_scan = (
            transitions.cd
            if transitions.cd is not None
            else jnp.zeros((T, D), dtype=observations.dtype)
        )
        return transitions.Ad, transitions.Qd, jnp.asarray(cd_scan, dtype=observations.dtype)

    if z_init is None:
        init_ref = jnp.broadcast_to(init_mean[None, :], (T, D))
        Ad_init, _Qd_init, cd_init = _transitions_at(init_ref)
        z_est = _predictive_latent_init(Ad_init, cd_init, init_mean)
    else:
        z_est = jnp.asarray(z_init, dtype=observations.dtype)

    Ad_curr, Qd_curr, cd_curr = _transitions_at(z_est)
    prior_terms_curr = build_gaussian_trajectory_prior_terms(
        Ad_curr,
        Qd_curr,
        cd_curr,
        init_mean,
        init_cov,
    )
    log_joint_curr = _row_joint_log_prob(
        z_est,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad_curr,
        cd=cd_curr,
        prior_terms=prior_terms_curr,
        H_rows=H_rows,
        d_rows=d_rows,
        R=R,
        obs_kernel=obs_kernel,
    )
    init_log_joint = log_joint_curr
    damping = jnp.asarray(_POINT_LM_DAMPING, dtype=z_est.dtype)
    active = jnp.asarray(True)
    n_iterations = jnp.asarray(0, dtype=jnp.int32)
    n_accepted_steps = jnp.asarray(0, dtype=jnp.int32)
    final_rel_change = jnp.asarray(jnp.nan, dtype=z_est.dtype)
    final_step_alpha = jnp.asarray(jnp.nan, dtype=z_est.dtype)
    final_step_norm = jnp.asarray(jnp.nan, dtype=z_est.dtype)

    eye = jnp.eye(D, dtype=z_est.dtype)
    for _ in range(max(n_ieks_iters, 1)):
        Ad_step, Qd_step, cd_step = _transitions_at(z_est)
        prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
            Ad_step,
            Qd_step,
            cd_step,
            init_mean,
            init_cov,
        )
        prior_terms_step = build_gaussian_trajectory_prior_terms(
            Ad_step,
            Qd_step,
            cd_step,
            init_mean,
            init_cov,
        )

        def _row_log_joint(
            latent_trajectory: jnp.ndarray,
            *,
            Ad_step=Ad_step,
            cd_step=cd_step,
            prior_terms_step=prior_terms_step,
        ) -> jnp.ndarray:
            return _row_joint_log_prob(
                latent_trajectory,
                observations=observations,
                obs_mask=obs_mask,
                Ad=Ad_step,
                cd=cd_step,
                prior_terms=prior_terms_step,
                H_rows=H_rows,
                d_rows=d_rows,
                R=R,
                obs_kernel=obs_kernel,
            )

        log_joint_prev = _row_log_joint(z_est)
        grads, J_t = _point_linearize(
            observations,
            obs_mask,
            H_rows,
            d_rows,
            R,
            obs_kernel,
            z_est,
        )
        tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)
        lower, diag, upper, rhs = _build_ieks_system_from_prior(
            prior_lower,
            prior_diag,
            prior_upper,
            prior_rhs,
            J_t,
            tilde_y,
        )
        diag = diag + damping * eye[None, :, :]
        z_newton = jnp.asarray(
            _solve_block_tridiagonal(lower, diag, upper, rhs),
            dtype=z_est.dtype,
        )
        step_direction = jnp.asarray(z_newton - z_est, dtype=z_est.dtype)
        step_norm = jnp.asarray(jnp.linalg.norm(step_direction), dtype=z_est.dtype)
        z_next, log_joint_next, accepted, accepted_alpha = _step_halving_search(
            z_est,
            step_direction,
            log_joint_prev,
            _row_log_joint,
            max_halvings=_POINT_LINE_SEARCH_MAX_HALVINGS,
        )
        rel_change = jnp.asarray(
            jnp.linalg.norm(z_next - z_est) / (1.0 + jnp.linalg.norm(z_est)),
            dtype=z_est.dtype,
        )
        damping_shrunk = jnp.maximum(
            damping * jnp.asarray(_POINT_LM_DAMPING_SHRINK, dtype=z_est.dtype),
            jnp.asarray(_POINT_LM_DAMPING_MIN, dtype=z_est.dtype),
        )
        damping_grown = jnp.minimum(
            damping * jnp.asarray(_POINT_LM_DAMPING_GROWTH, dtype=z_est.dtype),
            jnp.asarray(_POINT_LM_DAMPING_MAX, dtype=z_est.dtype),
        )
        accepted_full_step = accepted & (accepted_alpha > 0.999)
        damping_next = jnp.where(
            accepted_full_step,
            damping_shrunk,
            jnp.where(accepted, damping, damping_grown),
        )
        next_active = jnp.where(
            accepted,
            rel_change > _POINT_IEKS_CONVERGENCE_RTOL,
            damping_next < jnp.asarray(_POINT_LM_DAMPING_MAX, dtype=z_est.dtype),
        )

        z_est = jnp.where(active, z_next, z_est)
        log_joint_curr = jnp.where(active, log_joint_next, log_joint_curr)
        damping = jnp.where(active, damping_next, damping)
        n_iterations = n_iterations + active.astype(jnp.int32)
        n_accepted_steps = n_accepted_steps + (active & accepted).astype(jnp.int32)
        final_rel_change = jnp.where(active, rel_change, final_rel_change)
        final_step_alpha = jnp.where(active, accepted_alpha, final_step_alpha)
        final_step_norm = jnp.where(active, step_norm, final_step_norm)
        active = active & next_active

    # The IEKS iterations solve a latent fixed point. The outer parameter
    # gradient should not backpropagate through the discrete line-search path;
    # evaluate the final local-linearized system at the solved mode instead.
    z_mode = jax.lax.stop_gradient(z_est)
    Ad_final, Qd_final, cd_final = _transitions_at(z_mode)
    log_lik, mode_log_joint, laplace_logdet, min_chol_diag = _point_laplace_terms_from_mode(
        z_mode,
        observations,
        obs_mask,
        Ad_final,
        Qd_final,
        cd_final,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
    )
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=solver_kind,
        n_iterations=n_iterations,
        n_accepted_steps=n_accepted_steps,
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        final_rel_change=final_rel_change,
        final_damping=damping,
        final_step_alpha=final_step_alpha,
        final_step_norm=final_step_norm,
        laplace_logdet=laplace_logdet,
        min_chol_diag=min_chol_diag,
    )
    inner_eval_aux["latent_mode"] = z_mode
    return z_mode, log_lik, inner_eval_aux


def _ieks_smooth(
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H_rows,
    d_rows,
    R,
    init_mean,
    init_cov,
    obs_kernel,
    *,
    solver_kind: int = LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    n_ieks_iters=5,
    z_init: jnp.ndarray | None = None,
    build_measurement_objects=None,
    extra_params: LikelihoodExtraParams | None = None,
):
    """Run the Iterated Extended Kalman Smoother to find the MAP state trajectory."""
    return _point_ieks_laplace_core(
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        build_measurement_objects=build_measurement_objects,
        extra_params=extra_params,
        solver_kind=solver_kind,
        n_ieks_iters=n_ieks_iters,
        z_init=z_init,
    )


def _dense_support_laplace_log_lik(
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
    observation_support,
    n_newton_iters: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Dense Laplace approximation for interval-summary observation semantics."""
    T, D = observations.shape[0], init_mean.shape[0]
    flat_dim = T * D

    with jax.named_scope("map/dense_support_init"):
        z_init = _predictive_latent_init(Ad, cd, init_mean)
        prior_terms = build_gaussian_trajectory_prior_terms(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )

    def _joint_log_prob(z_flat):
        z = z_flat.reshape(T, D)
        prior_ll = trajectory_prior_log_prob_from_terms(z, Ad, cd, prior_terms)
        obs_ll = trajectory_observation_log_prob(
            z,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
        return prior_ll + obs_ll

    def _neg_log_prob(z_flat):
        return -_joint_log_prob(z_flat)

    z_flat = z_init.reshape(-1)
    init_log_joint = _joint_log_prob(z_flat)
    with jax.named_scope("map/dense_support_newton"):
        best_z = z_flat
        best_neg = _neg_log_prob(z_flat)
        for _ in range(max(n_newton_iters, 1)):
            grad = jax.grad(_neg_log_prob)(z_flat)
            hess = jax.hessian(_neg_log_prob)(z_flat)
            hess = symmetrize_with_jitter(hess, jitter=1e-4)
            step = jla.solve(hess, grad, assume_a="sym")
            # Backtracking: halve the step until the objective improves or
            # the step is too small.  Prevents the Newton iterate from
            # overshooting into numerically unstable regions.
            alpha = 1.0
            for _bt in range(6):
                z_cand = z_flat - alpha * step
                neg_cand = _neg_log_prob(z_cand)
                improved = jnp.isfinite(neg_cand) & (neg_cand < best_neg + 1.0)
                z_flat = jnp.where(improved, z_cand, z_flat)
                best_neg = jnp.where(improved, neg_cand, best_neg)
                alpha *= 0.5
            best_z = jnp.where(
                jnp.isfinite(best_neg) & (best_neg <= _neg_log_prob(best_z)),
                z_flat,
                best_z,
            )
        z_flat = best_z

    with jax.named_scope("map/dense_support_curvature"):
        mode_log_joint = _joint_log_prob(z_flat)
        hess = jax.hessian(_neg_log_prob)(z_flat)
        hess = symmetrize(hess)
        eigvals = jnp.linalg.eigvalsh(hess)
        min_eig = jnp.min(eigvals)
        logdet = jnp.sum(jnp.log(jnp.maximum(eigvals, 1e-6)))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * logdet
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
        n_iterations=jnp.asarray(max(n_newton_iters, 1), dtype=jnp.int32),
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        final_rel_change=(
            jnp.linalg.norm(z_flat - z_init.reshape(-1))
            / (1.0 + jnp.linalg.norm(z_init.reshape(-1)))
        ),
        laplace_logdet=logdet,
        min_chol_diag=jnp.sqrt(jnp.maximum(min_eig, 0.0)),
    )
    inner_eval_aux["latent_mode"] = z_flat.reshape(T, D)
    return log_lik, inner_eval_aux


def _dense_dynamic_support_laplace_log_lik(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    dynamics: RuntimeDynamics,
    time_intervals: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    mean_log_prob_fn,
    observation_support,
    n_newton_iters: int,
    *,
    transition_inputs: jnp.ndarray | None = None,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Dense interval-support Laplace path with local dynamics linearization."""
    T, D = observations.shape[0], init_mean.shape[0]
    flat_dim = T * D

    def _transitions_at(z_flat_curr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_path = z_flat_curr.reshape(T, D)
        transitions = build_discrete_transitions(
            dynamics,
            time_intervals,
            linearization_states=_transition_start_linearization_states(z_path, init_mean),
            transition_inputs=transition_inputs,
        )
        cd = (
            transitions.cd
            if transitions.cd is not None
            else jnp.zeros((T, D), dtype=observations.dtype)
        )
        cd = jnp.asarray(cd, dtype=observations.dtype)
        if cd.ndim == 1:
            cd = cd[:, None]
        return transitions.Ad, transitions.Qd, cd

    def _joint_log_prob_fixed(
        z_flat_eval: jnp.ndarray,
        Ad: jnp.ndarray,
        Qd: jnp.ndarray,
        cd: jnp.ndarray,
    ) -> jnp.ndarray:
        z = z_flat_eval.reshape(T, D)
        prior_terms = build_gaussian_trajectory_prior_terms(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )
        prior_ll = trajectory_prior_log_prob_from_terms(z, Ad, cd, prior_terms)
        obs_ll = trajectory_observation_log_prob(
            z,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
        return prior_ll + obs_ll

    with jax.named_scope("map/dense_dynamic_support_init"):
        if z_init is None:
            init_path = jnp.broadcast_to(init_mean[None, :], (T, D))
            Ad_init, _Qd_init, cd_init = _transitions_at(init_path.reshape(-1))
            z_flat = _predictive_latent_init(Ad_init, cd_init, init_mean).reshape(-1)
        else:
            z_flat = jnp.asarray(z_init, dtype=observations.dtype).reshape(-1)
        Ad_curr, Qd_curr, cd_curr = _transitions_at(z_flat)
        init_log_joint = _joint_log_prob_fixed(z_flat, Ad_curr, Qd_curr, cd_curr)

    final_rel_change = jnp.asarray(jnp.nan, dtype=z_flat.dtype)
    final_step_alpha = jnp.asarray(jnp.nan, dtype=z_flat.dtype)
    final_step_norm = jnp.asarray(jnp.nan, dtype=z_flat.dtype)
    n_accepted_steps = jnp.asarray(0, dtype=jnp.int32)

    with jax.named_scope("map/dense_dynamic_support_newton"):
        for _ in range(max(n_newton_iters, 1)):
            Ad_step, Qd_step, cd_step = _transitions_at(z_flat)

            def _neg_log_prob_fixed(
                z_flat_eval: jnp.ndarray,
                *,
                Ad_step=Ad_step,
                Qd_step=Qd_step,
                cd_step=cd_step,
            ) -> jnp.ndarray:
                return -_joint_log_prob_fixed(z_flat_eval, Ad_step, Qd_step, cd_step)

            neg_curr = _neg_log_prob_fixed(z_flat)
            grad = jax.grad(_neg_log_prob_fixed)(z_flat)
            hess = jax.hessian(_neg_log_prob_fixed)(z_flat)
            hess = symmetrize_with_jitter(hess, jitter=1e-4)
            step = jla.solve(hess, grad, assume_a="sym")
            step_norm = jnp.asarray(jnp.linalg.norm(step), dtype=z_flat.dtype)

            z_next = z_flat
            neg_next = neg_curr
            accepted = jnp.asarray(False)
            accepted_alpha = jnp.asarray(0.0, dtype=z_flat.dtype)
            alpha = 1.0
            for _bt in range(6):
                alpha_value = jnp.asarray(alpha, dtype=z_flat.dtype)
                z_cand = z_flat - alpha_value * step
                neg_cand = _neg_log_prob_fixed(z_cand)
                improved = jnp.isfinite(neg_cand) & (neg_cand < neg_next)
                first_accept = improved & ~accepted
                z_next = jnp.where(improved, z_cand, z_next)
                neg_next = jnp.where(improved, neg_cand, neg_next)
                accepted_alpha = jnp.where(first_accept, alpha_value, accepted_alpha)
                accepted = accepted | improved
                alpha *= 0.5

            rel_change = jnp.asarray(
                jnp.linalg.norm(z_next - z_flat) / (1.0 + jnp.linalg.norm(z_flat)),
                dtype=z_flat.dtype,
            )
            z_flat = z_next
            final_rel_change = rel_change
            final_step_alpha = accepted_alpha
            final_step_norm = step_norm
            n_accepted_steps = n_accepted_steps + accepted.astype(jnp.int32)

    with jax.named_scope("map/dense_dynamic_support_curvature"):
        Ad_final, Qd_final, cd_final = _transitions_at(z_flat)

        def _final_neg_log_prob_fixed(z_flat_eval: jnp.ndarray) -> jnp.ndarray:
            return -_joint_log_prob_fixed(z_flat_eval, Ad_final, Qd_final, cd_final)

        mode_log_joint = _joint_log_prob_fixed(z_flat, Ad_final, Qd_final, cd_final)
        hess = jax.hessian(_final_neg_log_prob_fixed)(z_flat)
        hess = symmetrize(hess)
        eigvals = jnp.linalg.eigvalsh(hess)
        min_eig = jnp.min(eigvals)
        logdet = jnp.sum(jnp.log(jnp.maximum(eigvals, 1e-6)))

    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * logdet
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
        n_iterations=jnp.asarray(max(n_newton_iters, 1), dtype=jnp.int32),
        n_accepted_steps=n_accepted_steps,
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        final_rel_change=final_rel_change,
        final_step_alpha=final_step_alpha,
        final_step_norm=final_step_norm,
        laplace_logdet=logdet,
        min_chol_diag=jnp.sqrt(jnp.maximum(min_eig, 0.0)),
    )
    inner_eval_aux["latent_mode"] = z_flat.reshape(T, D)
    return log_lik, inner_eval_aux
