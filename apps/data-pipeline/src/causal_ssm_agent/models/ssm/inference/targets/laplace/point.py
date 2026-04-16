"""Point-observation and linear-summary Laplace solvers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import discretize_linear_system_exact_batched
from causal_ssm_agent.models.ssm.inference.targets.base import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    build_likelihood_eval_aux,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    trajectory_observation_log_prob,
)

from .shared import (
    _POINT_IEKS_CONVERGENCE_RTOL,
    _POINT_LINE_SEARCH_MAX_HALVINGS,
    _POINT_LM_DAMPING,
    _POINT_LM_DAMPING_GROWTH,
    _POINT_LM_DAMPING_MAX,
    _POINT_LM_DAMPING_MIN,
    _POINT_LM_DAMPING_SHRINK,
    GaussianTrajectoryPriorTerms,
    LinearSummaryAccumulatorPlan,
    _block_banded_logdet,
    _build_gaussian_trajectory_prior_terms,
    _build_ieks_system_from_prior,
    _build_prior_tridiagonal_system,
    _factor_block_banded_cholesky,
    _predictive_latent_init,
    _solve_block_tridiagonal,
    _step_halving_search,
    _trajectory_prior_log_prob_from_terms,
)


def _build_linear_summary_augmented_system(
    *,
    plan: LinearSummaryAccumulatorPlan,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    support_kind_codes: jnp.ndarray,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Build augmented dynamics and per-row observation operators for linear interval summaries."""
    dtype = jnp.result_type(
        time_intervals,
        drift,
        diffusion_cov,
        cint,
        H,
        d,
        init_mean,
        init_cov,
    )
    time_intervals = jnp.asarray(time_intervals, dtype=dtype)
    drift = jnp.asarray(drift, dtype=dtype)
    diffusion_cov = jnp.asarray(diffusion_cov, dtype=dtype)
    cint = jnp.asarray(cint, dtype=dtype)
    H = jnp.asarray(H, dtype=dtype)
    d = jnp.asarray(d, dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)

    T = int(time_intervals.shape[0])
    n_latent = int(drift.shape[0])
    n_manifest = int(H.shape[0])
    n_accumulators = plan.n_accumulators
    augmented_dim = n_latent + n_accumulators

    drift_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    drift_aug = drift_aug.at[:n_latent, :n_latent].set(drift)
    if n_accumulators > 0:
        drift_aug = drift_aug.at[n_latent:, :n_latent].set(H[plan.accumulator_manifest_indices])

    diffusion_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    diffusion_aug = diffusion_aug.at[:n_latent, :n_latent].set(diffusion_cov)

    cint_aug = jnp.zeros((augmented_dim,), dtype=dtype)
    cint_aug = cint_aug.at[:n_latent].set(cint)
    if n_accumulators > 0:
        cint_aug = cint_aug.at[n_latent:].set(d[plan.accumulator_manifest_indices])

    Ad_aug, Qd_aug, cd_aug = discretize_linear_system_exact_batched(
        drift_aug,
        diffusion_aug,
        cint_aug,
        time_intervals,
    )
    if cd_aug is None:
        cd_aug = jnp.zeros((T, augmented_dim), dtype=dtype)
    else:
        cd_aug = jnp.asarray(cd_aug, dtype=dtype)
        if cd_aug.ndim == 1:
            cd_aug = cd_aug[:, None]

    init_mean_aug = jnp.concatenate(
        [
            init_mean,
            jnp.zeros((n_accumulators,), dtype=dtype),
        ],
        axis=0,
    )
    init_cov_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    init_cov_aug = init_cov_aug.at[:n_latent, :n_latent].set(init_cov)

    H_rows = jnp.zeros((T, n_manifest, augmented_dim), dtype=dtype)
    d_rows = jnp.zeros((T, n_manifest), dtype=dtype)

    point_manifest_indices = np.flatnonzero(np.asarray(support_kind_codes) == 0)
    if point_manifest_indices.size > 0:
        point_idx = jnp.asarray(point_manifest_indices, dtype=jnp.int64)
        H_rows = H_rows.at[:, point_idx, :n_latent].set(
            jnp.broadcast_to(H[point_idx], (T, point_idx.shape[0], n_latent))
        )
        d_rows = d_rows.at[:, point_idx].set(
            jnp.broadcast_to(d[point_idx], (T, point_idx.shape[0]))
        )

    emission_indices = np.asarray(plan.row_emission_accumulator_indices)
    emission_scales = np.asarray(plan.row_emission_scales, dtype=np.float64)
    for time_idx in range(T):
        for manifest_idx in range(n_manifest):
            accumulator_idx = int(emission_indices[time_idx, manifest_idx])
            if accumulator_idx < 0:
                continue
            H_rows = H_rows.at[time_idx, manifest_idx, n_latent + accumulator_idx].set(
                jnp.asarray(emission_scales[time_idx, manifest_idx], dtype=dtype)
            )

    reset_scales = jnp.ones((T, augmented_dim), dtype=dtype)
    if n_accumulators > 0:
        reset_scales = reset_scales.at[:, n_latent:].set(1.0 - plan.row_reset_mask.astype(dtype))
    if T > 1:
        Ad_aug = Ad_aug.at[1:].set(Ad_aug[1:] * reset_scales[:-1, None, :])

    return Ad_aug, Qd_aug, cd_aug, init_mean_aug, init_cov_aug, H_rows, d_rows


def _row_observation_log_prob(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Return the point-observation log-probability for per-row observation operators."""
    obs_mask_float = obs_mask.astype(latent_trajectory.dtype)
    return jnp.sum(
        jax.vmap(
            lambda y_t, z_t, mask_t, H_t, d_t: obs_kernel.emission_fn(
                y_t,
                z_t,
                H_t,
                d_t,
                R,
                mask_t,
            )
        )(observations, latent_trajectory, obs_mask_float, H_rows, d_rows)
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
    return _trajectory_prior_log_prob_from_terms(latent_trajectory, Ad, cd, prior_terms) + (
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
):
    """Run the Iterated Extended Kalman Smoother to find the MAP state trajectory.

    Accepts per-timestep observation operators H_rows (T, n_manifest, D)
    and d_rows (T, n_manifest). For shared H/d, broadcast before calling.

    Returns:
        z_smooth: (T, D) smoothed state means (MAP trajectory)
        log_lik_approx: scalar approximate log-likelihood
        inner_eval_aux: fixed-shape IEKS summary for host-side progress logs
    """
    T = observations.shape[0]
    D = init_mean.shape[0]
    cd_scan = cd if cd is not None else jnp.zeros((T, D))
    obs_mask_float = obs_mask.astype(observations.dtype)
    prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )
    prior_terms = _build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )

    def _emission_grad_hess(y_t, z_t, mask_t, H_t, d_t):
        return obs_kernel.emission_grad_hess_fn(y_t, z_t, H_t, d_t, R, mask_t)

    def _linearize(z_estimate: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        grads_and_hess = jax.vmap(_emission_grad_hess)(
            observations,
            z_estimate,
            obs_mask_float,
            H_rows,
            d_rows,
        )
        return grads_and_hess[0], grads_and_hess[1]

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

    def _newton_step(carry, _idx):
        (
            z_curr,
            log_joint_prev,
            damping,
            active,
            n_iterations,
            n_accepted_steps,
            last_rel_change,
            last_alpha,
            last_step_norm,
        ) = carry
        damping = jnp.asarray(damping, dtype=z_curr.dtype)
        carry_cast = (
            z_curr,
            log_joint_prev,
            damping,
            active,
            n_iterations,
            n_accepted_steps,
            last_rel_change,
            last_alpha,
            last_step_norm,
        )

        def _do_step(_):
            with jax.named_scope("laplace_em/ieks_linearize"):
                grads, J_t = _linearize(z_curr)

            with jax.named_scope("laplace_em/ieks_build_system"):
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

            with jax.named_scope("laplace_em/ieks_solve_system"):
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

        return jax.lax.cond(active, _do_step, lambda _: carry_cast, operand=None), None

    with jax.named_scope("laplace_em/ieks_iterations"):
        (
            (
                z_est,
                mode_log_joint,
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
            xs=jnp.arange(max(n_ieks_iters, 1)),
        )

    with jax.named_scope("laplace_em/ieks_log_likelihood"):
        _final_grads, final_J_t = _linearize(z_est)
        tilde_y_final = jax.vmap(lambda J, z, g: J @ z + g)(final_J_t, z_est, _final_grads)
        _lower, diag_final, upper_final, _rhs = _build_ieks_system_from_prior(
            prior_lower,
            prior_diag,
            prior_upper,
            prior_rhs,
            final_J_t,
            tilde_y_final,
        )
        chol_diag, _lower_factor = _factor_block_banded_cholesky(diag_final, upper_final[None, ...])
        flat_dim = T * D
        laplace_logdet = _block_banded_logdet(chol_diag)
        min_chol_diag = jnp.min(jnp.diagonal(chol_diag, axis1=1, axis2=2))
        log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * laplace_logdet

    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=solver_kind,
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
    inner_eval_aux["latent_mode"] = z_est
    return z_est, log_lik, inner_eval_aux


def _linear_summary_augmented_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel,
    plan: LinearSummaryAccumulatorPlan,
    support_kind_codes: jnp.ndarray,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """IEKS + Laplace path for linear interval summaries via accumulator augmentation."""
    base_cint = (
        jnp.asarray(cint, dtype=drift.dtype)
        if cint is not None
        else jnp.zeros((drift.shape[0],), dtype=drift.dtype)
    )
    (
        Ad_aug,
        Qd_aug,
        cd_aug,
        init_mean_aug,
        init_cov_aug,
        H_rows,
        d_rows,
    ) = _build_linear_summary_augmented_system(
        plan=plan,
        time_intervals=time_intervals,
        drift=drift,
        diffusion_cov=diffusion_cov,
        cint=base_cint,
        H=H,
        d=d,
        init_mean=init_mean,
        init_cov=init_cov,
        support_kind_codes=support_kind_codes,
    )
    return _ieks_smooth(
        observations,
        obs_mask,
        Ad_aug,
        Qd_aug,
        cd_aug,
        H_rows,
        d_rows,
        R,
        init_mean_aug,
        init_cov_aug,
        obs_kernel,
        solver_kind=LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
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

    with jax.named_scope("laplace_em/dense_support_init"):
        z_init = _predictive_latent_init(Ad, cd, init_mean)
        prior_terms = _build_gaussian_trajectory_prior_terms(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )

    def _joint_log_prob(z_flat):
        z = z_flat.reshape(T, D)
        prior_ll = _trajectory_prior_log_prob_from_terms(z, Ad, cd, prior_terms)
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
    with jax.named_scope("laplace_em/dense_support_newton"):
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

    with jax.named_scope("laplace_em/dense_support_curvature"):
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
