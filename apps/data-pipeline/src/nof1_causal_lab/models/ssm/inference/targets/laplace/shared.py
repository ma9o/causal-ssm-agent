"""Shared linear algebra and preprocessing helpers for MAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.core
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    get_support_kind_codes,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

_DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM = 160

_SUPPORT_AWARE_IEKS_CONVERGENCE_RTOL = 1e-3

_SUPPORT_AWARE_LM_DAMPING = 1e-3

_SUPPORT_AWARE_LM_DAMPING_MIN = 1e-6

_SUPPORT_AWARE_LM_DAMPING_MAX = 1e6

_SUPPORT_AWARE_LM_DAMPING_GROWTH = 10.0

_SUPPORT_AWARE_LM_DAMPING_SHRINK = 0.5

_SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS = 6

_POINT_IEKS_CONVERGENCE_RTOL = _SUPPORT_AWARE_IEKS_CONVERGENCE_RTOL

_POINT_LM_DAMPING = _SUPPORT_AWARE_LM_DAMPING

_POINT_LM_DAMPING_MIN = _SUPPORT_AWARE_LM_DAMPING_MIN

_POINT_LM_DAMPING_MAX = _SUPPORT_AWARE_LM_DAMPING_MAX

_POINT_LM_DAMPING_GROWTH = _SUPPORT_AWARE_LM_DAMPING_GROWTH

_POINT_LM_DAMPING_SHRINK = _SUPPORT_AWARE_LM_DAMPING_SHRINK

_POINT_LINE_SEARCH_MAX_HALVINGS = _SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS


@dataclass(frozen=True)
class SupportObservationWindowBatch:
    """Compiled anchored interval-summary windows padded to a common state length."""

    max_state_len: int
    state_lens: jnp.ndarray
    anchor_indices: jnp.ndarray
    start_indices: jnp.ndarray
    mask_full: jnp.ndarray
    prev_coeffs: jnp.ndarray
    curr_coeffs: jnp.ndarray
    weights: jnp.ndarray
    padded_state_indices: jnp.ndarray
    time_indices: jnp.ndarray
    valid_diag: jnp.ndarray
    cross_time_indices: jnp.ndarray
    valid_cross: jnp.ndarray


@dataclass(frozen=True)
class GaussianTrajectoryPriorTerms:
    """Precomputed Gaussian factors for the latent trajectory prior."""

    init_mean: jnp.ndarray
    init_chol: jnp.ndarray
    init_logdet: jnp.ndarray
    transition_chol: jnp.ndarray
    transition_logdet: jnp.ndarray


def _should_use_dense_support_laplace(*, n_time: int, n_latent: int) -> bool:
    """Use the dense exact support-aware Newton system on short trajectories.

    The banded support-aware path pays substantial Python/autodiff overhead per
    anchored window. For small latent trajectories, a single dense exact Hessian
    over the full latent path is materially faster and preserves the same model.
    """
    return n_time * n_latent <= _DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM


def _symmetrize_psd(mats: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize square matrices and optionally add diagonal jitter."""
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)
    return 0.5 * (mats + jnp.swapaxes(mats, -1, -2)) + jitter * eye


def _predictive_latent_init(
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
) -> jnp.ndarray:
    """Deterministic latent rollout under the mean dynamics."""
    cd = _coerce_transition_intercepts(
        cd,
        state_dim=int(Ad.shape[1]),
        dtype=jnp.result_type(Ad, cd, init_mean),
    )
    T = Ad.shape[0]
    z0 = Ad[0] @ init_mean + cd[0]
    if T == 1:
        return z0[None]

    def _step(z_prev, inputs):
        Ad_t, cd_t = inputs
        z_t = Ad_t @ z_prev + cd_t
        return z_t, z_t

    _, z_rest = jax.lax.scan(_step, z0, (Ad[1:], cd[1:]))
    return jnp.concatenate([z0[None], z_rest], axis=0)


def _batched_spd_solve(mats: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve a batch of SPD linear systems with matching right-hand sides."""
    return jax.vmap(lambda mat, b: jla.solve(mat, b, assume_a="pos"))(mats, rhs)


def _coerce_transition_intercepts(
    cd: jnp.ndarray,
    *,
    state_dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Normalize transition intercepts to shape (T, D)."""
    cd = jnp.asarray(cd, dtype=dtype)
    if cd.ndim == 1:
        if state_dim != 1:
            raise ValueError(
                "Transition intercepts must have shape (T, D) when the latent state "
                f"dimension is {state_dim}."
            )
        return cd[:, None]
    return cd


def _solve_spd_from_cholesky(chol: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve A x = rhs given a lower-triangular Cholesky factor A = L L^T."""
    y = jla.solve_triangular(chol, rhs, lower=True)
    return jla.solve_triangular(chol.T, y, lower=False)


def _logdet_from_cholesky(chol: jnp.ndarray) -> jnp.ndarray:
    """Log determinant from a lower-triangular Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(chol), 1e-12)))


def _gaussian_log_prob_from_cholesky(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    chol: jnp.ndarray,
    logdet: jnp.ndarray,
) -> jnp.ndarray:
    """Exact Gaussian log density using a precomputed Cholesky factor."""
    diff = value - mean
    whitened = jla.solve_triangular(chol, diff, lower=True)
    dim_term = value.shape[-1] * jnp.log(2.0 * jnp.pi)
    return -0.5 * (dim_term + logdet + whitened @ whitened)


def build_gaussian_trajectory_prior_terms(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> GaussianTrajectoryPriorTerms:
    """Precompute Gaussian factors for repeated latent-prior evaluations."""
    cd = _coerce_transition_intercepts(
        cd,
        state_dim=int(Ad.shape[1]),
        dtype=jnp.result_type(Ad, Qd, cd, init_mean, init_cov),
    )
    T = Ad.shape[0]
    init_pred_mean = Ad[0] @ init_mean + cd[0]
    init_pred_cov = symmetrize_with_jitter(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=jitter)
    init_chol = jnp.linalg.cholesky(init_pred_cov)
    init_logdet = _logdet_from_cholesky(init_chol)

    if T == 1:
        return GaussianTrajectoryPriorTerms(
            init_mean=init_pred_mean,
            init_chol=init_chol,
            init_logdet=init_logdet,
            transition_chol=jnp.zeros(
                (0, init_cov.shape[0], init_cov.shape[1]), dtype=init_cov.dtype
            ),
            transition_logdet=jnp.zeros((0,), dtype=init_cov.dtype),
        )

    transition_cov = symmetrize_with_jitter(Qd[1:], jitter=jitter)
    transition_chol = jax.vmap(jnp.linalg.cholesky)(transition_cov)
    transition_logdet = jax.vmap(_logdet_from_cholesky)(transition_chol)
    return GaussianTrajectoryPriorTerms(
        init_mean=init_pred_mean,
        init_chol=init_chol,
        init_logdet=init_logdet,
        transition_chol=transition_chol,
        transition_logdet=transition_logdet,
    )


def trajectory_prior_log_prob_from_terms(
    latent_trajectory: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
) -> jnp.ndarray:
    """Return log p(z_{1:T}) using precomputed Gaussian factors."""
    cd = _coerce_transition_intercepts(
        cd,
        state_dim=int(Ad.shape[1]),
        dtype=jnp.result_type(latent_trajectory, Ad, cd),
    )
    init_ll = _gaussian_log_prob_from_cholesky(
        latent_trajectory[0],
        prior_terms.init_mean,
        prior_terms.init_chol,
        prior_terms.init_logdet,
    )
    if latent_trajectory.shape[0] == 1:
        return init_ll

    transition_means = jax.vmap(lambda Ad_t, z_tm1, cd_t: Ad_t @ z_tm1 + cd_t)(
        Ad[1:],
        latent_trajectory[:-1],
        cd[1:],
    )
    transition_ll = jax.vmap(_gaussian_log_prob_from_cholesky)(
        latent_trajectory[1:],
        transition_means,
        prior_terms.transition_chol,
        prior_terms.transition_logdet,
    )
    return init_ll + jnp.sum(transition_ll)


def _build_prior_tridiagonal_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the latent-prior contribution for the IEKS tridiagonal system."""
    dtype = jnp.result_type(Ad, Qd, cd, init_mean, init_cov)
    Ad = jnp.asarray(Ad, dtype=dtype)
    Qd = jnp.asarray(Qd, dtype=dtype)
    cd = _coerce_transition_intercepts(cd, state_dim=int(Ad.shape[1]), dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)
    T, D = Ad.shape[:2]
    eye = jnp.eye(D, dtype=dtype)

    diag_blocks = jnp.zeros((T, D, D), dtype=dtype)
    rhs = jnp.zeros((T, D), dtype=dtype)
    lower = jnp.zeros((T, D, D), dtype=dtype)
    upper = jnp.zeros((T, D, D), dtype=dtype)

    prior_mean = Ad[0] @ init_mean + cd[0]
    prior_cov = _symmetrize_psd(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=jitter)
    prior_inv = jla.solve(prior_cov, eye, assume_a="pos")

    diag_blocks = diag_blocks.at[0].add(prior_inv)
    rhs = rhs.at[0].add(prior_inv @ prior_mean)

    if T == 1:
        return lower, _symmetrize_psd(diag_blocks, jitter=jitter), upper, rhs

    q_reg = _symmetrize_psd(Qd[1:], jitter=jitter)
    eye_batch = jnp.broadcast_to(eye, q_reg.shape)
    q_inv = _batched_spd_solve(q_reg, eye_batch)
    q_inv_a = _batched_spd_solve(q_reg, Ad[1:])
    q_inv_c = _batched_spd_solve(q_reg, cd[1:])

    lower = lower.at[1:].set(-q_inv_a)
    upper = upper.at[:-1].set(-jnp.swapaxes(q_inv_a, -1, -2))
    diag_blocks = diag_blocks.at[1:].add(q_inv)
    diag_blocks = diag_blocks.at[:-1].add(jnp.swapaxes(Ad[1:], -1, -2) @ q_inv_a)
    rhs = rhs.at[1:].add(q_inv_c)
    rhs = rhs.at[:-1].add(-jnp.einsum("tij,tj->ti", jnp.swapaxes(Ad[1:], -1, -2), q_inv_c))

    return lower, _symmetrize_psd(diag_blocks, jitter=jitter), upper, rhs


def _build_ieks_system_from_prior(
    prior_lower: jnp.ndarray,
    prior_diag: jnp.ndarray,
    prior_upper: jnp.ndarray,
    prior_rhs: jnp.ndarray,
    J_t: jnp.ndarray,
    tilde_y: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble IEKS normal equations from a precomputed prior system."""
    diag_blocks = prior_diag + J_t
    rhs = prior_rhs + tilde_y
    return prior_lower, _symmetrize_psd(diag_blocks, jitter=jitter), prior_upper, rhs


def _solve_block_tridiagonal(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a block-tridiagonal linear system via block Thomas elimination."""
    n = diag.shape[0]
    if n == 1:
        base_diag = _symmetrize_psd(diag[0], jitter=1e-6)
        return jla.solve(base_diag, rhs[0], assume_a="pos")[None]

    diag0 = _symmetrize_psd(diag[0], jitter=1e-6)
    chol0 = jnp.linalg.cholesky(diag0)
    rhs0 = rhs[0]

    def _forward_step(carry, inputs):
        chol_prev, rhs_prev = carry
        lower_i, diag_i, upper_prev, rhs_i = inputs
        solve_prev_upper = _solve_spd_from_cholesky(chol_prev, upper_prev)
        solve_prev_rhs = _solve_spd_from_cholesky(chol_prev, rhs_prev)
        schur = diag_i - lower_i @ solve_prev_upper
        rhs_tilde_i = rhs_i - lower_i @ solve_prev_rhs
        chol_i = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=1e-6))
        return (chol_i, rhs_tilde_i), (chol_i, rhs_tilde_i)

    (_, _), (chol_rest, rhs_rest) = jax.lax.scan(
        _forward_step,
        (chol0, rhs0),
        (lower[1:], diag[1:], upper[:-1], rhs[1:]),
    )
    chol_diag = jnp.concatenate([chol0[None], chol_rest], axis=0)
    rhs_tilde = jnp.concatenate([rhs0[None], rhs_rest], axis=0)

    x_last = _solve_spd_from_cholesky(chol_diag[-1], rhs_tilde[-1])

    def _backward_step(x_next, inputs):
        chol_i, upper_i, rhs_i = inputs
        rhs_eff = rhs_i - upper_i @ x_next
        x_i = _solve_spd_from_cholesky(chol_i, rhs_eff)
        return x_i, x_i

    _, x_rest = jax.lax.scan(
        _backward_step,
        x_last,
        (chol_diag[:-1], upper[:-1], rhs_tilde[:-1]),
        reverse=True,
    )
    return jnp.concatenate([x_rest, x_last[None]], axis=0)


def _build_prior_banded_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    bandwidth: int,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the Gaussian latent-prior contribution in block-banded form."""
    dtype = jnp.result_type(Ad, Qd, cd, init_mean, init_cov)
    Ad = jnp.asarray(Ad, dtype=dtype)
    Qd = jnp.asarray(Qd, dtype=dtype)
    cd = _coerce_transition_intercepts(cd, state_dim=int(Ad.shape[1]), dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)
    T, D = Ad.shape[:2]
    eye = jnp.eye(D, dtype=dtype)

    diag = jnp.zeros((T, D, D), dtype=dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=dtype)
    rhs = jnp.zeros((T, D), dtype=dtype)

    prior_mean = Ad[0] @ init_mean + cd[0]
    prior_cov = _symmetrize_psd(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=jitter)
    prior_inv = jla.solve(prior_cov, eye, assume_a="pos")

    diag = diag.at[0].add(prior_inv)
    rhs = rhs.at[0].add(prior_inv @ prior_mean)

    if T == 1:
        return diag, upper, rhs

    q_reg = _symmetrize_psd(Qd[1:], jitter=jitter)
    eye_batch = jnp.broadcast_to(eye, q_reg.shape)
    q_inv = _batched_spd_solve(q_reg, eye_batch)
    q_inv_a = _batched_spd_solve(q_reg, Ad[1:])
    q_inv_c = _batched_spd_solve(q_reg, cd[1:])

    diag = diag.at[1:].add(q_inv)
    diag = diag.at[:-1].add(jnp.swapaxes(Ad[1:], -1, -2) @ q_inv_a)
    rhs = rhs.at[1:].add(q_inv_c)
    rhs = rhs.at[:-1].add(-jnp.einsum("tij,tj->ti", jnp.swapaxes(Ad[1:], -1, -2), q_inv_c))

    if bandwidth >= 1:
        upper = upper.at[0, :-1].set(-jnp.swapaxes(q_inv_a, -1, -2))

    return diag, upper, rhs


def _factor_block_banded_cholesky(
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray | None = None,
    row_lower_bandwidths: jnp.ndarray | None = None,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Block-banded Cholesky factorization A = L L^T."""
    T, _D = diag.shape[:2]
    bandwidth = upper.shape[0]
    if row_upper_bandwidths is None:
        row_upper_bandwidths = jnp.full((T,), bandwidth, dtype=jnp.int32)
    if row_lower_bandwidths is None:
        row_lower_bandwidths = jnp.full((T,), bandwidth, dtype=jnp.int32)
    chol_diag = jnp.zeros_like(diag)
    lower = jnp.zeros_like(upper)

    def _factor_step(i, state):
        chol_diag_state, lower_state = state
        upper_bw_i = row_upper_bandwidths[i]
        lower_bw_i = row_lower_bandwidths[i]

        def _schur_offset(offset, schur):
            return jax.lax.cond(
                (i >= offset) & (offset <= lower_bw_i),
                lambda s: s - lower_state[offset - 1, i] @ lower_state[offset - 1, i].T,
                lambda s: s,
                schur,
            )

        schur = jax.lax.fori_loop(1, bandwidth + 1, _schur_offset, diag[i])
        l_ii = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=jitter))
        chol_diag_state = chol_diag_state.at[i].set(l_ii)

        def _update_lower_for_offset(offset_j, lower_curr):
            def _compute(curr_lower):
                lower_bw_j = row_lower_bandwidths[i + offset_j]

                def _cross_update(offset_k, schur_off):
                    cross_offset = offset_j + offset_k - 1
                    return jax.lax.cond(
                        (i >= offset_k)
                        & (offset_k <= lower_bw_i)
                        & (cross_offset < bandwidth)
                        & (cross_offset < lower_bw_j),
                        lambda s: (
                            s
                            - curr_lower[cross_offset, i + offset_j] @ curr_lower[offset_k - 1, i].T
                        ),
                        lambda s: s,
                        schur_off,
                    )

                schur_off = jax.lax.fori_loop(
                    1, bandwidth + 1, _cross_update, upper[offset_j - 1, i].T
                )
                l_ji = jla.solve_triangular(l_ii, schur_off.T, lower=True).T
                return curr_lower.at[offset_j - 1, i + offset_j].set(l_ji)

            return jax.lax.cond(
                (i + offset_j < T) & (offset_j <= upper_bw_i),
                _compute,
                lambda x: x,
                lower_curr,
            )

        lower_state = jax.lax.fori_loop(1, bandwidth + 1, _update_lower_for_offset, lower_state)
        return chol_diag_state, lower_state

    return jax.lax.fori_loop(0, T, _factor_step, (chol_diag, lower))


def _solve_block_banded_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    rhs: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray | None = None,
    row_lower_bandwidths: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Solve A x = rhs from block-banded Cholesky factors."""
    T = rhs.shape[0]
    bandwidth = lower.shape[0]
    if row_upper_bandwidths is None:
        row_upper_bandwidths = jnp.full((T,), bandwidth, dtype=jnp.int32)
    if row_lower_bandwidths is None:
        row_lower_bandwidths = jnp.full((T,), bandwidth, dtype=jnp.int32)
    y = jnp.zeros_like(rhs)

    def _forward_step(i, y_state):
        lower_bw_i = row_lower_bandwidths[i]

        def _forward_offset(offset, res):
            return jax.lax.cond(
                (i >= offset) & (offset <= lower_bw_i),
                lambda r: r - lower[offset - 1, i] @ y_state[i - offset],
                lambda r: r,
                res,
            )

        res = jax.lax.fori_loop(1, bandwidth + 1, _forward_offset, rhs[i])
        y_i = jla.solve_triangular(chol_diag[i], res, lower=True)
        return y_state.at[i].set(y_i)

    y = jax.lax.fori_loop(0, T, _forward_step, y)
    x = jnp.zeros_like(rhs)

    def _backward_step(rev_idx, x_state):
        i = T - 1 - rev_idx
        upper_bw_i = row_upper_bandwidths[i]

        def _backward_offset(offset, res):
            return jax.lax.cond(
                (i + offset < T) & (offset <= upper_bw_i),
                lambda r: r - lower[offset - 1, i + offset].T @ x_state[i + offset],
                lambda r: r,
                res,
            )

        res = jax.lax.fori_loop(1, bandwidth + 1, _backward_offset, y[i])
        x_i = jla.solve_triangular(chol_diag[i].T, res, lower=False)
        return x_state.at[i].set(x_i)

    return jax.lax.fori_loop(0, T, _backward_step, x)


def _block_banded_logdet(chol_diag: jnp.ndarray) -> jnp.ndarray:
    """Log determinant from block-banded Cholesky factors."""
    return 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diagonal(chol_diag, axis1=1, axis2=2), 1e-12)))


def _compute_profile_lower_bandwidths(row_upper_bandwidths: np.ndarray) -> np.ndarray:
    """Return the realized lower profile widths implied by symmetric upper widths."""
    T = int(row_upper_bandwidths.shape[0])
    max_bandwidth = int(np.max(row_upper_bandwidths, initial=0))
    row_lower_bandwidths = np.zeros((T,), dtype=np.int64)
    for row_idx in range(T):
        max_offset = min(row_idx, max_bandwidth)
        lower_bandwidth = 0
        for offset in range(1, max_offset + 1):
            if row_upper_bandwidths[row_idx - offset] >= offset:
                lower_bandwidth = offset
        row_lower_bandwidths[row_idx] = lower_bandwidth
    return row_lower_bandwidths


def _factor_block_profile_cholesky(
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact block-profile Cholesky factorization A = L L^T."""
    T, _D = diag.shape[:2]
    chol_diag = jnp.zeros_like(diag)
    lower = jnp.zeros_like(upper)

    def _factor_step(i, state):
        chol_diag_state, lower_state = state
        lower_bw_i = row_lower_bandwidths[i]
        upper_bw_i = row_upper_bandwidths[i]

        def _schur_cond(loop_state):
            offset, _schur = loop_state
            return offset <= lower_bw_i

        def _schur_body(loop_state):
            offset, schur = loop_state
            schur = schur - lower_state[offset - 1, i] @ lower_state[offset - 1, i].T
            return offset + 1, schur

        _offset_final, schur = jax.lax.while_loop(_schur_cond, _schur_body, (1, diag[i]))
        l_ii = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=jitter))
        chol_diag_state = chol_diag_state.at[i].set(l_ii)

        def _future_cond(loop_state):
            offset_j, _lower_curr = loop_state
            return offset_j <= upper_bw_i

        def _future_body(loop_state):
            offset_j, lower_curr = loop_state
            row_j = i + offset_j
            lower_bw_j = row_lower_bandwidths[row_j]

            def _cross_cond(cross_state):
                offset_k, _schur_off = cross_state
                return offset_k <= lower_bw_i

            def _cross_body(cross_state):
                offset_k, schur_off = cross_state
                cross_idx = offset_j + offset_k - 1
                schur_off = jax.lax.cond(
                    cross_idx < lower_bw_j,
                    lambda s: s - lower_curr[cross_idx, row_j] @ lower_curr[offset_k - 1, i].T,
                    lambda s: s,
                    schur_off,
                )
                return offset_k + 1, schur_off

            _cross_done, schur_off = jax.lax.while_loop(
                _cross_cond,
                _cross_body,
                (1, upper[offset_j - 1, i].T),
            )
            del _cross_done
            l_ji = jla.solve_triangular(l_ii, schur_off.T, lower=True).T
            lower_curr = lower_curr.at[offset_j - 1, row_j].set(l_ji)
            return offset_j + 1, lower_curr

        _future_done, lower_state = jax.lax.while_loop(
            _future_cond,
            _future_body,
            (1, lower_state),
        )
        del _future_done
        return chol_diag_state, lower_state

    return jax.lax.fori_loop(0, T, _factor_step, (chol_diag, lower))


def _solve_block_profile_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    rhs: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
) -> jnp.ndarray:
    """Solve A x = rhs from exact block-profile Cholesky factors."""
    T = rhs.shape[0]
    y = jnp.zeros_like(rhs)

    def _forward_step(i, y_state):
        lower_bw_i = row_lower_bandwidths[i]

        def _forward_cond(loop_state):
            offset, _res = loop_state
            return offset <= lower_bw_i

        def _forward_body(loop_state):
            offset, res = loop_state
            res = res - lower[offset - 1, i] @ y_state[i - offset]
            return offset + 1, res

        _offset_done, res = jax.lax.while_loop(_forward_cond, _forward_body, (1, rhs[i]))
        del _offset_done
        y_i = jla.solve_triangular(chol_diag[i], res, lower=True)
        return y_state.at[i].set(y_i)

    y = jax.lax.fori_loop(0, T, _forward_step, y)
    x = jnp.zeros_like(rhs)

    def _backward_step(rev_idx, x_state):
        i = T - 1 - rev_idx
        upper_bw_i = row_upper_bandwidths[i]

        def _backward_cond(loop_state):
            offset, _res = loop_state
            return offset <= upper_bw_i

        def _backward_body(loop_state):
            offset, res = loop_state
            res = res - lower[offset - 1, i + offset].T @ x_state[i + offset]
            return offset + 1, res

        _offset_done, res = jax.lax.while_loop(_backward_cond, _backward_body, (1, y[i]))
        del _offset_done
        x_i = jla.solve_triangular(chol_diag[i].T, res, lower=False)
        return x_state.at[i].set(x_i)

    return jax.lax.fori_loop(0, T, _backward_step, x)


def _selected_inverse_block(
    inv_diag: jnp.ndarray,
    inv_upper: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    i: int,
    j: int,
) -> jnp.ndarray:
    """Return block (i, j) from the packed inverse subset."""
    zero = jnp.zeros_like(inv_diag[0])

    def _diag_branch(_):
        return inv_diag[i]

    def _offdiag_branch(_):
        def _upper_branch(_):
            offset = j - i
            return jax.lax.cond(
                offset <= row_upper_bandwidths[i],
                lambda _: inv_upper[offset - 1, i],
                lambda _: zero,
                operand=None,
            )

        def _lower_branch(_):
            offset = i - j
            return jax.lax.cond(
                offset <= row_upper_bandwidths[j],
                lambda _: jnp.swapaxes(inv_upper[offset - 1, j], -1, -2),
                lambda _: zero,
                operand=None,
            )

        return jax.lax.cond(j > i, _upper_branch, _lower_branch, operand=None)

    return jax.lax.cond(i == j, _diag_branch, _offdiag_branch, operand=None)


def _block_profile_inverse_subset_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    _row_lower_bandwidths: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return diagonal and upper-profile blocks of A^{-1} from A = L L^T."""
    t_steps, block_dim = chol_diag.shape[:2]
    max_bandwidth = lower.shape[0]
    eye = jnp.eye(block_dim, dtype=chol_diag.dtype)
    inv_diag = jnp.zeros_like(chol_diag)
    inv_upper = jnp.zeros_like(lower)

    def _row_step(rev_i, state):
        inv_diag_state, inv_upper_state = state
        i = t_steps - 1 - rev_i
        l_ii = chol_diag[i]
        upper_bw_i = row_upper_bandwidths[i]

        def _offdiag_step(offset_j_zero, inv_upper_curr):
            offset_j = offset_j_zero + 1

            def _compute(curr):
                row_j = i + offset_j
                zero = jnp.zeros((block_dim, block_dim), dtype=chol_diag.dtype)

                def _sum_step(offset_k_zero, acc):
                    offset_k = offset_k_zero + 1

                    def _accumulate(a):
                        row_k = i + offset_k
                        l_ki = lower[offset_k - 1, row_k]
                        s_kj = _selected_inverse_block(
                            inv_diag_state,
                            curr,
                            row_upper_bandwidths,
                            row_k,
                            row_j,
                        )
                        return a + l_ki.T @ s_kj

                    return jax.lax.cond(
                        offset_k <= upper_bw_i,
                        _accumulate,
                        lambda a: a,
                        acc,
                    )

                schur_term = jax.lax.fori_loop(0, max_bandwidth, _sum_step, zero)
                s_ij = -jla.solve_triangular(l_ii.T, schur_term, lower=False)
                return curr.at[offset_j - 1, i].set(s_ij)

            return jax.lax.cond(
                offset_j <= upper_bw_i,
                _compute,
                lambda curr: curr,
                inv_upper_curr,
            )

        inv_upper_state = jax.lax.fori_loop(0, max_bandwidth, _offdiag_step, inv_upper_state)
        inv_l_ii = jla.solve_triangular(l_ii, eye, lower=True)
        diag_base = inv_l_ii.T @ inv_l_ii

        def _diag_sum_step(offset_k_zero, acc):
            offset_k = offset_k_zero + 1

            def _accumulate(a):
                row_k = i + offset_k
                l_ki = lower[offset_k - 1, row_k]
                s_ki = _selected_inverse_block(
                    inv_diag_state,
                    inv_upper_state,
                    row_upper_bandwidths,
                    row_k,
                    i,
                )
                return a + l_ki.T @ s_ki

            return jax.lax.cond(offset_k <= upper_bw_i, _accumulate, lambda a: a, acc)

        diag_schur = jax.lax.fori_loop(
            0,
            max_bandwidth,
            _diag_sum_step,
            jnp.zeros((block_dim, block_dim), dtype=chol_diag.dtype),
        )
        diag_i = diag_base - jla.solve_triangular(l_ii.T, diag_schur, lower=False)
        diag_i = 0.5 * (diag_i + diag_i.T)
        inv_diag_state = inv_diag_state.at[i].set(diag_i)
        return inv_diag_state, inv_upper_state

    return jax.lax.fori_loop(0, t_steps, _row_step, (inv_diag, inv_upper))


def block_profile_logdet_packed_cotangent(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    *,
    scale: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return cotangents for diag and packed upper blocks of log|A|."""
    inv_diag, inv_upper = _block_profile_inverse_subset_from_cholesky(
        chol_diag,
        lower,
        row_upper_bandwidths,
        row_lower_bandwidths,
    )
    return scale * inv_diag, 2.0 * scale * inv_upper


def _infer_support_groups(
    observation_support: ObservationSupportRuntime,
) -> tuple[tuple[SupportObservationWindowBatch, ...], int, jnp.ndarray]:
    """Compile anchored non-point observation windows into coarse exact-preserving buckets."""
    anchor_times = np.asarray(observation_support.anchor_times)
    support_kind_codes = np.asarray(get_support_kind_codes(observation_support))
    support_start_times = np.asarray(observation_support.support_start_times)
    prev_coeffs = np.asarray(observation_support.interval_prev_coeffs)
    curr_coeffs = np.asarray(observation_support.interval_curr_coeffs)
    weights = np.asarray(observation_support.interval_weights)
    emission_slots = np.asarray(observation_support.emission_slot_indices)
    T, n_manifest = emission_slots.shape

    compiled_windows: list[
        tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    max_bandwidth = 1 if T > 1 else 0
    row_upper_bandwidths = np.zeros((T,), dtype=np.int64)
    for anchor_idx in range(T):
        manifests = [
            manifest_idx
            for manifest_idx in range(n_manifest)
            if support_kind_codes[manifest_idx] == 1
            and emission_slots[anchor_idx, manifest_idx] >= 0
        ]
        if not manifests:
            continue

        manifest_windows: list[tuple[int, int, int]] = []
        start_idx = anchor_idx
        for manifest_idx in manifests:
            slot_idx = int(emission_slots[anchor_idx, manifest_idx])
            support_start = float(support_start_times[anchor_idx, manifest_idx])
            if not np.isfinite(support_start):
                raise ValueError(
                    "Support-aware Laplace requires finite support_start metadata "
                    f"for emitted interval observation manifest={manifest_idx} anchor_idx={anchor_idx}."
                )
            local_start_idx = int(np.searchsorted(anchor_times, support_start, side="right") - 1)
            local_start_idx = max(local_start_idx, 0)
            start_idx = min(start_idx, local_start_idx)
            manifest_windows.append((manifest_idx, slot_idx, local_start_idx))

        start_idx = max(start_idx, 0)
        max_bandwidth = max(max_bandwidth, anchor_idx - start_idx)

        mask_full = np.zeros((n_manifest,), dtype=np.float64)
        mask_full[manifests] = 1.0
        segment_len = anchor_idx - start_idx
        group_prev = np.zeros((segment_len, n_manifest), dtype=np.float64)
        group_curr = np.zeros((segment_len, n_manifest), dtype=np.float64)
        group_weights = np.zeros((segment_len, n_manifest), dtype=np.float64)
        for manifest_idx, slot_idx, local_start_idx in manifest_windows:
            offset = local_start_idx - start_idx
            local_segment_len = anchor_idx - local_start_idx
            if local_segment_len <= 0:
                continue
            group_prev[offset : offset + local_segment_len, manifest_idx] = prev_coeffs[
                local_start_idx + 1 : anchor_idx + 1,
                manifest_idx,
                slot_idx,
            ]
            group_curr[offset : offset + local_segment_len, manifest_idx] = curr_coeffs[
                local_start_idx + 1 : anchor_idx + 1,
                manifest_idx,
                slot_idx,
            ]
            group_weights[offset : offset + local_segment_len, manifest_idx] = weights[
                local_start_idx + 1 : anchor_idx + 1,
                manifest_idx,
                slot_idx,
            ]
        state_len = anchor_idx - start_idx + 1
        for row_idx in range(start_idx, anchor_idx):
            row_upper_bandwidths[row_idx] = max(row_upper_bandwidths[row_idx], anchor_idx - row_idx)
        compiled_windows.append(
            (
                state_len,
                anchor_idx,
                start_idx,
                mask_full,
                group_prev,
                group_curr,
                group_weights,
            )
        )

    def _support_bucket_state_len(state_len: int) -> int:
        return 1 if state_len <= 1 else 1 << (state_len - 1).bit_length()

    def _compile_support_batch(
        batch_state_len: int,
        windows_for_state_len: list[
            tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> SupportObservationWindowBatch:
        n_windows = len(windows_for_state_len)
        max_segment_len = batch_state_len - 1
        local_positions = np.arange(batch_state_len, dtype=np.int64)
        batch_bandwidth = max(batch_state_len - 1, 0)

        state_lens = np.zeros((n_windows,), dtype=np.int64)
        anchor_indices = np.zeros((n_windows,), dtype=np.int64)
        start_indices = np.zeros((n_windows,), dtype=np.int64)
        mask_full = np.zeros((n_windows, n_manifest), dtype=np.float64)
        padded_prev = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_curr = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_weights = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_state_indices = np.zeros((n_windows, batch_state_len), dtype=np.int64)
        time_indices = np.zeros((n_windows, batch_state_len), dtype=np.int64)
        valid_diag = np.zeros((n_windows, batch_state_len), dtype=bool)
        cross_time_indices = np.zeros(
            (batch_bandwidth, n_windows, batch_state_len),
            dtype=np.int64,
        )
        valid_cross = np.zeros((batch_bandwidth, n_windows, batch_state_len), dtype=bool)

        for window_idx, (
            state_len,
            anchor_idx,
            start_idx,
            mask_window,
            prev_window,
            curr_window,
            weights_window,
        ) in enumerate(windows_for_state_len):
            segment_len = state_len - 1
            state_lens[window_idx] = state_len
            anchor_indices[window_idx] = anchor_idx
            start_indices[window_idx] = start_idx
            mask_full[window_idx] = mask_window
            if segment_len > 0:
                padded_prev[window_idx, :segment_len] = prev_window
                padded_curr[window_idx, :segment_len] = curr_window
                padded_weights[window_idx, :segment_len] = weights_window

            raw_positions = start_idx + local_positions
            padded_state_indices[window_idx] = raw_positions
            time_indices[window_idx] = np.clip(raw_positions, 0, T - 1)
            valid_diag[window_idx, :state_len] = True

            for offset in range(1, batch_bandwidth + 1):
                cross_len = batch_state_len - offset
                cross_time_indices[offset - 1, window_idx, :cross_len] = np.clip(
                    raw_positions[:cross_len],
                    0,
                    T - 1,
                )
                valid_len = max(state_len - offset, 0)
                if valid_len > 0:
                    valid_cross[offset - 1, window_idx, :valid_len] = True

        return SupportObservationWindowBatch(
            max_state_len=batch_state_len,
            state_lens=jnp.asarray(state_lens, dtype=jnp.int32),
            anchor_indices=jnp.asarray(anchor_indices, dtype=jnp.int32),
            start_indices=jnp.asarray(start_indices, dtype=jnp.int32),
            mask_full=jnp.asarray(mask_full),
            prev_coeffs=jnp.asarray(padded_prev),
            curr_coeffs=jnp.asarray(padded_curr),
            weights=jnp.asarray(padded_weights),
            padded_state_indices=jnp.asarray(padded_state_indices, dtype=jnp.int32),
            time_indices=jnp.asarray(time_indices, dtype=jnp.int32),
            valid_diag=jnp.asarray(valid_diag),
            cross_time_indices=jnp.asarray(cross_time_indices, dtype=jnp.int32),
            valid_cross=jnp.asarray(valid_cross),
        )

    windows_by_state_len: dict[
        int,
        list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    for window in compiled_windows:
        bucket_state_len = _support_bucket_state_len(window[0])
        windows_by_state_len.setdefault(bucket_state_len, []).append(window)

    return (
        tuple(
            _compile_support_batch(state_len, windows_by_state_len[state_len])
            for state_len in sorted(windows_by_state_len)
        ),
        max_bandwidth,
        jnp.asarray(row_upper_bandwidths, dtype=jnp.int32),
    )


def _step_halving_search(
    z_curr: jnp.ndarray,
    step_direction: jnp.ndarray,
    current_log_joint: jnp.ndarray,
    objective_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    max_halvings: int = _SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backtracking step-halving line search for latent-mode Newton updates."""
    if max_halvings < 0:
        raise ValueError("max_halvings must be non-negative")
    zero_step = jnp.all(step_direction == 0)

    def _zero_step_result(_):
        return (
            z_curr,
            current_log_joint,
            jnp.asarray(True),
            jnp.asarray(1.0, dtype=z_curr.dtype),
        )

    def _run_search(_):
        alphas = jnp.asarray(
            [0.5**i for i in range(max_halvings + 1)],
            dtype=z_curr.dtype,
        )

        def _ls_step(carry, alpha):
            accepted, z_best, log_best, alpha_best = carry

            def _evaluate(_):
                z_cand = jnp.asarray(z_curr + alpha * step_direction, dtype=z_curr.dtype)
                cand_log_joint = jnp.asarray(
                    objective_fn(z_cand),
                    dtype=current_log_joint.dtype,
                )
                improved = jnp.isfinite(cand_log_joint) & (cand_log_joint >= current_log_joint)
                next_z = jnp.where(improved, z_cand, z_best)
                next_log = jnp.where(improved, cand_log_joint, log_best)
                next_alpha = jnp.where(improved, alpha, alpha_best)
                return (improved, next_z, next_log, next_alpha), None

            return jax.lax.cond(accepted, lambda _: (carry, None), _evaluate, operand=None)

        init_carry = (
            jnp.asarray(False),
            z_curr,
            current_log_joint,
            jnp.asarray(0.0, dtype=z_curr.dtype),
        )
        final_carry, _ = jax.lax.scan(_ls_step, init_carry, alphas)
        accepted, z_next, log_joint_next, alpha_next = final_carry
        return z_next, log_joint_next, accepted, alpha_next

    return jax.lax.cond(zero_step, _zero_step_result, _run_search, operand=None)


def _tree_contains_tracer(tree: Any) -> bool:
    """Whether any leaf in a pytree is currently a JAX tracer."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree))
