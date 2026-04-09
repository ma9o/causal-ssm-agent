"""Laplace-EM: IEKS + Laplace approximation with support-aware banded windows.

Implements Method 1 from the algorithmic specification:
1. Inner loop: Iterated Extended Kalman Smoother (IEKS) finds the mode of
   p(z_{1:T} | y_{1:T}, theta) via Newton iterations on the joint state posterior.
2. Laplace approximation: Gaussian approximation around the mode gives an
   approximate marginal likelihood log p(y_{1:T} | theta).
3. Outer loop: Optimize theta via gradient descent (MLE/MAP) or sample via NUTS,
   using the Laplace-approximated marginal likelihood as the log-density.

Works for any exponential-family emission (Gaussian, Poisson, Bernoulli, Gamma,
Student-t) with linear dynamics. The key requirement is twice-differentiable
log-emission density, which holds for all supported noise families.

The block-tridiagonal structure of the state-space Hessian makes the IEKS
O(T D^3) per iteration, and typically 3-8 iterations suffice for convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.engines.tempered_smc import run_tempered_smc
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    accumulate_support_statistics,
    expected_observation_mean,
    get_point_like_mask,
    get_summary_operator_codes,
    get_support_kind_codes,
    trajectory_observation_log_prob,
)

if TYPE_CHECKING:
    from causal_ssm_agent.artifacts.model_spec import DistributionFamily, LinkFunction
    from causal_ssm_agent.models.ssm.inference.targets.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm.inference.types import InferenceResult
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

logger = get_prefect_logger(__name__)


# ---------------------------------------------------------------------------
# Iterated Extended Kalman Smoother (IEKS)
# ---------------------------------------------------------------------------


_DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM = 160


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


def _batched_spd_solve(mats: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve a batch of SPD linear systems with matching right-hand sides."""
    return jax.vmap(lambda mat, b: jla.solve(mat, b, assume_a="pos"))(mats, rhs)


def _solve_spd_from_cholesky(chol: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve A x = rhs given a lower-triangular Cholesky factor A = L L^T."""
    y = jla.solve_triangular(chol, rhs, lower=True)
    return jla.solve_triangular(chol.T, y, lower=False)


def _build_prior_tridiagonal_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the latent-prior contribution for the IEKS tridiagonal system."""
    T, D = Ad.shape[:2]
    eye = jnp.eye(D, dtype=Ad.dtype)

    diag_blocks = jnp.zeros((T, D, D), dtype=Ad.dtype)
    rhs = jnp.zeros((T, D), dtype=Ad.dtype)
    lower = jnp.zeros((T, D, D), dtype=Ad.dtype)
    upper = jnp.zeros((T, D, D), dtype=Ad.dtype)

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


def _build_ieks_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    J_t: jnp.ndarray,
    tilde_y: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the block-tridiagonal IEKS normal equations."""
    lower, prior_diag, upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
        jitter=jitter,
    )
    diag_blocks = prior_diag + J_t
    rhs = prior_rhs + tilde_y
    return lower, _symmetrize_psd(diag_blocks, jitter=jitter), upper, rhs


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
    T, D = Ad.shape[:2]
    eye = jnp.eye(D, dtype=Ad.dtype)

    diag = jnp.zeros((T, D, D), dtype=Ad.dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=Ad.dtype)
    rhs = jnp.zeros((T, D), dtype=Ad.dtype)

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
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Block-banded Cholesky factorization A = L L^T."""
    T, _D = diag.shape[:2]
    bandwidth = upper.shape[0]
    chol_diag = jnp.zeros_like(diag)
    lower = jnp.zeros_like(upper)

    def _factor_step(i, state):
        chol_diag_state, lower_state = state
        schur = diag[i]

        for offset in range(1, bandwidth + 1):
            schur = jax.lax.cond(
                i >= offset,
                lambda s, off=offset: s - lower_state[off - 1, i] @ lower_state[off - 1, i].T,
                lambda s: s,
                schur,
            )

        l_ii = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=jitter))
        chol_diag_state = chol_diag_state.at[i].set(l_ii)

        def _update_lower_for_offset(lower_curr, offset_j: int):
            def _compute(curr_lower):
                schur_off = upper[offset_j - 1, i].T
                for offset_k in range(1, bandwidth + 1):
                    cross_offset = offset_j + offset_k - 1
                    schur_off = jax.lax.cond(
                        (i >= offset_k) & (cross_offset < bandwidth),
                        lambda s, off_k=offset_k, cross=cross_offset: (
                            s - curr_lower[cross, i + offset_j] @ curr_lower[off_k - 1, i].T
                        ),
                        lambda s: s,
                        schur_off,
                    )
                l_ji = jla.solve_triangular(l_ii, schur_off.T, lower=True).T
                return curr_lower.at[offset_j - 1, i + offset_j].set(l_ji)

            return jax.lax.cond(i + offset_j < T, _compute, lambda x: x, lower_curr)

        for offset_j in range(1, bandwidth + 1):
            lower_state = _update_lower_for_offset(lower_state, offset_j)

        return chol_diag_state, lower_state

    return jax.lax.fori_loop(0, T, _factor_step, (chol_diag, lower))


def _solve_block_banded_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve A x = rhs from block-banded Cholesky factors."""
    T = rhs.shape[0]
    bandwidth = lower.shape[0]
    y = jnp.zeros_like(rhs)

    def _forward_step(i, y_state):
        res = rhs[i]
        for offset in range(1, bandwidth + 1):
            res = jax.lax.cond(
                i >= offset,
                lambda r, off=offset: r - lower[off - 1, i] @ y_state[i - off],
                lambda r: r,
                res,
            )
        y_i = jla.solve_triangular(chol_diag[i], res, lower=True)
        return y_state.at[i].set(y_i)

    y = jax.lax.fori_loop(0, T, _forward_step, y)
    x = jnp.zeros_like(rhs)

    def _backward_step(rev_idx, x_state):
        i = T - 1 - rev_idx
        res = y[i]
        for offset in range(1, bandwidth + 1):
            res = jax.lax.cond(
                i + offset < T,
                lambda r, off=offset: r - lower[off - 1, i + off].T @ x_state[i + off],
                lambda r: r,
                res,
            )
        x_i = jla.solve_triangular(chol_diag[i].T, res, lower=False)
        return x_state.at[i].set(x_i)

    return jax.lax.fori_loop(0, T, _backward_step, x)


def _block_banded_logdet(chol_diag: jnp.ndarray) -> jnp.ndarray:
    """Log determinant from block-banded Cholesky factors."""
    return 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diagonal(chol_diag, axis1=1, axis2=2), 1e-12)))


def _infer_support_groups(
    observation_support: ObservationSupportRuntime,
) -> tuple[SupportObservationWindowBatch, int]:
    """Compile anchored non-point observation windows into a padded window table."""
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
    max_state_len = 1
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
        max_state_len = max(max_state_len, state_len)
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

    n_windows = len(compiled_windows)
    max_segment_len = max_state_len - 1
    state_lens = np.zeros((n_windows,), dtype=np.int32)
    anchor_indices = np.zeros((n_windows,), dtype=np.int32)
    start_indices = np.zeros((n_windows,), dtype=np.int32)
    mask_full = np.zeros((n_windows, n_manifest), dtype=np.float64)
    padded_prev = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
    padded_curr = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
    padded_weights = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)

    for window_idx, (
        state_len,
        anchor_idx,
        start_idx,
        mask_window,
        prev_window,
        curr_window,
        weights_window,
    ) in enumerate(compiled_windows):
        segment_len = state_len - 1
        state_lens[window_idx] = state_len
        anchor_indices[window_idx] = anchor_idx
        start_indices[window_idx] = start_idx
        mask_full[window_idx] = mask_window
        if segment_len > 0:
            padded_prev[window_idx, :segment_len] = prev_window
            padded_curr[window_idx, :segment_len] = curr_window
            padded_weights[window_idx, :segment_len] = weights_window

    return (
        SupportObservationWindowBatch(
            max_state_len=max_state_len,
            state_lens=jnp.asarray(state_lens, dtype=jnp.int32),
            anchor_indices=jnp.asarray(anchor_indices, dtype=jnp.int32),
            start_indices=jnp.asarray(start_indices, dtype=jnp.int32),
            mask_full=jnp.asarray(mask_full),
            prev_coeffs=jnp.asarray(padded_prev),
            curr_coeffs=jnp.asarray(padded_curr),
            weights=jnp.asarray(padded_weights),
        ),
        max_bandwidth,
    )


def _ieks_smooth(
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
    n_ieks_iters=5,
):
    """Run the Iterated Extended Kalman Smoother to find the MAP state trajectory.

    Args:
        observations: (T, n_manifest) observed data
        obs_mask: (T, n_manifest) boolean observation mask
        Ad: (T, D, D) discrete-time transition matrices
        Qd: (T, D, D) discrete-time process noise covariances
        cd: (T, D) discrete-time intercepts
        H: (n_manifest, D) measurement matrix
        d: (n_manifest,) measurement intercept
        R: (n_manifest, n_manifest) measurement noise covariance
        init_mean: (D,) initial state mean
        init_cov: (D, D) initial state covariance
        obs_kernel: ObservationKernel with emission_fn and emission_grad_hess_fn
        n_ieks_iters: number of IEKS iterations

    Returns:
        z_smooth: (T, D) smoothed state means (MAP trajectory)
        log_lik_approx: scalar approximate log-likelihood
    """
    T = observations.shape[0]
    D = init_mean.shape[0]

    # Compute gradient and Hessian of emission log-prob w.r.t. z_t
    def _emission_grad_hess(y_t, z_t, mask_t):
        return obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)

    # Initialize state estimates (zeros or prior mean)
    z_est = jnp.broadcast_to(init_mean, (T, D)).copy()

    cd_scan = cd if cd is not None else jnp.zeros((T, D))
    obs_mask_float = obs_mask.astype(jnp.float64)
    prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd_scan,
        init_mean,
        init_cov,
    )

    def _ieks_body(_, z_est):
        """Single IEKS iteration: linearize emissions and solve the normal equations."""
        with jax.named_scope("laplace_em/ieks_linearize"):
            grads_and_hess = jax.vmap(_emission_grad_hess)(observations, z_est, obs_mask_float)
            grads = grads_and_hess[0]  # (T, D)
            J_t = grads_and_hess[1]  # (T, D, D) negative Hessian

        with jax.named_scope("laplace_em/ieks_build_system"):
            tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)
            diag = prior_diag + J_t
            rhs = prior_rhs + tilde_y

        with jax.named_scope("laplace_em/ieks_solve_system"):
            return _solve_block_tridiagonal(prior_lower, diag, prior_upper, rhs)

    with jax.named_scope("laplace_em/ieks_iterations"):
        z_est = jax.lax.fori_loop(0, n_ieks_iters, _ieks_body, z_est)
    z_smooth = z_est

    # Compute approximate log-likelihood via prediction error decomposition
    # Sum of log p(y_t | y_{1:t-1}, theta) from the final filter pass
    with jax.named_scope("laplace_em/ieks_log_likelihood"):
        log_lik = _compute_laplace_log_lik(
            observations,
            obs_mask,
            z_smooth,
            Ad,
            Qd,
            cd_scan,
            init_mean,
            init_cov,
            obs_kernel,
            H,
            d,
            R,
        )

    return z_smooth, log_lik


def _compute_laplace_log_lik(
    observations,
    obs_mask,
    z_smooth,
    Ad,
    Qd,
    cd,
    init_mean,
    init_cov,
    obs_kernel,
    H,
    d,
    R,
):
    """Compute Laplace-approximated log-likelihood via prediction error decomposition.

    Uses the one-step-ahead Laplace formula. Linearizes the emission model
    around z_smooth (from the IEKS), runs a forward Kalman filter pass with
    the linearized model, and accumulates the per-step contribution:

        ll_t = l(z_filt_t) - 0.5||z_filt_t - z_pred_t||^2_{P_pred^-1}
             + 0.5 log(det P_filt_t / det P_pred_t)

    where z_filt_t is the filter mode (the correct z* for the one-step Laplace),
    NOT the smoother output. For the linear Gaussian case this reduces to the
    exact Kalman prediction error decomposition.
    """
    T, D = z_smooth.shape
    jitter = jnp.eye(D) * 1e-6
    mask_float = obs_mask.astype(jnp.float64)

    # Compute emission gradients and Hessians at the smoothed states (linearization point)
    def _emission_grad_hess(y_t, z_t, mask_t):
        return obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)

    with jax.named_scope("laplace_em/loglik_linearize"):
        all_grads_hess = jax.vmap(_emission_grad_hess)(observations, z_smooth, mask_float)
        grads = all_grads_hess[0]  # (T, D)
        J_t = all_grads_hess[1]  # (T, D, D) — emission info matrices

    def _step_ll(y_t, mask_t, z_pred, P_pred, J_obs, grad_obs, z_lin):
        """Single-step Laplace log-likelihood contribution.

        Returns (ll_t, z_filt, P_filt) for chaining the filter forward.
        """
        P_pred_reg = P_pred + jitter

        # Filter update (information form): P_filt^{-1} = P_pred^{-1} + J_obs
        P_pred_inv = jla.solve(P_pred_reg, jnp.eye(D), assume_a="pos")
        P_filt_inv = P_pred_inv + J_obs
        P_filt = jla.solve(P_filt_inv + jitter, jnp.eye(D), assume_a="pos")
        P_filt = 0.5 * (P_filt + P_filt.T) + jitter

        # Filter mean: z_filt = P_filt @ (P_pred_inv @ z_pred + tilde_y)
        tilde_y = J_obs @ z_lin + grad_obs
        z_filt = P_filt @ (P_pred_inv @ z_pred + tilde_y)

        # Emission log-prob at the filter mode z_filt (the correct z* for one-step Laplace)
        emission_ll = obs_kernel.emission_fn(y_t, z_filt, H, d, R, mask_t)

        # Log-determinant ratio: log(det P_filt / det P_pred)
        _, ld_filt = jnp.linalg.slogdet(P_filt)
        _, ld_pred = jnp.linalg.slogdet(P_pred_reg)
        log_det_ratio = ld_filt - ld_pred

        # Prior penalty: -0.5 * (z_filt - z_pred)^T P_pred^{-1} (z_filt - z_pred)
        diff = z_filt - z_pred
        mahal = diff @ jla.solve(P_pred_reg, diff, assume_a="pos")

        ll_t = emission_ll - 0.5 * mahal + 0.5 * log_det_ratio
        return ll_t, z_filt, P_filt

    # Time 0: predict from initial state
    z_pred_0 = Ad[0] @ init_mean + cd[0]
    P_pred_0 = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P_pred_0 = 0.5 * (P_pred_0 + P_pred_0.T)

    ll_0, z_filt_0, P_filt_0 = _step_ll(
        observations[0], mask_float[0], z_pred_0, P_pred_0, J_t[0], grads[0], z_smooth[0]
    )

    if T == 1:
        return jnp.array([ll_0])

    # Forward scan for t=1..T-1: predict, then compute ll_t
    def _forward_ll_step(carry, inputs):
        z_filt_prev, P_filt_prev = carry
        Ad_t, Qd_t, cd_t, z_lin_t, J_obs_t, grad_t, y_t, mask_t = inputs

        # Predict
        z_pred = Ad_t @ z_filt_prev + cd_t
        P_pred = Ad_t @ P_filt_prev @ Ad_t.T + Qd_t
        P_pred = 0.5 * (P_pred + P_pred.T)

        ll_t, z_filt, P_filt = _step_ll(y_t, mask_t, z_pred, P_pred, J_obs_t, grad_t, z_lin_t)

        return (z_filt, P_filt), ll_t

    with jax.named_scope("laplace_em/loglik_forward_pass"):
        _, ll_rest = jax.lax.scan(
            _forward_ll_step,
            (z_filt_0, P_filt_0),
            (
                Ad[1:],
                Qd[1:],
                cd[1:],
                z_smooth[1:],
                J_t[1:],
                grads[1:],
                observations[1:],
                mask_float[1:],
            ),
        )

    # Return (T,) cumulative log-normalizing constants matching LikelihoodBackend protocol.
    ll_all = jnp.concatenate([jnp.array([ll_0]), ll_rest])
    return jnp.cumsum(ll_all)


def _trajectory_prior_log_prob(
    latent_trajectory: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Return log p(z_{1:T} | theta) under the discretized latent dynamics."""
    from numpyro.distributions import MultivariateNormal

    T, D = latent_trajectory.shape
    jitter = jnp.eye(D, dtype=latent_trajectory.dtype) * 1e-6

    z0_pred = Ad[0] @ init_mean + cd[0]
    P0_pred = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P0_pred = 0.5 * (P0_pred + P0_pred.T) + jitter
    init_ll = MultivariateNormal(z0_pred, covariance_matrix=P0_pred).log_prob(latent_trajectory[0])

    if T == 1:
        return init_ll

    def _transition_ll(z_t, z_tm1, Ad_t, Qd_t, cd_t):
        mean = Ad_t @ z_tm1 + cd_t
        cov = 0.5 * (Qd_t + Qd_t.T) + jitter
        return MultivariateNormal(mean, covariance_matrix=cov).log_prob(z_t)

    trans_ll = jax.vmap(_transition_ll)(
        latent_trajectory[1:],
        latent_trajectory[:-1],
        Ad[1:],
        Qd[1:],
        cd[1:],
    )
    return init_ll + jnp.sum(trans_ll)


def _assemble_support_aware_observation_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
    support_windows: SupportObservationWindowBatch,
    point_like_mask: jnp.ndarray,
    window_derivatives,
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble exact Newton observation terms in block-banded form."""
    T, D = z_est.shape
    diag = jnp.zeros((T, D, D), dtype=z_est.dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=z_est.dtype)
    rhs = jnp.zeros((T, D), dtype=z_est.dtype)

    point_mask = obs_mask.astype(z_est.dtype) * point_like_mask[None, :]

    local_grads, local_hess = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
    )(observations, z_est, point_mask)
    diag = diag + local_hess
    rhs = rhs + jax.vmap(lambda j_t, z_t, g_t: j_t @ z_t + g_t)(local_hess, z_est, local_grads)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    max_state_len = support_windows.max_state_len
    if support_windows.state_lens.shape[0] == 0:
        return diag, upper, rhs

    padded_z = jnp.pad(z_est, ((0, max_state_len - 1), (0, 0)))

    def _extract_segment(start_idx):
        return jax.lax.dynamic_slice(padded_z, (start_idx, 0), (max_state_len, D))

    segment_states = jax.vmap(_extract_segment)(support_windows.start_indices)
    segment_flat = segment_states.reshape(segment_states.shape[0], -1)
    anchor_obs = clean_obs[support_windows.anchor_indices]
    grad_flat, hess_flat = window_derivatives(
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
    info_flat = -0.5 * (hess_flat + jnp.swapaxes(hess_flat, -1, -2))
    info_blocks = info_flat.reshape(-1, max_state_len, D, max_state_len, D).transpose(0, 1, 3, 2, 4)
    taylor_rhs = (jnp.einsum("gij,gj->gi", info_flat, segment_flat) + grad_flat).reshape(
        -1, max_state_len, D
    )

    local_positions = jnp.arange(max_state_len, dtype=support_windows.start_indices.dtype)
    time_indices = jnp.clip(
        support_windows.start_indices[:, None] + local_positions[None, :],
        0,
        T - 1,
    )
    valid_diag = local_positions[None, :] < support_windows.state_lens[:, None]
    diag_updates = info_blocks[:, jnp.arange(max_state_len), jnp.arange(max_state_len)]
    diag = diag.at[time_indices.reshape(-1)].add(
        (diag_updates * valid_diag[..., None, None]).reshape(-1, D, D)
    )
    rhs = rhs.at[time_indices.reshape(-1)].add((taylor_rhs * valid_diag[..., None]).reshape(-1, D))

    if bandwidth > 0:
        local_i = jnp.arange(max_state_len, dtype=support_windows.start_indices.dtype)
        local_j = jnp.arange(max_state_len, dtype=support_windows.start_indices.dtype)
        grid_i = jnp.broadcast_to(
            local_i[None, :, None], (segment_flat.shape[0], max_state_len, max_state_len)
        )
        grid_j = jnp.broadcast_to(
            local_j[None, None, :], (segment_flat.shape[0], max_state_len, max_state_len)
        )
        upper_offsets = grid_j - grid_i - 1
        valid_upper = (
            (grid_j > grid_i)
            & (grid_i < support_windows.state_lens[:, None, None])
            & (grid_j < support_windows.state_lens[:, None, None])
        )
        upper_times = jnp.clip(
            support_windows.start_indices[:, None, None] + grid_i,
            0,
            T - 1,
        )
        safe_offsets = jnp.clip(upper_offsets, 0, bandwidth - 1)
        upper = upper.at[safe_offsets.reshape(-1), upper_times.reshape(-1)].add(
            (info_blocks * valid_upper[..., None, None]).reshape(-1, D, D)
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
    """Build one exact window grad/Hessian evaluator for padded support windows."""

    def _window_log_prob_single(
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
        return mean_log_prob_fn(anchor_obs_single, expected_mean, R, mask_full_single)

    window_grad = jax.grad(_window_log_prob_single)
    window_hessian = jax.hessian(_window_log_prob_single)

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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        in_axes = (0, 0, 0, 0, 0, 0, 0, None, None, None)
        grad_flat = jax.vmap(window_grad, in_axes=in_axes)(
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
        hess_flat = jax.vmap(window_hessian, in_axes=in_axes)(
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
        return grad_flat, hess_flat

    return _batched_window_derivatives


def _support_aware_ieks_log_lik(
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
    support_windows: SupportObservationWindowBatch,
    bandwidth: int,
    n_ieks_iters: int,
) -> jnp.ndarray:
    """Sparse/banded support-aware IEKS + Laplace approximation."""
    T, D = observations.shape[0], init_mean.shape[0]
    prior_diag, prior_upper, prior_rhs = _build_prior_banded_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
        bandwidth,
    )
    summary_operator_codes = get_summary_operator_codes(observation_support)
    point_like_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )
    window_derivatives = _make_support_window_derivatives(
        max_state_len=support_windows.max_state_len,
        n_latent=D,
        n_manifest=observations.shape[1],
        summary_operator_codes=summary_operator_codes,
        obs_kernel=obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
    )

    z_est = jnp.broadcast_to(init_mean, (T, D)).copy()

    def _newton_step(_idx, z_curr):
        with jax.named_scope("laplace_em/support_aware_observation_system"):
            obs_diag, obs_upper, obs_rhs = _assemble_support_aware_observation_system(
                z_curr,
                observations,
                obs_mask,
                H,
                d,
                R,
                obs_kernel,
                support_windows,
                point_like_mask,
                window_derivatives,
                bandwidth,
            )
        system_diag = prior_diag + obs_diag
        system_upper = prior_upper + obs_upper
        system_rhs = prior_rhs + obs_rhs
        with jax.named_scope("laplace_em/support_aware_solve"):
            chol_diag, lower = _factor_block_banded_cholesky(system_diag, system_upper)
            return _solve_block_banded_from_cholesky(chol_diag, lower, system_rhs)

    with jax.named_scope("laplace_em/support_aware_newton"):
        z_est = jax.lax.fori_loop(0, max(n_ieks_iters, 1), _newton_step, z_est)

    with jax.named_scope("laplace_em/support_aware_final_hessian"):
        obs_diag, obs_upper, _obs_rhs = _assemble_support_aware_observation_system(
            z_est,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            support_windows,
            point_like_mask,
            window_derivatives,
            bandwidth,
        )
        system_diag = prior_diag + obs_diag
        system_upper = prior_upper + obs_upper
        chol_diag, _lower = _factor_block_banded_cholesky(system_diag, system_upper)

    flat_dim = T * D
    with jax.named_scope("laplace_em/support_aware_mode_log_joint"):
        mode_log_joint = _trajectory_prior_log_prob(z_est, Ad, Qd, cd, init_mean, init_cov) + (
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
    return (
        mode_log_joint
        + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi)
        - 0.5 * _block_banded_logdet(chol_diag)
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
) -> jnp.ndarray:
    """Dense Laplace approximation for interval-summary observation semantics."""
    T, D = observations.shape[0], init_mean.shape[0]
    flat_dim = T * D
    eye = jnp.eye(flat_dim, dtype=observations.dtype)

    def _predictive_init():
        z0 = Ad[0] @ init_mean + cd[0]
        if T == 1:
            return z0[None]

        def _step(z_prev, inputs):
            Ad_t, cd_t = inputs
            z_t = Ad_t @ z_prev + cd_t
            return z_t, z_t

        _, z_rest = jax.lax.scan(_step, z0, (Ad[1:], cd[1:]))
        return jnp.concatenate([z0[None], z_rest], axis=0)

    with jax.named_scope("laplace_em/dense_support_init"):
        z_init = _predictive_init()

    def _joint_log_prob(z_flat):
        z = z_flat.reshape(T, D)
        prior_ll = _trajectory_prior_log_prob(z, Ad, Qd, cd, init_mean, init_cov)
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
    with jax.named_scope("laplace_em/dense_support_newton"):
        best_z = z_flat
        best_neg = _neg_log_prob(z_flat)
        for _ in range(max(n_newton_iters, 1)):
            grad = jax.grad(_neg_log_prob)(z_flat)
            hess = jax.hessian(_neg_log_prob)(z_flat)
            hess = 0.5 * (hess + hess.T) + 1e-4 * eye
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
        hess = 0.5 * (hess + hess.T)
        eigvals = jnp.linalg.eigvalsh(hess)
        logdet = jnp.sum(jnp.log(jnp.maximum(eigvals, 1e-6)))
    return mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * logdet


# ---------------------------------------------------------------------------
# Laplace likelihood backend (for use in NumPyro model)
# ---------------------------------------------------------------------------


class LaplaceLikelihood:
    """Laplace-approximated likelihood backend.

    Computes log p(y|theta) via IEKS + Laplace approximation.
    Drop-in replacement for KalmanLikelihood / ParticleLikelihood.

    Accepts per-channel distribution and link lists to support heterogeneous
    observation models (e.g., channel 0 Gaussian, channel 1 Poisson).
    """

    checkpoint_loglik = False

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dists: list[DistributionFamily],
        manifest_links: list[LinkFunction],
        n_ieks_iters: int = 5,
        observation_support: ObservationSupportRuntime | None = None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dists = manifest_dists
        self.manifest_links = manifest_links
        self.n_ieks_iters = n_ieks_iters
        self.observation_support = observation_support
        if (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        ):
            self._support_windows, self._support_bandwidth = _infer_support_groups(
                observation_support
            )
        else:
            self._support_windows = SupportObservationWindowBatch(
                max_state_len=1,
                state_lens=jnp.zeros((0,), dtype=jnp.int32),
                anchor_indices=jnp.zeros((0,), dtype=jnp.int32),
                start_indices=jnp.zeros((0,), dtype=jnp.int32),
                mask_full=jnp.zeros((0, n_manifest), dtype=jnp.float64),
                prev_coeffs=jnp.zeros((0, 0, n_manifest), dtype=jnp.float64),
                curr_coeffs=jnp.zeros((0, 0, n_manifest), dtype=jnp.float64),
                weights=jnp.zeros((0, 0, n_manifest), dtype=jnp.float64),
            )
            self._support_bandwidth = 1 if n_latent > 0 else 0

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> jnp.ndarray:
        """Compute Laplace-approximated log-likelihood.

        Returns:
            (T,) cumulative log-normalizing constants, matching LikelihoodBackend protocol.
        """
        n = self.n_latent

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        # Pre-discretize CT -> DT
        with jax.named_scope("laplace_em/discretize_system"):
            Ad, Qd, cd = discretize_system_batched(
                ct_params.drift,
                ct_params.diffusion_cov,
                ct_params.cint,
                time_intervals,
            )
        if cd is None:
            cd = jnp.zeros((len(time_intervals), n))
        else:
            cd = jnp.asarray(cd)
            if cd.ndim == 1:
                cd = cd[:, None]

        with jax.named_scope("laplace_em/compile_measurement_semantics"):
            measurement_semantics = compile_measurement_semantics(
                self.manifest_dists,
                manifest_cov=measurement_params.manifest_cov,
                extra_params=extra_params,
                manifest_links=self.manifest_links,
                observation_support=self.observation_support,
            )
        obs_kernel = measurement_semantics.obs_kernel

        if (
            self.observation_support is not None
            and self.observation_support.requires_interval_summary_handling
        ):
            if _should_use_dense_support_laplace(
                n_time=clean_obs.shape[0],
                n_latent=self.n_latent,
            ):
                with jax.named_scope("laplace_em/dense_support_backend"):
                    return _dense_support_laplace_log_lik(
                        clean_obs,
                        obs_mask,
                        Ad,
                        Qd,
                        cd,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        initial_state.cov,
                        obs_kernel,
                        measurement_semantics.mean_log_prob_fn,
                        self.observation_support,
                        self.n_ieks_iters,
                    )
            with jax.named_scope("laplace_em/support_aware_backend"):
                return _support_aware_ieks_log_lik(
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    measurement_params.lambda_mat,
                    measurement_params.manifest_means,
                    measurement_params.manifest_cov,
                    initial_state.mean,
                    initial_state.cov,
                    obs_kernel,
                    measurement_semantics.mean_log_prob_fn,
                    self.observation_support,
                    self._support_windows,
                    self._support_bandwidth,
                    self.n_ieks_iters,
                )

        with jax.named_scope("laplace_em/ieks_backend"):
            _, log_lik = _ieks_smooth(
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                measurement_params.lambda_mat,
                measurement_params.manifest_means,
                measurement_params.manifest_cov,
                initial_state.mean,
                initial_state.cov,
                obs_kernel,
                n_ieks_iters=self.n_ieks_iters,
            )

        return log_lik


# ---------------------------------------------------------------------------
# fit_laplace_em: outer loop for parameter estimation
# ---------------------------------------------------------------------------


def fit_laplace_em(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    n_ieks_iters: int = 5,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via Laplace-EM with tempered SMC outer loop.

    Uses the Laplace-approximated marginal likelihood (via IEKS) as the
    log-density for a tempered SMC sampler over the parameter space.

    If the model has an explicit likelihood override (e.g. likelihood="kalman"),
    that backend is used instead of the Laplace approximation.
    """
    backend_label = "kalman" if model.likelihood == "kalman" else "laplace_ieks"
    logger.info(
        "Laplace-EM config: backend=%s n_outer=%s n_particles=%s n_mh=%s "
        "n_leapfrog=%s n_ieks_iters=%s adaptive_tempering=%s target_ess_ratio=%.2f "
        "waste_free=%s n_warmup=%s",
        backend_label,
        n_outer,
        n_csmc_particles,
        n_mh_steps,
        n_leapfrog,
        n_ieks_iters,
        adaptive_tempering,
        target_ess_ratio,
        waste_free,
        n_warmup if n_warmup is not None else 5,
    )
    with jax.profiler.TraceAnnotation("laplace_em/build_likelihood_backend"):
        if model.likelihood == "kalman":
            backend = model.make_likelihood_backend()
        else:
            backend = model.make_laplace_backend(n_ieks_iters)
    with jax.profiler.TraceAnnotation("laplace_em/run_tempered_smc"):
        return run_tempered_smc(
            model,
            observations,
            times,
            n_outer=n_outer,
            n_csmc_particles=n_csmc_particles,
            n_mh_steps=n_mh_steps,
            param_step_size=param_step_size,
            n_warmup=n_warmup,
            target_accept=target_accept,
            seed=seed,
            adaptive_tempering=adaptive_tempering,
            target_ess_ratio=target_ess_ratio,
            waste_free=waste_free,
            n_leapfrog=n_leapfrog,
            method_name="laplace_em",
            likelihood_backend=backend,
            extra_diagnostics={"n_ieks_iters": n_ieks_iters, "likelihood_backend": backend},
            print_prefix="Laplace-EM",
            reparam=reparam,
        )
