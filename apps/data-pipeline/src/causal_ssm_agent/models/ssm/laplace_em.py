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
from causal_ssm_agent.models.likelihoods.kernels import compile_measurement_semantics
from causal_ssm_agent.models.likelihoods.trajectory_observations import (
    accumulate_support_statistics,
    expected_observation_mean,
    get_point_like_mask,
    get_summary_operator_codes,
    get_support_kind_codes,
    trajectory_observation_log_prob,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.tempered_core import run_tempered_smc

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

logger = get_prefect_logger(__name__)


# ---------------------------------------------------------------------------
# Iterated Extended Kalman Smoother (IEKS)
# ---------------------------------------------------------------------------


_DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM = 160


@dataclass(frozen=True)
class SupportObservationGroupBatch:
    """One batch of anchored interval-summary windows with the same state length."""

    state_len: int
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
    T, D = J_t.shape[:2]
    eye = jnp.eye(D, dtype=J_t.dtype)

    diag_blocks = J_t
    rhs = tilde_y
    lower = jnp.zeros((T, D, D), dtype=J_t.dtype)
    upper = jnp.zeros((T, D, D), dtype=J_t.dtype)

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


def _solve_block_tridiagonal(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a block-tridiagonal linear system via recursive cyclic reduction."""
    n = diag.shape[0]
    if n == 1:
        base_diag = _symmetrize_psd(diag[0], jitter=1e-6)
        return jla.solve(base_diag, rhs[0], assume_a="pos")[None]

    D = diag.shape[-1]
    keep_np = np.arange(0, n, 2)
    odd_np = np.arange(1, n, 2)
    keep = jnp.asarray(keep_np)
    odd = jnp.asarray(odd_np)

    diag_keep = diag[keep]
    rhs_keep = rhs[keep]
    lower_keep = jnp.zeros_like(diag_keep)
    upper_keep = jnp.zeros_like(diag_keep)

    if keep_np.size > 1:
        left_rows = keep_np[1:]
        left_neighbors = left_rows - 1
        left_rhs = jnp.concatenate(
            [
                upper[left_neighbors],
                lower[left_neighbors],
                rhs[left_neighbors][..., None],
            ],
            axis=-1,
        )
        left_sol = _batched_spd_solve(diag[left_neighbors], left_rhs)
        left_to_upper = left_sol[:, :, :D]
        left_to_lower = left_sol[:, :, D : 2 * D]
        left_to_rhs = left_sol[:, :, 2 * D :].squeeze(-1)
        lower_i = lower[left_rows]
        diag_keep = diag_keep.at[1:].add(-lower_i @ left_to_upper)
        lower_keep = lower_keep.at[1:].set(-lower_i @ left_to_lower)
        rhs_keep = rhs_keep.at[1:].add(-jnp.einsum("tij,tj->ti", lower_i, left_to_rhs))

    if np.any(keep_np + 1 < n):
        right_rows = keep_np[keep_np + 1 < n]
        right_neighbors = right_rows + 1
        right_rhs = jnp.concatenate(
            [
                lower[right_neighbors],
                upper[right_neighbors],
                rhs[right_neighbors][..., None],
            ],
            axis=-1,
        )
        right_sol = _batched_spd_solve(diag[right_neighbors], right_rhs)
        right_to_lower = right_sol[:, :, :D]
        right_to_upper = right_sol[:, :, D : 2 * D]
        right_to_rhs = right_sol[:, :, 2 * D :].squeeze(-1)
        upper_i = upper[right_rows]
        n_right = right_rows.size
        diag_keep = diag_keep.at[:n_right].add(-upper_i @ right_to_lower)
        upper_keep = upper_keep.at[:n_right].set(-upper_i @ right_to_upper)
        rhs_keep = rhs_keep.at[:n_right].add(-jnp.einsum("tij,tj->ti", upper_i, right_to_rhs))

    diag_keep = _symmetrize_psd(diag_keep, jitter=1e-6)
    x_keep = _solve_block_tridiagonal(lower_keep, diag_keep, upper_keep, rhs_keep)

    solution = jnp.zeros((n, D), dtype=rhs.dtype).at[keep].set(x_keep)
    if odd_np.size == 0:
        return solution

    right_neighbors = np.minimum(odd_np + 1, n - 1)
    rhs_odd = (
        rhs[odd]
        - jnp.einsum("tij,tj->ti", lower[odd], solution[odd - 1])
        - jnp.einsum("tij,tj->ti", upper[odd], solution[jnp.asarray(right_neighbors)])
    )
    x_odd = _batched_spd_solve(_symmetrize_psd(diag[odd], jitter=1e-6), rhs_odd)
    return solution.at[odd].set(x_odd)


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

    for i in range(T):
        schur = diag[i]
        k_start = max(0, i - bandwidth)
        for k in range(k_start, i):
            l_ik = lower[i - k - 1, i]
            schur = schur - l_ik @ l_ik.T

        l_ii = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=jitter))
        chol_diag = chol_diag.at[i].set(l_ii)

        for j in range(i + 1, min(T, i + bandwidth + 1)):
            schur_off = upper[j - i - 1, i].T
            kk_start = max(0, i - bandwidth, j - bandwidth)
            for k in range(kk_start, i):
                l_jk = lower[j - k - 1, j]
                l_ik = lower[i - k - 1, i]
                schur_off = schur_off - l_jk @ l_ik.T

            l_ji = jla.solve_triangular(l_ii, schur_off.T, lower=True).T
            lower = lower.at[j - i - 1, j].set(l_ji)

    return chol_diag, lower


def _solve_block_banded_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve A x = rhs from block-banded Cholesky factors."""
    T = rhs.shape[0]
    bandwidth = lower.shape[0]
    y = jnp.zeros_like(rhs)

    for i in range(T):
        res = rhs[i]
        k_start = max(0, i - bandwidth)
        for k in range(k_start, i):
            res = res - lower[i - k - 1, i] @ y[k]
        y = y.at[i].set(jla.solve_triangular(chol_diag[i], res, lower=True))

    x = jnp.zeros_like(rhs)
    for i in range(T - 1, -1, -1):
        res = y[i]
        for j in range(i + 1, min(T, i + bandwidth + 1)):
            res = res - lower[j - i - 1, j].T @ x[j]
        x = x.at[i].set(jla.solve_triangular(chol_diag[i].T, res, lower=False))

    return x


def _block_banded_logdet(chol_diag: jnp.ndarray) -> jnp.ndarray:
    """Log determinant from block-banded Cholesky factors."""
    return 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diagonal(chol_diag, axis1=1, axis2=2), 1e-12)))


def _infer_support_groups(
    observation_support: ObservationSupportRuntime,
) -> tuple[tuple[SupportObservationGroupBatch, ...], int]:
    """Compile anchored non-point observation windows into block groups."""
    support_kind_codes = np.asarray(get_support_kind_codes(observation_support))
    prev_coeffs = np.asarray(observation_support.interval_prev_coeffs)
    curr_coeffs = np.asarray(observation_support.interval_curr_coeffs)
    weights = np.asarray(observation_support.interval_weights)
    emission_slots = np.asarray(observation_support.emission_slot_indices)
    T, n_manifest = emission_slots.shape
    tol = 1e-10

    grouped_windows: dict[
        int, list[tuple[int, int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ] = {}
    max_bandwidth = 1 if T > 1 else 0
    for anchor_idx in range(T):
        manifests = [
            manifest_idx
            for manifest_idx in range(n_manifest)
            if support_kind_codes[manifest_idx] == 1
            and emission_slots[anchor_idx, manifest_idx] >= 0
        ]
        if not manifests:
            continue

        start_idx = anchor_idx
        for manifest_idx in manifests:
            slot_idx = int(emission_slots[anchor_idx, manifest_idx])
            active_steps = np.where(
                np.abs(prev_coeffs[: anchor_idx + 1, manifest_idx, slot_idx])
                + np.abs(curr_coeffs[: anchor_idx + 1, manifest_idx, slot_idx])
                + np.abs(weights[: anchor_idx + 1, manifest_idx, slot_idx])
                > tol
            )[0]
            if active_steps.size > 0:
                start_idx = min(start_idx, int(active_steps.min()) - 1)

        start_idx = max(start_idx, 0)
        max_bandwidth = max(max_bandwidth, anchor_idx - start_idx)

        mask_full = np.zeros((n_manifest,), dtype=np.float32)
        mask_full[manifests] = 1.0
        segment_len = anchor_idx - start_idx
        group_prev = np.zeros((segment_len, n_manifest), dtype=np.float32)
        group_curr = np.zeros((segment_len, n_manifest), dtype=np.float32)
        group_weights = np.zeros((segment_len, n_manifest), dtype=np.float32)
        for manifest_idx in manifests:
            slot_idx = int(emission_slots[anchor_idx, manifest_idx])
            group_prev[:, manifest_idx] = prev_coeffs[
                start_idx + 1 : anchor_idx + 1, manifest_idx, slot_idx
            ]
            group_curr[:, manifest_idx] = curr_coeffs[
                start_idx + 1 : anchor_idx + 1, manifest_idx, slot_idx
            ]
            group_weights[:, manifest_idx] = weights[
                start_idx + 1 : anchor_idx + 1, manifest_idx, slot_idx
            ]
        state_len = anchor_idx - start_idx + 1
        grouped_windows.setdefault(state_len, []).append(
            (
                anchor_idx,
                start_idx,
                jnp.asarray(mask_full),
                jnp.asarray(group_prev),
                jnp.asarray(group_curr),
                jnp.asarray(group_weights),
            )
        )

    batches: list[SupportObservationGroupBatch] = []
    for state_len in sorted(grouped_windows):
        windows = grouped_windows[state_len]
        batches.append(
            SupportObservationGroupBatch(
                state_len=state_len,
                anchor_indices=jnp.asarray([window[0] for window in windows], dtype=jnp.int32),
                start_indices=jnp.asarray([window[1] for window in windows], dtype=jnp.int32),
                mask_full=jnp.stack([window[2] for window in windows], axis=0),
                prev_coeffs=jnp.stack([window[3] for window in windows], axis=0),
                curr_coeffs=jnp.stack([window[4] for window in windows], axis=0),
                weights=jnp.stack([window[5] for window in windows], axis=0),
            )
        )

    return tuple(batches), max_bandwidth


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
    obs_mask_float = obs_mask.astype(jnp.float32)

    def _ieks_body(_, z_est):
        """Single IEKS iteration: linearize emissions and solve the normal equations."""
        with jax.named_scope("laplace_em/ieks_linearize"):
            grads_and_hess = jax.vmap(_emission_grad_hess)(observations, z_est, obs_mask_float)
            grads = grads_and_hess[0]  # (T, D)
            J_t = grads_and_hess[1]  # (T, D, D) negative Hessian

        with jax.named_scope("laplace_em/ieks_build_system"):
            tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)
            lower, diag, upper, rhs = _build_ieks_system(
                Ad,
                Qd,
                cd_scan,
                init_mean,
                init_cov,
                J_t,
                tilde_y,
            )

        with jax.named_scope("laplace_em/ieks_solve_system"):
            return _solve_block_tridiagonal(lower, diag, upper, rhs)

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
    mask_float = obs_mask.astype(jnp.float32)

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
    mean_log_prob_fn,
    observation_support: ObservationSupportRuntime,
    support_groups: tuple[SupportObservationGroupBatch, ...],
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble exact Newton observation terms in block-banded form."""
    T, D = z_est.shape
    diag = jnp.zeros((T, D, D), dtype=z_est.dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=z_est.dtype)
    rhs = jnp.zeros((T, D), dtype=z_est.dtype)

    support_kind_codes = get_support_kind_codes(observation_support)
    summary_operator_codes = get_summary_operator_codes(observation_support)
    point_like_mask = get_point_like_mask(support_kind_codes, z_est.dtype)
    point_mask = obs_mask.astype(z_est.dtype) * point_like_mask[None, :]

    local_grads, local_hess = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
    )(observations, z_est, point_mask)
    diag = diag + local_hess
    rhs = rhs + jax.vmap(lambda j_t, z_t, g_t: j_t @ z_t + g_t)(local_hess, z_est, local_grads)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    n_manifest = observations.shape[1]

    for group_batch in support_groups:
        state_len = group_batch.state_len
        start_indices = group_batch.start_indices
        anchor_indices = group_batch.anchor_indices
        mask_full = group_batch.mask_full.astype(z_est.dtype)
        prev_coeffs = group_batch.prev_coeffs.astype(z_est.dtype)
        curr_coeffs = group_batch.curr_coeffs.astype(z_est.dtype)
        weights = group_batch.weights.astype(z_est.dtype)

        def _extract_segment(start_idx, *, _state_len: int = state_len):
            return jax.lax.dynamic_slice(z_est, (start_idx, 0), (_state_len, D))

        segment_states = jax.vmap(_extract_segment)(start_indices)
        segment_flat = segment_states.reshape(segment_states.shape[0], -1)
        anchor_obs = clean_obs[anchor_indices]

        def _window_log_prob_single(
            segment_flat_single: jnp.ndarray,
            mask_full_single: jnp.ndarray,
            prev_coeffs_single: jnp.ndarray,
            curr_coeffs_single: jnp.ndarray,
            weights_single: jnp.ndarray,
            anchor_obs_single: jnp.ndarray,
            *,
            _state_len: int = state_len,
        ) -> jnp.ndarray:
            states = segment_flat_single.reshape(_state_len, D)
            responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d))(states)

            if _state_len == 1:
                obs_sum = responses[-1]
                obs_sumsq = responses[-1] ** 2
                obs_weight = mask_full_single
            else:
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
                obs_sum = obs_sum.squeeze(-1)
                obs_sumsq = obs_sumsq.squeeze(-1)
                obs_weight = obs_weight.squeeze(-1)

            expected_mean = expected_observation_mean(
                responses[-1],
                obs_sum,
                obs_sumsq,
                obs_weight,
                summary_operator_codes,
            )
            return mean_log_prob_fn(anchor_obs_single, expected_mean, R, mask_full_single)

        window_grad = jax.grad(_window_log_prob_single)
        window_hessian = jax.hessian(_window_log_prob_single)
        grad_flat = jax.vmap(window_grad)(
            segment_flat,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
        )
        hess_flat = jax.vmap(window_hessian)(
            segment_flat,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
        )
        info_flat = -0.5 * (hess_flat + jnp.swapaxes(hess_flat, -1, -2))
        info_blocks = info_flat.reshape(-1, state_len, D, state_len, D).transpose(0, 1, 3, 2, 4)
        taylor_rhs = (jnp.einsum("gij,gj->gi", info_flat, segment_flat) + grad_flat).reshape(
            -1, state_len, D
        )

        for local_i in range(state_len):
            time_i = start_indices + local_i
            diag = diag.at[time_i].add(info_blocks[:, local_i, local_i])
            rhs = rhs.at[time_i].add(taylor_rhs[:, local_i])
            for local_j in range(local_i + 1, state_len):
                offset = local_j - local_i
                upper = upper.at[offset - 1, time_i].add(info_blocks[:, local_i, local_j])

    return diag, upper, rhs


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
    support_groups: tuple[SupportObservationGroupBatch, ...],
    bandwidth: int,
    n_ieks_iters: int,
) -> jnp.ndarray:
    """Sparse/banded support-aware IEKS + Laplace approximation."""
    T, D = observations.shape[0], init_mean.shape[0]

    z_est = jnp.broadcast_to(init_mean, (T, D)).copy()
    with jax.named_scope("laplace_em/support_aware_newton"):
        for _ in range(max(n_ieks_iters, 1)):
            with jax.named_scope("laplace_em/support_aware_prior_system"):
                prior_diag, prior_upper, prior_rhs = _build_prior_banded_system(
                    Ad,
                    Qd,
                    cd,
                    init_mean,
                    init_cov,
                    bandwidth,
                )
            with jax.named_scope("laplace_em/support_aware_observation_system"):
                obs_diag, obs_upper, obs_rhs = _assemble_support_aware_observation_system(
                    z_est,
                    observations,
                    obs_mask,
                    H,
                    d,
                    R,
                    obs_kernel,
                    mean_log_prob_fn,
                    observation_support,
                    support_groups,
                    bandwidth,
                )
            system_diag = prior_diag + obs_diag
            system_upper = prior_upper + obs_upper
            system_rhs = prior_rhs + obs_rhs
            with jax.named_scope("laplace_em/support_aware_solve"):
                chol_diag, lower = _factor_block_banded_cholesky(system_diag, system_upper)
                z_est = _solve_block_banded_from_cholesky(chol_diag, lower, system_rhs)

    with jax.named_scope("laplace_em/support_aware_final_hessian"):
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
            mean_log_prob_fn,
            observation_support,
            support_groups,
            bandwidth,
        )
        system_diag = prior_diag + obs_diag
        system_upper = prior_upper + obs_upper
        system_rhs = prior_rhs + obs_rhs
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
        for _ in range(max(n_newton_iters, 1)):
            grad = jax.grad(_neg_log_prob)(z_flat)
            hess = jax.hessian(_neg_log_prob)(z_flat)
            hess = 0.5 * (hess + hess.T) + 1e-4 * eye
            step = jla.solve(hess, grad, assume_a="sym")
            z_flat = z_flat - 0.5 * step

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
            self._support_groups, self._support_bandwidth = _infer_support_groups(
                observation_support
            )
        else:
            self._support_groups = ()
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
                self.manifest_dists[0],
                manifest_cov=measurement_params.manifest_cov,
                extra_params=extra_params,
                manifest_dists=self.manifest_dists,
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
                    self._support_groups,
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
