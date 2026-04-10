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
import jax.random as random
import jax.scipy.linalg as jla
import jax.scipy.optimize as jso
import numpy as np
import scipy.optimize as spo
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    accumulate_support_statistics,
    expected_observation_mean,
    get_point_like_mask,
    get_summary_operator_codes,
    get_support_kind_codes,
    trajectory_observation_log_prob,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult
from causal_ssm_agent.models.ssm.inference.utils import (
    _build_eval_fns,
    _discover_sites,
    extract_constrained_samples,
)
from causal_ssm_agent.models.ssm.parameterization import assemble_deterministics_from_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from causal_ssm_agent.artifacts.model_spec import DistributionFamily, LinkFunction
    from causal_ssm_agent.models.ssm.inference.targets.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

logger = get_prefect_logger(__name__)


# ---------------------------------------------------------------------------
# Iterated Extended Kalman Smoother (IEKS)
# ---------------------------------------------------------------------------


_DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM = 160
_SUPPORT_AWARE_IEKS_CONVERGENCE_RTOL = 1e-3
_SUPPORT_AWARE_LM_DAMPING = 1e-3
_SUPPORT_AWARE_LM_DAMPING_MIN = 1e-6
_SUPPORT_AWARE_LM_DAMPING_MAX = 1e6
_SUPPORT_AWARE_LM_DAMPING_GROWTH = 10.0
_SUPPORT_AWARE_LM_DAMPING_SHRINK = 0.5
_SUPPORT_AWARE_LINE_SEARCH_MAX_HALVINGS = 6


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


@dataclass(frozen=True)
class LaplaceModeOptimizationResult:
    """Unified outer-optimizer result for Laplace-EM parameter mode finding."""

    z_mode: jnp.ndarray
    objective_at_mode: float
    n_iters: int
    n_function_evals: int
    status: int
    success: bool
    optimizer: str
    init_log_posterior_best: float


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
    state_lens = np.zeros((n_windows,), dtype=np.int64)
    anchor_indices = np.zeros((n_windows,), dtype=np.int64)
    start_indices = np.zeros((n_windows,), dtype=np.int64)
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
            state_lens=jnp.asarray(state_lens, dtype=jnp.int64),
            anchor_indices=jnp.asarray(anchor_indices, dtype=jnp.int64),
            start_indices=jnp.asarray(start_indices, dtype=jnp.int64),
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
        P_filt = symmetrize_with_jitter(P_filt, jitter=1e-6)

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
    P_pred_0 = symmetrize(P_pred_0)

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
        P_pred = symmetrize(P_pred)

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

    T, _D = latent_trajectory.shape

    z0_pred = Ad[0] @ init_mean + cd[0]
    P0_pred = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P0_pred = symmetrize_with_jitter(P0_pred, jitter=1e-6)
    init_ll = MultivariateNormal(z0_pred, covariance_matrix=P0_pred).log_prob(latent_trajectory[0])

    if T == 1:
        return init_ll

    def _transition_ll(z_t, z_tm1, Ad_t, Qd_t, cd_t):
        mean = Ad_t @ z_tm1 + cd_t
        cov = symmetrize_with_jitter(Qd_t, jitter=1e-6)
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

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    point_mask = obs_mask.astype(z_est.dtype) * point_like_mask[None, :]

    local_grads, local_hess = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
    )(clean_obs, z_est, point_mask)
    diag = diag + local_hess
    rhs = rhs + jax.vmap(lambda j_t, z_t, g_t: j_t @ z_t + g_t)(local_hess, z_est, local_grads)

    max_state_len = support_windows.max_state_len
    if support_windows.state_lens.shape[0] == 0:
        return diag, upper, rhs

    padded_z = jnp.pad(z_est, ((0, max_state_len - 1), (0, 0)))

    def _extract_segment(start_idx):
        return jax.lax.dynamic_slice(padded_z, (start_idx, 0), (max_state_len, D))

    segment_states = jax.vmap(_extract_segment)(support_windows.start_indices)
    segment_flat = segment_states.reshape(segment_states.shape[0], -1)
    anchor_obs = clean_obs[support_windows.anchor_indices]
    grad_blocks, jac_blocks, mean_info = window_derivatives(
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

    local_positions = jnp.arange(max_state_len, dtype=support_windows.start_indices.dtype)
    time_indices = jnp.clip(
        support_windows.start_indices[:, None] + local_positions[None, :],
        0,
        T - 1,
    )
    valid_diag = local_positions[None, :] < support_windows.state_lens[:, None]
    diag = diag.at[time_indices.reshape(-1)].add(
        (diag_updates * valid_diag[..., None, None]).reshape(-1, D, D)
    )

    if bandwidth > 0:
        for offset in range(1, bandwidth + 1):
            left_jac = jac_blocks[:, :, :-offset, :]
            right_jac = jac_blocks[:, :, offset:, :]
            cross_updates = jnp.einsum("gmid,gmn,gnie->gide", left_jac, mean_info, right_jac)

            left_states = segment_states[:, :-offset, :]
            right_states = segment_states[:, offset:, :]
            taylor_rhs = taylor_rhs.at[:, :-offset, :].add(
                jnp.einsum("gide,gie->gid", cross_updates, right_states)
            )
            taylor_rhs = taylor_rhs.at[:, offset:, :].add(
                jnp.einsum("gide,gid->gie", cross_updates, left_states)
            )

            valid_cross = (
                local_positions[:-offset][None, :] < support_windows.state_lens[:, None]
            ) & (local_positions[offset:][None, :] < support_windows.state_lens[:, None])
            upper_times = jnp.clip(
                support_windows.start_indices[:, None] + local_positions[:-offset][None, :],
                0,
                T - 1,
            )
            upper = upper.at[offset - 1, upper_times.reshape(-1)].add(
                (cross_updates * valid_cross[..., None, None]).reshape(-1, D, D)
            )

    rhs = rhs.at[time_indices.reshape(-1)].add((taylor_rhs * valid_diag[..., None]).reshape(-1, D))

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
) -> jnp.ndarray:
    """Exact latent joint log-density used for support-aware step acceptance."""
    return _trajectory_prior_log_prob(z_est, Ad, Qd, cd, init_mean, init_cov) + (
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
    if max_halvings < 0:
        raise ValueError("max_halvings must be non-negative")

    alphas = jnp.asarray(
        [0.5**i for i in range(max_halvings + 1)],
        dtype=z_curr.dtype,
    )

    def _ls_step(carry, alpha):
        accepted, z_best, log_best, alpha_best = carry

        def _evaluate(_):
            z_cand = z_curr + alpha * step_direction
            cand_log_joint = objective_fn(z_cand)
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
    support_windows: SupportObservationWindowBatch,
    bandwidth: int,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Support-aware IEKS solve plus Laplace log-likelihood."""
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

    if z_init is None:
        z_est = _predictive_latent_init(Ad, cd, init_mean)
    else:
        z_est = jnp.asarray(z_init, dtype=observations.dtype)

    log_joint_curr = _support_aware_joint_log_prob(
        z_est,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad,
        Qd=Qd,
        cd=cd,
        H=H,
        d=d,
        R=R,
        init_mean=init_mean,
        init_cov=init_cov,
        obs_kernel=obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_support=observation_support,
    )

    def _newton_step(carry, _idx):
        z_curr, log_joint_prev, damping, active = carry
        damping = jnp.asarray(damping, dtype=z_curr.dtype)
        carry_cast = (z_curr, log_joint_prev, damping, active)

        def _do_step(_):
            damping_curr = damping
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
            system_diag = (
                prior_diag + obs_diag + damping_curr * jnp.eye(D, dtype=z_curr.dtype)[None, :, :]
            )
            system_upper = prior_upper + obs_upper
            system_rhs = prior_rhs + obs_rhs
            with jax.named_scope("laplace_em/support_aware_solve"):
                chol_diag, lower = _factor_block_banded_cholesky(system_diag, system_upper)
                z_newton = _solve_block_banded_from_cholesky(chol_diag, lower, system_rhs)

            step_direction = z_newton - z_curr
            z_next, log_joint_next, accepted, accepted_alpha = _support_aware_step_halving_search(
                z_curr,
                step_direction,
                log_joint_prev,
                lambda z: _support_aware_joint_log_prob(
                    z,
                    observations=observations,
                    obs_mask=obs_mask,
                    Ad=Ad,
                    Qd=Qd,
                    cd=cd,
                    H=H,
                    d=d,
                    R=R,
                    init_mean=init_mean,
                    init_cov=init_cov,
                    obs_kernel=obs_kernel,
                    mean_log_prob_fn=mean_log_prob_fn,
                    observation_support=observation_support,
                ),
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
                lambda _: (
                    damping_next < jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_MAX, dtype=z_curr.dtype)
                ),
                operand=None,
            )
            return z_next, log_joint_next, damping_next, next_active

        return jax.lax.cond(active, _do_step, lambda _: carry_cast, operand=None), None

    with jax.named_scope("laplace_em/support_aware_newton"):
        (z_est, log_joint_curr, _damping, _active), _ = jax.lax.scan(
            _newton_step,
            (
                z_est,
                log_joint_curr,
                jnp.asarray(_SUPPORT_AWARE_LM_DAMPING, dtype=z_est.dtype),
                jnp.asarray(True),
            ),
            xs=jnp.arange(max(n_ieks_iters, 1)),
        )

    mode_log_joint = log_joint_curr

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
    log_lik = (
        mode_log_joint
        + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi)
        - 0.5 * _block_banded_logdet(chol_diag)
    )
    return log_lik, z_est


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
    z_init: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sparse/banded support-aware IEKS + Laplace approximation."""
    log_lik, _z_mode = _support_aware_ieks_laplace(
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
        support_windows,
        bandwidth,
        n_ieks_iters,
        z_init=z_init,
    )
    return log_lik


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

    with jax.named_scope("laplace_em/dense_support_init"):
        z_init = _predictive_latent_init(Ad, cd, init_mean)

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
        self._support_mode_cache: jnp.ndarray | None = None
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
                state_lens=jnp.zeros((0,), dtype=jnp.int64),
                anchor_indices=jnp.zeros((0,), dtype=jnp.int64),
                start_indices=jnp.zeros((0,), dtype=jnp.int64),
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
            cache_inputs = (
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                obs_mask,
                extra_params,
            )
            can_reuse_support_mode = not _tree_contains_tracer(cache_inputs)
            support_mode_init = None
            if (
                can_reuse_support_mode
                and self._support_mode_cache is not None
                and self._support_mode_cache.shape == (clean_obs.shape[0], self.n_latent)
            ):
                support_mode_init = self._support_mode_cache
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
                log_lik, z_mode = _support_aware_ieks_laplace(
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
                    z_init=support_mode_init,
                )
                if can_reuse_support_mode:
                    self._support_mode_cache = jax.device_get(z_mode)
                return log_lik

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
# Canonical Laplace-EM: optimizer-backed parameter mode + Laplace posterior
# ---------------------------------------------------------------------------


def _build_laplace_em_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    trace_key: jnp.ndarray,
    likelihood_backend,
    reparam,
) -> dict[str, Any]:
    """Build the traced/JITed artifacts for optimizer-backed Laplace-EM."""
    site_info = _discover_sites(
        model,
        observations,
        times,
        trace_key,
        likelihood_backend,
        reparam=reparam,
    )
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)

    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model,
        observations,
        times,
        site_info,
        unravel_fn,
        likelihood_backend=likelihood_backend,
        reparam=reparam,
    )

    safe_floor = jnp.asarray(-1e30, dtype=observations.dtype)
    safe_ceiling = jnp.asarray(1e30, dtype=observations.dtype)

    def _log_posterior_fn(z: jnp.ndarray) -> jnp.ndarray:
        total = log_prior_unc_fn(z) + log_lik_fn(z)
        return jnp.where(jnp.isfinite(total), total, safe_floor)

    def _neg_log_posterior_fn(z: jnp.ndarray) -> jnp.ndarray:
        value = -_log_posterior_fn(z)
        return jnp.where(jnp.isfinite(value), value, safe_ceiling)

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "log_lik_fn": log_lik_fn,
        "log_prior_unc_fn": log_prior_unc_fn,
        "log_posterior_fn": _log_posterior_fn,
        "neg_log_posterior_fn": _neg_log_posterior_fn,
        "batch_log_posterior_jit": jax.jit(jax.vmap(_log_posterior_fn)),
    }


def _draw_laplace_init_candidates(
    rng_key: jnp.ndarray,
    site_info: dict[str, Any],
    *,
    dim: int,
    n_candidates: int,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample candidate parameter vectors from the prior in unconstrained space."""
    n_candidates = max(int(n_candidates), 1)
    if dim == 0:
        return rng_key, jnp.zeros((1, 0), dtype=dtype)

    parts = []
    for name in sorted(site_info.keys()):
        info = site_info[name]
        rng_key, sample_key = random.split(rng_key)
        constrained = info["distribution"].sample(sample_key, (n_candidates,))
        unconstrained = info["transform"].inv(constrained)
        parts.append(unconstrained.reshape(n_candidates, -1))

    candidates = jnp.concatenate(parts, axis=1)
    zeros = jnp.zeros((1, dim), dtype=candidates.dtype)
    return rng_key, jnp.concatenate([zeros, candidates], axis=0)


def _requires_support_aware_outer_optimizer(model) -> bool:
    """Use the support-aware outer optimizer for interval-summary models."""
    observation_support = getattr(model, "observation_support", None)
    return bool(
        observation_support is not None and observation_support.requires_interval_summary_handling
    )


def _optimize_laplace_parameter_mode(
    model,
    *,
    init_key: jnp.ndarray,
    dim: int,
    flat_example: jnp.ndarray,
    site_info: dict[str, Any],
    log_posterior_fn,
    neg_log_posterior_fn,
    batch_log_posterior_jit,
    observations: jnp.ndarray,
    n_init_samples: int,
    maxiter: int,
    tol: float,
) -> LaplaceModeOptimizationResult:
    """Find the parameter mode using the route appropriate for the model class."""
    if dim == 0:
        z_mode = flat_example
        return LaplaceModeOptimizationResult(
            z_mode=z_mode,
            objective_at_mode=float(jax.device_get(neg_log_posterior_fn(z_mode))),
            n_iters=0,
            n_function_evals=1,
            status=0,
            success=True,
            optimizer="BFGS",
            init_log_posterior_best=float(jax.device_get(log_posterior_fn(z_mode))),
        )

    if _requires_support_aware_outer_optimizer(model):
        z_init = flat_example
        init_log_posterior_best = float(jax.device_get(log_posterior_fn(z_init)))
        value_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_posterior_fn))
        cached_x: np.ndarray | None = None
        cached_fun: float | None = None
        cached_grad: np.ndarray | None = None

        def _value_and_grad(z_np: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal cached_x, cached_fun, cached_grad
            z_host = np.asarray(z_np, dtype=np.float64)
            if cached_x is not None and np.array_equal(z_host, cached_x):
                assert cached_fun is not None
                assert cached_grad is not None
                return cached_fun, cached_grad

            z = jnp.asarray(z_host, dtype=z_init.dtype)
            fun, grad = value_and_grad_fn(z)
            cached_x = z_host.copy()
            cached_fun = float(jax.device_get(fun))
            cached_grad = np.asarray(jax.device_get(grad), dtype=np.float64)
            return cached_fun, cached_grad

        def _objective(z_np: np.ndarray) -> float:
            fun, _grad = _value_and_grad(z_np)
            return fun

        def _gradient(z_np: np.ndarray) -> np.ndarray:
            _fun, grad = _value_and_grad(z_np)
            return grad

        opt_result = spo.minimize(
            _objective,
            x0=np.asarray(jax.device_get(z_init), dtype=np.float64),
            jac=_gradient,
            method="L-BFGS-B",
            tol=tol,
            options={"maxiter": maxiter, "disp": False},
        )
        z_mode = jnp.asarray(opt_result.x, dtype=z_init.dtype)
        return LaplaceModeOptimizationResult(
            z_mode=z_mode,
            objective_at_mode=float(opt_result.fun),
            n_iters=int(opt_result.nit),
            n_function_evals=int(opt_result.nfev),
            status=int(opt_result.status),
            success=bool(opt_result.success),
            optimizer="L-BFGS-B",
            init_log_posterior_best=init_log_posterior_best,
        )

    init_key, candidates = _draw_laplace_init_candidates(
        init_key,
        site_info,
        dim=dim,
        n_candidates=n_init_samples,
        dtype=observations.dtype,
    )
    del init_key
    init_scores = batch_log_posterior_jit(candidates)
    init_idx = int(jnp.argmax(init_scores))
    z_init = candidates[init_idx]

    opt_result = jso.minimize(
        neg_log_posterior_fn,
        z_init,
        method="BFGS",
        tol=tol,
        options={"maxiter": maxiter},
    )
    return LaplaceModeOptimizationResult(
        z_mode=opt_result.x,
        objective_at_mode=float(jax.device_get(opt_result.fun)),
        n_iters=int(opt_result.nit),
        n_function_evals=int(opt_result.nfev),
        status=int(opt_result.status),
        success=bool(jax.device_get(opt_result.success)),
        optimizer="BFGS",
        init_log_posterior_best=float(jax.device_get(init_scores[init_idx])),
    )


def _sample_laplace_parameter_posterior(
    rng_key: jnp.ndarray,
    z_mode: jnp.ndarray,
    neg_log_posterior_fn,
    *,
    num_samples: int,
    hessian_jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample an unconstrained Gaussian approximation around the parameter mode."""
    if num_samples < 1:
        raise ValueError("laplace_em requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    if dim == 0:
        return (
            jnp.zeros((num_samples, 0), dtype=z_mode.dtype),
            jnp.zeros((0, 0), dtype=z_mode.dtype),
            jnp.zeros((0,), dtype=z_mode.dtype),
        )

    with jax.named_scope("laplace_em/parameter_hessian"):
        hessian = jax.hessian(neg_log_posterior_fn)(z_mode)
        hessian = symmetrize_with_jitter(hessian, jitter=hessian_jitter)
        covariance = jla.solve(hessian, jnp.eye(dim, dtype=hessian.dtype), assume_a="pos")
        covariance = symmetrize_with_jitter(covariance, jitter=hessian_jitter)
        chol_cov = jnp.linalg.cholesky(covariance)

    with jax.named_scope("laplace_em/parameter_sampling"):
        eps = random.normal(rng_key, (num_samples, dim), dtype=z_mode.dtype)
        unc_samples = z_mode[None, :] + eps @ chol_cov.T

    return unc_samples, covariance, jnp.linalg.eigvalsh(hessian)


def _tree_contains_tracer(tree: Any) -> bool:
    """Whether any leaf in a pytree is currently a JAX tracer."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(tree))


def fit_laplace_em(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    num_samples: int = 1000,
    num_warmup: int | None = None,  # noqa: ARG001
    num_chains: int | None = None,  # noqa: ARG001
    seed: int = 0,
    n_ieks_iters: int = 5,
    maxiter: int = 100,
    tol: float = 1e-4,
    n_init_samples: int = 32,
    hessian_jitter: float = 1e-4,
    reparam=None,
    **kwargs: Any,
) -> InferenceResult:
    """Fit an approximate posterior with KFAS-style Laplace optimization.

    The latent-state side uses the existing IEKS/Laplace marginal likelihood
    backend. The outer loop then mirrors KFAS/Helske's optimizer-backed
    `fitSSM` pattern: find the parameter mode of the approximate marginal
    posterior, compute the local curvature there, and sample the resulting
    Gaussian approximation in unconstrained parameter space.
    """
    for ignored_key in ("svi_config", "nuts_config", "smc_config"):
        kwargs.pop(ignored_key, None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"fit_laplace_em got unexpected keyword arguments: {unknown}")

    rng_key = random.PRNGKey(seed)
    rng_key, trace_key, init_key, sample_key = random.split(rng_key, 4)

    backend_label = "kalman" if model.likelihood == "kalman" else "laplace_ieks"
    logger.info(
        "Laplace-EM config: backend=%s maxiter=%s tol=%s n_ieks_iters=%s "
        "n_init_samples=%s num_samples=%s",
        backend_label,
        maxiter,
        tol,
        n_ieks_iters,
        n_init_samples,
        num_samples,
    )

    with jax.profiler.TraceAnnotation("laplace_em/build_likelihood_backend"):
        if model.likelihood == "kalman":
            backend = model.make_likelihood_backend()
        else:
            backend = model.make_laplace_backend(n_ieks_iters)

    with jax.profiler.TraceAnnotation("laplace_em/build_bundle"):
        bundle = _build_laplace_em_bundle(
            model,
            observations,
            times,
            trace_key,
            backend,
            reparam,
        )

    dim = bundle["dim"]
    flat_example = bundle["flat_example"]
    site_info = bundle["site_info"]
    unravel_fn = bundle["unravel_fn"]
    log_lik_fn = bundle["log_lik_fn"]
    log_prior_unc_fn = bundle["log_prior_unc_fn"]
    log_posterior_fn = bundle["log_posterior_fn"]
    neg_log_posterior_fn = bundle["neg_log_posterior_fn"]
    batch_log_posterior_jit = bundle["batch_log_posterior_jit"]

    optimizer_name = "L-BFGS-B" if _requires_support_aware_outer_optimizer(model) else "BFGS"
    logger.info(
        "Laplace-EM outer optimizer: method=%s support_aware=%s",
        optimizer_name,
        _requires_support_aware_outer_optimizer(model),
    )
    with jax.profiler.TraceAnnotation("laplace_em/parameter_optimize"):
        mode_result = _optimize_laplace_parameter_mode(
            model,
            init_key=init_key,
            dim=dim,
            flat_example=flat_example,
            site_info=site_info,
            log_posterior_fn=log_posterior_fn,
            neg_log_posterior_fn=neg_log_posterior_fn,
            batch_log_posterior_jit=batch_log_posterior_jit,
            observations=observations,
            n_init_samples=n_init_samples,
            maxiter=maxiter,
            tol=tol,
        )

    z_mode = mode_result.z_mode
    mode_objective = mode_result.objective_at_mode
    nit = mode_result.n_iters
    nfev = mode_result.n_function_evals
    status = mode_result.status
    success = mode_result.success
    logger.info(
        "Laplace-EM mode found: optimizer=%s success=%s nit=%s nfev=%s objective=%.6f",
        mode_result.optimizer,
        success,
        nit,
        nfev,
        mode_objective,
    )

    mode_log_posterior = float(jax.device_get(log_posterior_fn(z_mode)))
    mode_log_likelihood = float(jax.device_get(log_lik_fn(z_mode)))
    mode_log_prior = float(jax.device_get(log_prior_unc_fn(z_mode)))
    if not np.isfinite(mode_log_posterior):
        raise RuntimeError("Laplace-EM failed to find a finite parameter mode.")

    logger.info(
        "Laplace-EM parameter Hessian: dim=%s sampling local Gaussian posterior",
        dim,
    )
    with jax.profiler.TraceAnnotation("laplace_em/sample_parameter_posterior"):
        unc_samples, covariance, hessian_eigvals = _sample_laplace_parameter_posterior(
            sample_key,
            z_mode,
            neg_log_posterior_fn,
            num_samples=num_samples,
            hessian_jitter=hessian_jitter,
        )
    logger.info("Laplace-EM parameter Hessian complete")

    if site_info:
        with jax.profiler.TraceAnnotation("laplace_em/extract_samples"):
            samples = extract_constrained_samples(
                unc_samples,
                site_info,
                unravel_fn,
                model.spec,
                reparam=reparam,
                model=model,
                observations=observations,
                times=times,
            )
    else:
        prior_runtime = model.get_prior_runtime_bundle()
        samples = assemble_deterministics_from_registry(
            {},
            model.spec,
            prior_runtime.registry,
            structure_runtime=model.structure_runtime,
            n_draws=num_samples,
        )

    hessian_condition_number = None
    if hessian_eigvals.size > 0:
        min_eig = float(jax.device_get(jnp.min(hessian_eigvals)))
        max_eig = float(jax.device_get(jnp.max(hessian_eigvals)))
        if min_eig > 0.0:
            hessian_condition_number = max_eig / min_eig

    diagnostics = {
        "optimizer": mode_result.optimizer,
        "success": success,
        "status": status,
        "n_iters": nit,
        "n_function_evals": nfev,
        "objective_at_mode": mode_objective,
        "mode_log_posterior": mode_log_posterior,
        "mode_log_likelihood": mode_log_likelihood,
        "mode_log_prior": mode_log_prior,
        "init_log_posterior_best": mode_result.init_log_posterior_best,
        "n_init_samples": n_init_samples,
        "n_ieks_iters": n_ieks_iters,
        "hessian_jitter": hessian_jitter,
        "hessian_condition_number": hessian_condition_number,
        "covariance_diag": np.asarray(jnp.diag(covariance)).tolist(),
        "likelihood_backend": backend,
    }

    logger.info(
        "Laplace-EM complete: success=%s status=%s nit=%s nfev=%s loglik=%.3f logpost=%.3f",
        success,
        status,
        nit,
        nfev,
        mode_log_likelihood,
        mode_log_posterior,
    )

    return InferenceResult(
        _samples=samples,
        method="laplace_em",
        diagnostics=diagnostics,
    )
