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

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import scipy.optimize as spo
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import (
    discretize_linear_system_exact_batched,
    discretize_system_batched,
)
from causal_ssm_agent.models.ssm.inference.targets.base import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_KALMAN_EXACT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    build_likelihood_eval_aux,
)
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
class LinearSummaryAccumulatorPlan:
    """Runtime plan for exact augmented-state interval summaries."""

    accumulator_manifest_indices: jnp.ndarray
    row_reset_mask: jnp.ndarray
    row_emission_accumulator_indices: jnp.ndarray
    row_emission_scales: jnp.ndarray

    @property
    def n_accumulators(self) -> int:
        return int(self.accumulator_manifest_indices.shape[0])


@dataclass(frozen=True)
class GaussianTrajectoryPriorTerms:
    """Precomputed Gaussian factors for the latent trajectory prior."""

    init_mean: jnp.ndarray
    init_chol: jnp.ndarray
    init_logdet: jnp.ndarray
    transition_chol: jnp.ndarray
    transition_logdet: jnp.ndarray


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
    optimizer_hess_inv: Any | None = None
    final_grad_norm: float | None = None
    final_eval_diagnostics: dict[str, Any] | None = None


_LINEAR_SUMMARY_SUPPORTED_DISTS = frozenset({"gaussian", "student_t"})
_LINEAR_SUMMARY_SUPPORTED_OPERATORS = frozenset({"mean", "sum"})


def _should_use_dense_support_laplace(*, n_time: int, n_latent: int) -> bool:
    """Use the dense exact support-aware Newton system on short trajectories.

    The banded support-aware path pays substantial Python/autodiff overhead per
    anchored window. For small latent trajectories, a single dense exact Hessian
    over the full latent path is materially faster and preserves the same model.
    """
    return n_time * n_latent <= _DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM


_SOLVER_KIND_LABELS = {
    LIKELIHOOD_SOLVER_KIND_KALMAN_EXACT: "kalman_exact",
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS: "point_ieks",
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS: "support_ieks",
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT: "dense_support",
}


def _elapsed_seconds(start: float) -> float:
    return time.monotonic() - start


def _scalar_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value, dtype=np.int64))


def _format_float(value: float | None, fmt: str = ".3e") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def _solver_label(kind: int) -> str:
    return _SOLVER_KIND_LABELS.get(kind, f"solver_{kind}")


def _hostify_inner_eval_diagnostics(aux: dict[str, Any]) -> dict[str, Any]:
    host = jax.device_get(aux)
    return {
        "solver_kind": _scalar_int(host["solver_kind"]),
        "n_iterations": _scalar_int(host["n_iterations"]),
        "n_accepted_steps": _scalar_int(host["n_accepted_steps"]),
        "init_log_joint": _scalar_float(host["init_log_joint"]),
        "final_log_joint": _scalar_float(host["final_log_joint"]),
        "final_rel_change": _scalar_float(host["final_rel_change"]),
        "final_damping": _scalar_float(host["final_damping"]),
        "final_step_alpha": _scalar_float(host["final_step_alpha"]),
        "final_step_norm": _scalar_float(host["final_step_norm"]),
        "laplace_logdet": _scalar_float(host["laplace_logdet"]),
        "min_chol_diag": _scalar_float(host["min_chol_diag"]),
    }


def _hostify_outer_eval_diagnostics(aux: dict[str, Any]) -> dict[str, Any]:
    host = jax.device_get(aux)
    return {
        "log_posterior": _scalar_float(host["log_posterior"]),
        "log_likelihood": _scalar_float(host["log_likelihood"]),
        "log_prior": _scalar_float(host["log_prior"]),
        "inner": _hostify_inner_eval_diagnostics(host["inner"]),
    }


def _inner_log_joint_gain(inner: dict[str, Any]) -> float | None:
    init_log_joint = inner["init_log_joint"]
    final_log_joint = inner["final_log_joint"]
    if not np.isfinite(init_log_joint) or not np.isfinite(final_log_joint):
        return None
    return final_log_joint - init_log_joint


def _log_outer_eval(
    *,
    label: str,
    elapsed_seconds: float,
    eval_count: int,
    objective: float,
    best_objective: float,
    delta_objective: float | None,
    grad_norm: float,
    step_norm: float | None,
    outer_diag: dict[str, Any],
) -> None:
    logger.info(
        "Laplace-EM outer %s: elapsed=%.1fs evals=%d objective=%.6f best=%.6f "
        "delta=%s grad_norm=%s step_norm=%s logpost=%.6f loglik=%.6f logprior=%.6f",
        label,
        elapsed_seconds,
        eval_count,
        objective,
        best_objective,
        _format_float(delta_objective),
        _format_float(grad_norm),
        _format_float(step_norm),
        outer_diag["log_posterior"],
        outer_diag["log_likelihood"],
        outer_diag["log_prior"],
    )
    inner = outer_diag["inner"]
    logger.info(
        "Laplace-EM inner %s: solver=%s n_iters=%d accepted=%d rel_change=%s "
        "damping=%s alpha=%s latent_gain=%s laplace_logdet=%s min_chol_diag=%s",
        label,
        _solver_label(inner["solver_kind"]),
        inner["n_iterations"],
        inner["n_accepted_steps"],
        _format_float(inner["final_rel_change"]),
        _format_float(inner["final_damping"]),
        _format_float(inner["final_step_alpha"]),
        _format_float(_inner_log_joint_gain(inner)),
        _format_float(inner["laplace_logdet"]),
        _format_float(inner["min_chol_diag"]),
    )


def _symmetrize_psd(mats: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize square matrices and optionally add diagonal jitter."""
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)
    return 0.5 * (mats + jnp.swapaxes(mats, -1, -2)) + jitter * eye


def _enum_value_lower(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _find_support_time_index(anchor_times: np.ndarray, target_time: float, *, tol: float = 1e-8) -> int:
    candidate_idx = int(np.searchsorted(anchor_times, target_time))
    if candidate_idx < len(anchor_times) and abs(float(anchor_times[candidate_idx]) - target_time) <= tol:
        return candidate_idx
    if candidate_idx > 0 and abs(float(anchor_times[candidate_idx - 1]) - target_time) <= tol:
        return candidate_idx - 1
    raise ValueError(
        "Linear interval-summary augmentation requires support boundaries to be present "
        f"on the model clock; missing boundary at time={target_time}."
    )


def _build_linear_summary_accumulator_plan(
    observation_support: ObservationSupportRuntime | None,
    manifest_dists: list[Any],
    manifest_links: list[Any],
) -> LinearSummaryAccumulatorPlan | None:
    """Return an augmentation plan when all interval summaries are linear in the latent state."""
    if observation_support is None or not observation_support.requires_interval_summary_handling:
        return None

    interval_manifest_indices: list[int] = []
    for manifest_idx, support_kind in enumerate(observation_support.support_kinds):
        if support_kind != "interval":
            continue
        dist_value = _enum_value_lower(manifest_dists[manifest_idx])
        link_value = _enum_value_lower(manifest_links[manifest_idx])
        summary_value = _enum_value_lower(observation_support.summary_operators[manifest_idx])
        if (
            dist_value not in _LINEAR_SUMMARY_SUPPORTED_DISTS
            or link_value != "identity"
            or summary_value not in _LINEAR_SUMMARY_SUPPORTED_OPERATORS
        ):
            return None
        interval_manifest_indices.append(manifest_idx)

    if not interval_manifest_indices:
        return None

    emission_slots = np.asarray(observation_support.emission_slot_indices)
    anchor_times = np.asarray(observation_support.anchor_times)
    support_start_times = np.asarray(observation_support.support_start_times)
    support_end_times = np.asarray(observation_support.support_end_times)
    support_summary_ops = list(observation_support.summary_operators)

    slot_to_accumulator: dict[tuple[int, int], int] = {}
    accumulator_manifest_indices: list[int] = []
    for manifest_idx in interval_manifest_indices:
        used_slots = sorted(
            {
                int(slot_idx)
                for slot_idx in emission_slots[:, manifest_idx].tolist()
                if int(slot_idx) >= 0
            }
        )
        if not used_slots:
            continue
        for slot_idx in used_slots:
            slot_to_accumulator[(manifest_idx, slot_idx)] = len(accumulator_manifest_indices)
            accumulator_manifest_indices.append(manifest_idx)

    if not accumulator_manifest_indices:
        return None

    n_time = int(anchor_times.shape[0])
    n_manifest = len(observation_support.manifest_names)
    n_accumulators = len(accumulator_manifest_indices)
    row_reset_mask = np.zeros((n_time, n_accumulators), dtype=bool)
    row_emission_accumulator_indices = np.full((n_time, n_manifest), -1, dtype=np.int64)
    row_emission_scales = np.zeros((n_time, n_manifest), dtype=np.float64)

    for anchor_idx in range(n_time):
        for manifest_idx in interval_manifest_indices:
            slot_idx = int(emission_slots[anchor_idx, manifest_idx])
            if slot_idx < 0:
                continue
            accumulator_idx = slot_to_accumulator[(manifest_idx, slot_idx)]
            support_start = float(support_start_times[anchor_idx, manifest_idx])
            support_end = float(support_end_times[anchor_idx, manifest_idx])
            if not np.isfinite(support_start) or not np.isfinite(support_end):
                raise ValueError(
                    "Linear interval-summary augmentation requires finite support bounds for "
                    f"manifest={observation_support.manifest_names[manifest_idx]!r} row={anchor_idx}."
                )
            duration = support_end - support_start
            if duration <= 1e-8:
                raise ValueError(
                    "Linear interval-summary augmentation requires positive support length for "
                    f"manifest={observation_support.manifest_names[manifest_idx]!r} row={anchor_idx}."
                )
            start_idx = _find_support_time_index(anchor_times, support_start)
            row_reset_mask[start_idx, accumulator_idx] = True
            row_emission_accumulator_indices[anchor_idx, manifest_idx] = accumulator_idx
            summary_value = _enum_value_lower(support_summary_ops[manifest_idx])
            row_emission_scales[anchor_idx, manifest_idx] = (
                1.0 / duration if summary_value == "mean" else 1.0
            )

    return LinearSummaryAccumulatorPlan(
        accumulator_manifest_indices=jnp.asarray(accumulator_manifest_indices, dtype=jnp.int64),
        row_reset_mask=jnp.asarray(row_reset_mask),
        row_emission_accumulator_indices=jnp.asarray(
            row_emission_accumulator_indices,
            dtype=jnp.int64,
        ),
        row_emission_scales=jnp.asarray(row_emission_scales),
    )


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


def _build_gaussian_trajectory_prior_terms(
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


def _trajectory_prior_log_prob_from_terms(
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
            state_lens=jnp.asarray(state_lens, dtype=jnp.int64),
            anchor_indices=jnp.asarray(anchor_indices, dtype=jnp.int64),
            start_indices=jnp.asarray(start_indices, dtype=jnp.int64),
            mask_full=jnp.asarray(mask_full),
            prev_coeffs=jnp.asarray(padded_prev),
            curr_coeffs=jnp.asarray(padded_curr),
            weights=jnp.asarray(padded_weights),
            padded_state_indices=jnp.asarray(padded_state_indices, dtype=jnp.int64),
            time_indices=jnp.asarray(time_indices, dtype=jnp.int64),
            valid_diag=jnp.asarray(valid_diag),
            cross_time_indices=jnp.asarray(cross_time_indices, dtype=jnp.int64),
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        reset_scales = reset_scales.at[:, n_latent:].set(
            1.0 - plan.row_reset_mask.astype(dtype)
        )
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
                z_newton = _solve_block_tridiagonal(lower, diag, upper, rhs)

            step_direction = z_newton - z_curr
            step_norm = jnp.linalg.norm(step_direction)
            z_next, log_joint_next, accepted, accepted_alpha = _step_halving_search(
                z_curr,
                step_direction,
                log_joint_prev,
                _row_log_joint,
                max_halvings=_POINT_LINE_SEARCH_MAX_HALVINGS,
            )

            rel_change = jnp.linalg.norm(z_next - z_curr) / (1.0 + jnp.linalg.norm(z_curr))
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

    return jax.lax.cond(zero_step, _zero_step_result, _run_search, operand=None)


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
            with jax.named_scope("laplace_em/support_aware_solve"):
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
                lambda _: (
                    damping_next < jnp.asarray(_SUPPORT_AWARE_LM_DAMPING_MAX, dtype=z_curr.dtype)
                ),
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

    with jax.named_scope("laplace_em/support_aware_newton"):
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
            _newton_step,
            (
                z_est,
                log_joint_curr,
                jnp.asarray(_SUPPORT_AWARE_LM_DAMPING, dtype=z_est.dtype),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
                jnp.asarray(jnp.nan, dtype=z_est.dtype),
            ),
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
    with jax.named_scope("laplace_em/support_aware_final_hessian"):
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

    # The support-aware Laplace path constructs runtime callables and custom-VJP
    # closures that are not remat-safe under large traced outer evaluations.
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
        self._point_mode_cache: jnp.ndarray | None = None
        self._support_mode_cache: jnp.ndarray | None = None
        self._linear_summary_mode_cache: jnp.ndarray | None = None
        self._support_window_derivatives = None
        self._support_window_derivatives_signature: tuple[Any, ...] | None = None
        self._linear_summary_plan = _build_linear_summary_accumulator_plan(
            observation_support,
            manifest_dists,
            manifest_links,
        )
        if observation_support is not None:
            self._support_kind_codes = get_support_kind_codes(observation_support)
            self._summary_operator_codes = get_summary_operator_codes(observation_support)
        else:
            self._support_kind_codes = jnp.zeros((n_manifest,), dtype=jnp.int64)
            self._summary_operator_codes = jnp.zeros((n_manifest,), dtype=jnp.int64)
        if (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        ):
            (
                self._support_window_batches,
                self._support_bandwidth,
                support_row_upper_bandwidths,
            ) = _infer_support_groups(observation_support)
            prior_row_upper_bandwidths = np.zeros(
                (len(observation_support.anchor_times),),
                dtype=np.int64,
            )
            if len(prior_row_upper_bandwidths) > 1:
                prior_row_upper_bandwidths[:-1] = 1
            full_row_upper_bandwidths = np.maximum(
                np.asarray(support_row_upper_bandwidths, dtype=np.int64),
                prior_row_upper_bandwidths,
            )
            self._support_row_upper_bandwidths = jnp.asarray(
                full_row_upper_bandwidths,
                dtype=jnp.int32,
            )
            self._support_row_lower_bandwidths = jnp.asarray(
                _compute_profile_lower_bandwidths(full_row_upper_bandwidths),
                dtype=jnp.int32,
            )
        else:
            self._support_window_batches = ()
            self._support_bandwidth = 1 if n_latent > 0 else 0
            self._support_row_upper_bandwidths = jnp.zeros((0,), dtype=jnp.int32)
            self._support_row_lower_bandwidths = jnp.zeros((0,), dtype=jnp.int32)

    def _build_support_window_derivatives(self, measurement_semantics) -> tuple[Any, ...]:
        return tuple(
            _make_support_window_derivatives(
                max_state_len=batch.max_state_len,
                n_latent=self.n_latent,
                n_manifest=self.n_manifest,
                summary_operator_codes=self._summary_operator_codes,
                obs_kernel=measurement_semantics.obs_kernel,
                mean_log_prob_fn=measurement_semantics.mean_log_prob_fn,
            )
            for batch in self._support_window_batches
        )

    def _get_support_window_derivatives(
        self,
        measurement_semantics,
        extra_params: dict | None,
        *,
        allow_cache: bool,
    ):
        if not allow_cache or extra_params is not None:
            return self._build_support_window_derivatives(measurement_semantics)

        signature = (
            measurement_semantics.manifest_dists,
            measurement_semantics.manifest_links,
            tuple(batch.max_state_len for batch in self._support_window_batches),
            self.n_latent,
            self.n_manifest,
        )
        if (
            self._support_window_derivatives is None
            or self._support_window_derivatives_signature != signature
        ):
            self._support_window_derivatives = self._build_support_window_derivatives(
                measurement_semantics
            )
            self._support_window_derivatives_signature = signature
        return self._support_window_derivatives

    def _compute_log_likelihood_impl(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        *,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
        include_aux: bool,
        allow_stateful_cache: bool,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray] | None]:
        """Shared Laplace likelihood implementation with explicit cache control."""
        n = self.n_latent

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        with jax.named_scope("laplace_em/compile_measurement_semantics"):
            measurement_semantics = compile_measurement_semantics(
                self.manifest_dists,
                manifest_cov=measurement_params.manifest_cov,
                extra_params=extra_params,
                manifest_links=self.manifest_links,
                observation_support=self.observation_support,
            )
        obs_kernel = measurement_semantics.obs_kernel

        def _discretize_base_system() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            return Ad, Qd, cd

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
            if self._linear_summary_plan is not None:
                can_reuse_linear_summary_mode = allow_stateful_cache and not _tree_contains_tracer(
                    cache_inputs
                )
                linear_summary_dim = self.n_latent + self._linear_summary_plan.n_accumulators
                linear_summary_mode_init = latent_mode_init
                if linear_summary_mode_init is not None and linear_summary_mode_init.shape != (
                    clean_obs.shape[0],
                    linear_summary_dim,
                ):
                    raise ValueError(
                        "Linear interval-summary warm start shape does not match the "
                        f"augmented latent dimension: expected {(clean_obs.shape[0], linear_summary_dim)}, "
                        f"received {tuple(linear_summary_mode_init.shape)}."
                    )
                if (
                    linear_summary_mode_init is None
                    and can_reuse_linear_summary_mode
                    and self._linear_summary_mode_cache is not None
                    and self._linear_summary_mode_cache.shape == (clean_obs.shape[0], linear_summary_dim)
                ):
                    linear_summary_mode_init = self._linear_summary_mode_cache
                with jax.named_scope("laplace_em/linear_summary_augmented_backend"):
                    z_mode, log_lik, inner_eval_aux = _linear_summary_augmented_ieks_laplace(
                        clean_obs,
                        obs_mask,
                        time_intervals,
                        ct_params.drift,
                        ct_params.diffusion_cov,
                        ct_params.cint,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        initial_state.cov,
                        obs_kernel,
                        self._linear_summary_plan,
                        self._support_kind_codes,
                        self.n_ieks_iters,
                        z_init=linear_summary_mode_init,
                    )
                    if can_reuse_linear_summary_mode:
                        self._linear_summary_mode_cache = jax.device_get(z_mode)
                    return log_lik, inner_eval_aux if include_aux else None

            def _build_support_measurement_objects(
                manifest_cov: jnp.ndarray,
                runtime_extra_params: dict | None,
            ):
                runtime_measurement_semantics = compile_measurement_semantics(
                    self.manifest_dists,
                    manifest_cov=manifest_cov,
                    extra_params=runtime_extra_params,
                    manifest_links=self.manifest_links,
                    observation_support=self.observation_support,
                )
                allow_runtime_cache = allow_stateful_cache and not _tree_contains_tracer(
                    (manifest_cov, runtime_extra_params)
                )
                return runtime_measurement_semantics, self._get_support_window_derivatives(
                    runtime_measurement_semantics,
                    runtime_extra_params,
                    allow_cache=allow_runtime_cache,
                )

            Ad, Qd, cd = _discretize_base_system()
            can_reuse_support_mode = allow_stateful_cache and not _tree_contains_tracer(
                cache_inputs
            )
            can_cache_window_derivatives = allow_stateful_cache and not _tree_contains_tracer(
                (measurement_params.manifest_cov, extra_params)
            )
            support_mode_init = latent_mode_init
            if (
                support_mode_init is None
                and can_reuse_support_mode
                and self._support_mode_cache is not None
                and self._support_mode_cache.shape == (clean_obs.shape[0], self.n_latent)
            ):
                support_mode_init = self._support_mode_cache
            if _should_use_dense_support_laplace(
                n_time=clean_obs.shape[0],
                n_latent=self.n_latent,
            ):
                with jax.named_scope("laplace_em/dense_support_backend"):
                    log_lik, inner_eval_aux = _dense_support_laplace_log_lik(
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
                return log_lik, inner_eval_aux if include_aux else None
            with jax.named_scope("laplace_em/support_aware_backend"):
                window_derivatives = self._get_support_window_derivatives(
                    measurement_semantics,
                    extra_params,
                    allow_cache=can_cache_window_derivatives,
                )
                log_lik, z_mode, inner_eval_aux = _support_aware_ieks_laplace(
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
                    self._support_window_batches,
                    self._support_bandwidth,
                    self._support_row_upper_bandwidths,
                    self._support_row_lower_bandwidths,
                    window_derivatives=window_derivatives,
                    build_measurement_objects=_build_support_measurement_objects,
                    extra_params=extra_params,
                    n_ieks_iters=self.n_ieks_iters,
                    z_init=support_mode_init,
                )
                if can_reuse_support_mode:
                    self._support_mode_cache = jax.device_get(z_mode)
                return log_lik, inner_eval_aux if include_aux else None

        cache_inputs = (
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask,
            extra_params,
        )
        can_reuse_point_mode = allow_stateful_cache and not _tree_contains_tracer(cache_inputs)
        point_mode_init = latent_mode_init
        if (
            point_mode_init is None
            and can_reuse_point_mode
            and self._point_mode_cache is not None
            and self._point_mode_cache.shape == (clean_obs.shape[0], self.n_latent)
        ):
            point_mode_init = self._point_mode_cache

        Ad, Qd, cd = _discretize_base_system()
        T_obs = clean_obs.shape[0]
        H_rows = jnp.broadcast_to(
            measurement_params.lambda_mat[None, :, :],
            (T_obs, *measurement_params.lambda_mat.shape),
        )
        d_rows = jnp.broadcast_to(
            measurement_params.manifest_means[None, :],
            (T_obs, *measurement_params.manifest_means.shape),
        )
        with jax.named_scope("laplace_em/ieks_backend"):
            z_mode, log_lik, inner_eval_aux = _ieks_smooth(
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                measurement_params.manifest_cov,
                initial_state.mean,
                initial_state.cov,
                obs_kernel,
                n_ieks_iters=self.n_ieks_iters,
                z_init=point_mode_init,
            )
            if can_reuse_point_mode:
                self._point_mode_cache = jax.device_get(z_mode)

        return log_lik, inner_eval_aux if include_aux else None

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute Laplace-approximated log-likelihood.

        Returns:
            (T,) cumulative log-normalizing constants, matching LikelihoodBackend protocol.
        """
        log_lik, _aux = self._compute_log_likelihood_impl(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            extra_params=extra_params,
            latent_mode_init=latent_mode_init,
            include_aux=False,
            allow_stateful_cache=False,
        )
        return log_lik

    def compute_log_likelihood_with_aux(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Laplace-approximated log-likelihood plus host-log aux."""
        log_lik, inner_eval_aux = self._compute_log_likelihood_impl(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            extra_params=extra_params,
            latent_mode_init=latent_mode_init,
            include_aux=True,
            allow_stateful_cache=True,
        )
        assert inner_eval_aux is not None
        return log_lik, inner_eval_aux


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

    log_lik_fn, log_prior_unc_fn, log_lik_with_aux_fn = _build_eval_fns(
        model,
        observations,
        times,
        site_info,
        unravel_fn,
        likelihood_backend=likelihood_backend,
        reparam=reparam,
        include_likelihood_aux=True,
    )

    safe_floor = jnp.asarray(-1e30, dtype=observations.dtype)
    safe_ceiling = jnp.asarray(1e30, dtype=observations.dtype)

    def _log_posterior_fn(z: jnp.ndarray, latent_mode_init=None) -> jnp.ndarray:
        total = log_prior_unc_fn(z) + log_lik_fn(z, latent_mode_init=latent_mode_init)
        return jnp.where(jnp.isfinite(total), total, safe_floor)

    def _neg_log_posterior_fn(z: jnp.ndarray, latent_mode_init=None) -> jnp.ndarray:
        value = -_log_posterior_fn(z, latent_mode_init=latent_mode_init)
        return jnp.where(jnp.isfinite(value), value, safe_ceiling)

    def _neg_log_posterior_with_aux_fn(
        z: jnp.ndarray,
        latent_mode_init=None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        log_lik, inner_eval_aux = log_lik_with_aux_fn(z, latent_mode_init=latent_mode_init)
        log_prior = log_prior_unc_fn(z)
        log_posterior = log_prior + log_lik
        neg_log_posterior = -log_posterior
        safe_value = jnp.where(jnp.isfinite(neg_log_posterior), neg_log_posterior, safe_ceiling)
        outer_aux = {
            "log_posterior": log_posterior,
            "log_likelihood": log_lik,
            "log_prior": log_prior,
            "inner": {key: value for key, value in inner_eval_aux.items() if key != "latent_mode"},
        }
        if "latent_mode" in inner_eval_aux:
            outer_aux["latent_mode"] = inner_eval_aux["latent_mode"]
        return safe_value, outer_aux

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "log_lik_fn": log_lik_fn,
        "log_prior_unc_fn": log_prior_unc_fn,
        "log_posterior_fn": _log_posterior_fn,
        "neg_log_posterior_fn": _neg_log_posterior_fn,
        "neg_log_posterior_with_aux_fn": _neg_log_posterior_with_aux_fn,
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
    _model,
    *,
    init_key: jnp.ndarray,
    dim: int,
    flat_example: jnp.ndarray,
    site_info: dict[str, Any],
    log_posterior_fn,
    neg_log_posterior_with_aux_fn,
    batch_log_posterior_jit,
    observations: jnp.ndarray,
    n_init_samples: int,
    maxiter: int,
    tol: float,
) -> LaplaceModeOptimizationResult:
    """Find the parameter mode using the route appropriate for the model class."""
    if dim == 0:
        z_mode = flat_example
        objective_at_mode, final_eval_aux = neg_log_posterior_with_aux_fn(
            z_mode, latent_mode_init=None
        )
        return LaplaceModeOptimizationResult(
            z_mode=z_mode,
            objective_at_mode=float(jax.device_get(objective_at_mode)),
            n_iters=0,
            n_function_evals=1,
            status=0,
            success=True,
            optimizer="L-BFGS-B",
            init_log_posterior_best=float(
                jax.device_get(log_posterior_fn(z_mode, latent_mode_init=None))
            ),
            optimizer_hess_inv=None,
            final_grad_norm=0.0,
            final_eval_diagnostics=_hostify_outer_eval_diagnostics(final_eval_aux),
        )

    support_aware_outer = _requires_support_aware_outer_optimizer(_model)
    if support_aware_outer:
        z_init = flat_example
        init_log_posterior_best: float | None = None
        logger.info("Laplace-EM init candidates skipped: support-aware outer optimizer")
    else:
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
        init_log_posterior_best = float(jax.device_get(init_scores[init_idx]))
        logger.info(
            "Laplace-EM init candidates: n_candidates=%d best_log_posterior=%.6f",
            int(candidates.shape[0]),
            init_log_posterior_best,
        )

    value_and_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda z, latent_mode_init: neg_log_posterior_with_aux_fn(
                z,
                latent_mode_init=latent_mode_init,
            ),
            argnums=0,
            has_aux=True,
        )
    )
    cached_x: np.ndarray | None = None
    cached_fun: float | None = None
    cached_grad: np.ndarray | None = None
    cached_aux: dict[str, Any] | None = None
    eval_count = 0
    optimize_started_at = time.monotonic()
    latent_mode_init: np.ndarray | None = None
    if support_aware_outer:
        _seed_objective, seed_aux = neg_log_posterior_with_aux_fn(z_init, latent_mode_init=None)
        del _seed_objective
        if "latent_mode" in seed_aux:
            latent_mode_init = np.asarray(jax.device_get(seed_aux["latent_mode"])).copy()
            logger.info("Laplace-EM seeded latent warm start before jitted value-and-grad compile")

    def _value_and_grad(z_np: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        nonlocal cached_x, cached_fun, cached_grad, cached_aux, eval_count, latent_mode_init
        z_host = np.asarray(z_np, dtype=np.float64)
        if cached_x is not None and np.array_equal(z_host, cached_x):
            assert cached_fun is not None
            assert cached_grad is not None
            assert cached_aux is not None
            return cached_fun, cached_grad, cached_aux

        z = jnp.asarray(z_host, dtype=z_init.dtype)
        latent_mode_arg = (
            None
            if latent_mode_init is None
            else jnp.asarray(latent_mode_init, dtype=observations.dtype)
        )
        (fun, aux), grad = value_and_grad_fn(z, latent_mode_arg)
        eval_count += 1
        cached_x = z_host.copy()
        cached_fun = float(jax.device_get(fun))
        cached_grad = np.asarray(jax.device_get(grad), dtype=np.float64)
        cached_aux = _hostify_outer_eval_diagnostics(aux)
        if "latent_mode" in aux:
            latent_mode_init = np.asarray(jax.device_get(aux["latent_mode"])).copy()
        else:
            latent_mode_init = None
        return cached_fun, cached_grad, cached_aux

    def _objective(z_np: np.ndarray) -> float:
        fun, _grad, _aux = _value_and_grad(z_np)
        return fun

    def _gradient(z_np: np.ndarray) -> np.ndarray:
        _fun, grad, _aux = _value_and_grad(z_np)
        return grad

    x0_np = np.asarray(jax.device_get(z_init), dtype=np.float64)
    init_fun, init_grad, init_aux = _value_and_grad(x0_np)
    if init_log_posterior_best is None:
        init_log_posterior_best = -init_fun
    _log_outer_eval(
        label="init",
        elapsed_seconds=_elapsed_seconds(optimize_started_at),
        eval_count=eval_count,
        objective=init_fun,
        best_objective=init_fun,
        delta_objective=None,
        grad_norm=float(np.linalg.norm(init_grad)),
        step_norm=None,
        outer_diag=init_aux,
    )

    iteration_count = 0
    best_objective = init_fun
    previous_fun = init_fun
    previous_x = x0_np.copy()

    def _callback(xk: np.ndarray) -> None:
        nonlocal iteration_count, best_objective, previous_fun, previous_x
        x_curr = np.asarray(xk, dtype=np.float64)
        fun, grad, aux = _value_and_grad(x_curr)
        iteration_count += 1
        best_objective = min(best_objective, fun)
        _log_outer_eval(
            label=f"iter {iteration_count}",
            elapsed_seconds=_elapsed_seconds(optimize_started_at),
            eval_count=eval_count,
            objective=fun,
            best_objective=best_objective,
            delta_objective=fun - previous_fun,
            grad_norm=float(np.linalg.norm(grad)),
            step_norm=float(np.linalg.norm(x_curr - previous_x)),
            outer_diag=aux,
        )
        previous_fun = fun
        previous_x = x_curr.copy()

    opt_result = spo.minimize(
        _objective,
        x0=x0_np,
        jac=_gradient,
        method="L-BFGS-B",
        tol=tol,
        options={"maxiter": maxiter},
        callback=_callback,
    )
    final_x = np.asarray(opt_result.x, dtype=np.float64)
    final_fun, final_grad, final_aux = _value_and_grad(final_x)
    return LaplaceModeOptimizationResult(
        z_mode=jnp.asarray(opt_result.x, dtype=z_init.dtype),
        objective_at_mode=float(final_fun),
        n_iters=int(opt_result.nit),
        n_function_evals=int(opt_result.nfev),
        status=int(opt_result.status),
        success=bool(opt_result.success),
        optimizer="L-BFGS-B",
        init_log_posterior_best=float(init_log_posterior_best),
        optimizer_hess_inv=getattr(opt_result, "hess_inv", None),
        final_grad_norm=float(np.linalg.norm(final_grad)),
        final_eval_diagnostics=final_aux,
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


def _sample_laplace_parameter_posterior_from_optimizer_hess_inv(
    rng_key: jnp.ndarray,
    z_mode: jnp.ndarray,
    optimizer_hess_inv,
    *,
    num_samples: int,
    hessian_jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample using the inverse-Hessian approximation returned by L-BFGS-B."""
    if num_samples < 1:
        raise ValueError("laplace_em requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    if dim == 0:
        return (
            jnp.zeros((num_samples, 0), dtype=z_mode.dtype),
            jnp.zeros((0, 0), dtype=z_mode.dtype),
            jnp.zeros((0,), dtype=z_mode.dtype),
        )
    if optimizer_hess_inv is None or not hasattr(optimizer_hess_inv, "todense"):
        raise RuntimeError("L-BFGS-B inverse-Hessian approximation is unavailable.")

    with jax.named_scope("laplace_em/optimizer_hess_inv"):
        covariance = jnp.asarray(
            np.asarray(optimizer_hess_inv.todense(), dtype=np.float64),
            dtype=z_mode.dtype,
        )
        covariance = symmetrize_with_jitter(covariance, jitter=hessian_jitter)
        chol_cov = jnp.linalg.cholesky(covariance)

    with jax.named_scope("laplace_em/parameter_sampling"):
        eps = random.normal(rng_key, (num_samples, dim), dtype=z_mode.dtype)
        unc_samples = z_mode[None, :] + eps @ chol_cov.T

    return unc_samples, covariance, jnp.zeros((0,), dtype=z_mode.dtype)


def _mode_only_parameter_posterior(
    z_mode: jnp.ndarray,
    *,
    num_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a degenerate posterior concentrated at the parameter mode."""
    if num_samples < 1:
        raise ValueError("laplace_em requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    unc_samples = jnp.broadcast_to(z_mode, (num_samples, dim))
    covariance = jnp.zeros((dim, dim), dtype=z_mode.dtype)
    hessian_eigvals = jnp.zeros((0,), dtype=z_mode.dtype)
    return unc_samples, covariance, hessian_eigvals


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
    compute_parameter_hessian: bool = True,
    parameter_covariance_method: Literal[
        "exact_hessian", "optimizer_hess_inv"
    ] = "optimizer_hess_inv",
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
    if parameter_covariance_method not in {"exact_hessian", "optimizer_hess_inv"}:
        raise ValueError(
            "parameter_covariance_method must be 'exact_hessian' or 'optimizer_hess_inv'."
        )

    rng_key = random.PRNGKey(seed)
    rng_key, trace_key, init_key, sample_key = random.split(rng_key, 4)

    backend_label = "kalman" if model.likelihood == "kalman" else "laplace_ieks"
    logger.info(
        "Laplace-EM config: backend=%s maxiter=%s tol=%s n_ieks_iters=%s "
        "n_init_samples=%s num_samples=%s compute_parameter_hessian=%s "
        "parameter_covariance_method=%s",
        backend_label,
        maxiter,
        tol,
        n_ieks_iters,
        n_init_samples,
        num_samples,
        compute_parameter_hessian,
        parameter_covariance_method,
    )

    phase_started_at = time.monotonic()
    logger.info("Laplace-EM phase start: phase=build_likelihood_backend")
    with jax.profiler.TraceAnnotation("laplace_em/build_likelihood_backend"):
        if model.likelihood == "kalman":
            backend = model.make_likelihood_backend()
        else:
            backend = model.make_laplace_backend(n_ieks_iters)
    logger.info(
        "Laplace-EM phase complete: phase=build_likelihood_backend elapsed=%.1fs backend=%s",
        _elapsed_seconds(phase_started_at),
        backend_label,
    )

    phase_started_at = time.monotonic()
    logger.info("Laplace-EM phase start: phase=build_bundle")
    with jax.profiler.TraceAnnotation("laplace_em/build_bundle"):
        bundle = _build_laplace_em_bundle(
            model,
            observations,
            times,
            trace_key,
            backend,
            reparam,
        )
    logger.info(
        "Laplace-EM phase complete: phase=build_bundle elapsed=%.1fs",
        _elapsed_seconds(phase_started_at),
    )

    dim = bundle["dim"]
    flat_example = bundle["flat_example"]
    site_info = bundle["site_info"]
    unravel_fn = bundle["unravel_fn"]
    log_posterior_fn = bundle["log_posterior_fn"]
    neg_log_posterior_fn = bundle["neg_log_posterior_fn"]
    neg_log_posterior_with_aux_fn = bundle["neg_log_posterior_with_aux_fn"]
    batch_log_posterior_jit = bundle["batch_log_posterior_jit"]
    logger.info("Laplace-EM bundle ready: parameter_dim=%d public_sites=%d", dim, len(site_info))

    logger.info(
        "Laplace-EM outer optimizer: method=%s support_aware=%s",
        "L-BFGS-B",
        _requires_support_aware_outer_optimizer(model),
    )
    phase_started_at = time.monotonic()
    logger.info("Laplace-EM phase start: phase=parameter_optimize")
    with jax.profiler.TraceAnnotation("laplace_em/parameter_optimize"):
        mode_result = _optimize_laplace_parameter_mode(
            model,
            init_key=init_key,
            dim=dim,
            flat_example=flat_example,
            site_info=site_info,
            log_posterior_fn=log_posterior_fn,
            neg_log_posterior_with_aux_fn=neg_log_posterior_with_aux_fn,
            batch_log_posterior_jit=batch_log_posterior_jit,
            observations=observations,
            n_init_samples=n_init_samples,
            maxiter=maxiter,
            tol=tol,
        )
    logger.info(
        "Laplace-EM phase complete: phase=parameter_optimize elapsed=%.1fs",
        _elapsed_seconds(phase_started_at),
    )

    z_mode = mode_result.z_mode
    mode_objective = mode_result.objective_at_mode
    nit = mode_result.n_iters
    nfev = mode_result.n_function_evals
    status = mode_result.status
    success = mode_result.success
    assert mode_result.final_eval_diagnostics is not None
    mode_eval = mode_result.final_eval_diagnostics
    mode_log_posterior = mode_eval["log_posterior"]
    mode_log_likelihood = mode_eval["log_likelihood"]
    mode_log_prior = mode_eval["log_prior"]
    mode_inner = mode_eval["inner"]
    logger.info(
        "Laplace-EM mode found: optimizer=%s success=%s nit=%s nfev=%s objective=%.6f",
        mode_result.optimizer,
        success,
        nit,
        nfev,
        mode_objective,
    )
    _log_outer_eval(
        label="mode",
        elapsed_seconds=0.0,
        eval_count=nfev,
        objective=mode_objective,
        best_objective=mode_objective,
        delta_objective=None,
        grad_norm=mode_result.final_grad_norm or 0.0,
        step_norm=None,
        outer_diag=mode_eval,
    )
    if not np.isfinite(mode_log_posterior):
        raise RuntimeError("Laplace-EM failed to find a finite parameter mode.")

    parameter_hessian_min_eig = None
    parameter_hessian_max_eig = None
    if compute_parameter_hessian:
        logger.info(
            "Laplace-EM parameter curvature: dim=%s method=%s sampling local Gaussian posterior",
            dim,
            parameter_covariance_method,
        )
        phase_started_at = time.monotonic()
        logger.info("Laplace-EM phase start: phase=parameter_curvature")
        with jax.profiler.TraceAnnotation("laplace_em/sample_parameter_posterior"):
            if parameter_covariance_method == "exact_hessian":
                unc_samples, covariance, hessian_eigvals = _sample_laplace_parameter_posterior(
                    sample_key,
                    z_mode,
                    neg_log_posterior_fn,
                    num_samples=num_samples,
                    hessian_jitter=hessian_jitter,
                )
            else:
                unc_samples, covariance, hessian_eigvals = (
                    _sample_laplace_parameter_posterior_from_optimizer_hess_inv(
                        sample_key,
                        z_mode,
                        mode_result.optimizer_hess_inv,
                        num_samples=num_samples,
                        hessian_jitter=hessian_jitter,
                    )
                )
        logger.info(
            "Laplace-EM phase complete: phase=parameter_curvature elapsed=%.1fs",
            _elapsed_seconds(phase_started_at),
        )
        parameter_posterior_strategy = "laplace_gaussian"
    else:
        logger.info("Laplace-EM parameter Hessian skipped; using deterministic mode samples")
        unc_samples, covariance, hessian_eigvals = _mode_only_parameter_posterior(
            z_mode,
            num_samples=num_samples,
        )
        parameter_posterior_strategy = "mode_only"

    if site_info:
        phase_started_at = time.monotonic()
        logger.info("Laplace-EM phase start: phase=extract_samples")
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
        logger.info(
            "Laplace-EM phase complete: phase=extract_samples elapsed=%.1fs draws=%d",
            _elapsed_seconds(phase_started_at),
            int(unc_samples.shape[0]),
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
        parameter_hessian_min_eig = float(jax.device_get(jnp.min(hessian_eigvals)))
        parameter_hessian_max_eig = float(jax.device_get(jnp.max(hessian_eigvals)))
        if parameter_hessian_min_eig > 0.0:
            hessian_condition_number = parameter_hessian_max_eig / parameter_hessian_min_eig

    if compute_parameter_hessian:
        if parameter_covariance_method == "exact_hessian":
            logger.info(
                "Laplace-EM parameter curvature exact_hessian: min_eig=%s max_eig=%s condition=%s",
                _format_float(parameter_hessian_min_eig),
                _format_float(parameter_hessian_max_eig),
                _format_float(hessian_condition_number),
            )
        else:
            covariance_diag = np.asarray(jax.device_get(jnp.diag(covariance)), dtype=np.float64)
            logger.info(
                "Laplace-EM parameter curvature optimizer_hess_inv: covariance_diag_min=%s covariance_diag_max=%s",
                _format_float(float(np.min(covariance_diag))),
                _format_float(float(np.max(covariance_diag))),
            )

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
        "mode_grad_norm": mode_result.final_grad_norm,
        "mode_inner_solver": _solver_label(mode_inner["solver_kind"]),
        "mode_inner_iterations": mode_inner["n_iterations"],
        "mode_inner_accepted_steps": mode_inner["n_accepted_steps"],
        "mode_inner_rel_change": mode_inner["final_rel_change"],
        "mode_inner_damping": mode_inner["final_damping"],
        "mode_inner_step_alpha": mode_inner["final_step_alpha"],
        "mode_inner_step_norm": mode_inner["final_step_norm"],
        "mode_inner_log_joint_gain": _inner_log_joint_gain(mode_inner),
        "mode_inner_laplace_logdet": mode_inner["laplace_logdet"],
        "mode_inner_min_chol_diag": mode_inner["min_chol_diag"],
        "init_log_posterior_best": mode_result.init_log_posterior_best,
        "n_init_samples": n_init_samples,
        "n_ieks_iters": n_ieks_iters,
        "compute_parameter_hessian": compute_parameter_hessian,
        "parameter_posterior_strategy": parameter_posterior_strategy,
        "parameter_covariance_method": parameter_covariance_method
        if compute_parameter_hessian
        else "mode_only",
        "hessian_jitter": hessian_jitter,
        "hessian_condition_number": hessian_condition_number,
        "parameter_hessian_min_eig": parameter_hessian_min_eig,
        "parameter_hessian_max_eig": parameter_hessian_max_eig,
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
        method="map",
        diagnostics=diagnostics,
    )
