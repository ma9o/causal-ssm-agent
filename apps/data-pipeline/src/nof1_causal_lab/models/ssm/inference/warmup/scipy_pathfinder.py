"""Scipy-driven Pathfinder: multi-start L-BFGS-B + best-ELBO Gaussian selection.

BlackJAX's public Pathfinder API is single-start and builds a JAX/JAXopt
program for the whole procedure: L-BFGS, line search, path history recovery,
inverse-Hessian factors, Gaussian sampling, and ELBO scoring. With our
IEKS/Laplace SSM target, that nests the expensive log-posterior inside a much
larger compiled program. Looping over starts through that API also makes each
start a separate Pathfinder call rather than one shared multistart program. At
T=1000 on A100 this blows up compile time to ~1700 s and HBM to ~25 GiB
(measured).

The scipy pattern jit-compiles the log-posterior value and the log-posterior
value+gradient kernels once. L-BFGS iteration, L-BFGS-history Hessian
reconstruction, ELBO computation, and multistart scheduling are plain
numpy/scipy/Python work outside XLA, with independent starts parallelized by a
thread pool. For this low-dimensional parameter target, that CPU-side
bookkeeping is cheap while compiled SSM kernels are reused across starts. ELBO
candidate scoring uses a batched value-only kernel so it does not pay for
gradients that Pathfinder discards, and it does not round-trip through Python
once per ELBO sample.

Algorithmically equivalent to Pathfinder (Zhang et al. 2022): at each
L-BFGS iterate form the local Gaussian approximation using the L-BFGS
quasi-Newton inverse-Hessian (via the two-loop recursion), compute ELBO by
Monte Carlo sampling, return the approximation with highest ELBO.
"""

from __future__ import annotations

import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import scipy.optimize

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScipyPathfinderResult:
    mean: np.ndarray  # (p,) — best-ELBO iterate
    chol: np.ndarray  # (p, p) lower-triangular — Cholesky of L-BFGS H^{-1} at that iterate
    best_elbo: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _ObjectiveEvaluation:
    x: np.ndarray
    neg_log_post: float
    objective_grad: np.ndarray
    log_post: float


@dataclass(frozen=True)
class _ScipyPathfinderStartResult:
    start_idx: int
    mean: np.ndarray | None
    chol: np.ndarray | None
    best_elbo: float
    diagnostics: dict[str, Any]


@functools.partial(jax.jit, static_argnames=("runtime_log_posterior_fn",))
def _scipy_pathfinder_value_batch_runtime(
    z_batch: jnp.ndarray,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_log_posterior_fn,
) -> jnp.ndarray:
    return jax.vmap(
        lambda z_arg: runtime_log_posterior_fn(
            z_arg,
            observations,
            times,
            latent_mode_init=None,
        )
    )(z_batch)


@functools.partial(jax.jit, static_argnames=("runtime_log_posterior_fn",))
def _scipy_pathfinder_value_and_grad_runtime(
    z: jnp.ndarray,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_log_posterior_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jax.value_and_grad(
        lambda z_arg: runtime_log_posterior_fn(
            z_arg,
            observations,
            times,
            latent_mode_init=None,
        )
    )(z)


def _flat_indices_for_sites(
    flat_example: jnp.ndarray,
    unravel_fn,
    site_names: tuple[str, ...],
) -> list[int]:
    """Return flat indices in the Pathfinder layout that belong to site names."""
    dim = int(flat_example.shape[0])
    site_for_idx: list[str | None] = [None] * dim
    for idx in range(dim):
        onehot = np.zeros(dim, dtype=np.float64)
        onehot[idx] = 1.0
        unraveled = unravel_fn(jnp.asarray(onehot, dtype=flat_example.dtype))
        for name, value in unraveled.items():
            if np.any(np.abs(np.asarray(value)) > 1e-10):
                site_for_idx[idx] = name
                break
    return [idx for idx, name in enumerate(site_for_idx) if name is not None and name in site_names]


def _lbfgs_two_loop_inverse_hessian_apply(
    history: list[tuple[np.ndarray, np.ndarray]],
    v: np.ndarray,
    gamma0: float,
) -> np.ndarray:
    """Apply L-BFGS quasi-Newton H_k^{-1} @ v via two-loop recursion.

    ``history`` is an ordered list of ``(s_i, y_i)`` curvature pairs (oldest
    first). ``gamma0`` scales the identity term in the initial Hessian
    approximation, typically ``<s_last, y_last> / <y_last, y_last>``.
    Nocedal & Wright Algorithm 7.4.
    """
    m = len(history)
    alpha = np.empty(m, dtype=np.float64)
    q = v.astype(np.float64, copy=True)
    for i in range(m - 1, -1, -1):
        s_i, y_i = history[i]
        rho_i = 1.0 / float(y_i @ s_i)
        alpha[i] = rho_i * float(s_i @ q)
        q = q - alpha[i] * y_i
    r = gamma0 * q
    for i in range(m):
        s_i, y_i = history[i]
        rho_i = 1.0 / float(y_i @ s_i)
        beta = rho_i * float(y_i @ r)
        r = r + (alpha[i] - beta) * s_i
    return r


def _form_lbfgs_inverse_hessian_matrix(
    history: list[tuple[np.ndarray, np.ndarray]],
    dim: int,
) -> np.ndarray:
    """Materialise the full ``(p, p)`` L-BFGS inverse-Hessian approximation.

    Computes each column of ``H^{-1}`` by applying the two-loop recursion to
    a unit vector. Cost: ``O(m * p^2)`` where ``m`` is the curvature-history
    length. For ``p = 80`` this is a few ms in numpy.
    """
    s_last, y_last = history[-1]
    yy = float(y_last @ y_last)
    gamma0 = float(s_last @ y_last) / yy if yy > 0.0 else 1.0
    H_inv = np.empty((dim, dim), dtype=np.float64)
    e = np.zeros(dim, dtype=np.float64)
    for j in range(dim):
        e[j] = 1.0
        H_inv[:, j] = _lbfgs_two_loop_inverse_hessian_apply(history, e, gamma0)
        e[j] = 0.0
    return H_inv


def _build_curvature_history(
    trajectory: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    memory: int,
    secant_tol: float = 1e-10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the most recent ``memory`` L-BFGS (s, y) pairs ending at iterate ``k``.

    Skips pairs that violate the curvature condition ``y^T s > tol`` — standard
    L-BFGS practice to keep ``H^{-1}`` PSD.
    """
    history: list[tuple[np.ndarray, np.ndarray]] = []
    lo = max(0, k - memory)
    for i in range(lo, k):
        s_i = trajectory[i + 1][0] - trajectory[i][0]
        y_i = trajectory[i + 1][1] - trajectory[i][1]
        if float(y_i @ s_i) > secant_tol:
            history.append((s_i, y_i))
    return history


def _resolve_elbo_screen_samples(
    *,
    elbo_samples: int,
    elbo_screen_samples: int | None,
) -> int:
    resolved = (
        min(8, int(elbo_samples)) if elbo_screen_samples is None else int(elbo_screen_samples)
    )
    if resolved < 1:
        raise ValueError("elbo_screen_samples must be >= 1.")
    if int(elbo_samples) < resolved:
        raise ValueError("elbo_samples must be >= elbo_screen_samples.")
    return int(resolved)


def _estimate_elbo_candidate_batch(
    log_post_batch_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    means: list[np.ndarray],
    chols: list[np.ndarray],
    *,
    elbo_samples: int,
    dim: int,
    candidate_batch_size: int,
) -> tuple[np.ndarray, int]:
    candidate_elbos = np.full(len(means), -np.inf, dtype=np.float64)
    n_batch_evaluations = 0

    for batch_start in range(0, len(means), candidate_batch_size):
        batch_means = means[batch_start : batch_start + candidate_batch_size]
        batch_chols = chols[batch_start : batch_start + candidate_batch_size]
        actual_batch_size = len(batch_means)
        if actual_batch_size == 0:
            continue
        while len(batch_means) < candidate_batch_size:
            batch_means.append(batch_means[-1])
            batch_chols.append(batch_chols[-1])

        mean_stack = np.stack(batch_means, axis=0)
        chol_stack = np.stack(batch_chols, axis=0)
        zeta = rng.standard_normal((candidate_batch_size, elbo_samples, dim))
        samples = mean_stack[:, None, :] + np.einsum("bni,bji->bnj", zeta, chol_stack)
        flat_samples = samples.reshape(candidate_batch_size * elbo_samples, dim)
        flat_log_post = np.asarray(log_post_batch_fn(flat_samples), dtype=np.float64)
        log_post_samples = flat_log_post.reshape(candidate_batch_size, elbo_samples)

        log_det_terms = np.log(np.abs(np.diagonal(chol_stack, axis1=1, axis2=2))).sum(axis=1)
        log_q = (
            -0.5 * np.sum(zeta**2, axis=2)
            - log_det_terms[:, None]
            - 0.5 * dim * np.log(2.0 * np.pi)
        )
        batch_elbos = np.mean(log_post_samples - log_q, axis=1)
        n_batch_evaluations += 1

        for offset in range(actual_batch_size):
            candidate_elbos[batch_start + offset] = float(batch_elbos[offset])

    return candidate_elbos, n_batch_evaluations


def _score_elbo_candidates(
    log_post_batch_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    means: list[np.ndarray],
    chols: list[np.ndarray],
    *,
    elbo_samples: int,
    elbo_screen_samples: int,
    elbo_refine_candidates: int,
    dim: int,
    candidate_batch_size: int,
) -> tuple[float, np.ndarray | None, np.ndarray | None, dict[str, int]]:
    if not means:
        return (
            -np.inf,
            None,
            None,
            {
                "n_elbo_batch_evaluations": 0,
                "n_elbo_screen_candidates": 0,
                "n_elbo_refine_candidates": 0,
                "best_elbo_candidate_index": -1,
            },
        )

    if len(means) <= elbo_refine_candidates or elbo_screen_samples == elbo_samples:
        elbos, n_batch_evaluations = _estimate_elbo_candidate_batch(
            log_post_batch_fn,
            rng,
            means,
            chols,
            elbo_samples=elbo_samples,
            dim=dim,
            candidate_batch_size=candidate_batch_size,
        )
        best_idx = int(np.argmax(elbos))
        if not np.isfinite(elbos[best_idx]):
            best_idx = -1
        return (
            float(elbos[best_idx]) if best_idx >= 0 else -np.inf,
            means[best_idx].copy() if best_idx >= 0 else None,
            chols[best_idx].copy() if best_idx >= 0 else None,
            {
                "n_elbo_batch_evaluations": int(n_batch_evaluations),
                "n_elbo_screen_candidates": len(means),
                "n_elbo_refine_candidates": len(means),
                "best_elbo_candidate_index": int(best_idx),
            },
        )

    screen_elbos, screen_batch_evaluations = _estimate_elbo_candidate_batch(
        log_post_batch_fn,
        rng,
        means,
        chols,
        elbo_samples=elbo_screen_samples,
        dim=dim,
        candidate_batch_size=candidate_batch_size,
    )
    finite_indices = np.flatnonzero(np.isfinite(screen_elbos))
    if finite_indices.size == 0:
        return (
            -np.inf,
            None,
            None,
            {
                "n_elbo_batch_evaluations": int(screen_batch_evaluations),
                "n_elbo_screen_candidates": len(means),
                "n_elbo_refine_candidates": 0,
                "best_elbo_candidate_index": -1,
            },
        )

    ordered_finite = finite_indices[np.argsort(screen_elbos[finite_indices])[::-1]]
    selected = ordered_finite[: int(elbo_refine_candidates)]
    if 0 not in selected and np.isfinite(screen_elbos[0]):
        selected = np.concatenate([selected, np.asarray([0], dtype=selected.dtype)])
    selected = np.unique(selected)
    selected.sort()
    refine_means = [means[int(idx)] for idx in selected]
    refine_chols = [chols[int(idx)] for idx in selected]
    refine_elbos, refine_batch_evaluations = _estimate_elbo_candidate_batch(
        log_post_batch_fn,
        rng,
        refine_means,
        refine_chols,
        elbo_samples=elbo_samples,
        dim=dim,
        candidate_batch_size=candidate_batch_size,
    )
    refine_best_offset = int(np.argmax(refine_elbos))
    if not np.isfinite(refine_elbos[refine_best_offset]):
        best_idx = -1
    else:
        best_idx = int(selected[refine_best_offset])

    return (
        float(refine_elbos[refine_best_offset]) if best_idx >= 0 else -np.inf,
        means[best_idx].copy() if best_idx >= 0 else None,
        chols[best_idx].copy() if best_idx >= 0 else None,
        {
            "n_elbo_batch_evaluations": int(screen_batch_evaluations + refine_batch_evaluations),
            "n_elbo_screen_candidates": len(means),
            "n_elbo_refine_candidates": len(selected),
            "best_elbo_candidate_index": int(best_idx),
        },
    )


def _run_pathfinder_start(
    log_post_batch_fn: Callable[[np.ndarray], np.ndarray],
    log_post_and_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    *,
    start_idx: int,
    n_starts: int,
    dim: int,
    maxiter: int,
    elbo_samples: int,
    elbo_screen_samples: int,
    elbo_refine_candidates: int,
    elbo_candidate_batch_size: int,
    lbfgs_memory: int,
    jitter: float,
    seed: int,
) -> _ScipyPathfinderStartResult:
    start_t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    logger.info(
        "scipy_pathfinder start %d/%d: starting L-BFGS-B from |x0|=%.2g",
        start_idx + 1,
        n_starts,
        float(np.linalg.norm(np.asarray(x0, dtype=np.float64))),
    )
    cached_eval: _ObjectiveEvaluation | None = None
    trajectory: list[tuple[np.ndarray, np.ndarray, float]] = []

    def _evaluate(x: np.ndarray) -> _ObjectiveEvaluation:
        nonlocal cached_eval
        x_np = np.asarray(x, dtype=np.float64)
        if cached_eval is not None and np.array_equal(x_np, cached_eval.x):
            return cached_eval
        log_post, grad_log_post = log_post_and_grad_fn(x_np)
        objective_grad = -np.asarray(grad_log_post, dtype=np.float64)
        cached_eval = _ObjectiveEvaluation(
            x=x_np.copy(),
            neg_log_post=float(-log_post),
            objective_grad=objective_grad.copy(),
            log_post=float(log_post),
        )
        return cached_eval

    def _append_accepted_iterate(
        x: np.ndarray,
        _trajectory: list[tuple[np.ndarray, np.ndarray, float]] = trajectory,
    ) -> None:
        evaluation = _evaluate(x)
        if _trajectory and np.array_equal(evaluation.x, _trajectory[-1][0]):
            return
        _trajectory.append(
            (
                evaluation.x.copy(),
                evaluation.objective_grad.copy(),
                evaluation.log_post,
            )
        )

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        evaluation = _evaluate(x)
        return evaluation.neg_log_post, evaluation.objective_grad

    _append_accepted_iterate(np.asarray(x0, dtype=np.float64))

    def callback(xk: np.ndarray) -> None:
        _append_accepted_iterate(xk)

    opt_result = scipy.optimize.minimize(
        objective,
        x0=np.asarray(x0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": int(maxiter)},
    )
    _append_accepted_iterate(np.asarray(opt_result.x, dtype=np.float64))
    logger.info(
        "scipy_pathfinder start %d/%d: L-BFGS-B done in %.1fs (nit=%d, nfev=%s, "
        "success=%s, status=%s, log_post=%.3f)",
        start_idx + 1,
        n_starts,
        time.monotonic() - start_t0,
        int(getattr(opt_result, "nit", 0)),
        int(getattr(opt_result, "nfev", -1)),
        bool(getattr(opt_result, "success", False)),
        int(getattr(opt_result, "status", -1)),
        float(-opt_result.fun),
    )

    # scipy's L-BFGS-B returns the final inverse-Hessian as a
    # ``LbfgsInvHessProduct`` — an always-PSD low-rank operator. Use it as
    # a guaranteed-valid candidate Gaussian at the final iterate; our own
    # per-iterate two-loop recursion then adds best-ELBO selection on top.
    candidate_means: list[np.ndarray] = []
    candidate_chols: list[np.ndarray] = []
    try:
        hess_inv_op = getattr(opt_result, "hess_inv", None)
        if hess_inv_op is not None:
            hess_inv_final = np.asarray(hess_inv_op.todense(), dtype=np.float64)
            hess_inv_final = 0.5 * (hess_inv_final + hess_inv_final.T)
            hess_inv_final = hess_inv_final + jitter * np.eye(dim, dtype=np.float64)
            l_final = np.linalg.cholesky(hess_inv_final)
            x_final = np.asarray(opt_result.x, dtype=np.float64).copy()
            candidate_means.append(x_final)
            candidate_chols.append(l_final)
    except (np.linalg.LinAlgError, ValueError, AttributeError):
        pass

    # Iterate along the accepted L-BFGS iterates, forming H^{-1} at each
    # point from the most recent curvature history and scoring the
    # resulting Gaussian by ELBO. trajectory[0] is the init; skip it since
    # we have no history yet.
    valid_iterate_count = 0
    traj_xg = [(pt[0], pt[1]) for pt in trajectory]

    for k in range(1, len(trajectory)):
        history = _build_curvature_history(traj_xg, k, lbfgs_memory)
        if not history:
            continue
        hess_inv = _form_lbfgs_inverse_hessian_matrix(history, dim)
        hess_inv = 0.5 * (hess_inv + hess_inv.T)
        hess_inv = hess_inv + jitter * np.eye(dim, dtype=np.float64)
        try:
            l_k = np.linalg.cholesky(hess_inv)
        except np.linalg.LinAlgError:
            continue

        x_k = trajectory[k][0]
        candidate_means.append(x_k.copy())
        candidate_chols.append(l_k.copy())
        valid_iterate_count += 1

    start_best_elbo, start_best_mean, start_best_chol, elbo_score_diagnostics = (
        _score_elbo_candidates(
            log_post_batch_fn,
            rng,
            candidate_means,
            candidate_chols,
            elbo_samples=elbo_samples,
            elbo_screen_samples=elbo_screen_samples,
            elbo_refine_candidates=elbo_refine_candidates,
            dim=dim,
            candidate_batch_size=elbo_candidate_batch_size,
        )
    )

    diagnostics = {
        "start_idx": int(start_idx),
        "n_trajectory_points": len(trajectory),
        "n_valid_iterates": int(valid_iterate_count),
        "n_elbo_candidates": len(candidate_means),
        **elbo_score_diagnostics,
        "n_lbfgs_iterations": int(opt_result.nit),
        "final_log_posterior": float(-opt_result.fun),
        "best_elbo_this_start": (float(start_best_elbo) if start_best_elbo > -np.inf else None),
        "scipy_success": bool(opt_result.success),
        "scipy_status": int(opt_result.status),
    }
    logger.info(
        "scipy_pathfinder start %d/%d: best ELBO this start = %s "
        "(n_valid_iterates=%d, total %.1fs)",
        start_idx + 1,
        n_starts,
        f"{start_best_elbo:.3f}" if np.isfinite(start_best_elbo) else "n/a",
        int(valid_iterate_count),
        time.monotonic() - start_t0,
    )
    return _ScipyPathfinderStartResult(
        start_idx=start_idx,
        mean=start_best_mean,
        chol=start_best_chol,
        best_elbo=float(start_best_elbo),
        diagnostics=diagnostics,
    )


def scipy_pathfinder(
    log_post_batch_fn: Callable[[np.ndarray], np.ndarray],
    log_post_and_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0_starts: list[np.ndarray],
    *,
    maxiter: int = 20,
    elbo_samples: int = 10,
    elbo_screen_samples: int | None = None,
    elbo_refine_candidates: int = 16,
    elbo_candidate_batch_size: int = 8,
    lbfgs_memory: int = 10,
    jitter: float = 1e-6,
    seed: int = 0,
    parallel_workers: int | None = None,
) -> ScipyPathfinderResult:
    """Multi-start scipy-driven Pathfinder.

    Parameters
    ----------
    log_post_batch_fn : callable
        ``x_batch -> log_posterior_vector``. Used for ELBO candidate scoring.
    log_post_and_grad_fn : callable
        ``x -> (log_posterior_scalar, gradient_vector)``. Accepts and returns
        numpy arrays. Used only for L-BFGS objective evaluations. Caller is
        responsible for any jit compilation inside.
    x0_starts : list[np.ndarray]
        One starting point per L-BFGS run. Length = number of multi-starts.
    maxiter : int
        L-BFGS outer-iteration budget per start.
    elbo_samples : int
        Monte Carlo samples used to refine ELBO at screened candidate iterates.
    elbo_screen_samples : int | None
        Monte Carlo samples used for the cheap first-pass ELBO screen. ``None``
        uses ``min(8, elbo_samples)``.
    elbo_refine_candidates : int
        Number of best screened candidate Gaussians per start to rescore with
        ``elbo_samples``.
    elbo_candidate_batch_size : int
        Number of candidate Gaussians scored in one batched log-posterior call.
    lbfgs_memory : int
        L-BFGS curvature-history depth (standard ``m``).
    jitter : float
        Added to ``H^{-1}`` diagonal before Cholesky for numerical stability.
    seed : int
        RNG seed for ELBO sample draws.
    parallel_workers : int | None
        Number of thread workers for independent L-BFGS starts. ``None`` uses
        one worker per start.

    Returns
    -------
    ScipyPathfinderResult with the best-ELBO Gaussian across all starts.
    """
    if not x0_starts:
        raise ValueError("scipy_pathfinder requires at least one start.")
    if elbo_candidate_batch_size < 1:
        raise ValueError("elbo_candidate_batch_size must be >= 1.")
    resolved_elbo_screen_samples = _resolve_elbo_screen_samples(
        elbo_samples=int(elbo_samples),
        elbo_screen_samples=elbo_screen_samples,
    )
    if elbo_refine_candidates < 1:
        raise ValueError("elbo_refine_candidates must be >= 1.")
    pf_t0 = time.monotonic()

    dim = int(x0_starts[0].shape[0])
    worker_count = len(x0_starts) if parallel_workers is None else int(parallel_workers)
    if worker_count < 1:
        raise ValueError("parallel_workers must be >= 1.")
    worker_count = min(worker_count, len(x0_starts))
    logger.info(
        "scipy_pathfinder: K=%d starts, dim=%d, maxiter=%d, lbfgs_memory=%d, "
        "elbo_samples=%d, elbo_screen_samples=%d, elbo_refine_candidates=%d, "
        "elbo_candidate_batch_size=%d, parallel_workers=%d",
        len(x0_starts),
        dim,
        int(maxiter),
        int(lbfgs_memory),
        int(elbo_samples),
        int(resolved_elbo_screen_samples),
        int(elbo_refine_candidates),
        int(elbo_candidate_batch_size),
        int(worker_count),
    )

    seed_sequence = np.random.SeedSequence(seed)
    start_seeds = [
        int(seed_state.generate_state(1, dtype=np.uint32)[0])
        for seed_state in seed_sequence.spawn(len(x0_starts))
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _run_pathfinder_start,
                log_post_batch_fn,
                log_post_and_grad_fn,
                np.asarray(x0, dtype=np.float64),
                start_idx=start_idx,
                n_starts=len(x0_starts),
                dim=dim,
                maxiter=int(maxiter),
                elbo_samples=int(elbo_samples),
                elbo_screen_samples=int(resolved_elbo_screen_samples),
                elbo_refine_candidates=int(elbo_refine_candidates),
                elbo_candidate_batch_size=int(elbo_candidate_batch_size),
                lbfgs_memory=int(lbfgs_memory),
                jitter=float(jitter),
                seed=start_seeds[start_idx],
            )
            for start_idx, x0 in enumerate(x0_starts)
        ]
        start_results = [future.result() for future in futures]

    start_results.sort(key=lambda result: result.start_idx)
    per_start_diagnostics = [result.diagnostics for result in start_results]
    best_result = max(start_results, key=lambda result: result.best_elbo)
    best_mean = best_result.mean
    best_chol = best_result.chol
    best_elbo = best_result.best_elbo

    if best_mean is None or best_chol is None:
        raise RuntimeError(
            "scipy_pathfinder found no valid Gaussian approximation across all starts; "
            "all accepted L-BFGS iterates and final inverse-Hessian candidates failed."
        )

    logger.info(
        "scipy_pathfinder: complete in %.1fs (best ELBO=%.3f across %d starts)",
        time.monotonic() - pf_t0,
        float(best_elbo),
        len(x0_starts),
    )

    finite_starts = sum(1 for d in per_start_diagnostics if d["best_elbo_this_start"] is not None)
    elbo_values = [
        d["best_elbo_this_start"]
        for d in per_start_diagnostics
        if d["best_elbo_this_start"] is not None
    ]
    return ScipyPathfinderResult(
        mean=np.asarray(best_mean, dtype=np.float64),
        chol=np.asarray(best_chol, dtype=np.float64),
        best_elbo=float(best_elbo),
        diagnostics={
            "n_starts": len(x0_starts),
            "n_starts_finite": int(finite_starts),
            "per_start": per_start_diagnostics,
            "elbo_samples": int(elbo_samples),
            "elbo_screen_samples": int(resolved_elbo_screen_samples),
            "elbo_refine_candidates": int(elbo_refine_candidates),
            "elbo_candidate_batch_size": int(elbo_candidate_batch_size),
            "lbfgs_memory": int(lbfgs_memory),
            "maxiter": int(maxiter),
            "parallel_workers": int(worker_count),
            "elbo_min": float(min(elbo_values)) if elbo_values else float("nan"),
            "elbo_max": float(max(elbo_values)) if elbo_values else float("nan"),
            "elbo_spread": (float(max(elbo_values) - min(elbo_values)) if elbo_values else 0.0),
        },
    )


def run_scipy_pathfinder_approximation(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    reparam,
    n_ieks_iters: int,
    num_elbo_samples: int,
    maxiter: int,
    n_pathfinder_starts: int = 8,
    pathfinder_parallel_workers: int | None = None,
    elbo_screen_samples: int | None = None,
    elbo_refine_candidates: int = 16,
    elbo_candidate_batch_size: int = 8,
    init_scale: float = 0.1,
) -> tuple[ScipyPathfinderResult, dict[str, Any]]:
    """Run scipy Pathfinder on the IEKS-marginal log-posterior for parameters."""
    if n_pathfinder_starts < 1:
        raise ValueError("n_pathfinder_starts must be >= 1.")
    from nof1_causal_lab.models.ssm.inference.warmup.map import _build_map_laplace_bundle

    total_t0 = time.monotonic()
    setup_t0 = time.monotonic()
    backend = model.make_laplace_backend(n_ieks_iters)
    laplace_bundle = _build_map_laplace_bundle(
        model, observations, times, trace_key, backend, reparam
    )
    runtime_log_post_fn = laplace_bundle["log_posterior_fn"]
    flat_example = laplace_bundle["flat_example"]
    flat_dtype = flat_example.dtype
    setup_seconds = time.monotonic() - setup_t0

    def log_post_batch_np(x_batch: np.ndarray) -> np.ndarray:
        xj = jnp.asarray(x_batch, dtype=flat_dtype)
        fx = _scipy_pathfinder_value_batch_runtime(
            xj,
            observations,
            times,
            runtime_log_posterior_fn=runtime_log_post_fn,
        )
        return np.asarray(jax.device_get(fx), dtype=np.float64)

    def log_post_and_grad_np(x: np.ndarray) -> tuple[float, np.ndarray]:
        xj = jnp.asarray(x, dtype=flat_dtype)
        fx, gx = _scipy_pathfinder_value_and_grad_runtime(
            xj,
            observations,
            times,
            runtime_log_posterior_fn=runtime_log_post_fn,
        )
        return float(fx), np.asarray(jax.device_get(gx), dtype=np.float64)

    seed_int = int(jax.device_get(random.randint(pathfinder_key, (), 0, 2**31 - 1)))
    rng = np.random.default_rng(seed_int)
    base_np = np.asarray(jax.device_get(flat_example), dtype=np.float64)
    resolved_elbo_screen_samples = _resolve_elbo_screen_samples(
        elbo_samples=int(num_elbo_samples),
        elbo_screen_samples=elbo_screen_samples,
    )
    compile_batch_sizes = sorted(
        {
            int(elbo_candidate_batch_size) * int(resolved_elbo_screen_samples),
            int(elbo_candidate_batch_size) * int(num_elbo_samples),
        }
    )
    compile_t0 = time.monotonic()
    log_post_and_grad_np(base_np)
    for batch_size in compile_batch_sizes:
        log_post_batch_np(np.repeat(base_np[None, :], int(batch_size), axis=0))
    jax_compile_seconds = time.monotonic() - compile_t0
    logger.info(
        "scipy_pathfinder: JAX compile/warm-up done in %.1fs (grad_shape=%s, value_batch_sizes=%s)",
        jax_compile_seconds,
        tuple(base_np.shape),
        compile_batch_sizes,
    )
    x0_starts = [
        base_np + float(init_scale) * rng.standard_normal(base_np.shape)
        for _ in range(int(n_pathfinder_starts))
    ]

    pathfinder_runtime_t0 = time.monotonic()
    result = scipy_pathfinder(
        log_post_batch_np,
        log_post_and_grad_np,
        x0_starts,
        maxiter=int(maxiter),
        elbo_samples=int(num_elbo_samples),
        elbo_screen_samples=elbo_screen_samples,
        elbo_refine_candidates=int(elbo_refine_candidates),
        elbo_candidate_batch_size=int(elbo_candidate_batch_size),
        seed=seed_int,
        parallel_workers=pathfinder_parallel_workers,
    )
    pathfinder_runtime_seconds = time.monotonic() - pathfinder_runtime_t0
    pathfinder_elbos = [
        float(item["best_elbo_this_start"])
        for item in result.diagnostics["per_start"]
        if item["best_elbo_this_start"] is not None
    ]
    diagnostics = {
        "n_pathfinder_starts": int(n_pathfinder_starts),
        "n_pathfinder_starts_finite": int(result.diagnostics["n_starts_finite"]),
        "pathfinder_parallel_workers": int(result.diagnostics["parallel_workers"]),
        "pathfinder_setup_seconds": float(setup_seconds),
        "pathfinder_jax_compile_seconds": float(jax_compile_seconds),
        "pathfinder_jax_compile_batch_sizes": compile_batch_sizes,
        "pathfinder_runtime_seconds": float(pathfinder_runtime_seconds),
        "pathfinder_total_seconds": float(time.monotonic() - total_t0),
        "best_pathfinder_elbo": float(result.best_elbo),
        "pathfinder_elbo": float(result.best_elbo),
        "pathfinder_elbo_min": float(result.diagnostics["elbo_min"]),
        "pathfinder_elbo_max": float(result.diagnostics["elbo_max"]),
        "pathfinder_elbo_spread": float(result.diagnostics["elbo_spread"]),
        "pathfinder_elbos": pathfinder_elbos,
        "pathfinder_maxiter": int(result.diagnostics["maxiter"]),
        "pathfinder_lbfgs_memory": int(result.diagnostics["lbfgs_memory"]),
        "pathfinder_elbo_samples": int(result.diagnostics["elbo_samples"]),
        "pathfinder_elbo_screen_samples": int(result.diagnostics["elbo_screen_samples"]),
        "pathfinder_elbo_refine_candidates": int(result.diagnostics["elbo_refine_candidates"]),
        "pathfinder_elbo_candidate_batch_size": int(
            result.diagnostics["elbo_candidate_batch_size"]
        ),
        "pathfinder_per_start": result.diagnostics["per_start"],
    }
    return result, diagnostics


def sample_scipy_pathfinder_init_positions(
    pathfinder_state: ScipyPathfinderResult,
    pathfinder_diagnostics: dict[str, Any],
    *,
    sample_key: jnp.ndarray,
    num_chains: int,
    dtype,
    pathfinder_init_scale: float | None = None,
    init_bundle: dict[str, Any] | None = None,
    prior_released_sites: tuple[str, ...] = (),
    prior_release_scale: float = 0.05,
    release_jitter_key: jnp.ndarray | None = None,
    method_label: str = "sampler",
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Sample per-chain initial positions from a fitted scipy Pathfinder state."""
    mean_np = np.asarray(jax.device_get(pathfinder_state.mean), dtype=np.float64)
    chol_np = np.asarray(jax.device_get(pathfinder_state.chol), dtype=np.float64)
    p_dim = mean_np.shape[0]
    zeta = random.normal(sample_key, (num_chains, p_dim), dtype=dtype)
    zeta_np = np.asarray(jax.device_get(zeta), dtype=np.float64)
    if pathfinder_init_scale is None:
        positions_np = mean_np[None, :] + zeta_np @ chol_np.T
        sampling_mode = "pathfinder_gaussian"
    else:
        positions_np = mean_np[None, :] + float(pathfinder_init_scale) * zeta_np
        sampling_mode = "mode_plus_scaled_normal"
    positions = jnp.asarray(positions_np, dtype=dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError(
            f"Pathfinder returned non-finite chain-init positions for {method_label}."
        )

    prior_site_indices: list[int] = []
    if init_bundle is not None and prior_released_sites:
        prior_site_indices = _flat_indices_for_sites(
            init_bundle["flat_example"],
            init_bundle["unravel_fn"],
            prior_released_sites,
        )
        if prior_site_indices:
            flat_example = jnp.asarray(init_bundle["flat_example"], dtype=dtype)
            dim = int(flat_example.shape[0])
            mask = np.zeros(dim, dtype=bool)
            for idx in prior_site_indices:
                mask[idx] = True
            mask_j = jnp.asarray(mask)
            jitter_key = (
                release_jitter_key
                if release_jitter_key is not None
                else random.fold_in(sample_key, 0xC0FFEE)
            )
            noise = random.normal(jitter_key, (num_chains, dim), dtype=dtype)
            prior_values = flat_example[None, :] + float(prior_release_scale) * noise
            positions = jnp.where(mask_j[None, :], prior_values, positions)

    diagnostics = {
        "init_method": "pathfinder",
        "pathfinder_sampling_mode": sampling_mode,
        "pathfinder_init_scale": pathfinder_init_scale,
        **pathfinder_diagnostics,
        "prior_released_site_names": list(prior_released_sites) if prior_site_indices else [],
        "prior_released_site_indices": prior_site_indices,
        "prior_release_scale": float(prior_release_scale) if prior_site_indices else 0.0,
    }
    return positions, diagnostics


def scipy_pathfinder_preconditioner_chol(
    pathfinder_state: ScipyPathfinderResult, jitter: float = 1e-6
) -> jnp.ndarray:
    """Return the covariance Cholesky carried by a scipy Pathfinder result."""
    del jitter
    return jnp.asarray(pathfinder_state.chol, dtype=jnp.float32)
