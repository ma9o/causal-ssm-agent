"""Scipy-driven Pathfinder: multi-start L-BFGS-B + best-ELBO Gaussian selection.

Why not ``blackjax.vi.pathfinder``: its JAX-native L-BFGS drives the entire
optimisation trajectory inside a jit-compiled ``lax.while_loop``, producing a
graph whose size is ``maxiter * lbfgs_history * num_elbo_samples`` larger than
the log-posterior graph itself. At T=1000 on A100 this blows up compile time
to ~1700 s and HBM to ~25 GiB (measured). The scipy pattern only jit-compiles
the log-posterior + gradient once (same compile cost as ``fit_map``); L-BFGS
iteration, L-BFGS-history Hessian reconstruction, and ELBO computation are
all plain numpy at Python level — orders of magnitude cheaper for one-shot
preconditioner/init builds.

Algorithmically equivalent to Pathfinder (Zhang et al. 2022): at each
L-BFGS iterate form the local Gaussian approximation using the L-BFGS
quasi-Newton inverse-Hessian (via the two-loop recursion), compute ELBO by
Monte Carlo sampling, return the approximation with highest ELBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.optimize

if TYPE_CHECKING:
    from collections.abc import Callable


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


def scipy_pathfinder(
    log_post_and_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0_starts: list[np.ndarray],
    *,
    maxiter: int = 20,
    elbo_samples: int = 10,
    lbfgs_memory: int = 10,
    jitter: float = 1e-6,
    seed: int = 0,
) -> ScipyPathfinderResult:
    """Multi-start scipy-driven Pathfinder.

    Parameters
    ----------
    log_post_and_grad_fn : callable
        ``x -> (log_posterior_scalar, gradient_vector)``. Accepts and returns
        numpy arrays. Caller is responsible for any jit compilation inside.
    x0_starts : list[np.ndarray]
        One starting point per L-BFGS run. Length = number of multi-starts.
    maxiter : int
        L-BFGS outer-iteration budget per start.
    elbo_samples : int
        Monte Carlo samples used to estimate ELBO at each candidate iterate.
    lbfgs_memory : int
        L-BFGS curvature-history depth (standard ``m``).
    jitter : float
        Added to ``H^{-1}`` diagonal before Cholesky for numerical stability.
    seed : int
        RNG seed for ELBO sample draws.

    Returns
    -------
    ScipyPathfinderResult with the best-ELBO Gaussian across all starts.
    """
    if not x0_starts:
        raise ValueError("scipy_pathfinder requires at least one start.")
    dim = int(x0_starts[0].shape[0])
    rng = np.random.default_rng(seed)

    best_mean: np.ndarray | None = None
    best_chol: np.ndarray | None = None
    best_elbo = -np.inf
    per_start_diagnostics: list[dict[str, Any]] = []

    for start_idx, x0 in enumerate(x0_starts):
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

        # scipy's L-BFGS-B returns the final inverse-Hessian as a
        # ``LbfgsInvHessProduct`` — an always-PSD low-rank operator. Use it as
        # a guaranteed-valid candidate Gaussian at the final iterate; our own
        # per-iterate two-loop recursion then adds best-ELBO selection on top.
        scipy_hess_inv_cand: tuple[np.ndarray, np.ndarray, float] | None = None
        try:
            hess_inv_op = getattr(opt_result, "hess_inv", None)
            if hess_inv_op is not None:
                H_inv_final = np.asarray(hess_inv_op.todense(), dtype=np.float64)
                H_inv_final = 0.5 * (H_inv_final + H_inv_final.T)
                H_inv_final = H_inv_final + jitter * np.eye(dim, dtype=np.float64)
                L_final = np.linalg.cholesky(H_inv_final)
                x_final = np.asarray(opt_result.x, dtype=np.float64).copy()
                # ELBO at the final iterate with the scipy hess_inv Gaussian.
                zeta = rng.standard_normal((elbo_samples, dim))
                samples = x_final[None, :] + zeta @ L_final.T
                log_post_samples = np.empty(elbo_samples, dtype=np.float64)
                for i in range(elbo_samples):
                    lp, _ = log_post_and_grad_fn(samples[i])
                    log_post_samples[i] = float(lp)
                log_det_term = float(np.log(np.abs(np.diag(L_final))).sum())
                log_q = (
                    -0.5 * np.sum(zeta**2, axis=1) - log_det_term - 0.5 * dim * np.log(2.0 * np.pi)
                )
                elbo_final = float(np.mean(log_post_samples - log_q))
                if np.isfinite(elbo_final):
                    scipy_hess_inv_cand = (x_final, L_final, elbo_final)
        except (np.linalg.LinAlgError, ValueError, AttributeError):
            pass

        # Iterate along the accepted L-BFGS iterates, forming H^{-1} at each
        # point from the most recent curvature history and scoring the
        # resulting Gaussian by ELBO. trajectory[0] is the init; skip it since
        # we have no history yet.
        start_best_elbo = -np.inf
        start_best_mean: np.ndarray | None = None
        start_best_chol: np.ndarray | None = None
        valid_iterate_count = 0

        # Pull out positions + gradients separately for history construction.
        traj_xg = [(pt[0], pt[1]) for pt in trajectory]

        for k in range(1, len(trajectory)):
            history = _build_curvature_history(traj_xg, k, lbfgs_memory)
            if not history:
                continue
            H_inv = _form_lbfgs_inverse_hessian_matrix(history, dim)
            H_inv = 0.5 * (H_inv + H_inv.T)
            H_inv = H_inv + jitter * np.eye(dim, dtype=np.float64)
            try:
                L_k = np.linalg.cholesky(H_inv)
            except np.linalg.LinAlgError:
                continue  # skip non-PSD reconstructions

            x_k = trajectory[k][0]
            # MC ELBO: E_q[log π] - E_q[log q]. q = N(x_k, L_k L_k^T).
            zeta = rng.standard_normal((elbo_samples, dim))
            samples = x_k[None, :] + zeta @ L_k.T
            log_post_at_samples = np.empty(elbo_samples, dtype=np.float64)
            for i in range(elbo_samples):
                lp, _ = log_post_and_grad_fn(samples[i])
                log_post_at_samples[i] = float(lp)
            # log q(sample) = -0.5 * ||zeta||^2 - sum(log diag(L_k)) - 0.5*p*log(2π)
            log_det_term = float(np.log(np.abs(np.diag(L_k))).sum())
            log_q = -0.5 * np.sum(zeta**2, axis=1) - log_det_term - 0.5 * dim * np.log(2.0 * np.pi)
            elbo_k = float(np.mean(log_post_at_samples - log_q))
            valid_iterate_count += 1

            if np.isfinite(elbo_k) and elbo_k > start_best_elbo:
                start_best_elbo = elbo_k
                start_best_mean = x_k.copy()
                start_best_chol = L_k.copy()

        # Consider scipy's own hess_inv candidate. Always PSD by construction,
        # so it rescues starts where our custom reconstruction never found a
        # valid Gaussian (can happen on tiny-T smoke models where the
        # (s, y) history is too short to be informative).
        if scipy_hess_inv_cand is not None:
            x_final, L_final, elbo_final = scipy_hess_inv_cand
            if elbo_final > start_best_elbo:
                start_best_elbo = elbo_final
                start_best_mean = x_final
                start_best_chol = L_final

        per_start_diagnostics.append(
            {
                "start_idx": int(start_idx),
                "n_trajectory_points": len(trajectory),
                "n_valid_iterates": int(valid_iterate_count),
                "n_lbfgs_iterations": int(opt_result.nit),
                "final_log_posterior": float(-opt_result.fun),
                "best_elbo_this_start": (
                    float(start_best_elbo) if start_best_elbo > -np.inf else None
                ),
                "scipy_success": bool(opt_result.success),
                "scipy_status": int(opt_result.status),
            }
        )

        if (
            start_best_elbo > best_elbo
            and start_best_mean is not None
            and start_best_chol is not None
        ):
            best_elbo = start_best_elbo
            best_mean = start_best_mean
            best_chol = start_best_chol

    if best_mean is None or best_chol is None:
        raise RuntimeError(
            "scipy_pathfinder found no valid Gaussian approximation across all starts; "
            "all accepted L-BFGS iterates and final inverse-Hessian candidates failed."
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
            "lbfgs_memory": int(lbfgs_memory),
            "maxiter": int(maxiter),
            "elbo_min": float(min(elbo_values)) if elbo_values else float("nan"),
            "elbo_max": float(max(elbo_values)) if elbo_values else float("nan"),
            "elbo_spread": (float(max(elbo_values) - min(elbo_values)) if elbo_values else 0.0),
        },
    )
