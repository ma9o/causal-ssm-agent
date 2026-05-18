"""MAP parameter estimation via L-BFGS + Laplace parameter posterior.

Implements the outer optimization loop for MAP inference:
1. Build likelihood backend (Kalman or IEKS/Laplace)
2. Find the parameter mode via L-BFGS-B
3. Approximate parameter posterior via Laplace Gaussian or optimizer Hessian inverse
4. Extract constrained samples from the approximate posterior
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import scipy.optimize as spo
from jax.flatten_util import ravel_pytree

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.targets.base import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_KALMAN_EXACT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.inference.utils import (
    _build_eval_fns,
    _discover_sites,
    extract_constrained_samples,
)
from nof1_causal_lab.models.ssm.parameterization import assemble_deterministics_from_registry

logger = get_prefect_logger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

_SOLVER_KIND_LABELS = {
    LIKELIHOOD_SOLVER_KIND_KALMAN_EXACT: "kalman_exact",
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS: "point_ieks",
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS: "support_ieks",
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT: "dense_support",
}


def _shape_dtype_signature(array: jnp.ndarray) -> tuple[tuple[int, ...], str]:
    return tuple(array.shape), str(jnp.dtype(array.dtype))


@functools.partial(jax.jit, static_argnames=("runtime_log_posterior_fn",))
def _batch_log_posterior_runtime(
    candidates: jnp.ndarray,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_log_posterior_fn,
) -> jnp.ndarray:
    return jax.vmap(
        lambda z: runtime_log_posterior_fn(z, observations, times, latent_mode_init=None)
    )(candidates)


@functools.partial(jax.jit, static_argnames=("runtime_neg_log_posterior_with_aux_fn",))
def _laplace_value_and_grad_runtime(
    z: jnp.ndarray,
    latent_mode_init,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_neg_log_posterior_with_aux_fn,
):
    def _objective(z_arg, latent_mode_init_arg):
        return runtime_neg_log_posterior_with_aux_fn(
            z_arg,
            observations,
            times,
            latent_mode_init=latent_mode_init_arg,
        )

    return jax.value_and_grad(_objective, argnums=0, has_aux=True)(z, latent_mode_init)


@functools.partial(jax.jit, static_argnames=("runtime_neg_log_posterior_fn",))
def _laplace_parameter_hessian_runtime(
    z_mode: jnp.ndarray,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_neg_log_posterior_fn,
) -> jnp.ndarray:
    return jax.hessian(
        lambda z: runtime_neg_log_posterior_fn(z, observations, times, latent_mode_init=None)
    )(z_mode)


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
        "MAP outer %s: elapsed=%.1fs evals=%d objective=%.6f best=%.6f "
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
        "MAP inner %s: solver=%s n_iters=%d accepted=%d rel_change=%s "
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


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaplaceModeOptimizationResult:
    """Unified outer-optimizer result for MAP parameter mode finding."""

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


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def _build_map_laplace_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    trace_key: jnp.ndarray,
    likelihood_backend,
    reparam,
) -> dict[str, Any]:
    """Build the traced/JITed artifacts for optimizer-backed MAP."""
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

    cache_key = (
        "map_laplace_runtime_bundle",
        id(likelihood_backend),
        id(reparam),
        _shape_dtype_signature(observations),
        _shape_dtype_signature(times),
    )

    def _build_runtime_bundle() -> dict[str, Any]:
        log_lik_fn, log_prior_unc_fn, log_lik_with_aux_fn = _build_eval_fns(
            model,
            observations,
            times,
            site_info,
            unravel_fn,
            likelihood_backend=likelihood_backend,
            reparam=reparam,
            include_likelihood_aux=True,
            runtime_observations_times=True,
        )

        safe_floor = jnp.asarray(-1e30, dtype=observations.dtype)
        safe_ceiling = jnp.asarray(1e30, dtype=observations.dtype)

        def _log_posterior_fn(
            z: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
            latent_mode_init=None,
        ) -> jnp.ndarray:
            total = log_prior_unc_fn(z) + log_lik_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )
            return jnp.where(jnp.isfinite(total), total, safe_floor)

        def _neg_log_posterior_fn(
            z: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
            latent_mode_init=None,
        ) -> jnp.ndarray:
            value = -_log_posterior_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )
            return jnp.where(jnp.isfinite(value), value, safe_ceiling)

        def _neg_log_posterior_with_aux_fn(
            z: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
            latent_mode_init=None,
        ) -> tuple[jnp.ndarray, dict[str, Any]]:
            log_lik, inner_eval_aux = log_lik_with_aux_fn(
                z,
                runtime_observations,
                runtime_times,
                latent_mode_init=latent_mode_init,
            )
            log_prior = log_prior_unc_fn(z)
            log_posterior = log_prior + log_lik
            neg_log_posterior = -log_posterior
            safe_value = jnp.where(jnp.isfinite(neg_log_posterior), neg_log_posterior, safe_ceiling)
            outer_aux = {
                "log_posterior": log_posterior,
                "log_likelihood": log_lik,
                "log_prior": log_prior,
                "inner": {
                    key: value for key, value in inner_eval_aux.items() if key != "latent_mode"
                },
            }
            if "latent_mode" in inner_eval_aux:
                outer_aux["latent_mode"] = inner_eval_aux["latent_mode"]
            return safe_value, outer_aux

        return {
            "log_lik_fn": log_lik_fn,
            "log_prior_unc_fn": log_prior_unc_fn,
            "log_posterior_fn": _log_posterior_fn,
            "neg_log_posterior_fn": _neg_log_posterior_fn,
            "neg_log_posterior_with_aux_fn": _neg_log_posterior_with_aux_fn,
        }

    if hasattr(model, "get_cached_artifact"):
        runtime_bundle = model.get_cached_artifact(cache_key, _build_runtime_bundle)
    else:
        runtime_bundle = _build_runtime_bundle()

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        **runtime_bundle,
    }


# ---------------------------------------------------------------------------
# Init candidate sampling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Outer optimization
# ---------------------------------------------------------------------------


def _optimize_laplace_parameter_mode(
    _model,
    *,
    init_key: jnp.ndarray,
    dim: int,
    flat_example: jnp.ndarray,
    site_info: dict[str, Any],
    runtime_log_posterior_fn,
    runtime_neg_log_posterior_with_aux_fn,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_init_samples: int,
    maxiter: int,
    tol: float,
) -> LaplaceModeOptimizationResult:
    """Find the parameter mode using the route appropriate for the model class."""
    if dim == 0:
        z_mode = flat_example
        objective_at_mode, final_eval_aux = runtime_neg_log_posterior_with_aux_fn(
            z_mode,
            observations,
            times,
            latent_mode_init=None,
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
                jax.device_get(
                    runtime_log_posterior_fn(
                        z_mode,
                        observations,
                        times,
                        latent_mode_init=None,
                    )
                )
            ),
            optimizer_hess_inv=None,
            final_grad_norm=0.0,
            final_eval_diagnostics=_hostify_outer_eval_diagnostics(final_eval_aux),
        )

    support_aware_outer = _requires_support_aware_outer_optimizer(_model)
    if support_aware_outer:
        z_init = flat_example
        init_log_posterior_best: float | None = None
        logger.info("MAP init candidates skipped: support-aware outer optimizer")
    else:
        init_key, candidates = _draw_laplace_init_candidates(
            init_key,
            site_info,
            dim=dim,
            n_candidates=n_init_samples,
            dtype=observations.dtype,
        )
        del init_key
        init_scores = _batch_log_posterior_runtime(
            candidates,
            observations,
            times,
            runtime_log_posterior_fn=runtime_log_posterior_fn,
        )
        init_idx = int(jnp.argmax(init_scores))
        z_init = candidates[init_idx]
        init_log_posterior_best = float(jax.device_get(init_scores[init_idx]))
        logger.info(
            "MAP init candidates: n_candidates=%d best_log_posterior=%.6f",
            int(candidates.shape[0]),
            init_log_posterior_best,
        )

    cached_x: np.ndarray | None = None
    cached_fun: float | None = None
    cached_grad: np.ndarray | None = None
    cached_aux: dict[str, Any] | None = None
    eval_count = 0
    optimize_started_at = time.monotonic()
    latent_mode_init: np.ndarray | None = None
    if support_aware_outer:
        _seed_objective, seed_aux = runtime_neg_log_posterior_with_aux_fn(
            z_init,
            observations,
            times,
            latent_mode_init=None,
        )
        del _seed_objective
        if "latent_mode" in seed_aux:
            latent_mode_init = np.asarray(jax.device_get(seed_aux["latent_mode"])).copy()
            logger.info("MAP seeded latent warm start before jitted value-and-grad compile")

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
        (fun, aux), grad = _laplace_value_and_grad_runtime(
            z,
            latent_mode_arg,
            observations,
            times,
            runtime_neg_log_posterior_with_aux_fn=runtime_neg_log_posterior_with_aux_fn,
        )
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


# ---------------------------------------------------------------------------
# Parameter posterior sampling
# ---------------------------------------------------------------------------


def _sample_laplace_parameter_posterior(
    rng_key: jnp.ndarray,
    z_mode: jnp.ndarray,
    runtime_neg_log_posterior_fn,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_samples: int,
    hessian_jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample an unconstrained Gaussian approximation around the parameter mode."""
    if num_samples < 1:
        raise ValueError("map requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    if dim == 0:
        return (
            jnp.zeros((num_samples, 0), dtype=z_mode.dtype),
            jnp.zeros((0, 0), dtype=z_mode.dtype),
            jnp.zeros((0,), dtype=z_mode.dtype),
        )

    with jax.named_scope("map/parameter_hessian"):
        hessian = _laplace_parameter_hessian_runtime(
            z_mode,
            observations,
            times,
            runtime_neg_log_posterior_fn=runtime_neg_log_posterior_fn,
        )
        hessian = symmetrize_with_jitter(hessian, jitter=hessian_jitter)
        covariance = jla.solve(hessian, jnp.eye(dim, dtype=hessian.dtype), assume_a="pos")
        covariance = symmetrize_with_jitter(covariance, jitter=hessian_jitter)
        chol_cov = jnp.linalg.cholesky(covariance)

    with jax.named_scope("map/parameter_sampling"):
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
        raise ValueError("map requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    if dim == 0:
        return (
            jnp.zeros((num_samples, 0), dtype=z_mode.dtype),
            jnp.zeros((0, 0), dtype=z_mode.dtype),
            jnp.zeros((0,), dtype=z_mode.dtype),
        )
    if optimizer_hess_inv is None or not hasattr(optimizer_hess_inv, "todense"):
        raise RuntimeError("L-BFGS-B inverse-Hessian approximation is unavailable.")

    with jax.named_scope("map/optimizer_hess_inv"):
        covariance = jnp.asarray(
            np.asarray(optimizer_hess_inv.todense(), dtype=np.float64),
            dtype=z_mode.dtype,
        )
        covariance = symmetrize_with_jitter(covariance, jitter=hessian_jitter)
        chol_cov = jnp.linalg.cholesky(covariance)

    with jax.named_scope("map/parameter_sampling"):
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
        raise ValueError("map requires num_samples >= 1")

    dim = int(z_mode.shape[0])
    unc_samples = jnp.broadcast_to(z_mode, (num_samples, dim))
    covariance = jnp.zeros((dim, dim), dtype=z_mode.dtype)
    hessian_eigvals = jnp.zeros((0,), dtype=z_mode.dtype)
    return unc_samples, covariance, hessian_eigvals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit_map(
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
    ``fitSSM`` pattern: find the parameter mode of the approximate marginal
    posterior, compute the local curvature there, and sample the resulting
    Gaussian approximation in unconstrained parameter space.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"fit_map got unexpected keyword arguments: {unknown}")
    if parameter_covariance_method not in {"exact_hessian", "optimizer_hess_inv"}:
        raise ValueError(
            "parameter_covariance_method must be 'exact_hessian' or 'optimizer_hess_inv'."
        )

    rng_key = random.PRNGKey(seed)
    rng_key, trace_key, init_key, sample_key = random.split(rng_key, 4)

    backend_label = "kalman" if model.likelihood == "kalman" else "laplace_ieks"
    logger.info(
        "MAP config: backend=%s maxiter=%s tol=%s n_ieks_iters=%s "
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
    logger.info("MAP phase start: phase=build_likelihood_backend")
    with jax.profiler.TraceAnnotation("map/build_likelihood_backend"):
        if model.likelihood == "kalman":
            backend = model.make_likelihood_backend()
        else:
            backend = model.make_laplace_backend(n_ieks_iters)
    logger.info(
        "MAP phase complete: phase=build_likelihood_backend elapsed=%.1fs backend=%s",
        _elapsed_seconds(phase_started_at),
        backend_label,
    )

    phase_started_at = time.monotonic()
    logger.info("MAP phase start: phase=build_bundle")
    with jax.profiler.TraceAnnotation("map/build_bundle"):
        bundle = _build_map_laplace_bundle(
            model,
            observations,
            times,
            trace_key,
            backend,
            reparam,
        )
    logger.info(
        "MAP phase complete: phase=build_bundle elapsed=%.1fs",
        _elapsed_seconds(phase_started_at),
    )

    dim = bundle["dim"]
    flat_example = bundle["flat_example"]
    site_info = bundle["site_info"]
    unravel_fn = bundle["unravel_fn"]
    log_posterior_fn = bundle["log_posterior_fn"]
    neg_log_posterior_fn = bundle["neg_log_posterior_fn"]
    neg_log_posterior_with_aux_fn = bundle["neg_log_posterior_with_aux_fn"]
    logger.info("MAP bundle ready: parameter_dim=%d public_sites=%d", dim, len(site_info))

    logger.info(
        "MAP outer optimizer: method=%s support_aware=%s",
        "L-BFGS-B",
        _requires_support_aware_outer_optimizer(model),
    )
    phase_started_at = time.monotonic()
    logger.info("MAP phase start: phase=parameter_optimize")
    with jax.profiler.TraceAnnotation("map/parameter_optimize"):
        mode_result = _optimize_laplace_parameter_mode(
            model,
            init_key=init_key,
            dim=dim,
            flat_example=flat_example,
            site_info=site_info,
            runtime_log_posterior_fn=log_posterior_fn,
            runtime_neg_log_posterior_with_aux_fn=neg_log_posterior_with_aux_fn,
            observations=observations,
            times=times,
            n_init_samples=n_init_samples,
            maxiter=maxiter,
            tol=tol,
        )
    logger.info(
        "MAP phase complete: phase=parameter_optimize elapsed=%.1fs",
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
        "MAP mode found: optimizer=%s success=%s nit=%s nfev=%s objective=%.6f",
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
        raise RuntimeError("MAP failed to find a finite parameter mode.")

    parameter_hessian_min_eig = None
    parameter_hessian_max_eig = None
    if compute_parameter_hessian:
        logger.info(
            "MAP parameter curvature: dim=%s method=%s sampling local Gaussian posterior",
            dim,
            parameter_covariance_method,
        )
        phase_started_at = time.monotonic()
        logger.info("MAP phase start: phase=parameter_curvature")
        with jax.profiler.TraceAnnotation("map/sample_parameter_posterior"):
            if parameter_covariance_method == "exact_hessian":
                unc_samples, covariance, hessian_eigvals = _sample_laplace_parameter_posterior(
                    sample_key,
                    z_mode,
                    neg_log_posterior_fn,
                    observations,
                    times,
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
            "MAP phase complete: phase=parameter_curvature elapsed=%.1fs",
            _elapsed_seconds(phase_started_at),
        )
        parameter_posterior_strategy = "laplace_gaussian"
    else:
        logger.info("MAP parameter Hessian skipped; using deterministic mode samples")
        unc_samples, covariance, hessian_eigvals = _mode_only_parameter_posterior(
            z_mode,
            num_samples=num_samples,
        )
        parameter_posterior_strategy = "mode_only"

    if site_info:
        phase_started_at = time.monotonic()
        logger.info("MAP phase start: phase=extract_samples")
        with jax.profiler.TraceAnnotation("map/extract_samples"):
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
            "MAP phase complete: phase=extract_samples elapsed=%.1fs draws=%d",
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
                "MAP parameter curvature exact_hessian: min_eig=%s max_eig=%s condition=%s",
                _format_float(parameter_hessian_min_eig),
                _format_float(parameter_hessian_max_eig),
                _format_float(hessian_condition_number),
            )
        else:
            covariance_diag = np.asarray(jax.device_get(jnp.diag(covariance)), dtype=np.float64)
            logger.info(
                "MAP parameter curvature optimizer_hess_inv: covariance_diag_min=%s covariance_diag_max=%s",
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
        # Full parameter covariance in the flat unconstrained layout used by
        # the auxiliary-Kalman bundle. Downstream MCMC methods can use this as
        # a preconditioner / mass matrix for MALA (see `build_mala_parameter_kernel`).
        "parameter_covariance": np.asarray(covariance),
        "likelihood_backend": backend,
    }

    logger.info(
        "MAP complete: success=%s status=%s nit=%s nfev=%s loglik=%.3f logpost=%.3f",
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


__all__ = [
    "LaplaceModeOptimizationResult",
    "_build_map_laplace_bundle",
    "_draw_laplace_init_candidates",
    "_elapsed_seconds",
    "_format_float",
    "_hostify_inner_eval_diagnostics",
    "_hostify_outer_eval_diagnostics",
    "_inner_log_joint_gain",
    "_log_outer_eval",
    "_mode_only_parameter_posterior",
    "_optimize_laplace_parameter_mode",
    "_requires_support_aware_outer_optimizer",
    "_sample_laplace_parameter_posterior",
    "_sample_laplace_parameter_posterior_from_optimizer_hess_inv",
    "_scalar_float",
    "_scalar_int",
    "_solver_label",
    "fit_map",
]
