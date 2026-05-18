"""Dataset-conditioned local geometry via multi-start MAP and Hessians."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import scipy.optimize as spo

from causal_ssm_agent.models.ssm.inference.targets.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.parameterization import SupportClass, sample_prior_unconstrained

from .context import ParametricIdContext, get_stage4b_sweep_context
from .results import MAPCurvatureResult, MAPGeometryResult, MAPOptimizationRun
from .sensitivity import (
    _interpretable_parameter_name_map,
    _normalized_direction_status,
    _split_scalar_parameter_name,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel

_NEGATIVE_EIGENVALUE_TOL = 1e-6
_BOUNDARY_RELATIVE_TOL = 0.05
_POSITIVE_BOUNDARY_UNCONSTRAINED_TOL = -4.0
_PARAMETER_LOADING_THRESHOLD = 0.1


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _condition_number(eigenvalues: jnp.ndarray) -> float | None:
    positive = eigenvalues[eigenvalues > NUMERICAL_EPSILON]
    if positive.size == 0 or positive.shape[0] != eigenvalues.shape[0]:
        return None
    smallest = float(jnp.min(positive))
    largest = float(jnp.max(positive))
    if smallest <= NUMERICAL_EPSILON:
        return None
    return largest / smallest


def _per_param_effective_eigenvalue(
    eigenvectors: jnp.ndarray, eigenvalues: jnp.ndarray
) -> jnp.ndarray:
    n_parameters = eigenvectors.shape[0]
    effective = jnp.full((n_parameters,), float(jnp.max(eigenvalues)))
    for param_idx in range(n_parameters):
        significant = jnp.abs(eigenvectors[param_idx, :]) > _PARAMETER_LOADING_THRESHOLD
        if jnp.any(significant):
            effective = effective.at[param_idx].set(
                jnp.min(jnp.where(significant, eigenvalues, jnp.inf))
            )
    return effective


def _raw_curvature_status(value: float, *, reference_scale: float) -> str:
    if value <= 0.0:
        return "fail"
    if value > 1e-3 * reference_scale:
        return "pass"
    if value > 1e-6 * reference_scale:
        return "warn"
    return "fail"


def _summarize_curvature(
    hessian: jnp.ndarray,
    *,
    prior_std: jnp.ndarray,
    scalar_names: list[str],
    interpretable_names: dict[str, str],
) -> MAPCurvatureResult:
    n_parameters = len(scalar_names)
    if n_parameters == 0:
        return MAPCurvatureResult(
            eigenvalues=[],
            normalized_eigenvalues=[],
            negative_direction_count=0,
            deficiency_count=0,
            positive_definite=True,
            condition_number=None,
            normalized_condition_number=None,
            weak_directions=[],
            per_parameter=[],
        )

    hessian = _symmetrize(hessian)
    hessian_norm = (prior_std[:, None] * hessian) * prior_std[None, :]

    eigvals, eigvecs = jnp.linalg.eigh(hessian)
    eigvals_norm, eigvecs_norm = jnp.linalg.eigh(hessian_norm)

    order = jnp.arange(n_parameters - 1, -1, -1)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals_norm = eigvals_norm[order]
    eigvecs_norm = eigvecs_norm[:, order]

    effective_eigvals = _per_param_effective_eigenvalue(eigvecs, eigvals)
    effective_norm_eigvals = _per_param_effective_eigenvalue(eigvecs_norm, eigvals_norm)

    negative_direction_count = int(jnp.sum(eigvals < -_NEGATIVE_EIGENVALUE_TOL))
    deficiency_count = int(jnp.sum(eigvals_norm < 1.0))
    positive_definite = bool(jnp.all(eigvals > _NEGATIVE_EIGENVALUE_TOL))

    representative_norm_v = np.asarray(eigvecs_norm, dtype=float)
    weak_directions = []
    highlighted_indices = [
        idx
        for idx, value in sorted(
            enumerate(np.asarray(eigvals_norm, dtype=float)),
            key=lambda item: item[1],
        )
        if _normalized_direction_status(float(value)) != "pass"
    ]
    for direction_idx in highlighted_indices:
        loadings = representative_norm_v[:, direction_idx].copy()
        top_indices = np.argsort(np.abs(loadings))[::-1][: min(15, loadings.shape[0])]
        if top_indices.size > 0 and loadings[top_indices[0]] < 0:
            loadings *= -1.0

        top_loadings = []
        for param_idx in top_indices:
            scalar_name = scalar_names[param_idx]
            loading = float(loadings[param_idx])
            top_loadings.append(
                {
                    "parameter": scalar_name,
                    "interpretable_parameter": interpretable_names[scalar_name],
                    "loading": loading,
                    "abs_loading": float(abs(loading)),
                }
            )

        normalized_eigenvalue = float(eigvals_norm[direction_idx])
        weak_directions.append(
            {
                "index": direction_idx + 1,
                "eigenvalue": float(eigvals[direction_idx]),
                "normalized_eigenvalue": normalized_eigenvalue,
                "status": _normalized_direction_status(normalized_eigenvalue),
                "top_loadings": top_loadings,
            }
        )

    raw_reference_scale = float(jnp.max(jnp.maximum(jnp.abs(eigvals), NUMERICAL_EPSILON)))
    per_parameter = []
    diag = jnp.diag(hessian)
    for param_idx, scalar_name in enumerate(scalar_names):
        eff_raw = float(effective_eigvals[param_idx])
        eff_norm = float(effective_norm_eigvals[param_idx])
        per_parameter.append(
            {
                "parameter": scalar_name,
                "interpretable_parameter": interpretable_names[scalar_name],
                "diagonal_curvature": float(diag[param_idx]),
                "effective_eigenvalue": eff_raw,
                "status": _raw_curvature_status(eff_raw, reference_scale=raw_reference_scale),
                "normalized_effective_eigenvalue": eff_norm,
                "normalized_status": _normalized_direction_status(eff_norm),
            }
        )

    return MAPCurvatureResult(
        eigenvalues=[float(value) for value in eigvals],
        normalized_eigenvalues=[float(value) for value in eigvals_norm],
        negative_direction_count=negative_direction_count,
        deficiency_count=deficiency_count,
        positive_definite=positive_definite,
        condition_number=_condition_number(eigvals),
        normalized_condition_number=_condition_number(eigvals_norm),
        weak_directions=weak_directions,
        per_parameter=per_parameter,
        parameter_names=list(scalar_names),
        eigenvectors_normalized=representative_norm_v.tolist(),
    )


def _boundary_issue_parameters(
    context: ParametricIdContext,
    prior_state,
    z_map: jnp.ndarray,
) -> list[str]:
    registry_by_name = {site.name: site for site in context.registry}
    unconstrained = context.unravel_fn(z_map)
    constrained = {name: context.transforms[name](value) for name, value in unconstrained.items()}

    issues: list[str] = []
    for scalar_name in context.scalar_names:
        site_name, flat_index = _split_scalar_parameter_name(scalar_name)
        site = registry_by_name.get(site_name)
        if site is None:
            continue

        unc_flat = jnp.asarray(unconstrained[site_name]).reshape(-1)
        con_flat = jnp.asarray(constrained[site_name]).reshape(-1)
        if flat_index >= unc_flat.shape[0] or flat_index >= con_flat.shape[0]:
            continue

        unc_value = float(unc_flat[flat_index])
        con_value = float(con_flat[flat_index])

        if site.support == SupportClass.CORRELATION:
            if 1.0 - abs(con_value) <= _BOUNDARY_RELATIVE_TOL:
                issues.append(scalar_name)
            continue

        params = prior_state.get(site_name, {})
        low = params.get("low")
        high = params.get("high")
        if low is not None and high is not None:
            low_arr = np.asarray(low, dtype=float).reshape(-1)
            high_arr = np.asarray(high, dtype=float).reshape(-1)
            if flat_index < low_arr.shape[0] and flat_index < high_arr.shape[0]:
                low_value = float(low_arr[flat_index])
                high_value = float(high_arr[flat_index])
                width = high_value - low_value
                if np.isfinite(low_value) and np.isfinite(high_value) and width > 0:
                    dist_to_edge = min(con_value - low_value, high_value - con_value)
                    if dist_to_edge / width <= _BOUNDARY_RELATIVE_TOL:
                        issues.append(scalar_name)
                        continue

        if (
            site.support == SupportClass.POSITIVE
            and unc_value <= _POSITIVE_BOUNDARY_UNCONSTRAINED_TOL
        ):
            issues.append(scalar_name)

    return sorted(set(issues))


def map_geometry_analysis(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    n_starts: int = 8,
    seed: int = 42,
    sweep_context: ParametricIdContext | None = None,
    optimizer_options: dict | None = None,
    parallel_workers: int | None = None,
    initial_starts: list[jnp.ndarray | np.ndarray] | None = None,
    initial_start_kinds: list[str] | None = None,
) -> MAPGeometryResult:
    """Run a multi-start MAP search, then compare H_lik and H_post at the best mode."""
    rng_key = random.PRNGKey(seed)
    context = sweep_context or get_stage4b_sweep_context(model)
    prior_state = model.get_prior_runtime_bundle().prior_state

    flat_dim = context.flat_dim
    scalar_names = context.scalar_names
    interpretable_names = _interpretable_parameter_name_map(model, scalar_names)

    def _log_likelihood(z):
        return context.log_lik_fn(z, observations, times)

    def _log_prior(z):
        return context.log_prior_unc_fn(z, prior_state)

    def _log_posterior(z):
        return _log_likelihood(z) + _log_prior(z)

    def _neg_log_posterior(z):
        value = -_log_posterior(z)
        return jnp.where(jnp.isfinite(value), value, jnp.array(1e10, dtype=z.dtype))

    if flat_dim == 0:
        likelihood_curvature = _summarize_curvature(
            jnp.zeros((0, 0), dtype=jnp.float64),
            prior_std=jnp.zeros((0,), dtype=jnp.float64),
            scalar_names=scalar_names,
            interpretable_names=interpretable_names,
        )
        return MAPGeometryResult(
            n_starts=1,
            n_successful_starts=1,
            best_start_index=0,
            map_log_posterior=float(_log_posterior(jnp.zeros((0,), dtype=jnp.float64))),
            map_log_likelihood=float(_log_likelihood(jnp.zeros((0,), dtype=jnp.float64))),
            map_log_prior=float(_log_prior(jnp.zeros((0,), dtype=jnp.float64))),
            final_grad_norm=0.0,
            runner_up_objective_gap=None,
            starts=[
                MAPOptimizationRun(
                    index=0,
                    start_kind="zero",
                    start_log_posterior=float(_log_posterior(jnp.zeros((0,), dtype=jnp.float64))),
                    log_posterior=float(_log_posterior(jnp.zeros((0,), dtype=jnp.float64))),
                    log_likelihood=float(_log_likelihood(jnp.zeros((0,), dtype=jnp.float64))),
                    log_prior=float(_log_prior(jnp.zeros((0,), dtype=jnp.float64))),
                    objective=float(_neg_log_posterior(jnp.zeros((0,), dtype=jnp.float64))),
                    success=True,
                    status=0,
                    message="no free parameters",
                    n_iters=0,
                    n_function_evals=1,
                    grad_norm=0.0,
                    distance_to_best=0.0,
                )
            ],
            likelihood_curvature=likelihood_curvature,
            posterior_curvature=likelihood_curvature,
            prior_rescued_parameters=[],
            boundary_parameters=[],
            z_map_unconstrained=[],
            prior_std_unconstrained=[],
        )

    n_candidate_draws = max(int(n_starts) * 4, 16)
    prior_draws, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=n_candidate_draws,
    )
    prior_draws_std, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=max(64, n_candidate_draws),
    )
    prior_std = jnp.std(prior_draws_std, axis=0)
    prior_std = jnp.maximum(prior_std, NUMERICAL_EPSILON)

    batch_log_posterior = jax.jit(jax.vmap(_log_posterior))
    if initial_starts is None:
        candidate_kinds = ["zero", "prior_median"] + [
            f"prior_draw_{idx}" for idx in range(int(prior_draws.shape[0]))
        ]
        candidates = jnp.concatenate(
            [
                jnp.zeros((1, flat_dim), dtype=prior_draws.dtype),
                jnp.median(prior_draws, axis=0, keepdims=True),
                prior_draws,
            ],
            axis=0,
        )
        candidate_scores = batch_log_posterior(candidates)
        candidate_scores = jnp.where(
            jnp.isfinite(candidate_scores),
            candidate_scores,
            jnp.asarray(-jnp.inf, dtype=candidate_scores.dtype),
        )
        order = np.asarray(jnp.argsort(candidate_scores)[::-1], dtype=int)
        selected = order[: max(int(n_starts), 1)]
        selected_starts = [
            (
                run_idx,
                candidates[candidate_idx],
                candidate_kinds[candidate_idx],
                float(candidate_scores[candidate_idx]),
            )
            for run_idx, candidate_idx in enumerate(selected)
        ]
    else:
        if not initial_starts:
            raise ValueError("initial_starts must contain at least one start.")
        if initial_start_kinds is None:
            start_kinds = [f"initial_start_{idx}" for idx in range(len(initial_starts))]
        else:
            if len(initial_start_kinds) != len(initial_starts):
                raise ValueError("initial_start_kinds must match initial_starts length.")
            start_kinds = list(initial_start_kinds)
        start_arrays = []
        for start in initial_starts:
            start_array = jnp.asarray(start, dtype=prior_draws.dtype).reshape(-1)
            if start_array.shape[0] != flat_dim:
                raise ValueError(
                    f"initial start has dimension {start_array.shape[0]}, expected {flat_dim}"
                )
            start_arrays.append(start_array)
        candidates = jnp.stack(start_arrays, axis=0)
        candidate_scores = batch_log_posterior(candidates)
        candidate_scores = jnp.where(
            jnp.isfinite(candidate_scores),
            candidate_scores,
            jnp.asarray(-jnp.inf, dtype=candidate_scores.dtype),
        )
        selected_starts = [
            (run_idx, candidates[run_idx], start_kinds[run_idx], float(candidate_scores[run_idx]))
            for run_idx in range(len(start_arrays))
        ]

    value_and_grad_fn = jax.jit(jax.value_and_grad(_neg_log_posterior))
    log_likelihood_jit = jax.jit(_log_likelihood)
    log_prior_jit = jax.jit(_log_prior)
    log_posterior_jit = jax.jit(_log_posterior)
    h_likelihood_fn = jax.jit(jax.hessian(lambda z: -_log_likelihood(z)))
    h_prior_fn = jax.jit(jax.hessian(lambda z: -_log_prior(z)))

    runs: list[MAPOptimizationRun] = []
    z_solutions: list[jnp.ndarray] = []

    def _optimize_one(index: int, start: jnp.ndarray, start_kind: str, start_lp: float):
        cached_x: np.ndarray | None = None
        cached_fun: float | None = None
        cached_grad: np.ndarray | None = None

        def _value_and_grad(x_np: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal cached_x, cached_fun, cached_grad
            x_host = np.asarray(x_np, dtype=np.float64)
            if cached_x is not None and np.array_equal(x_host, cached_x):
                assert cached_fun is not None
                assert cached_grad is not None
                return cached_fun, cached_grad

            z = jnp.asarray(x_host, dtype=start.dtype)
            fun, grad = value_and_grad_fn(z)
            cached_x = x_host.copy()
            cached_fun = float(jax.device_get(fun))
            cached_grad = np.asarray(jax.device_get(grad), dtype=np.float64)
            return cached_fun, cached_grad

        def _objective(x_np: np.ndarray) -> float:
            fun, _grad = _value_and_grad(x_np)
            return fun

        def _gradient(x_np: np.ndarray) -> np.ndarray:
            _fun, grad = _value_and_grad(x_np)
            return grad

        result = spo.minimize(
            _objective,
            np.asarray(jax.device_get(start), dtype=np.float64),
            method="L-BFGS-B",
            jac=_gradient,
            options=optimizer_options,
        )
        z_opt = jnp.asarray(result.x, dtype=start.dtype)
        grad_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=np.float64), ord=2))
        log_posterior = float(jax.device_get(log_posterior_jit(z_opt)))
        log_likelihood = float(jax.device_get(log_likelihood_jit(z_opt)))
        log_prior = float(jax.device_get(log_prior_jit(z_opt)))
        return (
            MAPOptimizationRun(
                index=index,
                start_kind=start_kind,
                start_log_posterior=float(start_lp),
                log_posterior=log_posterior,
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                objective=float(result.fun),
                success=bool(result.success) and np.isfinite(result.fun),
                status=int(getattr(result, "status", 0)),
                message=str(getattr(result, "message", "")),
                n_iters=int(getattr(result, "nit", 0)),
                n_function_evals=int(getattr(result, "nfev", 0)),
                grad_norm=grad_norm,
            ),
            z_opt,
        )

    worker_count = 1 if parallel_workers is None else int(parallel_workers)
    if worker_count < 1:
        raise ValueError("parallel_workers must be >= 1.")
    worker_count = min(worker_count, len(selected_starts))

    if selected_starts:
        # Trigger JIT compilation before threads start so workers share compiled
        # callables instead of racing to compile the same objective.
        warm_start = selected_starts[0][1]
        warm_value, warm_grad = value_and_grad_fn(warm_start)
        jax.block_until_ready(warm_value)
        jax.block_until_ready(warm_grad)

    if worker_count == 1:
        start_results = [
            _optimize_one(run_idx, start, start_kind, start_lp)
            for run_idx, start, start_kind, start_lp in selected_starts
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_optimize_one, run_idx, start, start_kind, start_lp)
                for run_idx, start, start_kind, start_lp in selected_starts
            ]
            start_results = [future.result() for future in futures]

    for run, z_opt in start_results:
        runs.append(run)
        z_solutions.append(z_opt)

    valid_indices = [
        idx
        for idx, run in enumerate(runs)
        if np.isfinite(run.objective) and np.isfinite(run.log_posterior)
    ]
    if not valid_indices:
        raise RuntimeError("multi-start MAP search produced no finite optimization result")

    best_idx = min(valid_indices, key=lambda idx: runs[idx].objective)
    z_map = z_solutions[best_idx]
    runner_up_objective_gap = None
    if len(valid_indices) > 1:
        sorted_objectives = sorted(runs[idx].objective for idx in valid_indices)
        runner_up_objective_gap = float(sorted_objectives[1] - sorted_objectives[0])

    best_host = np.asarray(jax.device_get(z_map), dtype=np.float64)
    updated_runs: list[MAPOptimizationRun] = []
    for idx, run in enumerate(runs):
        distance = float(
            np.linalg.norm(
                np.asarray(jax.device_get(z_solutions[idx]), dtype=np.float64) - best_host
            )
        )
        updated_runs.append(
            MAPOptimizationRun(
                index=run.index,
                start_kind=run.start_kind,
                start_log_posterior=run.start_log_posterior,
                log_posterior=run.log_posterior,
                log_likelihood=run.log_likelihood,
                log_prior=run.log_prior,
                objective=run.objective,
                success=run.success,
                status=run.status,
                message=run.message,
                n_iters=run.n_iters,
                n_function_evals=run.n_function_evals,
                grad_norm=run.grad_norm,
                distance_to_best=distance,
            )
        )

    h_likelihood = _symmetrize(h_likelihood_fn(z_map))
    # Reuse the full-grid likelihood Hessian; the posterior Hessian is additive.
    h_posterior = _symmetrize(h_likelihood + h_prior_fn(z_map))
    likelihood_curvature = _summarize_curvature(
        h_likelihood,
        prior_std=prior_std,
        scalar_names=scalar_names,
        interpretable_names=interpretable_names,
    )
    posterior_curvature = _summarize_curvature(
        h_posterior,
        prior_std=prior_std,
        scalar_names=scalar_names,
        interpretable_names=interpretable_names,
    )

    posterior_status = {
        entry["parameter"]: entry["normalized_status"]
        for entry in posterior_curvature.per_parameter
    }
    prior_rescued_parameters = sorted(
        entry["parameter"]
        for entry in likelihood_curvature.per_parameter
        if entry["normalized_status"] != "pass"
        and posterior_status.get(entry["parameter"]) == "pass"
    )

    best_run = updated_runs[best_idx]
    return MAPGeometryResult(
        n_starts=len(updated_runs),
        n_successful_starts=sum(1 for run in updated_runs if run.success),
        best_start_index=best_idx,
        map_log_posterior=best_run.log_posterior,
        map_log_likelihood=best_run.log_likelihood,
        map_log_prior=best_run.log_prior,
        final_grad_norm=best_run.grad_norm,
        runner_up_objective_gap=runner_up_objective_gap,
        starts=updated_runs,
        likelihood_curvature=likelihood_curvature,
        posterior_curvature=posterior_curvature,
        prior_rescued_parameters=prior_rescued_parameters,
        boundary_parameters=_boundary_issue_parameters(context, prior_state, z_map),
        z_map_unconstrained=best_host.tolist(),
        prior_std_unconstrained=[float(value) for value in prior_std],
    )
