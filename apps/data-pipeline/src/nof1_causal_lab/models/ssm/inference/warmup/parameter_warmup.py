"""Shared parameter warmup and preconditioner setup for blocked SSM MCMC.

Init positions (Pathfinder draws) and the preconditioner (Pathfinder or
Laplace/MAP covariance) are Gaussian approximations used only to warm-start the
sampler. Seeding from them is valid no matter how poor the linearization is: the
invariant distribution is the exact posterior for any seed, so the approximation
sets burn-in, not the target (Andrieu, Doucet & Holenstein 2010). init_positions
consumes only the location, never the covariance, so a bad linearization cannot
bias the result; the preconditioner uses the covariance but only as a proposal
scale that the Metropolis accept/reject corrects and adaptation re-tunes. These
Gaussians are seeds, never the reported posterior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.inference.warmup.scipy_pathfinder import (
    ScipyPathfinderResult,
    run_scipy_pathfinder_approximation,
    sample_scipy_pathfinder_init_positions,
    scipy_pathfinder_preconditioner_chol,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.bundle import CachedParticleRuntimeBundle
    from nof1_causal_lab.models.ssm.inference.types import WarmupProposal

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_RELEASED_SITE_NAMES: tuple[str, ...] = ("obs_df",)


@dataclass(frozen=True)
class ParameterWarmupResult:
    """Resolved parameter initialisation and preconditioning artifacts."""

    init_positions: jnp.ndarray | None
    init_diagnostics: UncheckedJsonObject
    preconditioner_chol: jnp.ndarray | None
    preconditioner_diagnostics: UncheckedJsonObject
    warmup_diagnostics: UncheckedJsonObject
    pathfinder_state: ScipyPathfinderResult | None
    pathfinder_diagnostics: UncheckedJsonObject | None


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


def _laplace_preconditioner_chol_from_map_result(
    map_result: WarmupProposal, jitter: float = 1e-6
) -> jnp.ndarray:
    """Build a parameter-kernel preconditioner Cholesky from ``fit_map`` covariance."""
    covariance = np.asarray(map_result.diagnostics["parameter_covariance"])
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + jitter * np.eye(covariance.shape[0], dtype=covariance.dtype)
    return jnp.asarray(np.linalg.cholesky(covariance), dtype=jnp.float32)


def _validate_initial_positions_override(
    initial_positions_override: jnp.ndarray,
    *,
    num_chains: int,
    dim: int,
    dtype,
) -> jnp.ndarray:
    init_positions = jnp.asarray(initial_positions_override, dtype=dtype)
    if init_positions.shape != (num_chains, dim):
        raise ValueError(
            "initial_positions_override must have shape (num_chains, dim); got "
            f"{init_positions.shape}"
        )
    return init_positions


def _pathfinder_preconditioner_diagnostics(
    pathfinder_diagnostics: UncheckedJsonObject,
) -> UncheckedJsonObject:
    return {
        "auto_preconditioner": True,
        "auto_preconditioner_method": "pathfinder",
        "auto_preconditioner_device": jax.default_backend(),
        "auto_preconditioner_n_pathfinder_starts": int(
            pathfinder_diagnostics["n_pathfinder_starts"]
        ),
        "auto_preconditioner_n_pathfinder_starts_finite": int(
            pathfinder_diagnostics["n_pathfinder_starts_finite"]
        ),
        "auto_preconditioner_best_pathfinder_elbo": float(
            pathfinder_diagnostics["best_pathfinder_elbo"]
        ),
        "auto_preconditioner_pathfinder_elbo_spread": float(
            pathfinder_diagnostics["pathfinder_elbo_spread"]
        ),
    }


def _log_pathfinder_completion(
    *,
    phase_label: str,
    started_at: float,
    pathfinder_diagnostics: UncheckedJsonObject,
) -> None:
    best_elbo = pathfinder_diagnostics.get("best_pathfinder_elbo")
    elbo_spread = pathfinder_diagnostics.get("pathfinder_elbo_spread")
    logger.info(
        "%s: scipy_pathfinder complete in %.1fs "
        "(setup=%.1fs, jax_compile=%.1fs, runtime=%.1fs, best_elbo=%s, "
        "elbo_spread=%s, n_starts_finite=%s)",
        phase_label,
        _phase_elapsed(started_at),
        float(pathfinder_diagnostics.get("pathfinder_setup_seconds", 0.0)),
        float(pathfinder_diagnostics.get("pathfinder_jax_compile_seconds", 0.0)),
        float(pathfinder_diagnostics.get("pathfinder_runtime_seconds", 0.0)),
        f"{best_elbo:.2f}" if isinstance(best_elbo, (int, float)) else "n/a",
        f"{elbo_spread:.2f}" if isinstance(elbo_spread, (int, float)) else "n/a",
        pathfinder_diagnostics.get("n_pathfinder_starts_finite", "n/a"),
    )


def prepare_parameter_warmup(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    bundle: CachedParticleRuntimeBundle,
    method_label: str,
    phase_label: str,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    sample_key: jnp.ndarray,
    reparam,
    seed: int,
    n_ieks_iters: int,
    num_chains: int,
    init_method: str,
    initial_positions_override: jnp.ndarray | None,
    init_scale: float,
    parameter_preconditioner_chol: jnp.ndarray | None,
    auto_preconditioner_method: str,
    auto_preconditioner_maxiter: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    pathfinder_parallel_workers: int | None,
    pathfinder_init_scale: float | None,
    prior_released_sites: tuple[str, ...] = DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    prior_release_scale: float = 0.05,
    release_jitter_key: jnp.ndarray | None = None,
) -> ParameterWarmupResult:
    """Resolve parameter init positions and the parameter-kernel preconditioner.

    The policy is intentionally centralized: if Pathfinder is needed for both
    initialization and preconditioning, this function runs the fit once and hands
    the same best-ELBO Gaussian to both consumers.
    """
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            f"Unsupported {method_label} init_method {init_method!r}. "
            "Supported: 'random' or 'pathfinder'."
        )
    if auto_preconditioner_method not in {"map", "none", "pathfinder"}:
        raise ValueError(
            f"Unsupported auto_preconditioner_method {auto_preconditioner_method!r}. "
            "Supported: 'map', 'none', or 'pathfinder'."
        )

    total_t0 = time.monotonic()
    dim = int(bundle.flat_example.shape[0])
    dtype = bundle.flat_example.dtype
    pathfinder_state: ScipyPathfinderResult | None = None
    pathfinder_diagnostics: UncheckedJsonObject | None = None
    init_positions: jnp.ndarray | None = None
    init_diagnostics: UncheckedJsonObject
    preconditioner_chol = parameter_preconditioner_chol

    pathfinder_consumers: list[str] = []
    if initial_positions_override is None and init_method == "pathfinder":
        pathfinder_consumers.append("init")
    if parameter_preconditioner_chol is None and auto_preconditioner_method == "pathfinder":
        pathfinder_consumers.append("preconditioner")

    if pathfinder_consumers:
        pathfinder_t0 = time.monotonic()
        logger.info(
            "%s: running scipy_pathfinder for %s (n_starts=%d, maxiter=%d, "
            "n_ieks_iters=%d, elbo_samples=%d, parallel_workers=%s)...",
            phase_label,
            "+".join(pathfinder_consumers),
            n_pathfinder_starts,
            pathfinder_maxiter,
            n_ieks_iters,
            pathfinder_num_elbo_samples,
            pathfinder_parallel_workers if pathfinder_parallel_workers is not None else "starts",
        )
        pathfinder_state, pathfinder_diagnostics = run_scipy_pathfinder_approximation(
            model,
            observations,
            times,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
            num_elbo_samples=pathfinder_num_elbo_samples,
            maxiter=pathfinder_maxiter,
            n_pathfinder_starts=n_pathfinder_starts,
            pathfinder_parallel_workers=pathfinder_parallel_workers,
        )
        _log_pathfinder_completion(
            phase_label=phase_label,
            started_at=pathfinder_t0,
            pathfinder_diagnostics=pathfinder_diagnostics,
        )
    else:
        logger.info("%s: scipy_pathfinder skipped", phase_label)

    init_t0 = time.monotonic()
    if initial_positions_override is not None:
        init_positions = _validate_initial_positions_override(
            initial_positions_override,
            num_chains=num_chains,
            dim=dim,
            dtype=dtype,
        )
        init_diagnostics = {"init_method": "user_provided"}
        init_source = "user_provided"
    elif init_method == "pathfinder":
        if pathfinder_state is None or pathfinder_diagnostics is None:
            raise RuntimeError("Pathfinder state missing for Pathfinder initialization.")
        init_positions, init_diagnostics = sample_scipy_pathfinder_init_positions(
            pathfinder_state,
            pathfinder_diagnostics,
            sample_key=sample_key,
            num_chains=num_chains,
            dtype=dtype,
            pathfinder_init_scale=pathfinder_init_scale,
            init_bundle=bundle,
            prior_released_sites=prior_released_sites,
            prior_release_scale=prior_release_scale,
            release_jitter_key=release_jitter_key,
            method_label=method_label,
        )
        init_source = "pathfinder"
    else:
        init_diagnostics = {"init_method": "random"}
        init_source = "random"
    logger.info(
        "%s: parameter init source=%s ready in %.1fs",
        phase_label,
        init_source,
        _phase_elapsed(init_t0),
    )

    preconditioner_t0 = time.monotonic()
    if parameter_preconditioner_chol is None:
        if auto_preconditioner_method == "pathfinder":
            if pathfinder_state is None or pathfinder_diagnostics is None:
                raise RuntimeError("Pathfinder state missing for Pathfinder preconditioner.")
            preconditioner_chol = jax.device_put(
                scipy_pathfinder_preconditioner_chol(pathfinder_state)
            )
            preconditioner_diagnostics = _pathfinder_preconditioner_diagnostics(
                pathfinder_diagnostics
            )
            preconditioner_source = "pathfinder"
        elif auto_preconditioner_method == "map":
            from nof1_causal_lab.models.ssm.inference.warmup.map import fit_map

            map_result = fit_map(
                model,
                observations,
                times,
                num_samples=1,
                seed=seed,
                n_ieks_iters=n_ieks_iters,
                maxiter=auto_preconditioner_maxiter,
                parameter_covariance_method="optimizer_hess_inv",
                reparam=reparam,
            )
            preconditioner_chol = _laplace_preconditioner_chol_from_map_result(map_result)
            preconditioner_diagnostics = {
                "auto_preconditioner": True,
                "auto_preconditioner_method": "map",
                "auto_preconditioner_maxiter": int(auto_preconditioner_maxiter),
            }
            preconditioner_source = "map"
        else:
            preconditioner_diagnostics = {
                "auto_preconditioner": False,
                "auto_preconditioner_method": "none",
            }
            preconditioner_source = "none"
    else:
        preconditioner_diagnostics = {"auto_preconditioner": False}
        preconditioner_source = "caller_provided"
    logger.info(
        "%s: parameter preconditioner source=%s ready in %.1fs",
        phase_label,
        preconditioner_source,
        _phase_elapsed(preconditioner_t0),
    )

    warmup_diagnostics = {
        "pathfinder_ran": bool(pathfinder_consumers),
        "pathfinder_run_count": 1 if pathfinder_consumers else 0,
        "pathfinder_consumers": pathfinder_consumers,
        "init_source": init_source,
        "preconditioner_source": preconditioner_source,
        "auto_preconditioner_method": auto_preconditioner_method,
        "dim": dim,
        "duration_seconds": _phase_elapsed(total_t0),
        "init_scale": float(init_scale),
        "pathfinder_init_scale": pathfinder_init_scale,
    }
    if pathfinder_diagnostics is not None:
        warmup_diagnostics.update(
            {
                "pathfinder_setup_seconds": pathfinder_diagnostics.get("pathfinder_setup_seconds"),
                "pathfinder_jax_compile_seconds": pathfinder_diagnostics.get(
                    "pathfinder_jax_compile_seconds"
                ),
                "pathfinder_runtime_seconds": pathfinder_diagnostics.get(
                    "pathfinder_runtime_seconds"
                ),
                "pathfinder_total_seconds": pathfinder_diagnostics.get("pathfinder_total_seconds"),
                "pathfinder_jax_compile_batch_sizes": pathfinder_diagnostics.get(
                    "pathfinder_jax_compile_batch_sizes"
                ),
            }
        )
    logger.info(
        "%s: parameter warmup complete in %.1fs (pathfinder_runs=%d, init=%s, preconditioner=%s)",
        phase_label,
        warmup_diagnostics["duration_seconds"],
        warmup_diagnostics["pathfinder_run_count"],
        init_source,
        preconditioner_source,
    )

    return ParameterWarmupResult(
        init_positions=init_positions,
        init_diagnostics=init_diagnostics,
        preconditioner_chol=preconditioner_chol,
        preconditioner_diagnostics=preconditioner_diagnostics,
        warmup_diagnostics=warmup_diagnostics,
        pathfinder_state=pathfinder_state,
        pathfinder_diagnostics=pathfinder_diagnostics,
    )
