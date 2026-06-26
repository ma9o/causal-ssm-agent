"""Prod-like particle MCMC inference check on the synthetic nonlinear fixture.

This script intentionally reuses ``scripts/benchmarks/synthetic_nonlinear.py`` so the
stress model stays identical to the benchmarking notebooks: three latent states,
mixed Gaussian/count/gamma observations, transition inputs, Hill saturation, and
multiplicative nonlinear drift terms.

It runs the production particle-MCMC ``fit(...)`` paths for the requested MPG
latent smoothers and optional PMMH baseline, then writes a JSON trace artifact
with per-sample acceptance, movement, label probabilities, derived label ESS,
complete log posterior history, particle ESS, backward-selection entropy,
reference-path hit rates, latent-path ESS approximations, aMALA proposal
diagnostics, and PMMH particle-filter diagnostics.
By default, one Pathfinder warmup is cached per support mode and passed into
every smoother fit as shared initial positions plus a shared parameter
preconditioner.

Example:
    uv run python scripts/benchmarks/benchmark.py
    uv run python scripts/benchmarks/benchmark.py --support-mode both --T 24

The default point-support mode is the nonlinear smoother smoke check. Interval
mode is an explicit caveat probe for the same fixture and currently exercises
the backend's affine-dynamics restriction before the smoother is reached.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "apps/data-pipeline/pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
for import_path in (
    REPO_ROOT / "apps/data-pipeline/src",
    REPO_ROOT / "apps/data-pipeline",
):
    path = str(import_path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _configure_jax_cache() -> None:
    cache_dir = REPO_ROOT / "scratchpad" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"


_configure_jax_cache()

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from evaluation.fixtures.synthetic_nonlinear import (
    TRUE_HILL_BY_SITE,
    TRUE_MULTIPLICATIVE_BY_SITE,
    build_synthetic_nonlinear_model,
    simulate_synthetic_nonlinear_data,
)
from evaluation.recovery.extraction import (
    autocorrelation_ess_1d,
    lag1_autocorrelation_1d,
    parameter_recovery,
    scalar_posterior_ess,
)

from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.bundle import (
    build_particle_runtime_bundle,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    MPGibbsDiagnosticMetric,
)
from nof1_causal_lab.models.ssm.inference.warmup.latent_init import (
    compute_ieks_latent_paths,
)
from nof1_causal_lab.models.ssm.inference.warmup.parameter_warmup import (
    DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    prepare_parameter_warmup,
)
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
    LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN,
)

logger = logging.getLogger(__name__)

EXPECTED_SMOOTHER_SELECTION = {
    "plain": "blocked_backward_sampling",
    # amala/amala_plus/amala_exact are dsmc leaf proposals; the dSMC tree's selection is
    # tree_stitch_combination (the old backward-sampling names predate the de-seq tree).
    "amala": "tree_stitch_combination",
    "amala_plus": "tree_stitch_combination",
    "amala_exact": "tree_stitch_combination",
    "dsmc": "tree_stitch_combination",
}
PMMH_BENCHMARK_METHOD = "pmmh"
AMALA_FAMILY_SMOOTHERS = frozenset({"amala", "amala_plus", "amala_exact"})
LATENT_PATH_MIXING_METRIC = "latent_path_mixing"
SCRIPT_DIAGNOSTIC_METRIC_VALUES = (
    *MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    LATENT_PATH_MIXING_METRIC,
)
PATHFINDER_CACHE_VERSION = 2
PATHFINDER_N_IEKS_ITERS = 6
PATHFINDER_PARAMETER_INIT_SCALE = 0.05


@dataclass(frozen=True)
class _PathfinderCache:
    path: Path
    source: str
    initial_positions: jnp.ndarray
    parameter_preconditioner_chol: jnp.ndarray
    # (num_chains, T, n_latent) IEKS smoothed paths at initial_positions: the
    # data-conditioned reference-path init (predictive simulation can diverge
    # at data-informed positions of nonlinear vector fields).
    init_latent_paths: jnp.ndarray
    diagnostics: dict[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class _FixtureContext:
    support_mode: str
    data: Any
    model: Any
    setup_seconds: float
    pathfinder_cache: _PathfinderCache | None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(jax.device_get(value)).tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _pathfinder_seed(args: argparse.Namespace, support_idx: int) -> int:
    base_seed = args.pathfinder_seed if args.pathfinder_seed is not None else args.seed
    return int(base_seed + 10_000 + support_idx)


def _pathfinder_cache_config(
    args: argparse.Namespace,
    *,
    support_mode: str,
    support_idx: int,
) -> dict[str, Any]:
    return {
        "version": PATHFINDER_CACHE_VERSION,
        "target": "synthetic_nonlinear",
        "support_mode": support_mode,
        "support_index": int(support_idx),
        "T": int(args.T),
        "data_seed": int(args.data_seed),
        "diffusion_scale": float(args.diffusion_scale),
        "num_chains": int(args.num_chains),
        "pathfinder_seed": _pathfinder_seed(args, support_idx),
        "n_ieks_iters": PATHFINDER_N_IEKS_ITERS,
        "pathfinder_num_elbo_samples": int(args.pathfinder_num_elbo_samples),
        "pathfinder_maxiter": int(args.pathfinder_maxiter),
        "n_pathfinder_starts": int(args.n_pathfinder_starts),
        "pathfinder_parallel_workers": args.pathfinder_parallel_workers,
        "pathfinder_init_scale": args.pathfinder_init_scale,
        "pathfinder_parameter_init_scale": PATHFINDER_PARAMETER_INIT_SCALE,
        "prior_released_sites": list(DEFAULT_PRIOR_RELEASED_SITE_NAMES),
        "reparam": None,
    }


def _pathfinder_cache_path(
    args: argparse.Namespace,
    *,
    support_mode: str,
    support_mode_count: int,
) -> Path:
    base_path = args.pathfinder_cache_path
    if support_mode_count == 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}.{support_mode}{base_path.suffix}")


def _load_pathfinder_cache(path: Path, *, expected_config: dict[str, Any]) -> _PathfinderCache:
    started = time.perf_counter()
    logger.info(
        "pathfinder cache load start: support_mode=%s path=%s",
        expected_config["support_mode"],
        path,
    )
    with np.load(path, allow_pickle=False) as payload:
        cache_config = json.loads(str(payload["config_json"].item()))
        if cache_config != expected_config:
            raise ValueError(
                "Pathfinder cache config does not match this run. "
                f"Use --pathfinder-cache-mode refresh to overwrite {path}."
            )
        diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        initial_positions = jnp.asarray(payload["initial_positions"])
        preconditioner_chol = jnp.asarray(payload["parameter_preconditioner_chol"])
        init_latent_paths = jnp.asarray(payload["init_latent_paths"])
    cache = _PathfinderCache(
        path=path,
        source="disk",
        initial_positions=initial_positions,
        parameter_preconditioner_chol=preconditioner_chol,
        init_latent_paths=init_latent_paths,
        diagnostics=diagnostics,
        elapsed_seconds=time.perf_counter() - started,
    )
    logger.info(
        "pathfinder cache load complete: support_mode=%s elapsed=%.2fs init_shape=%s preconditioner_shape=%s",
        expected_config["support_mode"],
        cache.elapsed_seconds,
        tuple(cache.initial_positions.shape),
        tuple(cache.parameter_preconditioner_chol.shape),
    )
    return cache


def _write_pathfinder_cache(
    cache: _PathfinderCache,
    *,
    config: dict[str, Any],
) -> None:
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache.path,
        initial_positions=np.asarray(jax.device_get(cache.initial_positions)),
        parameter_preconditioner_chol=np.asarray(
            jax.device_get(cache.parameter_preconditioner_chol)
        ),
        init_latent_paths=np.asarray(jax.device_get(cache.init_latent_paths)),
        config_json=np.asarray(json.dumps(_json_ready(config), sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(_json_ready(cache.diagnostics), sort_keys=True)),
    )


def _run_pathfinder_cache(
    args: argparse.Namespace,
    *,
    data: Any,
    model: Any,
    support_mode: str,
    support_idx: int,
    cache_path: Path,
) -> _PathfinderCache:
    started = time.perf_counter()
    logger.info(
        "pathfinder cache refresh start: support_mode=%s T=%d starts=%d maxiter=%d elbo_samples=%d",
        support_mode,
        int(args.T),
        int(args.n_pathfinder_starts),
        int(args.pathfinder_maxiter),
        int(args.pathfinder_num_elbo_samples),
    )
    base_key = random.PRNGKey(_pathfinder_seed(args, support_idx))
    trace_key, pathfinder_key, sample_key = random.split(base_key, 3)
    # Shared Pathfinder init bundle: use the EM-native Euler scheme unless only the
    # local-linear smoothers (plain / prior_predictive) are requested (mirrors fit.py's
    # per-method rule). The scheme only shapes the init objective, not the per-smoother
    # fits, which re-derive their own scheme via fit().
    requested = [s.strip() for s in args.smoothers.split(",")]
    cache_scheme = (
        LATENT_TRANSITION_EULER_MARUYAMA
        if any(s in ("amala", "amala_plus", "amala_exact") for s in requested)
        else LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN
    )
    bundle = build_particle_runtime_bundle(
        model,
        data.observations,
        data.times,
        scheme=cache_scheme,
        trace_key=trace_key,
        reparam=None,
    )
    warmup_result = prepare_parameter_warmup(
        model,
        data.observations,
        data.times,
        bundle=bundle,
        method_label="mpg_smoothers_synthetic_nonlinear_check",
        phase_label=f"shared pathfinder cache ({support_mode})",
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=sample_key,
        reparam=None,
        seed=_pathfinder_seed(args, support_idx),
        n_ieks_iters=PATHFINDER_N_IEKS_ITERS,
        num_chains=args.num_chains,
        init_method="pathfinder",
        initial_positions_override=None,
        init_scale=PATHFINDER_PARAMETER_INIT_SCALE,
        parameter_preconditioner_chol=None,
        auto_preconditioner_method="pathfinder",
        auto_preconditioner_maxiter=0,
        pathfinder_num_elbo_samples=args.pathfinder_num_elbo_samples,
        pathfinder_maxiter=args.pathfinder_maxiter,
        n_pathfinder_starts=args.n_pathfinder_starts,
        pathfinder_parallel_workers=args.pathfinder_parallel_workers,
        pathfinder_init_scale=args.pathfinder_init_scale,
        prior_released_sites=DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    )
    if warmup_result.init_positions is None:
        raise RuntimeError("Shared Pathfinder cache did not return initial positions.")
    if warmup_result.preconditioner_chol is None:
        raise RuntimeError("Shared Pathfinder cache did not return a preconditioner.")

    init_latent_paths = compute_ieks_latent_paths(
        model,
        data.observations,
        data.times,
        positions=warmup_result.init_positions,
        trace_key=trace_key,
        reparam=None,
        n_ieks_iters=PATHFINDER_N_IEKS_ITERS,
    )

    diagnostics = {
        "cache_config": _pathfinder_cache_config(
            args,
            support_mode=support_mode,
            support_idx=support_idx,
        ),
        "parameter_warmup": warmup_result.warmup_diagnostics,
        "init": warmup_result.init_diagnostics,
        "preconditioner": warmup_result.preconditioner_diagnostics,
        "pathfinder": warmup_result.pathfinder_diagnostics,
    }
    cache = _PathfinderCache(
        path=cache_path,
        source="computed",
        initial_positions=jnp.asarray(warmup_result.init_positions),
        parameter_preconditioner_chol=jnp.asarray(warmup_result.preconditioner_chol),
        init_latent_paths=init_latent_paths,
        diagnostics=diagnostics,
        elapsed_seconds=time.perf_counter() - started,
    )
    logger.info(
        "pathfinder cache refresh complete: support_mode=%s elapsed=%.2fs init_shape=%s preconditioner_shape=%s",
        support_mode,
        cache.elapsed_seconds,
        tuple(cache.initial_positions.shape),
        tuple(cache.parameter_preconditioner_chol.shape),
    )
    return cache


def _resolve_pathfinder_cache(
    args: argparse.Namespace,
    *,
    data: Any,
    model: Any,
    support_mode: str,
    support_idx: int,
    support_mode_count: int,
) -> _PathfinderCache | None:
    if args.pathfinder_cache_mode == "off":
        return None

    cache_path = _pathfinder_cache_path(
        args,
        support_mode=support_mode,
        support_mode_count=support_mode_count,
    )
    expected_config = _pathfinder_cache_config(
        args,
        support_mode=support_mode,
        support_idx=support_idx,
    )
    if args.pathfinder_cache_mode == "reuse" and cache_path.exists():
        print(f"loading Pathfinder cache for {support_mode} from {cache_path}", flush=True)
        return _load_pathfinder_cache(cache_path, expected_config=expected_config)

    print(f"running shared Pathfinder cache for {support_mode}...", flush=True)
    cache = _run_pathfinder_cache(
        args,
        data=data,
        model=model,
        support_mode=support_mode,
        support_idx=support_idx,
        cache_path=cache_path,
    )
    _write_pathfinder_cache(cache, config=expected_config)
    logger.info(
        "pathfinder cache persisted: support_mode=%s path=%s",
        support_mode,
        cache_path,
    )
    print(f"wrote Pathfinder cache for {support_mode} to {cache_path}", flush=True)
    return cache


def _prepare_fixture_context(
    args: argparse.Namespace,
    *,
    support_mode: str,
    support_idx: int,
    support_mode_count: int,
) -> _FixtureContext:
    include_interval_support = support_mode == "interval"
    setup_started = time.perf_counter()
    logger.info(
        "fixture setup start: support_mode=%s T=%d data_seed=%d",
        support_mode,
        int(args.T),
        int(args.data_seed),
    )
    data = simulate_synthetic_nonlinear_data(
        T=args.T, seed=args.data_seed, diffusion_scale=args.diffusion_scale
    )
    model = build_synthetic_nonlinear_model(
        data,
        include_interval_support=include_interval_support,
        diffusion_scale=args.diffusion_scale,
    )
    setup_seconds = time.perf_counter() - setup_started
    logger.info(
        "fixture setup complete: support_mode=%s elapsed=%.2fs observations_shape=%s",
        support_mode,
        setup_seconds,
        tuple(data.observations.shape),
    )
    pathfinder_cache = _resolve_pathfinder_cache(
        args,
        data=data,
        model=model,
        support_mode=support_mode,
        support_idx=support_idx,
        support_mode_count=support_mode_count,
    )
    return _FixtureContext(
        support_mode=support_mode,
        data=data,
        model=model,
        setup_seconds=setup_seconds,
        pathfinder_cache=pathfinder_cache,
    )


def _finite_summary(name: str, values: Any) -> dict[str, Any]:
    array = np.asarray(jax.device_get(values))
    finite = np.isfinite(array)
    if not bool(np.all(finite)):
        bad_count = int(array.size - int(np.count_nonzero(finite)))
        raise AssertionError(f"{name} contains {bad_count} non-finite values")
    if array.size == 0:
        return {"shape": list(array.shape), "mean": None, "std": None}
    return {
        "shape": list(array.shape),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _block_summaries(values: Any, *, block_size: int) -> dict[str, Any]:
    array = np.asarray(jax.device_get(values), dtype=np.float64)
    if array.ndim < 3:
        raise AssertionError(f"expected chains x samples x time values, got {array.shape}")
    time_count = int(array.shape[-1])
    rows = []
    for block_start in range(0, time_count, block_size):
        block_end = min(block_start + block_size, time_count)
        block_values = array[..., block_start:block_end]
        rows.append(
            {
                "block_index": block_start // block_size,
                "time_start": block_start,
                "time_end_exclusive": block_end,
                "min": float(np.min(block_values)),
                "mean": float(np.mean(block_values)),
                "max": float(np.max(block_values)),
            }
        )
    if not rows:
        return {"block_count": 0, "blocks": []}
    worst = min(rows, key=lambda row: row["min"])
    return {
        "block_count": len(rows),
        "worst_block_by_min": worst["block_index"],
        "blocks": rows,
    }


def _particle_smoother_health(
    extra_fields: dict[str, Any],
    *,
    block_size: int,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    health: dict[str, Any] = {
        "enabled": bool(
            selected_metrics
            & {
                MPGibbsDiagnosticMetric.PARTICLE_FILTER.value,
                MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value,
                MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value,
            }
        )
    }
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "forward_particle_ess_by_t" in extra_fields
    ):
        health["forward_particle_ess"] = {
            "summary": _finite_summary(
                "forward_particle_ess_by_t",
                extra_fields["forward_particle_ess_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_particle_ess_by_t"],
                block_size=block_size,
            ),
        }
        health["forward_log_weight_range"] = {
            "summary": _finite_summary(
                "forward_log_weight_range_by_t",
                extra_fields["forward_log_weight_range_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_log_weight_range_by_t"],
                block_size=block_size,
            ),
        }
        health["forward_log_weight_variance"] = {
            "summary": _finite_summary(
                "forward_log_weight_variance_by_t",
                extra_fields["forward_log_weight_variance_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_log_weight_variance_by_t"],
                block_size=block_size,
            ),
        }
    if (
        MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics
        and "backward_selection_ess_by_t" in extra_fields
    ):
        health["backward_selection_ess"] = {
            "summary": _finite_summary(
                "backward_selection_ess_by_t",
                extra_fields["backward_selection_ess_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["backward_selection_ess_by_t"],
                block_size=block_size,
            ),
        }
        health["backward_selection_entropy"] = {
            "summary": _finite_summary(
                "backward_selection_entropy_by_t",
                extra_fields["backward_selection_entropy_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["backward_selection_entropy_by_t"],
                block_size=block_size,
            ),
        }
        health["backward_selection_max_prob"] = _finite_summary(
            "backward_selection_max_prob_by_t",
            extra_fields["backward_selection_max_prob_by_t"],
        )
    if (
        MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics
        and "selected_particle_per_t" in extra_fields
    ):
        selected_by_t = np.asarray(
            jax.device_get(extra_fields["selected_particle_per_t"]),
            dtype=np.float64,
        )
        reference_hit_by_t = np.mean(selected_by_t == 0.0, axis=(0, 1))
        health["reference_path_hit_rate"] = _finite_summary(
            "reference_path_hit_rate",
            extra_fields["reference_path_hit_rate"],
        )
        health["reference_path_hit_rate_by_t"] = reference_hit_by_t.tolist()
        health["selected_particle_unique_count"] = _finite_summary(
            "selected_particle_unique_count",
            extra_fields["selected_particle_unique_count"],
        )
    return health


def _latent_path_mixing(
    result,
    *,
    max_sites: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "note": f"Enable {LATENT_PATH_MIXING_METRIC!r} to retain latent paths and compute latent ESS.",
        }
    latent_paths = result.diagnostics.get("latent_paths")
    scope = "post_warmup retained public latent paths at first/middle/final times"
    if latent_paths is not None and np.asarray(jax.device_get(latent_paths)).shape[1] == 0:
        latent_paths = result.diagnostics.get("warmup_latent_paths")
        scope = "warmup retained public latent paths at first/middle/final times"
    if latent_paths is None:
        return {
            "enabled": False,
            "note": "Run with retained latent paths to compute latent-path ESS.",
        }
    return _latent_path_mixing_from_paths(
        latent_paths,
        max_sites=max_sites,
        scope=scope,
    )


def _latent_path_mixing_from_paths(
    latent_paths: Any,
    *,
    max_sites: int,
    scope: str,
) -> dict[str, Any]:
    paths = np.asarray(jax.device_get(latent_paths), dtype=np.float64)
    if paths.ndim != 4:
        raise AssertionError(
            f"expected latent paths as chains x samples x time x dim, got {paths.shape}"
        )
    if paths.shape[1] == 0:
        return {
            "enabled": True,
            "scope": scope,
            "site_count": 0,
            "ess_approx_min": None,
            "ess_approx_median": None,
            "ess_approx_max": None,
            "sites": {},
        }
    _chains, _samples, time_count, latent_dim = paths.shape
    candidate_times = tuple(dict.fromkeys((0, time_count // 2, time_count - 1)))
    site_rows: dict[str, Any] = {}
    for time_idx in candidate_times:
        for dim_idx in range(latent_dim):
            if len(site_rows) >= max_sites:
                break
            site_key = f"t{time_idx}_dim{dim_idx}"
            chain_draws = paths[:, :, time_idx, dim_idx]
            ess_by_chain = [autocorrelation_ess_1d(draws) for draws in chain_draws]
            lag1_by_chain = [lag1_autocorrelation_1d(draws) for draws in chain_draws]
            finite_lag1 = [value for value in lag1_by_chain if value is not None]
            site_rows[site_key] = {
                "time_index": time_idx,
                "latent_dim": dim_idx,
                "ess_approx_min_chain": float(np.min(ess_by_chain)),
                "ess_approx_mean_chain": float(np.mean(ess_by_chain)),
                "lag1_autocorr_mean_chain": (
                    None if not finite_lag1 else float(np.mean(finite_lag1))
                ),
                "mean": float(np.mean(chain_draws)),
                "std": float(np.std(chain_draws)),
            }
        if len(site_rows) >= max_sites:
            break
    ess_values = np.asarray(
        [row["ess_approx_min_chain"] for row in site_rows.values()],
        dtype=np.float64,
    )
    return {
        "enabled": True,
        "scope": scope,
        "site_count": len(site_rows),
        "ess_approx_min": None if ess_values.size == 0 else float(np.min(ess_values)),
        "ess_approx_median": None if ess_values.size == 0 else float(np.median(ess_values)),
        "ess_approx_max": None if ess_values.size == 0 else float(np.max(ess_values)),
        "sites": site_rows,
    }


def _label_probability_diagnostics(label_log_probs: jnp.ndarray) -> dict[str, Any]:
    label_probs = jax.nn.softmax(label_log_probs, axis=-1)
    label_ess = 1.0 / jnp.sum(label_probs * label_probs, axis=-1)
    label_entropy = -jnp.sum(
        jnp.where(label_probs > 0.0, label_probs * jnp.log(label_probs), 0.0),
        axis=-1,
    )
    label_max_prob = jnp.max(label_probs, axis=-1)
    return {
        "final_label_probs": label_probs,
        "final_label_ess": label_ess,
        "final_label_entropy": label_entropy,
        "final_label_max_prob": label_max_prob,
    }


def _per_step_trace(
    *,
    extra_fields: dict[str, Any],
    complete_log_posterior: Any,
    smoother: str,
    selected_metrics: frozenset[str],
    scope: str,
) -> dict[str, Any]:
    complete_lp = jnp.asarray(complete_log_posterior)
    final_label_log_probs = jnp.asarray(extra_fields["final_label_log_probs"])
    label_diagnostics = _label_probability_diagnostics(final_label_log_probs)
    fields = {
        "parameter_accept_prob": extra_fields["parameter_accept_prob"],
        "latent_accept_prob": extra_fields["latent_accept_prob"],
        "selected_parameter_label": extra_fields["selected_parameter_label"],
        "selected_particle": extra_fields["selected_particle"],
        "latent_move_rms": extra_fields["latent_move_rms"],
        "latent_move_max_abs": extra_fields["latent_move_max_abs"],
        "latent_move_rms_per_t": extra_fields["latent_move_rms_per_t"],
        "complete_log_posterior": complete_lp,
        "final_label_log_probs": final_label_log_probs,
        **label_diagnostics,
    }
    if (
        MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics
        and "selected_particle_per_t" in extra_fields
    ):
        fields["selected_particle_per_t"] = extra_fields["selected_particle_per_t"]
        fields["reference_path_hit_rate"] = extra_fields["reference_path_hit_rate"]
        fields["selected_particle_unique_count"] = extra_fields["selected_particle_unique_count"]
    if (
        MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics
        and "parameter_jump_rms" in extra_fields
    ):
        fields["parameter_jump_rms"] = extra_fields["parameter_jump_rms"]
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "forward_particle_ess_by_t" in extra_fields
    ):
        fields["forward_particle_ess_by_t"] = extra_fields["forward_particle_ess_by_t"]
        fields["forward_log_weight_range_by_t"] = extra_fields["forward_log_weight_range_by_t"]
        fields["forward_log_weight_variance_by_t"] = extra_fields[
            "forward_log_weight_variance_by_t"
        ]
    if (
        MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics
        and "backward_selection_ess_by_t" in extra_fields
    ):
        fields["backward_selection_ess_by_t"] = extra_fields["backward_selection_ess_by_t"]
        fields["backward_selection_entropy_by_t"] = extra_fields["backward_selection_entropy_by_t"]
        fields["backward_selection_max_prob_by_t"] = extra_fields[
            "backward_selection_max_prob_by_t"
        ]
    if smoother in AMALA_FAMILY_SMOOTHERS:
        fields["amala_grad_norm_mean"] = extra_fields["amala_grad_norm_mean"]
        fields["amala_grad_norm_max"] = extra_fields["amala_grad_norm_max"]

    summaries = {
        name: _finite_summary(f"per_step[{name}]", values) for name, values in fields.items()
    }
    return {
        "scope": scope,
        "shape_convention": "chains x samples for scalar fields; chains x samples x ... otherwise",
        "fields": {name: _json_ready(values) for name, values in fields.items()},
        "summaries": summaries,
    }


def _check_label_log_probs(extra_fields: dict[str, Any]) -> dict[str, Any]:
    label_log_probs = jnp.asarray(extra_fields["final_label_log_probs"])
    log_norm = jax.scipy.special.logsumexp(label_log_probs, axis=-1)
    max_abs_error = None
    if log_norm.size:
        max_abs_error = float(jnp.max(jnp.abs(log_norm)))
        tolerance = 2e-2 if label_log_probs.dtype == jnp.float32 else 5e-5
        if not np.isfinite(max_abs_error) or max_abs_error > tolerance:
            raise AssertionError(
                f"final_label_log_probs are not normalized; max |logsumexp|={max_abs_error:.3g}"
            )
    selected_label = jnp.asarray(extra_fields["selected_parameter_label"])
    return {
        "final_label_log_probs": _finite_summary(
            "final_label_log_probs",
            label_log_probs,
        ),
        "final_label_log_probs_max_abs_log_norm_error": max_abs_error,
        "selected_parameter_label": _finite_summary(
            "selected_parameter_label",
            selected_label,
        ),
    }


def _phase_step_count(extra_fields: dict[str, Any]) -> int:
    if not extra_fields:
        return 0
    first_value = next(iter(extra_fields.values()))
    return int(np.asarray(jax.device_get(first_value)).shape[1])


def _phase_extra_fields(result) -> dict[str, dict[str, Any]]:
    phase_fields = result.diagnostics.get("marginal_particle_gibbs_phase_extra_fields")
    if isinstance(phase_fields, dict):
        return phase_fields
    phase_fields = result.diagnostics.get("particle_marginal_mh_phase_extra_fields")
    if isinstance(phase_fields, dict):
        return phase_fields
    return {
        "post_warmup": result.diagnostics["mcmc"].get_extra_fields(group_by_chain=True),
    }


def _phase_complete_log_posterior(result) -> dict[str, Any]:
    if "chain_estimated_log_posterior_history" in result.diagnostics:
        return {
            "warmup": result.diagnostics.get("warmup_estimated_log_posterior_history"),
            "post_warmup": result.diagnostics.get("chain_estimated_log_posterior_history"),
            "all": result.diagnostics.get("all_estimated_log_posterior_history"),
        }
    return {
        "warmup": result.diagnostics.get("warmup_complete_log_posterior_history"),
        "post_warmup": result.diagnostics.get("chain_complete_log_posterior_history"),
        "all": result.diagnostics.get("all_complete_log_posterior_history"),
    }


def _primary_diagnostic_phase(phase_fields: dict[str, dict[str, Any]]) -> str:
    if _phase_step_count(phase_fields.get("post_warmup", {})) > 0:
        return "post_warmup"
    if _phase_step_count(phase_fields.get("warmup", {})) > 0:
        return "warmup"
    return "all"


def _check_result(
    result,
    *,
    smoother: str,
    elapsed_seconds: float,
    ess_max_sites: int,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    diagnostics = result.diagnostics["marginal_particle_gibbs"]
    expected_latent = "plain" if smoother == "plain" else "dsmc"
    if diagnostics["latent_smoother"] != expected_latent:
        raise AssertionError(
            f"expected latent_smoother {expected_latent!r} (benchmark smoother {smoother!r}), "
            f"got {diagnostics['latent_smoother']!r}"
        )
    expected_selection = EXPECTED_SMOOTHER_SELECTION[smoother]
    if diagnostics["latent_smoother_selection"] != expected_selection:
        raise AssertionError(
            "unexpected smoother selection for "
            f"{smoother!r}: {diagnostics['latent_smoother_selection']!r}"
        )

    sample_summaries = {
        name: _finite_summary(f"samples[{name}]", values)
        for name, values in result.get_samples().items()
    }
    phase_fields = _phase_extra_fields(result)
    phase_complete_lp = _phase_complete_log_posterior(result)
    primary_phase = _primary_diagnostic_phase(phase_fields)
    extra_fields = phase_fields[primary_phase]
    extra_summaries = {
        "parameter_accept_prob": _finite_summary(
            "parameter_accept_prob",
            extra_fields["parameter_accept_prob"],
        ),
        "latent_accept_prob": _finite_summary(
            "latent_accept_prob",
            extra_fields["latent_accept_prob"],
        ),
        "latent_move_rms": _finite_summary(
            "latent_move_rms",
            extra_fields["latent_move_rms"],
        ),
        **_check_label_log_probs(extra_fields),
    }
    if (
        MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics
        and "parameter_jump_rms" in extra_fields
    ):
        extra_summaries["parameter_jump_rms"] = _finite_summary(
            "parameter_jump_rms",
            extra_fields["parameter_jump_rms"],
        )
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "forward_particle_ess_by_t" in extra_fields
    ):
        extra_summaries["forward_particle_ess_by_t"] = _finite_summary(
            "forward_particle_ess_by_t",
            extra_fields["forward_particle_ess_by_t"],
        )
    if (
        MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics
        and "backward_selection_ess_by_t" in extra_fields
    ):
        extra_summaries["backward_selection_ess_by_t"] = _finite_summary(
            "backward_selection_ess_by_t",
            extra_fields["backward_selection_ess_by_t"],
        )
    if (
        MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics
        and "reference_path_hit_rate" in extra_fields
    ):
        extra_summaries["reference_path_hit_rate"] = _finite_summary(
            "reference_path_hit_rate",
            extra_fields["reference_path_hit_rate"],
        )
        extra_summaries["selected_particle_unique_count"] = _finite_summary(
            "selected_particle_unique_count",
            extra_fields["selected_particle_unique_count"],
        )
    if smoother in AMALA_FAMILY_SMOOTHERS:
        extra_summaries["amala_grad_norm_mean"] = _finite_summary(
            "amala_grad_norm_mean",
            extra_fields["amala_grad_norm_mean"],
        )
        extra_summaries["amala_grad_norm_max"] = _finite_summary(
            "amala_grad_norm_max",
            extra_fields["amala_grad_norm_max"],
        )
        if diagnostics["amala_grad_norm_max"] < diagnostics["amala_grad_norm_mean"]:
            raise AssertionError("amala_grad_norm_max is smaller than amala_grad_norm_mean")

    diagnostic_summary = {
        "latent_kernel": diagnostics["latent_kernel"],
        "latent_smoother": diagnostics["latent_smoother"],
        "latent_smoother_selection": diagnostics["latent_smoother_selection"],
        "parameter_kernel": diagnostics["parameter_kernel"],
        "parameter_preconditioned": diagnostics["parameter_preconditioned"],
        "parameter_warmup": diagnostics.get("parameter_warmup"),
        "init_method": diagnostics.get("init_method"),
        "auto_preconditioner": diagnostics.get("auto_preconditioner"),
        "parameter_accept_rate": diagnostics["parameter_accept_rate"],
        "latent_update_fraction": diagnostics["latent_update_fraction"],
        "latent_move_rms_mean": diagnostics.get("latent_move_rms_mean"),
        "mcmc_phase_seconds": diagnostics["mcmc_phase_seconds"],
        "amala_delta_adapted": diagnostics.get("amala_delta_adapted"),
        "amala_delta_init": diagnostics.get("amala_delta_init"),
        "amala_target_accept": diagnostics.get("amala_target_accept"),
    }
    if diagnostics.get("final_latent_delta") is not None:
        final_delta = np.asarray(diagnostics["final_latent_delta"], dtype=np.float64)
        diagnostic_summary.update(
            {
                "final_latent_delta_min": float(np.min(final_delta)),
                "final_latent_delta_median": float(np.median(final_delta)),
                "final_latent_delta_max": float(np.max(final_delta)),
            }
        )
    if (
        MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics
        and "parameter_jump_rms" in extra_fields
    ):
        diagnostic_summary["parameter_jump_rms_mean"] = diagnostics.get("parameter_jump_rms_mean")
    if (
        MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics
        and "reference_path_hit_rate" in extra_fields
    ):
        diagnostic_summary["reference_path_hit_rate_mean"] = diagnostics.get(
            "reference_path_hit_rate_mean"
        )
        diagnostic_summary["selected_particle_unique_count_mean"] = diagnostics.get(
            "selected_particle_unique_count_mean"
        )
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "forward_particle_ess_by_t" in extra_fields
    ):
        diagnostic_summary["forward_particle_ess_min"] = diagnostics.get("forward_particle_ess_min")
        diagnostic_summary["forward_particle_ess_mean"] = diagnostics.get(
            "forward_particle_ess_mean"
        )
        diagnostic_summary["forward_log_weight_range_max"] = diagnostics.get(
            "forward_log_weight_range_max"
        )
        diagnostic_summary["forward_log_weight_variance_mean"] = diagnostics.get(
            "forward_log_weight_variance_mean"
        )
    if (
        MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics
        and "backward_selection_ess_by_t" in extra_fields
    ):
        diagnostic_summary["backward_selection_ess_min"] = diagnostics.get(
            "backward_selection_ess_min"
        )
        diagnostic_summary["backward_selection_ess_mean"] = diagnostics.get(
            "backward_selection_ess_mean"
        )
        diagnostic_summary["backward_selection_entropy_mean"] = diagnostics.get(
            "backward_selection_entropy_mean"
        )
        diagnostic_summary["backward_selection_max_prob_mean"] = diagnostics.get(
            "backward_selection_max_prob_mean"
        )
    if smoother in AMALA_FAMILY_SMOOTHERS:
        diagnostic_summary.update(
            {
                "amala_grad_norm_mean": diagnostics.get("amala_grad_norm_mean"),
                "amala_grad_norm_max": diagnostics.get("amala_grad_norm_max"),
            }
        )

    return {
        "diagnostic_primary_phase": primary_phase,
        "diagnostics": diagnostic_summary,
        "per_step_trace": _per_step_trace(
            extra_fields=extra_fields,
            complete_log_posterior=phase_complete_lp[primary_phase],
            smoother=smoother,
            selected_metrics=selected_metrics,
            scope=primary_phase,
        ),
        "per_phase_trace": {
            phase: _per_step_trace(
                extra_fields=fields,
                complete_log_posterior=phase_complete_lp[phase],
                smoother=smoother,
                selected_metrics=selected_metrics,
                scope=phase,
            )
            for phase, fields in phase_fields.items()
            if phase in {"warmup", "post_warmup"} and phase_complete_lp.get(phase) is not None
        },
        "particle_smoother_health": _particle_smoother_health(
            extra_fields,
            block_size=int(diagnostics["latent_block_size"]),
            selected_metrics=selected_metrics,
        ),
        "particle_smoother_health_by_phase": {
            phase: _particle_smoother_health(
                fields,
                block_size=int(diagnostics["latent_block_size"]),
                selected_metrics=selected_metrics,
            )
            for phase, fields in phase_fields.items()
            if phase in {"warmup", "post_warmup"} and _phase_step_count(fields) > 0
        },
        "posterior_ess": scalar_posterior_ess(
            result,
            max_sites=ess_max_sites,
            elapsed_seconds=elapsed_seconds,
        ),
        "parameter_recovery": parameter_recovery(
            result,
            elapsed_seconds=elapsed_seconds,
        ),
        "latent_path_mixing": _latent_path_mixing(
            result,
            max_sites=ess_max_sites,
            enabled=LATENT_PATH_MIXING_METRIC in selected_metrics,
        ),
        "latent_path_mixing_by_phase": (
            {}
            if LATENT_PATH_MIXING_METRIC not in selected_metrics
            else {
                phase: _latent_path_mixing_from_paths(
                    paths,
                    max_sites=ess_max_sites,
                    scope=f"{phase} retained public latent paths at first/middle/final times",
                )
                for phase, paths in {
                    "warmup": result.diagnostics.get("warmup_latent_paths"),
                    "post_warmup": result.diagnostics.get("latent_paths"),
                }.items()
                if paths is not None
            }
        ),
        "sample_summaries": sample_summaries,
        "extra_summaries": extra_summaries,
    }


def _pmmh_per_step_trace(
    *,
    extra_fields: dict[str, Any],
    selected_metrics: frozenset[str],
    scope: str,
) -> dict[str, Any]:
    fields = {
        "parameter_accept_prob": extra_fields["parameter_accept_prob"],
        "estimated_log_likelihood": extra_fields["estimated_log_likelihood"],
        "log_prior": extra_fields["log_prior"],
        "estimated_log_posterior": extra_fields["estimated_log_posterior"],
        "log_alpha": extra_fields["log_alpha"],
        "proposed_estimated_log_likelihood": extra_fields["proposed_estimated_log_likelihood"],
        "pf_ess_min": extra_fields["pf_ess_min"],
        "pf_ess_mean": extra_fields["pf_ess_mean"],
        "pf_log_weight_range_max": extra_fields["pf_log_weight_range_max"],
        "pf_log_weight_variance_mean": extra_fields["pf_log_weight_variance_mean"],
        "pf_log_likelihood_increment_variance": extra_fields[
            "pf_log_likelihood_increment_variance"
        ],
    }
    if (
        MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics
        and "parameter_jump_rms" in extra_fields
    ):
        fields["parameter_jump_rms"] = extra_fields["parameter_jump_rms"]
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "pf_ess_by_t" in extra_fields
    ):
        fields["pf_ess_by_t"] = extra_fields["pf_ess_by_t"]
        fields["pf_log_weight_range_by_t"] = extra_fields["pf_log_weight_range_by_t"]
        fields["pf_log_weight_variance_by_t"] = extra_fields["pf_log_weight_variance_by_t"]
        fields["pf_log_likelihood_increment_by_t"] = extra_fields[
            "pf_log_likelihood_increment_by_t"
        ]

    summaries = {
        name: _finite_summary(f"pmmh_per_step[{name}]", values) for name, values in fields.items()
    }
    return {
        "scope": scope,
        "shape_convention": "chains x samples for scalar fields; chains x samples x ... otherwise",
        "fields": {name: _json_ready(values) for name, values in fields.items()},
        "summaries": summaries,
    }


def _check_pmmh_result(
    result,
    *,
    elapsed_seconds: float,
    ess_max_sites: int,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    diagnostics = result.diagnostics["particle_marginal_mh"]
    if result.method != "particle_marginal_mh":
        raise AssertionError(f"expected particle_marginal_mh, got {result.method!r}")

    sample_summaries = {
        name: _finite_summary(f"samples[{name}]", values)
        for name, values in result.get_samples().items()
    }
    phase_fields = _phase_extra_fields(result)
    primary_phase = _primary_diagnostic_phase(phase_fields)
    extra_fields = phase_fields[primary_phase]
    extra_summaries = {
        "parameter_accept_prob": _finite_summary(
            "parameter_accept_prob",
            extra_fields["parameter_accept_prob"],
        ),
        "estimated_log_likelihood": _finite_summary(
            "estimated_log_likelihood",
            extra_fields["estimated_log_likelihood"],
        ),
        "estimated_log_posterior": _finite_summary(
            "estimated_log_posterior",
            extra_fields["estimated_log_posterior"],
        ),
        "log_alpha": _finite_summary("log_alpha", extra_fields["log_alpha"]),
        "pf_ess_min": _finite_summary("pf_ess_min", extra_fields["pf_ess_min"]),
        "pf_ess_mean": _finite_summary("pf_ess_mean", extra_fields["pf_ess_mean"]),
        "pf_log_weight_range_max": _finite_summary(
            "pf_log_weight_range_max",
            extra_fields["pf_log_weight_range_max"],
        ),
        "pf_log_weight_variance_mean": _finite_summary(
            "pf_log_weight_variance_mean",
            extra_fields["pf_log_weight_variance_mean"],
        ),
        "pf_log_likelihood_increment_variance": _finite_summary(
            "pf_log_likelihood_increment_variance",
            extra_fields["pf_log_likelihood_increment_variance"],
        ),
    }
    if (
        MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics
        and "parameter_jump_rms" in extra_fields
    ):
        extra_summaries["parameter_jump_rms"] = _finite_summary(
            "parameter_jump_rms",
            extra_fields["parameter_jump_rms"],
        )
    if (
        MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics
        and "pf_ess_by_t" in extra_fields
    ):
        extra_summaries["pf_ess_by_t"] = _finite_summary(
            "pf_ess_by_t",
            extra_fields["pf_ess_by_t"],
        )
        extra_summaries["pf_log_weight_range_by_t"] = _finite_summary(
            "pf_log_weight_range_by_t",
            extra_fields["pf_log_weight_range_by_t"],
        )

    final_step = np.asarray(diagnostics["final_param_step_size"], dtype=np.float64)
    diagnostic_summary = {
        "parameter_kernel": diagnostics["parameter_kernel"],
        "particle_likelihood_estimator": "bootstrap_filter",
        "parameter_preconditioned": diagnostics["parameter_preconditioned"],
        "parameter_warmup": diagnostics.get("parameter_warmup"),
        "init_method": diagnostics.get("init_method"),
        "auto_preconditioner": diagnostics.get("auto_preconditioner"),
        "parameter_accept_rate": diagnostics["parameter_accept_rate"],
        "parameter_jump_rms_mean": diagnostics.get("parameter_jump_rms_mean"),
        "mcmc_phase_seconds": diagnostics["mcmc_phase_seconds"],
        "first_step_seconds": diagnostics.get("first_step_seconds"),
        "sampling_loop_seconds": diagnostics.get("sampling_loop_seconds"),
        "n_particles": diagnostics["n_particles"],
        "param_target_accept": diagnostics["param_target_accept"],
        "param_step_size_initial": diagnostics["param_step_size_initial"],
        "final_param_step_size_min": float(np.min(final_step)),
        "final_param_step_size_median": float(np.median(final_step)),
        "final_param_step_size_max": float(np.max(final_step)),
        "estimated_log_likelihood_mean": diagnostics["estimated_log_likelihood_mean"],
        "estimated_log_posterior_mean": diagnostics["estimated_log_posterior_mean"],
        "pf_ess_min_mean": diagnostics["pf_ess_min_mean"],
        "pf_ess_mean_mean": diagnostics["pf_ess_mean_mean"],
        "pf_log_weight_range_max_mean": diagnostics["pf_log_weight_range_max_mean"],
        "pf_log_weight_variance_mean": diagnostics["pf_log_weight_variance_mean"],
        "pf_log_likelihood_increment_variance_mean": diagnostics[
            "pf_log_likelihood_increment_variance_mean"
        ],
    }
    return {
        "diagnostic_primary_phase": primary_phase,
        "diagnostics": diagnostic_summary,
        "per_step_trace": _pmmh_per_step_trace(
            extra_fields=extra_fields,
            selected_metrics=selected_metrics,
            scope=primary_phase,
        ),
        "per_phase_trace": {
            phase: _pmmh_per_step_trace(
                extra_fields=fields,
                selected_metrics=selected_metrics,
                scope=phase,
            )
            for phase, fields in phase_fields.items()
            if phase in {"warmup", "post_warmup"} and _phase_step_count(fields) > 0
        },
        "particle_smoother_health": {
            "enabled": False,
            "note": (
                "PMMH uses a bootstrap particle-filter likelihood estimator and "
                "does not expose conditional latent-smoother health diagnostics."
            ),
        },
        "particle_smoother_health_by_phase": {},
        "posterior_ess": scalar_posterior_ess(
            result,
            max_sites=ess_max_sites,
            elapsed_seconds=elapsed_seconds,
        ),
        "parameter_recovery": parameter_recovery(
            result,
            elapsed_seconds=elapsed_seconds,
        ),
        "latent_path_mixing": {
            "enabled": False,
            "note": "PMMH benchmark does not retain latent paths.",
        },
        "latent_path_mixing_by_phase": {},
        "sample_summaries": sample_summaries,
        "extra_summaries": extra_summaries,
    }


def _run_one(args, *, fixture: _FixtureContext, smoother: str, seed: int) -> dict[str, Any]:
    selected_metrics = _selected_diagnostic_metrics(args)
    production_metrics = _production_diagnostic_metrics(selected_metrics)
    retain_latent_paths = args.retain_latent_paths or (
        LATENT_PATH_MIXING_METRIC in selected_metrics
    )
    cached_init_positions = None
    cached_init_latents = None
    cached_preconditioner_chol = None
    init_method = args.init_method
    auto_preconditioner_method = "pathfinder" if args.init_method == "pathfinder" else "none"
    if fixture.pathfinder_cache is not None:
        cached_preconditioner_chol = fixture.pathfinder_cache.parameter_preconditioner_chol
        auto_preconditioner_method = "none"
        if args.init_positions == "cache":
            cached_init_positions = fixture.pathfinder_cache.initial_positions
            cached_init_latents = fixture.pathfinder_cache.init_latent_paths
            init_method = "pathfinder"
        else:
            # Random positions, cached preconditioner. The pathfinder mode of this
            # fixture has locally explosive dynamics: its posterior density is
            # finite but unconditional predictive simulation from it diverges, so
            # the predictive latent init NaNs (kernel now fails fast on that).
            init_method = "random"

    fit_started = time.perf_counter()
    logger.info(
        "fit start: support_mode=%s smoother=%s seed=%d warmup=%d samples=%d T=%d "
        "n_particles=%d n_parameter_particles=%d metrics_all=%s cache=%s",
        fixture.support_mode,
        smoother,
        int(seed),
        int(args.num_warmup),
        int(args.num_samples),
        int(args.T),
        int(args.n_particles),
        int(args.n_parameter_particles),
        bool(args.all_metrics),
        "yes" if fixture.pathfinder_cache is not None else "no",
    )
    if smoother == PMMH_BENCHMARK_METHOD:
        result = fit(
            fixture.model,
            observations=fixture.data.observations,
            times=fixture.data.times,
            method="particle_marginal_mh",
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            seed=seed,
            n_particles=args.n_particles,
            param_step_size=args.param_step_size,
            adaptation_scheme="simple",
            init_method=init_method,
            auto_preconditioner_method=auto_preconditioner_method,
            parameter_preconditioner_chol=cached_preconditioner_chol,
            initial_positions_override=cached_init_positions,
            init_scale=0.0,
            retain_latent_paths=False,
            compute_latent_posterior_summary=False,
            diagnostic_metrics_all=args.all_metrics,
            diagnostic_metrics=production_metrics,
            pathfinder_num_elbo_samples=args.pathfinder_num_elbo_samples,
            pathfinder_maxiter=args.pathfinder_maxiter,
            n_pathfinder_starts=args.n_pathfinder_starts,
            pathfinder_parallel_workers=args.pathfinder_parallel_workers,
            pathfinder_init_scale=args.pathfinder_init_scale,
            n_ieks_iters=PATHFINDER_N_IEKS_ITERS,
            reparam=None,
        )
    else:
        result = fit(
            fixture.model,
            observations=fixture.data.observations,
            times=fixture.data.times,
            method="marginal_particle_gibbs",
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            seed=seed,
            n_particles=args.n_particles,
            n_parameter_particles=args.n_parameter_particles,
            latent_block_size=args.latent_block_size,
            # Post-refactor API: "amala"/"amala_plus" are dsmc leaf proposals, not
            # latent smoothers; map the benchmark label to (latent_smoother, leaf).
            latent_smoother=("plain" if smoother == "plain" else "dsmc"),
            amala_delta_init=args.amala_delta_init,
            amala_delta_min=args.amala_delta_min,
            amala_delta_max=args.amala_delta_max,
            amala_target_accept=args.amala_target_accept,
            amala_adaptation_window=args.amala_adaptation_window,
            amala_adaptation_tolerance=args.amala_adaptation_tolerance,
            amala_adaptation_rho=args.amala_adaptation_rho,
            amala_adaptation_rho_min=args.amala_adaptation_rho_min,
            amala_adaptation_gamma=args.amala_adaptation_gamma,
            amala_kappa=args.amala_kappa,
            amala_grad_clip=args.amala_grad_clip,
            dsmc_leaf_proposal=(args.dsmc_leaf_proposal if smoother == "plain" else smoother),
            mgrad_grad_clip=args.mgrad_grad_clip,
            param_step_size=args.param_step_size,
            parameter_proposal=args.parameter_proposal,
            adaptation_scheme="simple",
            init_method=init_method,
            latent_init_method="predictive",
            auto_preconditioner_method=auto_preconditioner_method,
            parameter_preconditioner_chol=cached_preconditioner_chol,
            initial_positions_override=cached_init_positions,
            initial_latent_trajectories=cached_init_latents,
            init_scale=0.0,
            retain_latent_paths=retain_latent_paths,
            compute_latent_posterior_summary=not args.skip_latent_summary,
            diagnostic_metrics_all=args.all_metrics,
            diagnostic_metrics=production_metrics,
            pathfinder_num_elbo_samples=args.pathfinder_num_elbo_samples,
            pathfinder_maxiter=args.pathfinder_maxiter,
            n_pathfinder_starts=args.n_pathfinder_starts,
            pathfinder_parallel_workers=args.pathfinder_parallel_workers,
            pathfinder_init_scale=args.pathfinder_init_scale,
            reparam=None,
        )
    fit_seconds = time.perf_counter() - fit_started
    logger.info(
        "fit complete: support_mode=%s smoother=%s elapsed=%.2fs",
        fixture.support_mode,
        smoother,
        fit_seconds,
    )

    check_started = time.perf_counter()
    logger.info(
        "diagnostic validation start: support_mode=%s smoother=%s",
        fixture.support_mode,
        smoother,
    )
    if smoother == PMMH_BENCHMARK_METHOD:
        checked = _check_pmmh_result(
            result,
            elapsed_seconds=fit_seconds,
            ess_max_sites=args.ess_max_sites,
            selected_metrics=selected_metrics,
        )
    else:
        checked = _check_result(
            result,
            smoother=smoother,
            elapsed_seconds=fit_seconds,
            ess_max_sites=args.ess_max_sites,
            selected_metrics=selected_metrics,
        )
    check_seconds = time.perf_counter() - check_started
    logger.info(
        "diagnostic validation complete: support_mode=%s smoother=%s elapsed=%.2fs primary_phase=%s",
        fixture.support_mode,
        smoother,
        check_seconds,
        checked["diagnostic_primary_phase"],
    )
    shared_cache_seconds = (
        0.0 if fixture.pathfinder_cache is None else fixture.pathfinder_cache.elapsed_seconds
    )
    checked["elapsed_seconds"] = float(
        fixture.setup_seconds + shared_cache_seconds + fit_seconds + check_seconds
    )
    checked["pipeline_steps"] = {
        "fixture_and_model_setup_seconds": float(fixture.setup_seconds),
        "shared_pathfinder_cache_seconds": float(shared_cache_seconds),
        "production_fit_seconds": float(fit_seconds),
        "trace_validation_seconds": float(check_seconds),
    }
    checked["pathfinder_cache"] = (
        None
        if fixture.pathfinder_cache is None
        else {
            "path": fixture.pathfinder_cache.path,
            "source": fixture.pathfinder_cache.source,
            "initial_positions_shape": list(fixture.pathfinder_cache.initial_positions.shape),
            "parameter_preconditioner_chol_shape": list(
                fixture.pathfinder_cache.parameter_preconditioner_chol.shape
            ),
        }
    )
    checked["support_mode"] = fixture.support_mode
    checked["seed"] = int(seed)
    return checked


def _parse_smoothers(raw: str) -> tuple[str, ...]:
    smoothers = tuple(item.strip() for item in raw.split(",") if item.strip())
    allowed = {*EXPECTED_SMOOTHER_SELECTION, PMMH_BENCHMARK_METHOD}
    unknown = sorted(set(smoothers) - allowed)
    if unknown:
        raise ValueError(f"unknown smoothers: {unknown}; allowed: {sorted(allowed)}")
    if not smoothers:
        raise ValueError("at least one smoother is required")
    return smoothers


def _support_modes(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return ("point", "interval")
    return (raw,)


def _selected_diagnostic_metrics(args: argparse.Namespace) -> frozenset[str]:
    if args.all_metrics:
        return frozenset(SCRIPT_DIAGNOSTIC_METRIC_VALUES)
    return frozenset(args.metric or ())


def _production_diagnostic_metrics(selected_metrics: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        metric for metric in MPGIBBS_DIAGNOSTIC_METRIC_VALUES if metric in selected_metrics
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MPGibbs smoothers and the optional PMMH baseline on the synthetic "
            "nonlinear fixture."
        )
    )
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--n-parameter-particles", type=int, default=2)
    parser.add_argument("--latent-block-size", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=71)
    parser.add_argument(
        "--diffusion-scale",
        type=float,
        default=1.0,
        help="Multiply the true process-noise SD: 1.0 informative, >1 diffuse dynamics.",
    )
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--param-step-size", type=float, default=0.02)
    parser.add_argument(
        "--parameter-proposal",
        choices=("random_walk", "pseudo_langevin"),
        default="random_walk",
    )
    parser.add_argument("--amala-delta-init", type=float, default=1e-2)
    parser.add_argument("--amala-delta-min", type=float, default=1e-5)
    parser.add_argument("--amala-delta-max", type=float, default=1e1)
    parser.add_argument("--amala-target-accept", type=float, default=0.75)
    parser.add_argument("--amala-adaptation-window", type=int, default=100)
    parser.add_argument("--amala-adaptation-tolerance", type=float, default=0.05)
    parser.add_argument("--amala-adaptation-rho", type=float, default=0.5)
    parser.add_argument("--amala-adaptation-rho-min", type=float, default=1e-3)
    parser.add_argument("--amala-adaptation-gamma", type=float, default=-0.5)
    parser.add_argument("--amala-kappa", type=float, default=0.75)
    parser.add_argument("--amala-grad-clip", type=float, default=math.inf)
    parser.add_argument(
        "--dsmc-leaf-proposal",
        choices=("prior_predictive", "amala", "amala_plus", "amala_exact"),
        default="prior_predictive",
    )
    parser.add_argument("--mgrad-grad-clip", type=float, default=10.0)
    parser.add_argument(
        "--init-method",
        choices=("random", "pathfinder"),
        default="pathfinder",
        help=("Production init method used only when --pathfinder-cache-mode=off."),
    )
    parser.add_argument(
        "--init-positions",
        choices=("cache", "random"),
        default="cache",
        help=(
            "Initial parameter positions when a pathfinder cache is active: 'cache' "
            "starts at the cached pathfinder mode; 'random' draws prior-scale "
            "positions while still using the cached preconditioner (needed when the "
            "mode is explosive under the predictive latent init)."
        ),
    )
    parser.add_argument(
        "--pathfinder-cache-mode",
        choices=("reuse", "refresh", "off"),
        default="reuse",
        help=(
            "Run/load one shared Pathfinder warmup per support mode and pass its "
            "positions/preconditioner into every inference method. 'refresh' "
            "overwrites the cache; 'off' lets each fit use --init-method."
        ),
    )
    parser.add_argument(
        "--pathfinder-cache-path",
        type=Path,
        default=REPO_ROOT / "scratchpad" / "mpg_smoothers_synthetic_nonlinear_pathfinder_cache.npz",
        help="Relative paths resolve against the repo root, not the CWD.",
    )
    parser.add_argument("--pathfinder-seed", type=int, default=None)
    parser.add_argument("--pathfinder-num-elbo-samples", type=int, default=20)
    parser.add_argument("--pathfinder-maxiter", type=int, default=20)
    parser.add_argument("--n-pathfinder-starts", type=int, default=8)
    parser.add_argument("--pathfinder-parallel-workers", type=int, default=None)
    parser.add_argument("--pathfinder-init-scale", type=float, default=0.1)
    parser.add_argument("--ess-max-sites", type=int, default=24)
    parser.add_argument("--smoothers", default="plain,amala,amala_plus")
    parser.add_argument(
        "--all-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable every optional high-memory diagnostic metric group.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=SCRIPT_DIAGNOSTIC_METRIC_VALUES,
        default=None,
        help=(
            "Optional diagnostic metric group to collect when --no-all-metrics is used. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--support-mode",
        choices=("point", "interval", "both"),
        default="point",
        help=(
            "Observation-support mode. 'point' is the nonlinear smoother smoke "
            "check; 'interval' and 'both' probe the current interval-support caveat."
        ),
    )
    parser.add_argument(
        "--retain-latent-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--skip-latent-summary", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "scratchpad" / "mpg_smoothers_synthetic_nonlinear_check.json",
        help="Relative paths resolve against the repo root, not the CWD.",
    )
    args = parser.parse_args()
    # Resolve output paths against the repo root so artifacts always land in the
    # single root-level scratchpad regardless of the directory the script runs
    # from. A bare ``scratchpad/foo.json`` would otherwise follow the CWD and
    # spawn a second, nested scratchpad under ``apps/data-pipeline/``.
    for attr in ("pathfinder_cache_path", "output"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, REPO_ROOT / path)
    return args


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    if args.T < 8:
        raise ValueError("T must be at least 8 so the nonlinear fixture has sparse events")
    smoothers = _parse_smoothers(args.smoothers)
    support_modes = _support_modes(args.support_mode)
    selected_metrics = _selected_diagnostic_metrics(args)
    if not TRUE_HILL_BY_SITE or not TRUE_MULTIPLICATIVE_BY_SITE:
        raise AssertionError("synthetic fixture is not exercising nonlinear dynamics")
    logger.info(
        "benchmark start: T=%d warmup=%d samples=%d smoothers=%s support_modes=%s "
        "pathfinder_cache_mode=%s all_metrics=%s",
        int(args.T),
        int(args.num_warmup),
        int(args.num_samples),
        ",".join(smoothers),
        ",".join(support_modes),
        args.pathfinder_cache_mode,
        bool(args.all_metrics),
    )

    runs: dict[str, Any] = {}
    shared_setup: dict[str, Any] = {}
    for support_idx, support_mode in enumerate(support_modes):
        fixture = _prepare_fixture_context(
            args,
            support_mode=support_mode,
            support_idx=support_idx,
            support_mode_count=len(support_modes),
        )
        shared_setup[support_mode] = {
            "fixture_and_model_setup_seconds": float(fixture.setup_seconds),
            "pathfinder_cache": (
                None
                if fixture.pathfinder_cache is None
                else {
                    "path": fixture.pathfinder_cache.path,
                    "source": fixture.pathfinder_cache.source,
                    "elapsed_seconds": float(fixture.pathfinder_cache.elapsed_seconds),
                    "initial_positions_shape": list(
                        fixture.pathfinder_cache.initial_positions.shape
                    ),
                    "parameter_preconditioner_chol_shape": list(
                        fixture.pathfinder_cache.parameter_preconditioner_chol.shape
                    ),
                    "diagnostics": fixture.pathfinder_cache.diagnostics,
                }
            ),
        }
        for smoother_idx, smoother in enumerate(smoothers):
            run_key = f"{support_mode}:{smoother}"
            print(f"running {run_key}...", flush=True)
            runs[run_key] = _run_one(
                args,
                fixture=fixture,
                smoother=smoother,
                seed=args.seed + 100 * support_idx + smoother_idx,
            )

    payload = {
        "artifact": "mpg_smoothers_synthetic_nonlinear_check",
        "target": "synthetic_nonlinear",
        "config": {
            "T": args.T,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "log_level": args.log_level,
            "n_particles": args.n_particles,
            "n_parameter_particles": args.n_parameter_particles,
            "latent_block_size": args.latent_block_size,
            "amala_delta_init": args.amala_delta_init,
            "amala_delta_min": args.amala_delta_min,
            "amala_delta_max": args.amala_delta_max,
            "amala_target_accept": args.amala_target_accept,
            "amala_adaptation_window": args.amala_adaptation_window,
            "amala_adaptation_tolerance": args.amala_adaptation_tolerance,
            "amala_adaptation_rho": args.amala_adaptation_rho,
            "amala_adaptation_rho_min": args.amala_adaptation_rho_min,
            "amala_adaptation_gamma": args.amala_adaptation_gamma,
            "amala_kappa": args.amala_kappa,
            "dsmc_leaf_proposal": args.dsmc_leaf_proposal,
            "pathfinder_cache_mode": args.pathfinder_cache_mode,
            "pathfinder_cache_path": args.pathfinder_cache_path,
            "pathfinder_seed": args.pathfinder_seed,
            "pathfinder_n_ieks_iters": PATHFINDER_N_IEKS_ITERS,
            "pathfinder_num_elbo_samples": args.pathfinder_num_elbo_samples,
            "pathfinder_maxiter": args.pathfinder_maxiter,
            "n_pathfinder_starts": args.n_pathfinder_starts,
            "pathfinder_parallel_workers": args.pathfinder_parallel_workers,
            "pathfinder_init_scale": args.pathfinder_init_scale,
            "pathfinder_parameter_init_scale": PATHFINDER_PARAMETER_INIT_SCALE,
            "all_metrics": args.all_metrics,
            "diagnostic_metrics": sorted(selected_metrics),
            "retain_latent_paths": bool(
                args.retain_latent_paths or LATENT_PATH_MIXING_METRIC in selected_metrics
            ),
            "support_mode": args.support_mode,
            "smoothers": smoothers,
        },
        "nonlinear_fixture": {
            "hill_site_count": len(TRUE_HILL_BY_SITE),
            "multiplicative_site_count": len(TRUE_MULTIPLICATIVE_BY_SITE),
        },
        "shared_setup": shared_setup,
        "runs": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    logger.info("benchmark complete: output=%s run_count=%d", args.output, len(runs))
    print(f"wrote {args.output}", flush=True)
    print("all requested particle MCMC checks passed", flush=True)


if __name__ == "__main__":
    main()
