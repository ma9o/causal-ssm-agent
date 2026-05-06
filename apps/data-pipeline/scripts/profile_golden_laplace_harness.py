#!/usr/bin/env python3
# ruff: noqa: E402
"""Instrumented GOLDEN MAP profiling harness.

This script reconstructs the GOLDEN prepared runtime from saved artifacts and
profiles the support-aware MAP path at phase-level granularity.

It is intentionally wired to the same runtime-preparation code path used by
stage 5 so the measurements reflect production behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_run_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"golden-laplace-profile-{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = _default_repo_root()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root. Defaults to the current checkout root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_default_run_dir(),
        help="Output directory for traces, dumps, logs, and summary artifacts.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "data/.private/GOLDEN/run/stage2-model-data.parquet",
        help="Path to the GOLDEN stage-2 model data parquet.",
    )
    parser.add_argument(
        "--compiled-ssm-path",
        type=Path,
        default=repo_root / "data/.private/GOLDEN/run/stage4-compiled-ssm.json",
        help="Path to the compiled stage-4 SSM artifact.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for trace/init/sample key splitting.",
    )
    parser.add_argument(
        "--n-ieks-iters",
        type=int,
        default=3,
        help="Inner IEKS iteration budget passed to the Laplace backend.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1,
        help="Outer L-BFGS-B maxiter when running the optimizer probe.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Outer optimizer tolerance.",
    )
    parser.add_argument(
        "--n-init-samples",
        type=int,
        default=32,
        help="Outer initialization sample budget when applicable.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Dummy posterior draw count for parity with fit config logs.",
    )
    parser.add_argument(
        "--resource-interval",
        type=float,
        default=15.0,
        help="Seconds between background process resource log lines.",
    )
    parser.add_argument(
        "--skip-scalar-log-posterior",
        action="store_true",
        help="Skip the cold scalar log-posterior probe.",
    )
    parser.add_argument(
        "--skip-scalar-aux",
        action="store_true",
        help="Skip the cold scalar neg-log-posterior-with-aux probe.",
    )
    parser.add_argument(
        "--skip-jitted-value-and-grad",
        action="store_true",
        help="Skip the cold jitted value-and-grad probe.",
    )
    parser.add_argument(
        "--skip-warm-array-probe",
        action="store_true",
        help="Skip the second jitted probe that reuses the returned latent mode.",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip the full `_optimize_laplace_parameter_mode(..., maxiter=...)` probe.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable Perfetto trace capture.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Override Perfetto/JAX trace output directory.",
    )
    parser.add_argument(
        "--no-log-compiles",
        action="store_true",
        help="Disable `JAX_LOG_COMPILES=1`.",
    )
    parser.add_argument(
        "--no-xla-dump",
        action="store_true",
        help="Disable XLA HLO dump emission.",
    )
    parser.add_argument(
        "--xla-dump-dir",
        type=Path,
        default=None,
        help="Override XLA dump directory.",
    )
    parser.add_argument(
        "--xla-dump-pass-re",
        default="(AlgebraicSimplifier|HloRematerialization|algebraic|rematerial)",
        help="Regex passed to `--xla_dump_hlo_pass_re`.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Harness log verbosity.",
    )
    return parser.parse_args()


ARGS = _parse_args()
ARGS.run_dir = ARGS.run_dir.resolve()
ARGS.trace_dir = (ARGS.trace_dir or (ARGS.run_dir / "trace")).resolve()
ARGS.xla_dump_dir = (ARGS.xla_dump_dir or (ARGS.run_dir / "xla_dump")).resolve()
ARGS.run_dir.mkdir(parents=True, exist_ok=True)
if not ARGS.no_trace:
    ARGS.trace_dir.mkdir(parents=True, exist_ok=True)
if not ARGS.no_xla_dump:
    ARGS.xla_dump_dir.mkdir(parents=True, exist_ok=True)


def _append_xla_flag(flag: str) -> None:
    existing = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


if not ARGS.no_log_compiles:
    os.environ.setdefault("JAX_LOG_COMPILES", "1")
if not ARGS.no_xla_dump:
    _append_xla_flag(f"--xla_dump_to={ARGS.xla_dump_dir}")
    _append_xla_flag("--xla_dump_hlo_as_text")
    _append_xla_flag(f"--xla_dump_hlo_pass_re={ARGS.xla_dump_pass_re}")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import polars as pl

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.inference.methods.map import (
    _build_map_laplace_bundle,
    _hostify_outer_eval_diagnostics,
    _optimize_laplace_parameter_mode,
    _requires_support_aware_outer_optimizer,
)
from causal_ssm_agent.models.ssm_builder import prepare_model_runtime

LOGGER = logging.getLogger("golden_laplace_harness")


def _configure_logging() -> Path:
    log_path = ARGS.run_dir / "harness.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, ARGS.log_level))
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def _current_ps_snapshot(pid: int) -> dict[str, str]:
    cmd = ["ps", "-o", "rss=,%cpu=,etime=,state=", "-p", str(pid)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    parts = result.stdout.strip().split()
    rss_kb, cpu_pct, elapsed, state = parts[:4]
    return {
        "rss_kb": rss_kb,
        "cpu_pct": cpu_pct,
        "elapsed": elapsed,
        "state": state,
    }


class _ResourceSampler:
    def __init__(self, pid: int, interval_seconds: float):
        self._pid = pid
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(self._interval_seconds, 1.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                snapshot = _current_ps_snapshot(self._pid)
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:  # pragma: no cover
                LOGGER.warning("resource sample failed: %r", exc)
                continue
            LOGGER.info(
                "resource pid=%d rss_mb=%.1f cpu_pct=%s elapsed=%s state=%s",
                self._pid,
                int(snapshot["rss_kb"]) / 1024.0,
                snapshot["cpu_pct"],
                snapshot["elapsed"],
                snapshot["state"],
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _phase(
    phase_name: str,
    summary: dict[str, Any],
    fn,
):
    LOGGER.info("phase start: %s", phase_name)
    started_at = time.monotonic()
    with jax.profiler.TraceAnnotation(f"harness/{phase_name}"):
        result = fn()
    elapsed = time.monotonic() - started_at
    summary.setdefault("phase_seconds", {})[phase_name] = elapsed
    LOGGER.info("phase complete: %s elapsed=%.3fs", phase_name, elapsed)
    return result


def _summarize_support(runtime, backend) -> dict[str, Any]:
    support = runtime.observation_support
    summary: dict[str, Any] = {
        "has_support": support is not None,
        "requires_interval_summary_handling": bool(
            support is not None and support.requires_interval_summary_handling
        ),
        "max_active_windows": int(support.max_active_windows) if support is not None else 0,
        "manifest_names": list(runtime.manifest_names),
    }
    if support is None:
        return summary

    summary["support_kinds"] = list(support.support_kinds)
    summary["interval_summary_manifest_names"] = list(support.interval_summary_manifest_names)
    summary["summary_operators"] = list(support.summary_operators)

    if not summary["requires_interval_summary_handling"]:
        return summary

    batches = tuple(getattr(backend, "_support_window_batches", ()))
    row_upper_bandwidths = np.asarray(
        jax.device_get(backend._support_row_upper_bandwidths)
    )
    state_lens = []
    bucket_window_counts: dict[int, int] = {}
    for batch in batches:
        batch_state_lens = np.asarray(jax.device_get(batch.state_lens), dtype=np.int64)
        state_lens.append(batch_state_lens)
        bucket_window_counts[int(batch.max_state_len)] = int(batch_state_lens.size)

    if state_lens:
        all_state_lens = np.concatenate(state_lens)
        diag_block_terms = int(np.sum(all_state_lens))
        cross_block_terms = int(np.sum(all_state_lens * (all_state_lens - 1) // 2))
        summary.update(
            {
                "support_batches": len(batches),
                "support_batch_bucket_max_state_lens": [int(batch.max_state_len) for batch in batches],
                "support_batch_window_counts": bucket_window_counts,
                "support_total_windows": int(all_state_lens.size),
                "support_state_len_min": int(np.min(all_state_lens)),
                "support_state_len_median": float(np.median(all_state_lens)),
                "support_state_len_p95": float(np.percentile(all_state_lens, 95)),
                "support_state_len_max": int(np.max(all_state_lens)),
                "support_diag_block_terms": diag_block_terms,
                "support_cross_block_terms": cross_block_terms,
            }
        )
    summary.update(
        {
            "support_bandwidth": int(backend._support_bandwidth),
            "support_row_upper_bandwidth_max": int(np.max(row_upper_bandwidths, initial=0)),
            "support_row_upper_bandwidth_mean": float(np.mean(row_upper_bandwidths))
            if row_upper_bandwidths.size
            else 0.0,
            "support_row_upper_bandwidth_p95": float(np.percentile(row_upper_bandwidths, 95))
            if row_upper_bandwidths.size
            else 0.0,
        }
    )
    return summary


def _extract_latent_mode(aux: dict[str, Any]) -> np.ndarray | None:
    if "latent_mode" not in aux:
        return None
    return np.asarray(jax.device_get(aux["latent_mode"]), dtype=np.float64)


def _hostify_outer_eval_with_latent_mode(aux: dict[str, Any]) -> dict[str, Any]:
    host = _hostify_outer_eval_diagnostics(aux)
    latent_mode = _extract_latent_mode(aux)
    if latent_mode is not None:
        host["latent_mode"] = latent_mode
    return host


def _write_summary(summary: dict[str, Any]) -> None:
    summary_path = ARGS.run_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    LOGGER.info("wrote summary: %s", summary_path)


def main() -> int:
    log_path = _configure_logging()
    LOGGER.info("run dir: %s", ARGS.run_dir)
    LOGGER.info("log path: %s", log_path)
    LOGGER.info("trace enabled: %s", not ARGS.no_trace)
    LOGGER.info("trace dir: %s", ARGS.trace_dir if not ARGS.no_trace else "disabled")
    LOGGER.info("xla dump enabled: %s", not ARGS.no_xla_dump)
    LOGGER.info("xla dump dir: %s", ARGS.xla_dump_dir if not ARGS.no_xla_dump else "disabled")
    LOGGER.info("JAX_LOG_COMPILES=%s", os.environ.get("JAX_LOG_COMPILES", "0"))
    LOGGER.info("XLA_FLAGS=%s", os.environ.get("XLA_FLAGS", ""))

    summary: dict[str, Any] = {
        "args": vars(ARGS).copy(),
        "pid": os.getpid(),
        "jax_version": jax.__version__,
        "started_at_epoch": time.time(),
    }
    sampler = _ResourceSampler(os.getpid(), ARGS.resource_interval)
    sampler.start()

    trace_ctx = (
        jax.profiler.trace(
            ARGS.trace_dir,
            create_perfetto_trace=True,
        )
        if not ARGS.no_trace
        else contextlib.nullcontext()
    )

    try:
        with trace_ctx:
            compiled_ssm = _phase(
                "load_compiled_ssm",
                summary,
                lambda: json.loads(ARGS.compiled_ssm_path.read_text()),
            )
            data_for_model = _phase(
                "load_stage2_data",
                summary,
                lambda: pl.read_parquet(ARGS.data_path),
            )
            LOGGER.info(
                "loaded artifacts rows=%d compiled_keys=%d",
                len(data_for_model),
                len(compiled_ssm),
            )

            runtime = _phase(
                "prepare_runtime",
                summary,
                lambda: prepare_model_runtime(
                    data_for_model=data_for_model,
                    compiled_ssm=compiled_ssm,
                    sampler_config={"method": "map"},
                ),
            )
            observations = runtime.observations
            times = runtime.times
            model = runtime.model
            observed_cells = int(jnp.sum(~jnp.isnan(observations)).item())
            total_cells = int(observations.size)
            LOGGER.info(
                "prepared runtime timepoints=%d manifests=%d observed_cells=%d/%d",
                observations.shape[0],
                observations.shape[1],
                observed_cells,
                total_cells,
            )

            backend = _phase(
                "build_laplace_backend",
                summary,
                lambda: model.make_laplace_backend(ARGS.n_ieks_iters),
            )
            support_summary = _summarize_support(runtime, backend)
            summary["support_summary"] = support_summary
            LOGGER.info("support summary: %s", json.dumps(_jsonable(support_summary), sort_keys=True))

            rng_key = random.PRNGKey(ARGS.seed)
            rng_key, trace_key, init_key, _sample_key = random.split(rng_key, 4)
            reparam = AutoReparam(centered=0.0)

            bundle = _phase(
                "build_laplace_bundle",
                summary,
                lambda: _build_map_laplace_bundle(
                    model,
                    observations,
                    times,
                    trace_key,
                    backend,
                    reparam,
                ),
            )
            dim = int(bundle["dim"])
            summary["parameter_dim"] = dim
            LOGGER.info("bundle ready parameter_dim=%d support_aware_outer=%s", dim, _requires_support_aware_outer_optimizer(model))

            flat_example = bundle["flat_example"]
            log_posterior_fn = bundle["log_posterior_fn"]
            neg_log_posterior_with_aux_fn = bundle["neg_log_posterior_with_aux_fn"]
            batch_log_posterior_jit = bundle["batch_log_posterior_jit"]
            site_info = bundle["site_info"]
            z0 = flat_example

            cold_aux_host: dict[str, Any] | None = None
            cold_latent_mode: np.ndarray | None = None

            if not ARGS.skip_scalar_log_posterior:
                scalar_log_posterior = _phase(
                    "scalar_log_posterior",
                    summary,
                    lambda: float(jax.device_get(log_posterior_fn(z0, latent_mode_init=None))),
                )
                summary["scalar_log_posterior"] = scalar_log_posterior
                LOGGER.info("scalar log posterior=%s", scalar_log_posterior)

            if not ARGS.skip_scalar_aux:
                def _scalar_aux_probe():
                    objective, aux = neg_log_posterior_with_aux_fn(z0, latent_mode_init=None)
                    return float(jax.device_get(objective)), _hostify_outer_eval_with_latent_mode(aux)

                scalar_objective, cold_aux_host = _phase(
                    "scalar_neg_log_posterior_with_aux",
                    summary,
                    _scalar_aux_probe,
                )
                summary["scalar_neg_log_posterior_with_aux"] = scalar_objective
                summary["scalar_aux"] = cold_aux_host
                cold_latent_mode = (
                    np.asarray(cold_aux_host["latent_mode"], dtype=np.float64)
                    if "latent_mode" in cold_aux_host
                    else None
                )
                LOGGER.info(
                    "scalar aux objective=%s inner_solver=%s inner_iterations=%s",
                    scalar_objective,
                    cold_aux_host["inner"]["solver_kind"],
                    cold_aux_host["inner"]["n_iterations"],
                )

            if not ARGS.skip_jitted_value_and_grad:
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

                def _cold_vg_probe():
                    (fun, aux), grad = value_and_grad_fn(z0, None)
                    return (
                        float(jax.device_get(fun)),
                        float(np.linalg.norm(np.asarray(jax.device_get(grad), dtype=np.float64))),
                        _hostify_outer_eval_with_latent_mode(aux),
                    )

                cold_fun, cold_grad_norm, cold_vg_aux = _phase(
                    "jitted_value_and_grad_none",
                    summary,
                    _cold_vg_probe,
                )
                summary["jitted_value_and_grad_none"] = {
                    "objective": cold_fun,
                    "grad_norm": cold_grad_norm,
                    "aux": cold_vg_aux,
                }
                if cold_latent_mode is None and "latent_mode" in cold_vg_aux:
                    cold_latent_mode = np.asarray(cold_vg_aux["latent_mode"], dtype=np.float64)
                LOGGER.info(
                    "jitted value_and_grad none objective=%s grad_norm=%s",
                    cold_fun,
                    cold_grad_norm,
                )

                if not ARGS.skip_warm_array_probe and cold_latent_mode is not None:
                    latent_mode_arg = jnp.asarray(cold_latent_mode, dtype=observations.dtype)

                    def _warm_array_probe():
                        (fun, aux), grad = value_and_grad_fn(z0, latent_mode_arg)
                        return (
                            float(jax.device_get(fun)),
                            float(np.linalg.norm(np.asarray(jax.device_get(grad), dtype=np.float64))),
                            _hostify_outer_eval_with_latent_mode(aux),
                        )

                    warm_fun, warm_grad_norm, warm_aux = _phase(
                        "jitted_value_and_grad_array",
                        summary,
                        _warm_array_probe,
                    )
                    summary["jitted_value_and_grad_array"] = {
                        "objective": warm_fun,
                        "grad_norm": warm_grad_norm,
                        "aux": warm_aux,
                    }
                    LOGGER.info(
                        "jitted value_and_grad array objective=%s grad_norm=%s",
                        warm_fun,
                        warm_grad_norm,
                    )

            if not ARGS.skip_optimize:
                mode_result = _phase(
                    "optimize_parameter_mode",
                    summary,
                    lambda: _optimize_laplace_parameter_mode(
                        model,
                        init_key=init_key,
                        dim=dim,
                        flat_example=flat_example,
                        site_info=site_info,
                        log_posterior_fn=log_posterior_fn,
                        neg_log_posterior_with_aux_fn=neg_log_posterior_with_aux_fn,
                        batch_log_posterior_jit=batch_log_posterior_jit,
                        observations=observations,
                        n_init_samples=ARGS.n_init_samples,
                        maxiter=ARGS.maxiter,
                        tol=ARGS.tol,
                    ),
                )
                summary["optimize_result"] = {
                    "success": bool(mode_result.success),
                    "status": int(mode_result.status),
                    "n_iters": int(mode_result.n_iters),
                    "n_function_evals": int(mode_result.n_function_evals),
                    "objective_at_mode": float(mode_result.objective_at_mode),
                    "final_grad_norm": float(mode_result.final_grad_norm or 0.0),
                    "optimizer": mode_result.optimizer,
                    "init_log_posterior_best": float(mode_result.init_log_posterior_best),
                    "final_eval_diagnostics": mode_result.final_eval_diagnostics,
                }
                LOGGER.info(
                    "optimizer result success=%s status=%s nit=%s nfev=%s objective=%s",
                    mode_result.success,
                    mode_result.status,
                    mode_result.n_iters,
                    mode_result.n_function_evals,
                    mode_result.objective_at_mode,
                )

        final_snapshot = _current_ps_snapshot(os.getpid())
        summary["final_ps_snapshot"] = final_snapshot
        summary["completed"] = True
        _write_summary(summary)
        LOGGER.info("completed successfully")
        return 0
    except KeyboardInterrupt as exc:
        summary["completed"] = False
        summary["exception"] = repr(exc)
        summary["interrupted"] = True
        summary["final_ps_snapshot"] = _current_ps_snapshot(os.getpid())
        _write_summary(summary)
        LOGGER.warning("profiling harness interrupted")
        return 130
    except Exception as exc:
        summary["completed"] = False
        summary["exception"] = repr(exc)
        summary["final_ps_snapshot"] = _current_ps_snapshot(os.getpid())
        _write_summary(summary)
        LOGGER.exception("profiling harness failed")
        return 1
    finally:
        sampler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
