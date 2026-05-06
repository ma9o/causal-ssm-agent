#!/usr/bin/env python3
# ruff: noqa: E402
"""Instrumented synthetic MAP profiling harness.

This script builds a synthetic 10-latent mixed-support model with strong point
measurement geometry while matching GOLDEN's support-window scale:

- 1514 interval windows
- max state length 32
- support bandwidth 31

It then profiles the same support-aware MAP outer objective used by the
GOLDEN harness at phase-level granularity.
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
from types import SimpleNamespace
from typing import Any


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_run_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"synthetic-scaled-laplace-profile-{timestamp}"


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
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for synthetic data generation and optimizer keys.",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=10,
        help="Latent dimension of the synthetic recovery model.",
    )
    parser.add_argument(
        "--target-interval-windows",
        type=int,
        default=1514,
        help="Target number of emitted interval windows.",
    )
    parser.add_argument(
        "--max-window-len",
        type=int,
        default=32,
        help="Maximum number of latent states touched by one interval window.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Synthetic data dtype. Default matches GOLDEN's float64 path.",
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
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.artifacts.model_spec import DistributionFamily
from causal_ssm_agent.models.ssm import (
    SSMModel,
    SSMPriors,
    SSMSpec,
    discretize_system,
    full_diagonal_mask,
    zero_diagonal_mask,
    zero_loading_mask,
    zero_square_mask,
    zero_vector_mask,
)
from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.inference.methods.map import (
    _build_map_laplace_bundle,
    _hostify_outer_eval_diagnostics,
    _optimize_laplace_parameter_mode,
    _requires_support_aware_outer_optimizer,
)
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

LOGGER = logging.getLogger("synthetic_scaled_laplace_harness")


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


def _phase(phase_name: str, summary: dict[str, Any], fn):
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
    row_upper_bandwidths = np.asarray(jax.device_get(backend._support_row_upper_bandwidths))
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
                "support_batch_bucket_max_state_lens": [
                    int(batch.max_state_len) for batch in batches
                ],
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


def _dtype_pair() -> tuple[Any, Any]:
    if ARGS.dtype == "float32":
        return jnp.float32, np.float32
    return jnp.float64, np.float64


def _window_start_index(anchor_idx: int, max_window_len: int) -> int:
    return max(anchor_idx - (max_window_len - 1), 0)


def _trapezoid_mean(series: np.ndarray) -> float:
    if series.shape[0] <= 1:
        return float(series[-1])
    numerator = 0.5 * series[0] + float(np.sum(series[1:-1])) + 0.5 * series[-1]
    return numerator / float(series.shape[0] - 1)


def _build_scaled_mixed_support_runtime(
    times_np: np.ndarray,
    manifest_names: list[str],
    *,
    max_window_len: int,
) -> ObservationSupportRuntime:
    n_time = int(times_np.shape[0])
    n_manifest = len(manifest_names)
    n_point = n_manifest // 2

    support_start = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
    support_end = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
    prev_coeffs = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    curr_coeffs = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    weights = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    emission_slots = np.full((n_time, n_manifest), -1, dtype=np.int64)

    for anchor_idx in range(1, n_time):
        interval_manifest_idx = n_point + ((anchor_idx - 1) % n_point)
        start_idx = _window_start_index(anchor_idx, max_window_len)
        support_start[anchor_idx, interval_manifest_idx] = times_np[start_idx]
        support_end[anchor_idx, interval_manifest_idx] = times_np[anchor_idx]
        emission_slots[anchor_idx, interval_manifest_idx] = 0
        for step_idx in range(start_idx + 1, anchor_idx + 1):
            dt = times_np[step_idx] - times_np[step_idx - 1]
            prev_coeffs[step_idx, interval_manifest_idx, 0] = 0.5 * dt
            curr_coeffs[step_idx, interval_manifest_idx, 0] = 0.5 * dt
            weights[step_idx, interval_manifest_idx, 0] = dt

    return ObservationSupportRuntime(
        anchor_times=times_np,
        manifest_names=manifest_names,
        support_kinds=["point"] * n_point + ["interval"] * n_point,
        summary_operators=[None] * n_point + ["mean"] * n_point,
        anchor_policies=[None] * n_point + ["support_end"] * n_point,
        observation_windows=[None] * n_point + [f"{max_window_len}step"] * n_point,
        support_start_times=support_start,
        support_end_times=support_end,
        interval_prev_coeffs=prev_coeffs,
        interval_curr_coeffs=curr_coeffs,
        interval_weights=weights,
        emission_slot_indices=emission_slots,
    )


def _build_scaled_mixed_support_observations(
    point_observations: jnp.ndarray,
    *,
    max_window_len: int,
) -> jnp.ndarray:
    point_np = np.asarray(point_observations)
    n_time, n_manifest = point_np.shape
    n_point = n_manifest // 2
    mixed = np.full_like(point_np, np.nan)
    mixed[:, :n_point] = point_np[:, :n_point]

    for anchor_idx in range(1, n_time):
        interval_manifest_idx = n_point + ((anchor_idx - 1) % n_point)
        start_idx = _window_start_index(anchor_idx, max_window_len)
        series = point_np[start_idx : anchor_idx + 1, interval_manifest_idx]
        mixed[anchor_idx, interval_manifest_idx] = _trapezoid_mean(series)

    return jnp.asarray(mixed, dtype=point_observations.dtype)


def _sample_student_t_noise(
    rng_key: jnp.ndarray,
    *,
    df: float,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    normal_key, gamma_key = random.split(rng_key)
    z = random.normal(normal_key, shape, dtype=dtype)
    chi2 = 2.0 * random.gamma(gamma_key, df / 2.0, shape=shape, dtype=dtype)
    return z * jnp.sqrt(jnp.asarray(df, dtype=dtype) / chi2)


def _simulate_mixed_continuous_observations(
    *,
    drift_diag: jnp.ndarray,
    diffusion_diag: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_scales: jnp.ndarray,
    manifest_dists: list[DistributionFamily],
    t0_sd: jnp.ndarray,
    times: jnp.ndarray,
    rng_key: jnp.ndarray,
    obs_df: float,
) -> jnp.ndarray:
    """Simulate continuous observations with mixed Gaussian and Student-t noise."""
    n_latent = int(drift_diag.shape[0])
    n_manifest = int(lambda_mat.shape[0])
    dt = float(times[1] - times[0]) if times.shape[0] > 1 else 1.0

    Ad, Qd, _ = discretize_system(
        jnp.diag(drift_diag),
        jnp.diag(diffusion_diag**2),
        None,
        dt,
    )
    qd_chol = jla.cholesky(Qd + jnp.eye(n_latent, dtype=times.dtype) * 1e-8, lower=True)

    rng_key, init_key = random.split(rng_key)
    states = [t0_sd * random.normal(init_key, (n_latent,), dtype=times.dtype)]
    for _ in range(times.shape[0] - 1):
        rng_key, state_key = random.split(rng_key)
        states.append(
            states[-1] @ Ad.T + qd_chol @ random.normal(state_key, (n_latent,), dtype=times.dtype)
        )
    latent = jnp.stack(states)
    means = latent @ lambda_mat.T

    student_mask = jnp.asarray(
        [dist == DistributionFamily.STUDENT_T for dist in manifest_dists],
        dtype=bool,
    )
    obs_keys = random.split(rng_key, times.shape[0])
    draws: list[jnp.ndarray] = []
    for obs_key, mean in zip(obs_keys, means, strict=False):
        gaussian_key, student_key = random.split(obs_key)
        gaussian_noise = random.normal(gaussian_key, (n_manifest,), dtype=times.dtype)
        student_noise = _sample_student_t_noise(
            student_key,
            df=obs_df,
            shape=(n_manifest,),
            dtype=times.dtype,
        )
        base_noise = jnp.where(student_mask, student_noise, gaussian_noise)
        draws.append(mean + manifest_scales * base_noise)
    return jnp.stack(draws)


def _build_synthetic_runtime():
    dtype_jax, dtype_np = _dtype_pair()
    n_latent = int(ARGS.n_latent)
    n_time = int(ARGS.target_interval_windows) + 1
    n_manifest = 2 * n_latent

    true_drift_diag = -jnp.linspace(0.18, 0.45, n_latent, dtype=dtype_jax)
    true_diff_diag = jnp.linspace(0.10, 0.18, n_latent, dtype=dtype_jax)
    point_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=dtype_jax)
    interval_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=dtype_jax)
    true_obs_scale = jnp.concatenate([point_obs_scale, interval_obs_scale])
    true_obs_df = 3.0
    true_t0_sd = jnp.linspace(0.20, 0.32, n_latent, dtype=dtype_jax)

    times = jnp.arange(n_time, dtype=dtype_jax)
    lambda_mat = jnp.concatenate(
        [
            jnp.eye(n_latent, dtype=dtype_jax),
            jnp.eye(n_latent, dtype=dtype_jax),
        ],
        axis=0,
    )
    manifest_dists = [DistributionFamily.STUDENT_T] * n_latent + [
        DistributionFamily.GAUSSIAN
    ] * n_latent
    manifest_names = [
        *(f"y{i}_point" for i in range(n_latent)),
        *(f"y{i}_interval" for i in range(n_latent)),
    ]

    point_observations = _simulate_mixed_continuous_observations(
        drift_diag=true_drift_diag,
        diffusion_diag=true_diff_diag,
        lambda_mat=lambda_mat,
        manifest_scales=true_obs_scale,
        manifest_dists=manifest_dists,
        t0_sd=true_t0_sd,
        times=times,
        rng_key=random.PRNGKey(ARGS.seed),
        obs_df=true_obs_df,
    )
    observations = _build_scaled_mixed_support_observations(
        point_observations,
        max_window_len=int(ARGS.max_window_len),
    )
    observation_support = _build_scaled_mixed_support_runtime(
        np.asarray(times, dtype=np.float64),
        manifest_names,
        max_window_len=int(ARGS.max_window_len),
    )

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_diag_mask=full_diagonal_mask(n_latent),
        drift_offdiag_mask=zero_square_mask(n_latent),
        drift=jnp.zeros((n_latent, n_latent), dtype=dtype_jax),
        cint_mask=zero_vector_mask(n_latent),
        cint=jnp.zeros(n_latent, dtype=dtype_jax),
        lambda_mask=zero_loading_mask(n_manifest, n_latent),
        lambda_mat=lambda_mat,
        diffusion_chol_mask=np.diag(full_diagonal_mask(n_latent)),
        diffusion_chol=jnp.eye(n_latent, dtype=dtype_jax),
        manifest_means_mask=zero_vector_mask(n_manifest),
        manifest_means=jnp.zeros(n_manifest, dtype=dtype_jax),
        manifest_chol_diag_mask=full_diagonal_mask(n_manifest),
        manifest_chol=jnp.zeros((n_manifest, n_manifest), dtype=dtype_jax),
        t0_means_mask=zero_vector_mask(n_latent),
        t0_means=jnp.zeros(n_latent, dtype=dtype_jax),
        t0_chol_diag_mask=zero_diagonal_mask(n_latent),
        t0_correlation_mask=zero_square_mask(n_latent),
        t0_chol=jnp.diag(true_t0_sd),
        latent_names=[f"x{i}" for i in range(n_latent)],
        manifest_names=manifest_names,
        manifest_dists=manifest_dists,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.35, "sigma": 0.15},
        diffusion_diag={"sigma": 0.15},
        manifest_var_diag={"sigma": 0.15},
    )

    model = SSMModel(spec, priors, likelihood="particle")
    model.set_observation_support(observation_support)

    observed_cells = int(np.sum(np.isfinite(np.asarray(observations))))
    total_cells = int(np.asarray(observations).size)
    LOGGER.info(
        "synthetic runtime built: timepoints=%d manifests=%d observed_cells=%d/%d dtype=%s",
        n_time,
        n_manifest,
        observed_cells,
        total_cells,
        dtype_np.__name__,
    )

    return SimpleNamespace(
        observations=observations,
        times=times,
        model=model,
        manifest_names=manifest_names,
        observation_support=observation_support,
        true_drift_diag=true_drift_diag,
        true_diff_diag=true_diff_diag,
        true_obs_scale=true_obs_scale,
        true_obs_df=true_obs_df,
    )


def _validate_support_summary(summary: dict[str, Any]) -> None:
    expected_bandwidth = int(ARGS.max_window_len) - 1
    checks = {
        "support_total_windows": int(summary["support_total_windows"]),
        "support_state_len_max": int(summary["support_state_len_max"]),
        "support_bandwidth": int(summary["support_bandwidth"]),
        "support_row_upper_bandwidth_max": int(summary["support_row_upper_bandwidth_max"]),
    }
    expected = {
        "support_total_windows": int(ARGS.target_interval_windows),
        "support_state_len_max": int(ARGS.max_window_len),
        "support_bandwidth": expected_bandwidth,
        "support_row_upper_bandwidth_max": expected_bandwidth,
    }
    if checks != expected:
        raise RuntimeError(
            f"Synthetic support summary mismatch. expected={expected} observed={checks}"
        )


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
        jax.profiler.trace(ARGS.trace_dir, create_perfetto_trace=True)
        if not ARGS.no_trace
        else contextlib.nullcontext()
    )

    try:
        with trace_ctx:
            runtime = _phase(
                "build_synthetic_runtime",
                summary,
                _build_synthetic_runtime,
            )
            observations = runtime.observations
            times = runtime.times
            model = runtime.model

            backend = _phase(
                "build_laplace_backend",
                summary,
                lambda: model.make_laplace_backend(ARGS.n_ieks_iters),
            )
            support_summary = _summarize_support(runtime, backend)
            _validate_support_summary(support_summary)
            summary["support_summary"] = support_summary
            LOGGER.info(
                "support summary: %s", json.dumps(_jsonable(support_summary), sort_keys=True)
            )

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
            LOGGER.info(
                "bundle ready parameter_dim=%d support_aware_outer=%s",
                dim,
                _requires_support_aware_outer_optimizer(model),
            )

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
                    return float(jax.device_get(objective)), _hostify_outer_eval_with_latent_mode(
                        aux
                    )

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
                            float(
                                np.linalg.norm(np.asarray(jax.device_get(grad), dtype=np.float64))
                            ),
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
