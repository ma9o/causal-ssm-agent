"""Profile Laplace-EM inference with JAX Perfetto traces.

Usage:
    uv run python tools/profile_laplace_em_perfetto.py
    uv run python tools/profile_laplace_em_perfetto.py --run-dir ../../data/.private/SMALLGOLDEN/run
    uv run python tools/profile_laplace_em_perfetto.py --warmup-runs 0 --n-outer 2 --n-csmc-particles 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import polars as pl

from causal_ssm_agent.models.ssm_builder import (
    PreparedModelRuntime,
    prepare_model_runtime,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm import InferenceResult

ROOT = Path(__file__).parent.parent
REPO_ROOT = ROOT.parent.parent
DEFAULT_TRACE_ROOT = ROOT / "profiles" / "laplace-em"
DEFAULT_GOLDEN_RUN_DIR = REPO_ROOT / "data" / ".private" / "GOLDEN" / "run"


@dataclass(frozen=True)
class ProfilingWorkload:
    """Prepared runtime plus a short source description for profile metadata."""

    name: str
    source: str
    runtime: PreparedModelRuntime


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_default_compiled_data_path(run_dir: Path) -> Path:
    for candidate in ("stage2-model-data.parquet", "stage2-raw-data.parquet"):
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find stage2-model-data.parquet or stage2-raw-data.parquet under {run_dir}."
    )


def _build_compiled_run_runtime(
    run_dir: Path,
    data_path: Path | None,
    sampler_config: dict[str, Any],
) -> ProfilingWorkload:
    compiled_path = run_dir / "stage4-compiled-ssm.json"
    if not compiled_path.exists():
        raise FileNotFoundError(f"Missing compiled artifact: {compiled_path}")

    resolved_data_path = (
        data_path if data_path is not None else _find_default_compiled_data_path(run_dir)
    )
    compiled_ssm = _load_json(compiled_path)
    data_for_model = pl.read_parquet(resolved_data_path)
    runtime = prepare_model_runtime(
        data_for_model=data_for_model,
        compiled_ssm=compiled_ssm,
        sampler_config=sampler_config,
    )
    workload_name = run_dir.parent.name.lower()
    return ProfilingWorkload(
        name=workload_name,
        source=str(run_dir),
        runtime=runtime,
    )


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if callable(block_until_ready):
            block_until_ready()


def _run_fit(runtime: PreparedModelRuntime) -> tuple[InferenceResult, float]:
    started_at = time.perf_counter()
    result = runtime.builder.fit_prepared(runtime.observations, runtime.times)
    _block_tree(result.get_samples())
    _block_tree(result.diagnostics)
    return result, time.perf_counter() - started_at


def _find_perfetto_trace(trace_dir: Path) -> Path | None:
    traces = sorted(trace_dir.rglob("perfetto_trace.json.gz"))
    return traces[-1] if traces else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"shape": list(shape), "dtype": str(dtype)}
    return repr(value)


def _build_trace_dir(trace_root: Path, trace_name: str | None, workload_name: str) -> Path:
    if trace_name:
        trace_dir = trace_root / trace_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_dir = trace_root / f"{timestamp}-{workload_name}"
    trace_dir.mkdir(parents=True, exist_ok=False)
    return trace_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Laplace-EM inference and emit a Perfetto trace."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_GOLDEN_RUN_DIR,
        help="Compiled run directory to profile. Defaults to data/.private/GOLDEN/run.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override parquet input for the compiled run directory.",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="Directory under which the trace output directory will be created.",
    )
    parser.add_argument(
        "--trace-name",
        help="Optional explicit subdirectory name under --trace-root.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of unprofiled warmup runs to perform before tracing. Use 0 to include compilation.",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Number of profiled runs to record in the trace.",
    )
    parser.add_argument(
        "--create-perfetto-link",
        action="store_true",
        help="Ask JAX to emit a shareable Perfetto link in addition to local files.",
    )
    parser.add_argument(
        "--skip-memory-profile",
        action="store_true",
        help="Do not write device_memory_profile.pb after the profiled run.",
    )
    parser.add_argument("--n-outer", type=int, default=6)
    parser.add_argument("--n-csmc-particles", type=int, default=8)
    parser.add_argument("--n-mh-steps", type=int, default=3)
    parser.add_argument("--param-step-size", type=float, default=0.05)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-ieks-iters", type=int, default=3)
    parser.add_argument("--n-leapfrog", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adaptive-tempering",
        action="store_true",
        help="Use ESS-adaptive tempering instead of a fixed ladder.",
    )
    parser.add_argument("--target-ess-ratio", type=float, default=0.5)
    parser.add_argument(
        "--waste-free",
        action="store_true",
        help="Enable waste-free recycling. Requires n_csmc_particles %% n_mh_steps == 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler_config = {
        "method": "laplace_em",
        "n_outer": args.n_outer,
        "n_csmc_particles": args.n_csmc_particles,
        "n_mh_steps": args.n_mh_steps,
        "param_step_size": args.param_step_size,
        "n_warmup": args.n_warmup,
        "n_ieks_iters": args.n_ieks_iters,
        "n_leapfrog": args.n_leapfrog,
        "adaptive_tempering": args.adaptive_tempering,
        "target_ess_ratio": args.target_ess_ratio,
        "waste_free": args.waste_free,
        "seed": args.seed,
    }

    workload = _build_compiled_run_runtime(args.run_dir, args.data_path, sampler_config)

    trace_dir = _build_trace_dir(args.trace_root, args.trace_name, workload.name)
    print(f"JAX {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    print(f"Workload: {workload.name} ({workload.source})")
    print(f"Trace dir: {trace_dir}")
    print(
        "Prepared runtime: "
        f"obs_shape={tuple(workload.runtime.observations.shape)} "
        f"time_shape={tuple(workload.runtime.times.shape)} "
        f"wide_shape={workload.runtime.wide_data.shape}"
    )
    print(f"Sampler config: {json.dumps(sampler_config, indent=2)}")

    warmup_elapsed: list[float] = []
    for run_idx in range(args.warmup_runs):
        print(f"Warmup {run_idx + 1}/{args.warmup_runs}...")
        _result, elapsed = _run_fit(workload.runtime)
        warmup_elapsed.append(elapsed)
        print(f"  warmup elapsed={elapsed:.2f}s")

    profiled_elapsed: list[float] = []
    result: InferenceResult | None = None
    with jax.profiler.trace(
        trace_dir,
        create_perfetto_link=args.create_perfetto_link,
        create_perfetto_trace=True,
    ):
        for run_idx in range(args.profile_runs):
            with jax.profiler.StepTraceAnnotation(
                "laplace_em_profile/profiled_fit",
                run=run_idx + 1,
            ):
                print(f"Profiled run {run_idx + 1}/{args.profile_runs}...")
                result, elapsed = _run_fit(workload.runtime)
                profiled_elapsed.append(elapsed)
                print(f"  profiled elapsed={elapsed:.2f}s")

    if result is None:
        raise RuntimeError("No profiled run completed.")

    perfetto_trace = _find_perfetto_trace(trace_dir)
    memory_profile_path = trace_dir / "device_memory_profile.pb"
    if not args.skip_memory_profile:
        try:
            jax.profiler.save_device_memory_profile(memory_profile_path)
        except Exception as exc:
            print(f"Device memory profile failed: {exc}")
            memory_profile_path = None
    else:
        memory_profile_path = None

    samples = result.get_samples()
    metadata = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "workload": {
            "name": workload.name,
            "source": workload.source,
            "manifest_names": workload.runtime.manifest_names,
            "wide_shape": list(workload.runtime.wide_data.shape),
            "observations_shape": list(workload.runtime.observations.shape),
            "times_shape": list(workload.runtime.times.shape),
        },
        "sampler_config": sampler_config,
        "warmup_runs": args.warmup_runs,
        "profile_runs": args.profile_runs,
        "warmup_elapsed_seconds": warmup_elapsed,
        "profiled_elapsed_seconds": profiled_elapsed,
        "trace_dir": str(trace_dir),
        "perfetto_trace": str(perfetto_trace) if perfetto_trace is not None else None,
        "memory_profile": str(memory_profile_path) if memory_profile_path is not None else None,
        "result": {
            "method": result.method,
            "sample_shapes": {name: list(value.shape) for name, value in samples.items()},
            "diagnostics": _json_safe(result.diagnostics),
        },
    }
    metadata_path = trace_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Profile complete.")
    if perfetto_trace is not None:
        print(f"Perfetto trace: {perfetto_trace}")
    else:
        print("Perfetto trace file was not found under the trace directory.")
    if memory_profile_path is not None:
        print(f"Device memory profile: {memory_profile_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
