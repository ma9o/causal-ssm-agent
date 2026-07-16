"""Profiling harness for the MPGibbs latent smoothers — Modal GPU or local CPU.

Runs a scaling sweep (ms/step vs N and T) for the configured methods, which
*ranks* the wall-clock bottleneck:

  * dsmc ms/step ~16x when N quadruples  -> N^2 bridge bound (bandwidth/compute)
  * dsmc ms/step ~flat in N             -> kernel-launch bound
  * ms/step ~4x when T quadruples       -> T-linear (work/dispatch)
  * ms/step sublinear in T              -> span-bound (tree parallelism exploited)

The headline config also dumps a device-accurate HLO + cost-analysis +
access-pattern histogram and a jax.profiler trace.

Two execution targets, same core (`_run_benchmark`):

  GPU (Modal). Each config runs in its own GPU container (fan-out via .map), so
  the grid runs in parallel and any large-N OOM is isolated to its own container.
  The GPU is selected via the BENCHMARK_GPU env var (Modal binds resources at
  decoration time, so it cannot be a runtime CLI flag); it defaults to H100 (80GB):

    modal run evaluation/benchmarks/benchmark_gpu.py                     # H100 80GB (default)
    BENCHMARK_GPU=A100 modal run evaluation/benchmarks/benchmark_gpu.py
    BENCHMARK_GPU=H100 modal run evaluation/benchmarks/benchmark_gpu.py --headline-only
    BENCHMARK_GPU=H100 modal run evaluation/benchmarks/benchmark_gpu.py --headline-only --no-trace --no-compile-analysis
    BENCHMARK_GPU=H100 modal run evaluation/benchmarks/benchmark_gpu.py --headline-only --force-build

    modal volume get nof1-mpgibbs-prof /H100/dsmc_amala_exact_N512_T1024 ./h100_trace

  CPU (local, no Modal, no spend). The static rung — HLO, cost-analysis (FLOPs vs
  bytes => roofline), and the gather/dot/triangular-solve/while access-pattern
  histogram — is backend-portable and dumped at *compile* time (before the loop
  allocates), so it survives even if execution OOMs at the headline shape. The
  jax.profiler trace runs on CPU too (op *structure*, not GPU timings). This keeps
  the dev loop close: iterate the static analysis locally for free, reserve the
  GPU/ncu for confirming the one fusion the analysis points at.

    uv run python evaluation/benchmarks/benchmark_gpu.py --local                  # headline N512/T1024
    uv run python evaluation/benchmarks/benchmark_gpu.py --local --n 64 --t 128   # quick shape
    uv run python evaluation/benchmarks/benchmark_gpu.py --local --p 2 --no-trace # production P, analysis only

The model is the ``evaluation/fixtures/synthetic_nonlinear.py`` fixture (3 latent
states, mixed observation families), built with random init (no Pathfinder) —
numerics are irrelevant; only the compiled-kernel shapes/op-structure matter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Literal, NamedTuple

import modal

DEFAULT_GPU = "H100"
# Modal 1.3 binds gpu= at decoration time, so the card is chosen at import via env var
# (a main() CLI flag would be parsed too late to reach the @app.function spec).
GPU = os.environ.get("BENCHMARK_GPU", DEFAULT_GPU)
FORCE_BUILD = "--force-build" in sys.argv
# Direct `python benchmark_gpu.py --local` runs the headline config on local CPU
# (static HLO/cost/access-pattern + a CPU trace) with no Modal container or spend.
# Detected at import so we can skip building the (GPU) Modal image spec entirely.
LOCAL = "--local" in sys.argv


def _build_image() -> modal.Image:
    # Local-only: these dev-box paths and add_local_* references are valid when
    # launching, not inside the container (Modal re-imports this module there to
    # find the function). Build the spec only when local; the container's image is
    # already baked, so it just needs a placeholder that imports without touching FS.
    root = Path(__file__).resolve().parents[2]  # apps/data-pipeline/
    return (
        modal.Image.debian_slim(python_version="3.12", force_build=FORCE_BUILD)
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(root), groups=["dev", "cloud"], frozen=True)
        # CUDA 12 wheels (GPU-agnostic, one image for any card), pinned to uv.lock so the
        # resolve cannot drift JAX off 0.9.0.1 between benchmark runs.
        .uv_pip_install("jax[cuda12]==0.9.0.1")
        .env({"PYTHONPATH": "/root/src:/root"})
        .add_local_file(root / "config.yaml", remote_path="/root/config.yaml")
        .add_local_file(root / "pyproject.toml", remote_path="/root/pyproject.toml")
        .add_local_dir(root / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
        .add_local_dir(root / "evaluation", remote_path="/root/evaluation")
    )


# Local-CPU runs never dispatch to Modal, so skip the (slow) GPU image spec build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    if (LOCAL or not modal.is_local())
    else _build_image()
)
app = modal.App("nof1-mpgibbs-gpu-benchmark", image=image)
volume = modal.Volume.from_name("nof1-mpgibbs-prof", create_if_missing=True)


class MethodSpec(NamedTuple):
    smoother: Literal["dsmc"]
    leaf: Literal["amala_exact", "paid_mix"]
    block_coords: int
    trace: bool = False


class BenchmarkConfig(NamedTuple):
    smoother: Literal["dsmc"]
    leaf: Literal["amala_exact", "paid_mix"]
    block_coords: int
    n_particles: int
    t_steps: int
    trace: bool

    @property
    def tag(self) -> str:
        return (
            f"{self.smoother}/{self.leaf}/N{self.n_particles}/T{self.t_steps}/bc{self.block_coords}"
        )

    @property
    def profile_name(self) -> str:
        return f"{self.smoother}_{self.leaf}_N{self.n_particles}_T{self.t_steps}"


DEFAULT_BLOCK_COORDS = 256

# dsmc's pure-JAX (num_pairs, P, N, N) combine materialization can OOM on
# memory-tight cards. The default sweep targets the H100 80GB large-N regime.
N_GRID = (512,)
T_GRID = (1024,)
METHODS = (MethodSpec("dsmc", "amala_exact", DEFAULT_BLOCK_COORDS, trace=True),)


def _config(method: MethodSpec, n_particles: int, t_steps: int) -> BenchmarkConfig:
    return BenchmarkConfig(
        method.smoother,
        method.leaf,
        method.block_coords,
        n_particles,
        t_steps,
        method.trace,
    )


def _grid_configs(methods: tuple[MethodSpec, ...]) -> tuple[BenchmarkConfig, ...]:
    return tuple(
        _config(method, n_particles, t_steps)
        for method in methods
        for n_particles in N_GRID
        for t_steps in T_GRID
    )


def _max_grid_config(method: MethodSpec) -> BenchmarkConfig:
    return _config(method, max(N_GRID), max(T_GRID))


SWEEP_CONFIGS = _grid_configs(METHODS)
HEADLINE_METHOD = next(method for method in METHODS if method.trace)
HEADLINE_CONFIG = _max_grid_config(HEADLINE_METHOD)
NUM_PARAMETER_PARTICLES = 8
WARMUP_STEPS = 150
SAMPLE_STEPS = 150


def _run_benchmark(
    cfg: BenchmarkConfig,
    *,
    profile_dir: str | None,
    profile_compile_analysis: bool,
    profile_trace_start_step: int,
    profile_trace_steps: int,
    warmup_steps: int,
    sample_steps: int,
    num_parameter_particles: int,
) -> dict:
    """Build the fixture, time steady-state ms/step, dump profiling to ``profile_dir``.

    Backend-agnostic: the GPU worker (``run_config``) and the local-CPU ``__main__``
    path both call this. No Modal objects here — the caller resolves ``profile_dir``
    and commits any Volume. ``synthetic_nonlinear`` must be importable (callers put
    its directory on ``sys.path`` first); JAX picks the device from its environment.
    """
    import jax
    import jax.random as random
    from evaluation.fixtures.synthetic_nonlinear import (
        build_synthetic_nonlinear_model,
        simulate_synthetic_nonlinear_data,
    )

    from nof1_causal_lab.models.ssm.inference.bundle import build_particle_runtime_bundle
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
        build_marginal_particle_gibbs_kernel,
        run_marginal_particle_gibbs,
    )
    from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

    total_steps = warmup_steps + sample_steps
    print("JAX devices:", jax.devices(), "| config:", cfg.tag, flush=True)
    data = simulate_synthetic_nonlinear_data(T=cfg.t_steps, seed=71, diffusion_scale=1.0)
    model = build_synthetic_nonlinear_model(
        data, include_interval_support=False, diffusion_scale=1.0
    )
    bundle = build_particle_runtime_bundle(
        model,
        data.observations,
        data.times,
        scheme=LATENT_TRANSITION_EULER_MARUYAMA,
        trace_key=random.PRNGKey(0),
        reparam=None,
    )
    dim = int(bundle.cached.flat_example.shape[0])
    kernel = build_marginal_particle_gibbs_kernel(
        bundle,
        num_particles=cfg.n_particles,
        num_parameter_particles=num_parameter_particles,
        param_step_size=0.01,
        latent_smoother=cfg.smoother,
        dsmc_leaf_proposal=cfg.leaf,
        latent_block_coords=cfg.block_coords,
    )
    started = time.monotonic()
    run = run_marginal_particle_gibbs(
        bundle,
        kernel=kernel,
        num_warmup=warmup_steps,
        num_samples=sample_steps,
        num_chains=1,
        seed=0,
        adaptation_rate=0.0,
        init_scale=0.05,
        latent_delta=0.2,
        retain_latent_paths=False,
        compute_latent_posterior_summary=False,
        adaptation_scheme="simple",
        profile_dir=profile_dir,
        profile_compile_analysis=profile_compile_analysis,
        profile_runtime_trace=cfg.trace,
        profile_trace_start_step=profile_trace_start_step,
        profile_trace_steps=profile_trace_steps,
    )
    wall = time.monotonic() - started
    first = float(run["first_step_seconds"])
    loop = float(run["sampling_loop_seconds"])
    steady_ms = 1000.0 * (loop - first) / max(total_steps - 1, 1)
    row = {
        "smoother": cfg.smoother,
        "leaf": cfg.leaf,
        "block_coords": cfg.block_coords,
        "N": cfg.n_particles,
        "T": cfg.t_steps,
        "P": num_parameter_particles,
        "dim": dim,
        "compile_s": round(first, 3),
        "steady_ms_per_step": round(steady_ms, 3),
        "wall_s": round(wall, 2),
        "compile_analyzed": bool(profile_dir and profile_compile_analysis),
        "traced": bool(profile_dir and cfg.trace),
        "trace_steps": profile_trace_steps if profile_dir and cfg.trace else 0,
    }
    print("OK  ", cfg.tag, "->", row, flush=True)
    return row


# gpu=GPU is import-time (Modal 1.3 binds resources at decoration); select via the
# BENCHMARK_GPU env var. max_containers=10 stays within the account's concurrent-GPU cap;
# .map queues the rest. volumes= attaches the trace/HLO output volume (the headline
# config writes + commits here).
@app.function(
    gpu=GPU,
    timeout=1800,
    memory=32768,
    max_containers=10,
    volumes={"/prof": volume},
)
def run_config(
    cfg: BenchmarkConfig,
    gpu_tag: str,
    profile_compile_analysis: bool,
    profile_trace_start_step: int,
    profile_trace_steps: int,
    cuda_graphs: bool = False,
) -> dict:
    """Modal GPU worker: profile the headline config to the Volume; time the rest."""
    import os
    import traceback

    # CUDA graphs / command buffers: capture the per-step kernel sequence into
    # replayable graphs to amortize host launch overhead (the dispatch-bound regime).
    # Must be set before _run_benchmark imports jax so XLA reads it at backend init.
    if cuda_graphs:
        # FUSION command buffers (~10% once the dim-3 densities are precision-GEMMs;
        # a no-op before that, when cuSOLVER calls blocked capture). Measured: adding
        # CUBLAS,CUDNN now lowers without the prior "Unable to launch triangular solve"
        # crash but yields no further gain (190 vs 192 ms), and would re-crash if the
        # model regained many Gaussian solves — so FUSION is the robust default.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=FUSION"
        ).strip()

    is_headline = (
        cfg.smoother == HEADLINE_CONFIG.smoother
        and cfg.leaf == HEADLINE_CONFIG.leaf
        and cfg.block_coords == HEADLINE_CONFIG.block_coords
        and cfg.n_particles == HEADLINE_CONFIG.n_particles
        and cfg.t_steps == HEADLINE_CONFIG.t_steps
    )
    profile_dir = (
        f"/prof/{gpu_tag}/{cfg.profile_name}"
        if is_headline and (cfg.trace or profile_compile_analysis)
        else None
    )
    try:
        row = _run_benchmark(
            cfg,
            profile_dir=profile_dir,
            profile_compile_analysis=profile_compile_analysis,
            profile_trace_start_step=profile_trace_start_step,
            profile_trace_steps=profile_trace_steps,
            warmup_steps=WARMUP_STEPS,
            sample_steps=SAMPLE_STEPS,
            num_parameter_particles=NUM_PARAMETER_PARTICLES,
        )
        if profile_dir is not None:
            volume.commit()
        return row
    except Exception as exc:  # noqa: BLE001 — intentional: isolate per-config OOM/compile failure
        print("FAIL", cfg.tag, "->", traceback.format_exc(), flush=True)
        return {
            "smoother": cfg.smoother,
            "leaf": cfg.leaf,
            "block_coords": cfg.block_coords,
            "N": cfg.n_particles,
            "T": cfg.t_steps,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _result_ms(
    results: list[dict],
    method: MethodSpec,
    n_particles: int,
    t_steps: int,
) -> float | None:
    for result in results:
        if (
            result.get("smoother") == method.smoother
            and result.get("leaf") == method.leaf
            and result.get("block_coords") == method.block_coords
            and result.get("N") == n_particles
            and result.get("T") == t_steps
        ):
            return result.get("steady_ms_per_step")
    return None


def _scaling(results: list[dict], method: MethodSpec) -> None:
    lines = []
    if len(N_GRID) > 1:
        n_low = min(N_GRID)
        n_high = max(N_GRID)
        t_ref = max(T_GRID)
        low_ms = _result_ms(results, method, n_low, t_ref)
        high_ms = _result_ms(results, method, n_high, t_ref)
        if low_ms is not None and high_ms is not None:
            lines.append(
                f"    N {n_low}->{n_high} @T{t_ref}:  x{high_ms / low_ms:.1f}"
                "   (~quadratic => N^2-bound, ~1 => launch-bound)"
            )
    if len(T_GRID) > 1:
        t_low = min(T_GRID)
        t_high = max(T_GRID)
        n_ref = max(N_GRID)
        low_ms = _result_ms(results, method, n_ref, t_low)
        high_ms = _result_ms(results, method, n_ref, t_high)
        if low_ms is not None and high_ms is not None:
            lines.append(
                f"    T {t_low}->{t_high} @N{n_ref}: x{high_ms / low_ms:.1f}"
                "   (~linear => T-linear, ~1 => span-bound)"
            )
    if not lines:
        return
    print(f"\n  scaling for {method.smoother}({method.leaf}, bc={method.block_coords}):")
    for line in lines:
        print(line)


@app.local_entrypoint()
def main(
    headline_only: bool = False,
    comparison_only: bool = False,
    force_build: bool = False,
    trace: bool = True,
    compile_analysis: bool = True,
    trace_start_step: int = 0,
    trace_steps: int = 3,
    cuda_graphs: bool = False,
):
    # GPU is chosen via the BENCHMARK_GPU env var (see module docstring). --headline-only
    # re-runs just the trace target (1 GPU) once the grid is in hand.
    if force_build:
        print("Forcing Modal image rebuild for this run.")
    if comparison_only:
        configs = tuple(_max_grid_config(method._replace(trace=trace)) for method in METHODS)
    elif headline_only:
        configs = (_config(HEADLINE_METHOD._replace(trace=trace), max(N_GRID), max(T_GRID)),)
    else:
        configs = (
            tuple(
                _config(method._replace(trace=trace), n_particles, t_steps)
                for method in METHODS
                for n_particles in N_GRID
                for t_steps in T_GRID
            )
            if not trace
            else SWEEP_CONFIGS
        )
    # path-safe label (Modal allows multi-GPU "A100:2"); cuda-graphs runs get their
    # own namespace so they never collide with the baseline trace on the Volume.
    gpu_tag = GPU.replace(":", "x") + ("_cudagraphs" if cuda_graphs else "")
    results = list(
        run_config.map(
            configs,
            kwargs={
                "gpu_tag": gpu_tag,
                "profile_compile_analysis": compile_analysis,
                "profile_trace_start_step": trace_start_step,
                "profile_trace_steps": trace_steps,
                "cuda_graphs": cuda_graphs,
            },
        )
    )
    print(f"\n================ MPGibbs GPU benchmark [{GPU}] ================")
    hdr = (
        f"{'smoother':<11}{'leaf':<16}{'N':>5}{'T':>6}{'bc':>5}"
        f"{'dim':>5}{'compile_s':>11}{'ms/step':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(
                f"{r['smoother']:<11}{r['leaf']:<16}{r['N']:>5}{r['T']:>6}"
                f"{r['block_coords']:>5}{'':>5}  ERROR: {r['error']}"
            )
        else:
            print(
                f"{r['smoother']:<11}{r['leaf']:<16}{r['N']:>5}{r['T']:>6}"
                f"{r['block_coords']:>5}"
                f"{r['dim']:>5}{r['compile_s']:>11}{r['steady_ms_per_step']:>11}"
            )
    for method in METHODS:
        _scaling(results, method)
    headline_path = f"{gpu_tag}/{HEADLINE_CONFIG.profile_name}"
    if trace or compile_analysis:
        artifacts = []
        if trace:
            artifacts.append("Trace")
        if compile_analysis:
            artifacts.append("HLO")
        print(f"\n{' + '.join(artifacts)} for the headline config on Volume 'nof1-mpgibbs-prof':")
        print(f"  modal volume get nof1-mpgibbs-prof /{headline_path} ./{gpu_tag.lower()}_trace")
        if trace:
            print(
                f"  tensorboard --logdir ./{gpu_tag.lower()}_trace/run_loop   # Op Profile + Trace Viewer"
            )


def _run_local() -> int:
    """`python benchmark_gpu.py --local`: profile the headline config on local CPU.

    No Modal: JAX is pinned to CPU, ``synthetic_nonlinear`` is imported from this
    script's directory, and the HLO/cost/access-pattern/trace artifacts are written
    to a local ``--out`` directory instead of the Modal Volume.
    """
    import argparse
    import traceback

    parser = argparse.ArgumentParser(
        description="Local-CPU MPGibbs dsmc/amala_exact profiling (no Modal, no spend)."
    )
    parser.add_argument("--local", action="store_true", help="Required: run on local CPU.")
    parser.add_argument(
        "--n", type=int, default=max(N_GRID), help="num particles (default headline)"
    )
    parser.add_argument(
        "--t", type=int, default=max(T_GRID), help="num timesteps (default headline)"
    )
    parser.add_argument(
        "--p", type=int, default=NUM_PARAMETER_PARTICLES, help="num parameter particles (P)"
    )
    parser.add_argument(
        "--block-coords",
        type=int,
        default=DEFAULT_BLOCK_COORDS,
        help="number of latent coordinates proposed per update",
    )
    # Few steps: cost/HLO/access-patterns dump at compile time (before the loop), so a
    # couple of steps is enough to confirm execution + grab a short trace on CPU.
    parser.add_argument("--warmup", type=int, default=1, help="warmup steps (CPU: keep small)")
    parser.add_argument("--samples", type=int, default=1, help="sample steps (CPU: keep small)")
    parser.add_argument("--out", default="./prof", help="local artifact root (default ./prof)")
    parser.add_argument("--no-trace", action="store_true", help="skip the jax.profiler trace")
    parser.add_argument(
        "--no-compile-analysis", action="store_true", help="skip the HLO/cost/access dump"
    )
    parser.add_argument("--trace-start-step", type=int, default=0)
    parser.add_argument("--trace-steps", type=int, default=2)
    cli = parser.parse_args()
    if not cli.local:
        parser.error("Direct execution is local-only; pass --local (use `modal run ...` for GPU).")

    # JAX reads the platform at first import; set it before _run_benchmark imports jax.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    method = HEADLINE_METHOD
    cfg = BenchmarkConfig(
        method.smoother,
        method.leaf,
        cli.block_coords,
        cli.n,
        cli.t,
        not cli.no_trace,
    )
    profile_dir = f"{cli.out.rstrip('/')}/cpu/{cfg.profile_name}"
    print(
        f"[local-cpu] {cfg.tag}  P={cli.p}  warmup={cli.warmup} samples={cli.samples}"
        f"  -> {profile_dir}",
        flush=True,
    )
    try:
        _run_benchmark(
            cfg,
            profile_dir=profile_dir,
            profile_compile_analysis=not cli.no_compile_analysis,
            profile_trace_start_step=cli.trace_start_step,
            profile_trace_steps=cli.trace_steps,
            warmup_steps=cli.warmup,
            sample_steps=cli.samples,
            num_parameter_particles=cli.p,
        )
    except Exception:  # noqa: BLE001 — report, but the pre-loop static dump is already on disk
        traceback.print_exc()
        print(
            f"\nExecution failed (likely OOM at this shape). The compile-time static "
            f"analysis, if reached, was written to {profile_dir}/ before the loop ran.",
            flush=True,
        )
        return 1
    print(f"\nLocal CPU artifacts in {profile_dir}/:")
    if not cli.no_compile_analysis:
        print(
            "  run_batched_step.cost.json            # aggregate FLOPs / bytes-accessed (roofline)"
        )
        print(
            "  run_batched_step.hlo.txt              # optimized HLO (fusion clusters; (P,N,N) materialize)"
        )
        print(
            "  run_batched_step.access_patterns.json # gather/dot/triangular-solve/while histogram"
        )
    if not cli.no_trace:
        print(
            f"  tensorboard --logdir {profile_dir}/run_loop   # CPU op timeline (structure, not GPU timings)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_local())
