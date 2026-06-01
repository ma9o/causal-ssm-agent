"""Modal GPU profiling harness for the MPGibbs latent smoothers.

Runs a scaling sweep (ms/step vs N and T) for the configured methods on a Modal
GPU of your choice, which *ranks* the wall-clock bottleneck:

  * dsmc ms/step ~16x when N quadruples  -> N^2 bridge bound (bandwidth/compute)
  * dsmc ms/step ~flat in N             -> kernel-launch bound
  * ms/step ~4x when T quadruples       -> T-linear (work/dispatch)
  * ms/step sublinear in T              -> span-bound (tree parallelism exploited)

Each config runs in its own GPU container (fan-out via .map), so the grid runs in
parallel and any large-N OOM is isolated to its own container. The headline method
at max(N_GRID) and max(T_GRID) also dumps a device-accurate HLO + cost-analysis +
access-pattern histogram and a jax.profiler trace onto a Modal Volume, namespaced
per GPU so traces from different cards do not collide.

The GPU is selected via the BENCHMARK_GPU env var (Modal binds resources at decoration
time, so it cannot be a runtime CLI flag). It defaults to H100 (80GB); the image is
GPU-agnostic and built once, reused across cards:

  modal run scripts/benchmarks/benchmark_gpu.py                     # H100 80GB (default)
  BENCHMARK_GPU=A100 modal run scripts/benchmarks/benchmark_gpu.py
  BENCHMARK_GPU=H100 modal run scripts/benchmarks/benchmark_gpu.py --comparison-only
  BENCHMARK_GPU=H100 modal run scripts/benchmarks/benchmark_gpu.py --headline-only

  modal volume get nof1-mpgibbs-prof /H100/dsmc_amala_plus_N512_T1024 ./h100_trace

The model is the ``scripts/benchmarks/synthetic_nonlinear.py`` fixture (3 latent
states, mixed observation families), built with random init (no Pathfinder) —
numerics are irrelevant; only the compiled-kernel shapes/op-structure matter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import modal

DEFAULT_GPU = "H100"
# Modal 1.3 binds gpu= at decoration time, so the card is chosen at import via env var
# (a main() CLI flag would be parsed too late to reach the @app.function spec).
GPU = os.environ.get("BENCHMARK_GPU", DEFAULT_GPU)


def _build_image() -> modal.Image:
    # Local-only: these dev-box paths and add_local_* references are valid when
    # launching, not inside the container (Modal re-imports this module there to
    # find the function). Build the spec only when local; the container's image is
    # already baked, so it just needs a placeholder that imports without touching FS.
    root = Path(__file__).resolve().parents[2]  # apps/data-pipeline/
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(root), groups=["dev", "cloud"], frozen=True)
        # CUDA 12 wheels (GPU-agnostic, one image for any card), pinned to uv.lock so the
        # resolve cannot drift JAX off 0.9.0.1 between benchmark runs.
        .uv_pip_install("jax[cuda12]==0.9.0.1")
        .env({"PYTHONPATH": "/root/src"})
        .add_local_file(root / "config.yaml", remote_path="/root/config.yaml")
        .add_local_file(root / "pyproject.toml", remote_path="/root/pyproject.toml")
        .add_local_dir(root / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
        .add_local_file(
            root / "scripts" / "benchmarks" / "synthetic_nonlinear.py",
            remote_path="/root/scripts/synthetic_nonlinear.py",
        )
    )


image = _build_image() if modal.is_local() else modal.Image.debian_slim(python_version="3.12")
app = modal.App("nof1-mpgibbs-gpu-benchmark", image=image)
volume = modal.Volume.from_name("nof1-mpgibbs-prof", create_if_missing=True)


class MethodSpec(NamedTuple):
    smoother: str
    leaf: str
    block_size: int
    trace: bool = False


class BenchmarkConfig(NamedTuple):
    smoother: str
    leaf: str
    block_size: int
    n_particles: int
    t_steps: int

    @property
    def tag(self) -> str:
        return (
            f"{self.smoother}/{self.leaf}/N{self.n_particles}/T{self.t_steps}/bs{self.block_size}"
        )

    @property
    def profile_name(self) -> str:
        return f"{self.smoother}_{self.leaf}_N{self.n_particles}_T{self.t_steps}"


DEFAULT_BLOCK_SIZE = 256

# dsmc's pure-JAX (num_pairs, P, N, N) combine materialization can OOM on
# memory-tight cards. The default sweep targets the H100 80GB large-N regime.
N_GRID = (512,)
T_GRID = (1024,)
METHODS = (MethodSpec("dsmc", "amala_plus", DEFAULT_BLOCK_SIZE, trace=True),)


def _config(method: MethodSpec, n_particles: int, t_steps: int) -> BenchmarkConfig:
    return BenchmarkConfig(
        method.smoother,
        method.leaf,
        method.block_size,
        n_particles,
        t_steps,
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
COMPARISON_CONFIGS = tuple(_max_grid_config(method) for method in METHODS)
HEADLINE_CONFIG = _max_grid_config(next(method for method in METHODS if method.trace))
NUM_PARAMETER_PARTICLES = 8
WARMUP_STEPS = 150
SAMPLE_STEPS = 150


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
def run_config(cfg: BenchmarkConfig, gpu_tag: str) -> dict:
    """Build the fixture, time steady-state ms/step for one config; trace if headline."""
    import sys
    import time
    import traceback

    sys.path.insert(0, "/root/scripts")

    import jax
    import jax.random as random
    from synthetic_nonlinear import (
        build_synthetic_nonlinear_model,
        simulate_synthetic_nonlinear_data,
    )

    from nof1_causal_lab.models.ssm.inference.bundle import build_auxiliary_kalman_bundle
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
        build_marginal_particle_gibbs_kernel,
        run_marginal_particle_gibbs,
    )

    smoother = cfg.smoother
    leaf = cfg.leaf
    block = cfg.block_size
    n_particles = cfg.n_particles
    t_steps = cfg.t_steps
    tag = cfg.tag
    total_steps = WARMUP_STEPS + SAMPLE_STEPS
    try:
        print("JAX devices:", jax.devices(), "| config:", tag, flush=True)
        data = simulate_synthetic_nonlinear_data(T=t_steps, seed=71, diffusion_scale=1.0)
        model = build_synthetic_nonlinear_model(
            data, include_interval_support=False, diffusion_scale=1.0
        )
        bundle = build_auxiliary_kalman_bundle(
            model,
            data.observations,
            data.times,
            trace_key=random.PRNGKey(0),
            reparam=None,
            polya_gamma_num_terms=64,
            polya_gamma_sampler="truncated_sum",
            enable_polya_gamma=False,
            rbpf_mode="none",
            rbpf_marginalized_latent_indices=None,
        )
        dim = int(bundle["flat_example"].shape[0])
        kernel = build_marginal_particle_gibbs_kernel(
            bundle,
            num_particles=n_particles,
            num_parameter_particles=NUM_PARAMETER_PARTICLES,
            param_step_size=0.01,
            latent_smoother=smoother,
            dsmc_leaf_proposal=leaf,
            latent_block_size=block,
        )
        profile_dir = f"/prof/{gpu_tag}/{cfg.profile_name}" if cfg == HEADLINE_CONFIG else None
        started = time.monotonic()
        run = run_marginal_particle_gibbs(
            bundle,
            kernel=kernel,
            num_warmup=WARMUP_STEPS,
            num_samples=SAMPLE_STEPS,
            num_chains=1,
            seed=0,
            adaptation_rate=0.0,
            init_scale=0.05,
            latent_delta=0.2,
            retain_latent_paths=False,
            compute_latent_posterior_summary=False,
            adaptation_scheme="simple",
            profile_dir=profile_dir,
        )
        wall = time.monotonic() - started
        first = float(run["first_step_seconds"])
        loop = float(run["sampling_loop_seconds"])
        steady_ms = 1000.0 * (loop - first) / max(total_steps - 1, 1)
        if profile_dir is not None:
            volume.commit()
        row = {
            "smoother": smoother,
            "leaf": leaf,
            "block": block,
            "N": n_particles,
            "T": t_steps,
            "P": NUM_PARAMETER_PARTICLES,
            "dim": dim,
            "compile_s": round(first, 3),
            "steady_ms_per_step": round(steady_ms, 3),
            "wall_s": round(wall, 2),
            "traced": bool(profile_dir),
        }
        print("OK  ", tag, "->", row, flush=True)
        return row
    except Exception as exc:  # noqa: BLE001 — intentional: isolate per-config OOM/compile failure
        print("FAIL", tag, "->", traceback.format_exc(), flush=True)
        return {
            "smoother": smoother,
            "leaf": leaf,
            "block": block,
            "N": n_particles,
            "T": t_steps,
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
            and result.get("block") == method.block_size
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
    print(f"\n  scaling for {method.smoother}({method.leaf}, bs={method.block_size}):")
    for line in lines:
        print(line)


@app.local_entrypoint()
def main(headline_only: bool = False, comparison_only: bool = False):
    # GPU is chosen via the BENCHMARK_GPU env var (see module docstring). --headline-only
    # re-runs just the trace target (1 GPU) once the grid is in hand.
    if comparison_only:
        configs = COMPARISON_CONFIGS
    elif headline_only:
        configs = (HEADLINE_CONFIG,)
    else:
        configs = SWEEP_CONFIGS
    gpu_tag = GPU.replace(":", "x")  # path-safe label (Modal allows multi-GPU "A100:2")
    results = list(run_config.map(configs, kwargs={"gpu_tag": gpu_tag}))
    print(f"\n================ MPGibbs GPU benchmark [{GPU}] ================")
    hdr = (
        f"{'smoother':<11}{'leaf':<16}{'N':>5}{'T':>6}{'bs':>5}"
        f"{'dim':>5}{'compile_s':>11}{'ms/step':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(
                f"{r['smoother']:<11}{r['leaf']:<16}{r['N']:>5}{r['T']:>6}"
                f"{r['block']:>5}{'':>5}  ERROR: {r['error']}"
            )
        else:
            print(
                f"{r['smoother']:<11}{r['leaf']:<16}{r['N']:>5}{r['T']:>6}{r['block']:>5}"
                f"{r['dim']:>5}{r['compile_s']:>11}{r['steady_ms_per_step']:>11}"
            )
    for method in METHODS:
        _scaling(results, method)
    headline_path = f"{gpu_tag}/{HEADLINE_CONFIG.profile_name}"
    print("\nTrace + HLO for the headline config on Volume 'nof1-mpgibbs-prof':")
    print(f"  modal volume get nof1-mpgibbs-prof /{headline_path} ./{gpu_tag.lower()}_trace")
    print(
        f"  tensorboard --logdir ./{gpu_tag.lower()}_trace/run_loop   # Op Profile + Trace Viewer"
    )
