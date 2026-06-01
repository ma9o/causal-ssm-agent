"""Modal GPU profiling harness for the MPGibbs latent smoothers.

Runs a scaling sweep (ms/step vs N and T) for `plain` and `dsmc(amala_plus leaf)`
on a Modal GPU of your choice, which *ranks* the wall-clock bottleneck:

  * dsmc ms/step ~16x when N quadruples  -> N^2 bridge bound (bandwidth/compute)
  * dsmc ms/step ~flat in N             -> kernel-launch bound
  * ms/step ~4x when T quadruples       -> T-linear (work/dispatch)
  * ms/step sublinear in T              -> span-bound (tree parallelism exploited)

Each config runs in its own GPU container (fan-out via .map), so the grid runs in
parallel and any large-N OOM is isolated to its own container. The headline config
(dsmc+amala_plus, N=512, T=1024) also dumps a device-accurate HLO + cost-analysis
+ access-pattern histogram and a jax.profiler trace onto a Modal Volume, namespaced
per GPU so traces from different cards do not collide.

The GPU is selected via the BENCHMARK_GPU env var (Modal binds resources at decoration
time, so it cannot be a runtime CLI flag). It defaults to L4; the image is
GPU-agnostic and built once, reused across cards:

  modal run scripts/benchmarks/benchmark_gpu.py                     # L4 (default)
  BENCHMARK_GPU=A100 modal run scripts/benchmarks/benchmark_gpu.py
  BENCHMARK_GPU=H100 modal run scripts/benchmarks/benchmark_gpu.py --comparison-only
  BENCHMARK_GPU=H100 modal run scripts/benchmarks/benchmark_gpu.py --headline-only

  modal volume get nof1-mpgibbs-prof /A100/dsmc_amala_plus_N512_T1024 ./a100_trace

The model is the ``scripts/benchmarks/synthetic_nonlinear.py`` fixture (3 latent
states, mixed observation families), built with random init (no Pathfinder) —
numerics are irrelevant; only the compiled-kernel shapes/op-structure matter.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

DEFAULT_GPU = "L4"
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
        # resolve can't drift jax off 0.9.0.1 — the Triton combine kernel needs backend=.
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

# (latent_smoother, dsmc_leaf_proposal, latent_block_size)
PLAIN = ("plain", "prior_predictive", 256)
DSMC_AMALA = ("dsmc", "amala_plus", 256)
DSMC_PRIOR = ("dsmc", "prior_predictive", 256)
AMALA_PLUS = ("amala_plus", "prior_predictive", 16)

# dsmc's (num_pairs, N, N, P) level-0 materialization makes pure-JAX N=512 OOM on
# memory-tight cards (e.g. L4 24GB; fine on A100/H100/B200) — the reason the Triton
# kernel exists (per project_dsmc_scaling_regime). The grid centers on memory-safe N;
# N=512 is a single large-N probe (expected OOM on small cards, ~300ms on big ones).
N_GRID = (64, 128, 256)
T_GRID = (256, 1024)
SWEEP: list[tuple] = (
    [(*PLAIN, n, t) for n in N_GRID for t in T_GRID]
    + [(*DSMC_AMALA, n, t) for n in N_GRID for t in T_GRID]
    + [
        (*DSMC_AMALA, 512, 1024),  # large-N probe: OOM on <=24GB cards, ~300ms on >=40GB
        (*DSMC_PRIOR, 256, 1024),  # leaf-choice control (leaf is cheap; tree dominates)
        (*AMALA_PLUS, 128, 256),
        (*AMALA_PLUS, 128, 1024),
    ]
)
COMPARISON = [
    (*PLAIN, 512, 1024),
    (*DSMC_AMALA, 512, 1024),
]

HEADLINE = (
    "dsmc",
    "amala_plus",
    256,
    512,
    1024,
)  # trace + HLO dump target; large-N production config
NUM_PARAMETER_PARTICLES = 8
WARMUP_STEPS = 1
SAMPLE_STEPS = 20


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
def run_config(cfg: tuple, gpu_tag: str) -> dict:
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

    smoother, leaf, block, n_particles, t_steps = cfg
    tag = f"{smoother}/{leaf}/N{n_particles}/T{t_steps}/bs{block}"
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
        profile_dir = (
            f"/prof/{gpu_tag}/{smoother}_{leaf}_N{n_particles}_T{t_steps}"
            if cfg == HEADLINE
            else None
        )
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


def _scaling(results: list[dict], smoother: str, leaf: str) -> None:
    def ms(n, t):
        for r in results:
            if (
                r.get("smoother") == smoother
                and r.get("leaf") == leaf
                and r.get("N") == n
                and r.get("T") == t
            ):
                return r.get("steady_ms_per_step")
        return None

    print(f"\n  scaling for {smoother}({leaf}):")
    if ms(64, 1024) and ms(256, 1024):
        print(
            f"    N 64->256 @T1024:  x{ms(256, 1024) / ms(64, 1024):.1f}"
            "   (~16 => N^2-bound, ~1 => launch-bound)"
        )
    if ms(256, 256) and ms(256, 1024):
        print(
            f"    T 256->1024 @N256: x{ms(256, 1024) / ms(256, 256):.1f}"
            "   (~4 => T-linear, ~1 => span-bound)"
        )


@app.local_entrypoint()
def main(headline_only: bool = False, comparison_only: bool = False):
    # GPU is chosen via the BENCHMARK_GPU env var (see module docstring). --headline-only
    # re-runs just the trace target (1 GPU) once the grid is in hand.
    if comparison_only:
        configs = COMPARISON
    elif headline_only:
        configs = [HEADLINE]
    else:
        configs = SWEEP
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
    _scaling(results, "plain", "prior_predictive")
    _scaling(results, "dsmc", "amala_plus")
    headline_path = f"{gpu_tag}/dsmc_amala_plus_N{HEADLINE[3]}_T{HEADLINE[4]}"
    print("\nTrace + HLO for the headline config on Volume 'nof1-mpgibbs-prof':")
    print(f"  modal volume get nof1-mpgibbs-prof /{headline_path} ./{gpu_tag.lower()}_trace")
    print(
        f"  tensorboard --logdir ./{gpu_tag.lower()}_trace/run_loop   # Op Profile + Trace Viewer"
    )
