"""Modal H100 parameter-recovery A/B for the dsmc MPGibbs sampler: amala_exact vs amala_plus.

Runs benchmark.py's *actual* recovery flow (``benchmark.main()``) inside a Modal H100
container — same code path that passes on CPU, just GPU-accelerated. The local
benchmark.py is CPU-only (~1.6 s/step), so a 5k/5k (10k-step) chain is ~4.5 h on CPU but
~15 min on an H100 with the optimized sampler. We drive it via argv and read back its
JSON artifact, so there's no config drift vs the CPU run (released sites, pathfinder,
recovery extraction are all benchmark.py's own).

  modal run scripts/benchmarks/benchmark_recovery_gpu.py --num-warmup 5000 --num-samples 5000
  modal run scripts/benchmarks/benchmark_recovery_gpu.py --num-warmup 500 --num-samples 500   # quick cross-check vs CPU

The prebuilt Pathfinder cache (scratchpad/...pathfinder_cache.npz) is shipped into the
image and reused (--pathfinder-cache-mode reuse), saving the ~8 min pathfinder recompute.
The cache is keyed to the default config (T=1000, seed=1009, maxiter=50, starts=8, ...);
the argv below pins every cache-config field so benchmark.py's exact-match validation
passes. Running with a different --t/--seed/--pathfinder-* won't match the shipped cache
(benchmark.py raises with a clear message; regenerate the cache or use refresh).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("BENCHMARK_GPU", "H100")
FORCE_BUILD = "--force-build" in sys.argv
PATHFINDER_CACHE_NAME = "mpg_smoothers_synthetic_nonlinear_pathfinder_cache.npz"


def _build_image() -> modal.Image:
    root = Path(__file__).resolve().parents[2]  # apps/data-pipeline/
    image = (
        modal.Image.debian_slim(python_version="3.12", force_build=FORCE_BUILD)
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(root), groups=["dev", "cloud"], frozen=True)
        .uv_pip_install("jax[cuda12]==0.9.0.1")
        .env({"PYTHONPATH": "/root/src"})
        .add_local_file(root / "config.yaml", remote_path="/root/config.yaml")
        .add_local_file(root / "pyproject.toml", remote_path="/root/pyproject.toml")
        .add_local_dir(root / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
        .add_local_file(
            root / "scripts" / "benchmarks" / "synthetic_nonlinear.py",
            remote_path="/root/scripts/synthetic_nonlinear.py",
        )
        .add_local_file(
            root / "scripts" / "benchmarks" / "benchmark.py",
            remote_path="/root/scripts/benchmark.py",
        )
    )
    # Ship the prebuilt Pathfinder cache so the GPU run reuses it instead of
    # recomputing (~8 min). root.parents[1] is the worktree root (REPO_ROOT),
    # whose scratchpad/ holds the cache benchmark.py reads at /root/scratchpad.
    cache_src = root.parents[1] / "scratchpad" / PATHFINDER_CACHE_NAME
    if cache_src.exists():
        image = image.add_local_file(
            cache_src, remote_path=f"/root/scratchpad/{PATHFINDER_CACHE_NAME}"
        )
    return image


image = _build_image() if modal.is_local() else modal.Image.debian_slim(python_version="3.12")
app = modal.App("nof1-mpgibbs-recovery", image=image)


@app.function(gpu=GPU, timeout=5400, memory=65536)
def run_recovery(
    num_warmup: int,
    num_samples: int,
    t_steps: int,
    n_particles: int,
    n_parameter_particles: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    seed: int,
) -> dict:
    """Drive benchmark.main() for amala_exact + amala_plus (A/B) on H100, return its JSON."""
    import sys

    sys.path.insert(0, "/root/scripts")
    # benchmark.py resolves a repo root by walking up for apps/data-pipeline/pyproject.toml
    # and writes its JSON + pathfinder cache under <root>/scratchpad. Provide a dummy marker
    # so it resolves to /root in the container layout.
    Path("/root/apps/data-pipeline").mkdir(parents=True, exist_ok=True)
    Path("/root/apps/data-pipeline/pyproject.toml").touch()
    Path("/root/scratchpad").mkdir(parents=True, exist_ok=True)

    import jax

    print("JAX devices:", jax.devices(), flush=True)
    import benchmark

    out_path = Path("/root/scratchpad/mpg_smoothers_synthetic_nonlinear_check.json")
    sys.argv = [
        "benchmark.py",
        "--smoothers",
        "amala_exact,amala_plus",
        "--num-warmup",
        str(num_warmup),
        "--num-samples",
        str(num_samples),
        "--T",
        str(t_steps),
        "--n-particles",
        str(n_particles),
        "--n-parameter-particles",
        str(n_parameter_particles),
        "--pathfinder-maxiter",
        str(pathfinder_maxiter),
        "--n-pathfinder-starts",
        str(n_pathfinder_starts),
        "--seed",
        str(seed),
        # Pin every cache-config field to the shipped Pathfinder cache so
        # benchmark.py's exact-match validation reuses it instead of recomputing.
        "--data-seed",
        "71",
        "--diffusion-scale",
        "1.0",
        "--num-chains",
        "1",
        "--pathfinder-num-elbo-samples",
        "20",
        "--pathfinder-init-scale",
        "0.1",
        "--pathfinder-cache-mode",
        "reuse",
        # Cache positions are safe again: the v2 cache ships the IEKS smoothed
        # path per chain as the data-conditioned reference-path init, so the
        # explosive unconditional predictive simulation at the pathfinder mode
        # is no longer in the init path.
        "--init-positions",
        "cache",
        "--output",
        str(out_path),
    ]
    benchmark.main()
    return json.loads(out_path.read_text())


@app.local_entrypoint()
def main(
    num_warmup: int = 5000,
    num_samples: int = 5000,
    t: int = 1000,
    n_particles: int = 512,
    n_parameter_particles: int = 2,
    pathfinder_maxiter: int = 50,
    n_pathfinder_starts: int = 8,
    seed: int = 1009,
):
    out = run_recovery.remote(
        num_warmup,
        num_samples,
        t,
        n_particles,
        n_parameter_particles,
        pathfinder_maxiter,
        n_pathfinder_starts,
        seed,
    )
    print(f"\n========= recovery A/B [{GPU}] warmup={num_warmup} samples={num_samples} =========")
    for smoother in ("amala_exact", "amala_plus"):
        run = out["runs"][f"point:{smoother}"]
        rec = run["parameter_recovery"]
        summary = rec["summary"]
        ess = run.get("posterior_ess", {})
        lpm = run.get("latent_path_mixing", {})
        diag = run.get("diagnostics", {})
        print(f"\n---------------- {smoother} ----------------")
        print("  --- recovery (posterior vs truth) ---")
        for k in (
            "target_count",
            "mean_abs_error",
            "median_abs_error",
            "max_abs_error",
            "coverage_90",
        ):
            if k in summary:
                print(f"    {k}: {summary[k]}")
        print(f"    missing_targets: {list(rec.get('missing_targets', {}).keys())}")
        print("  --- ESS (out of samples) ---")
        print(
            f"    parameter ESS: min={ess.get('ess_approx_min')} "
            f"median={ess.get('ess_approx_median')} max={ess.get('ess_approx_max')}"
        )
        print(
            f"    latent-path ESS: min={lpm.get('ess_approx_min')} "
            f"median={lpm.get('ess_approx_median')} max={lpm.get('ess_approx_max')}"
        )
        print("  --- acceptance ---")
        for k in (
            "parameter_accept_rate",
            "post_warmup_parameter_accept_rate",
            "post_warmup_latent_accept_rate",
        ):
            if k in diag:
                print(f"    {k}: {diag[k]}")
    Path("./recovery_h100.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n  wrote ./recovery_h100.json")
