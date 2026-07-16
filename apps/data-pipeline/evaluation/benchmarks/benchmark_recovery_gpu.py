"""Modal GPU parameter-recovery comparison for current dSMC leaf proposals.

The worker invokes ``evaluation/benchmarks/benchmark.py`` so local and GPU runs
exercise the same production inference path. It compares ``amala_exact`` with
``paid_mix``; both use the exact Euler-Maruyama posterior target.

Running this entry point provisions a paid GPU:

    modal run evaluation/benchmarks/benchmark_recovery_gpu.py \
        --num-warmup 5000 --num-samples 5000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("BENCHMARK_GPU", "H100")
FORCE_BUILD = "--force-build" in sys.argv


def _build_image() -> modal.Image:
    root = Path(__file__).resolve().parents[2]
    return (
        modal.Image.debian_slim(python_version="3.12", force_build=FORCE_BUILD)
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(root), groups=["dev", "cloud"], frozen=True)
        .uv_pip_install("jax[cuda12]==0.9.0.1")
        .env({"PYTHONPATH": ("/root/apps/data-pipeline/src:/root/apps/data-pipeline")})
        .add_local_file(root / "config.yaml", remote_path="/root/config.yaml")
        .add_local_file(
            root / "pyproject.toml",
            remote_path="/root/apps/data-pipeline/pyproject.toml",
        )
        .add_local_dir(
            root / "src" / "nof1_causal_lab",
            remote_path="/root/apps/data-pipeline/src/nof1_causal_lab",
        )
        .add_local_dir(
            root / "evaluation",
            remote_path="/root/apps/data-pipeline/evaluation",
        )
    )


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
    """Run the current recovery comparison on one GPU worker."""
    import jax
    from evaluation.benchmarks import benchmark

    print("JAX devices:", jax.devices(), flush=True)
    out_path = Path("/root/scratchpad/mpg_synthetic_nonlinear_parameter_recovery.json")
    sys.argv = [
        "benchmark.py",
        "--proposals",
        "amala_exact,paid_mix",
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
) -> None:
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
    print(f"\n========= recovery [{GPU}] warmup={num_warmup} samples={num_samples} =========")
    for proposal in ("amala_exact", "paid_mix"):
        run = out["runs"][f"point:{proposal}"]
        recovery = run["parameter_recovery"]
        summary = recovery["summary"]
        ess = run["posterior_ess"]
        diagnostics = run["diagnostics"]
        print(f"\n---------------- {proposal} ----------------")
        for field in (
            "target_count",
            "mean_abs_error",
            "median_abs_error",
            "max_abs_error",
            "coverage_90",
        ):
            print(f"  {field}: {summary[field]}")
        print(f"  missing_targets: {list(recovery['missing_targets'])}")
        print(
            "  parameter ESS: "
            f"min={ess['ess_approx_min']} median={ess['ess_approx_median']} "
            f"max={ess['ess_approx_max']}"
        )
        print(f"  parameter_accept_rate: {diagnostics['parameter_accept_rate']}")
        print(f"  latent_update_fraction: {diagnostics['latent_update_fraction']}")
    Path("./recovery_h100.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("\nwrote ./recovery_h100.json")
