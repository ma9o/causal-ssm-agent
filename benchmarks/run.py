"""Unified recovery benchmark runner.

Replaces tools/recovery.py, tools/recovery_pgas.py, tools/recovery_hessmc2.py,
and tools/recovery_tempered_smc.py with a single script.

Usage:
    uv run python benchmarks/run.py --method svi --local       # local smoke
    uv run python benchmarks/run.py --method all --local       # all local
    modal run benchmarks/run.py --method pgas                   # GPU
    modal run benchmarks/run.py --method all                    # comparison
"""

from __future__ import annotations

import argparse
import time

from benchmarks.metrics import RecoveryResult, header, report_recovery
from benchmarks.problems.four_latent import FOUR_LATENT, RecoveryProblem

# ---------------------------------------------------------------------------
# Method configs: {method: {local: {...}, gpu: {...}, gpu_type, timeout}}
# ---------------------------------------------------------------------------

METHOD_CONFIGS = {
    "svi": {
        "local": {
            "T": 80,
            "num_steps": 1000,
            "num_samples": 200,
            "n_particles": 200,
            "learning_rate": 0.005,
        },
        "gpu": {
            "T": 200,
            "num_steps": 5000,
            "num_samples": 1000,
            "n_particles": 500,
            "learning_rate": 0.005,
        },
        "gpu_type": "L4",
        "timeout": 3600,
    },
    "nuts": {
        "local": {
            "T": 80,
            "num_warmup": 200,
            "num_samples": 200,
        },
        "gpu": {
            "T": 200,
            "num_warmup": 1000,
            "num_samples": 1000,
        },
        "gpu_type": "L4",
        "timeout": 3600,
    },
    "pgas": {
        "local": {
            "T": 80,
            "n_outer": 200,
            "n_csmc_particles": 30,
            "n_mh_steps": 10,
            "n_pf": 100,
            "n_warmup": 100,
            "param_step_size": 0.05,
        },
        "gpu": {
            "T": 200,
            "n_outer": 500,
            "n_csmc_particles": 100,  # N >= T/2 for good CSMC mixing
            "n_mh_steps": 15,
            "n_pf": 200,
            "n_warmup": 250,
            "param_step_size": 0.05,
        },
        "gpu_type": "A100",
        "timeout": 7200,
    },
    "tempered_smc": {
        "local": {
            "T": 80,
            "n_outer": 60,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 200,
            "n_warmup": 30,
            "param_step_size": 0.1,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 120,
            "n_mh_steps": 15,
            "n_pf": 500,
            "n_warmup": 100,
            "param_step_size": 0.1,
        },
        "gpu_type": "B200",
        "timeout": 3600,
    },
    # Baseline variants: disable all SOTA upgrades for A/B comparison
    "pgas_baseline": {
        "local": {
            "T": 80,
            "n_outer": 200,
            "n_csmc_particles": 30,
            "n_mh_steps": 10,
            "n_pf": 100,
            "n_warmup": 100,
            "param_step_size": 0.05,
        },
        "gpu": {
            "T": 200,
            "n_outer": 500,
            "n_csmc_particles": 100,  # N >= T/2 for good CSMC mixing
            "n_mh_steps": 15,
            "n_pf": 200,
            "n_warmup": 250,
            "param_step_size": 0.05,
        },
        "gpu_type": "A100",
        "timeout": 7200,
    },
    "tempered_smc_baseline": {
        "local": {
            "T": 80,
            "n_outer": 60,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 200,
            "n_warmup": 30,
            "param_step_size": 0.1,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 120,
            "n_mh_steps": 15,
            "n_pf": 500,
            "n_warmup": 100,
            "param_step_size": 0.1,
        },
        "gpu_type": "B200",
        "timeout": 3600,
    },
    "hessmc2": {
        "local": {
            "T": 80,
            "n_smc_particles": 64,
            "n_iterations": 15,
            "n_pf": 200,
            "warmup_iters": 5,
        },
        "gpu": {
            "T": 200,
            "n_smc_particles": 512,
            "n_iterations": 30,
            "n_pf": 500,
            "warmup_iters": 10,
        },
        "gpu_type": "B200",
        "timeout": 3600,
    },
}


def run_method(method: str, problem: RecoveryProblem, local: bool) -> RecoveryResult:
    """Run a single inference method and return structured results."""
    import jax
    import numpy as np

    from dsem_agent.models.ssm import SSMModel, fit

    cfg = METHOD_CONFIGS[method]["local" if local else "gpu"]
    T = cfg["T"]

    header("ENVIRONMENT")
    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"Method={method}  T={T}  local={local}")
    print()

    # Simulate
    problem.print_ground_truth()
    obs, times, latent = problem.simulate(T)

    header("SIMULATE")
    print(f"Latent  shape={latent.shape}  mean={np.array(latent.mean(0)).round(2)}")
    print(f"Obs     shape={obs.shape}  mean={np.array(obs.mean(0)).round(2)}")
    print()

    # Report args (shared across all methods)
    report_args = {
        "true_drift": problem.true_drift,
        "true_diff_diag": problem.true_diff_diag,
        "true_cint": problem.true_cint,
        "names": problem.latent_names,
        "n_latent": problem.n_latent,
    }

    # Method-specific fit
    header(f"FIT: {method.upper()}")

    if method == "svi":
        model = SSMModel(
            problem.spec,
            priors=problem.priors,
            n_particles=cfg["n_particles"],
            pf_seed=42,
        )
        t0 = time.perf_counter()
        result = fit(
            model,
            observations=obs,
            times=times,
            method="svi",
            num_steps=cfg["num_steps"],
            num_samples=cfg["num_samples"],
            learning_rate=cfg["learning_rate"],
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        losses = result.diagnostics["losses"]
        n = len(losses)
        ckpts = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        print(f"Done in {elapsed:.1f}s")
        print(
            "ELBO: ",
            "  ".join(f"{i}={float(losses[i]):.0f}" for i in ckpts),
        )
        print()
        # SVI has lambda in samples
        report_args.update(
            {
                "true_lambda": problem.true_lambda,
                "true_manifest_means": problem.true_manifest_means,
                "n_manifest": problem.n_manifest,
            }
        )

    elif method == "nuts":
        model = SSMModel(problem.spec, priors=problem.priors, likelihood="kalman")
        t0 = time.perf_counter()
        result = fit(
            model,
            observations=obs,
            times=times,
            method="nuts",
            num_warmup=cfg["num_warmup"],
            num_samples=cfg["num_samples"],
            num_chains=1,
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        total = cfg["num_warmup"] + cfg["num_samples"]
        print(f"Done in {elapsed:.1f}s ({elapsed / total:.2f}s/step)")
        print()
        # NUTS has lambda in samples
        report_args.update(
            {
                "true_lambda": problem.true_lambda,
                "true_manifest_means": problem.true_manifest_means,
                "n_manifest": problem.n_manifest,
            }
        )

    elif method in ("pgas", "pgas_baseline"):
        is_baseline = method.endswith("_baseline")
        model = SSMModel(
            problem.spec,
            priors=problem.priors,
            n_particles=cfg["n_pf"],
            pf_seed=42,
        )
        # SOTA flags: baseline disables all upgrades, upgraded enables them
        pgas_kwargs = {}
        if is_baseline:
            pgas_kwargs = {"block_sampling": False}
            print("  [BASELINE] block_sampling=False, svi_warmstart=on, HMC(L=5), dual_avg")
        else:
            pgas_kwargs = {"block_sampling": True}
            print("  [UPGRADED] block_sampling=True, svi_warmstart=on, HMC(L=5), dual_avg")
        print()
        t0 = time.perf_counter()
        result = fit(
            model,
            observations=obs,
            times=times,
            method="pgas",
            n_outer=cfg["n_outer"],
            n_csmc_particles=cfg["n_csmc_particles"],
            n_mh_steps=cfg["n_mh_steps"],
            langevin_step_size=0.0,
            param_step_size=cfg["param_step_size"],
            n_warmup=cfg["n_warmup"],
            seed=0,
            **pgas_kwargs,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        print()
        # PGAS-specific diagnostics
        rates = result.diagnostics["accept_rates"]
        print(
            f"  MALA accept: mean={sum(rates) / len(rates):.2f}  "
            f"final_step={result.diagnostics['param_step_size']:.4f}"
        )
        if result.diagnostics.get("svi_warmstart"):
            print("  SVI warmstart: ACTIVE")
        if result.diagnostics.get("gaussian_obs"):
            print("  Optimal proposal: ACTIVE")
        if result.diagnostics.get("block_sampling"):
            block_rates = result.diagnostics.get("block_accept_rates", {})
            for bname, brates in block_rates.items():
                print(f"  Block '{bname}' accept: mean={sum(brates) / len(brates):.2f}")
        print()

    elif method in ("tempered_smc", "tempered_smc_baseline"):
        is_baseline = method.endswith("_baseline")
        model = SSMModel(
            problem.spec,
            priors=problem.priors,
            n_particles=cfg["n_pf"],
            pf_seed=42,
        )
        # SOTA flags: baseline disables all upgrades
        smc_kwargs = {}
        if is_baseline:
            smc_kwargs = {"adaptive_tempering": False, "waste_free": False}
            print("  [BASELINE] adaptive_tempering=False, waste_free=False")
        else:
            smc_kwargs = {"waste_free": True, "adaptive_tempering": True}
            print("  [UPGRADED] adaptive_tempering=True, waste_free=True")
        print()
        t0 = time.perf_counter()
        result = fit(
            model,
            observations=obs,
            times=times,
            method="tempered_smc",
            n_outer=cfg["n_outer"],
            n_csmc_particles=cfg["n_csmc_particles"],
            n_mh_steps=cfg["n_mh_steps"],
            param_step_size=cfg["param_step_size"],
            n_warmup=cfg["n_warmup"],
            seed=0,
            **smc_kwargs,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        # Tempered SMC diagnostics
        beta_schedule = result.diagnostics.get("beta_schedule", [])
        if beta_schedule:
            print(f"  Tempering levels: {len(beta_schedule)}, final beta={beta_schedule[-1]:.4f}")
        if result.diagnostics.get("waste_free"):
            print("  Waste-free: ACTIVE")
        print()

    elif method == "hessmc2":
        model = SSMModel(
            problem.spec,
            priors=problem.priors,
            n_particles=cfg["n_pf"],
            pf_seed=42,
        )
        t0 = time.perf_counter()
        result = fit(
            model,
            observations=obs,
            times=times,
            method="hessmc2",
            n_smc_particles=cfg["n_smc_particles"],
            n_iterations=cfg["n_iterations"],
            proposal="hessian",
            step_size=0.5,
            warmup_iters=cfg["warmup_iters"],
            warmup_step_size=0.5,
            adapt_step_size=False,
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        print()

    else:
        raise ValueError(f"Unknown method: {method}")

    rmse, corr = report_recovery(method.upper(), result.get_samples(), **report_args)
    print(f"  Time={elapsed:.1f}s  RMSE={rmse:.4f}  Corr={corr:.4f}")

    return RecoveryResult(
        method=method,
        rmse=rmse,
        corr=corr,
        elapsed=elapsed,
        coverage={},
    )


def run(methods: list[str], local: bool):
    """Run one or more methods and print comparison."""
    summary = {}
    for method in methods:
        res = run_method(method, FOUR_LATENT, local)
        summary[method] = res

    if len(summary) > 1:
        header("COMPARISON")
        print(f"{'Method':<15s}  {'Time(s)':>8s}  {'RMSE':>8s}  {'Corr':>8s}")
        print("-" * 45)
        for method, res in summary.items():
            print(f"{method:<15s}  {res.elapsed:>8.1f}  {res.rmse:>8.4f}  {res.corr:>8.4f}")


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------

try:
    from benchmarks.modal_infra import make_modal_app

    # Use the heaviest GPU needed (B200) for the unified runner
    app, GPU = make_modal_app("dsem-recovery", "B200")
    HAS_MODAL = True
except Exception:
    HAS_MODAL = False


if HAS_MODAL:

    @app.function(gpu=GPU, timeout=7200)
    def recovery_remote(methods: list[str]):
        run(methods, local=False)

    @app.local_entrypoint()
    def modal_main(method: str = "all"):
        if method == "all":
            methods = list(METHOD_CONFIGS.keys())
        else:
            methods = [m.strip() for m in method.split(",")]
        recovery_remote.remote(methods)


# ---------------------------------------------------------------------------
# CLI entrypoint (local)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter recovery benchmarks")
    parser.add_argument(
        "--method",
        default="svi",
        help="Method name or 'all' (comma-separated for multiple)",
    )
    parser.add_argument("--local", action="store_true", help="Use small local settings")
    args = parser.parse_args()

    if args.method == "all":
        methods = list(METHOD_CONFIGS.keys())
    else:
        methods = [m.strip() for m in args.method.split(",")]

    for m in methods:
        if m not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method '{m}'. Available: {list(METHOD_CONFIGS.keys())}")

    run(methods, local=args.local)
