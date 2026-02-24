"""Unified recovery benchmark runner.

Replaces tools/recovery.py, tools/recovery_pgas.py, tools/recovery_hessmc2.py,
and tools/recovery_tempered_smc.py with a single script.

Usage:
    uv run python benchmarks/run.py --method svi --local       # local smoke
    uv run python benchmarks/run.py --method all --local       # all local
    modal run benchmarks/run.py --method pgas                   # GPU (uses config gpu_type)
    modal run benchmarks/run.py --method pgas --gpu A100        # GPU (override)
    modal run benchmarks/run.py --method all                    # comparison (B200)
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.metrics import RecoveryResult
    from benchmarks.problems.four_latent import RecoveryProblem

# Lazy-loaded problem registry (avoids JAX init at import time for Modal)
_PROBLEMS: dict[str, RecoveryProblem] | None = None


def _get_problems() -> dict[str, RecoveryProblem]:
    global _PROBLEMS
    if _PROBLEMS is None:
        from benchmarks.problems import ALL_PROBLEMS

        _PROBLEMS = ALL_PROBLEMS
    return _PROBLEMS

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
    "nuts_da": {
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
        "timeout": 7200,
    },
    "pgas": {
        "local": {
            "T": 80,
            "n_outer": 100,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 100,
            "n_warmup": 50,
            "param_step_size": 0.05,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 100,  # N >= T/2 for good CSMC mixing
            "n_mh_steps": 5,  # HMC(L=5) covers more ground per step
            "n_pf": 200,
            "n_warmup": 100,
            "param_step_size": 0.05,
        },
        "gpu_type": "A100",
        "timeout": 3600,
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
            "n_outer": 100,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 100,
            "n_warmup": 50,
            "param_step_size": 0.05,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 100,
            "n_mh_steps": 5,
            "n_pf": 200,
            "n_warmup": 100,
            "param_step_size": 0.05,
        },
        "gpu_type": "A100",
        "timeout": 3600,
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
    "laplace_em": {
        "local": {
            "T": 80,
            "n_outer": 60,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 200,
            "n_warmup": 30,
            "param_step_size": 0.1,
            "n_ieks_iters": 5,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 120,
            "n_mh_steps": 15,
            "n_pf": 500,
            "n_warmup": 100,
            "param_step_size": 0.1,
            "n_ieks_iters": 5,
        },
        "gpu_type": "B200",
        "timeout": 3600,
    },
    "structured_vi": {
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
    "dpf": {
        "local": {
            "T": 80,
            "n_outer": 60,
            "n_csmc_particles": 30,
            "n_mh_steps": 5,
            "n_pf": 200,
            "n_warmup": 30,
            "param_step_size": 0.1,
            "n_train_seqs": 10,
            "n_train_steps": 50,
            "n_particles_train": 16,
            "n_pf_particles": 50,
        },
        "gpu": {
            "T": 200,
            "n_outer": 200,
            "n_csmc_particles": 120,
            "n_mh_steps": 15,
            "n_pf": 500,
            "n_warmup": 100,
            "param_step_size": 0.1,
            "n_train_seqs": 50,
            "n_train_steps": 200,
            "n_particles_train": 32,
            "n_pf_particles": 100,
        },
        "gpu_type": "B200",
        "timeout": 7200,
    },
}


def run_method(method: str, problem: RecoveryProblem, local: bool) -> RecoveryResult:
    """Run a single inference method and return structured results."""
    import jax
    import numpy as np

    from benchmarks.metrics import RecoveryResult, header, report_recovery
    from causal_ssm_agent.models.ssm import SSMModel, fit

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

    elif method == "nuts_da":
        model = SSMModel(problem.spec, priors=problem.priors)
        t0 = time.perf_counter()
        # Forward optional kwargs (centered, dense_mass, etc.)
        da_kwargs = {}
        for k in (
            "centered",
            "dense_mass",
            "target_accept_prob",
            "max_tree_depth",
            "svi_warmstart",
            "svi_num_steps",
            "svi_learning_rate",
        ):
            if k in cfg:
                da_kwargs[k] = cfg[k]
        result = fit(
            model,
            observations=obs,
            times=times,
            method="nuts_da",
            num_warmup=cfg["num_warmup"],
            num_samples=cfg["num_samples"],
            num_chains=1,
            seed=0,
            **da_kwargs,
        )
        elapsed = time.perf_counter() - t0
        total = cfg["num_warmup"] + cfg["num_samples"]
        print(f"Done in {elapsed:.1f}s ({elapsed / total:.2f}s/step)")
        print()
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
            pgas_kwargs = {"svi_warmstart": False, "n_leapfrog": 1}
            print("  [BASELINE] no svi_warmstart, MALA(L=1), dual_avg")
        else:
            pgas_kwargs = {}  # svi_warmstart=True, n_leapfrog=5 by default
            print("  [UPGRADED] svi_warmstart=on, HMC(L=5), dual_avg")
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

    elif method == "laplace_em":
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
            method="laplace_em",
            n_outer=cfg["n_outer"],
            n_csmc_particles=cfg["n_csmc_particles"],
            n_mh_steps=cfg["n_mh_steps"],
            param_step_size=cfg["param_step_size"],
            n_warmup=cfg["n_warmup"],
            n_ieks_iters=cfg["n_ieks_iters"],
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        beta_schedule = result.diagnostics.get("beta_schedule", [])
        if beta_schedule:
            print(f"  Tempering levels: {len(beta_schedule)}, final beta={beta_schedule[-1]:.4f}")
        rates = result.diagnostics.get("accept_rates", [])
        if rates:
            print(f"  Accept rate: mean={sum(rates) / len(rates):.2f}")
        print()

    elif method == "structured_vi":
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
            method="structured_vi",
            n_outer=cfg["n_outer"],
            n_csmc_particles=cfg["n_csmc_particles"],
            n_mh_steps=cfg["n_mh_steps"],
            param_step_size=cfg["param_step_size"],
            n_warmup=cfg["n_warmup"],
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        beta_schedule = result.diagnostics.get("beta_schedule", [])
        if beta_schedule:
            print(f"  Tempering levels: {len(beta_schedule)}, final beta={beta_schedule[-1]:.4f}")
        rates = result.diagnostics.get("accept_rates", [])
        if rates:
            print(f"  Accept rate: mean={sum(rates) / len(rates):.2f}")
        print()

    elif method == "dpf":
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
            method="dpf",
            n_outer=cfg["n_outer"],
            n_csmc_particles=cfg["n_csmc_particles"],
            n_mh_steps=cfg["n_mh_steps"],
            param_step_size=cfg["param_step_size"],
            n_warmup=cfg["n_warmup"],
            n_train_seqs=cfg["n_train_seqs"],
            n_train_steps=cfg["n_train_steps"],
            n_particles_train=cfg["n_particles_train"],
            n_pf_particles=cfg["n_pf_particles"],
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s")
        beta_schedule = result.diagnostics.get("beta_schedule", [])
        if beta_schedule:
            print(f"  Tempering levels: {len(beta_schedule)}, final beta={beta_schedule[-1]:.4f}")
        rates = result.diagnostics.get("accept_rates", [])
        if rates:
            print(f"  Accept rate: mean={sum(rates) / len(rates):.2f}")
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


def run(methods: list[str], local: bool, problem_name: str = "four_latent"):
    """Run one or more methods and print comparison."""
    from benchmarks.metrics import header

    problems = _get_problems()
    if problem_name not in problems:
        raise ValueError(f"Unknown problem '{problem_name}'. Available: {list(problems.keys())}")
    problem = problems[problem_name]

    summary = {}
    for method in methods:
        res = run_method(method, problem, local)
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

# GPU priority ranking (most powerful first)
_GPU_PRIORITY = {"B200": 0, "A100": 1, "L4": 2}


def _resolve_modal_gpu() -> str:
    """Determine GPU type from CLI args at module import time.

    Modal requires GPU type at function registration (import time), so we peek
    at sys.argv before the local entrypoint is called.

    Priority: --gpu flag > method config gpu_type > B200 fallback.
    """
    import sys

    args = sys.argv

    # Explicit --gpu override
    for i, arg in enumerate(args):
        if arg == "--gpu" and i + 1 < len(args):
            return args[i + 1]

    # Derive from --method config (pick the most powerful GPU needed)
    for i, arg in enumerate(args):
        if arg == "--method" and i + 1 < len(args):
            method_str = args[i + 1]
            if method_str == "all":
                break  # fall through to B200 default
            methods = [m.strip() for m in method_str.split(",")]
            best = "L4"
            for m in methods:
                g = METHOD_CONFIGS.get(m, {}).get("gpu_type", "L4")
                if _GPU_PRIORITY.get(g, 99) < _GPU_PRIORITY.get(best, 99):
                    best = g
            return best

    return "B200"  # default for --method all or unrecognised


def _resolve_modal_timeout() -> int:
    """Max timeout across requested methods (default 7200)."""
    import sys

    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "--method" and i + 1 < len(args):
            method_str = args[i + 1]
            if method_str == "all":
                methods = list(METHOD_CONFIGS.keys())
            else:
                methods = [m.strip() for m in method_str.split(",")]
            return max(METHOD_CONFIGS.get(m, {}).get("timeout", 3600) for m in methods)
    return 7200


try:
    from benchmarks.modal_infra import make_modal_app

    _MODAL_GPU = _resolve_modal_gpu()
    _MODAL_TIMEOUT = _resolve_modal_timeout()
    app, GPU = make_modal_app("causal-ssm-recovery", _MODAL_GPU)
    HAS_MODAL = True
except Exception:
    HAS_MODAL = False


if HAS_MODAL:

    @app.function(gpu=GPU, timeout=_MODAL_TIMEOUT)
    def recovery_remote(methods: list[str], problem: str = "four_latent"):
        run(methods, local=False, problem_name=problem)

    @app.local_entrypoint()
    def modal_main(method: str = "all", gpu: str = "", problem: str = "four_latent"):  # noqa: ARG001 (gpu consumed at import time)
        if method == "all":
            methods = list(METHOD_CONFIGS.keys())
        else:
            methods = [m.strip() for m in method.split(",")]
        print(f"GPU: {_MODAL_GPU}  timeout: {_MODAL_TIMEOUT}s  problem: {problem}")
        recovery_remote.remote(methods, problem)


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
    parser.add_argument(
        "--gpu",
        default="",
        help="GPU type for Modal (L4, A100, B200). Ignored for --local runs.",
    )
    parser.add_argument(
        "--problem",
        default="four_latent",
        help="Problem name (four_latent, three_latent_robust)",
    )
    args = parser.parse_args()

    if args.method == "all":
        methods = list(METHOD_CONFIGS.keys())
    else:
        methods = [m.strip() for m in args.method.split(",")]

    for m in methods:
        if m not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method '{m}'. Available: {list(METHOD_CONFIGS.keys())}")

    run(methods, local=args.local, problem_name=args.problem)
