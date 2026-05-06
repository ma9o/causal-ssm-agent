"""Recovery comparison: aux_gibbs (default config) vs MAP+IEKS.

Uses the 10-latent mixed-support benchmark from run_map_mixed_support_recovery.
After the cleanup, aux_gibbs defaults are DA+Pathfinder — this script just
calls ``fit(..., method="aux_gibbs")`` with its defaults so a regression vs
MAP would immediately show up as width collapse or MAE blow-up.

Usage:
    uv run python scripts/run_aux_gibbs_vs_map_recovery.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_map_mixed_support_recovery import (
    _make_mixed_support_recovery_data,
    _summarize_family_recovery,
)

from causal_ssm_agent.models.ssm import SSMModel, fit


def _run_one(method: str, label: str, data: dict[str, Any], **fit_kwargs: Any) -> dict[str, Any]:
    model = SSMModel(data["spec"], data["priors"], likelihood="particle")
    model.set_observation_support(data["observation_support"])
    print(f"[compare] running {label}...", flush=True)
    t0 = time.perf_counter()
    result = fit(
        model,
        observations=data["observations"],
        times=data["times"],
        method=method,
        **fit_kwargs,
    )
    elapsed = time.perf_counter() - t0
    summary = _summarize_family_recovery(result.get_samples(), data)
    diag: dict[str, Any] = {}
    if method == "aux_gibbs":
        particle = result.diagnostics.get("aux_gibbs", {})
        for k in (
            "adaptation_scheme",
            "init_method",
            "latent_accept_rate",
            "parameter_accept_rate",
        ):
            if k in particle:
                diag[k] = particle[k]
        for k in ("final_latent_delta", "final_param_step_size"):
            if k in particle:
                diag[k + "_mean"] = float(jnp.asarray(particle[k]).mean())
    print(f"[compare] {label} done in {elapsed:.1f}s  diag={diag}")
    return {"label": label, "elapsed_seconds": elapsed, "summary": summary, "diag": diag}


def main() -> None:
    print(f"[compare] backend={jax.default_backend()}")
    data = _make_mixed_support_recovery_data(n_time=40)
    runs = [
        _run_one(
            "map",
            "map",
            data,
            num_samples=500,
            seed=0,
            n_ieks_iters=6,
            maxiter=200,
            parameter_covariance_method="optimizer_hess_inv",
        ),
        _run_one(
            "aux_gibbs",
            "aux_gibbs (defaults)",
            data,
            num_warmup=500,
            num_samples=500,
            num_chains=4,
            seed=0,
            latent_delta=0.01,
            param_step_size=0.03,
            init_scale=0.03,
        ),
    ]

    families = ("drift", "diffusion_sd", "obs_scale", "obs_df")
    metrics = ("mean_abs_error", "mean_ci_width", "coverage")
    print()
    header = f"{'family':<14}{'metric':<14}" + "".join(f"{r['label']:>26}" for r in runs)
    print(header)
    print("-" * len(header))
    for fam in families:
        for metric in metrics:
            row = f"{fam:<14}{metric:<14}"
            for r in runs:
                row += f"{r['summary'][fam][metric]:>26.4f}"
            print(row)

    print()
    print("elapsed_seconds: " + ", ".join(f"{r['label']}={r['elapsed_seconds']:.1f}" for r in runs))
    print()
    print(json.dumps(runs, indent=2, default=float, sort_keys=True))


if __name__ == "__main__":
    main()
