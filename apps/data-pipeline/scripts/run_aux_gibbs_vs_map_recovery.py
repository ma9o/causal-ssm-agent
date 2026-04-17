"""Compare joint aux_gibbs vs MAP+IEKS on the 10-latent mixed-support recovery.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nuts_mixed_support_recovery import (  # noqa: E402
    _make_mixed_support_recovery_data,
    _summarize_family_recovery,
)

from causal_ssm_agent.models.ssm import SSMModel, fit  # noqa: E402


def _run_one(method: str, data: dict[str, Any], **fit_kwargs: Any) -> dict[str, Any]:
    model = SSMModel(data["spec"], data["priors"], likelihood="particle")
    model.set_observation_support(data["observation_support"])
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
    sampler_diag: dict[str, Any] = {}
    if method == "aux_gibbs":
        sampler_diag = {
            k: v
            for k, v in result.diagnostics.get("aux_gibbs", {}).items()
            if not isinstance(v, dict)
        }
    return {
        "method": method,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "sampler_diag": sampler_diag,
    }


def main() -> None:
    n_time = 40
    print(f"[compare] backend={jax.default_backend()} devices={jax.devices()}")
    print(f"[compare] building 10-latent mixed-support recovery (n_time={n_time})")
    data = _make_mixed_support_recovery_data(n_time=n_time)

    # MAP + IEKS baseline.
    print("[compare] running method=map (MAP + IEKS + Laplace widths)...", flush=True)
    map_run = _run_one(
        "map",
        data,
        num_samples=400,
        seed=0,
        n_ieks_iters=6,
        maxiter=200,
        parameter_covariance_method="optimizer_hess_inv",
    )
    print(f"[compare] map done in {map_run['elapsed_seconds']:.1f}s")

    # Joint aux_gibbs (new kernel under eq 8 reparametrisation).
    print("[compare] running method=aux_gibbs (joint (x, theta) MH)...", flush=True)
    aux_run = _run_one(
        "aux_gibbs",
        data,
        num_warmup=2000,
        num_samples=400,
        num_chains=4,
        seed=0,
        latent_delta=0.01,
        param_step_size=0.03,
        latent_target_accept=0.5,
        param_target_accept=0.57,
        adaptation_rate=0.05,
        init_scale=0.03,
    )
    print(f"[compare] aux_gibbs done in {aux_run['elapsed_seconds']:.1f}s")
    print(f"[aux_gibbs diag] {aux_run['sampler_diag']}")

    print()
    print(f"{'family':<16}{'metric':<12}{'aux_gibbs':>12}{'map':>12}")
    print("-" * 52)
    for family in ("drift", "diffusion_sd", "obs_scale", "obs_df"):
        for metric in ("mean_abs_error", "mean_ci_width", "coverage"):
            av = aux_run["summary"][family][metric]
            mv = map_run["summary"][family][metric]
            print(f"{family:<16}{metric:<12}{av:>12.4f}{mv:>12.4f}")

    print()
    payload = {"aux_gibbs": aux_run, "map": map_run}
    print(json.dumps(payload, indent=2, default=float, sort_keys=True))


if __name__ == "__main__":
    main()
