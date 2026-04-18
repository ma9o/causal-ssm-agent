"""Compare aux_csmc against MAP + IEKS on the 10-latent mixed-support recovery.

Usage:
    uv run python scripts/run_aux_csmc_vs_map_recovery.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nuts_mixed_support_recovery import (
    _make_mixed_support_recovery_data,
    _summarize_family_recovery,
)

from causal_ssm_agent.models.ssm import SSMModel, fit


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    return value


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
    if method == "aux_csmc":
        sampler_diag = {
            k: v
            for k, v in result.diagnostics.get("aux_csmc", {}).items()
            if not isinstance(v, dict)
        }
    return {
        "method": method,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "sampler_diag": sampler_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-time", type=int, default=200)
    parser.add_argument("--map-num-samples", type=int, default=400)
    parser.add_argument("--map-maxiter", type=int, default=200)
    parser.add_argument("--map-n-ieks-iters", type=int, default=6)
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-delta", type=float, default=0.05)
    parser.add_argument("--latent-target-accept", type=float, default=0.5)
    parser.add_argument("--param-step-size", type=float, default=0.03)
    parser.add_argument("--param-target-accept", type=float, default=0.57)
    parser.add_argument("--adaptation-rate", type=float, default=0.05)
    parser.add_argument(
        "--adaptation-scheme",
        choices=("simple", "dual_averaging"),
        default="dual_averaging",
    )
    parser.add_argument("--init-scale", type=float, default=0.03)
    parser.add_argument("--n-csmc-particles", type=int, default=25)
    parser.add_argument(
        "--backward-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--init-method",
        choices=("random", "pathfinder"),
        default="pathfinder",
    )
    args = parser.parse_args()

    print(f"[compare] backend={jax.default_backend()} devices={jax.devices()}")
    print(f"[compare] building 10-latent mixed-support recovery (n_time={args.n_time})")
    data = _make_mixed_support_recovery_data(n_time=args.n_time)

    print("[compare] running method=map (MAP + IEKS + Laplace widths)...", flush=True)
    map_run = _run_one(
        "map",
        data,
        num_samples=args.map_num_samples,
        seed=args.seed,
        n_ieks_iters=args.map_n_ieks_iters,
        maxiter=args.map_maxiter,
        parameter_covariance_method="optimizer_hess_inv",
    )
    print(f"[compare] map done in {map_run['elapsed_seconds']:.1f}s")

    print(
        "[compare] running method=aux_csmc (joint (x, theta) blocked Gibbs with auxiliary cSMC)...",
        flush=True,
    )
    aux_run = _run_one(
        "aux_csmc",
        data,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        latent_delta=args.latent_delta,
        latent_target_accept=args.latent_target_accept,
        n_csmc_particles=args.n_csmc_particles,
        backward_sampling=args.backward_sampling,
        param_step_size=args.param_step_size,
        param_target_accept=args.param_target_accept,
        adaptation_rate=args.adaptation_rate,
        adaptation_scheme=args.adaptation_scheme,
        init_scale=args.init_scale,
        init_method=args.init_method,
    )
    print(f"[compare] aux_csmc done in {aux_run['elapsed_seconds']:.1f}s")
    print(f"[aux_csmc diag] {aux_run['sampler_diag']}")

    print()
    print(f"{'family':<16}{'metric':<12}{'aux_csmc':>12}{'map':>12}")
    print("-" * 52)
    for family in ("drift", "diffusion_sd", "obs_scale", "obs_df"):
        for metric in ("mean_abs_error", "mean_ci_width", "coverage"):
            av = aux_run["summary"][family][metric]
            mv = map_run["summary"][family][metric]
            print(f"{family:<16}{metric:<12}{av:>12.4f}{mv:>12.4f}")

    print()
    payload = {"aux_csmc": aux_run, "map": map_run}
    print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
