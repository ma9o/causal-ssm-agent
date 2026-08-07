"""Parameter-recovery benchmark for the production particle-MCMC sampler.

The benchmark compares the two supported dSMC leaf proposals on the synthetic
nonlinear fixture. Both proposals target the exact Euler-Maruyama transition;
the IEKS/Laplace machinery used by ``paid_mix`` only constructs its corrected
proposal and never substitutes for the reported posterior.

This is an evaluation entry point, not a test. Running it performs inference
and can be expensive:

    uv run python evaluation/benchmarks/benchmark.py
    uv run python evaluation/benchmarks/benchmark.py --proposals paid_mix --T 24
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

LeafProposal = Literal["amala_exact", "paid_mix"]
LEAF_PROPOSALS: tuple[LeafProposal, ...] = ("amala_exact", "paid_mix")
type BenchmarkRecord = dict[str, Any]

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "apps/data-pipeline/pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
DEFAULT_OUTPUT = REPO_ROOT / "scratchpad" / "mpg_synthetic_nonlinear_parameter_recovery.json"


def _configure_jax_cache() -> None:
    cache_dir = REPO_ROOT / "scratchpad" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"


def _parse_proposals(raw: str) -> tuple[LeafProposal, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(LEAF_PROPOSALS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown proposals: {unknown}; allowed: {list(LEAF_PROPOSALS)}"
        )
    if not values:
        raise argparse.ArgumentTypeError("at least one proposal is required")
    return cast("tuple[LeafProposal, ...]", values)


def _parse_metrics(raw: str) -> tuple[str, ...]:
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
        MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    )

    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(MPGIBBS_DIAGNOSTIC_METRIC_VALUES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown diagnostic metrics: "
            f"{unknown}; allowed: {list(MPGIBBS_DIAGNOSTIC_METRIC_VALUES)}"
        )
    return values


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    try:
        import jax
        import numpy as np

        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return str(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _diagnostic_summary(result: Any) -> BenchmarkRecord:
    diagnostics = result.diagnostics["marginal_particle_gibbs"]
    fields = (
        "latent_kernel",
        "latent_smoother",
        "latent_smoother_selection",
        "dsmc_leaf_proposal",
        "latent_transition_kind",
        "parameter_kernel",
        "parameter_preconditioned",
        "parameter_accept_rate",
        "latent_update_fraction",
        "latent_frozen_fraction",
        "latent_block_coords",
        "mcmc_phase_seconds",
        "diagnostic_metrics",
        "diagnostic_summary_phase",
        "final_param_step_size",
        "final_latent_delta",
    )
    return {field: _json_ready(diagnostics.get(field)) for field in fields}


def _run_one(
    *,
    proposal: LeafProposal,
    model: Any,
    observations: Any,
    times: Any,
    args: argparse.Namespace,
) -> BenchmarkRecord:
    from evaluation.recovery.extraction import parameter_recovery, scalar_posterior_ess

    from nof1_causal_lab.models.ssm.inference import fit

    logger.info("starting dSMC/%s", proposal)
    started = time.monotonic()
    result = fit(
        model,
        observations,
        times,
        method="marginal_particle_gibbs",
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        n_particles=args.n_particles,
        n_parameter_particles=args.n_parameter_particles,
        latent_smoother="dsmc",
        dsmc_leaf_proposal=proposal,
        latent_block_coords=args.latent_block_coords,
        latent_sign_flip_moves=args.latent_sign_flip_moves,
        diagnostic_metrics=args.diagnostic_metrics,
        init_method=args.init_method,
        init_scale=args.init_scale,
        pathfinder_num_elbo_samples=args.pathfinder_num_elbo_samples,
        pathfinder_maxiter=args.pathfinder_maxiter,
        n_pathfinder_starts=args.n_pathfinder_starts,
        pathfinder_parallel_workers=args.pathfinder_parallel_workers,
        pathfinder_init_scale=args.pathfinder_init_scale,
        auto_preconditioner_method=args.auto_preconditioner_method,
        auto_preconditioner_maxiter=args.auto_preconditioner_maxiter,
        n_ieks_iters=args.n_ieks_iters,
        param_step_size=args.param_step_size,
        param_step_size_min=args.param_step_size_min,
        param_step_size_max=args.param_step_size_max,
        param_target_accept=args.param_target_accept,
        adaptation_rate=args.adaptation_rate,
        amala_delta_init=args.amala_delta_init,
        amala_delta_min=args.amala_delta_min,
        amala_delta_max=args.amala_delta_max,
        amala_target_accept=args.amala_target_accept,
        paid_mix_z_weight=args.paid_mix_z_weight,
        paid_mix_pilot_weight=args.paid_mix_pilot_weight,
        paid_mix_pilot_var_scale=args.paid_mix_pilot_var_scale,
        paid_mix_wide_mult=args.paid_mix_wide_mult,
        retain_latent_paths=args.retain_latent_paths,
        compute_latent_posterior_summary=not args.skip_latent_posterior_summary,
    )
    elapsed_seconds = time.monotonic() - started
    logger.info("finished dSMC/%s in %.1fs", proposal, elapsed_seconds)
    return {
        "elapsed_seconds": elapsed_seconds,
        "diagnostics": _diagnostic_summary(result),
        "parameter_recovery": parameter_recovery(
            result,
            elapsed_seconds=elapsed_seconds,
        ),
        "posterior_ess": scalar_posterior_ess(
            result,
            max_sites=args.max_ess_sites,
            elapsed_seconds=elapsed_seconds,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", dest="t_steps", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71)
    parser.add_argument("--diffusion-scale", type=float, default=1.0)
    parser.add_argument("--proposals", type=_parse_proposals, default=LEAF_PROPOSALS)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--n-parameter-particles", type=int, default=2)
    parser.add_argument("--latent-block-coords", type=int)
    parser.add_argument("--latent-sign-flip-moves", action="store_true")
    parser.add_argument(
        "--diagnostic-metrics",
        type=_parse_metrics,
        default=("particle_identity", "parameter_movement"),
    )
    parser.add_argument("--init-method", choices=("random", "pathfinder"), default="pathfinder")
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--pathfinder-num-elbo-samples", type=int, default=20)
    parser.add_argument("--pathfinder-maxiter", type=int, default=50)
    parser.add_argument("--n-pathfinder-starts", type=int, default=8)
    parser.add_argument("--pathfinder-parallel-workers", type=int)
    parser.add_argument("--pathfinder-init-scale", type=float, default=0.1)
    parser.add_argument(
        "--auto-preconditioner-method",
        choices=("map", "none", "pathfinder"),
        default="pathfinder",
    )
    parser.add_argument("--auto-preconditioner-maxiter", type=int, default=200)
    parser.add_argument("--n-ieks-iters", type=int, default=6)
    parser.add_argument("--param-step-size", type=float, default=0.02)
    parser.add_argument("--param-step-size-min", type=float, default=1e-6)
    parser.add_argument("--param-step-size-max", type=float, default=1e3)
    parser.add_argument("--param-target-accept", type=float)
    parser.add_argument("--adaptation-rate", type=float, default=0.05)
    parser.add_argument("--amala-delta-init", type=float, default=0.2)
    parser.add_argument("--amala-delta-min", type=float, default=1e-4)
    parser.add_argument("--amala-delta-max", type=float, default=1e2)
    parser.add_argument("--amala-target-accept", type=float, default=0.574)
    parser.add_argument("--paid-mix-z-weight", type=float, default=0.85)
    parser.add_argument("--paid-mix-pilot-weight", type=float, default=0.10)
    parser.add_argument("--paid-mix-pilot-var-scale", type=float, default=0.25)
    parser.add_argument("--paid-mix-wide-mult", type=float, default=4.0)
    parser.add_argument("--retain-latent-paths", action="store_true")
    parser.add_argument("--skip-latent-posterior-summary", action="store_true")
    parser.add_argument("--max-ess-sites", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.n_parameter_particles < 2:
        raise ValueError("--n-parameter-particles must be at least 2")

    _configure_jax_cache()
    from evaluation.fixtures.synthetic_nonlinear import (
        build_synthetic_nonlinear_model,
        simulate_synthetic_nonlinear_data,
    )

    data = simulate_synthetic_nonlinear_data(
        T=args.t_steps,
        seed=args.data_seed,
        diffusion_scale=args.diffusion_scale,
    )
    model = build_synthetic_nonlinear_model(
        data,
        include_interval_support=False,
        diffusion_scale=args.diffusion_scale,
    )
    runs = {
        f"point:{proposal}": _run_one(
            proposal=proposal,
            model=model,
            observations=data.observations,
            times=data.times,
            args=args,
        )
        for proposal in args.proposals
    }
    artifact = {
        "artifact": "mpg_synthetic_nonlinear_parameter_recovery",
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture": "synthetic_nonlinear",
        "config": {
            "T": args.t_steps,
            "data_seed": args.data_seed,
            "diffusion_scale": args.diffusion_scale,
            "proposals": args.proposals,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "seed": args.seed,
            "n_particles": args.n_particles,
            "n_parameter_particles": args.n_parameter_particles,
            "latent_block_coords": args.latent_block_coords,
            "diagnostic_metrics": args.diagnostic_metrics,
            "init_method": args.init_method,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(artifact), indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
