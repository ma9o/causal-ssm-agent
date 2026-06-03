"""Particle marginal Metropolis-Hastings inference method."""

from __future__ import annotations

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.bundle import build_particle_runtime_bundle
from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared import (
    build_pmcmc_mcmc_result,
    extract_grouped_public_samples,
    prepare_pmcmc_parameter_warmup,
)
from nof1_causal_lab.models.ssm.inference.methods.particle_marginal_mh.kernel import (
    build_particle_marginal_mh_kernel,
    run_particle_marginal_mh,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

logger = logging.getLogger(__name__)

_DEFAULT_PARAM_STEP_SIZE_MIN = 1e-6
_DEFAULT_PARAM_STEP_SIZE_MAX = 1e3


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


def fit_particle_marginal_mh(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 4000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    n_particles: int = 128,
    diagnostic_metrics_all: bool = False,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = None,
    param_step_size: float = 0.02,
    param_step_size_min: float = _DEFAULT_PARAM_STEP_SIZE_MIN,
    param_step_size_max: float = _DEFAULT_PARAM_STEP_SIZE_MAX,
    param_target_accept: float | None = None,
    adaptation_rate: float = 0.05,
    adaptation_scheme: str = "simple",
    init_scale: float = 0.05,
    init_method: str = "pathfinder",
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 8,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = 0.1,
    auto_preconditioner_method: str = "pathfinder",
    auto_preconditioner_maxiter: int = 200,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    initial_positions_override: jnp.ndarray | None = None,
    n_ieks_iters: int = 6,
    reparam=None,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = False,
    **_kwargs: Any,
) -> InferenceResult:
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            "Unsupported particle_marginal_mh init_method "
            f"{init_method!r}. Supported: 'random' or 'pathfinder'."
        )
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            "Unsupported particle_marginal_mh adaptation_scheme "
            f"{adaptation_scheme!r}. Supported: 'simple' or 'dual_averaging'."
        )
    if retain_latent_paths:
        raise ValueError("particle_marginal_mh does not retain latent paths yet.")
    if compute_latent_posterior_summary:
        raise ValueError("particle_marginal_mh does not compute latent posterior summaries yet.")

    overall_t0 = time.monotonic()
    logger.info(
        "particle_marginal_mh entry: chains=%d warmup=%d samples=%d T=%d "
        "n_manifest=%d n_particles=%d init_method=%s",
        num_chains,
        num_warmup,
        num_samples,
        int(observations.shape[0]),
        int(observations.shape[1]) if observations.ndim >= 2 else 0,
        n_particles,
        init_method,
    )
    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key = random.split(base_key, 3)

    phase_t0 = time.monotonic()
    logger.info("phase 1/4: building PMMH runtime bundle...")
    bundle = build_particle_runtime_bundle(
        model,
        observations,
        times,
        scheme=LATENT_TRANSITION_EULER_MARUYAMA,
        trace_key=trace_key,
        reparam=reparam,
    )
    logger.info(
        "phase 1/4: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        _phase_elapsed(phase_t0),
        int(bundle["flat_example"].shape[0]),
        len(bundle.get("public_sites", [])),
    )

    phase_t0 = time.monotonic()
    warmup_result = prepare_pmcmc_parameter_warmup(
        model,
        observations,
        times,
        bundle=bundle,
        method_label="particle_marginal_mh",
        phase_label="phase 2/4",
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=pf_sample_key,
        reparam=reparam,
        seed=seed,
        n_ieks_iters=n_ieks_iters,
        num_chains=num_chains,
        init_method=init_method,
        initial_positions_override=initial_positions_override,
        init_scale=init_scale,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        auto_preconditioner_method=auto_preconditioner_method,
        auto_preconditioner_maxiter=auto_preconditioner_maxiter,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_parallel_workers=pathfinder_parallel_workers,
        pathfinder_init_scale=pathfinder_init_scale,
    )
    init_positions = warmup_result.init_positions
    parameter_preconditioner_chol = warmup_result.preconditioner_chol
    logger.info("phase 2/4: parameter warmup ready in %.1fs", _phase_elapsed(phase_t0))

    phase_t0 = time.monotonic()
    logger.info("phase 3/4: building PMMH kernel...")
    kernel = build_particle_marginal_mh_kernel(
        bundle,
        num_particles=n_particles,
        param_step_size=param_step_size,
        target_accept=param_target_accept,
        min_scale=param_step_size_min,
        max_scale=param_step_size_max,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        diagnostic_metrics_all=diagnostic_metrics_all,
        diagnostic_metrics=diagnostic_metrics,
    )
    run_result = run_particle_marginal_mh(
        bundle,
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        init_positions=init_positions,
        adaptation_scheme=adaptation_scheme,
    )
    mcmc_phase_seconds = _phase_elapsed(phase_t0)
    logger.info("phase 3/4: MCMC complete in %.1fs", mcmc_phase_seconds)

    phase_t0 = time.monotonic()
    logger.info("phase 4/4: extracting public posterior samples...")
    grouped_public_samples = extract_grouped_public_samples(
        run_result["grouped_positions"],
        bundle=bundle,
        model=model,
        observations=observations,
        times=times,
        num_chains=num_chains,
        num_samples=num_samples,
        reparam=reparam,
    )
    mcmc = build_pmcmc_mcmc_result(
        chain_samples=grouped_public_samples,
        chain_extra_fields=run_result["chain_extra_fields"],
        num_chains=num_chains,
        num_samples=num_samples,
        backend="particle_marginal_mh",
    )
    chain_extra_fields = run_result["chain_extra_fields"]
    summary_extra_fields = (
        chain_extra_fields if num_samples > 0 else run_result["warmup_chain_extra_fields"]
    )
    diagnostic_summary_phase = "post_warmup" if num_samples > 0 else "warmup"
    parameter_jump_rms_mean = (
        None
        if "parameter_jump_rms" not in summary_extra_fields
        else float(jnp.mean(summary_extra_fields["parameter_jump_rms"]))
    )
    diagnostics_summary = {
        "parameter_kernel": "pmmh_random_walk",
        "mcmc_phase_seconds": float(mcmc_phase_seconds),
        "first_step_seconds": float(run_result["first_step_seconds"]),
        "sampling_loop_seconds": float(run_result["sampling_loop_seconds"]),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "n_particles": int(n_particles),
        "param_step_size_initial": float(param_step_size),
        "param_step_size_min": float(param_step_size_min),
        "param_step_size_max": float(param_step_size_max),
        "param_target_accept": float(kernel.target_accept),
        "adaptation_scheme": adaptation_scheme,
        "parameter_preconditioned": bool(kernel.preconditioned),
        "latent_transition_kind": bundle["latent_transition_kind"],
        "diagnostic_summary_phase": diagnostic_summary_phase,
        "diagnostic_metrics_all": bool(diagnostic_metrics_all),
        "diagnostic_metrics": sorted(kernel.diagnostic_metrics),
        "parameter_accept_rate": float(jnp.mean(summary_extra_fields["parameter_accept_prob"])),
        "parameter_jump_rms_mean": parameter_jump_rms_mean,
        "estimated_log_likelihood_mean": float(
            jnp.mean(summary_extra_fields["estimated_log_likelihood"])
        ),
        "estimated_log_posterior_mean": float(
            jnp.mean(summary_extra_fields["estimated_log_posterior"])
        ),
        "pf_ess_min_mean": float(jnp.mean(summary_extra_fields["pf_ess_min"])),
        "pf_ess_mean_mean": float(jnp.mean(summary_extra_fields["pf_ess_mean"])),
        "pf_log_weight_range_max_mean": float(
            jnp.mean(summary_extra_fields["pf_log_weight_range_max"])
        ),
        "pf_log_weight_variance_mean": float(
            jnp.mean(summary_extra_fields["pf_log_weight_variance_mean"])
        ),
        "pf_log_likelihood_increment_variance_mean": float(
            jnp.mean(summary_extra_fields["pf_log_likelihood_increment_variance"])
        ),
        "initial_param_step_size": jax.device_get(run_result["initial_param_step_size"]).tolist(),
        "final_param_step_size": jax.device_get(run_result["final_param_step_size"]).tolist(),
        "chain_post_warmup_estimated_log_posterior_mean": jax.device_get(
            run_result["post_warmup_estimated_log_posterior_mean"]
        ).tolist(),
        "parameter_warmup": warmup_result.warmup_diagnostics,
        **warmup_result.init_diagnostics,
        **warmup_result.preconditioner_diagnostics,
    }
    diagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle["public_sites"]),
        "particle_marginal_mh": diagnostics_summary,
        "particle_marginal_mh_phase_extra_fields": {
            "warmup": run_result["warmup_chain_extra_fields"],
            "post_warmup": run_result["chain_extra_fields"],
            "all": run_result["all_chain_extra_fields"],
        },
        "chain_estimated_log_posterior_history": run_result["estimated_log_posterior_history"],
        "warmup_estimated_log_posterior_history": run_result[
            "warmup_estimated_log_posterior_history"
        ],
        "all_estimated_log_posterior_history": run_result["all_estimated_log_posterior_history"],
    }
    logger.info(
        "phase 4/4: posterior extraction complete in %.1fs. particle_marginal_mh total: %.1fs",
        _phase_elapsed(phase_t0),
        _phase_elapsed(overall_t0),
    )
    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="particle_marginal_mh",
        diagnostics=diagnostics,
    )
