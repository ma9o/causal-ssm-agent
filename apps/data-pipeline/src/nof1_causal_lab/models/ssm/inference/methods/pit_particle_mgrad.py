"""Particle-mGRAD latent sampler with hybrid Gibbs/NUTS parameter updates."""

from __future__ import annotations

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.models.ssm.inference.latent_trace import (
    LatentTraceConfig,
    build_latent_trace_diagnostics,
    validate_latent_trace_config,
)
from nof1_causal_lab.models.ssm.inference.methods.parameter_warmup import (
    DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    prepare_parameter_warmup,
)
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc import (
    AuxKalmanMCMCResult,
    build_auxiliary_kalman_bundle,
    build_hybrid_gibbs_nuts_parameter_kernel,
    build_pit_particle_mgrad_latent_kernel,
    initialize_ieks_latents,
    initialize_particle_smoother_latents,
    run_aux_kalman_mcmc,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.inference.utils import extract_constrained_samples

logger = logging.getLogger(__name__)

_DEFAULT_LATENT_DELTA_MIN = 1e-6
_DEFAULT_LATENT_DELTA_MAX = 1e3
_PARTICLE_MGRAD_TARGET_ACCEPT = 0.75
_PIT_AUX_CSMC_TARGET_ACCEPT = 0.50


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


def _scalar_or_summary(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        try:
            arr = np.asarray(value, dtype=float)
            return f"mean={float(arr.mean()):.4g}, shape={tuple(arr.shape)}"
        except (TypeError, ValueError):
            return repr(value)[:64]


def _default_latent_target_accept(latent_kernel_algorithm: str) -> float:
    if latent_kernel_algorithm == "particle_mgrad":
        return _PARTICLE_MGRAD_TARGET_ACCEPT
    if latent_kernel_algorithm == "pit_aux_csmc":
        return _PIT_AUX_CSMC_TARGET_ACCEPT
    raise ValueError(
        "Unknown latent_kernel_algorithm "
        f"{latent_kernel_algorithm!r}; expected 'particle_mgrad' or 'pit_aux_csmc'."
    )


def _fit_particle_latent_mcmc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    method_name: str,
    diagnostics_key: str,
    num_particles: int,
    particle_count_diagnostic_key: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    latent_delta: float,
    latent_delta_min: float | None,
    latent_delta_max: float | None,
    latent_target_accept: float | None,
    latent_kernel_algorithm: str,
    parameter_preconditioner_chol: jnp.ndarray | None,
    parameter_kernel: str,
    param_step_size: float,
    param_target_accept: float,
    param_max_num_doublings: int,
    adaptation_rate: float,
    init_scale: float,
    retain_latent_paths: bool,
    compute_latent_posterior_summary: bool,
    adaptation_scheme: str,
    init_method: str,
    latent_init_method: str,
    latent_init_num_particles: int,
    latent_init_guidance: str,
    n_ieks_iters: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    pathfinder_parallel_workers: int | None,
    pathfinder_init_scale: float | None,
    auto_preconditioner_method: str,
    auto_preconditioner_maxiter: int,
    initial_positions_override: jnp.ndarray | None,
    enable_polya_gamma: bool,
    polya_gamma_num_terms: int,
    polya_gamma_sampler: str,
    rbpf_mode: str,
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None,
    emit_per_t_log_alpha: bool,
    debug_particle_trace: bool,
    reparam,
) -> InferenceResult:
    if parameter_kernel != "hybrid_gibbs_nuts":
        raise ValueError(
            f"Unsupported {method_name} parameter kernel {parameter_kernel!r}. "
            "Supported: 'hybrid_gibbs_nuts'."
        )
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            f"Unsupported {method_name} init_method {init_method!r}. "
            "Supported: 'random' or 'pathfinder'."
        )
    if latent_init_method not in {"predictive", "particle_smoother", "ieks"}:
        raise ValueError(
            f"Unsupported {method_name} latent_init_method {latent_init_method!r}. "
            "Supported: 'predictive', 'particle_smoother', or 'ieks'."
        )
    if latent_init_guidance not in {"bootstrap", "bffg"}:
        raise ValueError(
            f"Unsupported {method_name} latent_init_guidance {latent_init_guidance!r}. "
            "Supported: 'bootstrap' or 'bffg'."
        )
    if latent_kernel_algorithm not in {"particle_mgrad", "pit_aux_csmc"}:
        raise ValueError(
            f"Unsupported {method_name} latent_kernel_algorithm {latent_kernel_algorithm!r}. "
            "Supported: 'particle_mgrad' or 'pit_aux_csmc'."
        )
    if latent_target_accept is None:
        latent_target_accept = _default_latent_target_accept(latent_kernel_algorithm)
    latent_trace_config = LatentTraceConfig(
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        debug_particle_trace=debug_particle_trace,
    )
    validate_latent_trace_config(method_name, latent_trace_config)
    if latent_delta_min is None:
        latent_delta_min = _DEFAULT_LATENT_DELTA_MIN
    if latent_delta_max is None:
        latent_delta_max = _DEFAULT_LATENT_DELTA_MAX

    overall_t0 = time.monotonic()
    T_time_log = int(observations.shape[0])
    n_manifest_log = int(observations.shape[1]) if observations.ndim >= 2 else 0
    logger.info(
        "%s entry: chains=%d warmup=%d samples=%d T=%d n_manifest=%d "
        "init_method=%s n_particles=%d parameter_kernel=%s adaptation_scheme=%s "
        "auto_preconditioner_method=%s debug_particle_trace=%s emit_per_t_log_alpha=%s",
        method_name,
        num_chains,
        num_warmup,
        num_samples,
        T_time_log,
        n_manifest_log,
        init_method,
        num_particles,
        parameter_kernel,
        adaptation_scheme,
        auto_preconditioner_method,
        debug_particle_trace,
        emit_per_t_log_alpha,
    )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key, latent_init_key = random.split(base_key, 4)

    phase_t0 = time.monotonic()
    logger.info("phase 1/5: building auxiliary Kalman bundle...")
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
        polya_gamma_num_terms=polya_gamma_num_terms,
        polya_gamma_sampler=polya_gamma_sampler,
        enable_polya_gamma=enable_polya_gamma,
        rbpf_mode=rbpf_mode,
        rbpf_marginalized_latent_indices=rbpf_marginalized_latent_indices,
    )
    if (
        latent_kernel_algorithm == "particle_mgrad"
        and str(bundle["rbpf_structure"]) == "conditional"
    ):
        raise ValueError(
            "particle_mgrad does not support conditional RBPF because the marginalized "
            "filter state is path-dependent; set latent_kernel_algorithm='pit_aux_csmc' "
            "for conditional RBPF."
        )
    logger.info(
        "phase 1/5: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        _phase_elapsed(phase_t0),
        int(bundle["flat_example"].shape[0]),
        len(bundle.get("public_sites", [])),
    )

    phase_t0 = time.monotonic()
    warmup_result = prepare_parameter_warmup(
        model,
        observations,
        times,
        bundle=bundle,
        method_label=method_name,
        phase_label="phase 2/5",
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
        prior_released_sites=DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    )
    init_positions = warmup_result.init_positions
    init_diagnostics = warmup_result.init_diagnostics
    parameter_preconditioner_chol = warmup_result.preconditioner_chol
    preconditioner_diagnostics = warmup_result.preconditioner_diagnostics
    logger.info("phase 2/5: parameter warmup ready in %.1fs", _phase_elapsed(phase_t0))

    initial_latent_trajectories = None
    if latent_init_method in {"particle_smoother", "ieks"}:
        phase_t0 = time.monotonic()
        if init_positions is None:
            init_keys = random.split(pf_sample_key, num_chains)
            init_noise = jax.vmap(
                lambda key: random.normal(
                    key,
                    bundle["flat_example"].shape,
                    dtype=bundle["flat_example"].dtype,
                )
            )(init_keys)
            init_positions = bundle["flat_example"][None, ...] + init_scale * init_noise
            init_diagnostics = {
                **init_diagnostics,
                "init_method": "random",
                "random_init_scale": float(init_scale),
            }
        seed_int = int(jax.device_get(random.randint(latent_init_key, (), 0, 2**31 - 1)))
        if latent_init_method == "particle_smoother":
            logger.info(
                "phase 2b/5: particle-smoother latent init (particles=%d, guidance=%s)...",
                latent_init_num_particles,
                latent_init_guidance,
            )
            initial_latent_trajectories, latent_init_diagnostics = (
                initialize_particle_smoother_latents(
                    bundle,
                    init_positions,
                    seed=seed_int,
                    num_particles=latent_init_num_particles,
                    guidance=latent_init_guidance,
                )
            )
        else:  # ieks
            logger.info(
                "phase 2b/5: IEKS latent init (n_ieks_iters=%d)...",
                n_ieks_iters,
            )
            initial_latent_trajectories, latent_init_diagnostics = initialize_ieks_latents(
                bundle,
                init_positions,
                model=model,
                seed=seed_int,
                n_ieks_iters=n_ieks_iters,
                reparam=reparam,
                trace_key=trace_key,
            )
        init_diagnostics = {**init_diagnostics, **latent_init_diagnostics}
        logger.info(
            "phase 2b/5: %s latent init complete in %.1fs",
            latent_init_method,
            _phase_elapsed(phase_t0),
        )
    else:
        init_diagnostics = {
            **init_diagnostics,
            "latent_init_method": "predictive",
        }

    phase_t0 = time.monotonic()
    logger.info(
        "phase 3/5: building %s latent kernel + %s "
        "parameter kernel (num_particles=%d, latent_delta=%.4g, param_step_size=%.4g, "
        "preconditioned=%s)...",
        latent_kernel_algorithm,
        parameter_kernel.upper(),
        num_particles,
        float(latent_delta),
        float(param_step_size),
        parameter_preconditioner_chol is not None,
    )
    latent_kernel_spec = build_pit_particle_mgrad_latent_kernel(
        bundle,
        delta=latent_delta,
        target_accept=latent_target_accept,
        num_particles=num_particles,
        min_scale=latent_delta_min,
        max_scale=latent_delta_max,
        debug_particle_trace=debug_particle_trace,
        latent_kernel_algorithm=latent_kernel_algorithm,
    )

    parameter_kernel_spec = build_hybrid_gibbs_nuts_parameter_kernel(
        bundle,
        step_size=param_step_size,
        target_accept=param_target_accept,
        max_num_doublings=param_max_num_doublings,
        preconditioner_chol=parameter_preconditioner_chol,
    )
    logger.info("phase 3/5: kernel specs ready in %.1fs", _phase_elapsed(phase_t0))

    phase_t0 = time.monotonic()
    logger.info(
        "phase 4/5: starting MCMC kernel; first call triggers JAX JIT compile of "
        "%s + hybrid Gibbs/NUTS + adaptation.",
        latent_kernel_spec["algorithm"],
    )
    run_result = run_aux_kalman_mcmc(
        bundle,
        latent_kernel=latent_kernel_spec,
        parameter_kernel=parameter_kernel_spec,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        retain_latent_paths=retain_latent_paths,
        adaptation_scheme=adaptation_scheme,
        init_positions=init_positions,
        initial_latent_trajectories=initial_latent_trajectories,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
    )
    mcmc_phase_seconds = _phase_elapsed(phase_t0)
    latent_acc = float(jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"]))
    param_acc = float(jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"]))
    latent_trace_diagnostics = build_latent_trace_diagnostics(
        run_result["chain_extra_fields"],
        latent_trace_config,
    )
    logger.info(
        "phase 4/5: MCMC kernel complete in %.1fs (latent_update_fraction=%.3f, "
        "param_acc=%.3f, final_latent_delta=%s, final_param_step_size=%s)",
        mcmc_phase_seconds,
        latent_acc,
        param_acc,
        _scalar_or_summary(run_result["final_latent_delta"]),
        _scalar_or_summary(run_result["final_param_step_size"]),
    )

    phase_t0 = time.monotonic()
    logger.info("phase 5/5: extracting + filtering public posterior samples...")
    flat_particles = run_result["grouped_positions"].reshape((-1, bundle["dim"]))
    constrained_samples = extract_constrained_samples(
        flat_particles,
        bundle["site_info"],
        bundle["unravel_fn"],
        model.spec,
        reparam=reparam,
        model=model,
        observations=observations,
        times=times,
    )
    public_samples = _filter_public_samples(constrained_samples, bundle["public_sites"])
    grouped_public_samples = {
        name: values.reshape((num_chains, num_samples, *values.shape[1:]))
        for name, values in public_samples.items()
    }
    diagnostic_likelihood_backend = model.make_laplace_backend(n_ieks_iters)
    mcmc = AuxKalmanMCMCResult(
        chain_samples=grouped_public_samples,
        chain_extra_fields=run_result["chain_extra_fields"],
        num_chains=num_chains,
        num_samples=num_samples,
        backend=method_name,
    )
    polya_gamma_plan = bundle["polya_gamma_plan"]
    rbpf_partition = bundle["rbpf_partition"]
    rbpf_observation_plan = bundle["rbpf_observation_plan"]
    kernel_diagnostics = {
        "latent_kernel": "pit_particle_mgrad",
        "latent_kernel_algorithm": latent_kernel_spec["algorithm"],
        "latent_kernel_family": latent_kernel_spec["family"],
        "latent_kernel_selection": latent_kernel_spec["selection"],
        "parallel_time": bool(latent_kernel_spec["parallel"]),
        "mcmc_phase_seconds": float(mcmc_phase_seconds),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "parameter_kernel": parameter_kernel,
        particle_count_diagnostic_key: num_particles,
        "adaptation_scheme": adaptation_scheme,
        "polya_gamma_enabled": bool(bundle["polya_gamma_enabled"]),
        "polya_gamma_sampler": str(bundle["polya_gamma_sampler"]),
        "polya_gamma_channels": int(jnp.sum(polya_gamma_plan.channel_mask)),
        "polya_gamma_num_terms": int(polya_gamma_plan.num_terms),
        "polya_gamma_max_integer_shape": polya_gamma_plan.max_integer_shape,
        "rbpf_enabled": bool(bundle["rbpf_enabled"]),
        "rbpf_requested": bool(bundle["rbpf_requested"]),
        "rbpf_mode": str(bundle["rbpf_mode"]),
        "rbpf_structure": str(bundle["rbpf_structure"]),
        "rbpf_carried_latent_indices": list(rbpf_partition.carried_latent_indices),
        "rbpf_marginalized_latent_indices": list(rbpf_partition.marginalized_latent_indices),
        "rbpf_partition_diagnostics": bundle["rbpf_partition_diagnostics"],
        "rbpf_observation_channels": int(jnp.sum(rbpf_observation_plan.channel_mask)),
        "rbpf_gaussian_observation_channels": int(
            jnp.sum(rbpf_observation_plan.gaussian_channel_mask)
        ),
        "rbpf_polya_gamma_observation_channels": int(
            jnp.sum(rbpf_observation_plan.polya_gamma_channel_mask)
        ),
        "latent_update_fraction": float(
            jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
        ),
        "target_accept": float(latent_target_accept),
        "parameter_accept_rate": float(
            jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
        ),
        "latent_delta_min": latent_delta_min,
        "latent_delta_max": latent_delta_max,
        "initial_param_step_size": run_result["initial_param_step_size"],
        "param_step_size_initial_guess": run_result["param_step_size_initial_guess"],
        "param_step_size_auto_tuned": run_result["param_step_size_auto_tuned"],
        "param_step_size_tuning_accept_prob": run_result[
            "param_step_size_tuning_accept_prob"
        ],
        "param_step_size_tuning_steps": run_result["param_step_size_tuning_steps"],
        "param_step_size_tuning_candidate_accept_prob": run_result[
            "param_step_size_tuning_candidate_accept_prob"
        ],
        "param_step_size_tuning_previous_accept_prob": run_result[
            "param_step_size_tuning_previous_accept_prob"
        ],
        "param_step_size_tuning_selected_previous": run_result[
            "param_step_size_tuning_selected_previous"
        ],
        "param_step_size_tuning_crossed": run_result["param_step_size_tuning_crossed"],
        "latent_adaptation_method": run_result["latent_adaptation_method"],
        "latent_window_adaptation_window_size": run_result[
            "latent_window_adaptation_window_size"
        ],
        "latent_window_acceptance_mean": run_result["latent_window_acceptance_mean"],
        "latent_window_acceptance_min": run_result["latent_window_acceptance_min"],
        "latent_window_acceptance_max": run_result["latent_window_acceptance_max"],
        "final_latent_delta": run_result["final_latent_delta"],
        "final_param_step_size": run_result["final_param_step_size"],
        "parameter_gibbs_block_count": int(parameter_kernel_spec["gibbs_block_count"]),
        "parameter_residual_dim": int(parameter_kernel_spec["residual_dim"]),
        "emit_per_t_log_alpha": bool(emit_per_t_log_alpha),
        "debug_particle_trace": bool(debug_particle_trace),
        "debug_particle_trace_fields": (
            list(latent_trace_diagnostics["particle_trace_fields"]) if debug_particle_trace else []
        ),
        "latent_trace": latent_trace_diagnostics,
        "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
            run_result["post_warmup_complete_log_posterior_mean"]
        ).tolist(),
        "parameter_warmup": warmup_result.warmup_diagnostics,
        **init_diagnostics,
        **preconditioner_diagnostics,
    }
    chain_extra_fields = run_result["chain_extra_fields"]
    if "latent_move_rms" in chain_extra_fields:
        kernel_diagnostics["latent_move_rms_mean"] = float(
            jnp.mean(chain_extra_fields["latent_move_rms"])
        )
    if "latent_move_rms_per_t" in chain_extra_fields:
        latent_dim = int(model.spec.n_latent)
        latent_move_rms_per_t = chain_extra_fields["latent_move_rms_per_t"]
        latent_esjd_per_draw = latent_dim * jnp.sum(latent_move_rms_per_t**2, axis=-1)
        latent_esjd_mean = float(jnp.mean(latent_esjd_per_draw))
        seconds_per_iteration = float(mcmc_phase_seconds) / float(num_warmup + num_samples)
        kernel_diagnostics["latent_esjd_mean"] = latent_esjd_mean
        kernel_diagnostics["latent_esjd_per_second"] = latent_esjd_mean / seconds_per_iteration
    if "latent_move_max_abs" in chain_extra_fields:
        kernel_diagnostics["latent_move_max_abs_mean"] = float(
            jnp.mean(chain_extra_fields["latent_move_max_abs"])
        )
    diagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle["public_sites"]),
        "likelihood_backend": diagnostic_likelihood_backend,
        diagnostics_key: kernel_diagnostics,
        "latent_posterior_summary": run_result["latent_posterior_summary"],
        "chain_complete_log_posterior_history": run_result["complete_log_posterior_history"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]

    logger.info(
        "phase 5/5: posterior extraction complete in %.1fs (n_public_sites=%d, "
        "draws_per_chain=%d). %s total: %.1fs",
        _phase_elapsed(phase_t0),
        len(grouped_public_samples),
        num_samples,
        method_name,
        _phase_elapsed(overall_t0),
    )

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method=method_name,
        diagnostics=diagnostics,
    )


def fit_pit_particle_mgrad(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 4000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    latent_delta: float = 0.2,
    latent_delta_min: float | None = _DEFAULT_LATENT_DELTA_MIN,
    latent_delta_max: float | None = _DEFAULT_LATENT_DELTA_MAX,
    latent_target_accept: float | None = None,
    latent_kernel_algorithm: str = "particle_mgrad",
    n_particles: int = 64,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_kernel: str = "hybrid_gibbs_nuts",
    param_step_size: float = 0.05,
    param_target_accept: float = 0.65,
    param_max_num_doublings: int = 10,
    adaptation_rate: float = 0.05,
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = True,
    adaptation_scheme: str = "dual_averaging",
    init_method: str = "pathfinder",
    latent_init_method: str = "ieks",
    latent_init_num_particles: int = 64,
    latent_init_guidance: str = "bffg",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 8,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = 0.1,
    auto_preconditioner_method: str = "pathfinder",
    auto_preconditioner_maxiter: int = 200,
    initial_positions_override: jnp.ndarray | None = None,
    enable_polya_gamma: bool = True,
    polya_gamma_num_terms: int = 64,
    polya_gamma_sampler: str = "truncated_sum",
    rbpf_mode: str = "none",
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None = None,
    emit_per_t_log_alpha: bool = False,
    debug_particle_trace: bool = False,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with the Particle-mGRAD latent kernel and parameter updates."""
    return _fit_particle_latent_mcmc(
        model,
        observations,
        times,
        method_name="pit_particle_mgrad",
        diagnostics_key="pit_particle_mgrad",
        num_particles=n_particles,
        particle_count_diagnostic_key="n_particles",
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        latent_delta=latent_delta,
        latent_delta_min=latent_delta_min,
        latent_delta_max=latent_delta_max,
        latent_target_accept=latent_target_accept,
        latent_kernel_algorithm=latent_kernel_algorithm,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        parameter_kernel=parameter_kernel,
        param_step_size=param_step_size,
        param_target_accept=param_target_accept,
        param_max_num_doublings=param_max_num_doublings,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        retain_latent_paths=retain_latent_paths,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
        adaptation_scheme=adaptation_scheme,
        init_method=init_method,
        latent_init_method=latent_init_method,
        latent_init_num_particles=latent_init_num_particles,
        latent_init_guidance=latent_init_guidance,
        n_ieks_iters=n_ieks_iters,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_parallel_workers=pathfinder_parallel_workers,
        pathfinder_init_scale=pathfinder_init_scale,
        auto_preconditioner_method=auto_preconditioner_method,
        auto_preconditioner_maxiter=auto_preconditioner_maxiter,
        initial_positions_override=initial_positions_override,
        enable_polya_gamma=enable_polya_gamma,
        polya_gamma_num_terms=polya_gamma_num_terms,
        polya_gamma_sampler=polya_gamma_sampler,
        rbpf_mode=rbpf_mode,
        rbpf_marginalized_latent_indices=rbpf_marginalized_latent_indices,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        debug_particle_trace=debug_particle_trace,
        reparam=reparam,
    )
