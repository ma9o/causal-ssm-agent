"""Marginalized Particle Gibbs inference method."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.bundle import (
    build_particle_runtime_bundle,
)
from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared import (
    build_pmcmc_mcmc_result,
    extract_grouped_public_samples,
    prepare_pmcmc_parameter_warmup,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
    _DEFAULT_AMALA_ADAPTATION_GAMMA,
    _DEFAULT_AMALA_ADAPTATION_RHO,
    _DEFAULT_AMALA_ADAPTATION_RHO_MIN,
    _DEFAULT_AMALA_ADAPTATION_TOLERANCE,
    _DEFAULT_AMALA_ADAPTATION_WINDOW,
    _DEFAULT_AMALA_DELTA_INIT,
    _DEFAULT_AMALA_DELTA_MAX,
    _DEFAULT_AMALA_DELTA_MIN,
    _DEFAULT_AMALA_GRAD_CLIP,
    _DEFAULT_AMALA_TARGET_ACCEPT,
    build_marginal_particle_gibbs_kernel,
    run_marginal_particle_gibbs,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceDiagnostics, InferenceResult
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
)

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
        DSMCLeafProposal,
    )

logger = logging.getLogger(__name__)

_DEFAULT_PARAM_STEP_SIZE_MIN = 1e-6
_DEFAULT_PARAM_STEP_SIZE_MAX = 1e3


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


def fit_marginal_particle_gibbs(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 4000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    n_particles: int = 64,
    n_parameter_particles: int = 2,
    latent_smoother: Literal["dsmc"] = "dsmc",
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "pseudo_langevin",
    amala_delta_init: float = _DEFAULT_AMALA_DELTA_INIT,
    amala_delta_min: float = _DEFAULT_AMALA_DELTA_MIN,
    amala_delta_max: float = _DEFAULT_AMALA_DELTA_MAX,
    amala_target_accept: float = _DEFAULT_AMALA_TARGET_ACCEPT,
    amala_adaptation_window: int = _DEFAULT_AMALA_ADAPTATION_WINDOW,
    amala_adaptation_tolerance: float = _DEFAULT_AMALA_ADAPTATION_TOLERANCE,
    amala_adaptation_rho: float = _DEFAULT_AMALA_ADAPTATION_RHO,
    amala_adaptation_rho_min: float = _DEFAULT_AMALA_ADAPTATION_RHO_MIN,
    amala_adaptation_gamma: float = _DEFAULT_AMALA_ADAPTATION_GAMMA,
    amala_kappa: float = 0.75,
    amala_grad_clip: float = _DEFAULT_AMALA_GRAD_CLIP,
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_exact",
    latent_block_coords: int | None = None,
    paid_mix_z_weight: float = 0.85,
    paid_mix_pilot_weight: float = 0.10,
    paid_mix_pilot_var_scale: float = 0.25,
    paid_mix_wide_mult: float = 4.0,
    latent_sign_flip_moves: bool = False,
    diagnostic_metrics_all: bool = False,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = None,
    param_step_size: float = 0.02,
    param_step_size_min: float = _DEFAULT_PARAM_STEP_SIZE_MIN,
    param_step_size_max: float = _DEFAULT_PARAM_STEP_SIZE_MAX,
    param_target_accept: float | None = None,
    adaptation_rate: float = 0.05,
    # "simple" suits m-PGibbs's noisy M=2 ensemble move-rate; dual_averaging
    # scatters/collapses the per-chain step there (see run_marginal_particle_gibbs).
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple",
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = True,
    init_method: Literal["random", "pathfinder"] = "pathfinder",
    latent_init_method: Literal["predictive"] = "predictive",
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 8,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = 0.1,
    auto_preconditioner_method: Literal["map", "none", "pathfinder"] = "pathfinder",
    auto_preconditioner_maxiter: int = 200,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    initial_positions_override: jnp.ndarray | None = None,
    # (num_chains, T, n_latent) data-conditioned reference paths (e.g. IEKS
    # smoothed paths at the init positions). Without this, the reference path
    # comes from unconditional predictive simulation, which can diverge at
    # data-informed positions of nonlinear vector fields.
    initial_latent_trajectories: jnp.ndarray | None = None,
    latent_delta: float = 0.2,
    n_ieks_iters: int = 6,
    reparam=None,
    profile_dir: str | None = None,
    profile_compile_analysis: bool = True,
    profile_runtime_trace: bool = True,
    profile_trace_start_step: int = 0,
    profile_trace_steps: int = 3,
) -> InferenceResult:
    """Fit an SSM with marginalized Particle Gibbs.

    This method targets the directly evaluable latent/parameter posterior using
    a collapsed Particle Gibbs update.
    """
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            "Unsupported marginal_particle_gibbs init_method "
            f"{init_method!r}. Supported: 'random' or 'pathfinder'."
        )
    if latent_init_method != "predictive":
        raise ValueError(
            "Unsupported marginal_particle_gibbs latent_init_method "
            f"{latent_init_method!r}. Supported: 'predictive'."
        )
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            "Unsupported marginal_particle_gibbs adaptation_scheme "
            f"{adaptation_scheme!r}. Supported: 'simple' or 'dual_averaging'."
        )

    overall_t0 = time.monotonic()
    logger.info(
        "marginal_particle_gibbs entry: chains=%d warmup=%d samples=%d T=%d "
        "n_manifest=%d n_particles=%d n_parameter_particles=%d init_method=%s",
        num_chains,
        num_warmup,
        num_samples,
        int(observations.shape[0]),
        int(observations.shape[1]) if observations.ndim >= 2 else 0,
        n_particles,
        n_parameter_particles,
        init_method,
    )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key = random.split(base_key, 3)

    phase_t0 = time.monotonic()
    logger.info("phase 1/4: building marginalized Particle Gibbs runtime bundle...")
    # The model is a continuous-time nonlinear SDE; the particle smoother always
    # discretizes it with the nonlinearity-preserving Euler-Maruyama scheme.
    # Linearised discretisation is confined to the warmup/init backend.
    scheme = LATENT_TRANSITION_EULER_MARUYAMA
    bundle = build_particle_runtime_bundle(
        model,
        observations,
        times,
        scheme=scheme,
        trace_key=trace_key,
        reparam=reparam,
    )
    logger.info(
        "phase 1/4: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        _phase_elapsed(phase_t0),
        int(bundle.cached.flat_example.shape[0]),
        len(bundle.cached.public_sites),
    )

    phase_t0 = time.monotonic()
    warmup_result = prepare_pmcmc_parameter_warmup(
        model,
        observations,
        times,
        bundle=bundle,
        method_label="marginal_particle_gibbs",
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

    pilot_means = pilot_vars = pilot_wide_vars = None
    if dsmc_leaf_proposal == "paid_mix":
        # The paid mixture leaf needs FIXED per-time pilot moments. The IEKS smoothed
        # paths at the warmup init positions provide them: means from the cross-chain
        # average, a per-coordinate scale from the paths' temporal spread plus the
        # cross-chain disagreement. These are proposal-side quantities of a fixed
        # (chain-independent) mixture component, so approximation cannot bias the
        # sampler — only the exactly-computed mixture density enters the weights —
        # and the init-only linearization policy is satisfied by construction.
        from nof1_causal_lab.models.ssm.inference.warmup.latent_init import (
            compute_ieks_latent_paths,
        )

        pilot_positions = (
            init_positions
            if init_positions is not None
            # Random init leaves positions to the runner; anchor the pilot at the
            # prior center instead (any fixed position yields a valid fixed proposal).
            else jnp.broadcast_to(
                bundle.cached.flat_example,
                (num_chains, int(bundle.cached.flat_example.shape[0])),
            )
        )
        ieks_paths = compute_ieks_latent_paths(
            model,
            observations,
            times,
            positions=pilot_positions,
            trace_key=trace_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
        )
        if initial_latent_trajectories is None:
            initial_latent_trajectories = ieks_paths
        pilot_means = jnp.mean(ieks_paths, axis=0)
        temporal_var = jnp.var(pilot_means, axis=0)
        cross_chain_var = jnp.var(ieks_paths, axis=0)
        var_floor = 1e-6 * (1.0 + temporal_var)
        core_var = temporal_var[None, :] + cross_chain_var + var_floor[None, :]
        pilot_vars = paid_mix_pilot_var_scale * core_var
        pilot_wide_vars = paid_mix_wide_mult * core_var

    sign_flip_spec = None
    if latent_sign_flip_moves:
        from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.flip import (
            build_sign_flip_spec,
        )

        sign_flip_spec = build_sign_flip_spec(model, bundle)

    phase_t0 = time.monotonic()
    logger.info("phase 3/4: building marginalized Particle Gibbs joint kernel...")
    kernel = build_marginal_particle_gibbs_kernel(
        bundle,
        num_particles=n_particles,
        num_parameter_particles=n_parameter_particles,
        param_step_size=param_step_size,
        target_accept=param_target_accept,
        min_scale=param_step_size_min,
        max_scale=param_step_size_max,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        parameter_proposal=parameter_proposal,
        latent_smoother=latent_smoother,
        latent_delta=latent_delta,
        amala_delta_init=amala_delta_init,
        amala_delta_min=amala_delta_min,
        amala_delta_max=amala_delta_max,
        amala_target_accept=amala_target_accept,
        amala_adaptation_window=amala_adaptation_window,
        amala_adaptation_tolerance=amala_adaptation_tolerance,
        amala_adaptation_rho=amala_adaptation_rho,
        amala_adaptation_rho_min=amala_adaptation_rho_min,
        amala_adaptation_gamma=amala_adaptation_gamma,
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        dsmc_leaf_proposal=dsmc_leaf_proposal,
        latent_block_coords=latent_block_coords,
        paid_mix_z_weight=paid_mix_z_weight,
        paid_mix_pilot_weight=paid_mix_pilot_weight,
        pilot_means=pilot_means,
        pilot_vars=pilot_vars,
        pilot_wide_vars=pilot_wide_vars,
        sign_flip_spec=sign_flip_spec,
        diagnostic_metrics_all=diagnostic_metrics_all,
        diagnostic_metrics=diagnostic_metrics,
    )
    run_result = run_marginal_particle_gibbs(
        bundle,
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        latent_delta=latent_delta,
        retain_latent_paths=retain_latent_paths,
        init_positions=init_positions,
        initial_latent_trajectories=initial_latent_trajectories,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
        adaptation_scheme=adaptation_scheme,
        profile_dir=profile_dir,
        profile_compile_analysis=profile_compile_analysis,
        profile_runtime_trace=profile_runtime_trace,
        profile_trace_start_step=profile_trace_start_step,
        profile_trace_steps=profile_trace_steps,
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
        backend="marginal_particle_gibbs",
    )
    diagnostic_likelihood_backend = model.make_laplace_backend(n_ieks_iters)
    chain_extra_fields = run_result["chain_extra_fields"]
    summary_extra_fields = (
        chain_extra_fields if num_samples > 0 else run_result["warmup_chain_extra_fields"]
    )
    diagnostic_summary_phase = "post_warmup" if num_samples > 0 else "warmup"
    kernel_diagnostics = cast(
        "JsonObject",
        {
            "latent_kernel": kernel.latent_smoother.algorithm,
            "latent_smoother": kernel.latent_smoother.name,
            "latent_smoother_algorithm": kernel.latent_smoother.algorithm,
            "latent_smoother_family": kernel.latent_smoother.family,
            "latent_smoother_selection": kernel.latent_smoother.selection,
            "latent_smoother_parallel": bool(kernel.latent_smoother.parallel),
            "latent_delta": float(latent_delta),
            "parameter_kernel": (
                "m_pgibbs_random_walk"
                if parameter_proposal == "random_walk"
                else "m_pgibbs_pseudo_langevin"
            ),
            "mcmc_phase_seconds": float(mcmc_phase_seconds),
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "n_particles": int(n_particles),
            "n_parameter_particles": int(n_parameter_particles),
            "parameter_proposal": parameter_proposal,
            "latent_backward_sampling": bool(kernel.latent_smoother.backward_sampling),
            "amala_delta_init": float(amala_delta_init),
            "amala_delta_min": float(amala_delta_min),
            "amala_delta_max": float(amala_delta_max),
            "amala_target_accept": float(amala_target_accept),
            "amala_adaptation_window": int(amala_adaptation_window),
            "amala_adaptation_tolerance": float(amala_adaptation_tolerance),
            "amala_adaptation_rho": float(amala_adaptation_rho),
            "amala_adaptation_rho_min": float(amala_adaptation_rho_min),
            "amala_adaptation_gamma": float(amala_adaptation_gamma),
            "amala_delta_adapted": bool(kernel.adapt_amala_delta),
            "amala_kappa": float(amala_kappa),
            "amala_grad_clip": float(amala_grad_clip),
            "dsmc_leaf_proposal": kernel.dsmc_leaf_proposal,
            "latent_transition_kind": bundle.cached.latent_transition_kind,
            "diagnostic_metrics_all": bool(diagnostic_metrics_all),
            "diagnostic_metrics": sorted(kernel.diagnostic_metrics),
            "param_step_size_initial": float(param_step_size),
            "param_step_size_min": float(param_step_size_min),
            "param_step_size_max": float(param_step_size_max),
            "param_target_accept": float(kernel.target_accept),
            "adaptation_scheme": adaptation_scheme,
            "parameter_preconditioned": bool(kernel.preconditioned),
            "diagnostic_summary_phase": diagnostic_summary_phase,
            "parameter_accept_rate": float(jnp.mean(summary_extra_fields["parameter_accept_prob"])),
            "latent_update_fraction": float(jnp.mean(summary_extra_fields["latent_accept_prob"])),
            "latent_frozen_fraction": float(jnp.mean(summary_extra_fields["latent_frozen_frac"])),
            "latent_block_coords": kernel.latent_block_coords,
            "latent_sign_flip_moves": bool(latent_sign_flip_moves),
            **(
                {
                    "sign_flip_accept_rate": float(
                        jnp.mean(summary_extra_fields["sign_flip_accept_prob"])
                    )
                }
                if "sign_flip_accept_prob" in summary_extra_fields
                else {}
            ),
            "initial_param_step_size": jax.device_get(
                run_result["initial_param_step_size"]
            ).tolist(),
            "final_param_step_size": jax.device_get(run_result["final_param_step_size"]).tolist(),
            "initial_latent_delta": jax.device_get(run_result["initial_latent_delta"]).tolist(),
            "final_latent_delta": jax.device_get(run_result["final_latent_delta"]).tolist(),
            "latent_init_method": "predictive",
            "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
                run_result["post_warmup_complete_log_posterior_mean"]
            ).tolist(),
            "parameter_warmup": warmup_result.warmup_diagnostics,
            **warmup_result.init_diagnostics,
            **warmup_result.preconditioner_diagnostics,
        },
    )
    if "latent_move_rms" in summary_extra_fields:
        kernel_diagnostics["latent_move_rms_mean"] = float(
            jnp.mean(summary_extra_fields["latent_move_rms"])
        )
    if "parameter_jump_rms" in summary_extra_fields:
        kernel_diagnostics["parameter_jump_rms_mean"] = float(
            jnp.mean(summary_extra_fields["parameter_jump_rms"])
        )
    if "reference_path_hit_rate" in summary_extra_fields:
        kernel_diagnostics["reference_path_hit_rate_mean"] = float(
            jnp.mean(summary_extra_fields["reference_path_hit_rate"])
        )
    if "selected_particle_unique_count" in summary_extra_fields:
        kernel_diagnostics["selected_particle_unique_count_mean"] = float(
            jnp.mean(summary_extra_fields["selected_particle_unique_count"])
        )
    if kernel.adapt_amala_delta:
        kernel_diagnostics["amala_grad_norm_mean"] = float(
            jnp.mean(summary_extra_fields["amala_grad_norm_mean"])
        )
        kernel_diagnostics["amala_grad_norm_max"] = float(
            jnp.max(summary_extra_fields["amala_grad_norm_max"])
        )
    diagnostics: InferenceDiagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle.cached.public_sites),
        "likelihood_backend": diagnostic_likelihood_backend,
        "marginal_particle_gibbs": kernel_diagnostics,
        "marginal_particle_gibbs_phase_extra_fields": {
            "warmup": run_result["warmup_chain_extra_fields"],
            "post_warmup": run_result["chain_extra_fields"],
            "all": run_result["all_chain_extra_fields"],
        },
        "latent_posterior_summary": run_result["latent_posterior_summary"],
        "chain_complete_log_posterior_history": run_result["complete_log_posterior_history"],
        "warmup_complete_log_posterior_history": run_result[
            "warmup_complete_log_posterior_history"
        ],
        "all_complete_log_posterior_history": run_result["all_complete_log_posterior_history"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]
    if run_result["warmup_latent_paths"] is not None:
        diagnostics["warmup_latent_paths"] = run_result["warmup_latent_paths"]
    if run_result["all_latent_paths"] is not None:
        diagnostics["all_latent_paths"] = run_result["all_latent_paths"]

    logger.info(
        "phase 4/4: posterior extraction complete in %.1fs. marginal_particle_gibbs total: %.1fs",
        _phase_elapsed(phase_t0),
        _phase_elapsed(overall_t0),
    )
    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="marginal_particle_gibbs",
        diagnostics=diagnostics,
    )
