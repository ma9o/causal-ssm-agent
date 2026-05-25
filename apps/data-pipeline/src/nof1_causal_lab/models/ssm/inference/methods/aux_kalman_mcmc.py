"""Auxiliary Kalman MCMC sampler: blocked aux-Kalman latent + hybrid Gibbs/NUTS parameter."""

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
    build_auxiliary_kalman_latent_kernel,
    build_hybrid_gibbs_nuts_parameter_kernel,
    initialize_ieks_latents,
    initialize_particle_smoother_latents,
    run_aux_kalman_mcmc,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.inference.utils import extract_constrained_samples

logger = logging.getLogger(__name__)


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


def fit_aux_kalman_mcmc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    latent_kernel: str = "kalman",
    latent_proposal_family: str = "eq8",
    latent_delta: float = 1e-3,
    latent_target_accept: float = 0.5,
    parameter_kernel: str = "hybrid_gibbs_nuts",
    param_step_size: float = 0.02,
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
    pathfinder_init_scale: float | None = None,
    parallel_filter: bool = True,
    latent_delta_profile: str = "scalar",
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    auto_preconditioner_maxiter: int = 200,
    auto_preconditioner_method: str = "pathfinder",
    initial_positions_override: jnp.ndarray | None = None,
    emit_per_t_log_alpha: bool = False,
    debug_particle_trace: bool = False,
    enable_polya_gamma: bool = True,
    polya_gamma_num_terms: int = 64,
    polya_gamma_sampler: str = "truncated_sum",
    rbpf_mode: str = "none",
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None = None,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with blocked aux-Kalman plus hybrid Gibbs/NUTS MCMC.

    ``parallel_filter`` toggles the Corenflos/Särkkä O(log T) associative
    Kalman filter and RTS sampler used inside the auxiliary-Kalman latent
    step. Turning it off falls back to a plain O(T) sequential ``lax.scan``
    filter (identical predict/update math) — useful for benchmarking or for
    very short trajectories where the per-step constant of the associative
    scan is larger than the gain from log-depth parallelism.

    ``latent_delta_profile`` selects how the latent step size δ is distributed
    across time steps, addressing the heterogeneously-informative-observation
    pathology described in Corenflos & Särkkä §4.4. Supported values:

    * ``"scalar"`` — a single global δ adapted by ``adaptation_scheme`` (the
      shipping default; closest to the plain eq-8 sampler).
    * ``"T_minus_one_third"`` — a single scalar δ fixed at
      ``latent_delta * T**(-1/3)`` and held frozen (Remark 3.1's worst-case
      MALA-rate bound; expects ``adaptation_scheme="simple"`` +
      ``adaptation_rate=0`` to actually hold the scale).
    * ``"informativeness"`` — a per-time-step δ_t ∝ 1 / n_observed_t, rescaled
      so the mean matches ``latent_delta``. Slots with many observed channels
      get a smaller δ_t; slots with none (missing / between-anchor interval
      summaries) get a larger one, so the global accept probability is no
      longer dominated by the most informative time step.

    ``latent_proposal_family`` selects the latent auxiliary-Kalman proposal:

    * ``"eq8"`` — the reparametrised auxiliary variable used by the original
      implementation in this repo, with TULAc-tamed observation gradients.
    * ``"eq10_11"`` — the non-reparametrised Corenflos & Särkkä (2025,
      eq. 10/11) auxiliary-variable/LGSSM construction, with the raw
      observation gradient replaced by the same TULAc-tamed gradient used in
      ``"eq8"``. This is a target-preserving proposal modification, not the
      literal raw-gradient Eq. 10/11 kernel.

    Auto-preconditioner: when ``parameter_preconditioner_chol`` is ``None``,
    ``auto_preconditioner_method`` selects how the residual-NUTS mass matrix is
    built. The default ``"pathfinder"`` reuses the same best-ELBO Pathfinder
    Gaussian approximation built for per-chain initialisation — so only one
    L-BFGS-style fit is driven, not two.

    * ``"pathfinder"`` (default) — reuse Pathfinder's fitted Gaussian
      approximation and pass its covariance Cholesky directly to the
      residual-NUTS kernel.
    * ``"map"`` — run a separate internal MAP+IEKS fit and use the
      L-BFGS-B inverse-Hessian approximation. ``auto_preconditioner_maxiter``
      controls the inner optimiser budget. Heavier than ``"pathfinder"``
      because it is a full second L-BFGS on the same objective; kept for
      callers that need exact Laplace covariance at the posterior mode.
    * ``"none"`` — leave the parameter kernel unpreconditioned.

    Provide a precomputed Cholesky to skip the auto-preconditioner step
    entirely.

    Per-parameter init (default ``init_method="pathfinder"``): Pathfinder's
    Gaussian approximation initialises regular flat indices; prior-released
    sites such as Student-t ``obs_df`` instead inherit the prior-median value
    plus a small per-chain jitter. The literature-standard "MAP/Pathfinder for
    regular parameters, prior mean for variance/df parameters" pattern is
    applied at flat-index granularity.
    """
    if latent_kernel != "kalman":
        raise ValueError(
            f"Unsupported aux_kalman_mcmc latent kernel {latent_kernel!r}. Supported: 'kalman'."
        )
    if latent_proposal_family not in {"eq8", "eq10_11"}:
        raise ValueError(
            f"Unsupported aux_kalman_mcmc latent proposal family {latent_proposal_family!r}. "
            "Supported: 'eq8' or 'eq10_11'."
        )
    if parameter_kernel != "hybrid_gibbs_nuts":
        raise ValueError(
            "Unsupported aux_kalman_mcmc parameter kernel "
            f"{parameter_kernel!r}. Supported: 'hybrid_gibbs_nuts'."
        )
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            f"Unsupported aux_kalman_mcmc init_method {init_method!r}. "
            "Supported: 'random' or 'pathfinder'."
        )
    if latent_init_method not in {"predictive", "particle_smoother", "ieks"}:
        raise ValueError(
            f"Unsupported aux_kalman_mcmc latent_init_method {latent_init_method!r}. "
            "Supported: 'predictive', 'particle_smoother', or 'ieks'."
        )
    if latent_init_guidance not in {"bootstrap", "bffg"}:
        raise ValueError(
            f"Unsupported aux_kalman_mcmc latent_init_guidance {latent_init_guidance!r}. "
            "Supported: 'bootstrap' or 'bffg'."
        )
    if latent_delta_profile not in {"scalar", "T_minus_one_third", "informativeness"}:
        raise ValueError(
            f"Unsupported latent_delta_profile {latent_delta_profile!r}. "
            "Supported: 'scalar', 'T_minus_one_third', 'informativeness'."
        )
    latent_trace_config = LatentTraceConfig(
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        debug_particle_trace=debug_particle_trace,
    )
    validate_latent_trace_config("aux_kalman_mcmc", latent_trace_config)
    overall_t0 = time.monotonic()
    T_time_log = int(observations.shape[0])
    n_manifest_log = int(observations.shape[1]) if observations.ndim >= 2 else 0
    logger.info(
        "aux_kalman_mcmc entry: chains=%d warmup=%d samples=%d T=%d n_manifest=%d "
        "init_method=%s parallel_filter=%s latent_kernel=%s parameter_kernel=%s "
        "auto_preconditioner_method=%s emit_per_t_log_alpha=%s",
        num_chains,
        num_warmup,
        num_samples,
        T_time_log,
        n_manifest_log,
        init_method,
        parallel_filter,
        latent_kernel,
        parameter_kernel,
        auto_preconditioner_method,
        emit_per_t_log_alpha,
    )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key, release_key, latent_init_key = random.split(
        base_key, 5
    )

    phase_t0 = time.monotonic()
    logger.info("phase 1/6: building auxiliary Kalman bundle...")
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
    logger.info(
        "phase 1/6: bundle ready in %.1fs (dim=%d, public_sites=%d)",
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
        method_label="aux_kalman_mcmc",
        phase_label="phase 2/6",
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
        release_jitter_key=release_key,
    )
    init_positions = warmup_result.init_positions
    init_diagnostics = warmup_result.init_diagnostics
    parameter_preconditioner_chol = warmup_result.preconditioner_chol
    preconditioner_diagnostics = warmup_result.preconditioner_diagnostics
    logger.info("phase 2/6: parameter warmup ready in %.1fs", _phase_elapsed(phase_t0))

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
                "phase 2b/6: particle-smoother latent init (particles=%d, guidance=%s)...",
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
        else:
            logger.info(
                "phase 2b/6: IEKS latent init (n_ieks_iters=%d)...",
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
            "phase 2b/6: %s latent init complete in %.1fs",
            latent_init_method,
            _phase_elapsed(phase_t0),
        )
    else:
        init_diagnostics = {
            **init_diagnostics,
            "latent_init_method": "predictive",
        }

    # Resolve the δ profile. "scalar" keeps a single global δ; the other two
    # variants freeze per-time step sizes — for them, the simple exponential
    # adapter at rate 0 is the "don't touch it" path.
    effective_delta = latent_delta
    delta_profile: jnp.ndarray | None = None
    T_time = int(observations.shape[0])
    if latent_delta_profile == "T_minus_one_third":
        effective_delta = float(latent_delta) * float(T_time) ** (-1.0 / 3.0)
    elif latent_delta_profile == "informativeness":
        obs_count = jnp.sum(~jnp.isnan(observations), axis=tuple(range(1, observations.ndim)))
        info_t = jnp.maximum(obs_count.astype(jnp.float32), 1.0)
        raw_weights = 1.0 / info_t
        normalised = raw_weights / jnp.mean(raw_weights)
        delta_profile = float(latent_delta) * normalised
    phase_t0 = time.monotonic()
    logger.info(
        "phase 4/6: building latent + parameter kernel specs (parallel_filter=%s, "
        "latent_proposal_family=%s, latent_delta_profile=%s, effective_delta=%.4g, "
        "param_step_size=%.4g)...",
        parallel_filter,
        latent_proposal_family,
        latent_delta_profile,
        float(effective_delta),
        float(param_step_size),
    )
    latent_kernel_spec = build_auxiliary_kalman_latent_kernel(
        bundle,
        delta=effective_delta,
        target_accept=latent_target_accept,
        proposal_family=latent_proposal_family,
        parallel=parallel_filter,
        delta_profile=delta_profile,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
    )
    parameter_kernel_spec = build_hybrid_gibbs_nuts_parameter_kernel(
        bundle,
        step_size=param_step_size,
        target_accept=param_target_accept,
        max_num_doublings=param_max_num_doublings,
        preconditioner_chol=parameter_preconditioner_chol,
    )
    logger.info("phase 4/6: kernel specs ready in %.1fs", _phase_elapsed(phase_t0))

    phase_t0 = time.monotonic()
    logger.info(
        "phase 5/6: starting MCMC kernel — first call triggers JAX JIT compile of the "
        "parallel-Kalman scan + hybrid Gibbs/NUTS step "
        "(this can take 1-5 min on a fresh container) "
        "before any sampling iteration begins...",
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
    latent_acc = float(jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"]))
    param_acc = float(jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"]))
    latent_trace_diagnostics = build_latent_trace_diagnostics(
        run_result["chain_extra_fields"],
        latent_trace_config,
    )

    def _scalar_or_summary(value: Any) -> str:
        # Latent / parameter step sizes can be scalar OR per-time-step / per-chain
        # arrays depending on the adaptation scheme. Render either flavor compactly.
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            try:
                arr = np.asarray(value, dtype=float)
                return f"mean={float(arr.mean()):.4g}, shape={tuple(arr.shape)}"
            except (TypeError, ValueError):
                return repr(value)[:64]

    logger.info(
        "phase 5/6: MCMC kernel complete in %.1fs (latent_acc=%.3f, param_acc=%.3f, "
        "final_latent_delta=%s, final_param_step_size=%s)",
        _phase_elapsed(phase_t0),
        latent_acc,
        param_acc,
        _scalar_or_summary(run_result["final_latent_delta"]),
        _scalar_or_summary(run_result["final_param_step_size"]),
    )

    phase_t0 = time.monotonic()
    logger.info("phase 6/6: extracting + filtering public posterior samples...")
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
    )
    polya_gamma_plan = bundle["polya_gamma_plan"]
    rbpf_partition = bundle["rbpf_partition"]
    rbpf_observation_plan = bundle["rbpf_observation_plan"]
    diagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle["public_sites"]),
        "likelihood_backend": diagnostic_likelihood_backend,
        "aux_kalman_mcmc": {
            "latent_kernel": latent_kernel,
            "latent_proposal_family": latent_proposal_family,
            "parameter_kernel": parameter_kernel,
            "adaptation_scheme": adaptation_scheme,
            "parallel_filter": parallel_filter,
            "latent_delta_profile": latent_delta_profile,
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
            "latent_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
            ),
            "parameter_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
            ),
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
            "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
                run_result["post_warmup_complete_log_posterior_mean"]
            ).tolist(),
            "emit_per_t_log_alpha": bool(emit_per_t_log_alpha),
            "latent_trace": latent_trace_diagnostics,
            "parameter_warmup": warmup_result.warmup_diagnostics,
            **init_diagnostics,
            **preconditioner_diagnostics,
        },
        "latent_posterior_summary": run_result["latent_posterior_summary"],
        "chain_complete_log_posterior_history": run_result["complete_log_posterior_history"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]

    logger.info(
        "phase 6/6: posterior extraction complete in %.1fs (n_public_sites=%d, "
        "draws_per_chain=%d). aux_kalman_mcmc total: %.1fs",
        _phase_elapsed(phase_t0),
        len(grouped_public_samples),
        num_samples,
        _phase_elapsed(overall_t0),
    )

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="aux_kalman_mcmc",
        diagnostics=diagnostics,
    )
