"""PIT Particle-mGRAD latent sampler with MALA parameter updates."""

from __future__ import annotations

import logging
import time
from typing import Any

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.models.ssm.inference.methods.map import _build_map_laplace_bundle
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc import (
    AuxKalmanMCMCResult,
    build_auxiliary_kalman_bundle,
    build_mala_parameter_kernel,
    build_nuts_parameter_kernel,
    build_pit_particle_mgrad_latent_kernel,
    initialize_ieks_latents,
    initialize_particle_smoother_latents,
    run_aux_kalman_mcmc,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from nof1_causal_lab.models.ssm.inference.utils import extract_constrained_samples

logger = logging.getLogger(__name__)


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


def _pathfinder_init_positions(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    sample_key: jnp.ndarray,
    reparam,
    n_ieks_iters: int,
    num_chains: int,
    num_elbo_samples: int,
    maxiter: int,
    dtype,
    n_pathfinder_starts: int = 1,
    pathfinder_init_scale: float | None = None,
    method_label: str = "pit_particle_mgrad",
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Run Pathfinder on the IEKS-marginal log-posterior for theta.

    See the aux_kalman_mcmc twin for the multi-start (``n_pathfinder_starts > 1``)
    semantics — same behaviour and diagnostics here.

    ``pathfinder_init_scale``:
        * ``None`` (default) — sample ``num_chains`` positions from Pathfinder's
          own Gaussian approximation via :func:`blackjax.pathfinder.sample`.
          This is the classical Pathfinder workflow; on ill-conditioned
          posteriors the Gaussian covariance can be dramatically wider than
          the target, scattering chains across distant basins.
        * ``float`` — take Pathfinder's mode ``state.position`` as the common
          centre and perturb each chain with ``pathfinder_init_scale * randn``.
          Works around the over-wide-Gaussian failure mode by initialising
          all chains in a controlled neighbourhood of the ELBO optimum.
    """
    if n_pathfinder_starts < 1:
        raise ValueError("n_pathfinder_starts must be >= 1.")
    pathfinder_t0 = time.monotonic()
    print(
        "pathfinder progress: "
        f"building laplace bundle starts={n_pathfinder_starts} "
        f"maxiter={maxiter} elbo_samples={num_elbo_samples}",
        flush=True,
    )
    backend = model.make_laplace_backend(n_ieks_iters)
    laplace_bundle = _build_map_laplace_bundle(
        model, observations, times, trace_key, backend, reparam
    )
    print(
        "pathfinder progress: "
        f"laplace bundle ready elapsed={time.monotonic() - pathfinder_t0:.1f}s "
        f"dim={int(laplace_bundle['flat_example'].shape[0])}",
        flush=True,
    )

    def _log_posterior_for_dataset(z):
        return laplace_bundle["log_posterior_fn"](
            z,
            observations,
            times,
            latent_mode_init=None,
        )

    start_keys = random.split(pathfinder_key, n_pathfinder_starts)
    states: list[Any] = []
    elbos: list[float] = []
    for start_idx, start_key in enumerate(start_keys, start=1):
        start_t0 = time.monotonic()
        print(
            "pathfinder progress: "
            f"start {start_idx}/{n_pathfinder_starts} begin "
            f"elapsed={start_t0 - pathfinder_t0:.1f}s",
            flush=True,
        )
        state_k, _ = pathfinder.approximate(
            start_key,
            _log_posterior_for_dataset,
            laplace_bundle["flat_example"],
            num_samples=num_elbo_samples,
            maxiter=maxiter,
        )
        elbo_k = float(jax.device_get(state_k.elbo))
        finite_position = bool(jax.device_get(jnp.all(jnp.isfinite(state_k.position))))
        finite_elbo = bool(jnp.isfinite(elbo_k))
        print(
            "pathfinder progress: "
            f"start {start_idx}/{n_pathfinder_starts} done "
            f"elapsed={time.monotonic() - pathfinder_t0:.1f}s "
            f"start_seconds={time.monotonic() - start_t0:.1f}s "
            f"elbo={elbo_k:.3f} finite_position={finite_position} finite_elbo={finite_elbo}",
            flush=True,
        )
        if not finite_position:
            continue
        if not finite_elbo:
            continue
        states.append(state_k)
        elbos.append(elbo_k)
    if not states:
        raise RuntimeError(
            "All pathfinder starts produced non-finite ELBO or position; "
            f"cannot seed {method_label} chains."
        )
    best_idx = int(max(range(len(elbos)), key=lambda i: elbos[i]))
    best_state = states[best_idx]
    if pathfinder_init_scale is None:
        positions, _log_q = pathfinder.sample(sample_key, best_state, num_samples=num_chains)
        sampling_mode = "pathfinder_gaussian"
    else:
        noise = random.normal(sample_key, (num_chains, best_state.position.shape[0]), dtype=dtype)
        positions = best_state.position[None, :] + float(pathfinder_init_scale) * noise
        sampling_mode = "mode_plus_scaled_normal"
    positions = jnp.asarray(positions, dtype=dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError(
            f"Pathfinder returned non-finite chain-init positions for {method_label}."
        )
    print(
        "pathfinder progress: "
        f"complete elapsed={time.monotonic() - pathfinder_t0:.1f}s "
        f"finite_starts={len(states)}/{n_pathfinder_starts} "
        f"best_elbo={elbos[best_idx]:.3f} "
        f"spread={max(elbos) - min(elbos):.3f} sampling_mode={sampling_mode}",
        flush=True,
    )
    diagnostics = {
        "init_method": "pathfinder",
        "pathfinder_sampling_mode": sampling_mode,
        "pathfinder_init_scale": pathfinder_init_scale,
        "n_pathfinder_starts": n_pathfinder_starts,
        "n_pathfinder_starts_finite": len(states),
        "best_pathfinder_elbo": elbos[best_idx],
        "pathfinder_elbo": elbos[best_idx],
        "pathfinder_elbo_min": min(elbos),
        "pathfinder_elbo_max": max(elbos),
        "pathfinder_elbo_spread": max(elbos) - min(elbos),
        "pathfinder_elbos": elbos,
    }
    return positions, diagnostics


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
    latent_target_accept: float,
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
    pathfinder_init_scale: float | None,
    initial_positions_override: jnp.ndarray | None,
    enable_polya_gamma: bool,
    polya_gamma_num_terms: int,
    polya_gamma_sampler: str,
    rbpf_mode: str,
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None,
    reparam,
) -> InferenceResult:
    if parameter_kernel not in {"mala", "nuts"}:
        raise ValueError(
            f"Unsupported {method_name} parameter kernel {parameter_kernel!r}. "
            "Supported: 'mala' or 'nuts'."
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

    overall_t0 = time.monotonic()
    T_time_log = int(observations.shape[0])
    n_manifest_log = int(observations.shape[1]) if observations.ndim >= 2 else 0
    logger.info(
        "%s entry: chains=%d warmup=%d samples=%d T=%d n_manifest=%d "
        "init_method=%s n_particles=%d parameter_kernel=%s adaptation_scheme=%s",
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
    logger.info(
        "phase 1/5: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        _phase_elapsed(phase_t0),
        int(bundle["flat_example"].shape[0]),
        len(bundle.get("public_sites", [])),
    )

    phase_t0 = time.monotonic()
    logger.info(
        "phase 2/5: building PIT dSMC particle-mGRAD latent kernel + %s "
        "parameter kernel (num_particles=%d, latent_delta=%.4g, param_step_size=%.4g)...",
        parameter_kernel.upper(),
        num_particles,
        float(latent_delta),
        float(param_step_size),
    )
    latent_kernel_spec = build_pit_particle_mgrad_latent_kernel(
        bundle,
        delta=latent_delta,
        target_accept=latent_target_accept,
        num_particles=num_particles,
        min_scale=latent_delta_min,
        max_scale=latent_delta_max,
    )

    if parameter_kernel == "mala":
        parameter_kernel_spec = build_mala_parameter_kernel(
            bundle,
            step_size=param_step_size,
            target_accept=param_target_accept,
            preconditioner_chol=parameter_preconditioner_chol,
        )
    else:
        parameter_kernel_spec = build_nuts_parameter_kernel(
            bundle,
            step_size=param_step_size,
            target_accept=param_target_accept,
            max_num_doublings=param_max_num_doublings,
            preconditioner_chol=parameter_preconditioner_chol,
        )
    logger.info("phase 2/5: kernel specs ready in %.1fs", _phase_elapsed(phase_t0))

    phase_t0 = time.monotonic()
    init_positions = None
    init_diagnostics: dict[str, Any] = {"init_method": init_method}
    if initial_positions_override is not None:
        init_positions = jnp.asarray(initial_positions_override, dtype=bundle["flat_example"].dtype)
        if init_positions.shape != (num_chains, int(bundle["flat_example"].shape[0])):
            raise ValueError(
                "initial_positions_override must have shape (num_chains, dim); got "
                f"{init_positions.shape}"
            )
        init_diagnostics = {"init_method": "user_provided"}
        logger.info(
            "phase 3/5: init positions = user_provided override (%.1fs)",
            _phase_elapsed(phase_t0),
        )
    elif init_method == "pathfinder":
        logger.info(
            "phase 3/5: blackjax-pathfinder warmup (n_starts=%d, maxiter=%d, "
            "n_ieks_iters=%d, elbo_samples=%d)...",
            n_pathfinder_starts,
            pathfinder_maxiter,
            n_ieks_iters,
            pathfinder_num_elbo_samples,
        )
        init_positions, init_diagnostics = _pathfinder_init_positions(
            model,
            observations,
            times,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
            sample_key=pf_sample_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
            num_chains=num_chains,
            num_elbo_samples=pathfinder_num_elbo_samples,
            maxiter=pathfinder_maxiter,
            dtype=bundle["flat_example"].dtype,
            n_pathfinder_starts=n_pathfinder_starts,
            pathfinder_init_scale=pathfinder_init_scale,
            method_label=method_name,
        )
        best_elbo_log = init_diagnostics.get("best_pathfinder_elbo")
        logger.info(
            "phase 3/5: pathfinder init complete in %.1fs (best_elbo=%s, "
            "n_starts_finite=%s, sampling_mode=%s)",
            _phase_elapsed(phase_t0),
            f"{best_elbo_log:.2f}" if isinstance(best_elbo_log, (int, float)) else "n/a",
            init_diagnostics.get("n_pathfinder_starts_finite", "n/a"),
            init_diagnostics.get("pathfinder_sampling_mode") or init_diagnostics.get("init_method"),
        )
    else:
        logger.info(
            "phase 3/5: init_method=random — chains start from prior draws (%.1fs)",
            _phase_elapsed(phase_t0),
        )

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
                "phase 3b/5: particle-smoother latent init (particles=%d, guidance=%s)...",
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
                "phase 3b/5: IEKS latent init (n_ieks_iters=%d)...",
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
            "phase 3b/5: %s latent init complete in %.1fs",
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
        "phase 4/5: starting MCMC kernel — first call triggers JAX JIT compile of "
        "the divide-and-conquer particle smoother (PIT dSMC) + MALA + adaptation.",
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
        compute_latent_posterior_summary=compute_latent_posterior_summary,
    )
    latent_acc = float(jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"]))
    param_acc = float(jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"]))
    logger.info(
        "phase 4/5: MCMC kernel complete in %.1fs (latent_update_fraction=%.3f, "
        "param_acc=%.3f, final_latent_delta=%s, final_param_step_size=%s)",
        _phase_elapsed(phase_t0),
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
        "latent_kernel_algorithm": "conditional_rbpf_csmc"
        if bundle["rbpf_structure"] == "conditional"
        else "pit_dsmc",
        "parallel_time": bundle["rbpf_structure"] != "conditional",
        "parameter_kernel": parameter_kernel,
        particle_count_diagnostic_key: num_particles,
        "adaptation_scheme": adaptation_scheme,
        "polya_gamma_enabled": bool(bundle["polya_gamma_enabled"]),
        "polya_gamma_sampler": str(bundle["polya_gamma_sampler"]),
        "polya_gamma_channels": int(jnp.sum(polya_gamma_plan.channel_mask)),
        "polya_gamma_num_terms": int(polya_gamma_plan.num_terms),
        "rbpf_enabled": bool(bundle["rbpf_enabled"]),
        "rbpf_requested": bool(bundle["rbpf_requested"]),
        "rbpf_mode": str(bundle["rbpf_mode"]),
        "rbpf_structure": str(bundle["rbpf_structure"]),
        "rbpf_carried_latent_indices": list(rbpf_partition.carried_latent_indices),
        "rbpf_marginalized_latent_indices": list(rbpf_partition.marginalized_latent_indices),
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
        "parameter_accept_rate": float(
            jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
        ),
        "latent_delta_min": latent_delta_min,
        "latent_delta_max": latent_delta_max,
        "final_latent_delta": run_result["final_latent_delta"],
        "final_param_step_size": run_result["final_param_step_size"],
        "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
            run_result["post_warmup_complete_log_posterior_mean"]
        ).tolist(),
        **init_diagnostics,
    }
    chain_extra_fields = run_result["chain_extra_fields"]
    if "latent_move_rms" in chain_extra_fields:
        kernel_diagnostics["latent_move_rms_mean"] = float(
            jnp.mean(chain_extra_fields["latent_move_rms"])
        )
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
    latent_delta_min: float | None = None,
    latent_delta_max: float | None = None,
    latent_target_accept: float = 0.5,
    n_particles: int = 64,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_kernel: str = "mala",
    param_step_size: float = 0.05,
    param_target_accept: float = 0.57,
    param_max_num_doublings: int = 10,
    adaptation_rate: float = 0.05,
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = True,
    adaptation_scheme: str = "simple",
    init_method: str = "pathfinder",
    latent_init_method: str = "ieks",
    latent_init_num_particles: int = 64,
    latent_init_guidance: str = "bffg",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 4,
    pathfinder_init_scale: float | None = 0.1,
    initial_positions_override: jnp.ndarray | None = None,
    enable_polya_gamma: bool = True,
    polya_gamma_num_terms: int = 64,
    polya_gamma_sampler: str = "truncated_sum",
    rbpf_mode: str = "none",
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None = None,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with the PIT Particle-mGRAD latent kernel and parameter updates."""
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
        pathfinder_init_scale=pathfinder_init_scale,
        initial_positions_override=initial_positions_override,
        enable_polya_gamma=enable_polya_gamma,
        polya_gamma_num_terms=polya_gamma_num_terms,
        polya_gamma_sampler=polya_gamma_sampler,
        rbpf_mode=rbpf_mode,
        rbpf_marginalized_latent_indices=rbpf_marginalized_latent_indices,
        reparam=reparam,
    )
