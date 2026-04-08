"""Shared tempered SMC loop for parameter inference.

Provides `fit_tempered_smc()` and `run_tempered_smc()`, the single
implementation site for tempered-SMC-backed inference.

Bridges the prior-posterior gap via a tempering ladder beta_0=0 -> beta_K=1,
with MH-corrected HMC mutations at each level. Supports adaptive tempering,
waste-free recycling, multi-step leapfrog, and precision preconditioning.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.autoreparam import reparam_cache_key
from causal_ssm_agent.models.ssm.inference import InferenceMethod, InferenceResult
from causal_ssm_agent.models.ssm.inference.engines.mcmc_utils import (
    HMC_TARGET_ACCEPT,
    RWM_TARGET_ACCEPT,
    compute_weighted_chol_mass,
    find_next_beta,
    hmc_step,
)
from causal_ssm_agent.models.ssm.inference.utils import (
    _build_eval_fns,
    _discover_sites,
    extract_constrained_samples,
)

logger = get_prefect_logger(__name__)


def _adapt_step_size(
    eps: float | jax.Array,
    avg_accept: float | jax.Array,
    target_accept: float | jax.Array,
    gain: float = 0.1,
) -> jax.Array:
    """Dual-averaging step size adaptation on log scale."""
    log_eps = jnp.log(jnp.array(eps)) + gain * (avg_accept - target_accept)
    return jnp.clip(jnp.exp(log_eps), 1e-5, 2.0)


def _build_tempered_smc_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    trace_key,
    likelihood_backend,
    reparam,
    n_mh_steps: int,
    n_leapfrog: int,
) -> dict[str, Any]:
    """Build the cached traced/JITed artifacts for a tempered SMC configuration."""
    site_info = _discover_sites(
        model, observations, times, trace_key, likelihood_backend, reparam=reparam
    )
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    dim = flat_example.shape[0]

    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model,
        observations,
        times,
        site_info,
        unravel_fn,
        likelihood_backend=likelihood_backend,
        reparam=reparam,
    )
    log_lik_val_and_grad = jax.value_and_grad(log_lik_fn)
    log_prior_unc_val_and_grad = jax.value_and_grad(log_prior_unc_fn)

    def _safe_lik_val_and_grad(z):
        with jax.named_scope("tempered_smc/likelihood_value_and_grad"):
            val, grad = log_lik_val_and_grad(z)
            safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
            safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_val, safe_grad

    batch_lik_val_and_grad = jax.jit(jax.vmap(_safe_lik_val_and_grad))

    def _tempered_val_and_grad(z, beta):
        with jax.named_scope("tempered_smc/tempered_target_value_and_grad"):
            lik_val, lik_grad = log_lik_val_and_grad(z)
            prior_val, prior_grad = log_prior_unc_val_and_grad(z)
            val = prior_val + beta * lik_val
            grad = prior_grad + beta * lik_grad
            safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
            safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_val, safe_grad

    def _hmc_scan_body(carry, rng_key, beta, eps, chol_mass):
        z, n_accept = carry

        def tempered_vg(z_):
            return _tempered_val_and_grad(z_, beta)

        z_new, accepted, _ = hmc_step(rng_key, z, tempered_vg, eps, chol_mass, n_leapfrog)
        return (z_new, n_accept + accepted.astype(jnp.int32)), None

    def _mutate_particle(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            return _hmc_scan_body(carry, key, beta, eps, chol_mass)

        (z_final, n_accept), _ = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return z_final, n_accept

    def _mutate_particle_wastefree(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), z_new

        (_, n_acc), all_z = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return all_z, n_acc

    def _mutate_batch(rng_key, particles, beta, eps, chol_mass):
        keys = random.split(rng_key, particles.shape[0])
        return jax.vmap(lambda k, z: _mutate_particle(k, z, beta, eps, chol_mass))(keys, particles)

    def _mutate_batch_wastefree(rng_key, particles_M, beta, eps, chol_mass):
        keys = random.split(rng_key, particles_M.shape[0])
        return jax.vmap(lambda k, z: _mutate_particle_wastefree(k, z, beta, eps, chol_mass))(
            keys, particles_M
        )

    def _pilot_adapt(rng_key, particles, eps, chol_mass, target_accept):
        zero = jnp.asarray(0.0, dtype=eps.dtype)

        def body(carry, step_idx):
            def _active(state):
                rng_key, particles, eps, _done, _avg_accept = state
                rng_key, mutate_key = random.split(rng_key)
                particles_new, n_accepts = _mutate_batch(
                    mutate_key, particles, zero, eps, chol_mass
                )
                avg_accept_new = jnp.mean(n_accepts) / n_mh_steps
                eps_new = _adapt_step_size(eps, avg_accept_new, target_accept, gain=0.5)
                converged = (step_idx >= 5) & (jnp.abs(avg_accept_new - target_accept) < 0.1)
                return rng_key, particles_new, eps_new, converged, avg_accept_new

            _rng_key, _particles, _eps, done, _avg_accept = carry
            new_state = jax.lax.cond(done, lambda state: state, _active, carry)
            return new_state, None

        init = (rng_key, particles, eps, jnp.asarray(False), jnp.asarray(0.0, dtype=eps.dtype))
        (rng_key, particles, eps, done, avg_accept), _ = jax.lax.scan(body, init, jnp.arange(30))
        return rng_key, particles, eps, done, avg_accept

    def _standard_mutation_rounds(rng_key, particles, beta_k, eps, chol_mass, target_accept):
        proposals_per_round = jnp.asarray(particles.shape[0] * n_mh_steps, dtype=eps.dtype)

        def body(carry, step_idx):
            def _active(state):
                (
                    rng_key,
                    particles,
                    eps,
                    _done,
                    total_accepts,
                    _n_rounds,
                    _round_accept_rate,
                ) = state
                rng_key, mutate_key = random.split(rng_key)
                particles_new, n_accepts = _mutate_batch(
                    mutate_key, particles, beta_k, eps, chol_mass
                )
                round_accepts = jnp.sum(n_accepts).astype(eps.dtype)
                round_accept_rate_new = round_accepts / proposals_per_round
                eps_new = _adapt_step_size(eps, round_accept_rate_new, target_accept)
                stop_now = (step_idx > 0) & (round_accept_rate_new > 0.2)
                return (
                    rng_key,
                    particles_new,
                    eps_new,
                    stop_now,
                    total_accepts + round_accepts,
                    jnp.asarray(step_idx + 1, dtype=jnp.int32),
                    round_accept_rate_new,
                )

            new_state = jax.lax.cond(carry[3], lambda state: state, _active, carry)
            return new_state, None

        init = (
            rng_key,
            particles,
            eps,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=eps.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=eps.dtype),
        )
        (
            (
                rng_key,
                particles,
                eps,
                _done,
                total_accepts,
                n_rounds,
                _last_round_accept_rate,
            ),
            _,
        ) = jax.lax.scan(body, init, jnp.arange(5))
        total_proposals = jnp.maximum(n_rounds.astype(eps.dtype) * proposals_per_round, 1.0)
        avg_accept = total_accepts / total_proposals
        return rng_key, particles, eps, avg_accept, n_rounds

    return {
        "dim": dim,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "batch_lik_val_and_grad": batch_lik_val_and_grad,
        "mutate_batch_jit": jax.jit(_mutate_batch),
        "mutate_batch_wastefree_jit": jax.jit(_mutate_batch_wastefree),
        "pilot_adapt_jit": jax.jit(_pilot_adapt),
        "standard_mutation_rounds_jit": jax.jit(_standard_mutation_rounds),
    }


def run_tempered_smc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    n_leapfrog: int = 5,
    method_name: InferenceMethod = "tempered_smc",
    likelihood_backend=None,
    extra_diagnostics: dict[str, Any] | None = None,
    print_prefix: str = "Tempered SMC",
    reparam=None,
) -> InferenceResult:
    """Run tempered SMC with preconditioned HMC/MALA mutations.

    This is the shared implementation used by tempered_smc, laplace_em,
    structured_vi, and dpf. Each method calls this with different
    method_name and extra_diagnostics.

    The posterior output is all N particles at beta=1.0 (the full posterior
    temperature), optionally refined with extra MCMC mutation rounds for
    mixing. This is the standard SMC output convention.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        n_outer: max tempering levels (safety bound for adaptive, exact for linear)
        n_csmc_particles: N -- number of parameter particles
        n_mh_steps: number of HMC mutation steps per round
        param_step_size: initial leapfrog step size (epsilon), adapted online
        n_warmup: extra MCMC mutation rounds at beta=1.0 for mixing (default: 5)
        target_accept: target MH acceptance rate (default: 0.44 for MALA, 0.65 for HMC)
        seed: random seed
        adaptive_tempering: use ESS-based bisection for tempering schedule
        target_ess_ratio: target ESS as fraction of N for adaptive tempering
        waste_free: use waste-free particle recycling
        n_leapfrog: number of leapfrog steps (1 = MALA, >1 = HMC)
        method_name: name for InferenceResult.method
        extra_diagnostics: additional diagnostics to merge into output
        print_prefix: prefix for progress messages

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    # Default target acceptance depends on n_leapfrog
    if target_accept is None:
        target_accept = HMC_TARGET_ACCEPT if n_leapfrog > 1 else RWM_TARGET_ACCEPT

    rng_key = random.PRNGKey(seed)
    N = n_csmc_particles

    # Validate waste-free constraint
    if waste_free and N % n_mh_steps != 0:
        raise ValueError(
            f"waste_free requires N % n_mh_steps == 0, got N={N}, n_mh_steps={n_mh_steps}"
        )

    if likelihood_backend is None:
        raise ValueError(
            "likelihood_backend is required. Use model.make_likelihood_backend() for the default."
        )

    started_at = time.monotonic()
    logger.info(
        "%s: preparing inference kernels (obs_shape=%s time_shape=%s n_mh=%s n_leapfrog=%s)",
        print_prefix,
        tuple(observations.shape),
        tuple(times.shape),
        n_mh_steps,
        n_leapfrog,
    )

    rng_key, trace_key = random.split(rng_key)
    reparam_key = reparam_cache_key(reparam)
    with jax.profiler.TraceAnnotation(f"{method_name}/build_tempered_smc_bundle"):
        if reparam_key is None:
            cached_bundle = _build_tempered_smc_bundle(
                model,
                observations,
                times,
                trace_key,
                likelihood_backend,
                reparam,
                n_mh_steps,
                n_leapfrog,
            )
        else:
            cache_key = (
                "tempered_smc_core",
                id(likelihood_backend),
                tuple(observations.shape),
                tuple(times.shape),
                n_mh_steps,
                n_leapfrog,
                *reparam_key,
            )
            cached_bundle = model.get_cached_artifact(
                cache_key,
                lambda: _build_tempered_smc_bundle(
                    model,
                    observations,
                    times,
                    trace_key,
                    likelihood_backend,
                    reparam,
                    n_mh_steps,
                    n_leapfrog,
                ),
            )

    D = cached_bundle["dim"]
    site_info = cached_bundle["site_info"]
    unravel_fn = cached_bundle["unravel_fn"]
    batch_lik_val_and_grad = cached_bundle["batch_lik_val_and_grad"]
    _mutate_batch_jit = cached_bundle["mutate_batch_jit"]
    _mutate_batch_wastefree_jit = cached_bundle["mutate_batch_wastefree_jit"]
    _pilot_adapt_jit = cached_bundle["pilot_adapt_jit"]
    _standard_mutation_rounds_jit = cached_bundle["standard_mutation_rounds_jit"]
    logger.info(
        "%s: inference kernels ready in %.1fs; parameter_dim=%s traced_sites=%s",
        print_prefix,
        time.monotonic() - started_at,
        D,
        len(site_info),
    )

    # 3. Initialize N particles from prior
    eps = jnp.asarray(param_step_size, dtype=observations.dtype)
    mode_tag = "adaptive" if adaptive_tempering else "linear"
    wf_tag = "+waste-free" if waste_free else ""
    hmc_tag = f"+HMC(L={n_leapfrog})" if n_leapfrog > 1 else ""
    logger.info(
        "%s [%s%s%s]: N=%s, K=%s, D=%s, n_mh=%s, eps=%s, target_accept=%s",
        print_prefix,
        mode_tag,
        wf_tag,
        hmc_tag,
        N,
        n_outer,
        D,
        n_mh_steps,
        float(jax.device_get(eps)),
        target_accept,
    )
    logger.info("  Initializing %s particles from prior...", N)

    with jax.profiler.TraceAnnotation(f"{method_name}/initialize_prior_particles"):
        parts = []
        for name in sorted(site_info.keys()):
            info = site_info[name]
            rng_key, sample_key = random.split(rng_key)
            prior_samples = info["distribution"].sample(sample_key, (N,))
            unc_samples = info["transform"].inv(prior_samples)
            parts.append(unc_samples.reshape(N, -1))

        particles = jnp.concatenate(parts, axis=1)  # (N, D)

        # Initial mass matrix from prior particle covariance (uniform weights)
        chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)
    logger.info(
        "  Prior particles ready in %.1fs; initial mass matrix estimated.",
        time.monotonic() - started_at,
    )

    # ===================================================================
    # Pilot: tune eps at prior (beta=0) before tempering
    # ===================================================================
    logger.info(
        "  Pilot: adapting step size at prior (elapsed=%.1fs)...", time.monotonic() - started_at
    )
    with jax.profiler.TraceAnnotation(f"{method_name}/pilot_adapt"):
        rng_key, particles, eps, pilot_converged, avg_accept_arr = _pilot_adapt_jit(
            rng_key,
            particles,
            eps,
            chol_mass,
            target_accept,
        )
    avg_accept = float(jax.device_get(avg_accept_arr))
    eps_host = float(jax.device_get(eps))
    if bool(jax.device_get(pilot_converged)):
        logger.info(
            "    pilot converged: accept=%.2f eps=%.4f elapsed=%.1fs",
            avg_accept,
            eps_host,
            time.monotonic() - started_at,
        )
    else:
        logger.info(
            "    pilot done: accept=%.2f eps=%.4f elapsed=%.1fs",
            avg_accept,
            eps_host,
            time.monotonic() - started_at,
        )

    # Recompute after pilot diversification
    logger.info("  Refreshing initial log-likelihood batch after pilot...")
    with jax.profiler.TraceAnnotation(f"{method_name}/initial_loglik_batch"):
        log_liks, _ = batch_lik_val_and_grad(particles)
        chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)
    logger.info(
        "  Initial log-likelihood batch ready in %.1fs; entering tempering ladder.",
        time.monotonic() - started_at,
    )

    logw = jnp.zeros(N)  # uniform weights at beta=0

    # Diagnostics
    accept_rates = []
    ess_history = []
    eps_history = []
    beta_schedule = []

    beta_prev = 0.0
    level = 0

    # Waste-free parameters
    M = N // n_mh_steps if waste_free else N  # resample count for waste-free

    # 5. Tempering loop
    while beta_prev < 1.0 and level < n_outer:
        with jax.profiler.TraceAnnotation(f"{method_name}/tempering_level_{level + 1}"):
            # a. Select next beta
            if adaptive_tempering:
                beta_k = find_next_beta(logw, log_liks, beta_prev, target_ess_ratio, N)
            else:
                beta_k = float(level + 1) / n_outer

            beta_schedule.append(beta_k)

            # b. Incremental reweight: logw += (beta_k - beta_{k-1}) * log_lik
            logw = logw + (beta_k - beta_prev) * log_liks

            # Normalize and compute ESS
            lse = jax.nn.logsumexp(logw)
            log_wn = logw - lse
            wn = jnp.exp(log_wn)
            ess_arr = 1.0 / jnp.sum(wn**2)
            ess = float(jax.device_get(ess_arr))
            ess_history.append(ess)
            mutation_tag = (
                " [waste-free]" if waste_free else " [will resample]" if ess < N / 2 else ""
            )
            logger.info(
                "  step %s entering mutation: beta %.3f -> %.3f  ESS=%.1f/%s%s  elapsed=%.1fs",
                level + 1,
                beta_prev,
                beta_k,
                ess,
                N,
                mutation_tag,
                time.monotonic() - started_at,
            )

            # c. Update mass matrix only when ESS is healthy
            if ess > N / 4:
                chol_mass = compute_weighted_chol_mass(particles, logw, D)

            # d. Resample and mutate
            if waste_free:
                with jax.profiler.TraceAnnotation(f"{method_name}/waste_free_mutation"):
                    rng_key, resample_key, mutate_key = random.split(rng_key, 3)
                    idx = _systematic_resample(resample_key, wn, M)
                    resampled = particles[idx]

                    all_trajs, n_accs = _mutate_batch_wastefree_jit(
                        mutate_key, resampled, beta_k, eps, chol_mass
                    )
                    particles = all_trajs.reshape(N, D)
                    logw = jnp.full(N, -jnp.log(float(N)))

                    avg_accept_arr = jnp.mean(n_accs) / n_mh_steps
                    avg_accept = float(jax.device_get(avg_accept_arr))
                    n_rounds = 1
                    eps = _adapt_step_size(eps, avg_accept_arr, target_accept)
            else:
                with jax.profiler.TraceAnnotation(f"{method_name}/standard_mutation"):
                    did_resample = False
                    if ess < N / 2:
                        rng_key, resample_key = random.split(rng_key)
                        idx = _systematic_resample(resample_key, wn, N)
                        particles = particles[idx]
                        log_liks = log_liks[idx]
                        logw = jnp.full(N, -jnp.log(float(N)))
                        did_resample = True

                    (
                        rng_key,
                        particles,
                        eps,
                        avg_accept_arr,
                        n_rounds_arr,
                    ) = _standard_mutation_rounds_jit(
                        rng_key,
                        particles,
                        beta_k,
                        eps,
                        chol_mass,
                        target_accept,
                    )
                    avg_accept = float(jax.device_get(avg_accept_arr))
                    n_rounds = int(jax.device_get(n_rounds_arr))

            accept_rates.append(avg_accept)
            eps_history.append(eps)

            with jax.profiler.TraceAnnotation(f"{method_name}/refresh_loglik_batch"):
                log_liks, _ = batch_lik_val_and_grad(particles)

        resamp_tag = ""
        if not waste_free and did_resample:
            resamp_tag = " [resampled]"
        elif waste_free:
            resamp_tag = " [waste-free]"

        logger.info(
            "  step %s done: beta=%.3f  ESS=%.1f/%s  accept=%.2f  eps=%.4f  "
            "rounds=%s%s  elapsed=%.1fs",
            level + 1,
            beta_k,
            ess,
            N,
            avg_accept,
            float(jax.device_get(eps)),
            n_rounds,
            resamp_tag,
            time.monotonic() - started_at,
        )

        beta_prev = beta_k
        level += 1

    actual_levels = level

    # 6. Extra MCMC rounds at beta=1.0 for posterior mixing.
    # n_warmup controls how many extra rounds to run (default: 5).
    n_mixing_rounds = n_warmup if n_warmup is not None else 5
    if n_mixing_rounds > 0 and beta_prev >= 1.0:
        logger.info(
            "  Running %s extra mixing rounds at beta=1.0 (elapsed=%.1fs)...",
            n_mixing_rounds,
            time.monotonic() - started_at,
        )
        with jax.profiler.TraceAnnotation(f"{method_name}/posterior_mixing"):
            for _mix_round in range(n_mixing_rounds):
                with jax.profiler.TraceAnnotation(f"{method_name}/mix_round_{_mix_round + 1}"):
                    rng_key, mutate_key = random.split(rng_key)
                    particles, n_accepts = _mutate_batch_jit(
                        mutate_key, particles, 1.0, eps, chol_mass
                    )
                    mix_accept_arr = jnp.mean(n_accepts) / n_mh_steps
                    mix_accept = float(jax.device_get(mix_accept_arr))
                    eps = _adapt_step_size(eps, mix_accept_arr, target_accept)
        logger.info(
            "    mixing done: accept=%.2f eps=%.4f elapsed=%.1fs",
            mix_accept,
            float(jax.device_get(eps)),
            time.monotonic() - started_at,
        )

    # Posterior = all N particles at beta=1.0
    chain_particles = particles  # (N, D)

    logger.info("  Extracting posterior samples from %s particles...", N)
    with jax.profiler.TraceAnnotation(f"{method_name}/extract_constrained_samples"):
        samples = extract_constrained_samples(
            chain_particles,
            site_info,
            unravel_fn,
            model.spec,
            reparam=reparam,
            model=model,
            observations=observations,
            times=times,
        )

    diagnostics = {
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "eps_history": [float(jax.device_get(step_eps)) for step_eps in eps_history],
        "beta_schedule": [float(jax.device_get(beta)) for beta in beta_schedule],
        "n_levels": actual_levels,
        "n_outer": n_outer,
        "n_csmc_particles": N,
        "n_mh_steps": n_mh_steps,
        "n_leapfrog": n_leapfrog,
        "param_step_size": param_step_size,
        "n_mixing_rounds": n_mixing_rounds,
        "target_accept": target_accept,
        "adaptive_tempering": adaptive_tempering,
        "waste_free": waste_free,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    logger.info(
        "%s complete in %.1fs across %s tempering levels.",
        print_prefix,
        time.monotonic() - started_at,
        actual_levels,
    )

    return InferenceResult(
        _samples=samples,
        method=method_name,
        diagnostics=diagnostics,
    )


def fit_tempered_smc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    n_leapfrog: int = 5,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters with the canonical tempered SMC entrypoint."""
    backend = model.make_likelihood_backend()
    return run_tempered_smc(
        model,
        observations,
        times,
        n_outer=n_outer,
        n_csmc_particles=n_csmc_particles,
        n_mh_steps=n_mh_steps,
        param_step_size=param_step_size,
        n_warmup=n_warmup,
        target_accept=target_accept,
        seed=seed,
        adaptive_tempering=adaptive_tempering,
        target_ess_ratio=target_ess_ratio,
        waste_free=waste_free,
        n_leapfrog=n_leapfrog,
        method_name="tempered_smc",
        likelihood_backend=backend,
        extra_diagnostics={"likelihood_backend": backend},
        print_prefix="Tempered SMC",
        reparam=reparam,
    )
