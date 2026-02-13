"""Shared tempered SMC loop for parameter inference.

Provides `run_tempered_smc()`, the single implementation of the tempered SMC
algorithm used by tempered_smc, laplace_em, structured_vi, and dpf.

Bridges the prior-posterior gap via a tempering ladder beta_0=0 -> beta_K=1,
with MH-corrected HMC mutations at each level. Supports adaptive tempering,
waste-free recycling, multi-step leapfrog, and precision preconditioning.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.models.ssm.inference import InferenceResult
from causal_ssm_agent.models.ssm.mcmc_utils import (
    compute_weighted_chol_mass,
    find_next_beta,
    hmc_step,
)
from causal_ssm_agent.models.ssm.utils import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)


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
    method_name: str = "tempered_smc",
    likelihood_backend=None,
    extra_diagnostics: dict[str, Any] | None = None,
    print_prefix: str = "Tempered SMC",
) -> InferenceResult:
    """Run tempered SMC with preconditioned HMC/MALA mutations.

    This is the shared implementation used by tempered_smc, laplace_em,
    structured_vi, and dpf. Each method calls this with different
    method_name and extra_diagnostics.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        n_outer: max tempering levels (safety bound for adaptive, exact for linear)
        n_csmc_particles: N -- number of parameter particles
        n_mh_steps: number of HMC mutation steps per round
        param_step_size: initial leapfrog step size (epsilon), adapted online
        n_warmup: tempering levels to discard as warmup (default: half of actual)
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
        target_accept = 0.65 if n_leapfrog > 1 else 0.44

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

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, trace_key, likelihood_backend)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    # 2. Build differentiable evaluators
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model,
        observations,
        times,
        site_info,
        unravel_fn,
        likelihood_backend=likelihood_backend,
    )

    # Safe value-and-grad for log-likelihood
    def _safe_lik_val_and_grad(z):
        val, grad = jax.value_and_grad(log_lik_fn)(z)
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    batch_lik_val_and_grad = jax.jit(jax.vmap(_safe_lik_val_and_grad))

    # Tempered target: log_prior + beta * log_lik
    def _tempered_val_and_grad(z, beta):
        lik_val, lik_grad = jax.value_and_grad(log_lik_fn)(z)
        prior_val, prior_grad = jax.value_and_grad(log_prior_unc_fn)(z)
        val = prior_val + beta * lik_val
        grad = prior_grad + beta * lik_grad
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    # HMC mutation kernel (single particle)
    def _hmc_scan_body(carry, rng_key, beta, eps, chol_mass):
        z, n_accept = carry

        def tempered_vg(z_):
            return _tempered_val_and_grad(z_, beta)

        z_new, accepted, _ = hmc_step(rng_key, z, tempered_vg, eps, chol_mass, n_leapfrog)
        return (z_new, n_accept + accepted.astype(jnp.int32)), None

    def _mutate_particle(rng_key, z, beta, eps, chol_mass):
        """Run n_mh_steps of preconditioned HMC on a single particle."""
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            return _hmc_scan_body(carry, key, beta, eps, chol_mass)

        (z_final, n_accept), _ = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return z_final, n_accept

    # Waste-free mutation: collect ALL intermediate states
    def _mutate_particle_wastefree(rng_key, z, beta, eps, chol_mass):
        """Run n_mh_steps of HMC, keeping all intermediate positions."""
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), z_new

        (_, n_acc), all_z = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return all_z, n_acc  # all_z: (n_mh_steps, D)

    # Vmap over particles, JIT the whole batch mutation
    def _mutate_batch(rng_key, particles, beta, eps, chol_mass):
        keys = random.split(rng_key, particles.shape[0])
        return jax.vmap(lambda k, z: _mutate_particle(k, z, beta, eps, chol_mass))(keys, particles)

    def _mutate_batch_wastefree(rng_key, particles_M, beta, eps, chol_mass):
        """Waste-free batch: mutate M particles, return (M, n_mh_steps, D)."""
        M = particles_M.shape[0]
        keys = random.split(rng_key, M)
        return jax.vmap(lambda k, z: _mutate_particle_wastefree(k, z, beta, eps, chol_mass))(
            keys, particles_M
        )

    _mutate_batch_jit = jax.jit(_mutate_batch)
    _mutate_batch_wastefree_jit = jax.jit(_mutate_batch_wastefree)

    # 3. Initialize N particles from prior
    eps = param_step_size
    mode_tag = "adaptive" if adaptive_tempering else "linear"
    wf_tag = "+waste-free" if waste_free else ""
    hmc_tag = f"+HMC(L={n_leapfrog})" if n_leapfrog > 1 else ""
    print(
        f"{print_prefix} [{mode_tag}{wf_tag}{hmc_tag}]: N={N}, K={n_outer}, D={D}, "
        f"n_mh={n_mh_steps}, eps={eps}, target_accept={target_accept}"
    )
    print(f"  Initializing {N} particles from prior...")

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

    # ===================================================================
    # Pilot: tune eps at prior (beta=0) before tempering
    # ===================================================================
    print("  Pilot: adapting step size at prior...")
    for pilot_step in range(30):
        rng_key, mutate_key = random.split(rng_key)
        particles_new, n_accepts = _mutate_batch_jit(mutate_key, particles, 0.0, eps, chol_mass)
        avg_accept = float(jnp.mean(n_accepts) / n_mh_steps)
        particles = particles_new

        # Aggressive adaptation during pilot
        log_eps = jnp.log(jnp.array(eps))
        log_eps = log_eps + 0.5 * (avg_accept - target_accept)
        eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

        if pilot_step >= 5 and abs(avg_accept - target_accept) < 0.1:
            print(
                f"    pilot converged at step {pilot_step + 1}: "
                f"accept={avg_accept:.2f} eps={eps:.4f}"
            )
            break
    else:
        print(f"    pilot done: accept={avg_accept:.2f} eps={eps:.4f}")

    # Recompute after pilot diversification
    log_liks, _ = batch_lik_val_and_grad(particles)
    chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)

    logw = jnp.zeros(N)  # uniform weights at beta=0

    # Diagnostics
    accept_rates = []
    ess_history = []
    eps_history = []
    beta_schedule = []
    chain_samples = []

    beta_prev = 0.0
    max_mutation_rounds = 5  # max extra rounds per tempering level (standard mode only)
    level = 0

    # Waste-free parameters
    M = N // n_mh_steps if waste_free else N  # resample count for waste-free

    # 5. Tempering loop
    while beta_prev < 1.0 and level < n_outer:
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
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        # c. Update mass matrix only when ESS is healthy
        if ess > N / 4:
            chol_mass = compute_weighted_chol_mass(particles, logw, D)

        # d. Resample and mutate
        if waste_free:
            # Waste-free: resample M particles, mutate each n_mh_steps times
            rng_key, resample_key, mutate_key = random.split(rng_key, 3)
            idx = _systematic_resample(resample_key, wn, M)
            resampled = particles[idx]

            all_trajs, n_accs = _mutate_batch_wastefree_jit(
                mutate_key, resampled, beta_k, eps, chol_mass
            )
            # all_trajs: (M, n_mh_steps, D) -> reshape to (N, D)
            particles = all_trajs.reshape(N, D)
            logw = jnp.full(N, -jnp.log(float(N)))

            avg_accept = float(jnp.mean(n_accs) / n_mh_steps)
            n_rounds = 1

            # Adapt step size
            log_eps = jnp.log(jnp.array(eps))
            log_eps = log_eps + 0.1 * (avg_accept - target_accept)
            eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))
        else:
            # Standard: resample if ESS < N/2, then adaptive mutation rounds
            did_resample = False
            if ess < N / 2:
                rng_key, resample_key = random.split(rng_key)
                idx = _systematic_resample(resample_key, wn, N)
                particles = particles[idx]
                log_liks = log_liks[idx]
                logw = jnp.full(N, -jnp.log(float(N)))
                did_resample = True

            total_accepts = 0
            total_proposals = 0
            for mutation_round in range(max_mutation_rounds):
                rng_key, mutate_key = random.split(rng_key)
                particles_new, n_accepts = _mutate_batch_jit(
                    mutate_key, particles, beta_k, eps, chol_mass
                )
                round_accepts = float(jnp.sum(n_accepts))
                total_accepts += round_accepts
                total_proposals += N * n_mh_steps
                particles = particles_new

                # Adapt step size after each round
                round_accept_rate = round_accepts / (N * n_mh_steps)
                log_eps = jnp.log(jnp.array(eps))
                log_eps = log_eps + 0.1 * (round_accept_rate - target_accept)
                eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

                # Stop early if acceptance is reasonable
                if mutation_round > 0 and round_accept_rate > 0.2:
                    break

            avg_accept = total_accepts / max(total_proposals, 1)
            n_rounds = mutation_round + 1

        accept_rates.append(avg_accept)
        eps_history.append(eps)

        # Recompute log-likelihoods for next incremental reweight
        log_liks, _ = batch_lik_val_and_grad(particles)

        # Store one sample (rotate through particles for coverage)
        chain_samples.append(particles[level % N])

        resamp_tag = ""
        if not waste_free and did_resample:
            resamp_tag = " [resampled]"
        elif waste_free:
            resamp_tag = " [waste-free]"

        print(
            f"  step {level + 1}  beta={beta_k:.3f}  ESS={ess:.1f}/{N}"
            f"  accept={avg_accept:.2f}  eps={eps:.4f}  rounds={n_rounds}{resamp_tag}"
        )

        beta_prev = beta_k
        level += 1

    # Determine warmup from actual levels used
    actual_levels = level
    if n_warmup is None:
        n_warmup = actual_levels // 2
    # Clamp warmup to leave at least 1 sample
    n_warmup = min(n_warmup, max(actual_levels - 1, 0))

    # 6. Post-process: discard warmup, transform to constrained space
    chain_particles = jnp.stack(chain_samples[n_warmup:], axis=0)  # (n_keep, D)

    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(chain_particles)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    diagnostics = {
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "eps_history": eps_history,
        "beta_schedule": beta_schedule,
        "n_levels": actual_levels,
        "n_outer": n_outer,
        "n_csmc_particles": N,
        "n_mh_steps": n_mh_steps,
        "n_leapfrog": n_leapfrog,
        "param_step_size": param_step_size,
        "n_warmup": n_warmup,
        "target_accept": target_accept,
        "adaptive_tempering": adaptive_tempering,
        "waste_free": waste_free,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)

    return InferenceResult(
        _samples=samples,
        method=method_name,
        diagnostics=diagnostics,
    )
