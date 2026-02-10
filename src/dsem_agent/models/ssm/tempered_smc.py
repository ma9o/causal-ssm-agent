"""Tempered SMC with preconditioned MALA mutations (fit_tempered_smc).

Bridges the prior-posterior gap via a tempering ladder beta_0=0 -> beta_K=1,
with MH-corrected MALA (1-step leapfrog HMC) mutations at each level.
This handles high-dimensional parameter spaces (D>>3) where importance-
weighted proposals (Hess-MC2) suffer ESS collapse.

Key features:
  - **Precision preconditioning**: Mass matrix M set to the weighted particle
    precision (inverse covariance), matching Stan/NUTS convention. This makes
    MALA proposals isotropic in posterior-standardized space (noise ~ eps*N(0,I)).
  - **Adaptive step size**: Robbins-Monro update targeting ~44% acceptance.
  - **Pilot adaptation**: Tunes eps at beta=0 (prior) before tempering starts.
  - **Guarded mass matrix**: Only updates from weighted covariance when ESS
    is healthy (> N/4), preventing degenerate mass matrices.
  - **Adaptive mutation rounds**: Runs extra MALA rounds at each tempering
    level when acceptance is low, ensuring particles properly diversify.

Algorithm:
  1. Initialize N particles from the prior in unconstrained space.
  2. Pilot: tune eps via MALA at beta=0 (targeting the prior).
  3. For each tempering level beta_k = k / n_outer:
     a. Incremental reweight: logw += (beta_k - beta_{k-1}) * log_lik(particles)
     b. Update mass matrix from weighted covariance (only if ESS > N/4)
     c. Systematic resample if ESS < N/2
     d. Mutate each particle with MALA (1-5 rounds of n_mh_steps)
     e. Adapt step size toward target acceptance rate
     f. Store one sample from the particle cloud
  4. Discard warmup, transform to constrained space, return InferenceResult.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree

from dsem_agent.models.ssm.hessmc2 import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)
from dsem_agent.models.ssm.inference import InferenceResult

# ---------------------------------------------------------------------------
# Preconditioned MALA step (full mass matrix)
# ---------------------------------------------------------------------------


def _mala_step(rng_key, z, log_target_val_and_grad, step_size, chol_mass):
    """Preconditioned MALA: 1-step leapfrog with full mass matrix M = L L^T.

    Args:
        rng_key: PRNG key
        z: current position (D,)
        log_target_val_and_grad: fn(z) -> (scalar, (D,))
        step_size: leapfrog epsilon (scalar JAX value)
        chol_mass: Cholesky factor of mass matrix (D, D), lower triangular

    Returns:
        z_new: accepted position (D,)
        accepted: bool scalar
        log_target_new: log target at accepted position
    """
    noise_key, accept_key = random.split(rng_key)
    D = z.shape[0]

    # Current value and gradient
    log_pi, grad = log_target_val_and_grad(z)
    grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # Sample momentum: p ~ N(0, M) where M = L L^T
    u = random.normal(noise_key, (D,))
    p = chol_mass @ u

    # Leapfrog: half-step momentum, full-step position, half-step momentum
    p_half = p + 0.5 * step_size * grad
    z_prop = z + step_size * jla.cho_solve((chol_mass, True), p_half)

    log_pi_prop, grad_prop = log_target_val_and_grad(z_prop)
    grad_prop = jnp.nan_to_num(grad_prop, nan=0.0, posinf=0.0, neginf=0.0)
    p_prop = p_half + 0.5 * step_size * grad_prop

    # Kinetic energy: 0.5 * p^T M^{-1} p = 0.5 * ||L^{-1} p||^2
    Linv_p = jla.solve_triangular(chol_mass, p, lower=True)
    Linv_p_prop = jla.solve_triangular(chol_mass, p_prop, lower=True)
    kinetic_old = 0.5 * jnp.dot(Linv_p, Linv_p)
    kinetic_new = 0.5 * jnp.dot(Linv_p_prop, Linv_p_prop)

    log_alpha = (log_pi_prop - kinetic_new) - (log_pi - kinetic_old)
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)

    accept_u = random.uniform(accept_key)
    accepted = jnp.log(accept_u) < log_alpha

    z_new = jnp.where(accepted, z_prop, z)
    log_target_new = jnp.where(accepted, log_pi_prop, log_pi)

    return z_new, accepted, log_target_new


# ---------------------------------------------------------------------------
# Empirical covariance helper
# ---------------------------------------------------------------------------


def _compute_weighted_chol_mass(particles, logw, D):
    """Compute Cholesky of precision (= inverse covariance) for HMC mass matrix.

    With M = precision, the leapfrog dynamics are isotropic in posterior-
    standardized space: MALA noise becomes eps * N(0, I) rather than
    eps * N(0, cov^{-1}). This matches the Stan/NUTS convention (where
    "inverse mass matrix" = covariance, so M = precision).

    For MALA (1-step leapfrog), isotropic proposals are essential since
    there's no trajectory adaptation to compensate for frequency mismatch.
    """
    wn = jnp.exp(logw - jax.nn.logsumexp(logw))
    mean = jnp.sum(wn[:, None] * particles, axis=0)
    centered = particles - mean
    cov = (centered * wn[:, None]).T @ centered
    cov_reg = cov + 1e-3 * jnp.eye(D)
    # Compute precision = cov^{-1} via Cholesky solve
    L_cov = jla.cholesky(cov_reg, lower=True)
    prec = jla.cho_solve((L_cov, True), jnp.eye(D))
    return jla.cholesky(prec, lower=True)


# ---------------------------------------------------------------------------
# Tempered SMC + MALA main sampler
# ---------------------------------------------------------------------------


def fit_tempered_smc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    langevin_step_size: float = 0.5,  # noqa: ARG001
    param_step_size: float = 0.1,
    n_warmup: int = 50,
    target_accept: float = 0.44,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via tempered SMC with preconditioned MALA mutations.

    Uses a linear tempering schedule beta_k = k/n_outer to gradually bridge
    from the prior to the posterior. At each level, preconditioned MALA
    mutations (with adaptive rounds) diversify the particle cloud.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices for hierarchical models
        n_outer: number of tempering levels (beta goes from 1/n_outer to 1.0)
        n_csmc_particles: N -- number of parameter particles
        n_mh_steps: number of MALA mutation steps per round
        langevin_step_size: unused (kept for API compatibility)
        param_step_size: initial leapfrog step size (epsilon), adapted online
        n_warmup: number of initial tempering levels to discard as warmup
        target_accept: target MH acceptance rate for step size adaptation
        seed: random seed

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    rng_key = random.PRNGKey(seed)
    N = n_csmc_particles

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    # 2. Build differentiable evaluators
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
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

    # JIT-compiled MALA kernel
    def _mala_scan_body(carry, rng_key, beta, eps, chol_mass):
        z, n_accept = carry

        def tempered_vg(z_):
            return _tempered_val_and_grad(z_, beta)

        z_new, accepted, _ = _mala_step(rng_key, z, tempered_vg, eps, chol_mass)
        return (z_new, n_accept + accepted.astype(jnp.int32)), None

    def _mutate_particle(rng_key, z, beta, eps, chol_mass):
        """Run n_mh_steps of preconditioned MALA on a single particle."""
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            return _mala_scan_body(carry, key, beta, eps, chol_mass)

        (z_final, n_accept), _ = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return z_final, n_accept

    # Vmap over particles, JIT the whole batch mutation
    def _mutate_batch(rng_key, particles, beta, eps, chol_mass):
        keys = random.split(rng_key, N)
        return jax.vmap(lambda k, z: _mutate_particle(k, z, beta, eps, chol_mass))(keys, particles)

    _mutate_batch_jit = jax.jit(_mutate_batch)

    # 3. Initialize N particles from prior
    eps = param_step_size
    print(
        f"Tempered SMC: N={N}, K={n_outer}, D={D}, n_mh={n_mh_steps}, "
        f"eps={eps}, target_accept={target_accept}"
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
    chol_mass = _compute_weighted_chol_mass(particles, jnp.zeros(N), D)

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
    chol_mass = _compute_weighted_chol_mass(particles, jnp.zeros(N), D)

    # 4. Tempering schedule: linear beta_k = k / n_outer
    betas = [float(k + 1) / n_outer for k in range(n_outer)]
    logw = jnp.zeros(N)  # uniform weights at beta=0

    # Diagnostics
    accept_rates = []
    ess_history = []
    eps_history = []
    chain_samples = []

    beta_prev = 0.0
    max_mutation_rounds = 5  # max extra MALA rounds per tempering level

    # 5. Tempering loop
    for k in range(n_outer):
        beta_k = betas[k]

        # a. Incremental reweight: logw += (beta_k - beta_{k-1}) * log_lik
        logw = logw + (beta_k - beta_prev) * log_liks

        # Normalize and compute ESS
        lse = jax.nn.logsumexp(logw)
        log_wn = logw - lse
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        # b. Update mass matrix only when ESS is healthy
        if ess > N / 4:
            chol_mass = _compute_weighted_chol_mass(particles, logw, D)

        # c. Resample if ESS < N/2
        did_resample = False
        if ess < N / 2:
            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resample(resample_key, wn, N)
            particles = particles[idx]
            log_liks = log_liks[idx]
            logw = jnp.full(N, -jnp.log(float(N)))
            did_resample = True

        # d. Adaptive mutation: run MALA rounds until acceptance is reasonable
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
        accept_rates.append(avg_accept)
        eps_history.append(eps)

        # Recompute log-likelihoods for next incremental reweight
        log_liks, _ = batch_lik_val_and_grad(particles)

        # f. Draw one sample (rotate through particles for coverage)
        chain_samples.append(particles[k % N])

        beta_prev = beta_k

        n_rounds = mutation_round + 1
        resamp_tag = " [resampled]" if did_resample else ""
        print(
            f"  step {k + 1}/{n_outer}  beta={beta_k:.3f}  ESS={ess:.1f}/{N}"
            f"  accept={avg_accept:.2f}  eps={eps:.4f}  rounds={n_rounds}{resamp_tag}"
        )

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

    return InferenceResult(
        _samples=samples,
        method="tempered_smc",
        diagnostics={
            "accept_rates": accept_rates,
            "ess_history": ess_history,
            "eps_history": eps_history,
            "n_outer": n_outer,
            "n_csmc_particles": N,
            "n_mh_steps": n_mh_steps,
            "param_step_size": param_step_size,
            "n_warmup": n_warmup,
            "target_accept": target_accept,
        },
    )
