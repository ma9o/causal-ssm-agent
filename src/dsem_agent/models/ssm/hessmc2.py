"""Hess-MC² inference: tempered SMC with gradient-based proposals.

Implements a tempered SMC sampler [Del Moral et al., 2006] for SSM parameter
estimation, with proposal kernels from the Hess-MC² paper:
- Random Walk (RW) proposals
- First-Order Langevin (MALA) proposals using gradient of log-posterior
- Second-Order (Hessian) proposals using curvature information

Tempering schedule gamma_0=0 < gamma_1 < ... < gamma_K=1 gradually transitions
from the prior p(theta) to the full posterior p(theta)*p(y|theta). At each step:
1. Re-weight particles by p(y|theta)^{delta_gamma}
2. Resample if ESS < N/2
3. Rejuvenate via MCMC moves targeting the current tempered posterior

The inner particle filter (or Kalman filter) provides a differentiable
log-likelihood, enabling gradient/Hessian computation via JAX autodiff.

Reference: Murphy et al., "Hess-MC²: Sequential Monte Carlo Squared using
Hessian Information and Second Order Proposals", 2025.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from numpyro import handlers

from dsem_agent.models.ssm.inference import InferenceResult, _eval_model

# ---------------------------------------------------------------------------
# Parameter packing: named dicts <-> flat vectors
# ---------------------------------------------------------------------------


class ParamPacker:
    """Pack/unpack named parameter dicts to/from flat JAX vectors."""

    def __init__(self, site_info: dict):
        self.names: list[str] = []
        self.slices: dict[str, tuple[int, int, tuple[int, ...]]] = {}
        offset = 0
        for name, info in site_info.items():
            shape = info["shape"]
            size = 1
            for s in shape:
                size *= s
            self.names.append(name)
            self.slices[name] = (offset, offset + size, shape)
            offset += size
        self.dim = offset

    def pack(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        parts = [params[n].flatten() for n in self.names]
        return jnp.concatenate(parts) if parts else jnp.array([])

    def unpack(self, vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
        result = {}
        for name in self.names:
            start, end, shape = self.slices[name]
            result[name] = vector[start:end].reshape(shape)
        return result


# ---------------------------------------------------------------------------
# Model tracing and differentiable evaluators
# ---------------------------------------------------------------------------


def _discover_sites(model, observations, times, subject_ids, rng_key):
    """Trace model once to discover sample sites (names, shapes, transforms)."""
    with handlers.seed(rng_seed=int(rng_key[0])):
        trace = handlers.trace(model.model).get_trace(observations, times, subject_ids)

    site_info = {}
    for name, site in trace.items():
        if (
            site["type"] == "sample"
            and not site.get("is_observed", False)
            and name != "log_likelihood"
        ):
            d = site["fn"]
            site_info[name] = {
                "shape": site["value"].shape,
                "distribution": d,
                "transform": dist.transforms.biject_to(d.support),
                "value": site["value"],
            }
    return site_info


def _build_eval_fns(model, observations, times, subject_ids, site_info, packer):
    """Build differentiable functions for log-likelihood and log-prior.

    Returns:
        log_lik_fn(z, pf_key) -> scalar log p(y|θ)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    def _constrain(z):
        unc = packer.unpack(z)
        return {name: transforms[name](unc[name]) for name in packer.names}, unc

    def log_lik_fn(z, pf_key):
        """Log-likelihood p(y|θ) via PF or Kalman."""
        con, _ = _constrain(z)
        model.pf_key = pf_key
        log_lik, _ = _eval_model(model.model, con, observations, times, subject_ids)
        return log_lik

    def log_prior_unc_fn(z):
        """Log-prior in unconstrained space: log p(T(z)) + log|J(z)|."""
        con, unc = _constrain(z)
        lp = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in packer.names)
        lj = sum(
            jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name]))
            for name in packer.names
        )
        return lp + lj

    return log_lik_fn, log_prior_unc_fn


# ---------------------------------------------------------------------------
# Outer SMC resampling
# ---------------------------------------------------------------------------


def _systematic_resampling_outer(key, log_weights, n):
    """Systematic resampling for the outer SMC sampler."""
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    cumsum = jnp.cumsum(weights)
    us = (random.uniform(key, ()) + jnp.arange(n)) / n
    idx = jnp.searchsorted(cumsum, us)
    return jnp.clip(idx, 0, n - 1)


# ---------------------------------------------------------------------------
# Proposals (leapfrog form from paper Sec III-A)
# ---------------------------------------------------------------------------


def _propose_rw(key, theta, step_size, _grad, _hess_diag):
    z = random.normal(key, theta.shape)
    return theta + step_size * z


def _propose_fo(key, theta, step_size, grad, _hess_diag):
    """MALA proposal: θ' = θ + (ε²/2)∇log π + ε·z."""
    z = random.normal(key, theta.shape)
    return theta + 0.5 * step_size**2 * grad + step_size * z


def _propose_so(key, theta, step_size, grad, hess_diag, fallback_ss):
    """Second-order proposal using diagonal Hessian as metric.

    M = -diag(∇²log π). If M > 0: θ' = θ + (ε²/2)M⁻¹∇ + ε·M^{-1/2}z.
    Else fallback to MALA with fallback_ss.
    """
    neg_hess_diag = -hess_diag
    is_psd = jnp.all(neg_hess_diag > 1e-8)

    def so_branch(key):
        inv_m = 1.0 / jnp.maximum(neg_hess_diag, 1e-8)
        inv_m_sqrt = jnp.sqrt(inv_m)
        z = random.normal(key, theta.shape)
        return theta + 0.5 * step_size**2 * inv_m * grad + step_size * inv_m_sqrt * z

    def fo_branch(key):
        z = random.normal(key, theta.shape)
        return theta + 0.5 * fallback_ss**2 * grad + fallback_ss * z

    return jax.lax.cond(is_psd, so_branch, fo_branch, key)


def _log_proposal_density(theta_new, theta_old, grad, hess_diag, step_size, proposal, fallback_ss):
    """Log q(θ'|θ) for MH accept/reject."""
    if proposal == "rw":
        diff = theta_new - theta_old
        return -0.5 * jnp.sum(diff**2) / step_size**2

    elif proposal == "mala":
        mean = theta_old + 0.5 * step_size**2 * grad
        diff = theta_new - mean
        return -0.5 * jnp.sum(diff**2) / step_size**2

    else:  # hessian
        neg_hess_diag = -hess_diag
        is_psd = jnp.all(neg_hess_diag > 1e-8)

        def so_density():
            inv_m = 1.0 / jnp.maximum(neg_hess_diag, 1e-8)
            mean = theta_old + 0.5 * step_size**2 * inv_m * grad
            var = step_size**2 * inv_m
            return -0.5 * jnp.sum((theta_new - mean) ** 2 / var + jnp.log(var))

        def fo_density():
            mean = theta_old + 0.5 * fallback_ss**2 * grad
            return -0.5 * jnp.sum((theta_new - mean) ** 2) / fallback_ss**2

        return jax.lax.cond(is_psd, so_density, fo_density)


# ---------------------------------------------------------------------------
# Diagonal Hessian via forward-over-reverse
# ---------------------------------------------------------------------------


def _diag_hessian(f, x, *args):
    """Compute diagonal of Hessian of f at x via forward-over-reverse."""
    grad_fn = jax.grad(f)

    def hvp_diag_element(basis_vec):
        _, jvp_val = jax.jvp(
            grad_fn, (x, *args), (basis_vec, *[jnp.zeros_like(a) for a in args])
        )
        return jnp.dot(basis_vec, jvp_val)

    eye = jnp.eye(x.shape[0])
    return jax.vmap(hvp_diag_element)(eye)


# ---------------------------------------------------------------------------
# Hess-MC² main sampler
# ---------------------------------------------------------------------------


def fit_hessmc2(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    n_smc_particles: int = 64,
    n_iterations: int = 20,
    n_mcmc_steps: int = 5,
    proposal: Literal["rw", "mala", "hessian"] = "mala",
    step_size: float = 0.1,
    fallback_step_size: float = 0.01,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters using Hess-MC² (tempered SMC with Langevin proposals).

    Uses likelihood tempering gamma: 0->1 over K steps to bridge from prior to
    posterior. At each temperature step, particles are rejuvenated via MCMC
    with gradient/Hessian-informed proposals.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices for hierarchical models
        n_smc_particles: N — number of parameter particles
        n_iterations: K — number of tempering steps
        n_mcmc_steps: number of MCMC rejuvenation moves per step
        proposal: "rw", "mala", or "hessian"
        step_size: ε — proposal step size
        fallback_step_size: step size when Hessian is not PSD (SO only)
        seed: random seed

    Returns:
        InferenceResult with posterior samples
    """
    rng_key = random.PRNGKey(seed)
    N = n_smc_particles
    K = n_iterations

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    packer = ParamPacker(site_info)
    D = packer.dim

    # 2. Build differentiable functions
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, packer
    )
    pf_key = model.pf_key

    def tempered_log_post(z, gamma):
        """Log of tempered posterior: log p(z) + gamma * log p(y|theta(z))."""
        ll = log_lik_fn(z, pf_key)
        lp = log_prior_unc_fn(z)
        return lp + gamma * ll

    # 3. Initialize N particles from prior (gamma=0, target is just the prior)
    particles = jnp.zeros((N, D))
    log_liks = jnp.zeros(N)

    print(f"Hess-MC²: N={N}, K={K}, D={D}, proposal={proposal}, ε={step_size}")
    print(f"  Initializing {N} particles from prior...")

    for i in range(N):
        rng_key, init_key = random.split(rng_key)
        with handlers.seed(rng_seed=int(init_key[0])):
            trace = handlers.trace(model.model).get_trace(
                observations, times, subject_ids
            )
        init_unc = {}
        for name, info in site_info.items():
            init_unc[name] = info["transform"].inv(trace[name]["value"])
        particles = particles.at[i].set(packer.pack(init_unc))
        # Compute log-likelihood for each particle
        ll = log_lik_fn(particles[i], pf_key)
        ll = jnp.where(jnp.isfinite(ll), ll, -1e30)
        log_liks = log_liks.at[i].set(ll)

    # Tempering schedule: linear gamma from 0 to 1
    gammas = jnp.linspace(0.0, 1.0, K + 1)

    # Track diagnostics
    ess_history = []
    resample_points = []
    accept_rates = []

    # 4. Tempered SMC loop
    for k in range(K):
        gamma_prev = float(gammas[k])
        gamma_curr = float(gammas[k + 1])
        delta_gamma = gamma_curr - gamma_prev

        # --- Step A: Re-weight by incremental likelihood ---
        # delta log w = (gamma_k - gamma_{k-1}) * log p(y|theta)
        log_weights = delta_gamma * log_liks

        # --- Step B: ESS check and resampling ---
        log_wn = log_weights - jax.nn.logsumexp(log_weights)
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        did_resample = False
        if ess < N / 2:
            resample_points.append(k)
            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resampling_outer(resample_key, log_weights, N)
            particles = particles[idx]
            log_liks = log_liks[idx]
            did_resample = True

        # --- Step C: MCMC rejuvenation targeting tempered posterior at gamma_k ---
        n_accepted = 0
        n_proposed = 0

        # Pre-compute gradients for current particles (needed for MALA/Hessian)
        grads = jnp.zeros((N, D))
        hess_diags = jnp.zeros((N, D))

        if proposal in ("mala", "hessian"):
            for i in range(N):
                g = jax.grad(tempered_log_post)(particles[i], gamma_curr)
                grads = grads.at[i].set(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))

        if proposal == "hessian":
            for i in range(N):
                hd = _diag_hessian(tempered_log_post, particles[i], gamma_curr)
                hess_diags = hess_diags.at[i].set(
                    jnp.nan_to_num(hd, nan=0.0, posinf=0.0, neginf=0.0)
                )

        for _m in range(n_mcmc_steps):
            for i in range(N):
                rng_key, prop_key, accept_key = random.split(rng_key, 3)

                # Propose
                if proposal == "rw":
                    theta_new = _propose_rw(
                        prop_key, particles[i], step_size, grads[i], hess_diags[i]
                    )
                elif proposal == "mala":
                    theta_new = _propose_fo(
                        prop_key, particles[i], step_size, grads[i], hess_diags[i]
                    )
                else:
                    theta_new = _propose_so(
                        prop_key, particles[i], step_size, grads[i],
                        hess_diags[i], fallback_step_size,
                    )

                # Evaluate proposed particle
                ll_new = log_lik_fn(theta_new, pf_key)
                ll_new = jnp.where(jnp.isfinite(ll_new), ll_new, -1e30)
                lp_new = log_prior_unc_fn(theta_new)
                lp_new = jnp.where(jnp.isfinite(lp_new), lp_new, -1e30)
                log_post_new = lp_new + gamma_curr * ll_new

                lp_old = log_prior_unc_fn(particles[i])
                log_post_old = lp_old + gamma_curr * log_liks[i]

                # MH acceptance ratio (includes proposal asymmetry for MALA/Hessian)
                log_alpha = log_post_new - log_post_old

                if proposal != "rw":
                    # Compute gradient at proposed point for reverse proposal
                    g_new = jax.grad(tempered_log_post)(theta_new, gamma_curr)
                    g_new = jnp.nan_to_num(g_new, nan=0.0, posinf=0.0, neginf=0.0)

                    hd_new = hess_diags[i]  # Approximate: reuse current Hessian
                    if proposal == "hessian":
                        hd_new = _diag_hessian(tempered_log_post, theta_new, gamma_curr)
                        hd_new = jnp.nan_to_num(hd_new, nan=0.0, posinf=0.0, neginf=0.0)

                    # Forward: q(θ'|θ), Reverse: q(θ|θ')
                    log_q_fwd = _log_proposal_density(
                        theta_new, particles[i], grads[i], hess_diags[i],
                        step_size, proposal, fallback_step_size,
                    )
                    log_q_rev = _log_proposal_density(
                        particles[i], theta_new, g_new, hd_new,
                        step_size, proposal, fallback_step_size,
                    )
                    log_alpha = log_alpha + log_q_rev - log_q_fwd

                # Accept/reject
                u = random.uniform(accept_key)
                accept = jnp.log(u) < log_alpha
                accept = accept & jnp.isfinite(log_post_new)

                if bool(accept):
                    particles = particles.at[i].set(theta_new)
                    log_liks = log_liks.at[i].set(ll_new)
                    if proposal in ("mala", "hessian"):
                        grads = grads.at[i].set(g_new)
                    if proposal == "hessian":
                        hess_diags = hess_diags.at[i].set(hd_new)
                    n_accepted += 1
                n_proposed += 1

        acc_rate = n_accepted / max(n_proposed, 1)
        accept_rates.append(acc_rate)
        resamp_tag = " [resampled]" if did_resample else ""
        print(
            f"  step {k + 1}/{K}  γ={gamma_curr:.3f}  "
            f"ESS={ess:.1f}/{N}  accept={acc_rate:.2f}{resamp_tag}"
        )

    # 5. Extract final samples in constrained space
    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in packer.names:
        vals = []
        for i in range(N):
            unc = packer.unpack(particles[i])
            vals.append(transforms[name](unc[name]))
        samples[name] = jnp.stack(vals)

    # Extract deterministic sites (drift, diffusion matrices, etc.)
    det_samples = _extract_deterministic_sites(
        model, observations, times, subject_ids, site_info, packer, particles,
    )
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="hessmc2",
        diagnostics={
            "ess_history": ess_history,
            "resample_points": resample_points,
            "accept_rates": accept_rates,
            "gammas": [float(g) for g in gammas],
            "n_smc_particles": N,
            "n_iterations": K,
            "proposal": proposal,
            "step_size": step_size,
        },
    )


def _extract_deterministic_sites(
    model, observations, times, subject_ids, site_info, packer, particles,
):
    """Run model with each particle to extract deterministic sites."""
    transforms = {name: info["transform"] for name, info in site_info.items()}
    N = particles.shape[0]
    det_samples: dict[str, list] = {}

    for i in range(N):
        unc = packer.unpack(particles[i])
        con = {name: transforms[name](unc[name]) for name in packer.names}

        with handlers.seed(rng_seed=0), handlers.substitute(data=con):
            trace = handlers.trace(model.model).get_trace(
                observations, times, subject_ids
            )

        for name, site in trace.items():
            if site["type"] == "deterministic":
                if name not in det_samples:
                    det_samples[name] = []
                det_samples[name].append(site["value"])

    return {name: jnp.stack(vals) for name, vals in det_samples.items()}
