"""Hess-MC² inference: SMC with gradient-based change-of-variables L-kernels.

Implements the SMC sampler from the Hess-MC² paper with momentum-based
proposals and change-of-variables (CoV) L-kernels:
- Random Walk (RW) proposals
- First-Order Langevin (MALA) proposals using gradient of log-posterior
- Second-Order (Hessian) proposals using full curvature information

All proposals are accepted — quality is controlled through importance weight
correction via the CoV L-kernel, not MH accept/reject. Gradients and Hessians
target the log-posterior (paper Eq 9, 11).

Reference: Murphy et al., "Hess-MC²: Sequential Monte Carlo Squared using
Hessian Information and Second Order Proposals", 2025.

Design Notes
------------
**No tempering (by design):** Unlike standard SMC samplers that use a tempering
ladder β_0=0 → β_K=1, Hess-MC² targets the full posterior π(θ) from iteration
1. The paper (Section III, Eq 24) defines initial weights as v_1^i = π(θ_1^i) /
q_1(θ_1^i) — the full posterior ratio, with no β scaling. The thesis is that
gradient- and Hessian-informed proposals provide sufficient exploration without
tempering. The authors' reference implementation (github.com/j-j-murphy/
SMC-Squared-Langevin, SMCsq_BASE.py line 137) confirms: ``logw = p_logpdf_x -
p_log_q0_x`` with no tempering parameter anywhere. Do NOT add tempering.

**Full Hessian:** The reference implementation uses the full DxD Hessian matrix:
``np.linalg.pinv(neg_hess)`` for inversion and full MVN sampling (proposals.py
lines 65-68). We match this: SO proposals use the full DxD negative Hessian as
the mass matrix, with Cholesky decomposition for sampling and solving. For
typical DSEM dimensions (D=5-30) the O(D³) cost is negligible compared to the
PF likelihood evaluation. Do NOT downgrade to diagonal Hessian.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpyro.distributions as dist
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from dsem_agent.models.ssm.inference import InferenceResult, _eval_model

if TYPE_CHECKING:
    from dsem_agent.models.ssm.model import SSMSpec

# ---------------------------------------------------------------------------
# Pure-JAX deterministic site assembly (replaces sequential numpyro tracing)
# ---------------------------------------------------------------------------


def _assemble_deterministics(
    samples: dict[str, jnp.ndarray], spec: SSMSpec
) -> dict[str, jnp.ndarray]:
    """Assemble deterministic sites from constrained samples, bypassing numpyro.

    Each deterministic site is a matrix assembled from the raw sample sites
    (e.g. drift_diag_pop, drift_offdiag_pop → drift matrix). This mirrors the
    assembly logic in SSMModel._sample_* but operates directly on the (N, ...)
    sample arrays, avoiding N sequential numpyro trace calls.

    """
    N = next(iter(samples.values())).shape[0]
    n_l, n_m = spec.n_latent, spec.n_manifest
    det = {}

    # -- drift: diag(-|drift_diag_pop|) + offdiag entries --
    if "drift_diag_pop" in samples:

        def _assemble_drift(drift_diag_pop, drift_offdiag_pop):
            drift_diag = -jnp.abs(drift_diag_pop)
            drift = jnp.diag(drift_diag)
            offdiag_idx = 0
            for i in range(n_l):
                for j in range(n_l):
                    if i != j:
                        drift = drift.at[i, j].set(drift_offdiag_pop[offdiag_idx])
                        offdiag_idx += 1
            return drift

        offdiag = samples.get("drift_offdiag_pop", jnp.zeros((N, 0)))
        det["drift"] = jax.vmap(_assemble_drift)(samples["drift_diag_pop"], offdiag)

    # -- diffusion: diag or full lower-triangular Cholesky --
    if "diffusion_diag_pop" in samples:

        def _assemble_diffusion_diag(diff_diag):
            return jnp.diag(diff_diag)

        def _assemble_diffusion_full(diff_diag, diff_lower):
            diffusion = jnp.diag(diff_diag)
            lower_idx = 0
            for i in range(n_l):
                for j in range(i):
                    diffusion = diffusion.at[i, j].set(diff_lower[lower_idx])
                    lower_idx += 1
            return diffusion

        if "diffusion_lower" in samples:
            det["diffusion"] = jax.vmap(_assemble_diffusion_full)(
                samples["diffusion_diag_pop"], samples["diffusion_lower"]
            )
        else:
            det["diffusion"] = jax.vmap(_assemble_diffusion_diag)(samples["diffusion_diag_pop"])

    # -- cint: passthrough --
    if "cint_pop" in samples:
        det["cint"] = samples["cint_pop"]

    # -- lambda: eye(n_m, n_l) + free loadings in rows n_l: --
    if not isinstance(spec.lambda_mat, jnp.ndarray):
        if "lambda_free" in samples:

            def _assemble_lambda(free_loadings):
                lam = jnp.eye(n_m, n_l)
                idx = 0
                for i in range(n_l, n_m):
                    for j in range(n_l):
                        lam = lam.at[i, j].set(free_loadings[idx])
                        idx += 1
                return lam

            det["lambda"] = jax.vmap(_assemble_lambda)(samples["lambda_free"])
        else:
            # n_m == n_l: lambda is just identity, no free params
            det["lambda"] = jnp.broadcast_to(jnp.eye(n_m, n_l), (N, n_m, n_l))

    # -- manifest_cov: diag(d) @ diag(d).T = diag(d²) --
    if "manifest_var_diag" in samples:
        det["manifest_cov"] = jax.vmap(lambda d: jnp.diag(d**2))(samples["manifest_var_diag"])
    elif isinstance(spec.manifest_var, jnp.ndarray):
        fixed_cov = spec.manifest_var @ spec.manifest_var.T
        det["manifest_cov"] = jnp.broadcast_to(fixed_cov, (N, n_m, n_m))

    # -- t0_means: passthrough or broadcast fixed --
    if "t0_means_pop" in samples:
        det["t0_means"] = samples["t0_means_pop"]
    elif isinstance(spec.t0_means, jnp.ndarray):
        det["t0_means"] = jnp.broadcast_to(spec.t0_means, (N, n_l))

    # -- t0_cov: diag(d²) or broadcast fixed --
    if "t0_var_diag" in samples:
        det["t0_cov"] = jax.vmap(lambda d: jnp.diag(d**2))(samples["t0_var_diag"])
    elif isinstance(spec.t0_var, jnp.ndarray):
        fixed_cov = spec.t0_var @ spec.t0_var.T
        det["t0_cov"] = jnp.broadcast_to(fixed_cov, (N, n_l, n_l))

    return det


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


def _build_eval_fns(model, observations, times, subject_ids, site_info, unravel_fn):
    """Build differentiable functions for log-likelihood and log-prior.

    Returns:
        log_lik_fn(z) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def _log_lik_fn(z):
        """Log-likelihood p(y|theta) via PF or Kalman."""
        con, _ = _constrain(z)
        log_lik, _ = _eval_model(model.model, con, observations, times, subject_ids)
        return log_lik

    # Checkpoint: recompute PF intermediates during backward pass instead of
    # storing them. Trades ~2x compute for O(1) memory in time-series length.
    log_lik_fn = jax.checkpoint(_log_lik_fn)

    def log_prior_unc_fn(z):
        """Log-prior in unconstrained space: log p(T(z)) + log|J(z)|."""
        con, unc = _constrain(z)
        lp = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in unc)
        lj = sum(
            jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name])) for name in unc
        )
        return lp + lj

    return log_lik_fn, log_prior_unc_fn


# ---------------------------------------------------------------------------
# CoV L-kernel density (full covariance)
# ---------------------------------------------------------------------------


def _log_cov_density(v, chol_M, step_size, D):
    """Log proposal density in v-space with full covariance and Jacobian correction.

    Computes 0.5 * (log|M| - v^T M^{-1} v) - D * log(eps) where M = L @ L^T,
    dropping the -D/2*log(2*pi) constant which cancels in L - q.

    For FO/RW pass chol_M = eye(D): log|M| = 0, M^{-1} = I.
    """
    log_det_M = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_M)))
    Linv_v = jla.solve_triangular(chol_M, v, lower=True)
    vMv = jnp.dot(Linv_v, Linv_v)
    return 0.5 * (log_det_M - vMv) - D * jnp.log(step_size)


# ---------------------------------------------------------------------------
# Pure-JAX proposal functions (vmappable, full Hessian)
# ---------------------------------------------------------------------------


def _propose_rw(x, grad, hessian, z, eps, eps_fb):  # noqa: ARG001
    """RW proposal (Eq 28): x_new = x + eps * z."""
    D = x.shape[0]
    v = z
    v_half = v
    x_new = x + eps * v
    return x_new, v, v_half, jnp.eye(D), eps


def _propose_fo(x, grad, hessian, z, eps, eps_fb):  # noqa: ARG001
    """First-order / MALA proposal with leapfrog structure (Eq 30-33)."""
    D = x.shape[0]
    v = z
    v_half = 0.5 * eps * grad + v
    x_new = x + eps * v_half
    return x_new, v, v_half, jnp.eye(D), eps


def _propose_so(x, grad, hessian, z, eps, eps_fb):
    """Second-order proposal using full Hessian with FO fallback (Eq 39-41).

    Uses M = -H (negative Hessian) as mass matrix. When M is PSD, proposals
    sample v ~ N(0, M) via Cholesky and solve M^{-1} via cho_solve. Falls back
    to identity (FO) when M is not PSD.
    """
    D = x.shape[0]
    neg_H = -hessian
    chol_M = jla.cholesky(neg_H + jnp.eye(D) * 1e-8, lower=True)
    is_psd = jnp.all(jnp.isfinite(chol_M))

    # Safe Cholesky: identity if not PSD
    chol_safe = jnp.where(is_psd, chol_M, jnp.eye(D))
    eps_used = jnp.where(is_psd, eps, eps_fb)

    # When chol_safe = I and eps_used = eps_fb, SO reduces to FO
    v = chol_safe @ z
    v_half = 0.5 * eps_used * grad + v
    x_new = x + eps_used * jla.cho_solve((chol_safe, True), v_half)

    return x_new, v, v_half, chol_safe, eps_used


# ---------------------------------------------------------------------------
# Pure-JAX reverse momentum functions (vmappable, full Hessian)
# ---------------------------------------------------------------------------


def _reverse_rw(v_half, grad_new, hessian_new, eps, eps_fb):  # noqa: ARG001
    """RW reverse: symmetric, v_new = v_half (Eq 29)."""
    D = v_half.shape[0]
    return v_half, jnp.eye(D), eps


def _reverse_fo(v_half, grad_new, hessian_new, eps, eps_fb):  # noqa: ARG001
    """FO reverse momentum kick (Eq 34)."""
    D = v_half.shape[0]
    v_new = 0.5 * eps * grad_new + v_half
    return v_new, jnp.eye(D), eps


def _reverse_so(v_half, grad_new, hessian_new, eps, eps_fb):
    """SO reverse with FO fallback (Eq 42, 44)."""
    D = v_half.shape[0]
    neg_H = -hessian_new
    chol_M = jla.cholesky(neg_H + jnp.eye(D) * 1e-8, lower=True)
    is_psd = jnp.all(jnp.isfinite(chol_M))

    chol_safe = jnp.where(is_psd, chol_M, jnp.eye(D))
    eps_used = jnp.where(is_psd, eps, eps_fb)

    v_new = 0.5 * eps_used * grad_new + v_half
    return v_new, chol_safe, eps_used


# ---------------------------------------------------------------------------
# Pure-JAX weight update (vmappable)
# ---------------------------------------------------------------------------


def _compute_weight(
    logw_old, log_post_new, log_post_old, v, v_new, fwd_chol, rev_chol, fwd_ss, rev_ss, D
):
    """Importance weight update with CoV L-kernel correction (Eq 25)."""
    log_L = _log_cov_density(-v_new, rev_chol, rev_ss, D)
    log_q = _log_cov_density(v, fwd_chol, fwd_ss, D)
    lw = logw_old + log_post_new - log_post_old + log_L - log_q
    lw = jnp.where(jnp.isfinite(log_post_new), lw, -jnp.inf)
    lw = jnp.where(jnp.isfinite(logw_old), lw, -jnp.inf)
    return lw


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
    proposal: Literal["rw", "mala", "hessian"] = "hessian",
    step_size: float = 0.1,
    fallback_step_size: float = 0.01,
    adapt_step_size: bool = True,
    warmup_iters: int = 0,
    warmup_step_size: float | None = None,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via Hess-MC² (SMC with CoV L-kernels).

    Uses importance-weighted SMC where all proposals are accepted and
    quality is controlled through change-of-variables L-kernel weight
    correction. Gradients/Hessians target the log-posterior (Eq 9, 11).

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices for hierarchical models
        n_smc_particles: N -- number of parameter particles
        n_iterations: K -- number of SMC iterations
        proposal: "rw", "mala", or "hessian"
        step_size: epsilon -- proposal step size
        fallback_step_size: step size when Hessian is not PSD (SO only)
        adapt_step_size: adapt step size based on ESS (default True)
        warmup_iters: number of initial RW iterations before switching to the
            requested proposal. Helps avoid initial particle collapse when
            priors are diffuse relative to the posterior.
        warmup_step_size: step size during warmup (defaults to step_size)
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
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    # 2. Build differentiable functions
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
    )

    # Gradient and Hessian target the log-POSTERIOR (paper Eq 9, 11)
    def log_post_fn(z):
        return log_lik_fn(z) + log_prior_unc_fn(z)

    # --- Batched evaluators: vmap over particles, JIT the whole batch ---
    def _safe_log_post(z):
        ll = log_lik_fn(z)
        lp = log_prior_unc_fn(z)
        return jnp.where(jnp.isfinite(ll) & jnp.isfinite(lp), lp + ll, -1e30)

    batch_log_post = jax.jit(jax.vmap(_safe_log_post))
    batch_log_prior = jax.jit(jax.vmap(log_prior_unc_fn))

    # Fused value-and-gradient: saves one full forward pass (including the
    # expensive PF likelihood) per call vs separate batch_log_post + batch_grad.
    # Reverse-mode AD already computes the forward value internally.
    def _safe_val_and_grad(z):
        val, grad = jax.value_and_grad(log_post_fn)(z)
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    batch_val_and_grad = jax.jit(jax.vmap(_safe_val_and_grad))

    if proposal == "hessian":

        def _safe_full_hessian(z):
            H = jax.hessian(log_post_fn)(z)
            return jnp.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        batch_hessian = jax.jit(jax.vmap(_safe_full_hessian))

    # Select proposal/reverse functions and build vmapped batches
    if proposal == "rw":
        _prop, _rev = _propose_rw, _reverse_rw
    elif proposal == "mala":
        _prop, _rev = _propose_fo, _reverse_fo
    else:
        _prop, _rev = _propose_so, _reverse_so

    propose_batch = jax.jit(jax.vmap(_prop, in_axes=(0, 0, 0, 0, None, None)))
    reverse_batch = jax.jit(jax.vmap(_rev, in_axes=(0, 0, 0, None, None)))
    weight_batch = jax.jit(jax.vmap(_compute_weight, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)))

    # Build RW batch functions for warmup (cheap — no grad/hessian needed)
    if warmup_iters > 0 and proposal != "rw":
        propose_batch_rw = jax.jit(jax.vmap(_propose_rw, in_axes=(0, 0, 0, 0, None, None)))
        reverse_batch_rw = jax.jit(jax.vmap(_reverse_rw, in_axes=(0, 0, 0, None, None)))
        _warmup_eps = warmup_step_size if warmup_step_size is not None else step_size

    # 3. Initialize N particles from prior — sample each distribution directly
    # via JAX (vectorized), bypassing numpyro handlers which aren't vmappable.
    # Sort by key name to match ravel_pytree's pytree leaf ordering.
    warmup_tag = f", warmup={warmup_iters}" if warmup_iters > 0 else ""
    print(f"Hess-MC²: N={N}, K={K}, D={D}, proposal={proposal}, eps={step_size}{warmup_tag}")
    print(f"  Initializing {N} particles from prior...")

    parts = []
    for name in sorted(site_info.keys()):
        info = site_info[name]
        rng_key, sample_key = random.split(rng_key)
        prior_samples = info["distribution"].sample(sample_key, (N,))
        unc_samples = info["transform"].inv(prior_samples)
        parts.append(unc_samples.reshape(N, -1))

    particles = jnp.concatenate(parts, axis=1)  # (N, D)

    # Batch evaluate all initial particles at once.
    # When warmup_iters > 0, the first iterations use RW which doesn't need
    # grads or hessians — skip the expensive initial computation.
    hessians = jnp.zeros((N, D, D))
    if warmup_iters > 0:
        log_posts = batch_log_post(particles)
        grads = jnp.zeros((N, D))
    elif proposal == "rw":
        log_posts = batch_log_post(particles)
        grads = jnp.zeros((N, D))
    else:
        log_posts, grads = batch_val_and_grad(particles)
        if proposal == "hessian":
            hessians = batch_hessian(particles)

    # Initial weights and tempering setup
    init_log_priors = batch_log_prior(particles)
    log_liks = log_posts - init_log_priors

    if warmup_iters > 0:
        # Tempered warmup: gradually increase β from 1/warmup to 1.0
        # β_k * log_lik is much less peaked than log_lik for small β
        betas = [float(i + 1) / warmup_iters for i in range(warmup_iters)]
        logw = betas[0] * log_liks
        beta_prev = betas[0]
    else:
        # Standard: log w = log [π(θ)/q(θ)] = log_post - log_prior = log_lik
        logw = log_liks

    # Diagnostics and recycling storage (Eq 26, [18])
    ess_history = []
    resample_points = []
    recycled_particles = [particles]
    recycled_logw = [logw]

    # 4. Main SMC loop
    for k in range(K):
        in_warmup = warmup_iters > 0 and k < warmup_iters

        # --- Tempering increment: reweight by lik^{β_k - β_{k-1}} ---
        if in_warmup and k > 0:
            beta_k = betas[k]
            logw = logw + (beta_k - beta_prev) * log_liks
            beta_prev = beta_k

        # At transition from warmup → main proposal, recompute grads/hessians
        # (warmup iterations only track log_posts, not grads/hessians)
        if warmup_iters > 0 and k == warmup_iters:
            if proposal != "rw":
                log_posts, grads = batch_val_and_grad(particles)
            if proposal == "hessian":
                hessians = batch_hessian(particles)

        # --- Normalize weights, ESS (single logsumexp) ---
        lse = jax.nn.logsumexp(logw)
        log_wn = logw - lse
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        # --- Resample if ESS < N/2 (carry grads and hessians) ---
        did_resample = False
        if ess < N / 2:
            resample_points.append(k)
            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resample(resample_key, wn, N)
            particles = particles[idx]
            log_posts = log_posts[idx]
            grads = grads[idx]
            hessians = hessians[idx]
            if in_warmup:
                log_liks = log_liks[idx]
            logw = jnp.full(N, -jnp.log(float(N)))
            did_resample = True

        # --- Draw noise for all particles at once ---
        rng_key, noise_key = random.split(rng_key)
        z_all = random.normal(noise_key, (N, D))

        # --- Vectorized proposal (pure JAX, no handlers) ---
        if in_warmup:
            particles_new, v_all, v_half_all, fwd_chol_all, fwd_ss_all = propose_batch_rw(
                particles, grads, hessians, z_all, _warmup_eps, fallback_step_size
            )
        else:
            particles_new, v_all, v_half_all, fwd_chol_all, fwd_ss_all = propose_batch(
                particles, grads, hessians, z_all, step_size, fallback_step_size
            )

        # --- Batch evaluate all new particles (vmapped) ---
        if in_warmup:
            # During warmup, skip expensive grad/hessian computation
            log_posts_new = batch_log_post(particles_new)
            grads_new = jnp.zeros((N, D))
            hessians_new = jnp.zeros((N, D, D))
        else:
            hessians_new = jnp.zeros((N, D, D))
            if proposal == "rw":
                log_posts_new = batch_log_post(particles_new)
                grads_new = jnp.zeros((N, D))
            else:
                log_posts_new, grads_new = batch_val_and_grad(particles_new)
            if proposal == "hessian":
                hessians_new = batch_hessian(particles_new)

        # --- Vectorized reverse momentum + weight update ---
        if in_warmup:
            v_new_all, rev_chol_all, rev_ss_all = reverse_batch_rw(
                v_half_all, grads_new, hessians_new, _warmup_eps, fallback_step_size
            )
            # Weight update uses tempered posterior: β*lik + prior
            beta_k = betas[k]
            log_priors_new = batch_log_prior(particles_new)
            log_liks_new = log_posts_new - log_priors_new
            tempered_new = beta_k * log_liks_new + log_priors_new
            log_priors_old = batch_log_prior(particles)
            tempered_old = beta_k * log_liks + log_priors_old
        else:
            v_new_all, rev_chol_all, rev_ss_all = reverse_batch(
                v_half_all, grads_new, hessians_new, step_size, fallback_step_size
            )
            tempered_new = log_posts_new
            tempered_old = log_posts

        logw = weight_batch(
            logw,
            tempered_new,
            tempered_old,
            v_all,
            v_new_all,
            fwd_chol_all,
            rev_chol_all,
            fwd_ss_all,
            rev_ss_all,
            D,
        )

        # Update state
        particles = particles_new
        log_posts = log_posts_new
        grads = grads_new
        hessians = hessians_new
        if in_warmup:
            log_liks = log_liks_new

        # Adapt step size based on ESS (only for main proposal, not warmup)
        if adapt_step_size and not in_warmup:
            ess_ratio = ess / N
            if ess_ratio > 0.8:
                step_size = min(step_size * 1.1, 1.0)
            elif ess_ratio < 0.2:
                step_size = max(step_size * 0.8, 1e-4)

        # Store for recycling (Eq 26, [18]) — only after warmup (β=1)
        if not in_warmup:
            recycled_particles.append(particles)
            recycled_logw.append(logw)

        resamp_tag = " [resampled]" if did_resample else ""
        eps_tag = f"  eps={step_size:.4f}" if adapt_step_size else ""
        beta_tag = f"  β={betas[k]:.2f}" if in_warmup else ""
        warmup_label = " [warmup/RW]" if in_warmup else ""
        print(f"  step {k + 1}/{K}  ESS={ess:.1f}/{N}{resamp_tag}{eps_tag}{beta_tag}{warmup_label}")

    # 5. Final resampling from recycled pool
    rng_key, final_key = random.split(rng_key)
    all_particles = jnp.concatenate(recycled_particles, axis=0)  # ((K+1)*N, D)
    all_logw = jnp.concatenate(recycled_logw, axis=0)  # ((K+1)*N,)
    idx = _systematic_resample(final_key, jnp.exp(all_logw - jax.nn.logsumexp(all_logw)), N)
    final_particles = all_particles[idx]

    # Extract final samples in constrained space (vmapped)
    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(final_particles)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="hessmc2",
        diagnostics={
            "ess_history": ess_history,
            "resample_points": resample_points,
            "n_smc_particles": N,
            "n_iterations": K,
            "proposal": proposal,
            "step_size": step_size,
            "warmup_iters": warmup_iters,
        },
    )
