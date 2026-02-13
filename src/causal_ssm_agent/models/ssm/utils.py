"""Shared utilities for SMC/MCMC inference backends.

Functions used by hessmc2, pgas, tempered_smc, and parametric_id:
- _discover_sites: trace model to discover sample sites
- _assemble_deterministics: build SSM matrices from constrained samples
- _build_eval_fns: build differentiable log-likelihood and log-prior evaluators
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree  # noqa: F401 — re-exported for callers
from numpyro import handlers

from causal_ssm_agent.models.ssm.inference import _eval_model

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(model, observations, times, rng_key, likelihood_backend):
    """Trace model once to discover sample sites (names, shapes, transforms)."""
    model_fn = functools.partial(model.model, likelihood_backend=likelihood_backend)
    with handlers.seed(rng_seed=int(rng_key[0])):
        trace = handlers.trace(model_fn).get_trace(observations, times)

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


# ---------------------------------------------------------------------------
# Pure-JAX deterministic site assembly
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
# Differentiable evaluators
# ---------------------------------------------------------------------------


def _build_eval_fns(model, observations, times, site_info, unravel_fn, likelihood_backend):
    """Build differentiable functions for log-likelihood and log-prior.

    Args:
        likelihood_backend: Likelihood backend instance to use for evaluation.

    Returns:
        log_lik_fn(z) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    model_fn = functools.partial(model.model, likelihood_backend=likelihood_backend)

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def _log_lik_fn(z):
        """Log-likelihood p(y|theta) via PF or Kalman."""
        con, _ = _constrain(z)
        log_lik, _ = _eval_model(model_fn, con, observations, times)
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
