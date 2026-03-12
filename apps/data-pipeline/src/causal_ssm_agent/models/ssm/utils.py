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

from causal_ssm_agent.models.ssm.assembler import SSMAssembler
from causal_ssm_agent.models.ssm.inference import _eval_model

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(model, observations, times, rng_key, likelihood_backend, reparam=None):
    """Trace model once to discover sample sites (names, shapes, transforms)."""
    model_fn = functools.partial(model.model, likelihood_backend=likelihood_backend)
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)
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
    (e.g. drift_diag_pop, drift_offdiag_pop → drift matrix). Uses SSMAssembler
    as the single source of truth for matrix construction (shared with
    SSMModel._sample_*).
    """
    N = next(iter(samples.values())).shape[0]
    n_l, n_m = spec.n_latent, spec.n_manifest
    asm = SSMAssembler(spec)
    det = {}

    if "drift_diag_pop" in samples:
        offdiag = samples.get(
            "drift_offdiag_pop",
            jnp.zeros((N, max(len(asm.offdiag_positions), 0))),
        )
        det["drift"] = jax.vmap(asm.assemble_drift)(samples["drift_diag_pop"], offdiag)

    if "diffusion_diag_pop" in samples:
        if "diffusion_lower" in samples:
            det["diffusion"] = jax.vmap(asm.assemble_diffusion)(
                samples["diffusion_diag_pop"], samples["diffusion_lower"]
            )
        else:
            det["diffusion"] = jax.vmap(asm.assemble_diffusion)(samples["diffusion_diag_pop"])

    if "cint_pop" in samples:
        det["cint"] = samples["cint_pop"]

    if "lambda_free" in samples and len(asm.lambda_free_positions) > 0:
        det["lambda"] = jax.vmap(asm.assemble_lambda)(samples["lambda_free"])
    else:
        det["lambda"] = jnp.broadcast_to(asm.lambda_template, (N, n_m, n_l))

    if "manifest_var_diag" in samples:
        det["manifest_cov"] = jax.vmap(lambda d: jnp.diag(d**2))(samples["manifest_var_diag"])
    elif isinstance(spec.manifest_var, jnp.ndarray):
        fixed_cov = spec.manifest_var @ spec.manifest_var.T
        det["manifest_cov"] = jnp.broadcast_to(fixed_cov, (N, n_m, n_m))

    if "t0_means_pop" in samples:
        det["t0_means"] = samples["t0_means_pop"]
    elif isinstance(spec.t0_means, jnp.ndarray):
        det["t0_means"] = jnp.broadcast_to(spec.t0_means, (N, n_l))

    if "t0_var_diag" in samples:
        det["t0_cov"] = jax.vmap(lambda d: jnp.diag(d**2))(samples["t0_var_diag"])
    elif isinstance(spec.t0_var, jnp.ndarray):
        fixed_cov = spec.t0_var @ spec.t0_var.T
        det["t0_cov"] = jnp.broadcast_to(fixed_cov, (N, n_l, n_l))

    return det


# ---------------------------------------------------------------------------
# Sample extraction from unconstrained particles
# ---------------------------------------------------------------------------


class _DummyLikelihoodBackend:
    """Dummy backend for model replay — returns zero log-likelihood."""

    def compute_log_likelihood(self, *_args, **_kwargs):
        return jnp.array(0.0)


def extract_constrained_samples(
    particles: jnp.ndarray,
    site_info: dict,
    unravel_fn,
    spec: SSMSpec,
    *,
    reparam=None,
    model=None,
    observations: jnp.ndarray | None = None,
    times: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Extract constrained samples from unconstrained particles and assemble deterministics.

    Shared by hessmc2, pgas, and tempered_core to avoid code duplication.

    When ``reparam`` is provided, replays the reparameterized model to recover
    original parameter names and assembled matrices via the model's own
    deterministic sites.

    Args:
        particles: (N, D) array of unconstrained parameter vectors
        site_info: site info dict from _discover_sites
        unravel_fn: function from ravel_pytree to unravel flat vectors
        spec: SSMSpec for assembling deterministic sites
        reparam: Optional reparameterization config (Strategy, dict, or None).
        model: SSMModel instance (required when reparam is provided).
        observations: Observed data (required when reparam is provided).
        times: Time points (required when reparam is provided).

    Returns:
        Dict of constrained samples including deterministic sites
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(particles)

    if reparam is None:
        det_samples = _assemble_deterministics(samples, spec)
        samples.update(det_samples)
        return samples

    # Reparam path: recover original parameter names + assemble deterministics.
    #
    # For AutoReparam(centered=0.0), LocScaleReparam creates auxiliary sample
    # sites named "{name}_decentered" with N(0,1) prior, and the original value
    # is: original = prior.loc + prior.scale * decentered.  This is a simple
    # vectorized op, so we reverse it without N sequential model replays.
    #
    # Non-reparameterized sites (HalfNormal, Gamma, TruncatedNormal, etc.)
    # keep their original names in the trace and are used directly.

    # Trace the non-reparameterized model to get original sample site distributions.
    base_replay_fn = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    with handlers.seed(rng_seed=0):
        public_trace = handlers.trace(base_replay_fn).get_trace(observations, times)

    # Recover original sample site values by reversing LocScaleReparam vectorized.
    original_samples: dict[str, jnp.ndarray] = {}
    for site_name, site in public_trace.items():
        if (
            site["type"] != "sample"
            or site.get("is_observed", False)
            or site_name in {"log_likelihood", "ll_per_timestep"}
        ):
            continue

        decentered_key = f"{site_name}_decentered"
        if decentered_key in samples:
            # LocScaleReparam: original = loc + scale * decentered
            # Unwrap Independent/Expanded/Masked to access loc/scale.
            d = site["fn"]
            while isinstance(
                d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
            ):
                d = d.base_dist
            original_samples[site_name] = d.loc + d.scale * samples[decentered_key]
        elif site_name in samples:
            # Not reparameterized — use directly
            original_samples[site_name] = samples[site_name]

    # Assemble deterministic matrices (drift, diffusion, lambda, etc.)
    det_samples = _assemble_deterministics(original_samples, spec)
    original_samples.update(det_samples)
    return original_samples


# ---------------------------------------------------------------------------
# Differentiable evaluators
# ---------------------------------------------------------------------------


def _build_eval_fns(
    model, observations, times, site_info, unravel_fn, likelihood_backend, reparam=None
):
    """Build differentiable functions for log-likelihood and log-prior.

    Args:
        likelihood_backend: Likelihood backend instance to use for evaluation.
        reparam: Optional reparameterization config (Strategy, dict, or None).

    Returns:
        log_lik_fn(z) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    model_fn = functools.partial(model.model, likelihood_backend=likelihood_backend)
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)

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
