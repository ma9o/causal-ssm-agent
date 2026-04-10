"""Shared utilities for SMC/MCMC inference backends.

Functions used by hessmc2, pgas, tempered_smc, and parametric_id:
- _discover_sites: trace model to discover sample sites
- _assemble_deterministics: build SSM matrices from constrained samples
- _build_eval_fns: build differentiable log-likelihood and log-prior evaluators
- _build_runtime_eval_fns_from_registry: compile-stable evaluators for diagnostics
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree  # noqa: F401 — re-exported for callers
from numpyro import handlers

from causal_ssm_agent.models.ssm.autoreparam import fixed_autoreparam_centering
from causal_ssm_agent.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES, MIN_DT
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_site_registry,
)
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(model, observations, times, rng_key, likelihood_backend, reparam=None):
    """Trace model once to discover sample sites (names, shapes, transforms).

    Site discovery is structural: it only needs the latent sample/deterministic
    sites emitted by ``model.model``. Tracing through the real likelihood backend
    can trigger large JAX/XLA compilations for support-aware Laplace and
    particle-filter backends before inference even starts, so discovery always
    replays the model with the dummy backend instead.
    """
    _ = likelihood_backend
    model_fn = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)
    with handlers.seed(rng_seed=int(rng_key[0])):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    site_info = {}
    for name, site in trace.items():
        if (
            site["type"] == "sample"
            and not site.get("is_observed", False)
            and name not in INTERNAL_DIAGNOSTIC_SITES
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
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
    *,
    registry=None,
    structure_runtime: SSMStructureRuntime | None = None,
) -> dict[str, jnp.ndarray]:
    """Thin wrapper over the registry-driven deterministic assembly path."""
    if structure_runtime is None:
        structure_runtime = SSMStructureRuntime(spec)
    if registry is None:
        registry = build_site_registry(spec, structure_runtime)
    return assemble_deterministics_from_registry(
        samples,
        spec,
        registry,
        structure_runtime=structure_runtime,
    )


# ---------------------------------------------------------------------------
# Sample extraction from unconstrained particles
# ---------------------------------------------------------------------------


class _DummyLikelihoodBackend:
    """Dummy backend for model replay — returns zero log-likelihood."""

    checkpoint_loglik = False

    def compute_log_likelihood(self, *_args, **_kwargs):
        return jnp.array(0.0)


def _unwrap_base_distribution(d: dist.Distribution) -> dist.Distribution:
    """Unwrap lightweight distribution wrappers to access base parameters."""
    while isinstance(d, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
        d = d.base_dist
    return d


def _build_original_sample_resolver(
    site_info: dict,
    *,
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    reparam=None,
):
    """Build a JAX-compatible resolver from reparameterized to original sample sites."""
    if reparam is None:
        return lambda samples: samples

    centered = fixed_autoreparam_centering(reparam)
    if centered is None:
        return None

    base_replay_fn = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    with handlers.seed(rng_seed=0):
        public_trace = handlers.trace(base_replay_fn).get_trace(observations, times)

    passthrough_sites: list[str] = []
    decentered_rules: list[tuple[str, str, jnp.ndarray, jnp.ndarray]] = []
    for site_name, site in public_trace.items():
        if (
            site["type"] != "sample"
            or site.get("is_observed", False)
            or site_name in INTERNAL_DIAGNOSTIC_SITES
        ):
            continue

        decentered_key = f"{site_name}_decentered"
        if decentered_key in site_info:
            d = _unwrap_base_distribution(site["fn"])
            loc = getattr(d, "loc", None)
            scale = getattr(d, "scale", None)
            if loc is None or scale is None:
                raise ValueError(
                    f"Decentered site {site_name!r} must use a loc/scale prior distribution"
                )
            decentered_rules.append(
                (site_name, decentered_key, jnp.asarray(loc), jnp.asarray(scale))
            )
        elif site_name in site_info:
            passthrough_sites.append(site_name)

    def _resolve(samples: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        original_samples = {name: samples[name] for name in passthrough_sites}
        for site_name, decentered_key, loc, scale in decentered_rules:
            decentered_value = samples[decentered_key]
            if centered == 0.0:
                value = loc + scale * decentered_value
            else:
                delta = decentered_value - centered * loc
                value = loc + jnp.power(scale, 1.0 - centered) * delta
            original_samples[site_name] = value
        return original_samples

    return _resolve


def _assemble_single_deterministics(
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
    *,
    structure_runtime: SSMStructureRuntime | None = None,
) -> dict[str, jnp.ndarray]:
    """Assemble deterministic sites for a single constrained parameter draw."""
    if not samples:
        return {}
    det = _assemble_deterministics(
        {name: value[None, ...] for name, value in samples.items()},
        spec,
        structure_runtime=structure_runtime,
    )
    return {name: value[0] for name, value in det.items()}


def _deterministics_to_likelihood_inputs(
    det: dict[str, jnp.ndarray],
) -> tuple[CTParams, MeasurementParams, InitialStateParams]:
    """Convert one deterministic draw into backend parameter dataclasses."""
    diffusion_chol = det["diffusion"]
    return (
        CTParams(
            drift=det["drift"],
            diffusion_cov=diffusion_chol @ diffusion_chol.T,
            cint=det["cint"],
        ),
        MeasurementParams(
            lambda_mat=det["lambda"],
            manifest_means=det["manifest_means"],
            manifest_cov=det["manifest_cov"],
        ),
        InitialStateParams(
            mean=det["t0_means"],
            cov=det["t0_cov"],
        ),
    )


def _assemble_likelihood_inputs(
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
    registry=None,
    structure_runtime: SSMStructureRuntime | None = None,
) -> tuple[CTParams, MeasurementParams, InitialStateParams, dict[str, jnp.ndarray] | None]:
    """Build backend-ready parameter tuples from constrained sample sites."""
    if structure_runtime is None:
        structure_runtime = SSMStructureRuntime(spec)
    det = _assemble_single_deterministics(
        samples,
        spec,
        structure_runtime=structure_runtime,
    )
    ct_params, measurement_params, initial_state = _deterministics_to_likelihood_inputs(det)

    runtime_registry = (
        registry if registry is not None else build_site_registry(spec, structure_runtime)
    )
    extra_params = assemble_extra_params_from_registry(spec, samples, runtime_registry)

    return (
        ct_params,
        measurement_params,
        initial_state,
        extra_params or None,
    )


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
        structure_runtime = (
            model.structure_runtime if model is not None else SSMStructureRuntime(spec)
        )
        det_samples = _assemble_deterministics(
            samples,
            spec,
            structure_runtime=structure_runtime,
        )
        samples.update(det_samples)
        return samples

    if observations is None or times is None:
        raise ValueError(
            "extract_constrained_samples requires observations and times when reparam is enabled"
        )

    sample_resolver = _build_original_sample_resolver(
        site_info,
        model=model,
        observations=observations,
        times=times,
        reparam=reparam,
    )
    if sample_resolver is None:
        raise ValueError(
            "extract_constrained_samples only supports no reparameterization "
            "or AutoReparam with fixed centering."
        )

    original_samples = sample_resolver(samples)

    # Assemble deterministic matrices (drift, diffusion, lambda, etc.)
    structure_runtime = model.structure_runtime if model is not None else SSMStructureRuntime(spec)
    det_samples = _assemble_deterministics(
        original_samples,
        spec,
        structure_runtime=structure_runtime,
    )
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
    sample_resolver = _build_original_sample_resolver(
        site_info,
        model=model,
        observations=observations,
        times=times,
        reparam=reparam,
    )
    structure_runtime = model.structure_runtime
    runtime_registry = build_site_registry(model.spec, structure_runtime)
    time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def _log_lik_fn(z):
        """Log-likelihood p(y|theta) via the configured backend only."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=structure_runtime,
        )
        lnc = likelihood_backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            extra_params=extra_params,
        )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    log_lik_fn = (
        jax.checkpoint(_log_lik_fn) if likelihood_backend.checkpoint_loglik else _log_lik_fn
    )

    def log_prior_unc_fn(z):
        """Log-prior in unconstrained space: log p(T(z)) + log|J(z)|."""
        con, unc = _constrain(z)
        lp = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in unc)
        lj = sum(
            jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name])) for name in unc
        )
        return lp + lj

    return log_lik_fn, log_prior_unc_fn


def _build_runtime_eval_fns_from_registry(
    spec,
    registry,
    unravel_fn,
    transforms,
    structure_runtime,
    likelihood_backend,
):
    """Build compile-stable evaluators that do not close over traced model state.

    This is intended for Stage 4b sweep-style diagnostics that repeatedly vary
    prior values while keeping the model topology fixed. The returned
    log-likelihood takes ``observations`` and ``times`` as runtime arguments so
    the same compiled closure can be reused across many sweeps in one process.
    """
    from causal_ssm_agent.models.ssm.parameterization import log_prior_unconstrained

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}

    def _log_lik_fn(z, observations, times):
        con = _constrain(z)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            con,
            spec,
            registry=registry,
            structure_runtime=structure_runtime,
        )
        time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)
        lnc = likelihood_backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            extra_params=extra_params,
        )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    log_lik_fn = (
        jax.checkpoint(_log_lik_fn) if likelihood_backend.checkpoint_loglik else _log_lik_fn
    )

    def log_prior_unc_fn(z, prior_state):
        return log_prior_unconstrained(z, unravel_fn, registry, prior_state)

    return log_lik_fn, log_prior_unc_fn
