"""Shared utilities for inference backends.

Functions used by MAP, SVI, blocked MCMC, and parametric-id diagnostics:
- _discover_sites: trace model to discover sample sites
- _assemble_deterministics: build SSM matrices from constrained samples
- _build_eval_fns: build differentiable log-likelihood and log-prior evaluators
- _build_runtime_eval_fns_from_registry: compile-stable evaluators for diagnostics
"""

from __future__ import annotations

import functools
import time
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
            input_effect=det.get("input_effect"),
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


def _block_until_ready_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _constrain_particles_batched(
    particles: jnp.ndarray,
    site_info: dict,
    unravel_fn,
) -> dict[str, jnp.ndarray]:
    """Map a batch of unconstrained particles to constrained site samples."""
    if not site_info:
        return {}

    transforms = {name: info["transform"] for name, info in site_info.items()}
    unconstrained = jax.vmap(unravel_fn)(particles)
    return {name: jax.vmap(transforms[name])(unconstrained[name]) for name in unconstrained}


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
    profiling: dict[str, float] | None = None,
) -> dict[str, jnp.ndarray]:
    """Extract constrained samples from unconstrained particles and assemble deterministics.

    Shared by parameter-space and blocked-MCMC backends to avoid code duplication.

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
    total_start = time.perf_counter()
    constrain_start = time.perf_counter()
    samples = _constrain_particles_batched(particles, site_info, unravel_fn)
    _block_until_ready_tree(samples)
    if profiling is not None:
        profiling["constrain_batched_seconds"] = time.perf_counter() - constrain_start

    if reparam is None:
        structure_runtime = (
            model.structure_runtime if model is not None else SSMStructureRuntime(spec)
        )
        det_start = time.perf_counter()
        det_samples = _assemble_deterministics(
            samples,
            spec,
            structure_runtime=structure_runtime,
        )
        _block_until_ready_tree(det_samples)
        if profiling is not None:
            profiling["deterministic_assembly_seconds"] = time.perf_counter() - det_start
        samples.update(det_samples)
        if profiling is not None:
            profiling["extract_constrained_samples_total_seconds"] = (
                time.perf_counter() - total_start
            )
        return samples

    if observations is None or times is None:
        raise ValueError(
            "extract_constrained_samples requires observations and times when reparam is enabled"
        )

    resolver_build_start = time.perf_counter()
    sample_resolver = _build_original_sample_resolver(
        site_info,
        model=model,
        observations=observations,
        times=times,
        reparam=reparam,
    )
    if profiling is not None:
        profiling["resolver_build_seconds"] = time.perf_counter() - resolver_build_start
    if sample_resolver is None:
        raise ValueError(
            "extract_constrained_samples only supports no reparameterization "
            "or AutoReparam with fixed centering."
        )

    resolve_start = time.perf_counter()
    original_samples = sample_resolver(samples)
    _block_until_ready_tree(original_samples)
    if profiling is not None:
        profiling["original_sample_resolution_seconds"] = time.perf_counter() - resolve_start

    # Assemble deterministic matrices (drift, diffusion, lambda, etc.)
    structure_runtime = model.structure_runtime if model is not None else SSMStructureRuntime(spec)
    det_start = time.perf_counter()
    det_samples = _assemble_deterministics(
        original_samples,
        spec,
        structure_runtime=structure_runtime,
    )
    _block_until_ready_tree(det_samples)
    if profiling is not None:
        profiling["deterministic_assembly_seconds"] = time.perf_counter() - det_start
    original_samples.update(det_samples)
    if profiling is not None:
        profiling["extract_constrained_samples_total_seconds"] = time.perf_counter() - total_start
    return original_samples


# ---------------------------------------------------------------------------
# Differentiable evaluators
# ---------------------------------------------------------------------------


def _build_eval_fns(
    model,
    observations,
    times,
    site_info,
    unravel_fn,
    likelihood_backend,
    reparam=None,
    *,
    include_likelihood_aux: bool = False,
    runtime_observations_times: bool = False,
):
    """Build differentiable functions for log-likelihood and log-prior.

    Args:
        likelihood_backend: Likelihood backend instance to use for evaluation.
        reparam: Optional reparameterization config (Strategy, dict, or None).

    Returns:
        When ``runtime_observations_times=False``:
        log_lik_fn(z) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
        log_lik_with_aux_fn(z) -> (scalar log p(y|theta), aux pytree), when requested

        When ``runtime_observations_times=True``:
        log_lik_fn(z, observations, times) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
        log_lik_with_aux_fn(z, observations, times) -> (scalar log p(y|theta), aux pytree),
        when requested
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
    bound_transition_inputs = getattr(model, "transition_inputs", None)

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def _log_lik_fn_bound(z, latent_mode_init=None):
        """Log-likelihood p(y|theta) via the configured backend only."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=structure_runtime,
        )
        time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)
        if latent_mode_init is None:
            lnc = likelihood_backend.compute_log_likelihood(
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
            )
        else:
            lnc = likelihood_backend.compute_log_likelihood(
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
                latent_mode_init=latent_mode_init,
            )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    def _log_lik_fn_runtime(z, runtime_observations, runtime_times, latent_mode_init=None):
        """Runtime-argument log-likelihood p(y|theta)."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=structure_runtime,
        )
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)
        runtime_transition_inputs = (
            None
            if bound_transition_inputs is None
            else bound_transition_inputs[: runtime_times.shape[0]]
        )
        if latent_mode_init is None:
            lnc = likelihood_backend.compute_log_likelihood(
                ct_params,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
            )
        else:
            lnc = likelihood_backend.compute_log_likelihood(
                ct_params,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
                latent_mode_init=latent_mode_init,
            )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    log_lik_base = _log_lik_fn_runtime if runtime_observations_times else _log_lik_fn_bound
    log_lik_fn = (
        jax.checkpoint(log_lik_base) if likelihood_backend.checkpoint_loglik else log_lik_base
    )

    def _log_lik_with_aux_fn_bound(z, latent_mode_init=None):
        """Log-likelihood p(y|theta) plus a fixed-shape backend aux payload."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=structure_runtime,
        )
        time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)
        if latent_mode_init is None:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
            )
        else:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
                latent_mode_init=latent_mode_init,
            )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        total_ll = jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)
        return total_ll, aux

    def _log_lik_with_aux_fn_runtime(
        z,
        runtime_observations,
        runtime_times,
        latent_mode_init=None,
    ):
        """Runtime-argument log-likelihood p(y|theta) plus fixed-shape aux."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=structure_runtime,
        )
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)
        runtime_transition_inputs = (
            None
            if bound_transition_inputs is None
            else bound_transition_inputs[: runtime_times.shape[0]]
        )
        if latent_mode_init is None:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                ct_params,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
            )
        else:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                ct_params,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
                latent_mode_init=latent_mode_init,
            )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        total_ll = jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)
        return total_ll, aux

    # The aux payload can include latent-mode state reused across outer evaluations.
    # Rematerializing that path leaks tracers through the returned aux tree.
    log_lik_with_aux_fn = (
        _log_lik_with_aux_fn_runtime if runtime_observations_times else _log_lik_with_aux_fn_bound
    )

    def log_prior_unc_fn(z):
        """Log-prior in unconstrained space: log p(T(z)) + log|J(z)|."""
        con, unc = _constrain(z)
        lp = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in unc)
        lj = sum(
            jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name])) for name in unc
        )
        return lp + lj

    if include_likelihood_aux:
        return log_lik_fn, log_prior_unc_fn, log_lik_with_aux_fn
    return log_lik_fn, log_prior_unc_fn


def _build_runtime_eval_fns_from_registry(
    spec,
    registry,
    unravel_fn,
    transforms,
    structure_runtime,
    likelihood_backend,
    transition_inputs=None,
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
        runtime_transition_inputs = (
            None if transition_inputs is None else transition_inputs[: times.shape[0]]
        )
        lnc = likelihood_backend.compute_log_likelihood(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            extra_params=extra_params,
            transition_inputs=runtime_transition_inputs,
        )
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    log_lik_fn = (
        jax.checkpoint(_log_lik_fn) if likelihood_backend.checkpoint_loglik else _log_lik_fn
    )

    def log_prior_unc_fn(z, prior_state):
        return log_prior_unconstrained(z, unravel_fn, registry, prior_state)

    return log_lik_fn, log_prior_unc_fn
