"""Shared utilities for inference backends.

Functions used by MAP, blocked MCMC, and post-fit diagnostics:
- _discover_sites: trace model to discover sample sites
- _assemble_deterministics: build SSM matrices from constrained samples
- _build_eval_fns: build differentiable log-likelihood and log-prior evaluators
- _build_runtime_eval_fns_from_registry: compile-stable evaluators for diagnostics
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from nof1_causal_lab.models.ssm.autoreparam import fixed_autoreparam_centering
from nof1_causal_lab.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES, MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
)
from nof1_causal_lab.models.ssm.dynamics.runtime import (
    build_vector_field_runtime_from_samples,
)
from nof1_causal_lab.models.ssm.inference.targets.base import (
    InitialStateParams,
    MeasurementParams,
    RuntimeDynamics,
)
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_site_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.model import SSMSpec


class SiteInfoEntry(TypedDict):
    """Typed trace metadata for one unconstrained NumPyro sample site."""

    shape: tuple[int, ...]
    distribution: dist.Distribution
    transform: dist.transforms.Transform
    value: jnp.ndarray


type SiteInfo = dict[str, SiteInfoEntry]


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(
    model, observations, times, rng_key, likelihood_backend, reparam=None
) -> SiteInfo:
    """Trace model once to discover sample sites (names, shapes, transforms).

    Site discovery is structural: it only needs the latent sample/deterministic
    sites emitted by ``model.model``. Tracing through the real likelihood backend
    can trigger large JAX/XLA compilations for support-aware Laplace before
    inference even starts, so discovery always replays the model with the dummy
    backend instead.
    """
    _ = likelihood_backend
    model_fn = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)
    with handlers.seed(rng_seed=int(rng_key[0])):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    site_info: SiteInfo = {}
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
) -> dict[str, jnp.ndarray]:
    """Thin wrapper over deterministic assembly owned by ``spec``."""
    return assemble_deterministics_from_registry(samples, spec)


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
    site_info: SiteInfo,
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


@dataclass(frozen=True)
class UnconstrainedSiteTransform:
    """Shared bijection between dict-keyed NumPyro sample sites and a
    flat unconstrained vector.

    The blocked MCMC paths need per-site ``biject_to(support)`` bijections,
    a ravel/unravel pair, and a combined ``log_prior(constrained) +
    log_abs_det_jacobian`` callable on the flat unconstrained representation.

    Built from a site-info dict where each entry has ``transform``
    (numpyro Transform), ``value`` (a constrained draw used to fix ravel
    order), and ``distribution`` (numpyro Distribution for the prior
    log-prob). Both call sites feed this builder; their statistical-model-specific
    layer (component-tuple shape, sample-resolver) sits on top.
    """

    flat_init: jnp.ndarray
    dim: int
    site_names: tuple[str, ...]
    constrain_dict: Callable[[jnp.ndarray], dict[str, jnp.ndarray]]
    unconstrain_dict: Callable[[jnp.ndarray], dict[str, jnp.ndarray]]
    log_prior_unc: Callable[[jnp.ndarray], jnp.ndarray]
    log_abs_det_jacobian: Callable[[jnp.ndarray], jnp.ndarray]


def build_unconstrained_site_transform(
    site_info: SiteInfo,
) -> UnconstrainedSiteTransform:
    """Generic builder over a NumPyro-trace-shaped site dict.

    ``site_info[name]`` must contain:
      - ``"transform"``: a NumPyro ``Transform`` (typically ``biject_to(support)``)
      - ``"value"``: an initial constrained value (only its shape/dtype matter)
      - ``"distribution"``: a NumPyro ``Distribution`` for ``log_prob`` evaluation

    Returns a closure-free dataclass; callers can add statistical-model-specific glue
    (component-layout reshaping, sample resolvers) on top without
    touching the bijection logic.
    """
    if not site_info:
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return UnconstrainedSiteTransform(
            flat_init=empty,
            dim=0,
            site_names=(),
            constrain_dict=lambda _z: {},
            unconstrain_dict=lambda _z: {},
            log_prior_unc=lambda _z: jnp.asarray(0.0),
            log_abs_det_jacobian=lambda _z: jnp.asarray(0.0),
        )

    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}
    init_unc = {name: transforms[name].inv(info["value"]) for name, info in site_info.items()}
    flat_init, unravel_fn = ravel_pytree(init_unc)
    flat_dtype = flat_init.dtype

    def constrain_dict(z: jnp.ndarray) -> dict[str, jnp.ndarray]:
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}

    def unconstrain_dict(z: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return unravel_fn(z)

    def log_abs_det_jacobian(z: jnp.ndarray) -> jnp.ndarray:
        unc = unravel_fn(z)
        total = jnp.asarray(0.0, dtype=flat_dtype)
        for name in unc:
            con = transforms[name](unc[name])
            total = total + jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con))
        return total

    def log_prior_unc(z: jnp.ndarray) -> jnp.ndarray:
        unc = unravel_fn(z)
        log_prior = jnp.asarray(0.0, dtype=flat_dtype)
        log_jac = jnp.asarray(0.0, dtype=flat_dtype)
        for name in unc:
            con = transforms[name](unc[name])
            log_prior = log_prior + jnp.sum(distributions[name].log_prob(con))
            log_jac = log_jac + jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con))
        return log_prior + log_jac

    return UnconstrainedSiteTransform(
        flat_init=flat_init,
        dim=int(flat_init.shape[0]),
        site_names=tuple(site_info.keys()),
        constrain_dict=constrain_dict,
        unconstrain_dict=unconstrain_dict,
        log_prior_unc=log_prior_unc,
        log_abs_det_jacobian=log_abs_det_jacobian,
    )


def _assemble_single_likelihood_deterministics(
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
) -> dict[str, jnp.ndarray]:
    """Assemble one draw's non-dynamics SSM pieces."""
    det: dict[str, jnp.ndarray] = {}
    det["diffusion"] = spec.diffusion_block.assemble(
        samples.get("diffusion_diag_free"),
        samples.get("diffusion_lower_free"),
    )
    det["input_effect"] = spec.input_effect_block.assemble(samples.get("input_effect_free"))
    det["static_state_sds"] = spec.static_state_sd_block.assemble(
        samples.get("static_state_sd_free")
    )
    det["lambda"] = spec.lambda_block.assemble(samples.get("lambda_free"))
    det["manifest_means"] = spec.manifest_means_block.assemble(samples.get("manifest_means_free"))
    manifest_chol = spec.manifest_chol_block.assemble(samples.get("manifest_var_diag_free"))
    det["manifest_cov"] = manifest_chol @ manifest_chol.T
    det["t0_means"] = spec.t0_means_block.assemble(samples.get("t0_means_free"))

    t0_cov = spec.t0_chol_block.assemble_cov(
        samples.get("t0_var_diag_free"),
        samples.get("t0_var_lower_free"),
    )
    static_sds = det["static_state_sds"]
    if static_sds.size:
        loadings = jnp.asarray(spec.static_factor_loadings)
        t0_cov = t0_cov + loadings @ jnp.diag(static_sds**2) @ loadings.T
    stable_cov, _ = stabilize_covariance_for_cholesky(
        0.5 * (t0_cov + t0_cov.T),
        min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
    )
    det["t0_cov"] = stable_cov
    return det


def _deterministics_to_likelihood_inputs(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    det: dict[str, jnp.ndarray],
) -> tuple[RuntimeDynamics, MeasurementParams, InitialStateParams]:
    """Convert one deterministic draw into backend parameter dataclasses."""
    diffusion_chol = det["diffusion"]
    vf_runtime = build_vector_field_runtime_from_samples(spec, samples)
    return (
        RuntimeDynamics(
            vector_field=vf_runtime.vector_field,
            vf_params=vf_runtime.vf_params,
            diffusion_cov=diffusion_chol @ diffusion_chol.T,
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
) -> tuple[RuntimeDynamics, MeasurementParams, InitialStateParams, dict[str, jnp.ndarray] | None]:
    """Build backend-ready parameter tuples from constrained sample sites."""
    det = _assemble_single_likelihood_deterministics(
        samples,
        spec,
    )
    dynamics, measurement_params, initial_state = _deterministics_to_likelihood_inputs(
        spec,
        samples,
        det,
    )

    runtime_registry = registry if registry is not None else build_site_registry(spec)
    extra_params = assemble_extra_params_from_registry(spec, samples, runtime_registry)

    return (
        dynamics,
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
        det_start = time.perf_counter()
        det_samples = _assemble_deterministics(samples, spec)
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
    det_start = time.perf_counter()
    det_samples = _assemble_deterministics(original_samples, spec)
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
    runtime_registry = build_site_registry(model.spec)
    bound_transition_inputs = getattr(model, "transition_inputs", None)

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def _log_lik_fn_bound(z, latent_mode_init=None):
        """Log-likelihood p(y|theta) via the configured backend only."""
        con, _ = _constrain(z)
        original_samples = con if sample_resolver is None else sample_resolver(con)
        dynamics, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
        )
        time_intervals = (
            jnp.diff(times, prepend=times[0]).at[0].set(jnp.asarray(MIN_DT, dtype=times.dtype))
        )
        if latent_mode_init is None:
            lnc = likelihood_backend.compute_log_likelihood(
                dynamics,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
            )
        else:
            lnc = likelihood_backend.compute_log_likelihood(
                dynamics,
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
        dynamics, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
        )
        time_intervals = (
            jnp.diff(runtime_times, prepend=runtime_times[0])
            .at[0]
            .set(jnp.asarray(MIN_DT, dtype=runtime_times.dtype))
        )
        runtime_transition_inputs = (
            None
            if bound_transition_inputs is None
            else bound_transition_inputs[: runtime_times.shape[0]]
        )
        if latent_mode_init is None:
            lnc = likelihood_backend.compute_log_likelihood(
                dynamics,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
            )
        else:
            lnc = likelihood_backend.compute_log_likelihood(
                dynamics,
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
        dynamics, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
        )
        time_intervals = (
            jnp.diff(times, prepend=times[0]).at[0].set(jnp.asarray(MIN_DT, dtype=times.dtype))
        )
        if latent_mode_init is None:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                dynamics,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=bound_transition_inputs,
            )
        else:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                dynamics,
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
        dynamics, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
        )
        time_intervals = (
            jnp.diff(runtime_times, prepend=runtime_times[0])
            .at[0]
            .set(jnp.asarray(MIN_DT, dtype=runtime_times.dtype))
        )
        runtime_transition_inputs = (
            None
            if bound_transition_inputs is None
            else bound_transition_inputs[: runtime_times.shape[0]]
        )
        if latent_mode_init is None:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                dynamics,
                measurement_params,
                initial_state,
                runtime_observations,
                time_intervals,
                extra_params=extra_params,
                transition_inputs=runtime_transition_inputs,
            )
        else:
            lnc, aux = likelihood_backend.compute_log_likelihood_with_aux(
                dynamics,
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
