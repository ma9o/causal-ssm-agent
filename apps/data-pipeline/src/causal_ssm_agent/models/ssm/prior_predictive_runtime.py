"""Compile-stable prior predictive runtime.

Builds prior predictive samples directly from compiled prior semantics or
``SSMPriors`` without tracing back through ``SSMModel.model()``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.posterior_predictive import simulate_posterior_predictive
from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec, assemble_sampled_extra_params
from causal_ssm_agent.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    build_prior_runtime_bundle,
    load_prior_runtime_bundle,
    sample_prior_unconstrained,
)
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily


def _ensure_discrete_metadata(spec: SSMSpec) -> None:
    """Require hydrated level counts before sampling discrete emissions."""
    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
    needs_levels = any(
        dist in (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL)
        for dist in manifest_dists
    )
    if needs_levels and spec.manifest_level_counts is None:
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts."
        )


def _constrain_prior_samples(
    z_samples: jnp.ndarray,
    runtime: PriorRuntimeBundle,
) -> dict[str, jnp.ndarray]:
    """Map unconstrained prior draws to constrained site samples."""
    if runtime.flat_dim == 0:
        return {}

    unconstrained = jax.vmap(runtime.unravel_fn)(z_samples)
    constrained: dict[str, jnp.ndarray] = {}
    for site in runtime.registry:
        constrained[site.name] = jax.vmap(runtime.transforms[site.name])(unconstrained[site.name])
    return constrained


def _assemble_extra_params_batched(
    spec: SSMSpec,
    constrained_samples: dict[str, jnp.ndarray],
    runtime: PriorRuntimeBundle,
    *,
    n_draws: int,
) -> dict[str, jnp.ndarray]:
    """Assemble per-draw observation/process hyperparameters."""
    likelihood_sites = [
        site.name for site in runtime.registry if site.assembly_group == "likelihood"
    ]
    if not likelihood_sites:
        return {}

    def _assemble_one(draw_idx):
        sampled_values = {
            site_name: constrained_samples[site_name][draw_idx]
            for site_name in likelihood_sites
            if site_name in constrained_samples
        }
        return assemble_sampled_extra_params(spec, sampled_values)

    return jax.vmap(_assemble_one)(jnp.arange(n_draws, dtype=jnp.int32))


def sample_prior_predictive_from_runtime(
    spec: SSMSpec,
    runtime: PriorRuntimeBundle,
    times: jnp.ndarray,
    *,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a prepared runtime bundle."""
    _ensure_discrete_metadata(spec)

    z_samples, _rng_key = sample_prior_unconstrained(
        random.PRNGKey(seed),
        runtime.registry,
        runtime.prior_state,
        n_samples=num_samples,
    )
    constrained_samples = _constrain_prior_samples(z_samples, runtime)
    deterministic_samples = assemble_deterministics_from_registry(
        constrained_samples,
        spec,
        runtime.registry,
        n_draws=num_samples,
    )
    extra_params = _assemble_extra_params_batched(
        spec,
        constrained_samples,
        runtime,
        n_draws=num_samples,
    )

    samples: dict[str, jnp.ndarray] = {}
    samples.update(constrained_samples)
    samples.update(deterministic_samples)
    samples.update(extra_params)
    samples["observations"] = simulate_posterior_predictive(
        samples,
        times,
        manifest_dist=spec.manifest_dist,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        n_subsample=num_samples,
        rng_seed=seed,
    )
    return samples


def sample_prior_predictive_from_compiled_semantics(
    spec: SSMSpec,
    compiled_prior_semantics: dict,
    times: jnp.ndarray,
    *,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from serialized compiled semantics."""
    runtime = load_prior_runtime_bundle(compiled_prior_semantics)
    return sample_prior_predictive_from_runtime(
        spec,
        runtime,
        times,
        num_samples=num_samples,
        seed=seed,
    )


def sample_prior_predictive_from_priors(
    spec: SSMSpec,
    priors: SSMPriors | None,
    times: jnp.ndarray,
    *,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from ``SSMPriors`` directly."""
    runtime = build_prior_runtime_bundle(spec, priors)
    return sample_prior_predictive_from_runtime(
        spec,
        runtime,
        times,
        num_samples=num_samples,
        seed=seed,
    )
