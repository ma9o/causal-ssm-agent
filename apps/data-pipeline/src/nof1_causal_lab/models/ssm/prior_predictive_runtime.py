"""Compile-stable prior predictive runtime.

Builds prior predictive samples directly from compiled prior semantics or
``SSMPriors`` without tracing back through ``SSMModel.model()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.predictive_simulation import simulate_predictive_observations
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_prior_runtime_bundle,
    load_prior_runtime_bundle,
    sample_prior_unconstrained,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMPriors, SSMSpec


def _ensure_discrete_metadata(spec: SSMSpec) -> None:
    """Require hydrated level counts before sampling discrete emissions."""
    needs_levels = any_family_needs_level_metadata(spec.manifest_dists)
    if needs_levels and spec.manifest_level_counts is None:
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts."
        )


def _assemble_extra_params_batched(
    spec: SSMSpec,
    constrained_samples: dict[str, jnp.ndarray],
    runtime: PriorRuntimeBundle,
    *,
    n_draws: int,
) -> dict[str, jnp.ndarray]:
    """Assemble per-draw observation/process hyperparameters."""
    if not any(site.assembly_group == "likelihood" for site in runtime.registry):
        return {}

    def _assemble_one(draw_idx):
        sampled_values = {
            site_name: values[draw_idx] for site_name, values in constrained_samples.items()
        }
        return assemble_extra_params_from_registry(spec, sampled_values, runtime.registry)

    return jax.vmap(_assemble_one)(jnp.arange(n_draws, dtype=jnp.int64))


def sample_prior_predictive_from_runtime(
    spec: SSMSpec,
    runtime: PriorRuntimeBundle,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
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
    constrained_samples = runtime.constrain_batched(z_samples)
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
    observations, observations_mask = simulate_predictive_observations(
        samples,
        times,
        diffusion_dists=spec.diffusion_dists,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        n_subsample=num_samples,
        rng_seed=seed,
        manifest_names=list(spec.manifest_names) if spec.manifest_names is not None else None,
    )
    samples["observations"] = observations
    samples["observations_mask"] = observations_mask
    return samples


def sample_prior_predictive_from_compiled_semantics(
    spec: SSMSpec,
    compiled_prior_semantics: dict,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from serialized compiled semantics."""
    runtime = load_prior_runtime_bundle(compiled_prior_semantics)
    return sample_prior_predictive_from_runtime(
        spec,
        runtime,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        num_samples=num_samples,
        seed=seed,
    )


def sample_prior_predictive_from_priors(
    spec: SSMSpec,
    priors: SSMPriors | None,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from ``SSMPriors`` directly."""
    runtime = build_prior_runtime_bundle(spec, priors)
    return sample_prior_predictive_from_runtime(
        spec,
        runtime,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        num_samples=num_samples,
        seed=seed,
    )
