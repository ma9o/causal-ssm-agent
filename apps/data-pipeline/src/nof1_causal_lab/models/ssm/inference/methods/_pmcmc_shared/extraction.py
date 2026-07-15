"""Posterior sample extraction shared by particle MCMC methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples
from nof1_causal_lab.models.ssm.inference.utils import extract_constrained_samples

if TYPE_CHECKING:
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.inference.bundle import ParticleRuntimeBundle


def build_pmcmc_mcmc_result(
    *,
    chain_samples: dict[str, jnp.ndarray],
    chain_extra_fields: dict[str, jnp.ndarray],
    num_chains: int,
    num_samples: int,
    backend: str,
) -> TrajectoryMCMCResult:
    return TrajectoryMCMCResult(
        chain_samples=chain_samples,
        chain_extra_fields=chain_extra_fields,
        num_chains=num_chains,
        num_samples=num_samples,
        backend=backend,
    )


def extract_grouped_public_samples(
    grouped_positions: jnp.ndarray,
    *,
    bundle: ParticleRuntimeBundle,
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    num_chains: int,
    num_samples: int,
    reparam,
) -> dict[str, jnp.ndarray]:
    flat_positions = grouped_positions.reshape((-1, bundle.cached.dim))
    constrained_samples = extract_constrained_samples(
        flat_positions,
        bundle.cached.site_info,
        bundle.cached.unravel_fn,
        model.spec,
        reparam=reparam,
        model=model,
        observations=observations,
        times=times,
    )
    public_samples = _filter_public_samples(constrained_samples, bundle.cached.public_sites)
    return {
        name: values.reshape((num_chains, num_samples, *values.shape[1:]))
        for name, values in public_samples.items()
    }
