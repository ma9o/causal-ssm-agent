"""Leaf factory for concrete likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm_observation_metadata import ObservationSupportRuntime


def build_laplace_backend(
    spec: SSMSpec,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import (
        get_per_channel_links,
        get_per_channel_manifest,
    )
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood

    return LaplaceLikelihood(
        n_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        manifest_dists=get_per_channel_manifest(spec),
        manifest_links=get_per_channel_links(spec),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )


def make_likelihood_backend(
    spec: SSMSpec,
    likelihood: Literal["particle", "kalman"] = "particle",
    n_particles: int = 200,
    pf_key: jnp.ndarray | None = None,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a likelihood backend from model configuration."""
    from nof1_causal_lab.models.ssm.inference.structure import plan_inference_structure

    if pf_key is None:
        pf_key = jax.random.PRNGKey(0)

    inference_structure = plan_inference_structure(
        spec,
        likelihood=likelihood,
        observation_support=observation_support,
    )

    if inference_structure.structural_backend == "kalman":
        from nof1_causal_lab.models.ssm.inference.targets.kalman import KalmanLikelihood

        return KalmanLikelihood(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
        )

    from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import (
        get_per_channel_links,
        get_per_channel_manifest,
        get_per_variable_diffusion,
    )

    per_var = list(get_per_variable_diffusion(spec))
    per_obs = list(get_per_channel_manifest(spec))
    per_links = list(get_per_channel_links(spec))

    if inference_structure.structural_backend == "composed":
        from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
        from nof1_causal_lab.models.ssm.inference.targets.kalman import KalmanLikelihood
        from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood

        partition = inference_structure.first_pass_partition
        if partition is None:
            raise ValueError("Composed likelihood path requires an active first-pass partition")

        n_k = len(partition.kalman_idx)
        n_obs_k = len(partition.obs_kalman_idx)
        n_p = len(partition.particle_idx)
        n_obs_p = len(partition.obs_particle_idx)

        particle_diffs: list[DistributionFamily | str] = [
            per_var[int(i)] for i in partition.particle_idx
        ]
        particle_obs_dists: list[DistributionFamily | str] = [
            per_obs[int(k)] for k in partition.obs_particle_idx
        ]
        particle_obs_links: list[LinkFunction | str | None] = [
            per_links[int(k)] for k in partition.obs_particle_idx
        ]

        return ComposedLikelihood(
            partition=partition,
            kalman_backend=KalmanLikelihood(
                n_latent=n_k,
                n_manifest=n_obs_k,
            ),
            particle_backend=ParticleLikelihood(
                n_latent=n_p,
                n_manifest=n_obs_p,
                n_particles=n_particles,
                rng_key=pf_key,
                manifest_dists=particle_obs_dists,
                diffusion_dists=particle_diffs,
                block_rb=spec.second_pass_rb,
                manifest_links=particle_obs_links,
                observation_support=None,
            ),
        )

    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood

    return ParticleLikelihood(
        n_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        n_particles=n_particles,
        rng_key=pf_key,
        manifest_dists=per_obs,
        diffusion_dists=per_var,
        block_rb=False
        if observation_support is not None
        and observation_support.requires_interval_summary_handling
        else spec.second_pass_rb,
        manifest_links=per_links,
        observation_support=observation_support,
    )
