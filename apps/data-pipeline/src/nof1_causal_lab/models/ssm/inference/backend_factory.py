"""Leaf factory for concrete marginal likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime


def build_laplace_backend(
    spec: SSMSpec,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood
    from nof1_causal_lab.models.ssm.inference.targets.spec_metadata import (
        get_per_channel_links,
        get_per_channel_manifest,
    )

    return LaplaceLikelihood(
        n_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        manifest_dists=get_per_channel_manifest(spec),
        manifest_links=get_per_channel_links(spec),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )
