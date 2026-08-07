"""Helpers for reading likelihood and diffusion metadata from ``SSMSpec``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily, LinkFunction
    from nof1_causal_lab.models.ssm.model import SSMSpec


def get_per_variable_diffusion(spec: SSMSpec) -> list[DistributionFamily]:
    """Return the canonical per-variable diffusion noise families."""
    return list(spec.diffusion_dists)


def has_student_t_diffusion(spec: SSMSpec) -> bool:
    """Return whether any latent process uses Student-t diffusion noise."""
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily

    return DistributionFamily.STUDENT_T in set(get_per_variable_diffusion(spec))


def get_per_channel_manifest(spec: SSMSpec) -> list[DistributionFamily]:
    """Return the canonical per-channel observation noise families."""
    return list(spec.manifest_dists)


def get_per_channel_links(spec: SSMSpec) -> list[LinkFunction]:
    """Resolve per-channel link functions."""
    from nof1_causal_lab.models.ssm.execution.observation_families import (
        resolve_manifest_families_and_links,
    )

    if spec.manifest_links is not None:
        return list(spec.manifest_links)
    _, links = resolve_manifest_families_and_links(get_per_channel_manifest(spec))
    return links
