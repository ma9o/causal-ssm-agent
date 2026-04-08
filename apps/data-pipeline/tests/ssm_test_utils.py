"""Test helpers for constructing explicit SSMSpec instances."""

from __future__ import annotations

from typing import Any

from causal_ssm_agent.models.ssm.model import SSMSpec, full_drift_mask, zero_loading_mask


def make_ssm_spec(**kwargs: Any) -> SSMSpec:
    """Build an SSMSpec with explicit default structural masks."""
    n_latent = kwargs["n_latent"]
    n_manifest = kwargs["n_manifest"]
    kwargs.setdefault("drift_mask", full_drift_mask(n_latent))
    kwargs.setdefault("lambda_mask", zero_loading_mask(n_manifest, n_latent))
    return SSMSpec(**kwargs)
