"""Leaf observation/process hyperparameter assembly for SSM likelihoods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec


def assemble_sampled_extra_params(
    spec: SSMSpec,
    sampled_values: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Assemble likelihood hyperparameters and derived observation metadata."""
    extra_params: dict[str, jnp.ndarray] = {}
    manifest_dist_set = set(spec.manifest_dists)

    scalar_keys = (
        "obs_df",
        "obs_shape",
        "obs_r",
        "obs_concentration",
        "proc_df",
    )
    for key in scalar_keys:
        if key in sampled_values:
            extra_params[key] = sampled_values[key]

    if spec.manifest_level_counts is None:
        return extra_params

    level_counts_list = list(spec.manifest_level_counts)
    level_counts = jnp.asarray(level_counts_list, dtype=jnp.int32)
    extra_params["obs_level_counts"] = level_counts

    max_levels = max(level_counts_list) if level_counts_list else 0
    max_cutpoints = max(max_levels - 1, 0)

    if any_family_needs_level_metadata(manifest_dist_set) and max_cutpoints <= 0:
        raise ValueError(
            "ordered_logistic/categorical requires manifest_level_counts with at least 2 levels"
        )

    if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set:
        ordered_base = sampled_values["obs_ordered_base"]
        if max_cutpoints > 1:
            ordered_gaps = sampled_values["obs_ordered_gaps"]
        else:
            ordered_gaps = jnp.zeros((spec.n_manifest, 0), dtype=ordered_base.dtype)

        # Cutpoints are NOT centered: the threshold base is the channel-side
        # location parameter, identified against the construct's latent-side
        # location anchor (see docs/reference/statistical-model-spec/identification.md).
        # Centering here would both kill the base (it cancels exactly) and
        # over-anchor constructs whose location is already pinned elsewhere.
        raw_cutpoints = jnp.concatenate(
            [
                ordered_base[:, None],
                ordered_base[:, None] + jnp.cumsum(ordered_gaps, axis=1),
            ],
            axis=1,
        )
        cutpoint_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(
            level_counts[:, None] - 1,
            0,
        )
        extra_params["obs_ordered_cutpoints"] = jnp.where(cutpoint_mask, raw_cutpoints, 0.0)

    if DistributionFamily.CATEGORICAL in manifest_dist_set:
        cat_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(level_counts[:, None] - 1, 0)
        extra_params["obs_cat_intercepts"] = jnp.where(
            cat_mask,
            sampled_values["obs_cat_intercepts"],
            0.0,
        )
        extra_params["obs_cat_slopes"] = jnp.where(
            cat_mask,
            sampled_values["obs_cat_slopes"],
            0.0,
        )

    return extra_params
