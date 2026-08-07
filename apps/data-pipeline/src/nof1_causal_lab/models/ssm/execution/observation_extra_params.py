"""Observation-channel metadata operations shared by likelihood consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams

PER_CHANNEL_OBSERVATION_EXTRA_PARAM_KEYS = frozenset(
    {
        "obs_level_counts",
        "obs_ordered_cutpoints",
        "obs_cat_intercepts",
        "obs_cat_slopes",
    }
)


def slice_observation_extra_params(
    extra_params: LikelihoodExtraParams | None,
    channel_indices: list[int],
    *,
    source_channel_count: int,
) -> LikelihoodExtraParams | None:
    """Slice declared per-channel metadata against a known source layout.

    Scalar and site-wide parameters are preserved verbatim. Per-channel
    parameters must describe the complete source layout; this prevents JAX's
    out-of-bounds gather semantics from silently selecting the final channel.
    """
    if extra_params is None:
        return None
    if source_channel_count < 0:
        raise ValueError("source_channel_count must be non-negative")
    if any(index < 0 or index >= source_channel_count for index in channel_indices):
        raise ValueError(
            "channel_indices must be within the source observation layout: "
            f"indices={channel_indices}, source_channel_count={source_channel_count}"
        )

    index_array = jnp.asarray(channel_indices, dtype=jnp.int32)
    sliced = dict(extra_params)
    for key in PER_CHANNEL_OBSERVATION_EXTRA_PARAM_KEYS.intersection(extra_params):
        value = jnp.asarray(extra_params[key])
        if value.ndim == 0 or value.shape[0] != source_channel_count:
            raise ValueError(
                f"{key} must have leading dimension {source_channel_count}; got shape {value.shape}"
            )
        sliced[key] = value[index_array]
    return sliced
