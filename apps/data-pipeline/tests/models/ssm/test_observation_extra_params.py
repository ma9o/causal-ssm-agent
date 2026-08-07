"""Tests for explicit observation-channel metadata slicing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.models.ssm.execution.observation_extra_params import (
    slice_observation_extra_params,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams


def test_slice_observation_extra_params_preserves_none_and_site_wide_values() -> None:
    assert slice_observation_extra_params(None, [0], source_channel_count=1) is None

    scalar_array = jnp.asarray(7.0)
    params: LikelihoodExtraParams = {
        "obs_df": scalar_array,
        "obs_shape": 2.0,
        "custom_matrix": jnp.arange(6).reshape(3, 2),
    }
    sliced = slice_observation_extra_params(params, [2, 0], source_channel_count=3)

    assert sliced is not None
    assert sliced["obs_df"] is scalar_array
    assert sliced["obs_shape"] == 2.0
    assert sliced["custom_matrix"] is params["custom_matrix"]


def test_slice_observation_extra_params_slices_declared_vectors_and_matrices() -> None:
    params: LikelihoodExtraParams = {
        "obs_level_counts": jnp.asarray([2, 3, 4]),
        "obs_ordered_cutpoints": jnp.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    }

    sliced = slice_observation_extra_params(params, [2, 0], source_channel_count=3)

    assert sliced is not None
    np.testing.assert_array_equal(sliced["obs_level_counts"], [4, 2])
    np.testing.assert_array_equal(
        sliced["obs_ordered_cutpoints"],
        [[4.0, 5.0, 6.0], [1.0, 0.0, 0.0]],
    )


def test_slice_observation_extra_params_reorders_a_full_current_layout() -> None:
    params: LikelihoodExtraParams = {"obs_cat_slopes": jnp.asarray([[10.0], [20.0]])}

    sliced = slice_observation_extra_params(params, [1, 0], source_channel_count=2)

    assert sliced is not None
    np.testing.assert_array_equal(sliced["obs_cat_slopes"], [[20.0], [10.0]])


@pytest.mark.parametrize(
    "value",
    [jnp.asarray(1), jnp.asarray([2, 3]), jnp.ones((4, 2))],
)
def test_slice_observation_extra_params_rejects_malformed_channel_metadata(
    value: jnp.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="obs_level_counts must have leading dimension 3",
    ):
        slice_observation_extra_params(
            {"obs_level_counts": value},
            [0],
            source_channel_count=3,
        )


@pytest.mark.parametrize("indices", [[-1], [3], [0, 4]])
def test_slice_observation_extra_params_rejects_out_of_range_indices(
    indices: list[int],
) -> None:
    with pytest.raises(ValueError, match="channel_indices must be within"):
        slice_observation_extra_params(
            {"obs_level_counts": jnp.asarray([2, 3, 4])},
            indices,
            source_channel_count=3,
        )
