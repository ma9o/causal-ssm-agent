"""Focused tests for Stage 4b interval-summary moment propagation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    compile_observation_operator,
)
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction
from causal_ssm_agent.utils.parametric_id import (
    _predict_observation_components,
    _project_response_covariance_blocks,
)
from tests.ssm_test_utils import make_ssm_spec


def _overlapping_interval_mean_support() -> ObservationSupportRuntime:
    """Build one-manifest overlapping interval windows on two reusable slots."""
    interval_prev_coeffs = np.zeros((4, 1, 2), dtype=np.float64)
    interval_weights = np.zeros((4, 1, 2), dtype=np.float64)

    # Row 2 emits mean(r0, r1) from slot 0.
    interval_prev_coeffs[1, 0, 0] = 1.0
    interval_prev_coeffs[2, 0, 0] = 1.0
    interval_weights[1, 0, 0] = 1.0
    interval_weights[2, 0, 0] = 1.0

    # Row 3 emits mean(r1, r2) from slot 1, overlapping the first window.
    interval_prev_coeffs[2, 0, 1] = 1.0
    interval_prev_coeffs[3, 0, 1] = 1.0
    interval_weights[2, 0, 1] = 1.0
    interval_weights[3, 0, 1] = 1.0

    return ObservationSupportRuntime(
        anchor_times=np.array([0.0, 1.0, 2.0, 3.0]),
        manifest_names=["score_mean"],
        support_kinds=["interval"],
        summary_operators=["mean"],
        anchor_policies=["end"],
        observation_windows=["1d"],
        support_start_times=np.full((4, 1), np.nan),
        support_end_times=np.full((4, 1), np.nan),
        interval_prev_coeffs=interval_prev_coeffs,
        interval_curr_coeffs=np.zeros((4, 1, 2), dtype=np.float64),
        interval_weights=interval_weights,
        emission_slot_indices=np.array([[-1], [-1], [0], [1]], dtype=np.int32),
    )


def test_interval_mean_projection_uses_full_window_covariance():
    """Windowed mean covariances should include off-diagonal response terms."""
    response_means = jnp.array([[0.2], [0.4], [0.1], [0.3]], dtype=jnp.float32)
    response_cov = jnp.array(
        [
            [1.0, 0.3, 0.1, 0.0],
            [0.3, 2.0, 0.5, 0.2],
            [0.1, 0.5, 3.0, 0.7],
            [0.0, 0.2, 0.7, 4.0],
        ],
        dtype=jnp.float32,
    )
    response_cov_blocks = response_cov.reshape(4, 1, 4, 1).transpose(0, 2, 1, 3)
    observation_operator = compile_observation_operator(_overlapping_interval_mean_support())

    emitted_means, same_covs, lag1_covs, semantic_mask = _project_response_covariance_blocks(
        response_means,
        response_cov_blocks,
        observation_operator,
    )

    assert semantic_mask[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert float(emitted_means[2, 0]) == pytest.approx((0.2 + 0.4) / 2.0)
    assert float(emitted_means[3, 0]) == pytest.approx((0.4 + 0.1) / 2.0)

    expected_same_row_2 = (1.0 + 2.0 + (2.0 * 0.3)) / 4.0
    expected_same_row_3 = (2.0 + 3.0 + (2.0 * 0.5)) / 4.0
    expected_lag1_row_3 = (0.3 + 2.0 + 0.1 + 0.5) / 4.0

    assert float(same_covs[0, 0, 0]) == pytest.approx(0.0)
    assert float(same_covs[1, 0, 0]) == pytest.approx(0.0)
    assert float(same_covs[2, 0, 0]) == pytest.approx(expected_same_row_2)
    assert float(same_covs[3, 0, 0]) == pytest.approx(expected_same_row_3)
    assert float(lag1_covs[2, 0, 0]) == pytest.approx(expected_lag1_row_3)


def test_predict_observation_components_keeps_point_interval_same_row_covariance():
    """Mixed point/interval rows should retain same-row point-interval covariance."""
    drift = -0.5
    transition = np.exp(drift)
    observation_support = ObservationSupportRuntime(
        anchor_times=np.array([0.0, 1.0, 2.0]),
        manifest_names=["point_score", "window_score"],
        support_kinds=["point", "interval"],
        summary_operators=[None, "mean"],
        anchor_policies=[None, "end"],
        observation_windows=["1d", "1d"],
        support_start_times=np.array(
            [
                [np.nan, np.nan],
                [np.nan, 0.0],
                [np.nan, 1.0],
            ],
        ),
        support_end_times=np.array(
            [
                [np.nan, np.nan],
                [np.nan, 1.0],
                [np.nan, 2.0],
            ],
        ),
        interval_prev_coeffs=np.array(
            [
                [[0.0], [0.0]],
                [[0.0], [0.5]],
                [[0.0], [0.5]],
            ],
        ),
        interval_curr_coeffs=np.array(
            [
                [[0.0], [0.0]],
                [[0.0], [0.5]],
                [[0.0], [0.5]],
            ],
        ),
        interval_weights=np.array(
            [
                [[0.0], [0.0]],
                [[0.0], [1.0]],
                [[0.0], [1.0]],
            ],
        ),
        emission_slot_indices=np.array([[-1, -1], [-1, 0], [-1, 0]], dtype=np.int32),
    )
    spec = make_ssm_spec(
        n_latent=1,
        n_manifest=2,
        drift=jnp.array([[drift]], dtype=jnp.float32),
        diffusion=jnp.zeros((1, 1)),
        lambda_mat=jnp.ones((2, 1)),
        manifest_var=jnp.array([[0.25, 0.05], [0.05, 0.16]], dtype=jnp.float32),
        manifest_means=jnp.zeros(2),
        t0_means=jnp.zeros(1),
        t0_var=jnp.eye(1),
        manifest_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[
            LinkFunction.IDENTITY,
            LinkFunction.IDENTITY,
        ],
    )
    det = {
        "drift": jnp.array([[drift]], dtype=jnp.float32),
        "diffusion": jnp.zeros((1, 1), dtype=jnp.float32),
        "lambda": jnp.ones((2, 1), dtype=jnp.float32),
        "manifest_cov": jnp.array([[0.25, 0.05], [0.05, 0.16]], dtype=jnp.float32),
        "manifest_means": jnp.zeros(2, dtype=jnp.float32),
        "t0_means": jnp.zeros(1, dtype=jnp.float32),
        "t0_cov": jnp.eye(1, dtype=jnp.float32),
        "cint": jnp.zeros(1, dtype=jnp.float32),
    }

    emitted_means, same_covs, lag1_covs, obs_noise_sd, semantic_mask = (
        _predict_observation_components(
            det,
            {},
            spec,
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
            observation_support=observation_support,
        )
    )

    assert emitted_means.shape == (3, 2)
    assert semantic_mask.tolist() == [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    expected_row1_point_interval = transition * (1.0 + transition) / 2.0 + 0.05
    expected_row2_point_interval = transition**3 * (1.0 + transition) / 2.0 + 0.05
    expected_lag1_interval_interval = transition * (1.0 + transition) ** 2 / 4.0
    assert float(same_covs[1, 0, 1]) == pytest.approx(expected_row1_point_interval)
    assert float(same_covs[2, 0, 1]) == pytest.approx(expected_row2_point_interval)
    assert float(lag1_covs[1, 1, 1]) == pytest.approx(expected_lag1_interval_interval)
    assert float(obs_noise_sd[1, 1]) == pytest.approx(np.sqrt(0.16))
