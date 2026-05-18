"""Tests for ComposedLikelihood two-level Kalman+PF backend.

Uses mock backends to verify parameter extraction and sub-LL summation
without requiring actual Kalman/PF computation.
"""

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import RBPartition

# =============================================================================
# Mock backends that record their inputs
# =============================================================================


class RecordingBackend:
    """Mock backend that records the arguments it receives and returns a known LL."""

    def __init__(self, return_val):
        self.calls = []
        self.return_val = return_val

    def compute_log_likelihood(
        self,
        ct_params,
        measurement_params,
        initial_state,
        observations,
        time_intervals,
        obs_mask=None,
        extra_params=None,
        transition_inputs=None,
    ):
        self.calls.append(
            {
                "ct_params": ct_params,
                "measurement_params": measurement_params,
                "initial_state": initial_state,
                "observations": observations,
                "time_intervals": time_intervals,
                "obs_mask": obs_mask,
                "extra_params": extra_params,
                "transition_inputs": transition_inputs,
            }
        )
        T = observations.shape[0]
        return jnp.full(T, self.return_val)


# =============================================================================
# Shared fixtures
# =============================================================================


def _make_full_model(n_latent=4, n_manifest=4, T=10):
    """Create full-size model parameters for testing."""
    A = jnp.eye(n_latent) * -0.5 + jnp.ones((n_latent, n_latent)) * 0.01
    Q = jnp.eye(n_latent) * 0.1
    c = jnp.zeros(n_latent)
    H = (
        jnp.eye(n_manifest, n_latent)
        if n_manifest == n_latent
        else jnp.ones((n_manifest, n_latent))
    )
    d = jnp.zeros(n_manifest)
    R = jnp.eye(n_manifest) * 0.5
    m0 = jnp.zeros(n_latent)
    P0 = jnp.eye(n_latent)

    ct = CTParams(drift=A, diffusion_cov=Q, cint=c)
    meas = MeasurementParams(lambda_mat=H, manifest_means=d, manifest_cov=R)
    init = InitialStateParams(mean=m0, cov=P0)
    obs = jnp.ones((T, n_manifest))
    dt = jnp.ones(T) * 0.1
    mask = jnp.ones((T, n_manifest), dtype=bool)

    return ct, meas, init, obs, dt, mask


def _make_partition_2_2():
    """Partition: latents {0,1} → Kalman, {2,3} → PF. Obs follows same split."""
    return RBPartition(
        kalman_idx=np.array([0, 1], dtype=np.int64),
        particle_idx=np.array([2, 3], dtype=np.int64),
        obs_kalman_idx=np.array([0, 1], dtype=np.int64),
        obs_particle_idx=np.array([2, 3], dtype=np.int64),
    )


# =============================================================================
# Tests
# =============================================================================


class TestComposedLikelihood:
    def test_sub_ll_summation(self):
        """Total LL should be element-wise sum of sub-LLs."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=1.0)
        particle = RecordingBackend(return_val=2.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        result = composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        assert result.shape == (10,)
        assert jnp.allclose(result, 3.0)  # 1.0 + 2.0

    def test_drift_subblock_extraction(self):
        """Kalman backend should receive drift[0:2, 0:2], PF should receive drift[2:4, 2:4]."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        # Kalman should get 2x2 sub-blocks for latents {0,1}
        k_ct = kalman.calls[0]["ct_params"]
        assert k_ct.drift.shape == (2, 2)
        assert k_ct.diffusion_cov.shape == (2, 2)
        assert k_ct.cint.shape == (2,)

        # PF should get 2x2 sub-blocks for latents {2,3}
        p_ct = particle.calls[0]["ct_params"]
        assert p_ct.drift.shape == (2, 2)
        assert p_ct.diffusion_cov.shape == (2, 2)

    def test_measurement_subblock_extraction(self):
        """Measurement params should be sliced by obs and latent partition indices."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        # Kalman: H[obs_kalman, kalman] = H[0:2, 0:2]
        k_meas = kalman.calls[0]["measurement_params"]
        assert k_meas.lambda_mat.shape == (2, 2)
        assert k_meas.manifest_means.shape == (2,)
        assert k_meas.manifest_cov.shape == (2, 2)

        # PF: H[obs_particle, particle] = H[2:4, 2:4]
        p_meas = particle.calls[0]["measurement_params"]
        assert p_meas.lambda_mat.shape == (2, 2)

    def test_initial_state_extraction(self):
        """Initial state should be split by latent partition."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        k_init = kalman.calls[0]["initial_state"]
        assert k_init.mean.shape == (2,)
        assert k_init.cov.shape == (2, 2)

        p_init = particle.calls[0]["initial_state"]
        assert p_init.mean.shape == (2,)
        assert p_init.cov.shape == (2, 2)

    def test_observations_split(self):
        """Observations should be split by obs partition indices."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        assert kalman.calls[0]["observations"].shape == (10, 2)
        assert particle.calls[0]["observations"].shape == (10, 2)

    def test_obs_mask_split(self):
        """obs_mask should be split the same way as observations."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        ct, meas, init, obs, dt, mask = _make_full_model()
        # Create a non-trivial mask
        mask = mask.at[:, 1].set(False)
        composed.compute_log_likelihood(ct, meas, init, obs, dt, obs_mask=mask)

        k_mask = kalman.calls[0]["obs_mask"]
        assert k_mask.shape == (10, 2)
        # Column 1 of Kalman's obs_mask (which is obs channel 1) should be False
        assert not bool(k_mask[0, 1])

    def test_correct_subblock_values(self):
        """Verify exact values of extracted sub-blocks, not just shapes."""
        partition = _make_partition_2_2()
        kalman = RecordingBackend(return_val=0.0)
        particle = RecordingBackend(return_val=0.0)
        composed = ComposedLikelihood(partition, kalman, particle)

        # Create distinguishable drift matrix
        A = jnp.arange(16).reshape(4, 4).astype(float)
        ct = CTParams(drift=A, diffusion_cov=jnp.eye(4), cint=jnp.arange(4.0))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(4), manifest_means=jnp.zeros(4), manifest_cov=jnp.eye(4)
        )
        init = InitialStateParams(mean=jnp.zeros(4), cov=jnp.eye(4))
        obs = jnp.ones((5, 4))
        dt = jnp.ones(5)

        composed.compute_log_likelihood(ct, meas, init, obs, dt)

        # Kalman block: latents {0,1} → A[0:2, 0:2]
        k_drift = kalman.calls[0]["ct_params"].drift
        expected_k = jnp.array([[0, 1], [4, 5]], dtype=float)
        assert jnp.allclose(k_drift, expected_k)

        # Particle block: latents {2,3} → A[2:4, 2:4]
        p_drift = particle.calls[0]["ct_params"].drift
        expected_p = jnp.array([[10, 11], [14, 15]], dtype=float)
        assert jnp.allclose(p_drift, expected_p)

        # cint split
        k_cint = kalman.calls[0]["ct_params"].cint
        assert jnp.allclose(k_cint, jnp.array([0.0, 1.0]))
        p_cint = particle.calls[0]["ct_params"].cint
        assert jnp.allclose(p_cint, jnp.array([2.0, 3.0]))
