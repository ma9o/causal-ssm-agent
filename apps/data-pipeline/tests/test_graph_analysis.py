"""Tests for first-pass Rao-Blackwellization graph analysis.

Covers: drift sparsity, observation dependency, per-variable distribution
resolution, and the main analyze_first_pass_rb decomposition logic.
"""

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.distributions import DistributionFamily
from causal_ssm_agent.models.ssm.inference import select_default_method
from causal_ssm_agent.models.ssm.inference.structure import plan_inference_structure
from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import (
    RBPartition,
    analyze_first_pass_rb,
    compute_drift_sparsity,
    compute_obs_dependency,
    get_per_channel_links,
    get_per_channel_manifest,
    get_per_variable_diffusion,
    kalman_block_profile_indices,
)
from causal_ssm_agent.models.ssm.model import SSMSpec, full_diagonal_mask
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.ssm_test_utils import combined_drift_mask, make_ssm_spec

# =============================================================================
# Helper
# =============================================================================


def _make_spec(**kwargs) -> SSMSpec:
    """Create an SSMSpec with defaults for testing."""
    defaults = {
        "n_latent": 2,
        "n_manifest": 2,
        "lambda_mat": jnp.eye(2),
    }
    defaults.update(kwargs)
    return make_ssm_spec(**defaults)


# =============================================================================
# get_per_variable_diffusion
# =============================================================================


class TestGetPerVariableDiffusion:
    def test_defaults_to_gaussian_per_latent(self):
        """Missing diffusion_dists defaults to Gaussian for every latent."""
        spec = _make_spec(n_latent=3, n_manifest=3, lambda_mat=jnp.eye(3))
        result = get_per_variable_diffusion(spec)
        assert result == [DistributionFamily.GAUSSIAN] * 3

    def test_per_variable_override(self):
        """Authored diffusion_dists are preserved."""
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
                DistributionFamily.GAUSSIAN,
            ],
        )
        result = get_per_variable_diffusion(spec)
        assert result[1] == DistributionFamily.STUDENT_T


# =============================================================================
# get_per_channel_manifest / links
# =============================================================================


class TestGetPerChannelManifest:
    def test_scalar_broadcast(self):
        spec = _make_spec()
        result = get_per_channel_manifest(spec)
        assert result == [DistributionFamily.GAUSSIAN] * 2

    def test_per_channel_override(self):
        spec = _make_spec(manifest_dists=[DistributionFamily.POISSON, DistributionFamily.GAUSSIAN])
        result = get_per_channel_manifest(spec)
        assert result[0] == DistributionFamily.POISSON
        assert result[1] == DistributionFamily.GAUSSIAN


class TestGetPerChannelLinks:
    def test_scalar_broadcast(self):
        spec = _make_spec()
        result = get_per_channel_links(spec)
        assert result == [LinkFunction.IDENTITY] * 2

    def test_per_channel_override(self):
        spec = _make_spec(manifest_links=[LinkFunction.LOG, LinkFunction.IDENTITY])
        result = get_per_channel_links(spec)
        assert result[0] == LinkFunction.LOG


# =============================================================================
# compute_drift_sparsity
# =============================================================================


class TestComputeDriftSparsity:
    def test_free_drift_all_nonzero(self):
        """Free drift → all entries could be nonzero."""
        spec = _make_spec()
        mask = compute_drift_sparsity(SSMStructureRuntime(spec))
        assert mask.shape == (2, 2)
        assert mask.all()

    def test_drift_mask_used_directly(self):
        """drift_mask is used directly when set."""
        dm = np.array([[True, False], [True, True]])
        spec = _make_spec(drift_mask=dm)
        mask = compute_drift_sparsity(SSMStructureRuntime(spec))
        np.testing.assert_array_equal(mask, combined_drift_mask(spec))

    def test_fixed_drift_sparsity(self):
        """Fixed drift matrix: nonzero entries detected."""
        A = jnp.array([[0.5, 0.0], [0.3, -0.2]])
        spec = _make_spec(drift=A)
        mask = compute_drift_sparsity(SSMStructureRuntime(spec))
        expected = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(mask, expected)


# =============================================================================
# compute_obs_dependency
# =============================================================================


class TestComputeObsDependency:
    def test_loading_mask_all_deps(self):
        """Dense loading mask marks every observation channel as dependent."""
        spec = _make_spec(
            lambda_mat=jnp.zeros((2, 2)),
            lambda_mask=np.ones((2, 2), dtype=bool),
        )
        dep = compute_obs_dependency(SSMStructureRuntime(spec))
        assert dep.shape == (2, 2)
        assert dep.all()

    def test_fixed_lambda_sparsity(self):
        """Fixed lambda: detect nonzero entries."""
        H = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        spec = _make_spec(n_manifest=3, lambda_mat=H)
        dep = compute_obs_dependency(SSMStructureRuntime(spec))
        expected = np.array([[True, False], [False, True], [True, True]])
        np.testing.assert_array_equal(dep, expected)

    def test_lambda_mask_union_with_fixed(self):
        """lambda_mask adds free positions to fixed nonzeros."""
        H = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        lm = np.array([[False, True], [True, False]])
        spec = _make_spec(lambda_mat=H, lambda_mask=lm)
        dep = compute_obs_dependency(SSMStructureRuntime(spec))
        # Fixed: (0,0)=True; Mask: (0,1)=True, (1,0)=True
        expected = np.array([[True, True], [True, False]])
        np.testing.assert_array_equal(dep, expected)


# =============================================================================
# analyze_first_pass_rb — the core partition algorithm
# =============================================================================


class TestAnalyzeFirstPassRB:
    def test_all_gaussian_diagonal(self):
        """Fully Gaussian diagonal model → all variables go to Kalman."""
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
            drift_mask=np.eye(3, dtype=bool),  # diagonal → no cross-coupling
        )
        partition = analyze_first_pass_rb(spec)

        assert partition.has_kalman_block
        assert not partition.has_particle_block
        np.testing.assert_array_equal(partition.kalman_idx, [0, 1, 2])
        assert len(partition.particle_idx) == 0
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0, 1, 2])
        assert len(partition.obs_particle_idx) == 0

    def test_all_nongaussian(self):
        """All non-Gaussian diffusion → everything goes to PF."""
        spec = _make_spec(
            diffusion_dists=[DistributionFamily.STUDENT_T, DistributionFamily.STUDENT_T]
        )
        partition = analyze_first_pass_rb(spec)

        assert not partition.has_kalman_block
        assert partition.has_particle_block
        assert len(partition.kalman_idx) == 0
        np.testing.assert_array_equal(partition.particle_idx, [0, 1])

    def test_mixed_gaussian_nongaussian_decoupled(self):
        """Mixed model with diagonal drift: Gaussian block goes to Kalman."""
        # Latent 0: Gaussian diffusion, Latent 1: Student-t diffusion
        # Diagonal drift → no coupling
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),  # diagonal, no cross-coupling
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        partition = analyze_first_pass_rb(spec)

        np.testing.assert_array_equal(partition.kalman_idx, [0])
        np.testing.assert_array_equal(partition.particle_idx, [1])
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0])
        np.testing.assert_array_equal(partition.obs_particle_idx, [1])

    def test_coupled_prevents_kalman(self):
        """Gaussian variable coupled to non-Gaussian via drift → goes to PF."""
        # Latent 0: Gaussian, Latent 1: Student-t, but drift couples them
        spec = _make_spec(
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
            drift_mask=np.ones((2, 2), dtype=bool),  # fully coupled
        )
        partition = analyze_first_pass_rb(spec)

        # Both go to PF because the Gaussian variable is coupled to non-Gaussian
        assert not partition.has_kalman_block
        np.testing.assert_array_equal(partition.particle_idx, [0, 1])

    def test_shared_obs_goes_to_particle(self):
        """Obs channel depending on both Kalman and PF latents → goes to PF."""
        # 3 latents: 0,1 Gaussian decoupled, 2 Student-t
        # Obs 2 depends on latent 0 and 2 → mixed dependency
        H = jnp.array(
            [
                [1.0, 0.0, 0.0],  # obs 0 → latent 0 only (Kalman)
                [0.0, 1.0, 0.0],  # obs 1 → latent 1 only (Kalman)
                [0.5, 0.0, 0.5],  # obs 2 → latent 0 + 2 (mixed → PF)
            ]
        )
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=H,
            drift_mask=np.eye(3, dtype=bool),  # diagonal
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
            ],
        )
        partition = analyze_first_pass_rb(spec)

        np.testing.assert_array_equal(partition.kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.particle_idx, [2])
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.obs_particle_idx, [2])


class TestKalmanBlockProfileIndices:
    def test_includes_initial_state_lower_triangle_when_t0_is_free(self):
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
            drift=jnp.diag(jnp.array([-0.5, -0.5, -0.5])),
            diffusion=jnp.eye(3),
            diffusion_mask=np.diag(full_diagonal_mask(3)),
        )
        partition = RBPartition(
            kalman_idx=np.array([0, 2]),
            particle_idx=np.array([1]),
            obs_kalman_idx=np.array([0, 2]),
            obs_particle_idx=np.array([1]),
        )

        indices = kalman_block_profile_indices(
            partition, structure_runtime=SSMStructureRuntime(spec)
        )

        assert 13 in indices

    def test_respects_sparse_initial_state_correlation_mask(self):
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
            drift=jnp.diag(jnp.array([-0.5, -0.5, -0.5])),
            diffusion=jnp.eye(3),
            diffusion_mask=np.diag(full_diagonal_mask(3)),
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True
        spec.t0_correlation_mask = mask
        partition = RBPartition(
            kalman_idx=np.array([0, 2]),
            particle_idx=np.array([1]),
            obs_kalman_idx=np.array([0, 2]),
            obs_particle_idx=np.array([1]),
        )

        indices = kalman_block_profile_indices(
            partition, structure_runtime=SSMStructureRuntime(spec)
        )

        assert indices.count(12) == 1
        assert 13 not in indices
        assert 14 not in indices

    def test_respects_sparse_manifest_variance_mask(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift=jnp.diag(jnp.array([-0.5, -0.5])),
            diffusion=jnp.eye(2),
            diffusion_mask=np.diag(full_diagonal_mask(2)),
            manifest_var=jnp.diag(jnp.array([0.3, 0.0])),
            manifest_var_mask=np.array([False, True]),
            t0_means=jnp.zeros(2),
            t0_var=jnp.eye(2),
        )
        partition = RBPartition(
            kalman_idx=np.array([1]),
            particle_idx=np.array([0]),
            obs_kalman_idx=np.array([1]),
            obs_particle_idx=np.array([0]),
        )

        indices = kalman_block_profile_indices(
            partition, structure_runtime=SSMStructureRuntime(spec)
        )

        assert 1 in indices
        assert 2 in indices

    def test_nongaussian_obs_prevents_kalman(self):
        """Gaussian diffusion but Poisson observation → variable goes to PF."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
            manifest_dists=[DistributionFamily.POISSON, DistributionFamily.GAUSSIAN],
        )
        partition = analyze_first_pass_rb(spec)

        # Latent 0 has Gaussian diffusion but Poisson observation → PF
        # Latent 1 has Gaussian diffusion + Gaussian obs → Kalman
        np.testing.assert_array_equal(partition.kalman_idx, [1])
        np.testing.assert_array_equal(partition.particle_idx, [0])

    def test_partition_properties(self):
        """has_kalman_block and has_particle_block properties work correctly."""
        p_full_kalman = RBPartition(
            kalman_idx=np.array([0, 1]),
            particle_idx=np.array([], dtype=np.int64),
            obs_kalman_idx=np.array([0, 1]),
            obs_particle_idx=np.array([], dtype=np.int64),
        )
        assert p_full_kalman.has_kalman_block
        assert not p_full_kalman.has_particle_block

        p_full_pf = RBPartition(
            kalman_idx=np.array([], dtype=np.int64),
            particle_idx=np.array([0, 1]),
            obs_kalman_idx=np.array([], dtype=np.int64),
            obs_particle_idx=np.array([0, 1]),
        )
        assert not p_full_pf.has_kalman_block
        assert p_full_pf.has_particle_block

    def test_has_particle_block_obs_only(self):
        """has_particle_block is True when obs_particle_idx is non-empty, even if particle_idx is empty."""
        p = RBPartition(
            kalman_idx=np.array([0, 1]),
            particle_idx=np.array([], dtype=np.int64),
            obs_kalman_idx=np.array([0]),
            obs_particle_idx=np.array([1]),
        )
        assert p.has_kalman_block
        assert p.has_particle_block

    def test_zero_dep_gaussian_channel_goes_to_kalman(self):
        """Zero-dependency Gaussian+identity obs channel is assigned to Kalman."""
        # 2 latents, 3 obs: obs 2 has zero loadings (no deps)
        H = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],  # zero-dep channel
            ]
        )
        spec = _make_spec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=H,
            drift_mask=np.eye(2, dtype=bool),
        )
        partition = analyze_first_pass_rb(spec)

        np.testing.assert_array_equal(partition.kalman_idx, [0, 1])
        assert len(partition.particle_idx) == 0
        # All three obs channels should be Kalman (including the zero-dep one)
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0, 1, 2])
        assert len(partition.obs_particle_idx) == 0
        assert not partition.has_particle_block

    def test_zero_dep_nongaussian_channel_goes_to_particle(self):
        """Zero-dependency non-Gaussian obs channel is assigned to particle."""
        # 2 latents, 3 obs: obs 2 has zero loadings but Poisson noise
        H = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],  # zero-dep channel
            ]
        )
        spec = _make_spec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=H,
            drift_mask=np.eye(2, dtype=bool),
            manifest_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.POISSON,
            ],
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY, LinkFunction.LOG],
        )
        partition = analyze_first_pass_rb(spec)

        np.testing.assert_array_equal(partition.kalman_idx, [0, 1])
        assert len(partition.particle_idx) == 0
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.obs_particle_idx, [2])
        # has_particle_block should be True due to the non-Kalman obs channel
        assert partition.has_particle_block

    def test_free_drift_without_mask_couples_all(self):
        """Free drift without mask → full coupling → no Kalman block if any non-Gaussian."""
        spec = _make_spec(
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
            # No drift_mask → all entries nonzero → full coupling
        )
        partition = analyze_first_pass_rb(spec)

        # Full coupling with one non-Gaussian → everything to PF
        assert not partition.has_kalman_block

    def test_three_block_partition(self):
        """Three decoupled blocks: two Gaussian (Kalman), one Student-t (PF)."""
        # Block structure via drift_mask: variables {0,1}, {2,3}, {4}
        dm = np.zeros((5, 5), dtype=bool)
        dm[0, 0] = dm[0, 1] = dm[1, 0] = dm[1, 1] = True  # block {0,1}
        dm[2, 2] = dm[2, 3] = dm[3, 2] = dm[3, 3] = True  # block {2,3}
        dm[4, 4] = True  # block {4}

        spec = _make_spec(
            n_latent=5,
            n_manifest=5,
            lambda_mat=jnp.eye(5),
            drift_mask=dm,
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,  # 0 → Kalman
                DistributionFamily.GAUSSIAN,  # 1 → Kalman (same block as 0)
                DistributionFamily.GAUSSIAN,  # 2 → Kalman
                DistributionFamily.STUDENT_T,  # 3 → PF (contaminates block {2,3})
                DistributionFamily.GAUSSIAN,  # 4 → Kalman
            ],
        )
        partition = analyze_first_pass_rb(spec)

        # Block {0,1}: all Gaussian → Kalman
        # Block {2,3}: mixed → PF
        # Block {4}: Gaussian → Kalman
        np.testing.assert_array_equal(partition.kalman_idx, [0, 1, 4])
        np.testing.assert_array_equal(partition.particle_idx, [2, 3])

    def test_nonidentity_link_prevents_kalman(self):
        """Gaussian diffusion + Gaussian obs but non-identity link → goes to PF."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
            manifest_links=[LinkFunction.LOG, LinkFunction.IDENTITY],
        )
        partition = analyze_first_pass_rb(spec)

        # Latent 0 has log link → PF despite Gaussian everything else
        # Latent 1 has identity link → Kalman
        np.testing.assert_array_equal(partition.kalman_idx, [1])
        np.testing.assert_array_equal(partition.particle_idx, [0])

    def test_all_nonidentity_links_no_kalman(self):
        """All non-identity links → no Kalman block."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        partition = analyze_first_pass_rb(spec)

        assert not partition.has_kalman_block
        assert partition.has_particle_block

    def test_dense_loading_mask_no_split(self):
        """Dense loading mask leaves no exclusive observation channels."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=jnp.zeros((2, 2)),
            lambda_mask=np.ones((2, 2), dtype=bool),
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        partition = analyze_first_pass_rb(spec)
        # Var 0 is decoupled in drift, but dense loadings mean all obs depend on both vars.
        np.testing.assert_array_equal(partition.kalman_idx, [0])
        assert len(partition.obs_kalman_idx) == 0  # no exclusive obs

    def test_g_feeds_s_prevents_split(self):
        """A[s,g] != 0 → g must go to particle (S depends on G)."""
        drift = jnp.array([[-0.5, 0.0], [0.2, -0.3]])  # A[1,0] = 0.2: S <- G
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            drift=drift,
            lambda_mat=jnp.eye(2),
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        partition = analyze_first_pass_rb(spec)
        assert len(partition.kalman_idx) == 0

    def test_s_feeds_g_prevents_split(self):
        """A[g,s] != 0 → g must go to particle (G depends on S)."""
        drift = jnp.array([[-0.5, 0.15], [0.0, -0.3]])  # A[0,1] = 0.15: G <- S
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            drift=drift,
            lambda_mat=jnp.eye(2),
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        partition = analyze_first_pass_rb(spec)
        assert len(partition.kalman_idx) == 0

    def test_partial_split_3var(self):
        """2 Gaussian isolated + 1 Student-t → 2 in Kalman."""
        # Drift: 3x3 with [0,1] block-diagonal, [2] separate
        drift = jnp.array(
            [
                [-0.5, 0.1, 0.0],
                [0.1, -0.3, 0.0],
                [0.0, 0.0, -0.8],
            ]
        )
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            drift=drift,
            lambda_mat=jnp.eye(3),
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
            ],
        )
        partition = analyze_first_pass_rb(spec)
        np.testing.assert_array_equal(partition.kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.particle_idx, [2])

    def test_shared_obs_prevents_kalman(self):
        """Dense lambda couples G and S vars into shared obs → no Kalman obs."""
        # All 2 obs depend on all 2 latent vars
        lam = jnp.ones((2, 2))
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=lam,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        partition = analyze_first_pass_rb(spec)
        # Var 0 is Gaussian and decoupled in drift, but all obs depend on both vars
        # → no exclusive obs for var 0 → obs_kalman is empty
        np.testing.assert_array_equal(partition.kalman_idx, [0])
        np.testing.assert_array_equal(partition.particle_idx, [1])
        assert len(partition.obs_kalman_idx) == 0
        np.testing.assert_array_equal(partition.obs_particle_idx, [0, 1])

    def test_independent_blocks_clean_split(self):
        """Block-diagonal drift with 2 Gaussian + 1 Student-t → clean 3-var split."""
        n = 3
        m = 3
        # Block-diagonal drift: stable diagonal
        drift = jnp.diag(jnp.array([-0.5, -0.5, -0.5]))
        # Block-diagonal lambda: obs 0,1 → latent 0,1; obs 2 → latent 2
        lam = jnp.eye(3)
        spec = _make_spec(
            n_latent=n,
            n_manifest=m,
            drift=drift,
            lambda_mat=lam,
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
            ],
        )
        partition = analyze_first_pass_rb(spec)
        np.testing.assert_array_equal(partition.kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.particle_idx, [2])
        np.testing.assert_array_equal(partition.obs_kalman_idx, [0, 1])
        np.testing.assert_array_equal(partition.obs_particle_idx, [2])


# =============================================================================
# select_default_method — structural inference routing
# =============================================================================


class TestSelectDefaultMethod:
    def test_non_point_observation_support_routes_to_nuts(self):
        """Interval-summary observations should route to nuts."""
        spec = _make_spec()
        support = ObservationSupportRuntime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["y1", "y2"],
            support_kinds=["interval", "point"],
            summary_operators=["mean", "last"],
            anchor_policies=["support_end", "support_end"],
            observation_windows=["1d", "1d"],
            support_start_times=np.array([[np.nan, np.nan], [0.0, np.nan]]),
            support_end_times=np.array([[np.nan, np.nan], [1.0, np.nan]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0]], [[0.5], [0.0]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0]], [[0.5], [0.0]]]),
            interval_weights=np.array([[[0.0], [0.0]], [[1.0], [0.0]]]),
            emission_slot_indices=np.array([[-1, -1], [0, -1]], dtype=np.int64),
        )

        assert select_default_method(spec, observation_support=support) == "nuts"

    def test_gaussian_model_routes_to_nuts(self):
        """Fully Gaussian model with identity links → nuts."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
        )
        assert select_default_method(spec) == "nuts"

    def test_poisson_obs_routes_to_nuts(self):
        """Poisson observations → nuts."""
        spec = _make_spec(
            manifest_dists=[DistributionFamily.POISSON, DistributionFamily.POISSON],
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"

    def test_explicit_kalman_override_routes_to_nuts(self):
        """Explicit likelihood override should drive auto routing to nuts."""
        spec = _make_spec(
            manifest_dists=[DistributionFamily.POISSON, DistributionFamily.POISSON],
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        assert select_default_method(spec, likelihood="kalman") == "nuts"


class TestPlanInferenceStructure:
    def test_interval_summary_support_uses_particle_path_and_disables_first_pass(self):
        spec = _make_spec()
        support = ObservationSupportRuntime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["y1", "y2"],
            support_kinds=["interval", "point"],
            summary_operators=["mean", "last"],
            anchor_policies=["support_end", "support_end"],
            observation_windows=["1d", "1d"],
            support_start_times=np.array([[np.nan, np.nan], [0.0, np.nan]]),
            support_end_times=np.array([[np.nan, np.nan], [1.0, np.nan]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0]], [[0.5], [0.0]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0]], [[0.5], [0.0]]]),
            interval_weights=np.array([[[0.0], [0.0]], [[1.0], [0.0]]]),
            emission_slot_indices=np.array([[-1, -1], [0, -1]], dtype=np.int64),
        )

        plan = plan_inference_structure(spec, observation_support=support)

        assert plan.structural_backend == "particle"
        assert plan.resolved_method == "nuts"
        assert plan.method_override is None
        assert plan.first_pass_partition is None

    def test_separable_mixed_model_uses_composed_path(self):
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            lambda_mat=jnp.eye(3),
            drift_mask=np.eye(3, dtype=bool),
            diffusion_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
            ],
        )

        plan = plan_inference_structure(spec)

        assert plan.structural_backend == "composed"
        assert plan.resolved_method == "nuts"
        assert plan.method_override is None
        assert plan.first_pass_partition is not None

    def test_shared_observations_disable_executable_first_pass_split(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=jnp.ones((2, 2)),
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )

        plan = plan_inference_structure(spec)

        assert plan.structural_backend == "particle"
        assert plan.resolved_method == "nuts"
        assert plan.method_override is None
        assert plan.first_pass_partition is None

    def test_first_pass_disabled_keeps_full_kalman_backend_for_fully_gaussian_model(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
            first_pass_rb=False,
        )

        plan = plan_inference_structure(spec)

        assert plan.structural_backend == "kalman"
        assert plan.resolved_method == "nuts"
        assert plan.method_override is None
        assert plan.first_pass_partition is None

    def test_explicit_method_override_is_preserved_in_plan(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
        )

        plan = plan_inference_structure(spec, method_override="map")

        assert plan.structural_backend == "kalman"
        assert plan.resolved_method == "map"
        assert plan.method_override == "map"
        assert plan.first_pass_partition is None

    def test_aux_gibbs_override_is_preserved_in_plan(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
        )

        plan = plan_inference_structure(spec, method_override="aux_gibbs")

        assert plan.structural_backend == "kalman"
        assert plan.resolved_method == "aux_gibbs"
        assert plan.method_override == "aux_gibbs"
        assert plan.first_pass_partition is None

    def test_aux_csmc_override_is_preserved_in_plan(self):
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            drift_mask=np.eye(2, dtype=bool),
        )

        plan = plan_inference_structure(spec, method_override="aux_csmc")

        assert plan.structural_backend == "kalman"
        assert plan.resolved_method == "aux_csmc"
        assert plan.method_override == "aux_csmc"
        assert plan.first_pass_partition is None

    def test_student_t_diffusion_routes_to_nuts(self):
        """Student-t diffusion noise → nuts."""
        spec = _make_spec(
            diffusion_dists=[DistributionFamily.STUDENT_T, DistributionFamily.STUDENT_T],
        )
        assert select_default_method(spec) == "nuts"

    def test_mixed_model_routes_to_nuts(self):
        """Mixed Gaussian + non-Gaussian with coupling → nuts."""
        spec = _make_spec(
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        assert select_default_method(spec) == "nuts"

    def test_gaussian_with_log_link_routes_to_nuts(self):
        """Gaussian noise but log link → non-Kalman → nuts."""
        spec = _make_spec(
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"

    def test_bernoulli_routes_to_nuts(self):
        """Bernoulli observations → nuts."""
        spec = _make_spec(
            manifest_dists=[DistributionFamily.BERNOULLI, DistributionFamily.BERNOULLI],
            manifest_links=[LinkFunction.LOGIT, LinkFunction.LOGIT],
        )
        assert select_default_method(spec) == "nuts"

    def test_gamma_routes_to_nuts(self):
        """Gamma observations → nuts."""
        spec = _make_spec(
            manifest_dists=[DistributionFamily.GAMMA, DistributionFamily.GAMMA],
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"

    def test_negative_binomial_routes_to_nuts(self):
        """Negative binomial observations → nuts."""
        spec = _make_spec(
            manifest_dists=[
                DistributionFamily.NEGATIVE_BINOMIAL,
                DistributionFamily.NEGATIVE_BINOMIAL,
            ],
            manifest_links=[LinkFunction.LOG, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"

    def test_beta_routes_to_nuts(self):
        """Beta observations → nuts."""
        spec = _make_spec(
            manifest_dists=[DistributionFamily.BETA, DistributionFamily.BETA],
            manifest_links=[LinkFunction.LOGIT, LinkFunction.LOGIT],
        )
        assert select_default_method(spec) == "nuts"

    def test_per_channel_mixed_routes_to_nuts(self):
        """Per-channel mixed distributions: one Poisson channel → nuts."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.POISSON],
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"

    def test_large_gaussian_model_routes_to_nuts(self):
        """Larger fully Gaussian model → nuts."""
        n = 5
        spec = _make_spec(
            n_latent=n,
            n_manifest=n,
            lambda_mat=jnp.eye(n),
            drift_mask=np.eye(n, dtype=bool),
        )
        assert select_default_method(spec) == "nuts"

    def test_zero_dep_gaussian_channel_still_routes_nuts(self):
        """Gaussian zero-dep channel doesn't break pure-Kalman routing → nuts."""
        H = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],  # zero-dep Gaussian+identity
            ]
        )
        spec = _make_spec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=H,
            drift_mask=np.eye(2, dtype=bool),
        )
        assert select_default_method(spec) == "nuts"

    def test_zero_dep_nongaussian_channel_routes_to_nuts(self):
        """Non-Gaussian zero-dep channel prevents pure-Kalman → nuts."""
        H = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],  # zero-dep Poisson
            ]
        )
        spec = _make_spec(
            n_latent=2,
            n_manifest=3,
            lambda_mat=H,
            drift_mask=np.eye(2, dtype=bool),
            manifest_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.GAUSSIAN,
                DistributionFamily.POISSON,
            ],
            manifest_links=[LinkFunction.IDENTITY, LinkFunction.IDENTITY, LinkFunction.LOG],
        )
        assert select_default_method(spec) == "nuts"
