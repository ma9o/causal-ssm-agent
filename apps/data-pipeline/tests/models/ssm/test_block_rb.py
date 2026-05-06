"""Tests for block Rao-Blackwell particle filter (BIRCH-style graph decomposition).

Test hierarchy:
1. Partition logic (pure logic, fast)
2. Matrix block extraction (linear algebra, fast)
3. Degenerate case equivalence (all-G = RBPF, all-S = bootstrap)
4. Independent block decomposition (additive LL)
5. Cross-coupling correctness (S->G, shared obs)
6. Variance reduction (block RBPF < bootstrap PF variance)
7. Gradient flow (jax.grad produces finite gradients)
8. Parameter recovery (barebone bootstrap, then block RBPF)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.ssm.inference.targets.block_rb import (
    extract_obs_subblocks,
    extract_subblocks,
    partition_indices,
)
from tests.models.ssm.block_rb_support import (
    make_mixed_params as _make_mixed_params,
)
from tests.models.ssm.block_rb_support import (
    run_block_rbpf as _run_block_rbpf,
)
from tests.models.ssm.block_rb_support import (
    run_bootstrap_pf as _run_bootstrap_pf,
)
from tests.models.ssm.block_rb_support import (
    run_full_rbpf as _run_full_rbpf,
)
from tests.models.ssm.block_rb_support import (
    simulate_data as _simulate_data,
)

# =============================================================================
# Level 1: Partition Logic
# =============================================================================


class TestPartitionIndices:
    """Test partition_indices correctly separates Gaussian vs sampled."""

    def test_all_gaussian(self):
        g_idx, s_idx = partition_indices(["gaussian", "gaussian", "gaussian"])
        np.testing.assert_array_equal(g_idx, [0, 1, 2])
        assert s_idx.shape[0] == 0

    def test_all_student_t(self):
        g_idx, s_idx = partition_indices(["student_t", "student_t"])
        assert g_idx.shape[0] == 0
        np.testing.assert_array_equal(s_idx, [0, 1])

    def test_mixed(self):
        g_idx, s_idx = partition_indices(
            ["gaussian", "student_t", "gaussian", "gaussian", "student_t"]
        )
        np.testing.assert_array_equal(g_idx, [0, 2, 3])
        np.testing.assert_array_equal(s_idx, [1, 4])

    def test_single_each(self):
        g_idx, s_idx = partition_indices(["gaussian", "student_t"])
        np.testing.assert_array_equal(g_idx, [0])
        np.testing.assert_array_equal(s_idx, [1])

    def test_single_gaussian(self):
        g_idx, s_idx = partition_indices(["gaussian"])
        np.testing.assert_array_equal(g_idx, [0])
        assert s_idx.shape[0] == 0


# =============================================================================
# Level 2: Matrix Block Extraction
# =============================================================================


class TestExtractSubblocks:
    """Test matrix sub-block extraction."""

    def test_basic_extraction(self):
        A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.float32)
        Q = jnp.eye(3) * 0.1
        c = jnp.array([10.0, 20.0, 30.0])
        g_idx = jnp.array([0, 2], dtype=jnp.int64)
        s_idx = jnp.array([1], dtype=jnp.int64)

        blocks = extract_subblocks(A, Q, c, g_idx, s_idx)

        # A_gg: rows [0,2], cols [0,2]
        np.testing.assert_array_equal(blocks["A_gg"], [[1, 3], [7, 9]])
        # A_gs: rows [0,2], cols [1]
        np.testing.assert_array_equal(blocks["A_gs"], [[2], [8]])
        # A_sg: rows [1], cols [0,2]
        np.testing.assert_array_equal(blocks["A_sg"], [[4, 6]])
        # A_ss: rows [1], cols [1]
        np.testing.assert_array_equal(blocks["A_ss"], [[5]])
        # c_g, c_s
        np.testing.assert_array_equal(blocks["c_g"], [10.0, 30.0])
        np.testing.assert_array_equal(blocks["c_s"], [20.0])

    def test_round_trip(self):
        """Reassembling sub-blocks should recover the original matrix."""
        n = 4
        key = random.PRNGKey(0)
        A = random.normal(key, (n, n))
        Q = jnp.eye(n) * 0.1
        c = jnp.arange(n, dtype=jnp.float32)
        g_idx = jnp.array([0, 3], dtype=jnp.int64)
        s_idx = jnp.array([1, 2], dtype=jnp.int64)

        blocks = extract_subblocks(A, Q, c, g_idx, s_idx)

        # Reconstruct A
        all_idx = jnp.concatenate([g_idx, s_idx])
        inv_perm = jnp.argsort(all_idx)

        A_reorder = jnp.block(
            [
                [blocks["A_gg"], blocks["A_gs"]],
                [blocks["A_sg"], blocks["A_ss"]],
            ]
        )
        A_recovered = A_reorder[jnp.ix_(inv_perm, inv_perm)]
        np.testing.assert_allclose(A_recovered, A, atol=1e-6)

    def test_obs_subblocks(self):
        H = jnp.array([[1, 0, 2], [0, 3, 1]], dtype=jnp.float32)
        g_idx = jnp.array([0, 2], dtype=jnp.int64)
        s_idx = jnp.array([1], dtype=jnp.int64)

        H_g, H_s = extract_obs_subblocks(H, g_idx, s_idx)
        np.testing.assert_array_equal(H_g, [[1, 2], [0, 1]])
        np.testing.assert_array_equal(H_s, [[0], [3]])


# =============================================================================
# Level 3: Degenerate Case Equivalence
# =============================================================================


class TestDegenerateEquivalence:
    """Block RBPF with all-G or all-S must match existing implementations."""

    def test_all_gaussian_matches_full_rbpf(self):
        """All-Gaussian block RBPF should match full RBPF exactly."""
        ct, meas, init = _make_mixed_params(n_g=2, n_s=0, n_manifest=2, cross_coupling=False)
        key = random.PRNGKey(123)
        obs, dt = _simulate_data(key, ct, meas, init, T=20)

        ll_block = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "gaussian"],
            rng_key=random.PRNGKey(42),
        )
        ll_full = _run_full_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            rng_key=random.PRNGKey(42),
        )

        assert jnp.isfinite(ll_block)
        assert jnp.isfinite(ll_full)
        np.testing.assert_allclose(float(ll_block), float(ll_full), atol=1e-3)

    def test_all_sampled_matches_bootstrap(self):
        """All-sampled block RBPF should match bootstrap PF (high df ~ Gaussian)."""
        ct, meas, init = _make_mixed_params(n_g=0, n_s=2, n_manifest=2, cross_coupling=False)
        key = random.PRNGKey(456)
        obs, dt = _simulate_data(key, ct, meas, init, T=20)

        # Both use Student-t with high df (≈ Gaussian)
        ll_block = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["student_t", "student_t"],
            rng_key=random.PRNGKey(42),
            extra_params={"proc_df": 100.0},
        )
        ll_boot = _run_bootstrap_pf(
            ct,
            meas,
            init,
            obs,
            dt,
            rng_key=random.PRNGKey(42),
            extra_params={"proc_df": 100.0},
        )

        assert jnp.isfinite(ll_block)
        assert jnp.isfinite(ll_boot)
        np.testing.assert_allclose(float(ll_block), float(ll_boot), atol=1e-3)


# =============================================================================
# Level 4: Independent Block Decomposition
# =============================================================================


class TestIndependentBlocks:
    """With no cross-coupling, LL should decompose additively."""

    def test_additive_ll_decomposition(self):
        """LL(block_rbpf) ≈ LL(kalman on G) + LL(bootstrap on S)."""
        # Build independent 1G + 1S model (block-diagonal drift)
        ct, meas, init = _make_mixed_params(
            n_g=1,
            n_s=1,
            n_manifest=2,
            cross_coupling=False,
        )
        key = random.PRNGKey(789)
        obs, dt = _simulate_data(key, ct, meas, init, T=20)

        # Block RBPF on the full model
        ll_block = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "student_t"],
            n_particles=500,
            rng_key=random.PRNGKey(42),
            extra_params={"proc_df": 100.0},
        )

        # Kalman filter on G-block alone
        ct_g = CTParams(
            drift=ct.drift[:1, :1],
            diffusion_cov=ct.diffusion_cov[:1, :1],
            cint=ct.cint[:1],
        )
        meas_g = MeasurementParams(
            lambda_mat=meas.lambda_mat[:1, :1],
            manifest_means=meas.manifest_means[:1],
            manifest_cov=meas.manifest_cov[:1, :1],
        )
        init_g = InitialStateParams(mean=init.mean[:1], cov=init.cov[:1, :1])
        ll_kalman_g = _run_full_rbpf(
            ct_g,
            meas_g,
            init_g,
            obs[:, :1],
            dt,
            n_particles=500,
            rng_key=random.PRNGKey(42),
        )

        # Bootstrap PF on S-block alone
        ct_s = CTParams(
            drift=ct.drift[1:, 1:],
            diffusion_cov=ct.diffusion_cov[1:, 1:],
            cint=ct.cint[1:],
        )
        meas_s = MeasurementParams(
            lambda_mat=meas.lambda_mat[1:, 1:],
            manifest_means=meas.manifest_means[1:],
            manifest_cov=meas.manifest_cov[1:, 1:],
        )
        init_s = InitialStateParams(mean=init.mean[1:], cov=init.cov[1:, 1:])
        ll_boot_s = _run_bootstrap_pf(
            ct_s,
            meas_s,
            init_s,
            obs[:, 1:],
            dt,
            n_particles=500,
            rng_key=random.PRNGKey(42),
            extra_params={"proc_df": 100.0},
        )

        # Block RBPF ≈ sum of independent sub-LLs (approximately, due to PF variance)
        ll_sum = float(ll_kalman_g) + float(ll_boot_s)
        assert jnp.isfinite(ll_block)
        np.testing.assert_allclose(float(ll_block), ll_sum, atol=5.0)


# =============================================================================
# Level 5: Cross-Coupling
# =============================================================================


class TestCrossCoupling:
    """Tests for models with edges between Gaussian and sampled blocks."""

    def test_s_to_g_coupling_finite(self):
        """S->G coupling: LL should be finite and sensible."""
        ct, meas, init = _make_mixed_params(n_g=1, n_s=1, n_manifest=2)
        key = random.PRNGKey(111)
        obs, dt = _simulate_data(key, ct, meas, init, T=20)

        ll = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "student_t"],
            n_particles=200,
            extra_params={"proc_df": 5.0},
        )
        assert jnp.isfinite(ll)
        assert float(ll) < 0  # LL should be negative

    def test_higher_dim_mixed(self):
        """3G + 2S with mixed coupling: LL is finite."""
        ct, meas, init = _make_mixed_params(
            n_g=3,
            n_s=2,
            n_manifest=5,
            cross_coupling=True,
        )
        key = random.PRNGKey(444)
        obs, dt = _simulate_data(key, ct, meas, init, T=15)

        ll = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "gaussian", "gaussian", "student_t", "student_t"],
            n_particles=200,
            extra_params={"proc_df": 5.0},
        )
        assert jnp.isfinite(ll)


# =============================================================================
# Level 6: Variance Reduction
# =============================================================================


# =============================================================================
# Level 7: Gradient Flow
# =============================================================================


class TestGradientFlow:
    """jax.grad must flow through block RBPF callbacks."""

    def test_grad_finite(self):
        """Gradient of LL w.r.t. drift diagonal should be finite."""
        _, meas, init = _make_mixed_params(n_g=1, n_s=1, n_manifest=2)
        key = random.PRNGKey(666)
        drift_base = jnp.array([[-0.5, 0.15], [0.2, -0.5]])
        ct = CTParams(
            drift=drift_base,
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.zeros(2),
        )
        obs, dt = _simulate_data(key, ct, meas, init, T=15)

        def ll_fn(drift_diag):
            drift = drift_base.at[jnp.diag_indices(2)].set(-jnp.abs(drift_diag))
            ct_local = CTParams(
                drift=drift,
                diffusion_cov=jnp.eye(2) * 0.1,
                cint=jnp.zeros(2),
            )
            return _run_block_rbpf(
                ct_local,
                meas,
                init,
                obs,
                dt,
                diffusion_dists=["gaussian", "student_t"],
                n_particles=50,
                rng_key=random.PRNGKey(42),
                extra_params={"proc_df": 5.0},
            )

        grad_fn = jax.grad(ll_fn)
        grad_val = grad_fn(jnp.array([0.5, 0.5]))
        assert jnp.all(jnp.isfinite(grad_val)), f"Gradient not finite: {grad_val}"


# =============================================================================
# Level 8: Parameter Recovery
# =============================================================================
