"""Tests for block Rao-Blackwell particle filter (BIRCH-style graph decomposition).

Test hierarchy:
1. Partition logic (pure logic, fast)
2. Matrix block extraction (linear algebra, fast)
3. Degenerate case equivalence (all-G = RBPF, all-S = bootstrap)
4. Independent block decomposition (additive LL)
5. Cross-coupling correctness (S->G, shared obs)
6. Variance reduction (block RBPF < bootstrap PF variance)
7. Gradient flow (jax.grad produces finite gradients)
8. Parameter recovery (barebone NUTS + bootstrap, then NUTS + block RBPF)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import pytest

from causal_ssm_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.likelihoods.block_rb import (
    extract_obs_subblocks,
    extract_subblocks,
    partition_indices,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_mixed_params(n_g=1, n_s=1, n_manifest=2, cross_coupling=True):
    """Build test parameters for a mixed Gaussian/non-Gaussian model.

    Creates a stable system with n_g Gaussian + n_s sampled latent variables.
    """
    n = n_g + n_s

    # Drift: stable diagonal, optional cross-coupling
    drift = jnp.diag(jnp.full(n, -0.5))
    if cross_coupling and n_g > 0 and n_s > 0:
        # G->S coupling
        drift = drift.at[n_g, 0].set(0.2)
        # S->G coupling
        drift = drift.at[0, n_g].set(0.15)

    # Diffusion: diagonal
    diffusion_cov = jnp.eye(n) * 0.1

    ct_params = CTParams(
        drift=drift,
        diffusion_cov=diffusion_cov,
        cint=jnp.zeros(n),
    )
    meas_params = MeasurementParams(
        lambda_mat=jnp.eye(n_manifest, n),
        manifest_means=jnp.zeros(n_manifest),
        manifest_cov=jnp.eye(n_manifest) * 0.1,
    )
    init = InitialStateParams(
        mean=jnp.zeros(n),
        cov=jnp.eye(n),
    )
    return ct_params, meas_params, init


def _simulate_data(key, ct_params, meas_params, init, T=30):
    """Simulate Gaussian observations from a linear-Gaussian latent process."""
    n = init.mean.shape[0]
    n_manifest = meas_params.lambda_mat.shape[0]

    k1, k2 = random.split(key)
    states = [init.mean]
    dt = 1.0
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        drift_effect = ct_params.drift @ states[-1] * dt
        noise = random.normal(k_step, (n,)) * jnp.sqrt(0.1 * dt)
        states.append(states[-1] + drift_effect + noise)
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    noise = random.normal(k2, (T, n_manifest)) * jnp.sqrt(0.1)
    observations = eta + noise

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def _simulate_data_exact(key, ct_params, meas_params, init, T=30):
    """Simulate Gaussian observations using exact CT->DT discretization.

    Unlike _simulate_data (Euler), this matches the filter's matrix-exponential
    discretization for unbiased parameter recovery.
    """
    from causal_ssm_agent.models.ssm.discretization import discretize_system

    n = init.mean.shape[0]
    n_manifest = meas_params.lambda_mat.shape[0]
    dt = 1.0

    Ad, Qd, cd = discretize_system(ct_params.drift, ct_params.diffusion_cov, ct_params.cint, dt)
    if cd is None:
        cd = jnp.zeros(n)
    cd = cd.flatten()
    chol_Qd = jla.cholesky(Qd + jnp.eye(n) * 1e-6, lower=True)

    k1, k2 = random.split(key)
    states = [init.mean]
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        mean = Ad @ states[-1] + cd
        states.append(mean + chol_Qd @ random.normal(k_step, (n,)))
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    chol_R = jla.cholesky(meas_params.manifest_cov + jnp.eye(n_manifest) * 1e-8, lower=True)
    noise = (chol_R @ random.normal(k2, (n_manifest, T))).T
    observations = eta + noise

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def _simulate_poisson_data(key, ct_params, meas_params, init, T=30):
    """Simulate Poisson count observations using exact CT->DT discretization."""
    from causal_ssm_agent.models.ssm.discretization import discretize_system

    n = init.mean.shape[0]
    dt = 1.0

    Ad, Qd, cd = discretize_system(ct_params.drift, ct_params.diffusion_cov, ct_params.cint, dt)
    if cd is None:
        cd = jnp.zeros(n)
    cd = cd.flatten()
    chol_Qd = jla.cholesky(Qd + jnp.eye(n) * 1e-6, lower=True)

    k1, k2 = random.split(key)
    states = [init.mean]
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        mean = Ad @ states[-1] + cd
        states.append(mean + chol_Qd @ random.normal(k_step, (n,)))
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    rates = jnp.exp(jnp.clip(eta, -5.0, 5.0))
    observations = random.poisson(k2, rates).astype(jnp.float32)

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def _run_block_rbpf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    diffusion_dists,
    manifest_dist="gaussian",
    n_particles=200,
    rng_key=None,
    extra_params=None,
):
    """Run block RBPF with per-variable diffusion dists."""
    from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dist=manifest_dist,
        diffusion_dist=diffusion_dists,
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
        extra_params=extra_params,
    )


def _run_full_rbpf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dist="gaussian",
    n_particles=200,
    rng_key=None,
):
    """Run full RBPF (all Gaussian)."""
    from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dist=manifest_dist,
        diffusion_dist="gaussian",
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
    )


def _run_bootstrap_pf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dist="gaussian",
    diffusion_dist="student_t",
    n_particles=200,
    rng_key=None,
    extra_params=None,
):
    """Run bootstrap PF (all sampled)."""
    from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    ep = {"proc_df": 100.0}
    if extra_params:
        ep.update(extra_params)

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dist=manifest_dist,
        diffusion_dist=diffusion_dist,
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
        extra_params=ep,
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
        g_idx = jnp.array([0, 2], dtype=jnp.int32)
        s_idx = jnp.array([1], dtype=jnp.int32)

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
        g_idx = jnp.array([0, 3], dtype=jnp.int32)
        s_idx = jnp.array([1, 2], dtype=jnp.int32)

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
        g_idx = jnp.array([0, 2], dtype=jnp.int32)
        s_idx = jnp.array([1], dtype=jnp.int32)

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

    def test_full_coupling_finite(self):
        """Full cross-coupling (G<->S): LL should be finite."""
        ct, meas, init = _make_mixed_params(
            n_g=1,
            n_s=1,
            n_manifest=2,
            cross_coupling=True,
        )
        key = random.PRNGKey(222)
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

    def test_deterministic_with_fixed_seed(self):
        """Block RBPF should be deterministic with fixed RNG key."""
        ct, meas, init = _make_mixed_params(n_g=1, n_s=1, n_manifest=2)
        key = random.PRNGKey(333)
        obs, dt = _simulate_data(key, ct, meas, init, T=15)

        ll1 = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "student_t"],
            rng_key=random.PRNGKey(99),
            extra_params={"proc_df": 5.0},
        )
        ll2 = _run_block_rbpf(
            ct,
            meas,
            init,
            obs,
            dt,
            diffusion_dists=["gaussian", "student_t"],
            rng_key=random.PRNGKey(99),
            extra_params={"proc_df": 5.0},
        )
        assert float(ll1) == float(ll2)

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


class TestVarianceReduction:
    """Block RBPF should have lower LL variance than full bootstrap PF."""

    @pytest.mark.slow
    def test_variance_reduction(self):
        """Block RBPF variance < bootstrap PF variance on same model."""
        ct, meas, init = _make_mixed_params(
            n_g=1,
            n_s=1,
            n_manifest=2,
            cross_coupling=False,
        )
        key = random.PRNGKey(555)
        obs, dt = _simulate_data(key, ct, meas, init, T=20)

        n_runs = 30
        n_particles = 100

        # Block RBPF (Gaussian variable marginalized)
        block_lls = []
        for i in range(n_runs):
            ll = _run_block_rbpf(
                ct,
                meas,
                init,
                obs,
                dt,
                diffusion_dists=["gaussian", "student_t"],
                n_particles=n_particles,
                rng_key=random.PRNGKey(i),
                extra_params={"proc_df": 100.0},
            )
            block_lls.append(float(ll))

        # Full bootstrap PF (everything sampled)
        boot_lls = []
        for i in range(n_runs):
            ll = _run_bootstrap_pf(
                ct,
                meas,
                init,
                obs,
                dt,
                n_particles=n_particles,
                rng_key=random.PRNGKey(i),
                extra_params={"proc_df": 100.0},
            )
            boot_lls.append(float(ll))

        var_block = np.var(block_lls)
        var_boot = np.var(boot_lls)

        # Block RBPF should have strictly lower variance
        assert var_block < var_boot, (
            f"Block RBPF variance ({var_block:.4f}) should be < "
            f"bootstrap PF variance ({var_boot:.4f})"
        )


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


class TestParameterRecovery:
    """Parameter recovery: first with barebone bootstrap, then with block RBPF.

    This establishes:
    1. The model is identifiable (bootstrap NUTS recovers params).
    2. Block RBPF preserves correctness (also recovers params).
    """

    @pytest.mark.slow
    def test_parameter_recovery_bootstrap_baseline(self):
        """Baseline: bootstrap PF + SVI recovers drift diagonal."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        # Ground truth model: 1G + 1S, Gaussian obs
        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        drift_true = jnp.diag(true_drift_diag)
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(drift=drift_true, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))

        # Simulate data (short for speed)
        key = random.PRNGKey(777)
        obs, dt = _simulate_data(key, ct_true, meas, init, T=30)

        # Define NumPyro model with bootstrap PF
        def model(observations, time_intervals):
            drift_diag = numpyro.sample(
                "drift_diag",
                dist.Normal(-0.5, 0.5).expand([n]),
            )
            drift = jnp.diag(-jnp.abs(drift_diag))
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=50,
                rng_key=random.PRNGKey(0),
                manifest_dist="gaussian",
                diffusion_dist="student_t",
            )
            ll = backend.compute_log_likelihood(
                ct,
                meas,
                init,
                observations,
                time_intervals,
                extra_params={"proc_df": 100.0},
            )
            numpyro.factor("ll", ll)

        guide = AutoNormal(model)
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        # JIT the update for speed
        jit_update = jax.jit(svi.update)
        for _step in range(300):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        recovered = -jnp.abs(params["drift_diag_auto_loc"])

        np.testing.assert_allclose(
            np.sort(recovered),
            np.sort(np.array(true_drift_diag)),
            atol=0.3,
        )

    @pytest.mark.slow
    def test_parameter_recovery_block_rbpf(self):
        """Block RBPF + SVI also recovers drift diagonal."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        drift_true = jnp.diag(true_drift_diag)
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(drift=drift_true, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))

        key = random.PRNGKey(777)
        obs, dt = _simulate_data(key, ct_true, meas, init, T=30)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample(
                "drift_diag",
                dist.Normal(-0.5, 0.5).expand([n]),
            )
            drift = jnp.diag(-jnp.abs(drift_diag))
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=100,
                rng_key=random.PRNGKey(0),
                manifest_dist="gaussian",
                diffusion_dist=["gaussian", "student_t"],
            )
            ll = backend.compute_log_likelihood(
                ct,
                meas,
                init,
                observations,
                time_intervals,
                extra_params={"proc_df": 100.0},
            )
            numpyro.factor("ll", ll)

        guide = AutoNormal(model)
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _step in range(500):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        recovered = -jnp.abs(params["drift_diag_auto_loc"])

        # Check each parameter is in the right ballpark (order doesn't matter)
        recovered_sorted = np.sort(np.array(recovered))
        true_sorted = np.sort(np.array(true_drift_diag))
        np.testing.assert_allclose(recovered_sorted, true_sorted, atol=0.35)

    @pytest.mark.slow
    def test_parameter_recovery_cross_coupled_drift(self):
        """Block RBPF + SVI recovers cross-coupled 2x2 drift (diagonal + off-diagonal)."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_diag = jnp.array([-0.3, -0.7])
        true_offdiag = jnp.array([0.15, 0.2])  # A[0,1]=S->G, A[1,0]=G->S
        n = 2
        drift_true = jnp.array([[true_diag[0], true_offdiag[0]], [true_offdiag[1], true_diag[1]]])
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(drift=drift_true, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))

        key = random.PRNGKey(888)
        obs, dt = _simulate_data_exact(key, ct_true, meas, init, T=50)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.3).expand([n]))
            drift_offdiag = numpyro.sample("drift_offdiag", dist.Normal(0.0, 0.3).expand([n]))
            drift = jnp.zeros((n, n))
            drift = drift.at[0, 0].set(-jnp.abs(drift_diag[0]))
            drift = drift.at[1, 1].set(-jnp.abs(drift_diag[1]))
            drift = drift.at[0, 1].set(drift_offdiag[0])
            drift = drift.at[1, 0].set(drift_offdiag[1])
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=100,
                rng_key=random.PRNGKey(0),
                manifest_dist="gaussian",
                diffusion_dist=["gaussian", "student_t"],
            )
            ll = backend.compute_log_likelihood(
                ct,
                meas,
                init,
                observations,
                time_intervals,
                extra_params={"proc_df": 100.0},
            )
            numpyro.factor("ll", ll)

        guide = AutoNormal(model)
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _step in range(500):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        recovered_diag = -jnp.abs(params["drift_diag_auto_loc"])
        recovered_offdiag = params["drift_offdiag_auto_loc"]

        # Diagonal entries
        np.testing.assert_allclose(np.sort(recovered_diag), np.sort(np.array(true_diag)), atol=0.35)
        # Off-diagonal coupling terms
        np.testing.assert_allclose(recovered_offdiag, np.array(true_offdiag), atol=0.25)

    @pytest.mark.slow
    def test_parameter_recovery_higher_dim_1g_2s(self):
        """Block RBPF + SVI recovers drift diagonal for 3-variable (1G+2S) model.

        Tests the asymmetric partition with more sampled than Gaussian variables.
        The 2D S-block is sampled, 1D G-block is Kalman-marginalized.
        """
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_drift_diag = jnp.array([-0.3, -0.5, -0.8])
        n = 3
        drift_true = jnp.diag(true_drift_diag)
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(drift=drift_true, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))

        key = random.PRNGKey(1111)
        obs, dt = _simulate_data_exact(key, ct_true, meas, init, T=50)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.3).expand([n]))
            drift = jnp.diag(-jnp.abs(drift_diag))
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

            # 1G + 2S: variable 0 is Gaussian, variables 1,2 are sampled
            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=200,
                rng_key=random.PRNGKey(0),
                manifest_dist="gaussian",
                diffusion_dist=["gaussian", "student_t", "student_t"],
            )
            ll = backend.compute_log_likelihood(
                ct,
                meas,
                init,
                observations,
                time_intervals,
                extra_params={"proc_df": 100.0},
            )
            numpyro.factor("ll", ll)

        guide = AutoNormal(model)
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _step in range(600):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        recovered = -jnp.abs(params["drift_diag_auto_loc"])

        np.testing.assert_allclose(
            np.sort(recovered), np.sort(np.array(true_drift_diag)), atol=0.35
        )

    @pytest.mark.slow
    def test_parameter_recovery_poisson_obs(self):
        """Block RBPF with Poisson obs: LL landscape peaks near true params.

        Exercises the non-Gaussian quadrature path in log_potential and the
        EKF linearized update with the d_eff correction in propagate_sample.
        Tests that: (1) LL is finite, (2) true params score higher than
        wrong params, (3) gradients are finite.
        """
        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        drift_true = jnp.diag(true_drift_diag)
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(drift=drift_true, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.ones(n) * 1.5,  # baseline log-rate ~1.5 → rate ~4.5
            manifest_cov=jnp.eye(n) * 0.1,  # not used by Poisson emission
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))

        key = random.PRNGKey(1234)
        obs, dt = _simulate_poisson_data(key, ct_true, meas, init, T=50)

        def compute_ll(drift_diag_vals):
            drift = jnp.diag(-jnp.abs(drift_diag_vals))
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))
            return _run_block_rbpf(
                ct,
                meas,
                init,
                obs,
                dt,
                diffusion_dists=["gaussian", "student_t"],
                n_particles=200,
                rng_key=random.PRNGKey(42),
                manifest_dist="poisson",
                extra_params={"proc_df": 100.0},
            )

        # 1. LL at true params is finite
        ll_true = compute_ll(jnp.abs(true_drift_diag))
        assert jnp.isfinite(ll_true), f"LL at true params not finite: {ll_true}"

        # 2. LL at true params > LL at distant wrong params
        ll_wrong = compute_ll(jnp.array([1.5, 0.05]))
        assert float(ll_true) > float(ll_wrong), (
            f"LL at true ({float(ll_true):.1f}) should exceed "
            f"LL at wrong params ({float(ll_wrong):.1f})"
        )

        # 3. Gradient at true params is finite (differentiable through quadrature)
        grad_fn = jax.grad(compute_ll)
        grad_val = grad_fn(jnp.abs(true_drift_diag))
        assert jnp.all(jnp.isfinite(grad_val)), f"Gradient not finite: {grad_val}"
