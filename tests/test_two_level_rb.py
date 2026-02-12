"""Tests for two-level greedy Rao-Blackwellization.

Test hierarchy:
1. Graph Analysis Pure Logic (sparsity, dependency, per-var resolution)
2. Partition Analysis (analyze_first_pass_rb: splitting, edge cases)
3. Degenerate Equivalence (all-Kalman == Kalman, all-particle == particle)
4. Additive LL Decomposition (independent blocks: composed ≈ sum of parts)
5. Pipeline Integration (make_likelihood_backend dispatching)
6. Variance Reduction (two-level < pure PF variance)
7. Gradient Flow (jax.grad finite through ComposedLikelihood)
8. Parameter Recovery (SVI recovers drift with two-level RB)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import pytest

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.graph_analysis import (
    analyze_first_pass_rb,
    compute_drift_sparsity,
    compute_obs_dependency,
    get_per_channel_manifest,
    get_per_variable_diffusion,
)
from dsem_agent.models.ssm.model import NoiseFamily, SSMSpec

# =============================================================================
# Helpers
# =============================================================================


def _make_separable_spec(
    n_g: int = 2,
    n_s: int = 1,
    n_obs_g: int = 2,
    n_obs_s: int = 1,
    cross_coupling: bool = False,
) -> SSMSpec:
    """Build an SSMSpec with block-diagonal drift/lambda and mixed diffusion_dists.

    Variables 0..n_g-1 are Gaussian, n_g..n_g+n_s-1 are Student-t.
    Observations 0..n_obs_g-1 map to Gaussian vars, n_obs_g..n_obs_g+n_obs_s-1 to Student-t.
    """
    n = n_g + n_s
    m = n_obs_g + n_obs_s

    # Block-diagonal drift: stable diagonal
    drift = np.diag(np.full(n, -0.5))
    if cross_coupling and n_g > 0 and n_s > 0:
        drift[n_g, 0] = 0.2  # S <- G
        drift[0, n_g] = 0.15  # G <- S
    drift = jnp.array(drift, dtype=jnp.float32)

    # Block-diagonal lambda: obs_g -> latent_g, obs_s -> latent_s
    lam = np.zeros((m, n))
    for i in range(min(n_obs_g, n_g)):
        lam[i, i] = 1.0
    for i in range(min(n_obs_s, n_s)):
        lam[n_obs_g + i, n_g + i] = 1.0
    lambda_mat = jnp.array(lam, dtype=jnp.float32)

    # Per-variable diffusion dists
    diffusion_dists = [NoiseFamily.GAUSSIAN] * n_g + [NoiseFamily.STUDENT_T] * n_s

    return SSMSpec(
        n_latent=n,
        n_manifest=m,
        drift=drift,
        diffusion=jnp.eye(n) * 0.3,
        lambda_mat=lambda_mat,
        manifest_var=jnp.eye(m) * 0.1,
        manifest_means=jnp.zeros(m),
        t0_means=jnp.zeros(n),
        t0_var=jnp.eye(n),
        diffusion_dist=NoiseFamily.GAUSSIAN,
        manifest_dist=NoiseFamily.GAUSSIAN,
        diffusion_dists=diffusion_dists,
    )


def _make_separable_params(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1):
    """Build CT params, measurement params, and initial state for a separable model."""
    n = n_g + n_s
    m = n_obs_g + n_obs_s

    drift = jnp.diag(jnp.full(n, -0.5))
    diffusion_cov = jnp.eye(n) * 0.1

    ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

    lam = np.zeros((m, n))
    for i in range(min(n_obs_g, n_g)):
        lam[i, i] = 1.0
    for i in range(min(n_obs_s, n_s)):
        lam[n_obs_g + i, n_g + i] = 1.0

    meas = MeasurementParams(
        lambda_mat=jnp.array(lam, dtype=jnp.float32),
        manifest_means=jnp.zeros(m),
        manifest_cov=jnp.eye(m) * 0.1,
    )
    init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))
    return ct, meas, init


def _simulate_separable_data(key, ct_params, meas_params, init, T=30):
    """Simulate data using exact CT->DT discretization."""
    from dsem_agent.models.ssm.discretization import discretize_system

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
    for _ in range(T - 1):
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


def _run_composed(spec, ct, meas, init, obs, dt, n_particles=200, extra_params=None):
    """Build backend via make_likelihood_backend() and compute LL."""
    from dsem_agent.models.ssm.model import SSMModel

    model = SSMModel(spec=spec, n_particles=n_particles)
    backend = model.make_likelihood_backend()
    return backend.compute_log_likelihood(
        ct,
        meas,
        init,
        obs,
        dt,
        extra_params=extra_params,
    )


# =============================================================================
# Level 1: Graph Analysis Pure Logic
# =============================================================================


class TestGraphAnalysis:
    """Test graph analysis utility functions."""

    def test_drift_sparsity_free(self):
        """drift="free" -> all-True mask."""
        spec = SSMSpec(n_latent=3, n_manifest=3, drift="free")
        mask = compute_drift_sparsity(spec)
        assert mask.shape == (3, 3)
        assert mask.all()

    def test_drift_sparsity_fixed_diag(self):
        """Diagonal drift -> only diagonal True."""
        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift=jnp.diag(jnp.array([-0.5, -0.3, -0.8])),
        )
        mask = compute_drift_sparsity(spec)
        np.testing.assert_array_equal(mask, np.eye(3, dtype=bool))

    def test_drift_sparsity_fixed_sparse(self):
        """Sparse drift -> matches nonzero pattern."""
        A = jnp.array([[-0.5, 0.0, 0.2], [0.0, -0.3, 0.0], [0.1, 0.0, -0.8]])
        spec = SSMSpec(n_latent=3, n_manifest=3, drift=A)
        mask = compute_drift_sparsity(spec)
        expected = np.array([[True, False, True], [False, True, False], [True, False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_obs_dependency_free(self):
        """lambda="free" -> all-True."""
        spec = SSMSpec(n_latent=2, n_manifest=3, lambda_mat="free")
        dep = compute_obs_dependency(spec)
        assert dep.shape == (3, 2)
        assert dep.all()

    def test_obs_dependency_identity(self):
        """Identity lambda -> diagonal True."""
        spec = SSMSpec(n_latent=2, n_manifest=2, lambda_mat=jnp.eye(2))
        dep = compute_obs_dependency(spec)
        np.testing.assert_array_equal(dep, np.eye(2, dtype=bool))

    def test_obs_dependency_sparse(self):
        """Sparse lambda -> matches nonzero pattern."""
        lam = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        spec = SSMSpec(n_latent=2, n_manifest=3, lambda_mat=lam)
        dep = compute_obs_dependency(spec)
        expected = np.array([[True, False], [False, True], [True, True]])
        np.testing.assert_array_equal(dep, expected)

    def test_per_var_diffusion_scalar(self):
        """Single diffusion_dist broadcasts to all vars."""
        spec = SSMSpec(n_latent=3, n_manifest=3, diffusion_dist=NoiseFamily.STUDENT_T)
        result = get_per_variable_diffusion(spec)
        assert result == ["student_t", "student_t", "student_t"]

    def test_per_var_diffusion_list(self):
        """Per-var diffusion_dists returned as-is."""
        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T, NoiseFamily.GAUSSIAN],
        )
        result = get_per_variable_diffusion(spec)
        assert result == ["gaussian", "student_t", "gaussian"]

    def test_per_channel_manifest_scalar(self):
        """Single manifest_dist broadcasts to all channels."""
        spec = SSMSpec(n_latent=2, n_manifest=3, manifest_dist=NoiseFamily.POISSON)
        result = get_per_channel_manifest(spec)
        assert result == ["poisson", "poisson", "poisson"]


# =============================================================================
# Level 2: Partition Analysis
# =============================================================================


class TestAnalyzeFirstPassRB:
    """Test analyze_first_pass_rb for various model structures."""

    def test_all_gaussian_all_kalman(self):
        """All Gaussian dynamics + Gaussian obs -> everything in Kalman block."""
        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift=jnp.diag(jnp.array([-0.5, -0.3, -0.8])),
            lambda_mat=jnp.eye(3),
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        p = analyze_first_pass_rb(spec)
        np.testing.assert_array_equal(p.kalman_idx, [0, 1, 2])
        assert len(p.particle_idx) == 0
        np.testing.assert_array_equal(p.obs_kalman_idx, [0, 1, 2])
        assert len(p.obs_particle_idx) == 0
        assert p.has_kalman_block
        assert not p.has_particle_block

    def test_all_nongaussian_all_particle(self):
        """All Student-t dynamics -> everything in particle block."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=jnp.eye(2),
            diffusion_dist=NoiseFamily.STUDENT_T,
        )
        p = analyze_first_pass_rb(spec)
        assert len(p.kalman_idx) == 0
        np.testing.assert_array_equal(p.particle_idx, [0, 1])
        assert not p.has_kalman_block
        assert p.has_particle_block

    def test_independent_blocks_clean_split(self):
        """Block-diagonal drift with mixed dists -> clean split."""
        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        p = analyze_first_pass_rb(spec)
        np.testing.assert_array_equal(p.kalman_idx, [0, 1])
        np.testing.assert_array_equal(p.particle_idx, [2])
        np.testing.assert_array_equal(p.obs_kalman_idx, [0, 1])
        np.testing.assert_array_equal(p.obs_particle_idx, [2])

    def test_cross_coupling_prevents_split(self):
        """Full drift coupling -> no split possible."""
        n = 3
        drift = jnp.ones((n, n)) * 0.1
        drift = drift.at[jnp.diag_indices(n)].set(-0.5)
        spec = SSMSpec(
            n_latent=n,
            n_manifest=n,
            drift=drift,
            lambda_mat=jnp.eye(n),
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        # Gaussian vars are coupled to Student-t var -> all in particle
        assert len(p.kalman_idx) == 0
        np.testing.assert_array_equal(p.particle_idx, [0, 1, 2])

    def test_partial_split_3var(self):
        """2 Gaussian isolated + 1 Student-t -> 2 in Kalman."""
        # Drift: 3x3 with [0,1] block-diagonal, [2] separate
        drift = jnp.array(
            [
                [-0.5, 0.1, 0.0],
                [0.1, -0.3, 0.0],
                [0.0, 0.0, -0.8],
            ]
        )
        lam = jnp.eye(3)
        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift=drift,
            lambda_mat=lam,
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        np.testing.assert_array_equal(p.kalman_idx, [0, 1])
        np.testing.assert_array_equal(p.particle_idx, [2])

    def test_shared_obs_prevents_kalman(self):
        """Lambda couples G and S vars into shared obs -> no Kalman split."""
        # All 2 obs depend on all 2 latent vars
        lam = jnp.ones((2, 2))
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=lam,
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        # Var 0 is Gaussian and decoupled in drift, but all obs depend on both vars
        # -> no exclusive obs for var 0 -> obs_kalman is empty
        # However var 0 is still structurally decoupled in drift...
        # The key: Kalman filter needs its own observations. If no obs channels
        # are exclusive to the Kalman block, the Kalman filter gets no data.
        # The partition still identifies var 0 as Kalman-eligible based on drift,
        # but the obs channels go to particle since they depend on both.
        # This is correct: Kalman handles unobserved states, particle uses all obs.
        np.testing.assert_array_equal(p.kalman_idx, [0])
        np.testing.assert_array_equal(p.particle_idx, [1])
        assert len(p.obs_kalman_idx) == 0
        np.testing.assert_array_equal(p.obs_particle_idx, [0, 1])

    def test_free_drift_no_split(self):
        """drift="free" -> no structural zeros -> no split possible."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            lambda_mat=jnp.eye(2),
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        assert len(p.kalman_idx) == 0

    def test_free_lambda_no_split(self):
        """lambda="free" -> no exclusive obs channels -> no Kalman obs."""
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat="free",
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        # Var 0 is decoupled in drift, but lambda="free" means all obs depend on all vars
        np.testing.assert_array_equal(p.kalman_idx, [0])
        assert len(p.obs_kalman_idx) == 0  # no exclusive obs

    def test_g_feeds_s_prevents_split(self):
        """A[s,g] != 0 -> g must go to particle (S depends on G)."""
        drift = jnp.array([[-0.5, 0.0], [0.2, -0.3]])  # A[1,0] = 0.2: S <- G
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=drift,
            lambda_mat=jnp.eye(2),
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        assert len(p.kalman_idx) == 0

    def test_s_feeds_g_prevents_split(self):
        """A[g,s] != 0 -> g must go to particle (G depends on S)."""
        drift = jnp.array([[-0.5, 0.15], [0.0, -0.3]])  # A[0,1] = 0.15: G <- S
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=drift,
            lambda_mat=jnp.eye(2),
            diffusion_dists=[NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T],
        )
        p = analyze_first_pass_rb(spec)
        assert len(p.kalman_idx) == 0


# =============================================================================
# Level 3: Degenerate Equivalence
# =============================================================================


class TestDegenerateEquivalence:
    """All-Gaussian or all-particle should match simple backends."""

    def test_all_gaussian_matches_kalman(self):
        """All Gaussian -> ComposedLikelihood (all Kalman) == KalmanLikelihood."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        ct, meas, init = _make_separable_params(n_g=2, n_s=0, n_obs_g=2, n_obs_s=0)
        obs, dt = _simulate_separable_data(random.PRNGKey(42), ct, meas, init, T=20)

        # Direct Kalman
        kalman = KalmanLikelihood(n_latent=2, n_manifest=2)
        ll_kalman = kalman.compute_log_likelihood(ct, meas, init, obs, dt)

        # Via make_likelihood_backend with all-Gaussian spec
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=ct.drift,
            lambda_mat=meas.lambda_mat,
            diffusion=jnp.eye(2) * 0.3,
            manifest_var=jnp.eye(2) * 0.1,
            manifest_means=jnp.zeros(2),
            t0_means=jnp.zeros(2),
            t0_var=jnp.eye(2),
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        ll_composed = _run_composed(spec, ct, meas, init, obs, dt)

        assert jnp.isfinite(ll_kalman)
        assert jnp.isfinite(ll_composed)
        np.testing.assert_allclose(float(ll_composed), float(ll_kalman), atol=1e-3)

    def test_all_nongaussian_matches_particle(self):
        """All Student-t -> fallthrough to ParticleLikelihood (no composed wrapper)."""
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood

        ct, meas, init = _make_separable_params(n_g=0, n_s=2, n_obs_g=0, n_obs_s=2)
        obs, dt = _simulate_separable_data(random.PRNGKey(42), ct, meas, init, T=20)

        # Direct particle
        pf = ParticleLikelihood(
            n_latent=2,
            n_manifest=2,
            n_particles=300,
            rng_key=random.PRNGKey(42),
            diffusion_dist="student_t",
        )
        ll_pf = pf.compute_log_likelihood(
            ct,
            meas,
            init,
            obs,
            dt,
            extra_params={"proc_df": 100.0},
        )

        # Via make_likelihood_backend with all-Student-t spec
        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=ct.drift,
            lambda_mat=meas.lambda_mat,
            diffusion=jnp.eye(2) * 0.3,
            manifest_var=jnp.eye(2) * 0.1,
            manifest_means=jnp.zeros(2),
            t0_means=jnp.zeros(2),
            t0_var=jnp.eye(2),
            diffusion_dist=NoiseFamily.STUDENT_T,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        ll_composed = _run_composed(
            spec,
            ct,
            meas,
            init,
            obs,
            dt,
            n_particles=300,
            extra_params={"proc_df": 100.0},
        )

        assert jnp.isfinite(ll_pf)
        assert jnp.isfinite(ll_composed)
        # Both use PF so variance means they won't match exactly, but should be close
        np.testing.assert_allclose(float(ll_composed), float(ll_pf), atol=5.0)

    def test_trivial_partition_returns_simple_backend(self):
        """No composed wrapper when partition is trivial."""
        from dsem_agent.models.likelihoods.composed import ComposedLikelihood
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.ssm.model import SSMModel

        # All Gaussian -> should get KalmanLikelihood, not ComposedLikelihood
        spec_g = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=jnp.eye(2),
            diffusion_dist=NoiseFamily.GAUSSIAN,
        )
        model_g = SSMModel(spec=spec_g)
        backend_g = model_g.make_likelihood_backend()
        assert isinstance(backend_g, KalmanLikelihood)

        # All Student-t -> should get ParticleLikelihood, not ComposedLikelihood
        spec_s = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.diag(jnp.array([-0.5, -0.3])),
            lambda_mat=jnp.eye(2),
            diffusion_dist=NoiseFamily.STUDENT_T,
        )
        model_s = SSMModel(spec=spec_s)
        backend_s = model_s.make_likelihood_backend()
        assert isinstance(backend_s, ParticleLikelihood)
        assert not isinstance(backend_s, ComposedLikelihood)


# =============================================================================
# Level 4: Additive LL Decomposition
# =============================================================================


class TestAdditiveLLDecomposition:
    """For independent blocks, composed LL ≈ Kalman(G) + PF(S)."""

    def test_independent_2g_1s_additive(self):
        """LL(composed) ≈ LL(kalman on G-block) + LL(PF on S-block)."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood

        ct, meas, init = _make_separable_params(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        obs, dt = _simulate_separable_data(random.PRNGKey(123), ct, meas, init, T=25)

        # Composed via spec
        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        ll_composed = _run_composed(
            spec,
            ct,
            meas,
            init,
            obs,
            dt,
            n_particles=500,
            extra_params={"proc_df": 100.0},
        )

        # Kalman on G-block
        ct_g = CTParams(
            drift=ct.drift[:2, :2],
            diffusion_cov=ct.diffusion_cov[:2, :2],
            cint=ct.cint[:2],
        )
        meas_g = MeasurementParams(
            lambda_mat=meas.lambda_mat[:2, :2],
            manifest_means=meas.manifest_means[:2],
            manifest_cov=meas.manifest_cov[:2, :2],
        )
        init_g = InitialStateParams(mean=init.mean[:2], cov=init.cov[:2, :2])
        kalman = KalmanLikelihood(n_latent=2, n_manifest=2)
        ll_kalman = kalman.compute_log_likelihood(ct_g, meas_g, init_g, obs[:, :2], dt)

        # PF on S-block
        ct_s = CTParams(
            drift=ct.drift[2:, 2:],
            diffusion_cov=ct.diffusion_cov[2:, 2:],
            cint=ct.cint[2:],
        )
        meas_s = MeasurementParams(
            lambda_mat=meas.lambda_mat[2:, 2:],
            manifest_means=meas.manifest_means[2:],
            manifest_cov=meas.manifest_cov[2:, 2:],
        )
        init_s = InitialStateParams(mean=init.mean[2:], cov=init.cov[2:, 2:])
        pf = ParticleLikelihood(
            n_latent=1,
            n_manifest=1,
            n_particles=500,
            rng_key=random.PRNGKey(0),
            diffusion_dist="student_t",
        )
        ll_pf = pf.compute_log_likelihood(
            ct_s,
            meas_s,
            init_s,
            obs[:, 2:],
            dt,
            extra_params={"proc_df": 100.0},
        )

        ll_sum = float(ll_kalman) + float(ll_pf)
        assert jnp.isfinite(ll_composed)
        np.testing.assert_allclose(float(ll_composed), ll_sum, atol=5.0)

    def test_independent_1g_2s_additive(self):
        """Different dimensionality: 1G + 2S."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood

        ct, meas, init = _make_separable_params(n_g=1, n_s=2, n_obs_g=1, n_obs_s=2)
        obs, dt = _simulate_separable_data(random.PRNGKey(456), ct, meas, init, T=25)

        spec = _make_separable_spec(n_g=1, n_s=2, n_obs_g=1, n_obs_s=2)
        ll_composed = _run_composed(
            spec,
            ct,
            meas,
            init,
            obs,
            dt,
            n_particles=500,
            extra_params={"proc_df": 100.0},
        )

        # Kalman on G-block (1 var, 1 obs)
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
        kalman = KalmanLikelihood(n_latent=1, n_manifest=1)
        ll_kalman = kalman.compute_log_likelihood(ct_g, meas_g, init_g, obs[:, :1], dt)

        # PF on S-block (2 vars, 2 obs)
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
        pf = ParticleLikelihood(
            n_latent=2,
            n_manifest=2,
            n_particles=500,
            rng_key=random.PRNGKey(0),
            diffusion_dist="student_t",
        )
        ll_pf = pf.compute_log_likelihood(
            ct_s,
            meas_s,
            init_s,
            obs[:, 1:],
            dt,
            extra_params={"proc_df": 100.0},
        )

        ll_sum = float(ll_kalman) + float(ll_pf)
        assert jnp.isfinite(ll_composed)
        np.testing.assert_allclose(float(ll_composed), ll_sum, atol=5.0)


# =============================================================================
# Level 5: Pipeline Integration
# =============================================================================


class TestMakeLikelihoodBackend:
    """Test make_likelihood_backend dispatching."""

    def test_creates_composed_for_mixed_separable(self):
        """Mixed separable spec -> ComposedLikelihood."""
        from dsem_agent.models.likelihoods.composed import ComposedLikelihood
        from dsem_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ComposedLikelihood)

    def test_first_pass_rb_false_skips_analysis(self):
        """first_pass_rb=False -> no ComposedLikelihood, just ParticleLikelihood."""
        from dsem_agent.models.likelihoods.composed import ComposedLikelihood
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        # Override first_pass_rb
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            first_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        assert not isinstance(backend, ComposedLikelihood)

    def test_second_pass_rb_false_forces_bootstrap(self):
        """second_pass_rb=False -> ParticleLikelihood with block_rb=False."""
        from dsem_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        # Should still create composed (first pass is on) but particle sub-backend
        # has block_rb=False
        from dsem_agent.models.likelihoods.composed import ComposedLikelihood

        assert isinstance(backend, ComposedLikelihood)
        # The particle sub-backend should not use block RBPF
        assert not backend.particle_backend._block_rb

    def test_both_toggles_off_pure_bootstrap(self):
        """Both toggles off -> pure bootstrap PF."""
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            first_pass_rb=False,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        # Should not use Rao-Blackwellization at all
        assert backend.diffusion_dist != "mixed"
        assert not backend._block_rb

    def test_kalman_override_bypasses_analysis(self):
        """likelihood="kalman" bypasses first-pass analysis entirely."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec, likelihood="kalman")
        backend = model.make_likelihood_backend()
        assert isinstance(backend, KalmanLikelihood)


# =============================================================================
# Level 6: Variance Reduction
# =============================================================================


class TestVarianceReduction:
    """Two-level RB should have lower variance than pure PF."""

    @pytest.mark.slow
    def test_two_level_lower_variance(self):
        """Two-level composed LL has lower variance than pure bootstrap PF."""
        ct, meas, init = _make_separable_params(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        obs, dt = _simulate_separable_data(random.PRNGKey(789), ct, meas, init, T=25)
        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)

        n_runs = 20
        n_particles = 100

        # Two-level composed
        from dsem_agent.models.ssm.model import SSMModel

        composed_lls = []
        for i in range(n_runs):
            model = SSMModel(spec=spec, n_particles=n_particles, pf_seed=i)
            backend = model.make_likelihood_backend()
            ll = backend.compute_log_likelihood(
                ct,
                meas,
                init,
                obs,
                dt,
                extra_params={"proc_df": 100.0},
            )
            composed_lls.append(float(ll))

        # Pure bootstrap PF (no RB at all)
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood

        boot_lls = []
        for i in range(n_runs):
            pf = ParticleLikelihood(
                n_latent=3,
                n_manifest=3,
                n_particles=n_particles,
                rng_key=random.PRNGKey(i),
                diffusion_dist="student_t",
                block_rb=False,
            )
            ll = pf.compute_log_likelihood(
                ct,
                meas,
                init,
                obs,
                dt,
                extra_params={"proc_df": 100.0},
            )
            boot_lls.append(float(ll))

        var_composed = np.var(composed_lls)
        var_boot = np.var(boot_lls)
        assert var_composed < var_boot, (
            f"Composed variance ({var_composed:.4f}) should be < "
            f"bootstrap variance ({var_boot:.4f})"
        )


# =============================================================================
# Level 7: Gradient Flow
# =============================================================================


class TestGradientFlow:
    """jax.grad must flow through ComposedLikelihood."""

    def test_grad_through_composed(self):
        """Gradient of composed LL w.r.t. drift diagonal is finite."""
        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        ct, meas, init = _make_separable_params(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        obs, dt = _simulate_separable_data(random.PRNGKey(999), ct, meas, init, T=15)

        def ll_fn(drift_diag):
            n = 3
            drift = jnp.diag(-jnp.abs(drift_diag))
            ct_local = CTParams(
                drift=drift,
                diffusion_cov=jnp.eye(n) * 0.1,
                cint=jnp.zeros(n),
            )
            return _run_composed(
                spec,
                ct_local,
                meas,
                init,
                obs,
                dt,
                n_particles=50,
                extra_params={"proc_df": 100.0},
            )

        grad_fn = jax.grad(ll_fn)
        grad_val = grad_fn(jnp.array([0.5, 0.5, 0.5]))
        assert jnp.all(jnp.isfinite(grad_val)), f"Gradient not finite: {grad_val}"

    def test_grad_finite_all_params(self):
        """Gradients finite w.r.t. all CT params (drift diag + diffusion diag)."""
        spec = _make_separable_spec(n_g=1, n_s=1, n_obs_g=1, n_obs_s=1)
        ct, meas, init = _make_separable_params(n_g=1, n_s=1, n_obs_g=1, n_obs_s=1)
        obs, dt = _simulate_separable_data(random.PRNGKey(888), ct, meas, init, T=15)

        def ll_fn(params_vec):
            drift = jnp.diag(-jnp.abs(params_vec[:2]))
            diff_cov = jnp.diag(jnp.abs(params_vec[2:4]))
            ct_local = CTParams(drift=drift, diffusion_cov=diff_cov, cint=jnp.zeros(2))
            return _run_composed(
                spec,
                ct_local,
                meas,
                init,
                obs,
                dt,
                n_particles=50,
                extra_params={"proc_df": 100.0},
            )

        grad_fn = jax.grad(ll_fn)
        grad_val = grad_fn(jnp.array([0.5, 0.5, 0.1, 0.1]))
        assert jnp.all(jnp.isfinite(grad_val)), f"Gradient not finite: {grad_val}"


# =============================================================================
# Level 8: Parameter Recovery
# =============================================================================


class TestParameterRecovery:
    """SVI recovers drift params with two-level RB."""

    @pytest.mark.slow
    def test_recovery_two_level_rb(self):
        """Two-level RB + SVI recovers drift diagonal."""
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

        key = random.PRNGKey(1234)
        obs, dt = _simulate_separable_data(key, ct_true, meas, init, T=50)

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.3).expand([n]))
            drift = jnp.diag(-jnp.abs(drift_diag))
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from dsem_agent.models.ssm.model import SSMModel

            ssm_model = SSMModel(spec=spec, n_particles=100)
            backend = ssm_model.make_likelihood_backend()
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
        for _ in range(500):
            svi_state, _ = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        recovered = -jnp.abs(params["drift_diag_auto_loc"])

        np.testing.assert_allclose(
            np.sort(recovered),
            np.sort(np.array(true_drift_diag)),
            atol=0.35,
        )
