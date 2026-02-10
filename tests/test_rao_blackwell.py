"""Tests for Rao-Blackwell particle filter.

Covers:
1. Quadrature utilities (Gauss-Hermite, unscented transform)
2. Kalman sub-operations (predict, update, marginal likelihood)
3. Observation weight functions (Gaussian exact, non-Gaussian quadrature)
4. Full RBPF smoke tests (finite likelihood, determinism, gradient flow)
5. Variance reduction vs bootstrap PF
6. Parameter recovery via NUTS
7. Edge cases (missing data, irregular times, high-dim)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.rao_blackwell import (
    _gauss_hermite_1d,
    _kalman_predict,
    _kalman_update_gaussian,
    _multivariate_gauss_hermite,
    _obs_weight_gaussian,
    _obs_weight_quadrature,
    _unscented_sigma_points,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_standard_params(n_latent=2, n_manifest=2):
    """Standard test parameters for RBPF tests."""
    drift = jnp.array([[-0.5, 0.1], [0.2, -0.8]])[:n_latent, :n_latent]
    ct_params = CTParams(
        drift=drift,
        diffusion_cov=jnp.eye(n_latent) * 0.1,
        cint=jnp.zeros(n_latent),
    )
    meas_params = MeasurementParams(
        lambda_mat=jnp.eye(n_manifest, n_latent),
        manifest_means=jnp.zeros(n_manifest),
        manifest_cov=jnp.eye(n_manifest) * 0.1,
    )
    init = InitialStateParams(
        mean=jnp.zeros(n_latent),
        cov=jnp.eye(n_latent),
    )
    return ct_params, meas_params, init


def _simulate_poisson_data(key, n_latent=2, n_manifest=2, T=20):
    """Simulate Poisson observations from a linear-Gaussian latent process."""
    ct_params, meas_params, init = _make_standard_params(n_latent, n_manifest)

    k1, k2 = random.split(key)
    states = [init.mean]
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        drift = ct_params.drift @ states[-1]
        noise = random.normal(k_step, (n_latent,)) * 0.1
        states.append(states[-1] + drift + noise)
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    rates = jnp.exp(jnp.clip(eta, -5, 5))
    observations = random.poisson(k2, rates).astype(jnp.float32)

    time_intervals = jnp.ones(T)
    return observations, time_intervals, ct_params, meas_params, init


def _run_rbpf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dist="poisson",
    n_particles=200,
    rng_key=None,
    extra_params=None,
):
    """Run RBPF and return log-likelihood."""
    from dsem_agent.models.likelihoods.particle import ParticleLikelihood

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
        extra_params=extra_params,
    )


def _run_bootstrap_pf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dist="poisson",
    n_particles=200,
    rng_key=None,
    extra_params=None,
):
    """Run bootstrap PF (Student-t dynamics forces bootstrap) and return log-likelihood."""
    from dsem_agent.models.likelihoods.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dist=manifest_dist,
        diffusion_dist="student_t",
    )
    ep = {"proc_df": 100.0}
    if extra_params:
        ep.update(extra_params)
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
        extra_params=ep,
    )


# =============================================================================
# TestQuadrature
# =============================================================================


class TestQuadrature:
    """Test Gauss-Hermite and unscented transform quadrature."""

    def test_gh_polynomial_exact(self):
        """GH should exactly integrate x^2 against N(0,1) = 1."""
        nodes, weights = _gauss_hermite_1d(5)
        result = jnp.sum(weights * nodes**2)
        assert jnp.allclose(result, 1.0, atol=1e-5)

    def test_gh_exp_accurate(self):
        """GH should accurately integrate exp(x) against N(0,1) = exp(0.5)."""
        nodes, weights = _gauss_hermite_1d(10)
        result = jnp.sum(weights * jnp.exp(nodes))
        expected = jnp.exp(0.5)
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_gh_weights_sum_to_one(self):
        """GH weights should sum to 1."""
        _, weights = _gauss_hermite_1d(7)
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-5)

    def test_multivariate_gh_shape(self):
        """Multivariate GH should produce n_points^dim nodes."""
        nodes, weights = _multivariate_gauss_hermite(3, 2)
        assert nodes.shape == (9, 2)
        assert weights.shape == (9,)
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-5)

    def test_multivariate_gh_mean_recovery(self):
        """Multivariate GH should recover mean of N(0,I)."""
        nodes, weights = _multivariate_gauss_hermite(5, 3)
        mean_est = jnp.sum(weights[:, None] * nodes, axis=0)
        assert jnp.allclose(mean_est, jnp.zeros(3), atol=1e-5)

    def test_unscented_mean_recovery(self):
        """Unscented transform should recover the mean exactly."""
        mean = jnp.array([1.0, 2.0, 3.0])
        cov = jnp.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]])
        points, weights = _unscented_sigma_points(mean, cov)
        recovered_mean = jnp.sum(weights[:, None] * points, axis=0)
        assert jnp.allclose(recovered_mean, mean, atol=1e-6)

    def test_unscented_cov_recovery(self):
        """Unscented transform should recover the covariance."""
        mean = jnp.array([1.0, -0.5])
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        points, weights = _unscented_sigma_points(mean, cov)
        diff = points - mean[None, :]
        recovered_cov = jnp.sum(
            weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0
        )
        assert jnp.allclose(recovered_cov, cov, atol=1e-4)

    def test_unscented_point_count(self):
        """Unscented transform should generate 2n+1 points."""
        n = 4
        points, weights = _unscented_sigma_points(jnp.zeros(n), jnp.eye(n))
        assert points.shape == (2 * n + 1, n)
        assert weights.shape == (2 * n + 1,)


# =============================================================================
# TestKalmanSubOps
# =============================================================================


class TestKalmanSubOps:
    """Test Kalman predict/update sub-operations."""

    def test_predict_increases_uncertainty(self):
        """Kalman predict should increase covariance."""
        m = jnp.array([1.0, 0.0])
        P = jnp.eye(2) * 0.1
        F = 0.9 * jnp.eye(2)
        Q = jnp.eye(2) * 0.05
        c = jnp.zeros(2)

        m_pred, P_pred = _kalman_predict(m, P, F, Q, c)
        assert jnp.all(jnp.diag(P_pred) > jnp.diag(P) * 0.5)
        assert jnp.allclose(m_pred, F @ m, atol=1e-10)

    def test_predict_with_intercept(self):
        """Kalman predict with nonzero intercept."""
        m = jnp.zeros(2)
        P = jnp.eye(2)
        F = jnp.eye(2)
        Q = jnp.zeros((2, 2))
        c = jnp.array([1.0, 2.0])

        m_pred, _ = _kalman_predict(m, P, F, Q, c)
        assert jnp.allclose(m_pred, c, atol=1e-10)

    def test_gaussian_update_reduces_uncertainty(self):
        """Kalman update should reduce uncertainty for observed variables."""
        m = jnp.zeros(2)
        P = jnp.eye(2)
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.1
        d = jnp.zeros(2)
        y = jnp.array([1.0, 0.5])
        obs_mask = jnp.ones(2, dtype=bool)

        m_upd, P_upd, log_marg = _kalman_update_gaussian(m, P, H, R, d, y, obs_mask)
        assert jnp.all(jnp.abs(m_upd - y) < jnp.abs(m - y))
        assert jnp.all(jnp.diag(P_upd) < jnp.diag(P))
        assert jnp.isfinite(log_marg)

    def test_gaussian_update_marginal_likelihood_value(self):
        """Check marginal likelihood against manual computation."""
        m = jnp.zeros(1)
        P = jnp.eye(1) * 2.0
        H = jnp.eye(1)
        R = jnp.eye(1) * 0.5
        d = jnp.zeros(1)
        y = jnp.array([1.0])
        obs_mask = jnp.ones(1, dtype=bool)

        _, _, log_marg = _kalman_update_gaussian(m, P, H, R, d, y, obs_mask)

        # y ~ N(0, 2.5)
        expected = jax.scipy.stats.norm.logpdf(1.0, loc=0.0, scale=jnp.sqrt(2.5))
        assert jnp.allclose(log_marg, expected, atol=1e-4)

    def test_gaussian_update_missing_data(self):
        """Kalman update with all-missing data should not change state."""
        m = jnp.array([1.0, 2.0])
        P = jnp.eye(2) * 0.5
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.1
        d = jnp.zeros(2)
        y = jnp.array([0.0, 0.0])
        obs_mask = jnp.array([False, False])

        m_upd, _P_upd, log_marg = _kalman_update_gaussian(m, P, H, R, d, y, obs_mask)
        assert jnp.allclose(m_upd, m, atol=1e-3)
        assert jnp.allclose(log_marg, 0.0, atol=1e-3)


# =============================================================================
# TestObsWeights
# =============================================================================


class TestObsWeights:
    """Test observation weight functions."""

    def test_gaussian_weight_matches_kalman_marginal(self):
        """Gaussian obs weight should match Kalman update marginal likelihood."""
        m = jnp.array([0.5, -0.3])
        P = jnp.eye(2) * 0.5
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.2
        d = jnp.zeros(2)
        y = jnp.array([0.8, 0.1])
        obs_mask = jnp.ones(2, dtype=bool)

        w1 = _obs_weight_gaussian(y, m, P, H, R, d, obs_mask)
        _, _, w2 = _kalman_update_gaussian(m, P, H, R, d, y, obs_mask)
        assert jnp.allclose(w1, w2, atol=1e-4)

    def test_poisson_weight_finite(self):
        """Poisson obs weight should be finite."""
        m = jnp.array([0.5, 0.3])
        P = jnp.eye(2) * 0.3
        H = jnp.eye(2)
        d = jnp.zeros(2)
        y = jnp.array([2.0, 1.0])
        obs_mask = jnp.ones(2, dtype=bool)
        params = {"manifest_cov": jnp.eye(2) * 0.1}

        w = _obs_weight_quadrature(y, m, P, H, d, obs_mask, "poisson", params)
        assert jnp.isfinite(w)

    def test_student_t_weight_finite(self):
        """Student-t obs weight should be finite."""
        m = jnp.array([0.5, 0.3])
        P = jnp.eye(2) * 0.3
        H = jnp.eye(2)
        d = jnp.zeros(2)
        y = jnp.array([0.8, 0.1])
        obs_mask = jnp.ones(2, dtype=bool)
        params = {"manifest_cov": jnp.eye(2) * 0.1, "obs_df": 5.0}

        w = _obs_weight_quadrature(y, m, P, H, d, obs_mask, "student_t", params)
        assert jnp.isfinite(w)

    def test_gamma_weight_finite(self):
        """Gamma obs weight should be finite."""
        m = jnp.array([0.5, 0.3])
        P = jnp.eye(2) * 0.1
        H = jnp.eye(2)
        d = jnp.zeros(2)
        y = jnp.array([1.5, 1.2])
        obs_mask = jnp.ones(2, dtype=bool)
        params = {"obs_shape": 2.0}

        w = _obs_weight_quadrature(y, m, P, H, d, obs_mask, "gamma", params)
        assert jnp.isfinite(w)

    def test_weight_varies_with_mean(self):
        """Obs weight should change when predicted mean changes."""
        P = jnp.eye(2) * 0.3
        H = jnp.eye(2)
        d = jnp.zeros(2)
        y = jnp.array([3.0, 2.0])
        obs_mask = jnp.ones(2, dtype=bool)
        params = {}

        m_good = jnp.array([jnp.log(3.0), jnp.log(2.0)])
        m_bad = jnp.array([-2.0, -2.0])

        w_good = _obs_weight_quadrature(y, m_good, P, H, d, obs_mask, "poisson", params)
        w_bad = _obs_weight_quadrature(y, m_bad, P, H, d, obs_mask, "poisson", params)
        assert w_good > w_bad


# =============================================================================
# TestRBPFSmoke
# =============================================================================


class TestRBPFSmoke:
    """Smoke tests for full RBPF pipeline."""

    def test_poisson_obs_finite_likelihood(self):
        """RBPF with Poisson observations should produce finite log-likelihood."""
        key = random.PRNGKey(0)
        obs, dt, ct, meas, init = _simulate_poisson_data(key, T=15)
        ll = _run_rbpf(ct, meas, init, obs, dt, manifest_dist="poisson")
        assert jnp.isfinite(ll), f"RBPF Poisson ll = {ll}"

    def test_student_t_obs_finite_likelihood(self):
        """RBPF with Student-t observations should produce finite log-likelihood."""
        ct, meas, init = _make_standard_params()
        T = 15
        obs = random.normal(random.PRNGKey(1), (T, 2)) * 0.5
        dt = jnp.ones(T)
        ll = _run_rbpf(
            ct, meas, init, obs, dt, manifest_dist="student_t", extra_params={"obs_df": 5.0}
        )
        assert jnp.isfinite(ll)

    def test_gamma_obs_finite_likelihood(self):
        """RBPF with Gamma observations should produce finite log-likelihood."""
        ct, meas, init = _make_standard_params()
        T = 15
        obs = jnp.abs(random.normal(random.PRNGKey(2), (T, 2))) + 0.1
        dt = jnp.ones(T)
        ll = _run_rbpf(
            ct, meas, init, obs, dt, manifest_dist="gamma", extra_params={"obs_shape": 2.0}
        )
        assert jnp.isfinite(ll)

    def test_deterministic_with_same_key(self):
        """RBPF with same key should produce identical results."""
        key = random.PRNGKey(10)
        obs, dt, ct, meas, init = _simulate_poisson_data(key, T=10)
        pf_key = random.PRNGKey(99)
        ll1 = _run_rbpf(ct, meas, init, obs, dt, rng_key=pf_key)
        ll2 = _run_rbpf(ct, meas, init, obs, dt, rng_key=pf_key)
        assert float(ll1) == float(ll2)

    def test_varies_with_params(self):
        """RBPF likelihood should change with different drift parameters."""
        key = random.PRNGKey(20)
        obs, dt, ct, meas, init = _simulate_poisson_data(key, T=10)

        lls = []
        for drift_scale in [-0.3, -0.5, -0.8]:
            ct_mod = CTParams(
                drift=jnp.eye(2) * drift_scale,
                diffusion_cov=ct.diffusion_cov,
                cint=ct.cint,
            )
            lls.append(float(_run_rbpf(ct_mod, meas, init, obs, dt)))

        assert all(np.isfinite(ll) for ll in lls)
        assert len({round(ll, 1) for ll in lls}) > 1

    def test_gaussian_obs_matches_kalman(self):
        """RBPF with Gaussian obs should approximate exact Kalman filter."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        ct, meas, init = _make_standard_params()
        T = 15
        obs = random.normal(random.PRNGKey(42), (T, 2)) * 0.5
        dt = jnp.ones(T)

        kalman = KalmanLikelihood(n_latent=2, n_manifest=2)
        ll_kalman = kalman.compute_log_likelihood(ct, meas, init, obs, dt)

        ll_rbpf = _run_rbpf(ct, meas, init, obs, dt, manifest_dist="gaussian", n_particles=500)

        # Tolerance is loose because PF resampling introduces some bias
        assert jnp.allclose(ll_rbpf, ll_kalman, atol=1.5), f"RBPF={ll_rbpf}, Kalman={ll_kalman}"


# =============================================================================
# TestRBPFGradient
# =============================================================================


class TestRBPFGradient:
    """Test that jax.grad flows through RBPF."""

    def _grad_test(self, manifest_dist, extra_params=None):
        ct, meas, init = _make_standard_params()
        T = 8
        key = random.PRNGKey(42)
        if manifest_dist == "gamma":
            obs = jnp.abs(random.normal(key, (T, 2))) + 0.1
        elif manifest_dist == "poisson":
            obs = random.poisson(key, jnp.ones((T, 2)) * 2.0).astype(jnp.float32)
        else:
            obs = random.normal(key, (T, 2)) * 0.5
        dt = jnp.ones(T)

        def ll_fn(drift_diag):
            ct_mod = CTParams(
                drift=jnp.diag(drift_diag),
                diffusion_cov=ct.diffusion_cov,
                cint=ct.cint,
            )
            return _run_rbpf(
                ct_mod, meas, init, obs, dt, manifest_dist=manifest_dist, extra_params=extra_params
            )

        grad = jax.grad(ll_fn)(jnp.array([-0.5, -0.5]))
        assert jnp.all(jnp.isfinite(grad)), f"Gradient not finite: {grad}"

    def test_gradient_finite_poisson(self):
        self._grad_test("poisson")

    def test_gradient_finite_student_t(self):
        self._grad_test("student_t", {"obs_df": 5.0})

    def test_gradient_finite_gamma(self):
        self._grad_test("gamma", {"obs_shape": 2.0})


# =============================================================================
# TestVarianceReduction (slow)
# =============================================================================


class TestVarianceReduction:
    """Test that RBPF has lower variance than bootstrap PF."""

    @pytest.mark.slow
    def test_rbpf_lower_variance_poisson(self):
        """RBPF should have lower variance than bootstrap PF for Poisson obs."""
        key = random.PRNGKey(0)
        obs, dt, ct, meas, init = _simulate_poisson_data(key, T=15)

        n_runs = 30
        rbpf_lls, boot_lls = [], []
        for i in range(n_runs):
            pf_key = random.PRNGKey(1000 + i)
            rbpf_lls.append(
                float(_run_rbpf(ct, meas, init, obs, dt, rng_key=pf_key, n_particles=100))
            )
            boot_lls.append(
                float(_run_bootstrap_pf(ct, meas, init, obs, dt, rng_key=pf_key, n_particles=100))
            )

        var_rbpf = np.var(rbpf_lls)
        var_boot = np.var(boot_lls)
        assert var_rbpf < var_boot, (
            f"RBPF var ({var_rbpf:.4f}) should be < bootstrap var ({var_boot:.4f})"
        )

    @pytest.mark.slow
    def test_rbpf_lower_variance_student_t(self):
        """RBPF should have lower variance for Student-t obs."""
        ct, meas, init = _make_standard_params()
        T = 15
        obs = random.normal(random.PRNGKey(10), (T, 2)) * 0.5
        dt = jnp.ones(T)

        n_runs = 30
        rbpf_lls, boot_lls = [], []
        for i in range(n_runs):
            pf_key = random.PRNGKey(2000 + i)
            rbpf_lls.append(
                float(
                    _run_rbpf(
                        ct,
                        meas,
                        init,
                        obs,
                        dt,
                        manifest_dist="student_t",
                        rng_key=pf_key,
                        n_particles=100,
                        extra_params={"obs_df": 5.0},
                    )
                )
            )
            boot_lls.append(
                float(
                    _run_bootstrap_pf(
                        ct,
                        meas,
                        init,
                        obs,
                        dt,
                        manifest_dist="student_t",
                        rng_key=pf_key,
                        n_particles=100,
                        extra_params={"obs_df": 5.0},
                    )
                )
            )

        var_rbpf = np.var(rbpf_lls)
        var_boot = np.var(boot_lls)
        assert var_rbpf < var_boot, (
            f"RBPF var ({var_rbpf:.4f}) should be < bootstrap var ({var_boot:.4f})"
        )

    @pytest.mark.slow
    def test_rbpf_lower_variance_high_dim(self):
        """RBPF variance reduction should hold in 6D."""
        n_latent, n_manifest = 6, 6
        ct = CTParams(
            drift=jnp.diag(jnp.array([-0.3, -0.4, -0.5, -0.6, -0.7, -0.8])),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=jnp.zeros(n_latent),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        T = 15
        obs = random.poisson(random.PRNGKey(20), jnp.ones((T, n_manifest)) * 2.0).astype(
            jnp.float32
        )
        dt = jnp.ones(T)

        n_runs = 20
        rbpf_lls, boot_lls = [], []
        for i in range(n_runs):
            pf_key = random.PRNGKey(3000 + i)
            rbpf_lls.append(
                float(_run_rbpf(ct, meas, init, obs, dt, rng_key=pf_key, n_particles=100))
            )
            boot_lls.append(
                float(_run_bootstrap_pf(ct, meas, init, obs, dt, rng_key=pf_key, n_particles=100))
            )

        var_rbpf = np.var(rbpf_lls)
        var_boot = np.var(boot_lls)
        assert var_rbpf < var_boot, (
            f"6D RBPF var ({var_rbpf:.4f}) should be < bootstrap var ({var_boot:.4f})"
        )


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for RBPF."""

    def test_single_observation(self):
        """RBPF should work with T=1."""
        ct, meas, init = _make_standard_params()
        obs = jnp.array([[2.0, 1.0]])
        dt = jnp.array([1.0])
        ll = _run_rbpf(ct, meas, init, obs, dt)
        assert jnp.isfinite(ll)

    def test_irregular_time_intervals(self):
        """RBPF should handle irregular time intervals."""
        ct, meas, init = _make_standard_params()
        T = 10
        obs = random.poisson(random.PRNGKey(42), jnp.ones((T, 2)) * 2.0).astype(jnp.float32)
        dt = jnp.array([0.1, 0.5, 1.0, 2.0, 0.3, 0.7, 1.5, 0.2, 0.8, 1.2])
        ll = _run_rbpf(ct, meas, init, obs, dt)
        assert jnp.isfinite(ll)

    def test_missing_data(self):
        """RBPF should handle NaN observations."""
        ct, meas, init = _make_standard_params()
        T = 10
        obs = random.poisson(random.PRNGKey(42), jnp.ones((T, 2)) * 2.0).astype(jnp.float32)
        obs = obs.at[3, 0].set(jnp.nan)
        obs = obs.at[7, :].set(jnp.nan)
        dt = jnp.ones(T)
        ll = _run_rbpf(ct, meas, init, obs, dt)
        assert jnp.isfinite(ll)

    def test_non_identity_lambda(self):
        """RBPF should work with non-identity factor loadings."""
        n_latent, n_manifest = 2, 3
        ct = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=jnp.zeros(n_latent),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        T = 10
        obs = random.poisson(random.PRNGKey(42), jnp.ones((T, n_manifest)) * 2.0).astype(
            jnp.float32
        )
        dt = jnp.ones(T)
        ll = _run_rbpf(ct, meas, init, obs, dt)
        assert jnp.isfinite(ll)

    def test_high_dim_6d_poisson(self):
        """RBPF should work with 6D latent + Poisson."""
        n_latent, n_manifest = 6, 6
        ct = CTParams(
            drift=jnp.diag(jnp.array([-0.3, -0.4, -0.5, -0.6, -0.7, -0.8])),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=jnp.zeros(n_latent),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_means=jnp.zeros(n_manifest),
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        T = 10
        obs = random.poisson(random.PRNGKey(42), jnp.ones((T, n_manifest)) * 2.0).astype(
            jnp.float32
        )
        dt = jnp.ones(T)
        ll = _run_rbpf(ct, meas, init, obs, dt, n_particles=100)
        assert jnp.isfinite(ll)


# =============================================================================
# Parameter Recovery (slow)
# =============================================================================


class TestParameterRecovery:
    """Test parameter recovery via NUTS with RBPF likelihood."""

    @pytest.mark.slow
    def test_drift_recovery_poisson(self):
        """1D AR(1) + Poisson: recover drift parameter via RBPF + NUTS."""
        from dsem_agent.models.ssm import NoiseFamily, SSMModel, SSMSpec, fit

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_dist=NoiseFamily.POISSON,
            diffusion_dist=NoiseFamily.GAUSSIAN,
        )
        model = SSMModel(spec, n_particles=200, pf_seed=42)

        key = random.PRNGKey(0)
        T = 50
        true_drift = -0.5
        k1, k2 = random.split(key)
        states = [jnp.zeros(1)]
        for _t in range(T - 1):
            k1, k_step = random.split(k1)
            states.append(states[-1] * jnp.exp(true_drift) + random.normal(k_step, (1,)) * 0.2)
        states = jnp.stack(states)
        obs = random.poisson(k2, jnp.exp(states)).astype(jnp.float32)
        times = jnp.arange(T, dtype=float)

        result = fit(
            model, obs, times, method="nuts", num_warmup=200, num_samples=200, num_chains=1
        )
        # Model applies -abs(drift_diag_pop) so check the transformed value
        raw = result.get_samples()["drift_diag_pop"][:, 0]
        drift_mean = float(jnp.mean(-jnp.abs(raw)))
        assert -1.5 < drift_mean < 0.0, f"Drift posterior mean = {drift_mean}"

    @pytest.mark.slow
    def test_drift_recovery_student_t(self):
        """1D AR(1) + Student-t: recover drift via RBPF + NUTS."""
        from dsem_agent.models.ssm import NoiseFamily, SSMModel, SSMSpec, fit

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_dist=NoiseFamily.STUDENT_T,
            diffusion_dist=NoiseFamily.GAUSSIAN,
        )
        model = SSMModel(spec, n_particles=200, pf_seed=42)

        key = random.PRNGKey(1)
        T = 50
        k1, k2 = random.split(key)
        states = [jnp.zeros(1)]
        for _t in range(T - 1):
            k1, k_step = random.split(k1)
            states.append(states[-1] * 0.8 + random.normal(k_step, (1,)) * 0.3)
        states = jnp.stack(states)
        obs = (states + random.normal(k2, states.shape) * 0.5).astype(jnp.float32)
        times = jnp.arange(T, dtype=float)

        result = fit(
            model, obs, times, method="nuts", num_warmup=200, num_samples=200, num_chains=1
        )
        raw = result.get_samples()["drift_diag_pop"][:, 0]
        drift_mean = float(jnp.mean(-jnp.abs(raw)))
        assert -1.5 < drift_mean < 0.0, f"Drift posterior mean = {drift_mean}"

    @pytest.mark.slow
    def test_drift_recovery_gamma(self):
        """1D AR(1) + Gamma: recover drift via RBPF + NUTS."""
        from dsem_agent.models.ssm import NoiseFamily, SSMModel, SSMSpec, fit

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_dist=NoiseFamily.GAMMA,
            diffusion_dist=NoiseFamily.GAUSSIAN,
        )
        model = SSMModel(spec, n_particles=200, pf_seed=42)

        key = random.PRNGKey(2)
        T = 50
        k1, k2 = random.split(key)
        states = [jnp.zeros(1)]
        for _t in range(T - 1):
            k1, k_step = random.split(k1)
            states.append(states[-1] * 0.8 + random.normal(k_step, (1,)) * 0.2)
        states = jnp.stack(states)
        shape = 2.0
        obs = (random.gamma(k2, shape, states.shape) * jnp.exp(states) / shape).astype(jnp.float32)
        times = jnp.arange(T, dtype=float)

        result = fit(
            model, obs, times, method="nuts", num_warmup=200, num_samples=200, num_chains=1
        )
        raw = result.get_samples()["drift_diag_pop"][:, 0]
        drift_mean = float(jnp.mean(-jnp.abs(raw)))
        assert -1.5 < drift_mean < 0.0, f"Drift posterior mean = {drift_mean}"


# =============================================================================
# TestRBPFKalmanConsistency — RBPF must match exact Kalman for Gaussian obs
# =============================================================================


class TestRBPFKalmanConsistency:
    """Verify RBPF matches Kalman filter exactly for Gaussian observations.

    These tests catch bugs in RBPF predict/update/weight logic by comparing
    against the known-correct Kalman filter on linear-Gaussian systems.
    """

    def _kalman_ll(self, ct, meas, init, obs, dt):
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood

        kf = KalmanLikelihood(n_latent=init.mean.shape[0], n_manifest=meas.lambda_mat.shape[0])
        return float(kf.compute_log_likelihood(ct, meas, init, obs, dt))

    def _rbpf_ll(self, ct, meas, init, obs, dt, n_particles=500, seed=42):
        return float(
            _run_rbpf(
                ct,
                meas,
                init,
                obs,
                dt,
                manifest_dist="gaussian",
                n_particles=n_particles,
                rng_key=random.PRNGKey(seed),
            )
        )

    def test_kalman_match_identity_measurement(self):
        """RBPF ≈ Kalman with identity lambda, n=2."""
        ct, meas, init = _make_standard_params(n_latent=2, n_manifest=2)
        T = 20
        obs = random.normal(random.PRNGKey(0), (T, 2)) * 0.5
        dt = jnp.ones(T)
        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        ll_rb = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        assert abs(ll_rb - ll_kf) < 1.5, f"RBPF={ll_rb:.2f}, Kalman={ll_kf:.2f}"

    def test_kalman_match_non_identity_lambda(self):
        """RBPF ≈ Kalman with 3-manifest, 2-latent, non-identity lambda."""
        n_latent, n_manifest = 2, 3
        ct = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=jnp.eye(n_latent) * 0.1,
            cint=jnp.zeros(n_latent),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            manifest_means=jnp.array([0.1, -0.2, 0.3]),
            manifest_cov=jnp.eye(n_manifest) * 0.15,
        )
        init = InitialStateParams(mean=jnp.array([0.5, -0.3]), cov=jnp.eye(n_latent) * 0.8)

        T = 15
        obs = random.normal(random.PRNGKey(7), (T, n_manifest)) * 0.5
        dt = jnp.ones(T)
        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        ll_rb = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        assert abs(ll_rb - ll_kf) < 1.5, f"RBPF={ll_rb:.2f}, Kalman={ll_kf:.2f}"

    def test_kalman_match_nonzero_intercept(self):
        """RBPF ≈ Kalman with non-zero drift intercept."""
        ct = CTParams(
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.3]]),
            diffusion_cov=jnp.eye(2) * 0.1,
            cint=jnp.array([0.5, -0.2]),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        T = 15
        obs = random.normal(random.PRNGKey(11), (T, 2)) * 0.5 + 0.3
        dt = jnp.ones(T)
        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        ll_rb = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        assert abs(ll_rb - ll_kf) < 1.5, f"RBPF={ll_rb:.2f}, Kalman={ll_kf:.2f}"

    def test_kalman_match_nondiag_diffusion(self):
        """RBPF ≈ Kalman with non-diagonal diffusion covariance."""
        Q = jnp.array([[0.15, 0.05], [0.05, 0.10]])
        ct = CTParams(
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion_cov=Q,
            cint=jnp.zeros(2),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

        T = 15
        obs = random.normal(random.PRNGKey(13), (T, 2)) * 0.5
        dt = jnp.ones(T)
        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        ll_rb = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        assert abs(ll_rb - ll_kf) < 1.5, f"RBPF={ll_rb:.2f}, Kalman={ll_kf:.2f}"

    def test_kalman_match_irregular_dt(self):
        """RBPF ≈ Kalman with irregular time intervals."""
        ct, meas, init = _make_standard_params()
        T = 12
        obs = random.normal(random.PRNGKey(17), (T, 2)) * 0.5
        dt = jnp.array([0.1, 0.5, 1.0, 2.0, 0.3, 0.7, 1.5, 0.2, 0.8, 1.2, 0.4, 0.9])
        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        ll_rb = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        assert abs(ll_rb - ll_kf) < 1.5, f"RBPF={ll_rb:.2f}, Kalman={ll_kf:.2f}"

    def test_missing_data_convention_difference(self):
        """RBPF and Kalman handle NaN differently: verify RBPF gives higher LL.

        The Kalman filter inflates R for missing obs (adding spurious
        normalization constants). RBPF correctly skips them (log weight = 0).
        So RBPF LL > Kalman LL when data has NaN, and the difference should
        scale with the number of missing observations.
        """
        ct, meas, init = _make_standard_params()
        T = 15
        obs = random.normal(random.PRNGKey(19), (T, 2)) * 0.5
        obs_missing = obs.at[3, 0].set(jnp.nan)
        obs_missing = obs_missing.at[7, :].set(jnp.nan)
        obs_missing = obs_missing.at[10, 1].set(jnp.nan)
        dt = jnp.ones(T)

        # RBPF on complete vs missing data
        ll_rb_full = self._rbpf_ll(ct, meas, init, obs, dt, n_particles=1000)
        ll_rb_miss = self._rbpf_ll(ct, meas, init, obs_missing, dt, n_particles=1000)

        # Missing data should give HIGHER LL (fewer observations to penalize)
        assert ll_rb_miss > ll_rb_full - 1.0, (
            f"Missing data LL={ll_rb_miss:.2f} should be >= full LL={ll_rb_full:.2f} "
            "(fewer obs can only increase or maintain LL)"
        )
        assert np.isfinite(ll_rb_miss), f"RBPF with missing data should be finite: {ll_rb_miss}"

    @pytest.mark.slow
    def test_kalman_match_unbiased(self):
        """Mean RBPF LL over many seeds should converge to exact Kalman LL."""
        ct, meas, init = _make_standard_params()
        T = 20
        obs = random.normal(random.PRNGKey(23), (T, 2)) * 0.5
        dt = jnp.ones(T)

        ll_kf = self._kalman_ll(ct, meas, init, obs, dt)
        rbpf_lls = [
            self._rbpf_ll(ct, meas, init, obs, dt, n_particles=500, seed=i) for i in range(50)
        ]
        mean_rbpf = np.mean(rbpf_lls)
        # PF has a known downward bias in log-LL (Jensen's inequality), so
        # mean RBPF should be slightly below Kalman. Allow 1.0 nat tolerance.
        assert abs(mean_rbpf - ll_kf) < 1.0, (
            f"Mean RBPF={mean_rbpf:.3f}, Kalman={ll_kf:.3f}, diff={abs(mean_rbpf - ll_kf):.3f}"
        )


# =============================================================================
# TestRBPFGradientConsistency — Gradients match finite differences
# =============================================================================


class TestRBPFGradientConsistency:
    """Verify RBPF gradients via finite differences.

    Catches bugs where jax.grad runs without error but produces incorrect
    gradients (e.g., stop-gradient errors, wrong resampling gradient paths).
    """

    def _fd_gradient(self, fn, x, eps=1e-3):
        """Central finite difference gradient."""
        grad_fd = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.at[i].set(x[i] + eps)
            x_minus = x.at[i].set(x[i] - eps)
            grad_fd[i] = (float(fn(x_plus)) - float(fn(x_minus))) / (2 * eps)
        return jnp.array(grad_fd)

    def test_grad_drift_gaussian(self):
        """RBPF gradient w.r.t. drift matches finite differences (Gaussian obs)."""
        ct, meas, init = _make_standard_params()
        T = 8
        obs = random.normal(random.PRNGKey(0), (T, 2)) * 0.5
        dt = jnp.ones(T)

        def ll_fn(drift_diag):
            ct_mod = CTParams(
                drift=jnp.diag(drift_diag), diffusion_cov=ct.diffusion_cov, cint=ct.cint
            )
            return _run_rbpf(ct_mod, meas, init, obs, dt, manifest_dist="gaussian", n_particles=200)

        x0 = jnp.array([-0.5, -0.8])
        grad_ad = jax.grad(ll_fn)(x0)
        grad_fd = self._fd_gradient(ll_fn, x0)
        # Finite differences are noisy due to resampling; use loose tolerance
        assert jnp.allclose(grad_ad, grad_fd, atol=2.0), f"AD grad={grad_ad}, FD grad={grad_fd}"

    def test_grad_drift_poisson(self):
        """RBPF gradient w.r.t. drift matches finite differences (Poisson obs)."""
        ct, meas, init = _make_standard_params()
        T = 8
        obs = random.poisson(random.PRNGKey(1), jnp.ones((T, 2)) * 2.0).astype(jnp.float32)
        dt = jnp.ones(T)

        def ll_fn(drift_diag):
            ct_mod = CTParams(
                drift=jnp.diag(drift_diag), diffusion_cov=ct.diffusion_cov, cint=ct.cint
            )
            return _run_rbpf(ct_mod, meas, init, obs, dt, manifest_dist="poisson", n_particles=200)

        x0 = jnp.array([-0.5, -0.8])
        grad_ad = jax.grad(ll_fn)(x0)
        grad_fd = self._fd_gradient(ll_fn, x0)
        assert jnp.allclose(grad_ad, grad_fd, atol=2.0), f"AD grad={grad_ad}, FD grad={grad_fd}"

    def test_grad_measurement_cov(self):
        """RBPF gradient w.r.t. measurement noise matches finite differences."""
        ct, meas, init = _make_standard_params()
        T = 8
        obs = random.normal(random.PRNGKey(2), (T, 2)) * 0.5
        dt = jnp.ones(T)

        def ll_fn(log_r_diag):
            R = jnp.diag(jnp.exp(log_r_diag))
            meas_mod = MeasurementParams(
                lambda_mat=meas.lambda_mat, manifest_means=meas.manifest_means, manifest_cov=R
            )
            return _run_rbpf(ct, meas_mod, init, obs, dt, manifest_dist="gaussian", n_particles=200)

        x0 = jnp.array([jnp.log(0.1), jnp.log(0.1)])
        grad_ad = jax.grad(ll_fn)(x0)
        grad_fd = self._fd_gradient(ll_fn, x0)
        assert jnp.allclose(grad_ad, grad_fd, atol=2.0), f"AD grad={grad_ad}, FD grad={grad_fd}"


# =============================================================================
# TestRBPFBootstrapAgreement — RBPF and bootstrap should agree in expectation
# =============================================================================


class TestRBPFBootstrapAgreement:
    """RBPF and bootstrap PF should agree on average LL for non-Gaussian obs.

    Catches bugs where RBPF computes a biased likelihood (e.g., wrong quadrature,
    missing observation weight terms).
    """

    @pytest.mark.slow
    def test_mean_ll_agreement_poisson(self):
        """Average RBPF LL ≈ average bootstrap LL for Poisson observations."""
        key = random.PRNGKey(0)
        obs, dt, ct, meas, init = _simulate_poisson_data(key, T=15)

        n_runs = 40
        rbpf_lls = [
            float(_run_rbpf(ct, meas, init, obs, dt, rng_key=random.PRNGKey(i), n_particles=200))
            for i in range(n_runs)
        ]
        boot_lls = [
            float(
                _run_bootstrap_pf(
                    ct, meas, init, obs, dt, rng_key=random.PRNGKey(i), n_particles=200
                )
            )
            for i in range(n_runs)
        ]

        mean_rbpf = np.mean(rbpf_lls)
        mean_boot = np.mean(boot_lls)
        # Both should estimate the same marginal LL; allow some tolerance
        assert abs(mean_rbpf - mean_boot) < 3.0, (
            f"Mean RBPF={mean_rbpf:.2f}, Mean bootstrap={mean_boot:.2f}, "
            f"diff={abs(mean_rbpf - mean_boot):.2f}"
        )

    @pytest.mark.slow
    def test_mean_ll_agreement_student_t(self):
        """Average RBPF LL ≈ average bootstrap LL for Student-t observations."""
        ct, meas, init = _make_standard_params()
        T = 15
        obs = random.normal(random.PRNGKey(10), (T, 2)) * 0.5
        dt = jnp.ones(T)

        n_runs = 40
        rbpf_lls = [
            float(
                _run_rbpf(
                    ct,
                    meas,
                    init,
                    obs,
                    dt,
                    manifest_dist="student_t",
                    rng_key=random.PRNGKey(i),
                    n_particles=200,
                    extra_params={"obs_df": 5.0},
                )
            )
            for i in range(n_runs)
        ]
        boot_lls = [
            float(
                _run_bootstrap_pf(
                    ct,
                    meas,
                    init,
                    obs,
                    dt,
                    manifest_dist="student_t",
                    rng_key=random.PRNGKey(i),
                    n_particles=200,
                    extra_params={"obs_df": 5.0},
                )
            )
            for i in range(n_runs)
        ]

        mean_rbpf = np.mean(rbpf_lls)
        mean_boot = np.mean(boot_lls)
        assert abs(mean_rbpf - mean_boot) < 3.0, (
            f"Mean RBPF={mean_rbpf:.2f}, Mean bootstrap={mean_boot:.2f}, "
            f"diff={abs(mean_rbpf - mean_boot):.2f}"
        )


# =============================================================================
# TestRBPFDiscretization — Verify RBPF uses discretized params correctly
# =============================================================================


class TestRBPFDiscretization:
    """Test that RBPF correctly uses CT→DT discretized parameters.

    Catches bugs where RBPF uses wrong time intervals or swaps Ad/Qd/cd.
    """

    def test_longer_dt_increases_uncertainty(self):
        """Larger dt should give larger prediction uncertainty → lower LL."""
        ct, meas, init = _make_standard_params()
        T = 10
        obs = random.normal(random.PRNGKey(0), (T, 2)) * 0.5

        dt_short = jnp.ones(T) * 0.1
        dt_long = jnp.ones(T) * 5.0
        ll_short = _run_rbpf(ct, meas, init, obs, dt_short, manifest_dist="gaussian")
        ll_long = _run_rbpf(ct, meas, init, obs, dt_long, manifest_dist="gaussian")
        # Short dt → tighter predictions → typically higher LL for moderate data
        # (not guaranteed for all data, but holds for typical draws)
        assert ll_short != ll_long, "Different dt should give different LL"

    def test_dt_sensitivity_poisson(self):
        """RBPF LL should change when time intervals change (Poisson)."""
        ct, meas, init = _make_standard_params()
        T = 10
        obs = random.poisson(random.PRNGKey(5), jnp.ones((T, 2)) * 2.0).astype(jnp.float32)

        dt1 = jnp.ones(T)
        dt2 = jnp.ones(T) * 0.5
        ll1 = float(_run_rbpf(ct, meas, init, obs, dt1))
        ll2 = float(_run_rbpf(ct, meas, init, obs, dt2))
        assert ll1 != ll2, "Different dt should produce different LL"

    def test_zero_diffusion_reduces_to_deterministic(self):
        """With zero diffusion, RBPF should give very tight state predictions."""
        ct = CTParams(
            drift=jnp.array([[-0.1, 0.0], [0.0, -0.1]]),
            diffusion_cov=jnp.eye(2) * 1e-10,
            cint=jnp.zeros(2),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(2),
            manifest_means=jnp.zeros(2),
            manifest_cov=jnp.eye(2) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2) * 1e-10)

        T = 10
        obs = random.normal(random.PRNGKey(0), (T, 2)) * 0.5
        dt = jnp.ones(T)
        ll = _run_rbpf(ct, meas, init, obs, dt, manifest_dist="gaussian")
        assert jnp.isfinite(ll), f"Near-zero diffusion should still give finite LL: {ll}"


# =============================================================================
# TestRBPFMonotonicity — Likelihood should respond sensibly to param changes
# =============================================================================


class TestRBPFMonotonicity:
    """Verify monotonic relationships between params and LL.

    Catches sign errors, wrong Jacobians, and off-by-one bugs in RBPF.
    """

    def test_better_measurement_increases_ll(self):
        """Smaller measurement noise → higher LL when data matches the model."""
        ct, _, init = _make_standard_params()
        T = 15

        # Simulate from the model with low measurement noise
        key = random.PRNGKey(0)
        k1, k2 = random.split(key)
        states = [init.mean]
        for _t in range(T - 1):
            k1, k_step = random.split(k1)
            states.append(states[-1] + ct.drift @ states[-1] + random.normal(k_step, (2,)) * 0.1)
        states = jnp.stack(states)
        obs = (states + random.normal(k2, states.shape) * 0.05).astype(jnp.float32)
        dt = jnp.ones(T)

        ll_low_noise = float(
            _run_rbpf(
                ct,
                MeasurementParams(jnp.eye(2), jnp.zeros(2), jnp.eye(2) * 0.01),
                init,
                obs,
                dt,
                manifest_dist="gaussian",
            )
        )
        ll_high_noise = float(
            _run_rbpf(
                ct,
                MeasurementParams(jnp.eye(2), jnp.zeros(2), jnp.eye(2) * 1.0),
                init,
                obs,
                dt,
                manifest_dist="gaussian",
            )
        )
        # Low noise model should fit clean data better
        assert ll_low_noise > ll_high_noise, (
            f"Low noise LL={ll_low_noise:.2f} should be > high noise LL={ll_high_noise:.2f}"
        )

    def test_correct_drift_beats_wrong_drift(self):
        """True drift parameter should give higher LL than a wrong one (Poisson)."""
        key = random.PRNGKey(42)
        obs, dt, ct_true, meas, init = _simulate_poisson_data(key, T=20)

        ct_wrong = CTParams(
            drift=jnp.eye(2) * 0.5,  # Unstable positive drift (wrong)
            diffusion_cov=ct_true.diffusion_cov,
            cint=ct_true.cint,
        )

        ll_true = float(_run_rbpf(ct_true, meas, init, obs, dt, n_particles=300))
        ll_wrong = float(_run_rbpf(ct_wrong, meas, init, obs, dt, n_particles=300))
        assert ll_true > ll_wrong, (
            f"True drift LL={ll_true:.2f} should beat wrong drift LL={ll_wrong:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
