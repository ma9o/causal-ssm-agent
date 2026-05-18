"""Tests for Rao-Blackwell particle filter helper functions.

Covers: Gauss-Hermite quadrature, unscented sigma points,
        Kalman predict, and Kalman update.
"""

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.targets.rao_blackwell import (
    _gauss_hermite_1d,
    _kalman_predict,
    _kalman_update_gaussian,
    _multivariate_gauss_hermite,
    _unscented_sigma_points,
)

# =============================================================================
# Gauss-Hermite quadrature
# =============================================================================


class TestGaussHermite1D:
    def test_weights_sum_to_one(self):
        """Weights should sum to 1 (probability measure)."""
        _nodes, weights = _gauss_hermite_1d(5)
        assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-10)

    def test_integrates_x_squared(self):
        """int x^2 * N(x|0,1) dx = 1 (variance of standard normal)."""
        nodes, weights = _gauss_hermite_1d(5)
        result = jnp.sum(weights * nodes**2)
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_integrates_x_to_zero(self):
        """int x * N(x|0,1) dx = 0 (mean of standard normal)."""
        nodes, weights = _gauss_hermite_1d(5)
        result = jnp.sum(weights * nodes)
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_node_count(self):
        """Should return exactly n_points nodes and weights."""
        nodes, weights = _gauss_hermite_1d(7)
        assert nodes.shape == (7,)
        assert weights.shape == (7,)


class TestMultivariateGaussHermite:
    def test_1d_matches_1d(self):
        """dim=1 should match the 1D version."""
        nodes_1d, weights_1d = _gauss_hermite_1d(5)
        nodes_mv, weights_mv = _multivariate_gauss_hermite(5, 1)
        assert jnp.allclose(nodes_mv[:, 0], nodes_1d)
        assert jnp.allclose(weights_mv, weights_1d)

    def test_2d_output_shape(self):
        """2D with 3 points should give 9 nodes."""
        nodes, weights = _multivariate_gauss_hermite(3, 2)
        assert nodes.shape == (9, 2)
        assert weights.shape == (9,)

    def test_2d_weights_sum_to_one(self):
        """Multivariate weights should still sum to 1."""
        _nodes, weights = _multivariate_gauss_hermite(5, 2)
        assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-6)

    def test_integrates_x1_squared(self):
        """int x1^2 * N(x|0,I) dx = 1."""
        nodes, weights = _multivariate_gauss_hermite(5, 2)
        result = jnp.sum(weights * nodes[:, 0] ** 2)
        assert jnp.isclose(result, 1.0, atol=1e-5)


# =============================================================================
# Unscented sigma points
# =============================================================================


class TestUnscentedSigmaPoints:
    def test_point_count(self):
        """Should generate 2n+1 sigma points."""
        n = 3
        mean = jnp.zeros(n)
        cov = jnp.eye(n)
        points, weights = _unscented_sigma_points(mean, cov)
        assert points.shape == (2 * n + 1, n)
        assert weights.shape == (2 * n + 1,)

    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        mean = jnp.zeros(2)
        cov = jnp.eye(2)
        _points, weights = _unscented_sigma_points(mean, cov)
        assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-6)

    def test_weighted_mean_recovers_mean(self):
        """Weighted average of sigma points should recover the mean."""
        mean = jnp.array([1.0, -2.0, 3.0])
        cov = jnp.eye(3) * 0.5
        points, weights = _unscented_sigma_points(mean, cov)
        recovered = jnp.sum(weights[:, None] * points, axis=0)
        assert jnp.allclose(recovered, mean, atol=1e-5)

    def test_center_point_is_mean(self):
        """First sigma point should be the mean."""
        mean = jnp.array([5.0, -3.0])
        cov = jnp.eye(2)
        points, _weights = _unscented_sigma_points(mean, cov)
        assert jnp.allclose(points[0], mean)


# =============================================================================
# Kalman predict
# =============================================================================


class TestKalmanPredict:
    def test_identity_transition(self):
        """With F=I, Q=0, c=0: m_pred=m, P_pred=P."""
        n = 2
        m = jnp.array([1.0, 2.0])
        P = jnp.eye(n) * 0.5
        F = jnp.eye(n)
        Q = jnp.zeros((n, n))
        c = jnp.zeros(n)
        m_pred, P_pred = _kalman_predict(m, P, F, Q, c)
        assert jnp.allclose(m_pred, m)
        assert jnp.allclose(P_pred, P, atol=1e-7)

    def test_adds_process_noise(self):
        """P_pred should be P + Q when F=I."""
        n = 2
        m = jnp.zeros(n)
        P = jnp.eye(n) * 0.5
        F = jnp.eye(n)
        Q = jnp.eye(n) * 0.3
        c = jnp.zeros(n)
        _m_pred, P_pred = _kalman_predict(m, P, F, Q, c)
        assert jnp.allclose(P_pred, jnp.eye(n) * 0.8, atol=1e-7)

    def test_intercept(self):
        """c should shift the predicted mean."""
        n = 2
        m = jnp.array([1.0, 2.0])
        P = jnp.eye(n)
        F = jnp.eye(n)
        Q = jnp.zeros((n, n))
        c = jnp.array([0.5, -0.5])
        m_pred, _P_pred = _kalman_predict(m, P, F, Q, c)
        assert jnp.allclose(m_pred, jnp.array([1.5, 1.5]))

    def test_symmetry(self):
        """P_pred should be symmetric."""
        F = jnp.array([[0.9, 0.1], [0.0, 0.8]])
        P = jnp.array([[1.0, 0.2], [0.2, 0.5]])
        Q = jnp.eye(2) * 0.1
        m = jnp.zeros(2)
        c = jnp.zeros(2)
        _m_pred, P_pred = _kalman_predict(m, P, F, Q, c)
        assert jnp.allclose(P_pred, P_pred.T, atol=1e-10)


# =============================================================================
# Kalman update (Gaussian)
# =============================================================================


class TestKalmanUpdateGaussian:
    def test_perfect_observation_reduces_variance(self):
        """Observing should reduce posterior variance."""
        n_latent = 2
        n_manifest = 2
        m = jnp.zeros(n_latent)
        P = jnp.eye(n_latent) * 2.0
        H = jnp.eye(n_manifest, n_latent)
        R = jnp.eye(n_manifest) * 0.1
        d = jnp.zeros(n_manifest)
        y = jnp.array([1.0, 2.0])
        mask = jnp.ones(n_manifest)
        m_upd, P_upd, _lml = _kalman_update_gaussian(m, P, H, R, d, y, mask)
        # Posterior variance should be less than prior
        assert jnp.all(jnp.diag(P_upd) < jnp.diag(P))
        # Mean should move toward observation
        assert jnp.allclose(m_upd, y, atol=0.5)

    def test_all_missing_no_update(self):
        """With all channels missing, state should remain unchanged."""
        n = 2
        m = jnp.array([1.0, 2.0])
        P = jnp.eye(n) * 0.5
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.1
        d = jnp.zeros(n)
        y = jnp.array([999.0, 999.0])
        mask = jnp.zeros(n)
        m_upd, _P_upd, _lml = _kalman_update_gaussian(m, P, H, R, d, y, mask)
        # Should be almost unchanged (large variance inflation)
        assert jnp.allclose(m_upd, m, atol=0.01)

    def test_log_marginal_finite(self):
        """Log marginal likelihood should be finite."""
        n = 2
        m = jnp.zeros(n)
        P = jnp.eye(n)
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.5
        d = jnp.zeros(n)
        y = jnp.array([0.5, -0.5])
        mask = jnp.ones(n)
        _m_upd, _P_upd, lml = _kalman_update_gaussian(m, P, H, R, d, y, mask)
        assert jnp.isfinite(lml)
        assert lml < 0.0  # log-prob is negative

    def test_update_symmetry(self):
        """Updated covariance should be symmetric."""
        n = 2
        m = jnp.zeros(n)
        P = jnp.array([[1.0, 0.3], [0.3, 0.8]])
        H = jnp.eye(n)
        R = jnp.eye(n) * 0.5
        d = jnp.zeros(n)
        y = jnp.array([1.0, 2.0])
        mask = jnp.ones(n)
        _m_upd, P_upd, _lml = _kalman_update_gaussian(m, P, H, R, d, y, mask)
        assert jnp.allclose(P_upd, P_upd.T, atol=1e-6)
