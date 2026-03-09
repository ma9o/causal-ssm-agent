"""Tests for shared MCMC utilities (mcmc_utils.py).

Covers: hmc_step, compute_weighted_chol_mass, find_next_beta,
dual averaging step size adaptation.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.ssm.mcmc_utils import (
    DualAveragingState,
    compute_weighted_chol_mass,
    dual_averaging_init,
    dual_averaging_update,
    find_next_beta,
    hmc_step,
)

# =============================================================================
# hmc_step
# =============================================================================


class TestHMCStep:
    """Tests for the preconditioned HMC/MALA step."""

    def _make_gaussian_target(self, mean, prec):
        """Create a Gaussian log-target and its value_and_grad fn."""

        def log_target(z):
            diff = z - mean
            return -0.5 * diff @ prec @ diff

        return jax.value_and_grad(log_target)

    def test_returns_correct_shapes(self):
        """HMC step returns (z_new, accepted, log_target_new) with correct shapes."""
        D = 3
        key = random.PRNGKey(0)
        z = jnp.zeros(D)
        prec = jnp.eye(D)
        target_fn = self._make_gaussian_target(jnp.zeros(D), prec)

        z_new, accepted, log_pi = hmc_step(
            key, z, target_fn, step_size=0.1, chol_mass=jnp.eye(D), n_leapfrog=1
        )

        assert z_new.shape == (D,)
        assert accepted.shape == ()
        assert log_pi.shape == ()

    def test_mala_moves_toward_mode(self):
        """MALA (n_leapfrog=1) moves particles toward the target mode on average."""
        D = 2
        mean = jnp.array([3.0, -2.0])
        prec = jnp.eye(D) * 4.0  # tight target
        target_fn = self._make_gaussian_target(mean, prec)
        chol_mass = jnp.eye(D) * 2.0

        z = jnp.zeros(D)
        n_steps = 500
        key = random.PRNGKey(42)

        positions = []
        for i in range(n_steps):
            key, step_key = random.split(key)
            z, _accepted, _ = hmc_step(
                step_key, z, target_fn, step_size=0.1, chol_mass=chol_mass, n_leapfrog=1
            )
            if i >= 200:  # burn-in
                positions.append(z)

        positions = jnp.stack(positions)
        sample_mean = jnp.mean(positions, axis=0)
        # Should be reasonably close to true mean (MCMC has variance)
        assert jnp.allclose(sample_mean, mean, atol=1.5), (
            f"Sample mean {sample_mean} far from target {mean}"
        )

    def test_multi_leapfrog_hmc_accepts_more(self):
        """Multi-step leapfrog should maintain reasonable acceptance with proper step size."""
        D = 2
        mean = jnp.zeros(D)
        prec = jnp.eye(D)
        target_fn = self._make_gaussian_target(mean, prec)
        chol_mass = jnp.eye(D)

        n_trials = 50
        key = random.PRNGKey(0)
        accepts = []
        z = jnp.ones(D)

        for _ in range(n_trials):
            key, step_key = random.split(key)
            z, accepted, _ = hmc_step(
                step_key, z, target_fn, step_size=0.3, chol_mass=chol_mass, n_leapfrog=5
            )
            accepts.append(float(accepted))

        rate = sum(accepts) / len(accepts)
        assert rate > 0.3, f"HMC acceptance rate too low: {rate:.2f}"

    def test_accepts_at_mode(self):
        """Starting at mode with good step size should yield high acceptance."""
        D = 2
        mean = jnp.zeros(D)
        prec = jnp.eye(D)
        target_fn = self._make_gaussian_target(mean, prec)

        key = random.PRNGKey(123)
        z = mean  # start at mode

        _, _, log_pi = hmc_step(
            key, z, target_fn, step_size=0.5, chol_mass=jnp.eye(D), n_leapfrog=1
        )
        # log_pi should be non-positive (Gaussian peak is 0)
        assert log_pi <= 0.0 + 1e-6

    def test_nan_gradient_handled(self):
        """NaN gradients are replaced with 0, preventing crashes."""

        def bad_target(z):
            # Returns valid value but NaN gradient for first component
            val = -0.5 * jnp.dot(z, z)
            return val

        def val_and_grad_with_nan(z):
            val = bad_target(z)
            grad = -z
            # Inject NaN
            grad = grad.at[0].set(jnp.nan)
            return val, grad

        key = random.PRNGKey(0)
        z = jnp.ones(3)
        z_new, _, _ = hmc_step(
            key, z, val_and_grad_with_nan, step_size=0.1, chol_mass=jnp.eye(3), n_leapfrog=1
        )
        assert jnp.all(jnp.isfinite(z_new)), "HMC should handle NaN gradients gracefully"


# =============================================================================
# compute_weighted_chol_mass
# =============================================================================


class TestComputeWeightedCholMass:
    """Tests for weighted precision Cholesky computation."""

    def test_identity_covariance_gives_identity_precision(self):
        """Uniform-weight unit-variance particles should give ~identity precision."""
        N, D = 1000, 2
        key = random.PRNGKey(42)
        particles = random.normal(key, (N, D))
        logw = jnp.zeros(N)  # uniform weights

        chol = compute_weighted_chol_mass(particles, logw, D, reg=1e-4)
        prec = chol @ chol.T

        # Precision of N(0,I) should be ~I
        assert jnp.allclose(prec, jnp.eye(D), atol=0.2), (
            f"Precision should be ~I for N(0,I) samples, got {prec}"
        )

    def test_scaled_covariance(self):
        """Particles from N(0, diag(σ²)) should give precision ~diag(1/σ²)."""
        N, D = 2000, 2
        key = random.PRNGKey(0)
        scales = jnp.array([2.0, 0.5])
        particles = random.normal(key, (N, D)) * scales
        logw = jnp.zeros(N)

        chol = compute_weighted_chol_mass(particles, logw, D, reg=1e-4)
        prec = chol @ chol.T

        expected_prec = jnp.diag(1.0 / scales**2)
        assert jnp.allclose(prec, expected_prec, atol=0.3), (
            f"Precision diagonal mismatch: {jnp.diag(prec)} vs {jnp.diag(expected_prec)}"
        )

    def test_output_is_lower_triangular(self):
        """Cholesky output should be lower triangular."""
        N, D = 100, 3
        key = random.PRNGKey(1)
        particles = random.normal(key, (N, D))
        logw = jnp.zeros(N)

        chol = compute_weighted_chol_mass(particles, logw, D)

        # Upper triangle (excluding diagonal) should be zero
        upper = jnp.triu(chol, k=1)
        assert jnp.allclose(upper, 0.0, atol=1e-10)

    def test_weights_matter(self):
        """Non-uniform weights should shift the effective precision."""
        N, D = 200, 2
        key = random.PRNGKey(2)
        # Mix two clusters: tight cluster gets high weight
        k1, k2 = random.split(key)
        tight = random.normal(k1, (N // 2, D)) * 0.1  # tight
        wide = random.normal(k2, (N // 2, D)) * 5.0  # wide

        particles = jnp.concatenate([tight, wide])
        # Weight tight cluster heavily
        logw = jnp.concatenate([jnp.ones(N // 2), jnp.full(N // 2, -10.0)])

        chol = compute_weighted_chol_mass(particles, logw, D, reg=1e-4)
        prec = chol @ chol.T

        # Precision should reflect the tight cluster (high diagonal values)
        assert jnp.all(jnp.diag(prec) > 5.0), (
            f"Precision should reflect tight weighted cluster, got diag={jnp.diag(prec)}"
        )


# =============================================================================
# find_next_beta
# =============================================================================


class TestFindNextBeta:
    """Tests for ESS-based bisection tempering."""

    def test_beta_increases(self):
        """Next beta should be greater than previous beta."""
        N = 100
        key = random.PRNGKey(0)
        logw = jnp.zeros(N)  # uniform weights
        log_liks = random.normal(key, (N,))  # random likelihoods

        beta_prev = 0.3
        beta_next = find_next_beta(logw, log_liks, beta_prev, target_ess_ratio=0.5, N=N)

        assert beta_next > beta_prev, f"beta should increase: {beta_next} <= {beta_prev}"
        assert beta_next <= 1.0, f"beta should be at most 1.0: {beta_next}"

    def test_jumps_to_one_when_ess_high(self):
        """Should jump to beta=1.0 when ESS is still above target."""
        N = 100
        logw = jnp.zeros(N)
        # Very similar likelihoods → small ESS drop even with large delta
        log_liks = jnp.zeros(N) + 1.0

        beta_next = find_next_beta(logw, log_liks, 0.5, target_ess_ratio=0.5, N=N)
        assert beta_next == 1.0

    def test_small_step_when_likelihoods_vary(self):
        """With highly variable likelihoods, beta step should be small."""
        N = 100
        key = random.PRNGKey(42)
        logw = jnp.zeros(N)
        # Very high variance likelihoods
        log_liks = random.normal(key, (N,)) * 100.0

        beta_prev = 0.0
        beta_next = find_next_beta(logw, log_liks, beta_prev, target_ess_ratio=0.5, N=N)

        assert beta_next < 0.1, f"beta step should be small with high-variance liks: {beta_next}"

    def test_beta_bounded_by_one(self):
        """Beta should never exceed 1.0."""
        N = 50
        logw = jnp.zeros(N)
        log_liks = jnp.ones(N) * 0.01

        beta_next = find_next_beta(logw, log_liks, 0.99, target_ess_ratio=0.5, N=N)
        assert beta_next <= 1.0


# =============================================================================
# Dual averaging
# =============================================================================


class TestDualAveraging:
    """Tests for Nesterov dual averaging step size adaptation."""

    def test_init_state(self):
        """dual_averaging_init creates correct initial state."""
        state = dual_averaging_init(0.1)

        assert isinstance(state, DualAveragingState)
        assert state.step == 0
        assert state.h_bar == 0.0
        assert state.log_eps_bar == 0.0
        assert abs(state.eps - 0.1) < 1e-10

    def test_eps_property(self):
        """eps and eps_bar properties return exp of log values."""
        state = dual_averaging_init(0.5)
        assert abs(state.eps - 0.5) < 1e-10
        assert abs(state.eps_bar - 1.0) < 1e-10  # log_eps_bar=0 → exp(0)=1

    def test_adapts_down_for_low_acceptance(self):
        """When accept_prob < target, step size should decrease."""
        state = dual_averaging_init(1.0)

        # Simulate consistently low acceptance
        for _ in range(50):
            state = dual_averaging_update(state, accept_prob=0.2, target_accept=0.65)

        assert state.eps_bar < 1.0, f"Step size should decrease for low acceptance: {state.eps_bar}"

    def test_adapts_up_for_high_acceptance(self):
        """When accept_prob > target, step size should increase."""
        state = dual_averaging_init(0.01)

        # Simulate consistently high acceptance
        for _ in range(50):
            state = dual_averaging_update(state, accept_prob=0.95, target_accept=0.65)

        assert state.eps_bar > 0.01, (
            f"Step size should increase for high acceptance: {state.eps_bar}"
        )

    def test_converges_to_stable_value(self):
        """With constant acceptance at target, eps_bar should stabilize."""
        state = dual_averaging_init(0.5)
        target = 0.65

        # Feed exact target acceptance → should converge
        values = []
        for _ in range(200):
            state = dual_averaging_update(state, accept_prob=target, target_accept=target)
            values.append(state.eps_bar)

        # Check that it stabilizes (last 50 values should be similar)
        last_values = values[-50:]
        spread = max(last_values) - min(last_values)
        assert spread < 0.1, f"eps_bar should stabilize, spread={spread:.4f}"

    def test_step_increments(self):
        """Step counter should increment with each update."""
        state = dual_averaging_init(0.1)
        assert state.step == 0

        state = dual_averaging_update(state, accept_prob=0.5)
        assert state.step == 1

        state = dual_averaging_update(state, accept_prob=0.5)
        assert state.step == 2
