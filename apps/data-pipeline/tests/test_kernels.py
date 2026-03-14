"""Tests for kernel layer: observation and transition kernel factories.

Covers: response functions, variance functions, build_observation_kernel,
build_transition_kernel.
"""

import jax
import jax.numpy as jnp
import pytest

from causal_ssm_agent.models.likelihoods.kernels import (
    _response_probit,
    _response_sigmoid,
    build_observation_kernel,
    build_transition_kernel,
)
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

# =============================================================================
# Response functions (inverse links)
# =============================================================================


class TestResponseFunctions:
    def test_sigmoid_range(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        result = _response_sigmoid(x)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)
        assert jnp.isclose(result[1], 0.5)

    def test_probit_range(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        result = _response_probit(x)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)
        assert jnp.isclose(result[1], 0.5)


# =============================================================================
# build_observation_kernel
# =============================================================================


class TestBuildObservationKernel:
    def test_gaussian_is_gaussian(self):
        R = jnp.eye(2)
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY, manifest_cov=R
        )
        assert kernel.is_gaussian

    def test_poisson_not_gaussian(self):
        kernel = build_observation_kernel(DistributionFamily.POISSON, LinkFunction.LOG)
        assert not kernel.is_gaussian

    def test_student_t_not_gaussian(self):
        R = jnp.eye(2)
        kernel = build_observation_kernel(
            DistributionFamily.STUDENT_T, LinkFunction.IDENTITY, manifest_cov=R
        )
        assert not kernel.is_gaussian

    def test_bernoulli_kernel(self):
        kernel = build_observation_kernel(DistributionFamily.BERNOULLI, LinkFunction.LOGIT)
        assert not kernel.is_gaussian
        # Variance fn should produce diagonal matrix
        mean = jnp.array([0.5, 0.3])
        var = kernel.variance_fn(mean)
        assert var.shape == (2, 2)
        assert jnp.isclose(var[0, 0], 0.25)  # 0.5 * 0.5

    def test_poisson_variance(self):
        kernel = build_observation_kernel(DistributionFamily.POISSON, LinkFunction.LOG)
        mean = jnp.array([3.0, 5.0])
        var = kernel.variance_fn(mean)
        assert var.shape == (2, 2)
        assert jnp.isclose(var[0, 0], 3.0)  # Var = mean for Poisson

    def test_negative_binomial_variance(self):
        kernel = build_observation_kernel(
            DistributionFamily.NEGATIVE_BINOMIAL,
            LinkFunction.LOG,
            extra_params={"obs_r": 10.0},
        )
        mean = jnp.array([5.0])
        var = kernel.variance_fn(mean)
        # Var = mu + mu^2 / r = 5 + 25/10 = 7.5
        assert jnp.isclose(var[0, 0], 7.5)

    def test_gamma_variance(self):
        kernel = build_observation_kernel(
            DistributionFamily.GAMMA,
            LinkFunction.LOG,
            extra_params={"obs_shape": 2.0},
        )
        mean = jnp.array([4.0])
        var = kernel.variance_fn(mean)
        # Var = mean^2 / shape = 16 / 2 = 8
        assert jnp.isclose(var[0, 0], 8.0)

    def test_beta_variance(self):
        kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": 9.0},
        )
        mean = jnp.array([0.5])
        var = kernel.variance_fn(mean)
        # Var = p(1-p) / (phi+1) = 0.25 / 10 = 0.025
        assert jnp.isclose(var[0, 0], 0.025)

    def test_ordered_logistic_variance(self):
        kernel = build_observation_kernel(
            DistributionFamily.ORDERED_LOGISTIC,
            LinkFunction.CUMULATIVE_LOGIT,
            extra_params={
                "obs_level_counts": jnp.array([3, 4]),
                "obs_ordered_cutpoints": jnp.array([[-1.0, 1.0, 0.0], [-1.5, 0.0, 1.5]]),
            },
        )
        eta = jnp.array([0.0, 0.5])
        var = kernel.variance_fn(eta)
        assert var.shape == (2, 2)
        assert jnp.all(jnp.diag(var) > 0)

    def test_categorical_variance(self):
        kernel = build_observation_kernel(
            DistributionFamily.CATEGORICAL,
            LinkFunction.SOFTMAX,
            extra_params={
                "obs_level_counts": jnp.array([3]),
                "obs_cat_intercepts": jnp.array([[-1.0, 0.5]]),
                "obs_cat_slopes": jnp.array([[0.2, -0.4]]),
            },
        )
        eta = jnp.array([0.7])
        var = kernel.variance_fn(eta)
        assert var.shape == (1, 1)
        assert var[0, 0] > 0

    def test_unsupported_link_raises(self):
        with pytest.raises(ValueError, match="No response function"):
            build_observation_kernel(
                DistributionFamily.GAUSSIAN, "nonexistent_link", manifest_cov=jnp.eye(2)
            )

    def test_gaussian_response_is_identity(self):
        R = jnp.eye(2)
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY, manifest_cov=R
        )
        x = jnp.array([1.0, -2.0])
        assert jnp.allclose(kernel.response_fn(x), x)

    def test_gaussian_without_cov_raises_on_variance_call(self):
        kernel = build_observation_kernel(DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY)
        with pytest.raises(RuntimeError, match="requires manifest_cov"):
            kernel.variance_fn(jnp.array([1.0]))


# =============================================================================
# build_transition_kernel
# =============================================================================


class TestBuildTransitionKernel:
    def test_gaussian_is_gaussian(self):
        kernel = build_transition_kernel(DistributionFamily.GAUSSIAN)
        assert kernel.is_gaussian

    def test_student_t_not_gaussian(self):
        kernel = build_transition_kernel(DistributionFamily.STUDENT_T)
        assert not kernel.is_gaussian

    def test_gaussian_noise_shape(self):
        kernel = build_transition_kernel(DistributionFamily.GAUSSIAN)
        key = jax.random.PRNGKey(0)
        chol_Q = jnp.eye(3) * 0.1
        noise = kernel.sample_noise_fn(key, chol_Q)
        assert noise.shape == (3,)

    def test_student_t_noise_shape(self):
        kernel = build_transition_kernel(
            DistributionFamily.STUDENT_T, extra_params={"proc_df": 5.0}
        )
        key = jax.random.PRNGKey(0)
        chol_Q = jnp.eye(2) * 0.5
        noise = kernel.sample_noise_fn(key, chol_Q)
        assert noise.shape == (2,)

    def test_unsupported_diffusion_raises(self):
        with pytest.raises(ValueError, match="No transition kernel"):
            build_transition_kernel(DistributionFamily.POISSON)

    def test_gaussian_noise_mean_near_zero(self):
        """Gaussian process noise should have approximately zero mean."""
        kernel = build_transition_kernel(DistributionFamily.GAUSSIAN)
        chol_Q = jnp.eye(2) * 0.1
        keys = jax.random.split(jax.random.PRNGKey(42), 1000)
        samples = jax.vmap(lambda k: kernel.sample_noise_fn(k, chol_Q))(keys)
        assert jnp.allclose(samples.mean(axis=0), jnp.zeros(2), atol=0.05)
