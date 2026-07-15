"""Tests for the observation-kernel layer.

Covers variance functions and build_observation_kernel.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts import LikelihoodSpec, LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
    compile_observation_model,
)
from tests.ssm_spec_fixtures import block_ssm_spec, full_dense_matrix_dynamics_spec


class TestBuildObservationKernel:
    def test_likelihood_spec_rejects_invalid_family_link_pair(self):
        with pytest.raises(ValidationError, match="invalid for gaussian"):
            LikelihoodSpec(
                variable="y",
                distribution=DistributionFamily.GAUSSIAN,
                link=LinkFunction.LOG,
                reasoning="test",
            )

    def test_direct_ssm_spec_rejects_invalid_family_link_pair(self):
        with pytest.raises(ValueError, match="invalid for observation family 'gaussian'"):
            block_ssm_spec(
                n_latent=1,
                dynamics_spec=full_dense_matrix_dynamics_spec(1),
                manifest_dists=[DistributionFamily.GAUSSIAN],
                manifest_links=[LinkFunction.LOG],
            )

    def test_compiled_model_shares_predictor_semantics_for_likelihood_and_sampling(self):
        manifest_cov = jnp.eye(1)
        model = compile_observation_model(
            [DistributionFamily.POISSON],
            manifest_cov=manifest_cov,
            manifest_links=[LinkFunction.LOG],
        )
        eta = jnp.array([jnp.log(3.0)])
        log_prob = model.kernel.log_prob_fn(
            jnp.array([2.0]),
            eta,
            manifest_cov,
            jnp.ones(1),
        )
        expected = jax.scipy.stats.poisson.logpmf(2.0, 3.0)
        assert jnp.isclose(log_prob, expected)

        draws = model.point_sampler.sample_point_trajectory(
            jax.random.PRNGKey(0),
            eta[None, :],
        )
        assert draws.shape == (1, 1)
        assert draws[0, 0] >= 0
        assert jnp.isclose(draws[0, 0], jnp.rint(draws[0, 0]))

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

    def test_beta_kernel_accepts_traced_positive_site(self):
        @jax.jit
        def _build_variance(obs_concentration):
            kernel = build_observation_kernel(
                DistributionFamily.BETA,
                LinkFunction.LOGIT,
                extra_params={"obs_concentration": obs_concentration},
            )
            return kernel.variance_fn(jnp.array([0.5]))

        var = _build_variance(jnp.array(9.0))
        assert jnp.isclose(var[0, 0], 0.025)

    def test_beta_kernel_rejects_nonpositive_concrete_site(self):
        with pytest.raises(ValueError, match="obs_concentration must be positive"):
            build_observation_kernel(
                DistributionFamily.BETA,
                LinkFunction.LOGIT,
                extra_params={"obs_concentration": 0.0},
            )

    def test_gamma_inverse_response_marks_invalid_eta_as_nan(self):
        kernel = build_observation_kernel(
            DistributionFamily.GAMMA,
            LinkFunction.INVERSE,
            extra_params={"obs_shape": 2.0},
        )
        response = kernel.response_fn(jnp.array([2.0, -0.5]))
        assert jnp.isclose(response[0], 0.5)
        assert jnp.isnan(response[1])

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
        invalid_link = cast("LinkFunction", "nonexistent_link")
        with pytest.raises(ValueError, match="Unknown link function"):
            build_observation_kernel(
                DistributionFamily.GAUSSIAN, invalid_link, manifest_cov=jnp.eye(2)
            )

    def test_recognized_but_invalid_family_link_pair_raises(self):
        with pytest.raises(ValueError, match="invalid for observation family 'gaussian'"):
            build_observation_kernel(
                DistributionFamily.GAUSSIAN,
                LinkFunction.LOG,
                manifest_cov=jnp.eye(2),
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
