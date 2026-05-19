"""Slow inference-strategy integration tests."""

import functools

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import SSMModel, fit
from nof1_causal_lab.models.ssm.autoreparam import AutoReparam
from nof1_causal_lab.models.ssm.discretization import discretize_system_batched
from nof1_causal_lab.models.ssm.inference import _eval_model
from nof1_causal_lab.models.ssm.inference.shared import _apply_reparam
from nof1_causal_lab.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.inference.targets.emissions import (
    get_mean_param_log_prob_fn,
)
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace import (
    LaplaceLikelihood,
    _dense_support_laplace_log_lik,
)
from nof1_causal_lab.models.ssm.inference.utils import _build_eval_fns, _discover_sites
from tests.ssm_test_utils import (
    diagonal_diffusion_kwargs,
    make_observation_support_runtime,
    make_ssm_spec,
)

pytestmark = pytest.mark.slow


class TestLaplaceSupportAware:
    def test_laplace_backend_handles_window_average(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )

        assert jnp.isfinite(ll)

    def test_laplace_banded_matches_dense_reference(self):
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            manifest_links=[LinkFunction.IDENTITY],
            n_ieks_iters=2,
            observation_support=support,
        )
        ct_params = CTParams(
            drift=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion_cov=jnp.array([[0.1]], dtype=jnp.float32),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = InitialStateParams(
            mean=jnp.array([0.0], dtype=jnp.float32),
            cov=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            time_intervals,
        )
        assert cd is not None
        if cd.ndim == 1:
            cd = cd[:, None]
        obs_kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=meas_params.manifest_cov,
        )
        mean_log_prob_fn = get_mean_param_log_prob_fn(DistributionFamily.GAUSSIAN)

        banded = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
        )
        dense, _inner_eval_aux = _dense_support_laplace_log_lik(
            jnp.nan_to_num(observations, nan=0.0),
            ~jnp.isnan(observations),
            Ad,
            Qd,
            cd,
            meas_params.lambda_mat,
            meas_params.manifest_means,
            meas_params.manifest_cov,
            init.mean,
            init.cov,
            obs_kernel,
            mean_log_prob_fn,
            support,
            2,
        )

        assert banded == pytest.approx(float(dense), rel=1e-2, abs=1e-2)


class TestParameterRecoveryMAP:
    """Parameter recovery tests using MAP."""

    def test_drift_diagonal_recovery(self):
        true_drift_diag = jnp.array([-0.6, -0.9])

        key = random.PRNGKey(42)
        T = 60
        n_latent = 2
        dt = 0.5
        discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))
        process_noise = 0.3

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * process_noise
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.1
        times = jnp.arange(T, dtype=float) * dt

        spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            **diagonal_diffusion_kwargs(2),
        )
        model = SSMModel(spec)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="map",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = result.get_samples()
        drift_diag_samples = samples["drift_diag_free"]

        for i, true_val in enumerate(true_drift_diag):
            posterior_mean = jnp.mean(drift_diag_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.5, (
                f"Drift[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )

    def test_diffusion_recovery(self):
        true_diffusion_diag = jnp.array([0.4, 0.4])
        true_drift_diag = jnp.array([-0.5, -0.5])

        key = random.PRNGKey(123)
        T = 80
        n_latent = 2
        dt = 0.5
        discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key, subkey = random.split(key)
            noise = random.normal(subkey, (n_latent,)) * true_diffusion_diag
            new_state = discrete_coef @ states[-1] + noise
            states.append(new_state)

        key, subkey = random.split(key)
        observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.05
        times = jnp.arange(T, dtype=float) * dt

        spec = make_ssm_spec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            **diagonal_diffusion_kwargs(2),
        )
        model = SSMModel(spec)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="map",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        samples = result.get_samples()
        diffusion_samples = samples["diffusion_diag_free"]

        for i, true_val in enumerate(true_diffusion_diag):
            posterior_mean = jnp.mean(diffusion_samples[:, i])
            assert abs(posterior_mean - true_val) < 0.4, (
                f"Diffusion[{i}] posterior mean {float(posterior_mean):.3f} "
                f"far from true {float(true_val):.3f}"
            )


class TestPureJaxLikelihoodEvaluator:
    """The pure-JAX likelihood path should match NumPyro replay exactly."""

    @staticmethod
    def _build_poisson_case():
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
            manifest_dists=[DistributionFamily.POISSON],
            manifest_links=[LinkFunction.LOG],
            manifest_means=jnp.array([jnp.log(4.0)], dtype=jnp.float32),
        )
        model = SSMModel(spec)
        observations = jnp.array([[4.0], [3.0], [5.0], [6.0]], dtype=jnp.float32)
        times = jnp.arange(observations.shape[0], dtype=jnp.float32) * 0.5
        return model, observations, times

    @staticmethod
    def _assert_log_likelihood_match(reparam) -> None:
        model, observations, times = TestPureJaxLikelihoodEvaluator._build_poisson_case()
        backend = model.make_likelihood_backend()
        site_info = _discover_sites(
            model,
            observations,
            times,
            random.PRNGKey(0),
            backend,
            reparam=reparam,
        )
        example_unc = {
            name: info["transform"].inv(info["value"]) for name, info in site_info.items()
        }
        z0, unravel_fn = ravel_pytree(example_unc)
        log_lik_fn, _ = _build_eval_fns(
            model,
            observations,
            times,
            site_info,
            unravel_fn,
            likelihood_backend=backend,
            reparam=reparam,
        )

        base_model_fn = functools.partial(model.model, likelihood_backend=backend)
        replay_model_fn = _apply_reparam(base_model_fn, reparam)
        constrained = {
            name: site_info[name]["transform"](unravel_fn(z0)[name]) for name in site_info
        }
        replay_ll, _ = _eval_model(replay_model_fn, constrained, observations, times)

        np.testing.assert_allclose(
            np.asarray(log_lik_fn(z0)),
            np.asarray(replay_ll),
            rtol=1e-6,
            atol=1e-6,
        )
        grads = jax.grad(log_lik_fn)(z0)
        assert jnp.all(jnp.isfinite(grads))

    def test_log_likelihood_matches_model_replay_without_reparam(self):
        self._assert_log_likelihood_match(reparam=None)

    def test_log_likelihood_matches_model_replay_with_fixed_autoreparam(self):
        self._assert_log_likelihood_match(reparam=AutoReparam(centered=0.0))
