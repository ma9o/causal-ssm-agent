"""Slow inference-strategy integration tests."""

import functools

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.distributions import DistributionFamily
from causal_ssm_agent.models.ssm import SSMModel, fit
from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference import _apply_reparam, _eval_model
from causal_ssm_agent.models.ssm.inference.methods.laplace_em import (
    LaplaceLikelihood,
    _dense_support_laplace_log_lik,
)
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.ssm.inference.targets.emissions import (
    get_mean_param_log_prob_fn,
)
from causal_ssm_agent.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
)
from causal_ssm_agent.models.ssm.inference.targets.particle import (
    ParticleLikelihood,
    SSMAdapter,
)
from causal_ssm_agent.models.ssm.inference.utils import _build_eval_fns, _discover_sites
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.ssm_test_utils import diagonal_diffusion_kwargs, make_ssm_spec

pytestmark = pytest.mark.slow


def _support_runtime(**kwargs) -> ObservationSupportRuntime:
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime(**kwargs)


class TestLaplaceSupportAware:
    def test_laplace_backend_handles_window_average(self):
        support = _support_runtime(
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
        support = _support_runtime(
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




class TestStudentTProcessNoise:
    """Tests for Student-t process noise variance calibration."""

    def test_student_t_process_noise_variance_matches_qd(self):
        from causal_ssm_agent.models.ssm.discretization import discretize_system

        n_latent, n_manifest = 1, 1
        df = 5.0
        dt = 1.0

        drift = jnp.array([[-0.5]])
        diffusion_cov = jnp.array([[0.3**2]])
        _, Qd, _ = discretize_system(drift, diffusion_cov, None, dt)

        adapter = SSMAdapter(
            n_latent,
            n_manifest,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            diffusion_dists=[DistributionFamily.STUDENT_T],
            manifest_links=[LinkFunction.IDENTITY],
        )

        params = {
            "drift": drift,
            "diffusion_cov": diffusion_cov,
            "lambda_mat": jnp.eye(1),
            "manifest_means": jnp.zeros(1),
            "manifest_cov": jnp.eye(1),
            "t0_mean": jnp.zeros(n_latent),
            "t0_cov": jnp.eye(n_latent),
            "proc_df": df,
        }

        key = random.PRNGKey(0)
        n_samples = 400
        keys = random.split(key, n_samples)
        x_prev = jnp.zeros(n_latent)

        samples = jax.vmap(lambda k: adapter.transition_sample(k, x_prev, params, dt))(keys)
        sample_var = jnp.var(samples, axis=0)[0]
        target_var = Qd[0, 0]

        assert jnp.isfinite(sample_var)
        assert jnp.allclose(sample_var, target_var, rtol=0.25, atol=0.05), (
            f"Sample var {float(sample_var):.4f} vs Qd {float(target_var):.4f}"
        )


class TestParameterRecoveryPF:
    """Parameter recovery tests using particle filter + NUTS."""

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
        model = SSMModel(spec, n_particles=200)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="nuts",
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
        model = SSMModel(spec, n_particles=200)

        result = fit(
            model,
            observations=observations,
            times=times,
            method="nuts",
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


class TestHighDimNonlinear:
    """Test PF+NUTS on a high-dimensional nonlinear model."""

    def test_high_dimensional_poisson_pf_nuts(self):
        import jax.scipy.linalg as jla

        from causal_ssm_agent.models.ssm.discretization import discretize_system

        n_latent, n_manifest = 6, 6
        T = 30
        dt = 0.4
        true_drift = -0.4
        proc_df = 4.0

        drift = true_drift * jnp.eye(n_latent) + 0.05 * (
            jnp.ones((n_latent, n_latent)) - jnp.eye(n_latent)
        )
        diffusion_cov = jnp.eye(n_latent) * 0.2**2

        key = random.PRNGKey(123)
        key, key_cross, key_noise, key_obs = random.split(key, 4)
        lambda_base = 0.7 * jnp.eye(n_manifest, n_latent)
        cross = random.normal(key_cross, (n_manifest, n_latent)) * 0.05
        lambda_mat = lambda_base + cross
        manifest_means = jnp.ones(n_manifest) * jnp.log(5.0)

        Ad, Qd, _ = discretize_system(drift, diffusion_cov, None, dt)
        chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)

        states = [jnp.zeros(n_latent)]
        for _ in range(T - 1):
            key_noise, key_z, key_chi2 = random.split(key_noise, 3)
            z = random.normal(key_z, (n_latent,))
            chi2 = random.gamma(key_chi2, proc_df / 2.0) * 2.0
            scale = jnp.sqrt((proc_df - 2.0) / chi2)
            noise = chol @ (z * scale)
            states.append(Ad @ states[-1] + noise)
        latent = jnp.stack(states)

        eta = jax.vmap(lambda x: lambda_mat @ x + manifest_means)(latent)
        eta = jnp.clip(eta, -10.0, 6.0)
        rates = jnp.exp(eta)
        observations = random.poisson(key_obs, rates).astype(jnp.float32)
        time_intervals = jnp.ones(T) * dt

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=None)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=jnp.eye(n_manifest) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n_latent), cov=jnp.eye(n_latent))

        backend = ParticleLikelihood(
            n_latent=n_latent,
            n_manifest=n_manifest,
            n_particles=300,
            manifest_dists=["poisson"],
            diffusion_dists=["student_t"],
            manifest_links=["log"],
        )
        ll = backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params={"proc_df": proc_df},
        )

        assert jnp.all(jnp.isfinite(ll)), f"Non-finite LL on high-dim model: {ll}"


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
        model = SSMModel(spec, n_particles=40)
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


class TestSVIBackend:
    """Slow SVI convergence coverage."""

    def test_svi_losses_decrease(self):
        spec = make_ssm_spec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            **diagonal_diffusion_kwargs(1),
        )
        model = SSMModel(spec, likelihood="kalman")

        T = 15
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 1)) * 0.5
        times = jnp.arange(T, dtype=jnp.float32) * 0.5

        result = fit(
            model,
            observations=observations,
            times=times,
            method="svi",
            num_steps=200,
            num_samples=10,
        )

        losses = result.diagnostics["losses"]
        n = len(losses)
        early_mean = float(jnp.mean(losses[: n // 10]))
        late_mean = float(jnp.mean(losses[-n // 10 :]))
        assert late_mean < early_mean, (
            f"SVI loss did not decrease: early={early_mean:.1f}, late={late_mean:.1f}"
        )


