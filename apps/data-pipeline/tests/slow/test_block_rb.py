"""Slow block-RB tests that exercise repeated particle inference."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from tests.test_block_rb import (
    _make_mixed_params,
    _run_block_rbpf,
    _run_bootstrap_pf,
    _simulate_data,
    _simulate_data_exact,
    _simulate_poisson_data,
)

pytestmark = pytest.mark.slow


class TestVarianceReduction:
    def test_variance_reduction(self):
        ct, meas, init = _make_mixed_params(n_g=1, n_s=1, n_manifest=2, cross_coupling=False)
        obs, dt = _simulate_data(random.PRNGKey(555), ct, meas, init, T=20)

        block_lls = []
        for i in range(30):
            ll = _run_block_rbpf(
                ct,
                meas,
                init,
                obs,
                dt,
                diffusion_dists=["gaussian", "student_t"],
                n_particles=100,
                rng_key=random.PRNGKey(i),
                extra_params={"proc_df": 100.0},
            )
            block_lls.append(float(ll))

        boot_lls = []
        for i in range(30):
            ll = _run_bootstrap_pf(
                ct,
                meas,
                init,
                obs,
                dt,
                n_particles=100,
                rng_key=random.PRNGKey(i),
                extra_params={"proc_df": 100.0},
            )
            boot_lls.append(float(ll))

        assert np.var(block_lls) < np.var(boot_lls)


class TestParameterRecovery:
    def test_parameter_recovery_bootstrap_baseline(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(
            drift=jnp.diag(true_drift_diag),
            diffusion_cov=diffusion_cov,
            cint=jnp.zeros(n),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))
        obs, dt = _simulate_data(random.PRNGKey(777), ct_true, meas, init, T=30)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.5).expand((n,)))
            ct = CTParams(
                drift=jnp.diag(-jnp.abs(drift_diag)),
                diffusion_cov=diffusion_cov,
                cint=jnp.zeros(n),
            )

            from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=50,
                rng_key=random.PRNGKey(0),
                manifest_dists=["gaussian"] * n,
                diffusion_dists=["student_t"] * n,
            )
            numpyro.factor(
                "ll",
                backend.compute_log_likelihood(
                    ct,
                    meas,
                    init,
                    observations,
                    time_intervals,
                    extra_params={"proc_df": 100.0},
                )[-1],
            )

        svi = SVI(model, AutoNormal(model), numpyro.optim.Adam(0.01), loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _ in range(300):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        recovered = -jnp.abs(svi.get_params(svi_state)["drift_diag_auto_loc"])
        np.testing.assert_allclose(np.sort(recovered), np.sort(np.array(true_drift_diag)), atol=0.3)

    def test_parameter_recovery_block_rbpf(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(
            drift=jnp.diag(true_drift_diag),
            diffusion_cov=diffusion_cov,
            cint=jnp.zeros(n),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))
        obs, dt = _simulate_data(random.PRNGKey(777), ct_true, meas, init, T=30)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.5).expand((n,)))
            ct = CTParams(
                drift=jnp.diag(-jnp.abs(drift_diag)),
                diffusion_cov=diffusion_cov,
                cint=jnp.zeros(n),
            )

            from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=100,
                rng_key=random.PRNGKey(0),
                manifest_dists=["gaussian"] * n,
                diffusion_dists=["gaussian", "student_t"],
            )
            numpyro.factor(
                "ll",
                backend.compute_log_likelihood(
                    ct,
                    meas,
                    init,
                    observations,
                    time_intervals,
                    extra_params={"proc_df": 100.0},
                )[-1],
            )

        svi = SVI(model, AutoNormal(model), numpyro.optim.Adam(0.01), loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _ in range(500):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        recovered = -jnp.abs(svi.get_params(svi_state)["drift_diag_auto_loc"])
        np.testing.assert_allclose(
            np.sort(np.array(recovered)),
            np.sort(np.array(true_drift_diag)),
            atol=0.35,
        )

    def test_parameter_recovery_cross_coupled_drift(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_diag = jnp.array([-0.3, -0.7])
        true_offdiag = jnp.array([0.15, 0.2])
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
        obs, dt = _simulate_data_exact(random.PRNGKey(888), ct_true, meas, init, T=50)

        def model(observations, time_intervals):
            drift_diag = jnp.asarray(
                numpyro.sample("drift_diag", dist.Normal(-0.5, 0.3).expand((n,)))
            )
            drift_offdiag = jnp.asarray(
                numpyro.sample("drift_offdiag", dist.Normal(0.0, 0.3).expand((n,)))
            )
            drift = jnp.zeros((n, n))
            drift = drift.at[0, 0].set(-jnp.abs(drift_diag[0]))
            drift = drift.at[1, 1].set(-jnp.abs(drift_diag[1]))
            drift = drift.at[0, 1].set(drift_offdiag[0])
            drift = drift.at[1, 0].set(drift_offdiag[1])
            ct = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=jnp.zeros(n))

            from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=100,
                rng_key=random.PRNGKey(0),
                manifest_dists=["gaussian"] * n,
                diffusion_dists=["gaussian", "student_t"],
            )
            numpyro.factor(
                "ll",
                backend.compute_log_likelihood(
                    ct,
                    meas,
                    init,
                    observations,
                    time_intervals,
                    extra_params={"proc_df": 100.0},
                )[-1],
            )

        svi = SVI(model, AutoNormal(model), numpyro.optim.Adam(0.01), loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _ in range(500):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        params = svi.get_params(svi_state)
        np.testing.assert_allclose(
            np.sort(-jnp.abs(params["drift_diag_auto_loc"])),
            np.sort(np.array(true_diag)),
            atol=0.35,
        )
        np.testing.assert_allclose(
            params["drift_offdiag_auto_loc"], np.array(true_offdiag), atol=0.25
        )

    def test_parameter_recovery_higher_dim_1g_2s(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal

        true_drift_diag = jnp.array([-0.3, -0.5, -0.8])
        n = 3
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(
            drift=jnp.diag(true_drift_diag),
            diffusion_cov=diffusion_cov,
            cint=jnp.zeros(n),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.zeros(n),
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))
        obs, dt = _simulate_data_exact(random.PRNGKey(1111), ct_true, meas, init, T=50)

        def model(observations, time_intervals):
            drift_diag = numpyro.sample("drift_diag", dist.Normal(-0.5, 0.3).expand((n,)))
            ct = CTParams(
                drift=jnp.diag(-jnp.abs(drift_diag)),
                diffusion_cov=diffusion_cov,
                cint=jnp.zeros(n),
            )

            from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

            backend = ParticleLikelihood(
                n_latent=n,
                n_manifest=n,
                n_particles=200,
                rng_key=random.PRNGKey(0),
                manifest_dists=["gaussian"] * n,
                diffusion_dists=["gaussian", "student_t", "student_t"],
            )
            numpyro.factor(
                "ll",
                backend.compute_log_likelihood(
                    ct,
                    meas,
                    init,
                    observations,
                    time_intervals,
                    extra_params={"proc_df": 100.0},
                )[-1],
            )

        svi = SVI(model, AutoNormal(model), numpyro.optim.Adam(0.01), loss=Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(0), obs, dt)
        jit_update = jax.jit(svi.update)
        for _ in range(600):
            svi_state, _loss = jit_update(svi_state, obs, dt)

        recovered = -jnp.abs(svi.get_params(svi_state)["drift_diag_auto_loc"])
        np.testing.assert_allclose(
            np.sort(recovered), np.sort(np.array(true_drift_diag)), atol=0.35
        )

    def test_parameter_recovery_poisson_obs(self):
        true_drift_diag = jnp.array([-0.3, -0.7])
        n = 2
        diffusion_cov = jnp.eye(n) * 0.1
        ct_true = CTParams(
            drift=jnp.diag(true_drift_diag),
            diffusion_cov=diffusion_cov,
            cint=jnp.zeros(n),
        )
        meas = MeasurementParams(
            lambda_mat=jnp.eye(n),
            manifest_means=jnp.ones(n) * 1.5,
            manifest_cov=jnp.eye(n) * 0.1,
        )
        init = InitialStateParams(mean=jnp.zeros(n), cov=jnp.eye(n))
        obs, dt = _simulate_poisson_data(random.PRNGKey(1234), ct_true, meas, init, T=50)

        def compute_ll(drift_diag_vals):
            ct = CTParams(
                drift=jnp.diag(-jnp.abs(drift_diag_vals)),
                diffusion_cov=diffusion_cov,
                cint=jnp.zeros(n),
            )
            return _run_block_rbpf(
                ct,
                meas,
                init,
                obs,
                dt,
                diffusion_dists=["gaussian", "student_t"],
                n_particles=200,
                rng_key=random.PRNGKey(42),
                manifest_dists=["poisson", "poisson"],
                extra_params={"proc_df": 100.0},
            )

        ll_true = compute_ll(jnp.abs(true_drift_diag))
        ll_wrong = compute_ll(jnp.array([1.5, 0.05]))
        assert jnp.isfinite(ll_true)
        assert float(ll_true) > float(ll_wrong)
        assert jnp.all(jnp.isfinite(jax.grad(compute_ll)(jnp.abs(true_drift_diag))))
