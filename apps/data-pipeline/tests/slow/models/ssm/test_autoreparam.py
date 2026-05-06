"""Slow AutoReparam integration tests."""

import jax
import jax.numpy as jnp
import pytest
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from tests.models.ssm._support import simple_normal_model
from tests.ssm_test_utils import make_ssm_spec

pytestmark = pytest.mark.slow


class TestEndToEndSVI:
    def test_svi_then_predictive(self):
        strategy = AutoReparam(centered=0.0)
        model = strategy(simple_normal_model)
        guide = AutoNormal(model)
        svi = SVI(model, guide, Adam(1e-3), Trace_ELBO())

        svi_state = svi.init(jax.random.PRNGKey(0))
        for _ in range(3):
            svi_state, _loss = svi.update(svi_state)

        params = svi.get_params(svi_state)
        predictive = Predictive(model, guide=guide, params=params, num_samples=5)
        samples = predictive(jax.random.PRNGKey(1))
        assert "x" in samples
        assert "y" in samples

    def test_learnable_centering(self):
        strategy = AutoReparam(centered=None)
        model = strategy(simple_normal_model)
        guide = AutoNormal(model)
        svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())

        svi_state = svi.init(jax.random.PRNGKey(0))
        for _ in range(20):
            svi_state, _loss = svi.update(svi_state)

        params = svi.get_params(svi_state)
        centering_params = [k for k in params if "_centered" in k]
        assert len(centering_params) > 0


def _make_simple_ssm():
    from causal_ssm_agent.models.ssm.model import SSMModel

    return SSMModel(spec=make_ssm_spec(n_latent=2, n_manifest=2), likelihood="kalman")


class TestAutoReparamSSM:
    def test_fit_svi_with_reparam(self):
        from causal_ssm_agent.models.ssm.inference import fit

        model = _make_simple_ssm()
        observations = jnp.zeros((10, 2))
        times = jnp.linspace(0, 1, 10)

        result = fit(
            model,
            observations,
            times,
            method="svi",
            reparam=AutoReparam(centered=0.0),
            num_steps=50,
            num_samples=10,
            seed=42,
        )

        assert result.method == "svi"
        samples = result.get_samples()
        assert len(samples) > 0
        for value in samples.values():
            assert jnp.all(jnp.isfinite(value))

    def test_fit_map_filters_auxiliary_sites(self):
        from causal_ssm_agent.models.ssm.inference import fit

        model = _make_simple_ssm()
        observations = jnp.zeros((8, 2))
        times = jnp.linspace(0, 1, 8)

        result = fit(
            model,
            observations,
            times,
            method="map",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
        )

        sample_names = set(result.get_samples())
        assert "drift_base_decay_free" in sample_names
        assert "diffusion_diag_free" in sample_names
        assert all("_decentered" not in name for name in sample_names)

        diag = result.get_mcmc_diagnostics()
        assert diag is None
