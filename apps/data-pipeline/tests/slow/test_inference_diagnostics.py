"""Slow diagnostics tests that require real posterior inference."""

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from causal_ssm_agent.models.ssm.inference import InferenceResult

pytestmark = pytest.mark.slow


def _toy_model(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))
    mu = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


@pytest.fixture(scope="module")
def mcmc_result():
    key = random.PRNGKey(0)
    n_obs = 30
    x = jnp.linspace(-2, 2, n_obs)
    y = 1.0 + 2.5 * x + 0.5 * random.normal(key, (n_obs,))

    kernel = NUTS(_toy_model)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=2)
    mcmc.run(
        random.PRNGKey(1), x, y, extra_fields=("diverging", "num_steps", "accept_prob", "energy")
    )

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="nuts",
        diagnostics={"mcmc": mcmc},
    )


class TestMCMCDiagnostics:
    def test_basic_diagnostics(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert diag is not None
        assert "per_parameter" in diag
        assert len(diag["per_parameter"]) == 3
        for p in diag["per_parameter"]:
            assert "parameter" in p
            assert "r_hat" in p
            assert "ess_bulk" in p

    def test_ess_tail_values_positive(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        for p in diag["per_parameter"]:
            assert "ess_tail" in p
            assert p["ess_tail"] > 0

    def test_mcse_values_positive(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        for p in diag["per_parameter"]:
            assert "mcse_mean" in p
            assert p["mcse_mean"] > 0

    def test_trace_data_has_finite_values(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert len(diag["trace_data"]) == 3
        assert {t["parameter"] for t in diag["trace_data"]} == {"alpha", "beta", "sigma"}
        for trace in diag["trace_data"]:
            assert len(trace["chains"]) == 2
            for chain in trace["chains"]:
                assert len(chain) > 0

    def test_rank_histograms_have_valid_bins(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert len(diag["rank_histograms"]) == 3
        assert {h["parameter"] for h in diag["rank_histograms"]} == {"alpha", "beta", "sigma"}
        for hist in diag["rank_histograms"]:
            assert hist["n_bins"] > 0
            for chain_entry in hist["chains"]:
                counts = chain_entry["counts"]
                assert len(counts) == hist["n_bins"]
                assert all(c >= 0 for c in counts)
                assert sum(counts) > 0

    def test_sampler_stats(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "num_divergences" in diag
        assert "tree_depth_mean" in diag
        assert "accept_prob_mean" in diag
        assert diag["num_chains"] == 2
        assert diag["num_samples"] == 200


class TestLOODiagnostics:
    def test_loo_basic(self, mcmc_result):
        key = random.PRNGKey(0)
        n_obs = 30
        x = jnp.linspace(-2, 2, n_obs)
        y = 1.0 + 2.5 * x + 0.5 * random.normal(key, (n_obs,))

        loo = mcmc_result.get_loo_diagnostics(model_fn=_toy_model, observations=x, times=y)
        assert loo is not None
        assert "elpd_loo" in loo
        assert "p_loo" in loo
        assert "se" in loo
        assert "pareto_k" in loo
        assert len(loo["pareto_k"]) == n_obs
        assert loo["n_bad_k"] == 0

    def test_loo_without_model_returns_none(self, mcmc_result):
        assert mcmc_result.get_loo_diagnostics() is None

    def test_loo_smc_path(self):
        key = random.PRNGKey(0)
        n_obs = 30
        x = jnp.linspace(-2, 2, n_obs)
        y = 1.0 + 2.5 * x + 0.5 * random.normal(key, (n_obs,))

        def _ssm_style_model(x, y):
            alpha = numpyro.sample("alpha", dist.Normal(0, 10))
            beta_p = numpyro.sample("beta", dist.Normal(0, 5))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            mu = alpha + beta_p * x
            ll_per_t = dist.Normal(mu, sigma).log_prob(y)
            numpyro.deterministic("ll_per_timestep", ll_per_t)
            numpyro.factor("log_likelihood", jnp.sum(ll_per_t))

        kernel = NUTS(_ssm_style_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
        mcmc.run(random.PRNGKey(42), x, y)

        smc_result = InferenceResult(
            _samples=mcmc.get_samples(),
            method="map",
            diagnostics={},
        )
        loo = smc_result.get_loo_diagnostics(model_fn=_ssm_style_model, observations=x, times=y)
        assert loo is not None
        assert "elpd_loo" in loo
        assert "p_loo" in loo
        assert "se" in loo
        assert "pareto_k" in loo
        assert len(loo["pareto_k"]) == n_obs
        assert loo["observation_unit"] == "timestep"


class TestPosteriorMarginals:
    def test_marginals(self, mcmc_result):
        marginals = mcmc_result.get_posterior_marginals()
        assert len(marginals) == 3
        assert {m["parameter"] for m in marginals} == {"alpha", "beta", "sigma"}
        for m in marginals:
            assert len(m["x_values"]) == len(m["density"])
            assert all(d >= 0 for d in m["density"])
            assert m["hdi_3"] < m["mean"] < m["hdi_97"]


class TestPosteriorPairs:
    def test_pairs(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        assert len(pairs) == 3
        for p in pairs:
            assert "param_x" in p
            assert "param_y" in p
            assert len(p["x_values"]) == len(p["y_values"])
            assert len(p["x_values"]) <= 200

    def test_divergent_field(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        for p in pairs:
            if "divergent" in p:
                assert len(p["divergent"]) == len(p["x_values"])
                assert not any(p["divergent"])


class TestEnergyInMCMCDiagnostics:
    def test_energy_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "energy" in diag
        energy = diag["energy"]
        assert "energy_hist" in energy
        assert "energy_transition_hist" in energy
        assert "bfmi" in energy
        assert len(energy["bfmi"]) == 2
