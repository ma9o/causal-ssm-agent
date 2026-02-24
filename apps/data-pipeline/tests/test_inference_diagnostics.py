"""Tests for enriched MCMC diagnostics extraction (trace data, rank histograms, ESS-tail, LOO)."""

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from causal_ssm_agent.models.ssm.inference import (
    InferenceResult,
    _build_energy_diagnostics,
    _build_rank_histograms,
    _build_trace_data,
    _param_marginal,
)

numpyro.set_host_device_count(2)


def _toy_model(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))
    mu = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


@pytest.fixture(scope="module")
def mcmc_result():
    """Run a toy MCMC and return InferenceResult."""
    key = random.PRNGKey(0)
    N = 30
    x = jnp.linspace(-2, 2, N)
    y = 1.0 + 2.5 * x + 0.5 * random.normal(key, (N,))

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


class TestBuildTraceData:
    def test_trace_data_shape(self):
        chain_samples = {
            "a": jnp.ones((2, 100)),
            "b": jnp.ones((2, 100, 3)),
        }
        traces = _build_trace_data(chain_samples, max_points=50)
        # "a" is scalar -> 1 trace, "b" is 3-dim -> 3 traces
        assert len(traces) == 4
        assert traces[0]["parameter"] == "a"
        assert len(traces[0]["chains"]) == 2
        assert len(traces[0]["chains"][0]["values"]) == 50

    def test_trace_data_thinning(self):
        chain_samples = {"x": jnp.ones((2, 1000))}
        traces = _build_trace_data(chain_samples, max_points=100)
        assert len(traces[0]["chains"][0]["values"]) == 100


class TestBuildRankHistograms:
    def test_rank_histogram_structure(self):
        key = random.PRNGKey(42)
        chain_samples = {"x": random.normal(key, (2, 200))}
        hists = _build_rank_histograms(chain_samples, n_bins=10)
        assert len(hists) == 1
        assert hists[0]["parameter"] == "x"
        assert hists[0]["n_bins"] == 10
        assert hists[0]["expected_per_bin"] == 20.0  # 200 / 10
        assert len(hists[0]["chains"]) == 2
        assert len(hists[0]["chains"][0]["counts"]) == 10
        # Total counts should equal n_samples
        assert sum(hists[0]["chains"][0]["counts"]) == 200

    def test_skips_multidim(self):
        chain_samples = {"matrix": jnp.ones((2, 100, 3, 3))}
        hists = _build_rank_histograms(chain_samples)
        assert len(hists) == 0


class TestParamMarginal:
    def test_marginal_structure(self):
        values = random.normal(random.PRNGKey(0), (500,))
        m = _param_marginal("test", values, n_bins=30)
        assert m["parameter"] == "test"
        assert len(m["x_values"]) == 30
        assert len(m["density"]) == 30
        assert m["hdi_3"] < m["mean"] < m["hdi_97"]
        # Density should be non-negative
        assert all(d >= 0 for d in m["density"])


class TestMCMCDiagnostics:
    def test_basic_diagnostics(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert diag is not None
        assert "per_parameter" in diag
        assert len(diag["per_parameter"]) == 3  # alpha, beta, sigma
        for p in diag["per_parameter"]:
            assert "parameter" in p
            assert "r_hat" in p
            assert "ess_bulk" in p

    def test_ess_tail_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        # ArviZ should provide ess_tail
        has_tail = any("ess_tail" in p for p in diag["per_parameter"])
        assert has_tail, "ESS-tail should be present (requires arviz)"

    def test_mcse_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        has_mcse = any("mcse_mean" in p for p in diag["per_parameter"])
        assert has_mcse, "MCSE should be present (requires arviz)"

    def test_trace_data_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "trace_data" in diag
        assert len(diag["trace_data"]) == 3
        for trace in diag["trace_data"]:
            assert "parameter" in trace
            assert "chains" in trace
            assert len(trace["chains"]) == 2  # 2 chains

    def test_rank_histograms_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "rank_histograms" in diag
        assert len(diag["rank_histograms"]) == 3
        for hist in diag["rank_histograms"]:
            assert "parameter" in hist
            assert "n_bins" in hist
            assert "chains" in hist

    def test_sampler_stats(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "num_divergences" in diag
        assert "tree_depth_mean" in diag
        assert "accept_prob_mean" in diag
        assert diag["num_chains"] == 2
        # num_samples may be None if _num_samples is not set on MCMC object
        assert diag["num_samples"] is None or diag["num_samples"] == 200

    def test_svi_returns_none(self):
        result = InferenceResult(
            _samples={"x": jnp.ones(10)},
            method="svi",
            diagnostics={},
        )
        assert result.get_mcmc_diagnostics() is None


class TestLOODiagnostics:
    def test_loo_basic(self, mcmc_result):
        key = random.PRNGKey(0)
        N = 30
        x = jnp.linspace(-2, 2, N)
        y = 1.0 + 2.5 * x + 0.5 * random.normal(key, (N,))

        loo = mcmc_result.get_loo_diagnostics(
            model_fn=_toy_model,
            observations=x,  # model takes x as first arg
            times=y,  # model takes y as second arg (obs)
        )
        assert loo is not None
        assert "elpd_loo" in loo
        assert "p_loo" in loo
        assert "se" in loo
        assert "pareto_k" in loo
        assert len(loo["pareto_k"]) == N
        assert loo["n_bad_k"] == 0  # toy model should have no bad k

    def test_loo_without_model_returns_none(self, mcmc_result):
        assert mcmc_result.get_loo_diagnostics() is None


class TestPosteriorMarginals:
    def test_marginals(self, mcmc_result):
        marginals = mcmc_result.get_posterior_marginals()
        assert len(marginals) == 3  # alpha, beta, sigma
        for m in marginals:
            assert "parameter" in m
            assert "x_values" in m
            assert "density" in m
            assert "mean" in m
            assert "hdi_3" in m


class TestPosteriorPairs:
    def test_pairs(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        # 3 scalar params -> 3 pairs (3 choose 2)
        assert len(pairs) == 3
        for p in pairs:
            assert "param_x" in p
            assert "param_y" in p
            assert len(p["x_values"]) == len(p["y_values"])
            assert len(p["x_values"]) <= 200

    def test_divergent_field(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        # Toy model should converge — either no divergent key or all False
        for p in pairs:
            if "divergent" in p:
                assert len(p["divergent"]) == len(p["x_values"])


class TestBuildEnergyDiagnostics:
    def test_energy_shape_2d(self):
        energy = jnp.ones((2, 200))
        result = _build_energy_diagnostics(energy, n_bins=20)
        assert "energy_hist" in result
        assert "energy_transition_hist" in result
        assert "bfmi" in result
        assert len(result["bfmi"]) == 2
        assert len(result["energy_hist"]["bin_centers"]) == 20
        assert len(result["energy_hist"]["density"]) == 20
        assert len(result["energy_transition_hist"]["bin_centers"]) == 20

    def test_energy_shape_1d(self):
        energy = jnp.ones(400)
        result = _build_energy_diagnostics(energy, n_bins=15)
        assert len(result["bfmi"]) == 1
        assert len(result["energy_hist"]["bin_centers"]) == 15

    def test_bfmi_reasonable(self):
        # Random energy => BFMI should be a positive finite number
        key = random.PRNGKey(99)
        energy = random.normal(key, (2, 500))
        result = _build_energy_diagnostics(energy)
        for b in result["bfmi"]:
            assert b > 0
            assert b < 10  # sanity bound


class TestEnergyInMCMCDiagnostics:
    def test_energy_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "energy" in diag
        energy = diag["energy"]
        assert "energy_hist" in energy
        assert "energy_transition_hist" in energy
        assert "bfmi" in energy
        assert len(energy["bfmi"]) == 2  # 2 chains
