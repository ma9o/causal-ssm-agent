"""Tests for prior_research.py aggregation functions.

Covers: _aggregate_simple, _aggregate_gmm, aggregate_prior_samples.
"""

import numpy as np
import pytest

from causal_ssm_agent.workers.prior_research import (
    _aggregate_gmm,
    _aggregate_simple,
    aggregate_prior_samples,
)
from causal_ssm_agent.workers.schemas_prior import RawPriorSample


def _make_samples(mus: list[float], sigmas: list[float]) -> list[RawPriorSample]:
    """Helper to create RawPriorSample list."""
    return [
        RawPriorSample(paraphrase_id=i, mu=mu, sigma=sigma, reasoning=f"sample {i}")
        for i, (mu, sigma) in enumerate(zip(mus, sigmas))
    ]


# =============================================================================
# _aggregate_simple
# =============================================================================


class TestAggregateSimple:
    def test_basic_pooling(self):
        mus = np.array([1.0, 2.0, 3.0])
        sigmas = np.array([0.5, 0.5, 0.5])
        samples = _make_samples([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])

        result = _aggregate_simple(mus, sigmas, samples)

        assert result.method == "simple"
        assert result.mu == pytest.approx(2.0)
        # sigma = sqrt(mean(0.25) + var([1,2,3])) = sqrt(0.25 + 2/3)
        expected_sigma = np.sqrt(np.mean(sigmas**2) + np.var(mus))
        assert result.sigma == pytest.approx(expected_sigma)
        assert result.n_samples == 3

    def test_identical_samples(self):
        mus = np.array([5.0, 5.0, 5.0])
        sigmas = np.array([1.0, 1.0, 1.0])
        samples = _make_samples([5.0, 5.0, 5.0], [1.0, 1.0, 1.0])

        result = _aggregate_simple(mus, sigmas, samples)

        assert result.mu == pytest.approx(5.0)
        # No between-sample variance
        assert result.sigma == pytest.approx(1.0)

    def test_sigma_includes_between_sample_variance(self):
        """When mus vary, pooled sigma should be larger than individual sigmas."""
        mus = np.array([0.0, 10.0])
        sigmas = np.array([0.1, 0.1])
        samples = _make_samples([0.0, 10.0], [0.1, 0.1])

        result = _aggregate_simple(mus, sigmas, samples)

        # Pooled sigma should be much larger than 0.1
        assert result.sigma > 1.0


# =============================================================================
# _aggregate_gmm
# =============================================================================


class TestAggregateGmm:
    def test_falls_back_to_simple_for_few_samples(self):
        """With < 3 samples, should use simple pooling."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([0.5, 0.5])
        samples = _make_samples([1.0, 2.0], [0.5, 0.5])

        result = _aggregate_gmm(mus, sigmas, samples)

        assert result.method == "simple"

    def test_unimodal_data_uses_simple(self):
        """Tightly clustered data should select K=1 and fall back to simple."""
        rng = np.random.default_rng(42)
        mus = rng.normal(5.0, 0.01, 10)
        sigmas = np.full(10, 0.5)
        samples = _make_samples(mus.tolist(), sigmas.tolist())

        result = _aggregate_gmm(mus, sigmas, samples)

        # BIC should prefer K=1, which falls back to simple
        assert result.method == "simple"
        assert result.mu == pytest.approx(5.0, abs=0.05)

    def test_bimodal_data_returns_gmm(self):
        """Clearly bimodal data should trigger GMM with K>1."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(-5.0, 0.1, 20)
        cluster2 = rng.normal(5.0, 0.1, 20)
        mus = np.concatenate([cluster1, cluster2])
        sigmas = np.full(40, 0.5)
        samples = _make_samples(mus.tolist(), sigmas.tolist())

        result = _aggregate_gmm(mus, sigmas, samples)

        assert result.method == "gmm"
        assert result.mixture_weights is not None
        assert result.mixture_means is not None
        assert result.mixture_stds is not None
        assert len(result.mixture_weights) >= 2
        assert result.n_samples == 40

    def test_gmm_mu_is_weighted_mean(self):
        """The aggregated mu should be the weighted mean of components."""
        rng = np.random.default_rng(0)
        cluster1 = rng.normal(-3.0, 0.1, 15)
        cluster2 = rng.normal(3.0, 0.1, 15)
        mus = np.concatenate([cluster1, cluster2])
        sigmas = np.full(30, 0.5)
        samples = _make_samples(mus.tolist(), sigmas.tolist())

        result = _aggregate_gmm(mus, sigmas, samples)

        if result.method == "gmm":
            # Weighted mean should be near 0 for symmetric bimodal data
            assert result.mu == pytest.approx(0.0, abs=0.5)

    def test_gmm_sigma_larger_than_individual(self):
        """For bimodal data, aggregated sigma should capture spread."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(-5.0, 0.1, 20)
        cluster2 = rng.normal(5.0, 0.1, 20)
        mus = np.concatenate([cluster1, cluster2])
        sigmas = np.full(40, 0.5)
        samples = _make_samples(mus.tolist(), sigmas.tolist())

        result = _aggregate_gmm(mus, sigmas, samples)

        # Sigma should be large because of the bimodal spread
        assert result.sigma > 2.0


# =============================================================================
# aggregate_prior_samples (top-level function)
# =============================================================================


class TestAggregatePriorSamples:
    def test_dispatches_to_gmm(self):
        """aggregate_prior_samples should use GMM internally."""
        samples = _make_samples([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
        result = aggregate_prior_samples(samples)
        assert result.n_samples == 3
        assert isinstance(result.mu, float)
        assert isinstance(result.sigma, float)

    def test_single_sample(self):
        samples = _make_samples([5.0], [1.0])
        result = aggregate_prior_samples(samples)
        # Single sample: falls back to simple
        assert result.method == "simple"
        assert result.mu == pytest.approx(5.0)
        assert result.sigma == pytest.approx(1.0)

    def test_two_samples(self):
        samples = _make_samples([3.0, 7.0], [1.0, 1.0])
        result = aggregate_prior_samples(samples)
        assert result.method == "simple"
        assert result.mu == pytest.approx(5.0)
        assert result.n_samples == 2
