"""Stage 4 Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search (cacheable, run once)
2. LLM prior elicitation based on evidence (can be retried with feedback)
3. Optional AutoElicit-style paraphrased prompting for robust aggregation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
import numpy as np

from nof1_causal_lab.utils.openrouter_client import acquire_limiter

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.workers.schemas_prior import (
    AggregatedPrior,
    PriorProposal,
    RawPriorSample,
)

logger = logging.getLogger(__name__)


async def search_parameter_literature(
    query: str,
) -> list[dict]:
    """Search Exa for literature relevant to a query.

    This is separated from elicitation so results can be cached and reused
    across retry loops without re-hitting the Exa API.

    Uses Exa deep search with structured output schema.

    Args:
        query: Search query string

    Returns:
        List of source dicts with title, url, snippet, effect_size
    """
    from nof1_causal_lab.utils.config import get_secret_async

    api_key = await get_secret_async("EXA_API_KEY")
    if not api_key:
        return []

    try:
        from exa_py import AsyncExa

        exa = AsyncExa(api_key=api_key)

        exa_query = (
            f"Empirical effect sizes related to: {query}. "
            "Meta-analyses, systematic reviews, standardized effect sizes."
        )

        await acquire_limiter("exa")

        result = await exa.search_and_contents(
            exa_query,
            num_results=5,
            type="auto",
            highlights=True,
            category="research paper",
        )

        sources = []
        for r in result.results:
            sources.append(
                {
                    "title": r.title or "",
                    "url": r.url or "",
                    "snippet": " ".join(r.highlights) if r.highlights else (r.text or "")[:500],
                    "effect_size": "",
                }
            )
        return sources

    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Exa search failed; continuing without search results: %s", exc)
        return []


def aggregate_prior_samples(
    samples: list[RawPriorSample],
) -> AggregatedPrior:
    """Aggregate multiple prior elicitations into a single prior using GMM.

    Uses Gaussian Mixture Model with BIC model selection (K=1,2,3).
    Falls back to simple pooling if GMM fails or selects K=1.

    Args:
        samples: List of raw prior samples from paraphrased prompts

    Returns:
        AggregatedPrior with aggregated parameters
    """
    mus = np.array([s.mu for s in samples])
    sigmas = np.array([s.sigma for s in samples])

    return _aggregate_gmm(mus, sigmas, samples)


def _aggregate_simple(
    mus: np.ndarray,
    sigmas: np.ndarray,
    samples: list[RawPriorSample],
) -> AggregatedPrior:
    """Simple pooling: mu_pooled = mean(mu_k), sigma_pooled = sqrt(mean(sigma_k^2) + var(mu_k))."""
    mu_pooled = float(np.mean(mus))
    # Total variance = within-sample variance + between-sample variance
    sigma_pooled = float(np.sqrt(np.mean(sigmas**2) + np.var(mus)))

    return AggregatedPrior(
        method="simple",
        mu=mu_pooled,
        sigma=sigma_pooled,
        n_samples=len(samples),
    )


def _aggregate_gmm(
    mus: np.ndarray,
    sigmas: np.ndarray,
    samples: list[RawPriorSample],
) -> AggregatedPrior:
    """GMM aggregation with BIC model selection for multimodal detection."""
    from sklearn.mixture import GaussianMixture

    # Need at least 3 samples to fit GMM
    if len(mus) < 3:
        return _aggregate_simple(mus, sigmas, samples)

    # Reshape for sklearn
    X = mus.reshape(-1, 1)

    # Try K=1,2,3 and select by BIC
    best_bic = np.inf
    best_gmm = None
    best_k = 1

    for k in range(1, min(4, len(mus))):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            logger.warning("GMM fitting failed for k=%d: %s", k, exc)
            continue

    if best_gmm is None or best_k == 1:
        # Fall back to simple if GMM fails or selects K=1
        return _aggregate_simple(mus, sigmas, samples)

    # Extract GMM parameters (always populated after fit)
    assert best_gmm.weights_ is not None
    assert best_gmm.means_ is not None
    assert best_gmm.covariances_ is not None
    weights = best_gmm.weights_.tolist()
    means = best_gmm.means_.flatten().tolist()
    stds = np.sqrt(best_gmm.covariances_.flatten()).tolist()

    # For mu, use weighted mean of GMM components
    mu_pooled = float(np.sum(best_gmm.weights_ * best_gmm.means_.flatten()))

    # For sigma, combine GMM variance with between-sample variance
    gmm_variance = float(
        np.sum(
            best_gmm.weights_
            * (best_gmm.covariances_.flatten() + (best_gmm.means_.flatten() - mu_pooled) ** 2)
        )
    )
    sigma_pooled = float(np.sqrt(gmm_variance + np.mean(sigmas**2)))

    return AggregatedPrior(
        method="gmm",
        mu=mu_pooled,
        sigma=sigma_pooled,
        mixture_weights=weights,
        mixture_means=means,
        mixture_stds=stds,
        n_samples=len(samples),
    )


def get_default_prior(parameter: ParameterSpec) -> PriorProposal:
    """Get a default prior when research fails.

    Args:
        parameter: The parameter spec

    Returns:
        Default PriorProposal based on parameter role/constraint
    """
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterConstraint, ParameterRole

    # AR priors live on the baseline DT persistence scale in (0, 1).
    if parameter.role == ParameterRole.AR_COEFFICIENT:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.POSITIVE:
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}
    elif parameter.constraint == ParameterConstraint.NEGATIVE:
        distribution = PriorDistributionFamily.TRUNCATED_NORMAL
        params = {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0}
    elif parameter.constraint == ParameterConstraint.UNIT_INTERVAL:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.CORRELATION:
        distribution = PriorDistributionFamily.UNIFORM
        params = {"lower": -1.0, "upper": 1.0}
    else:
        distribution = PriorDistributionFamily.NORMAL
        params = {"mu": 0.0, "sigma": 0.5}

    # Adjust based on role
    if parameter.role in (ParameterRole.RESIDUAL_SD, ParameterRole.STATIC_STATE_SD):
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}

    return PriorProposal(
        parameter=parameter.name,
        distribution=distribution,
        params=params,
        sources=[],
        reasoning=f"Default prior for {parameter.role.value} parameter",
    )
