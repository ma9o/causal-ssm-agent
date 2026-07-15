"""model-spec Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search (cacheable, run once)
2. LLM prior elicitation based on evidence (can be retried with feedback)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from nof1_causal_lab.utils.openrouter_client import acquire_limiter

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.workers.schemas_prior import (
    PriorProposal,
    prior_params_model,
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
        params=prior_params_model(distribution, params),
        sources=[],
        reasoning=f"Default prior for {parameter.role.value} parameter",
    )
