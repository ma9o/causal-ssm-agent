"""Stage 4 Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search (cacheable, run once)
2. LLM prior elicitation based on evidence (can be retried with feedback)
3. Optional AutoElicit-style paraphrased prompting for robust aggregation
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import numpy as np
from pydantic import ValidationError

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.openrouter_client import acquire_limiter

if TYPE_CHECKING:
    from causal_ssm_agent.artifacts.model_spec import ParameterSpec
from causal_ssm_agent.distributions import PriorDistributionFamily
from causal_ssm_agent.utils.llm import (
    GenerateFn,
    make_validation_tool,
    parse_json_response,
)
from causal_ssm_agent.workers.prompts.prior_research import (
    SYSTEM as PRIOR_RESEARCH_SYSTEM,
)
from causal_ssm_agent.workers.prompts.prior_research import (
    generate_paraphrased_prompts,
)
from causal_ssm_agent.workers.schemas_prior import (
    AggregatedPrior,
    PriorProposal,
    RawPriorSample,
)

logger = get_prefect_logger(__name__)


def _make_prior_tool() -> tuple[object, dict]:
    """Create a validation tool for prior proposals using PriorProposal schema."""

    def _validate(data: dict) -> tuple[object, list[str]]:
        try:
            validated = PriorProposal.model_validate(data)
            return validated.model_dump(), []
        except ValidationError as e:
            return None, [str(e)]

    return make_validation_tool(
        name="validate_prior_proposal",
        description="Validate prior distribution proposal JSON.",
        param_name="prior_json",
        param_description="The JSON string containing the prior proposal.",
        validator=_validate,
        capture_key="prior",
        capture_result=True,
    )


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
    from causal_ssm_agent.utils.config import get_secret_async

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


async def run_gmm_elicitation(
    parameter_name: str,
    parameter_role: str,
    parameter_constraint: str,
    context: str,
    question: str,
    model_name: str,
    n_paraphrases: int = 10,
    max_tool_turns: int = 40,
) -> str:
    """Run paraphrased prior elicitation and return GMM-aggregated result.

    Used as the implementation behind the ``elicit_prior_gmm`` tool in the
    agentic Stage 4 flow.

    Returns:
        Formatted string with the aggregated prior for the LLM to use.
    """
    from causal_ssm_agent.utils.llm import make_generate_fn

    prompts = generate_paraphrased_prompts(
        parameter_name=parameter_name,
        parameter_role=parameter_role,
        parameter_constraint=parameter_constraint,
        parameter_description=context,
        question=question,
        literature_context="",
        n_paraphrases=n_paraphrases,
    )

    generate = make_generate_fn(model_name, max_tool_turns=max_tool_turns)
    tasks = [_elicit_single_paraphrase(i, prompt, generate) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    samples = [r for r in results if r is not None]
    if not samples:
        return (
            f"GMM elicitation failed for {parameter_name}: "
            "all paraphrase prompts returned errors. Use domain reasoning."
        )

    aggregated = aggregate_prior_samples(samples)

    lines = [
        f"GMM-aggregated prior for '{parameter_name}' "
        f"(from {aggregated.n_samples} prompts, method={aggregated.method}):",
        f"  Pooled: Normal(mu={aggregated.mu:.4f}, sigma={aggregated.sigma:.4f})",
    ]
    if aggregated.mixture_weights and len(aggregated.mixture_weights) > 1:
        for k, (w, m, s) in enumerate(
            zip(
                aggregated.mixture_weights,
                aggregated.mixture_means or [],
                aggregated.mixture_stds or [],
                strict=True,
            )
        ):
            lines.append(f"  Component {k + 1} (w={w:.2f}): Normal(mu={m:.4f}, sigma={s:.4f})")

    return "\n".join(lines)


async def _elicit_single_paraphrase(
    paraphrase_id: int,
    prompt: str,
    generate: GenerateFn,
) -> RawPriorSample | None:
    """Elicit a prior from a single paraphrased prompt.

    Args:
        paraphrase_id: Index of the paraphrase template
        prompt: The formatted prompt to use
        generate: Async generate function

    Returns:
        RawPriorSample or None if parsing fails
    """
    messages = [
        {"role": "system", "content": PRIOR_RESEARCH_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        tool, capture = _make_prior_tool()
        completion = await generate(messages, [tool])

        prior_data = capture.get("prior")
        if prior_data is None:
            prior_data = parse_json_response(completion)

        # Extract mu/sigma from params
        params = prior_data.get("params", {})
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)

        return RawPriorSample(
            paraphrase_id=paraphrase_id,
            mu=mu,
            sigma=sigma,
            reasoning=prior_data.get("reasoning", ""),
        )
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "Failed to parse prior elicitation response for paraphrase %d: %s",
            paraphrase_id,
            exc,
        )
        return None


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
    from causal_ssm_agent.artifacts.model_spec import ParameterConstraint, ParameterRole

    # AR priors live on the DT persistence scale in (0, 1).
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
