"""Stage 4 Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search based on search_context (cacheable, run once)
2. LLM prior elicitation based on evidence (can be retried with feedback)
3. Optional AutoElicit-style paraphrased prompting for robust aggregation
"""

import asyncio
import logging

import numpy as np

from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
from causal_ssm_agent.utils.llm import WorkerGenerateFn, parse_json_response
from causal_ssm_agent.workers.prompts.prior_research import (
    SYSTEM as PRIOR_RESEARCH_SYSTEM,
)
from causal_ssm_agent.workers.prompts.prior_research import (
    USER as PRIOR_RESEARCH_USER,
)
from causal_ssm_agent.workers.prompts.prior_research import (
    format_literature_for_parameter,
    generate_paraphrased_prompts,
)
from causal_ssm_agent.workers.schemas_prior import (
    AggregatedPrior,
    PriorProposal,
    PriorResearchResult,
    PriorSource,
    RawPriorSample,
)

logger = logging.getLogger(__name__)


async def search_parameter_literature(
    parameter: ParameterSpec,
    model: str | None = None,
    timeout_ms: int | None = None,
) -> list[dict]:
    """Search Exa for literature relevant to this specific parameter.

    This is separated from elicitation so results can be cached and reused
    across retry loops without re-hitting the Exa API.

    Args:
        parameter: The parameter spec with search_context
        model: Exa model to use (reads from config if None)
        timeout_ms: Timeout for Exa research (reads from config if None)

    Returns:
        List of source dicts with title, url, snippet, effect_size
    """
    from causal_ssm_agent.utils.config import get_secret

    api_key = get_secret("EXA_API_KEY")
    if not api_key:
        return []

    # Read from config if not provided
    if model is None or timeout_ms is None:
        from causal_ssm_agent.utils.config import get_config

        config = get_config()
        lit_config = config.stage4_prior_elicitation.literature_search
        if model is None:
            model = lit_config.model
        if timeout_ms is None:
            timeout_ms = lit_config.timeout_ms

    try:
        from exa_py import AsyncExa

        exa = AsyncExa(api_key=api_key)

        # Build search instructions from parameter context
        instructions = f"""\
Search for empirical effect sizes related to:

{parameter.search_context}

Focus on:
1. Meta-analyses and systematic reviews (highest priority)
2. Large-scale longitudinal studies
3. Standardized effect sizes (Cohen's d, correlation r, standardized beta)

Report specific numerical values when available.
"""

        output_schema = {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "effect_size": {"type": "string"},
                        },
                    },
                },
            },
        }

        research = await exa.research.create(
            instructions=instructions,
            output_schema=output_schema,
            model=model,
        )

        result = await exa.research.poll_until_finished(
            research.id,
            timeout_ms=timeout_ms,
        )

        if result.status != "completed" or not result.data:
            return []

        return result.data.get("sources", [])

    except Exception:
        # Don't fail the pipeline if Exa search fails
        return []


async def _elicit_single_paraphrase(
    paraphrase_id: int,
    prompt: str,
    generate: WorkerGenerateFn,
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
        completion = await generate(messages, None)
        prior_data = parse_json_response(completion)

        # Extract mu/sigma from params
        params = prior_data.get("params", {})
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)

        return RawPriorSample(
            paraphrase_id=paraphrase_id,
            mu=mu,
            sigma=sigma,
            confidence=prior_data.get("confidence", 0.5),
            reasoning=prior_data.get("reasoning", ""),
        )
    except Exception:
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
        except Exception:
            continue

    if best_gmm is None or best_k == 1:
        # Fall back to simple if GMM fails or selects K=1
        return _aggregate_simple(mus, sigmas, samples)

    # Extract GMM parameters
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


async def elicit_prior(
    parameter: ParameterSpec,
    question: str,
    generate: WorkerGenerateFn,
    literature_context: str,
    literature_sources: list[dict] | None = None,
    feedback: str | None = None,
    n_paraphrases: int = 1,
) -> PriorResearchResult:
    """Elicit a prior for a parameter using LLM (no Exa search).

    This is the core elicitation function, separated from literature search
    so it can be called in a retry loop with validation feedback without
    re-hitting the Exa API.

    Args:
        parameter: The parameter spec from ModelSpec
        question: The research question for context
        generate: Async generate function (messages, tools) -> str
        literature_context: Pre-formatted literature evidence string
        literature_sources: Raw source dicts (for metadata)
        feedback: Optional validation feedback from a previous failed attempt
        n_paraphrases: Number of paraphrased prompts (1 = original single-shot)

    Returns:
        PriorResearchResult with proposed prior
    """
    if literature_sources is None:
        literature_sources = []

    if n_paraphrases <= 1:
        return await _research_single_prior_single_shot(
            parameter=parameter,
            question=question,
            generate=generate,
            literature_context=literature_context,
            literature_sources=literature_sources,
            feedback=feedback,
        )
    else:
        return await _research_single_prior_paraphrased(
            parameter=parameter,
            question=question,
            generate=generate,
            literature_context=literature_context,
            literature_sources=literature_sources,
            n_paraphrases=n_paraphrases,
            feedback=feedback,
        )


async def research_single_prior(
    parameter: ParameterSpec,
    question: str,
    generate: WorkerGenerateFn,
    enable_literature: bool = True,
    n_paraphrases: int = 1,
) -> PriorResearchResult:
    """Research and propose a prior for a single parameter.

    Convenience wrapper that runs both Exa search and LLM elicitation.
    For retry loops, use search_parameter_literature() + elicit_prior() separately.

    Args:
        parameter: The parameter spec from ModelSpec
        question: The research question for context
        generate: Async generate function (messages, tools) -> str
        enable_literature: Whether to search Exa for literature
        n_paraphrases: Number of paraphrased prompts (1 = original single-shot)

    Returns:
        PriorResearchResult with proposed prior
    """
    # Search literature if enabled
    literature_sources: list[dict] = []
    if enable_literature:
        literature_sources = await search_parameter_literature(parameter)

    # Format literature for prompt
    literature_context = format_literature_for_parameter(literature_sources)

    return await elicit_prior(
        parameter=parameter,
        question=question,
        generate=generate,
        literature_context=literature_context,
        literature_sources=literature_sources,
        n_paraphrases=n_paraphrases,
    )


async def _research_single_prior_single_shot(
    parameter: ParameterSpec,
    question: str,
    generate: WorkerGenerateFn,
    literature_context: str,
    literature_sources: list[dict],
    feedback: str | None = None,
) -> PriorResearchResult:
    """Original single-shot prior elicitation."""
    # Build messages
    user_content = PRIOR_RESEARCH_USER.format(
        parameter_name=parameter.name,
        parameter_role=parameter.role.value,
        parameter_constraint=parameter.constraint.value,
        parameter_description=parameter.description,
        question=question,
        literature_context=literature_context,
    )

    if feedback:
        user_content += f"\n\n## Validation Feedback (from previous attempt)\n\n{feedback}"

    messages = [
        {"role": "system", "content": PRIOR_RESEARCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    # Generate prior proposal
    completion = await generate(messages, None)

    # Parse JSON response
    prior_data = parse_json_response(completion)

    # Build sources from literature + LLM response
    sources = []
    for src in prior_data.get("sources", []):
        sources.append(
            PriorSource(
                title=src.get("title", "Unknown"),
                url=src.get("url"),
                snippet=src.get("snippet", ""),
                effect_size=src.get("effect_size"),
            )
        )

    # Warn if LLM response is missing key fields before falling back to defaults
    if "distribution" not in prior_data:
        logger.warning(
            "Prior elicitation for '%s': LLM response missing 'distribution', defaulting to Normal",
            parameter.name,
        )
    if "params" not in prior_data:
        logger.warning(
            "Prior elicitation for '%s': LLM response missing 'params', "
            "defaulting to {'mu': 0.0, 'sigma': 1.0}",
            parameter.name,
        )

    # Build prior proposal
    proposal = PriorProposal(
        parameter=parameter.name,
        distribution=prior_data.get("distribution", "Normal"),
        params=prior_data.get("params", {"mu": 0.0, "sigma": 1.0}),
        sources=sources,
        confidence=prior_data.get("confidence", 0.5),
        reasoning=prior_data.get("reasoning", ""),
    )

    return PriorResearchResult(
        parameter=parameter.name,
        proposal=proposal,
        literature_found=len(literature_sources) > 0,
        raw_response=completion,
    )


async def _research_single_prior_paraphrased(
    parameter: ParameterSpec,
    question: str,
    generate: WorkerGenerateFn,
    literature_context: str,
    literature_sources: list[dict],
    n_paraphrases: int,
    feedback: str | None = None,
) -> PriorResearchResult:
    """AutoElicit-style paraphrased prior elicitation with GMM aggregation."""
    # Generate paraphrased prompts
    prompts = generate_paraphrased_prompts(
        parameter_name=parameter.name,
        parameter_role=parameter.role.value,
        parameter_constraint=parameter.constraint.value,
        parameter_description=parameter.description,
        question=question,
        literature_context=literature_context,
        n_paraphrases=n_paraphrases,
    )

    # Append validation feedback to each prompt if provided
    if feedback:
        feedback_section = f"\n\n## Validation Feedback (from previous attempt)\n\n{feedback}"
        prompts = [p + feedback_section for p in prompts]

    # Elicit priors in parallel
    tasks = [_elicit_single_paraphrase(i, prompt, generate) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # Filter out failed elicitations
    samples = [r for r in results if r is not None]

    if not samples:
        # All paraphrases failed, fall back to single-shot
        return await _research_single_prior_single_shot(
            parameter=parameter,
            question=question,
            generate=generate,
            literature_context=literature_context,
            literature_sources=literature_sources,
            feedback=feedback,
        )

    # Aggregate samples using GMM
    aggregated = aggregate_prior_samples(samples)

    # Build proposal using aggregated values
    proposal = PriorProposal(
        parameter=parameter.name,
        distribution="Normal",  # Aggregation produces Normal params
        params={"mu": aggregated.mu, "sigma": aggregated.sigma},
        sources=[],  # Sources come from literature, not paraphrases
        confidence=float(np.mean([s.confidence for s in samples])),
        reasoning=f"Aggregated from {len(samples)} paraphrased elicitations using GMM.",
    )

    return PriorResearchResult(
        parameter=parameter.name,
        proposal=proposal,
        literature_found=len(literature_sources) > 0,
        raw_response=f"Aggregated {len(samples)} paraphrase responses",
        aggregation=aggregated,
    )


def get_default_prior(parameter: ParameterSpec) -> PriorProposal:
    """Get a default prior when research fails.

    Args:
        parameter: The parameter spec

    Returns:
        Default PriorProposal based on parameter role/constraint
    """
    from causal_ssm_agent.orchestrator.schemas_model import ParameterConstraint, ParameterRole

    # Choose distribution based on constraint
    if parameter.constraint == ParameterConstraint.POSITIVE:
        distribution = "HalfNormal"
        params = {"sigma": 1.0}
    elif parameter.constraint == ParameterConstraint.UNIT_INTERVAL:
        distribution = "Beta"
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.CORRELATION:
        distribution = "Uniform"
        params = {"lower": -1.0, "upper": 1.0}
    else:
        distribution = "Normal"
        params = {"mu": 0.0, "sigma": 0.5}

    # Adjust based on role
    if parameter.role == ParameterRole.AR_COEFFICIENT:
        distribution = "Beta"
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.role == ParameterRole.RESIDUAL_SD:
        distribution = "HalfNormal"
        params = {"sigma": 1.0}

    return PriorProposal(
        parameter=parameter.name,
        distribution=distribution,
        params=params,
        sources=[],
        confidence=0.3,  # Low confidence for default
        reasoning=f"Default prior for {parameter.role.value} parameter",
    )
