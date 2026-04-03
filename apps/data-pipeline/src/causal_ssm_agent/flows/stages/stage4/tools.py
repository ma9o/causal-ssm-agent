"""Stage 4 tooling helpers."""

from __future__ import annotations

from typing import Any


async def search_literature(query: str) -> str:
    """Search Exa for empirical literature, return formatted results."""
    from causal_ssm_agent.workers.prior_research import search_parameter_literature
    from causal_ssm_agent.workers.prompts.prior_research import format_literature_for_parameter

    sources = await search_parameter_literature(query)
    if not sources:
        return "No relevant literature found for this query."
    return format_literature_for_parameter(sources)


def make_search_tool(state: Any) -> Any:
    """Create a search_literature Tool for pipeline use."""
    from causal_ssm_agent.utils.openrouter_client import Tool

    async def _execute(*, query: str, parameter_name: str) -> str:
        state.search_queries[parameter_name] = query
        cached = state.search_cache.get(query)
        if cached is not None:
            return cached
        result = await search_literature(query)
        state.search_cache[query] = result
        return result

    return Tool(
        name="search_literature",
        description="Search for empirical literature about effect sizes for model parameters.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for empirical literature about effect sizes.",
                },
                "parameter_name": {
                    "type": "string",
                    "description": "Name of the parameter this search is for (e.g. 'beta_stress_sleep').",
                },
            },
            "required": ["query", "parameter_name"],
            "additionalProperties": False,
        },
        execute=_execute,
    )


def make_elicit_prior_gmm_tool(
    question: str,
    model_name: str,
    n_paraphrases: int = 10,
    max_tool_turns: int = 40,
) -> Any:
    """Create an ``elicit_prior_gmm`` tool for the agentic Stage 4 flow."""
    from causal_ssm_agent.utils.openrouter_client import Tool

    async def _execute(
        *,
        parameter_name: str,
        parameter_role: str,
        parameter_constraint: str,
        context: str,
    ) -> str:
        from causal_ssm_agent.workers.prior_research import run_gmm_elicitation

        return await run_gmm_elicitation(
            parameter_name=parameter_name,
            parameter_role=parameter_role,
            parameter_constraint=parameter_constraint,
            context=context,
            question=question,
            model_name=model_name,
            n_paraphrases=n_paraphrases,
            max_tool_turns=max_tool_turns,
        )

    return Tool(
        name="elicit_prior_gmm",
        description=(
            "Run robust paraphrased prior elicitation with GMM aggregation "
            "for a single parameter. Returns an aggregated prior estimate."
        ),
        parameters={
            "type": "object",
            "properties": {
                "parameter_name": {
                    "type": "string",
                    "description": "Name of the parameter (e.g. 'beta_stress_depression')",
                },
                "parameter_role": {
                    "type": "string",
                    "description": (
                        "Role such as fixed_effect, ar_coefficient, residual_sd, "
                        "measurement_error_sd, loading, or observation_hyperparameter"
                    ),
                },
                "parameter_constraint": {
                    "type": "string",
                    "description": "Constraint: none, positive, unit_interval, correlation",
                },
                "context": {
                    "type": "string",
                    "description": "What this parameter represents, for literature grounding",
                },
            },
            "required": ["parameter_name", "parameter_role", "parameter_constraint", "context"],
            "additionalProperties": False,
        },
        execute=_execute,
    )
