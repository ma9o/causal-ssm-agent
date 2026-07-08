"""model-spec tooling helpers."""

from __future__ import annotations

from typing import Any


async def search_literature(query: str) -> str:
    """Search Exa for empirical literature, return formatted results."""
    from nof1_causal_lab.workers.prior_research import search_parameter_literature
    from nof1_causal_lab.workers.prompts.prior_research import format_literature_for_parameter

    sources = await search_parameter_literature(query)
    if not sources:
        return "No relevant literature found for this query."
    return format_literature_for_parameter(sources)


def make_search_tool(state: Any) -> Any:
    """Create a search_literature Tool for pipeline use."""
    from nof1_causal_lab.utils.openrouter_client import Tool

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
