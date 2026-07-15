"""model-spec tooling helpers."""

from __future__ import annotations


async def search_literature(query: str) -> str:
    """Search Exa for empirical literature, return formatted results."""
    from nof1_causal_lab.workers.prior_research import search_parameter_literature
    from nof1_causal_lab.workers.prompts.prior_research import format_literature_for_parameter

    sources = await search_parameter_literature(query)
    if not sources:
        return "No relevant literature found for this query."
    return format_literature_for_parameter(sources)
