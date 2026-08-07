"""Single-source registry for the public model-spec literature-search tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

if TYPE_CHECKING:
    from nof1_causal_lab.flows.artifact_contracts import ToolContract


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for empirical literature about effect sizes.")


async def execute_public_search_literature(
    _ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    """Execute the public model-spec literature-search tool."""
    from nof1_causal_lab.flows.transitions.model_spec.tools import search_literature

    query = args.get("query", "")
    if not query:
        return {"result": "Error: query is required"}
    result = await search_literature(query)
    return {"result": result}


def build_model_spec_public_tool_contracts() -> list[ToolContract]:
    """Materialize the stateless public model-spec tool contract."""
    from nof1_causal_lab.flows.contracts_base import ToolContract

    return [
        ToolContract(
            name="search_literature",
            description="Search for empirical literature about effect sizes for model parameters.",
            input_schema=SearchLiteratureInput,
        )
    ]
