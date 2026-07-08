"""latent-structure: Latent Structure Proposal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompting import templates

if TYPE_CHECKING:
    from nof1_causal_lab.utils.agent_session import ScopedSessionFactory


@dataclass
class LatentStructureResult:
    """Result of latent-structure: latent structure proposal."""

    latent_structure: dict

    @property
    def n_constructs(self) -> int:
        """Number of constructs in the model."""
        return len(self.latent_structure.get("constructs", []))

    @property
    def n_edges(self) -> int:
        """Number of edges in the model."""
        return len(self.latent_structure.get("edges", []))


async def run_latent_structure(
    question: str,
    session_factory: ScopedSessionFactory,
) -> LatentStructureResult:
    """Run the latent-structure flow: latent structure proposal + self-review.

    Opens one :class:`AgentSession` with the validation tool bound, sends
    the proposal prompt, then sends the review follow-up in the same
    session so the model keeps its prior reasoning context.
    """
    from nof1_causal_lab.flows.context_tool_factory import make_context_tool
    from nof1_causal_lab.flows.transitions.latent_structure.grounding import (
        latent_structure_grounding,
    )

    tool, capture = make_context_tool(
        name="validate_latent_structure",
        description="Validate latent structure JSON.",
        param_name="structure_json",
        param_description="The JSON string containing the latent structure.",
        compute_fn=latent_structure_grounding,
    )

    async with session_factory.open(
        system_prompt=templates.SYSTEM,
        tools=[tool],
        log_label="latent-structure",
    ) as session:
        await session.turn(templates.USER.format(question=question))
        await session.turn(templates.REVIEW)

    if not capture.get("latent_structure"):
        raise ValueError("No valid latent structure produced")

    return LatentStructureResult(latent_structure=capture["latent_structure"])
