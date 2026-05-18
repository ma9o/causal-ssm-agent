"""Stage 1a: Latent Model Proposal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompting import templates

if TYPE_CHECKING:
    from causal_ssm_agent.utils.agent_session import StageSessionFactory


@dataclass
class Stage1aResult:
    """Result of Stage 1a: latent model proposal."""

    latent_model: dict

    @property
    def n_constructs(self) -> int:
        """Number of constructs in the model."""
        return len(self.latent_model.get("constructs", []))

    @property
    def n_edges(self) -> int:
        """Number of edges in the model."""
        return len(self.latent_model.get("edges", []))


async def run_stage1a(
    question: str,
    session_factory: StageSessionFactory,
) -> Stage1aResult:
    """Run the Stage 1a flow: latent model proposal + self-review.

    Opens one :class:`AgentSession` with the validation tool bound, sends
    the proposal prompt, then sends the review follow-up in the same
    session so the model keeps its prior reasoning context.
    """
    from causal_ssm_agent.flows.stage_tool_factory import make_stage_tool
    from causal_ssm_agent.flows.stages.stage1a.grounding import stage1a_grounding

    tool, capture = make_stage_tool(
        name="validate_latent_model",
        description="Validate latent model JSON.",
        param_name="structure_json",
        param_description="The JSON string containing the latent model.",
        compute_fn=stage1a_grounding,
    )

    async with session_factory.open(
        system_prompt=templates.SYSTEM,
        tools=[tool],
        log_label="stage-1a",
    ) as session:
        await session.turn(templates.USER.format(question=question))
        await session.turn(templates.REVIEW)

    if not capture.get("latent_model"):
        raise ValueError("No valid latent model produced")

    return Stage1aResult(latent_model=capture["latent_model"])
