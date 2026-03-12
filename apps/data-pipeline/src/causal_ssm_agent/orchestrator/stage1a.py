"""Stage 1a: Latent Model Proposal.

Core logic for Stage 1a, decoupled from Prefect and model-client frameworks.
Uses dependency injection for the LLM generate function.
"""

from dataclasses import dataclass

from causal_ssm_agent.utils.llm import OrchestratorGenerateFn

from .prompts import latent_model


@dataclass
class Stage1aResult:
    """Result of Stage 1a: latent model proposal."""

    latent_model: dict
    outcome_name: str
    treatments: list[str]

    @property
    def n_constructs(self) -> int:
        """Number of constructs in the model."""
        return len(self.latent_model.get("constructs", []))

    @property
    def n_edges(self) -> int:
        """Number of edges in the model."""
        return len(self.latent_model.get("edges", []))


@dataclass
class Stage1aMessages:
    """Message builders for Stage 1a prompts."""

    question: str

    def proposal_messages(self) -> list[dict]:
        """Build messages for initial latent model proposal."""
        return [
            {"role": "system", "content": latent_model.SYSTEM},
            {"role": "user", "content": latent_model.USER.format(question=self.question)},
        ]


async def run_stage1a(
    question: str,
    generate: OrchestratorGenerateFn,
) -> Stage1aResult:
    """
    Run the full Stage 1a flow: latent model proposal with self-review.

    This is the core logic, decoupled from any framework. The caller provides
    a `generate` function that handles LLM calls.

    Args:
        question: The causal research question
        generate: Async function (messages, tools, follow_ups) -> completion

    Returns:
        Stage1aResult with latent model, outcome_name, and treatments
    """
    from causal_ssm_agent.flows.stages.stage_tools import make_stage_tool, stage1a_grounding

    msgs = Stage1aMessages(question)

    tool, capture = make_stage_tool(
        name="validate_latent_model",
        description="Validate latent model JSON. Returns outcome and treatments on success.",
        param_name="structure_json",
        param_description="The JSON string containing the latent model.",
        compute_fn=stage1a_grounding,
    )

    await generate(msgs.proposal_messages(), [tool], [latent_model.REVIEW])

    if not capture.get("latent_model"):
        raise ValueError("No valid latent model produced")

    return Stage1aResult(
        latent_model=capture["latent_model"],
        outcome_name=capture["outcome_name"],
        treatments=capture["treatments"],
    )
