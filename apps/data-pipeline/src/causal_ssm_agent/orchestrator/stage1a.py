"""Stage 1a: Latent Model Proposal.

Core logic for Stage 1a, decoupled from Prefect/Inspect frameworks.
Uses dependency injection for the LLM generate function.
"""

from dataclasses import dataclass

from causal_ssm_agent.utils.llm import (
    OrchestratorGenerateFn,
    make_validate_latent_model_tool,
    parse_json_response,
)

from .prompts import latent_model
from .schemas import LatentModel


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
        Stage1aResult with the latent model
    """
    msgs = Stage1aMessages(question)

    # Step 1: Initial proposal with self-review
    # The tool captures the last valid LatentModel so we don't depend
    # on the final completion being valid JSON (the review follow-up
    # may return prose or empty).
    proposal_msgs = msgs.proposal_messages()
    tool, capture = make_validate_latent_model_tool()

    completion = await generate(proposal_msgs, [tool], [latent_model.REVIEW])

    # Prefer the captured result from the validation tool
    latent = capture.get("latent")
    if latent is None:
        # Fallback: try parsing the final completion directly
        latent = parse_json_response(completion)
    LatentModel.model_validate(latent)  # Validate schema

    return Stage1aResult(latent_model=latent)
