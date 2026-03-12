"""Stage 1b: Measurement Model with Identifiability.

Core logic for Stage 1b, decoupled from Prefect and model-client frameworks.
Uses dependency injection for the LLM generate function.

The fat validation tool checks both structural validity AND causal identifiability,
giving the LLM rich feedback to self-correct (add proxies, fix indicators) within
a single tool loop — no imperative multi-step orchestration.
"""

import json
import logging
from dataclasses import dataclass

from causal_ssm_agent.utils.identifiability import analyze_unobserved_constructs
from causal_ssm_agent.utils.llm import OrchestratorGenerateFn

from .prompts import measurement_model

logger = logging.getLogger(__name__)


@dataclass
class Stage1bResult:
    """Result of Stage 1b: measurement model with identifiability status."""

    measurement_model: dict
    identifiability_status: dict
    causal_spec: dict
    marginalization_analysis: dict | None = None


@dataclass
class Stage1bMessages:
    """Message builders for Stage 1b prompts."""

    question: str
    latent_model: dict
    chunks: list[str]
    dataset_summary: str = ""

    def proposal_messages(self) -> list[dict]:
        """Build messages for initial measurement proposal."""
        return [
            {"role": "system", "content": measurement_model.SYSTEM},
            {
                "role": "user",
                "content": measurement_model.USER.format(
                    question=self.question,
                    latent_model_json=json.dumps(self.latent_model, indent=2),
                    dataset_summary=self.dataset_summary or "Not provided",
                    chunks="\n".join(self.chunks),
                ),
            },
        ]


async def run_stage1b(
    question: str,
    latent_model: dict,
    chunks: list[str],
    generate: OrchestratorGenerateFn,
    dataset_summary: str = "",
) -> Stage1bResult:
    """
    Run the full Stage 1b flow: measurement proposal with identifiability checking.

    The fat validation tool checks both structural validity and causal identifiability.
    When identifiability fails, it returns rich feedback so the LLM can add proxy
    indicators and resubmit — all within a single tool loop.

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        chunks: Data chunks for operationalization
        generate: Async function (messages, tools, follow_ups) -> completion
        dataset_summary: Optional description of the dataset

    Returns:
        Stage1bResult with measurement model, identifiability, and marginalization
    """
    from causal_ssm_agent.flows.stages.stage_tools import make_stage_tool, stage1b_grounding

    msgs = Stage1bMessages(question, latent_model, chunks, dataset_summary)

    # Single fat tool — validates structure + checks identifiability
    tool, capture = make_stage_tool(
        name="validate_measurement_model",
        description="Validate measurement model JSON, check compiler constraints, and verify causal identifiability.",
        param_name="measurement_json",
        param_description="The JSON string containing the measurement model.",
        compute_fn=lambda data: stage1b_grounding(data, latent_model),
    )

    await generate(msgs.proposal_messages(), [tool], [measurement_model.REVIEW])

    causal_spec = capture.get("causal_spec")
    if causal_spec is None:
        raise ValueError("No valid measurement model produced")

    # Extract fields from the captured causal_spec
    measurement = causal_spec.get("measurement", {})
    id_info = causal_spec.get("identifiability") or {}
    id_status = {
        "identifiable_treatments": id_info.get("identifiable_treatments", {}),
        "non_identifiable_treatments": id_info.get("non_identifiable_treatments", {}),
    }

    # Deterministic post-processing: which unobserved constructs can be marginalized
    marginalization = analyze_unobserved_constructs(latent_model, measurement, id_status)

    return Stage1bResult(
        measurement_model=measurement,
        identifiability_status=id_status,
        causal_spec=causal_spec,
        marginalization_analysis=marginalization,
    )
