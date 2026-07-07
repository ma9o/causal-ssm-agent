"""Stage 1b: Measurement Structure with Identifiability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompting import templates

if TYPE_CHECKING:
    from nof1_causal_lab.utils.agent_session import StageSessionFactory

logger = logging.getLogger(__name__)


@dataclass
class Stage1bResult:
    """Result of Stage 1b: measurement structure with identifiability status."""

    measurement_structure: dict
    identifiability_status: dict
    causal_design: dict


def _build_stage1b_user_prompt(
    question: str,
    latent_structure: dict,
    chunks: list[str],
    dataset_summary: str,
) -> str:
    return templates.USER.format(
        question=question,
        latent_structure_json=json.dumps(latent_structure, indent=2),
        dataset_summary=dataset_summary or "Not provided",
        chunks="\n".join(chunks),
    )


async def run_stage1b(
    question: str,
    latent_structure: dict,
    chunks: list[str],
    session_factory: StageSessionFactory,
    dataset_summary: str = "",
) -> Stage1bResult:
    """Run the Stage 1b flow: measurement structure + identifiability check.

    The fat validation tool checks structural validity and causal
    identifiability together. When identifiability fails, the tool
    returns rich feedback the model can use to add proxy indicators in
    the next turn — all within one session.
    """
    from nof1_causal_lab.flows.stage_tool_factory import make_stage_tool
    from nof1_causal_lab.flows.stages.stage1b.grounding import stage1b_grounding

    tool, capture = make_stage_tool(
        name="validate_measurement_structure",
        description=(
            "Validate measurement structure JSON, check compiler constraints, "
            "and verify causal identifiability."
        ),
        param_name="measurement_json",
        param_description="The JSON string containing the measurement structure.",
        compute_fn=lambda data: stage1b_grounding(data, latent_structure),
    )

    async with session_factory.open(
        system_prompt=templates.SYSTEM,
        tools=[tool],
        log_label="stage-1b",
    ) as session:
        await session.turn(
            _build_stage1b_user_prompt(question, latent_structure, chunks, dataset_summary)
        )
        await session.turn(templates.REVIEW)

    causal_design = capture.get("causal_design")
    if causal_design is None:
        raise ValueError("No valid measurement structure produced")

    measurement = causal_design.get("measurement", {})
    id_info = causal_design.get("identifiability") or {}
    id_status = {
        "identifiable_treatments": id_info.get("identifiable_treatments", {}),
        "non_identifiable_treatments": id_info.get("non_identifiable_treatments", {}),
    }
    graph_info = capture.get("graph_info")
    if graph_info is not None:
        id_status["graph_info"] = graph_info

    return Stage1bResult(
        measurement_structure=measurement,
        identifiability_status=id_status,
        causal_design=causal_design,
    )
