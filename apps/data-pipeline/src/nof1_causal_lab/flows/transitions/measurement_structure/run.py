"""measurement-structure: Measurement Structure proposal."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompting import templates

if TYPE_CHECKING:
    from nof1_causal_lab.utils.agent_session import ScopedSessionFactory

logger = logging.getLogger(__name__)


@dataclass
class MeasurementStructureResult:
    """Result of measurement-structure: validated measurement structure."""

    measurement_structure: dict


def _build_measurement_structure_user_prompt(
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


async def run_measurement_structure(
    question: str,
    latent_structure: dict,
    chunks: list[str],
    session_factory: ScopedSessionFactory,
    dataset_summary: str = "",
) -> MeasurementStructureResult:
    """Run the measurement-structure measurement-structure proposal flow."""
    from nof1_causal_lab.flows.context_tool_factory import make_context_tool
    from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
        measurement_structure_grounding,
    )

    tool, capture = make_context_tool(
        name="validate_measurement_structure",
        description="Validate measurement structure JSON and compiler constraints.",
        param_name="measurement_json",
        param_description="The JSON string containing the measurement structure.",
        compute_fn=lambda data: measurement_structure_grounding(data, latent_structure),
    )

    async with session_factory.open(
        system_prompt=templates.SYSTEM,
        tools=[tool],
        log_label="measurement-structure",
    ) as session:
        await session.turn(
            _build_measurement_structure_user_prompt(
                question, latent_structure, chunks, dataset_summary
            )
        )
        await session.turn(templates.REVIEW)

    measurement = capture.get("measurement_structure")
    if measurement is None:
        raise ValueError("No valid measurement structure produced")

    return MeasurementStructureResult(measurement_structure=measurement)
