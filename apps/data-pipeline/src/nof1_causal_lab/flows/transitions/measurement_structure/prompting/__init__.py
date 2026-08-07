"""measurement-structure prompt helpers."""

import json

from nof1_causal_lab.flows.transitions.measurement_structure.prompting import templates
from nof1_causal_lab.json_types import UncheckedJsonObject


def build_measurement_structure_user_prompt(
    question: str,
    latent_structure: UncheckedJsonObject,
    chunks: list[str],
    dataset_summary: str,
) -> str:
    return templates.USER.format(
        question=question,
        latent_structure_json=json.dumps(latent_structure, indent=2),
        dataset_summary=dataset_summary or "Not provided",
        chunks="\n".join(chunks),
    )
