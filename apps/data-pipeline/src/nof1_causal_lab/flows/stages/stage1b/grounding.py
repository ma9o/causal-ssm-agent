"""Stage 1b grounding."""

from __future__ import annotations


def stage1b_grounding(data: dict, latent_structure: dict) -> tuple[dict | None, str]:
    """Validate measurement structure against schema and compiler constraints."""
    from nof1_causal_lab.artifacts.latent_structure import LatentStructure
    from nof1_causal_lab.models.ssm.compile.artifact import (
        validate_measurement_structure_for_compilation,
    )

    latent = LatentStructure.model_validate(latent_structure)
    validated, errors = validate_measurement_structure_for_compilation(data, latent)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    assert validated is not None
    measurement = validated.model_dump(mode="json")
    return {"measurement_structure": measurement}, "VALID"
