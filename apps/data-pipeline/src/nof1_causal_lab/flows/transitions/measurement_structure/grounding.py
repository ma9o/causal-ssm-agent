"""measurement-structure grounding."""

from __future__ import annotations

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


def measurement_structure_grounding(
    data: UncheckedJsonObject,
    latent_structure: UncheckedJsonObject,
) -> tuple[UncheckedJsonObject | None, str]:
    """Validate authored measurement and known-input declarations."""
    from pydantic import ValidationError

    from nof1_causal_lab.artifacts.causal_design import KnownInput, ScientificOnlyConstruct
    from nof1_causal_lab.artifacts.latent_structure import LatentStructure
    from nof1_causal_lab.flows.transitions.measurement_structure.assemble import (
        build_causal_design,
    )
    from nof1_causal_lab.models.ssm.compile.artifact import (
        validate_measurement_structure_for_compilation,
    )
    from nof1_causal_lab.models.structural import build_structural_plan

    latent = LatentStructure.model_validate(latent_structure)
    validated, errors = validate_measurement_structure_for_compilation(data, latent)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    assert validated is not None
    measurement = validated.model_dump(mode="json")
    raw_known_inputs = data.get("known_inputs")
    if not isinstance(raw_known_inputs, list):
        return None, "VALIDATION ERRORS:\n- 'known_inputs' must be a list"
    raw_scientific_only = data.get("scientific_only_constructs")
    if not isinstance(raw_scientific_only, list):
        return None, "VALIDATION ERRORS:\n- 'scientific_only_constructs' must be a list"

    try:
        known_inputs = [
            KnownInput.model_validate(item).model_dump(mode="json", by_alias=True)
            for item in raw_known_inputs
        ]
        scientific_only_constructs = [
            ScientificOnlyConstruct.model_validate(item).model_dump(mode="json", by_alias=True)
            for item in raw_scientific_only
        ]
        candidate = build_causal_design(
            latent_structure,
            measurement,
            known_inputs=known_inputs,
            scientific_only_constructs=scientific_only_constructs,
        )
        build_structural_plan(candidate)
    except (TypeError, ValidationError, ValueError) as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"

    return {
        "measurement_structure": measurement,
        "known_inputs": known_inputs,
        "scientific_only_constructs": scientific_only_constructs,
    }, "VALID"
