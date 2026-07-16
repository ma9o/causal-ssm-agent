"""measurement-structure grounding."""

from __future__ import annotations


def measurement_structure_grounding(data: dict, latent_structure: dict) -> tuple[dict | None, str]:
    """Validate authored measurement and known-input declarations."""
    from pydantic import ValidationError

    from nof1_causal_lab.artifacts.causal_design import KnownInput
    from nof1_causal_lab.artifacts.latent_structure import LatentStructure
    from nof1_causal_lab.flows.transitions.measurement_structure.assemble import (
        build_causal_design,
    )
    from nof1_causal_lab.models.ssm.compile.artifact import (
        collect_estimation_projection_compile_errors,
        validate_measurement_structure_for_compilation,
    )

    latent = LatentStructure.model_validate(latent_structure)
    validated, errors = validate_measurement_structure_for_compilation(data, latent)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    assert validated is not None
    measurement = validated.model_dump(mode="json")
    raw_known_inputs = data.get("known_inputs")
    if not isinstance(raw_known_inputs, list):
        return None, "VALIDATION ERRORS:\n- 'known_inputs' must be a list"

    try:
        known_inputs = [
            KnownInput.model_validate(item).model_dump(mode="json", by_alias=True)
            for item in raw_known_inputs
        ]
        candidate = build_causal_design(
            latent_structure,
            measurement,
            known_inputs=known_inputs,
        )
    except (TypeError, ValidationError, ValueError) as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"

    compile_errors = collect_estimation_projection_compile_errors(candidate)
    if compile_errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in compile_errors)

    return {
        "measurement_structure": measurement,
        "known_inputs": known_inputs,
    }, "VALID"
