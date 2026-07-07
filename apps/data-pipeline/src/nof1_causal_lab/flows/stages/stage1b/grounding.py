"""Stage 1b grounding."""

from __future__ import annotations

from typing import Any


def stage1b_grounding(data: dict, latent_structure: dict) -> tuple[dict | None, str]:
    """Validate measurement structure, check identifiability, build CausalDesign."""
    from nof1_causal_lab.artifacts.latent_structure import LatentStructure
    from nof1_causal_lab.models.ssm.compile.artifact import (
        collect_estimation_projection_compile_errors,
        validate_measurement_structure_for_compilation,
    )
    from nof1_causal_lab.utils.identifiability import check_identifiability

    from .assemble import build_causal_design

    latent = LatentStructure.model_validate(latent_structure)
    validated, errors = validate_measurement_structure_for_compilation(data, latent)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    assert validated is not None
    measurement = validated.model_dump(mode="json")

    id_result = check_identifiability(latent_structure, measurement)
    id_status = {
        "identifiable_treatments": id_result.get("identifiable_treatments", {}),
        "non_identifiable_treatments": id_result.get("non_identifiable_treatments", {}),
    }
    if "graph_info" in id_result:
        id_status["graph_info"] = id_result["graph_info"]

    known_inputs = data.get("known_inputs") if isinstance(data.get("known_inputs"), list) else None
    causal_design = build_causal_design(
        latent_structure, measurement, id_status, known_inputs=known_inputs
    )
    estimation_errors = collect_estimation_projection_compile_errors(causal_design)
    if estimation_errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in estimation_errors)
    stage_output: dict[str, Any] = {"causal_design": causal_design}
    if "graph_info" in id_result:
        stage_output["graph_info"] = id_result["graph_info"]

    if id_result.get("non_identifiable_treatments"):
        feedback = _format_identifiability_feedback(id_result, latent_structure)
        return stage_output, feedback

    return stage_output, "VALID"


def _format_identifiability_feedback(id_result: dict, latent_structure: dict) -> str:
    """Rich feedback when model is valid but not fully identifiable."""
    lines = [
        "Structure is VALID but causal effects are NOT fully identifiable.",
        "",
        "Non-identifiable effects:",
    ]
    non_id = id_result.get("non_identifiable_treatments", {})
    construct_names = {c["name"] for c in latent_structure.get("constructs", [])}

    all_confounders: set[str] = set()
    for treatment, info in sorted(non_id.items()):
        if not isinstance(info, dict):
            lines.append(f"  - {treatment}: {info}")
            continue
        confounders = info.get("confounders", [])
        notes = info.get("notes", "")
        if confounders:
            lines.append(f"  - {treatment}: blocked by {', '.join(confounders)}")
            all_confounders.update(c for c in confounders if c in construct_names)
        elif notes:
            lines.append(f"  - {treatment}: {notes}")

    if all_confounders:
        lines.extend(
            [
                "",
                "To fix: add proxy indicators for the blocking confounders and resubmit",
                "the COMPLETE measurement structure (all existing indicators + new proxies).",
                f"Confounders needing proxies: {', '.join(sorted(all_confounders))}",
                "",
                "A proxy is an observable variable in the dataset that correlates with",
                "the unobserved confounder. Add it as a new indicator with the confounder",
                "as its construct_name.",
                "",
                "If no suitable proxy exists in the data, proceed — those effects will",
                "remain non-identifiable and be flagged in downstream analysis.",
            ]
        )

    return "\n".join(lines)
