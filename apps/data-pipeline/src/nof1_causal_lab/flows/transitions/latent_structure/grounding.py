"""latent-structure grounding."""

from nof1_causal_lab.json_types import UncheckedJsonObject


def latent_structure_grounding(
    data: UncheckedJsonObject,
) -> tuple[UncheckedJsonObject | None, str]:
    """Validate latent structure."""
    from nof1_causal_lab.artifacts.latent_structure import validate_latent_structure

    _result, errors = validate_latent_structure(data)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return {"latent_structure": data}, "VALID"
