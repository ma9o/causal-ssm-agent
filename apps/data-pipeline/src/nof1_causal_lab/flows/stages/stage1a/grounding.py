"""Stage 1a grounding."""


def stage1a_grounding(data: dict) -> tuple[dict | None, str]:
    """Validate latent structure."""
    from nof1_causal_lab.artifacts.latent_structure import validate_latent_structure

    _result, errors = validate_latent_structure(data)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return {"latent_structure": data}, "VALID"
