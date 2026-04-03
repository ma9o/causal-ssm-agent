"""Stage 1a grounding."""


def stage1a_grounding(data: dict) -> tuple[dict | None, str]:
    """Validate latent model."""
    from causal_ssm_agent.orchestrator.schemas import validate_latent_model

    _result, errors = validate_latent_model(data)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return {"latent_model": data}, "VALID"
