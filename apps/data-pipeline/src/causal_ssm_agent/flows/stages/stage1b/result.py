"""Stage 1b result shaping."""

from __future__ import annotations

from typing import Any

from ... import get_prefect_logger

logger = get_prefect_logger(__name__)


def finalize_stage1b_result(
    result: dict[str, Any],
    *,
    latent_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize Stage 1b derived fields from a causal spec payload."""
    from causal_ssm_agent.utils.causal_spec import get_estimable_treatments, get_outcome_name

    finalized = dict(result)
    causal_spec = finalized.get("causal_spec", {}) or {}
    treatments = list(get_estimable_treatments(causal_spec))
    outcome_name = get_outcome_name(causal_spec) or get_outcome_name(latent_model or {}) or ""
    identifiability = causal_spec.get("identifiability", {}) or {}
    non_identifiable = identifiability.get("non_identifiable_treatments", {})

    if non_identifiable:
        logger.warning("NON-IDENTIFIABLE TREATMENT EFFECTS (excluded from analysis):")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                logger.warning(
                    "  - %s → %s (blocked by: %s)",
                    treatment,
                    outcome_name,
                    ", ".join(blockers),
                )
            elif notes:
                logger.warning("  - %s → %s (%s)", treatment, outcome_name, notes)
            else:
                logger.warning("  - %s → %s", treatment, outcome_name)
        treatments = [t for t in treatments if t not in non_identifiable]
        logger.info(
            "Retaining %d estimable intervention targets after identifiability filtering",
            len(treatments),
        )

    if not treatments:
        logger.warning(
            "No retained estimation-stage intervention targets remain for %s",
            outcome_name or "the outcome",
        )

    if treatments and not non_identifiable:
        outcome = "success"
        fail_reason = None
    elif not treatments:
        outcome = "fail"
        fail_reason = "no_estimable_treatments"
    else:
        outcome = "warn"
        fail_reason = None

    finalized["_identified_treatments"] = treatments
    finalized["outcome"] = outcome
    if fail_reason is not None:
        finalized["fail_reason"] = fail_reason
    else:
        finalized.pop("fail_reason", None)
    return finalized
