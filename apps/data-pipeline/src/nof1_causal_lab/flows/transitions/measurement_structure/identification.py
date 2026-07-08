"""Identification-report derivation for measurement-structure structures."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def derive_identification_report(
    causal_design: dict[str, Any],
    *,
    latent_structure: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute the positive identification report from a causal design."""
    from nof1_causal_lab.utils.causal_design import get_outcome_name

    outcome_name = get_outcome_name(causal_design) or get_outcome_name(latent_structure or {}) or ""
    identifiability = causal_design.get("identifiability", {}) or {}
    identifiable = identifiability.get("identifiable_treatments", {}) or {}
    non_identifiable = identifiability.get("non_identifiable_treatments", {}) or {}
    treatments = list(identifiable.keys())

    if non_identifiable:
        logger.warning("NON-IDENTIFIABLE TREATMENT EFFECTS (excluded from analysis):")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                logger.warning(
                    "  - %s -> %s (blocked by: %s)",
                    treatment,
                    outcome_name,
                    ", ".join(blockers),
                )
            elif notes:
                logger.warning("  - %s -> %s (%s)", treatment, outcome_name, notes)
            else:
                logger.warning("  - %s -> %s", treatment, outcome_name)
        logger.info(
            "Retaining %d estimable intervention targets after identifiability filtering",
            len(treatments),
        )

    if not treatments:
        logger.warning(
            "No estimable intervention targets remain for %s; "
            "identification_report artifact withheld",
            outcome_name or "the outcome",
        )
        return None
    return {
        "outcome_name": outcome_name,
        "estimable_treatments": treatments,
        "non_identifiable_treatments": non_identifiable,
    }
