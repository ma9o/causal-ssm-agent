"""Stage 1b result shaping: split the LLM proposal into machine artifacts.

The proposal becomes three artifacts:

- ``causal_spec`` — the structural + measurement model (always)
- ``identification_report`` — the identification finding (always, even when
  negative: "nothing estimable under this spec" is a result, not a failure)
- ``estimands`` — ONLY when at least one treatment is identifiable; its
  absence is what structurally disables fitting and interventions

This module is also the derivation used when a human/LLM *writes* an edited
``causal_spec`` directly: identification is pure computation over the spec,
so the write executor fans out the same report/estimands artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1bArtifacts:
    """Payloads for the artifacts a stage-1b run (or causal_spec write) yields."""

    causal_spec_payload: dict[str, Any]
    identification_report: dict[str, Any]
    estimands: dict[str, Any] | None


def derive_identification_artifacts(
    causal_spec: dict[str, Any],
    *,
    latent_model: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compute (identification_report, estimands|None) from a causal spec."""
    from nof1_causal_lab.utils.causal_spec import get_estimable_treatments, get_outcome_name

    treatments = list(get_estimable_treatments(causal_spec))
    outcome_name = get_outcome_name(causal_spec) or get_outcome_name(latent_model or {}) or ""
    identifiability = causal_spec.get("identifiability", {}) or {}
    non_identifiable = identifiability.get("non_identifiable_treatments", {}) or {}

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

    report = {
        "outcome_name": outcome_name,
        "estimable_treatments": treatments,
        "non_identifiable_treatments": non_identifiable,
    }
    if not treatments:
        logger.warning(
            "No estimable intervention targets remain for %s — "
            "estimands artifact withheld (fit chain stays disabled)",
            outcome_name or "the outcome",
        )
        return report, None
    return report, {"outcome": outcome_name, "treatments": treatments}


def split_stage1b_result(
    result: dict[str, Any],
    *,
    latent_model: dict[str, Any] | None = None,
) -> Stage1bArtifacts:
    """Split a raw stage-1b LLM result into the three machine artifacts."""
    payload = dict(result)
    causal_spec = payload.get("causal_spec", {}) or {}
    report, estimands = derive_identification_artifacts(causal_spec, latent_model=latent_model)
    return Stage1bArtifacts(
        causal_spec_payload=payload,
        identification_report=report,
        estimands=estimands,
    )
