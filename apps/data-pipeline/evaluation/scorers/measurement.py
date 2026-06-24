"""Measurement-model (Stage 1b) scoring with the identifiability bonus.

Lifted out of the Inspect Stage 1b eval so the scoring is a single source shared
by the eval and any registry measurement row. ``score_measurement_model`` is the
core logic; ``MeasurementModelScorer`` is the registry-shaped wrapper.

``result`` is duck-typed to the Stage 1b result (``.measurement_model``,
``.identifiability_status``); ``latent`` is the proposed :class:`LatentModel`.
"""

from __future__ import annotations

from typing import Any

from nof1_causal_lab.artifacts.measurement_model import MeasurementModel


def score_measurement_model(result: Any, latent: Any) -> dict[str, Any]:
    """Score a Stage 1b result.

    Scoring rules:
    - +2 per valid indicator (references a known construct)
    - +1 valid dtype, +1 valid aggregation, +1 specific how_to_measure (>50 chars)
    - +2 per extra indicator on a multiply-measured construct
    - +10 if all treatments are identifiable
    """
    breakdown: list[str] = []
    indicator_points: dict[str, Any] = {}
    total = 0.0

    try:
        measurement = MeasurementModel.model_validate(result.measurement_model)
    except Exception as e:  # noqa: BLE001
        return {"total": 0.0, "breakdown": f"Invalid measurement model: {e}", "error": True}

    construct_names = {c.name for c in latent.constructs}
    indicators_per_construct: dict[str, int] = {}

    for indicator in measurement.indicators:
        pts = 0
        details = []

        if indicator.construct_name in construct_names:
            pts += 2
            details.append(f"+2 valid construct '{indicator.construct_name}'")
            indicators_per_construct[indicator.construct_name] = (
                indicators_per_construct.get(indicator.construct_name, 0) + 1
            )
        else:
            details.append(f"+0 unknown construct '{indicator.construct_name}'")

        valid_dtypes = {"continuous", "binary", "count", "ordinal", "categorical"}
        if indicator.measurement_dtype in valid_dtypes:
            pts += 1
            details.append("+1 valid dtype")

        pts += 1  # Valid aggregation (schema-validated)
        details.append("+1 valid aggregation")

        if len(indicator.how_to_measure) > 50:
            pts += 1
            details.append("+1 specific how_to_measure")

        indicator_points[indicator.name] = {"points": pts, "details": details}
        total += pts

    for construct, count in indicators_per_construct.items():
        if count > 1:
            bonus = (count - 1) * 2
            total += bonus
            breakdown.append(f"+{bonus} multi-indicator for '{construct}' ({count})")

    non_id = len(result.identifiability_status.get("non_identifiable_treatments", {}))
    if non_id == 0:
        breakdown.append("+10 ALL identifiable!")
        total += 10
    else:
        breakdown.append(f"+0 {non_id} treatments not identifiable")

    breakdown.insert(0, f"INDICATORS ({len(measurement.indicators)}):")
    for name, info in indicator_points.items():
        breakdown.append(f"  {name}: {info['points']} pts")
    breakdown.append(f"\nTOTAL: {total} points")

    return {
        "total": total,
        "indicators": indicator_points,
        "breakdown": "\n".join(breakdown),
        "indicators_per_construct": indicators_per_construct,
    }
