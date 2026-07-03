"""Stage 3 validation entrypoint."""

from __future__ import annotations

import polars as pl

from nof1_causal_lab.flows.stages.stage3.rules import (
    RULES,
    ValidationContext,
    build_indicator_audits,
    derive_validation_status,
    no_data_validation_result,
    run_rules,
)

# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


def validate_extraction(
    causal_spec: dict,
    dataframes: list[pl.DataFrame],
) -> dict:
    """Validate semantic properties of extracted data.

    Runs all ``RULES`` against the extracted data and reduces findings
    into a keyed indicator audit map plus dataset-level issues.

    Args:
        causal_spec: The full causal spec with measurement model
        dataframes: List of DataFrames with columns (indicator, value, anchor_time)

    Returns:
        Dict with:
            - is_valid: bool
            - indicators: per-indicator profile + validation
            - dataset_issues: cross-indicator validation findings
    """
    dataframes = [df for df in dataframes if df is not None and not df.is_empty()]
    if not dataframes:
        return no_data_validation_result()

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return no_data_validation_result()

    from nof1_causal_lab.utils.causal_spec import get_constructs, get_indicators

    indicators = get_indicators(causal_spec)
    indicator_names: set[str] = {ind["name"] for ind in indicators if ind.get("name")}
    indicator_lookup = {ind["name"]: ind for ind in indicators if ind.get("name")}

    constructs = get_constructs(causal_spec)
    construct_lookup = {c["name"]: c for c in constructs if c.get("name")}

    model_clock_str = causal_spec.get("measurement", {}).get("model_clock")
    model_clock_hours: float | None = None
    if model_clock_str:
        import contextlib

        from nof1_causal_lab.artifacts.duration import parse_duration_to_hours

        with contextlib.suppress(ValueError):
            model_clock_hours = parse_duration_to_hours(model_clock_str)

    validation_ctx = ValidationContext(
        combined=combined,
        indicators=indicators,
        indicator_names=indicator_names,
        indicator_lookup=indicator_lookup,
        construct_lookup=construct_lookup,
        model_clock_hours=model_clock_hours,
    )

    indicator_issues, indicator_health, dataset_issues = run_rules(
        RULES,
        validation_ctx,
    )

    indicator_audits = build_indicator_audits(
        indicator_names=indicator_names,
        indicator_lookup=indicator_lookup,
        model_data=combined,
        indicator_issues=indicator_issues,
        indicator_health=indicator_health,
    )

    all_issues = [*indicator_issues, *dataset_issues]
    status = derive_validation_status(all_issues)

    return {
        "is_valid": status["is_valid"],
        "indicators": indicator_audits,
        "dataset_issues": dataset_issues,
    }
