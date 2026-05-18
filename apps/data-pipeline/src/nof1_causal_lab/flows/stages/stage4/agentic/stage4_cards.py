"""Stage 4 prompt-local context card builders.

Compact per-indicator, per-construct, and per-parameter context cards used
in Stage 4 LLM prompts.  Extracted from ``stage4_orchestrator`` so that
the orchestrator stays focused on the structural plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.models.ssm_spec_translation import get_construct_dt_days
from nof1_causal_lab.utils.causal_spec import (
    build_reference_indicator_lookup,
    get_constructs,
    get_estimation_edges,
    get_estimation_state_order,
    get_manifest_indicators,
    get_outcome_name,
)

from .stage4_parameter_surfaces import build_stage4_parameter_surface_index

if TYPE_CHECKING:
    from .stage4_skeleton import Stage4Skeleton


def build_model_topology(causal_spec: dict) -> dict[str, Any]:
    """Build compact fixed model context for the Stage 4 prompt."""
    model_dt_days = get_construct_dt_days(causal_spec)
    return {
        "model_clock": causal_spec.get("measurement", {}).get("model_clock"),
        "model_interval_days": model_dt_days,
        "outcome": get_outcome_name(causal_spec),
        "latent_edges": [
            {
                "cause": edge["cause"],
                "effect": edge["effect"],
                "lagged": bool(edge.get("lagged", True)),
                "description": edge.get("description"),
            }
            for edge in get_estimation_edges(causal_spec)
        ],
    }


def build_distribution_cards(
    causal_spec: dict,
    indicator_audits: dict[str, dict[str, Any]] | None,
    skeleton: Stage4Skeleton,
) -> list[dict[str, Any]]:
    """Build compact cards for indicators whose likelihoods need judgment."""
    indicator_lookup = {
        indicator["name"]: indicator for indicator in get_manifest_indicators(causal_spec)
    }

    cards: list[dict[str, Any]] = []
    for item in skeleton.ambiguous_indicators:
        variable = item["variable"]
        indicator = indicator_lookup.get(variable, {})
        audit = (indicator_audits or {}).get(variable) or {}
        validation = audit.get("validation") or {}

        option_rows: list[dict[str, Any]] = []
        if "fixed_distribution" in item:
            option_rows.append(
                {
                    "distribution": item["fixed_distribution"],
                    "links": item.get("valid_links", []),
                    "distribution_fixed": True,
                }
            )
        else:
            for distribution in item.get("valid_distributions", []):
                option_rows.append(
                    {
                        "distribution": distribution,
                        "links": item.get("link_options", {}).get(distribution, []),
                        "distribution_fixed": False,
                    }
                )

        cards.append(
            {
                "variable": variable,
                "construct": indicator.get("construct_name"),
                "measurement_dtype": indicator.get("measurement_dtype"),
                "aggregation": indicator.get("aggregation"),
                "observation_window": indicator.get("observation_window"),
                "effective_window": indicator.get("observation_window")
                or causal_spec.get("measurement", {}).get("model_clock"),
                "how_to_measure": indicator.get("how_to_measure"),
                "options": option_rows,
                "profile": _compact_profile(audit.get("profile") or {}),
                "validation_issues": [
                    f"{issue['severity']} {issue['issue_type']}"
                    for issue in validation.get("issues") or []
                ],
            }
        )
    return cards


def build_construct_scale_cards(
    causal_spec: dict,
    indicator_audits: dict[str, dict[str, Any]] | None,
    skeleton: Stage4Skeleton | None = None,
) -> list[dict[str, Any]]:
    """Build one construct-local scale card per construct."""
    from .stage4_skeleton import indicators_per_construct

    model_clock = causal_spec.get("measurement", {}).get("model_clock")
    retained_state_order = get_estimation_state_order(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_constructs(causal_spec)
    }
    constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]
    indicators = get_manifest_indicators(causal_spec)
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    indicators_per_construct = indicators_per_construct(indicators)
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
    ambiguous_indicator_names = {
        item["variable"] for item in (skeleton.ambiguous_indicators if skeleton else [])
    }

    cards: list[dict[str, Any]] = []
    for construct in constructs:
        construct_name = construct["name"]
        cards.append(
            {
                "construct": construct_name,
                "description": construct.get("description"),
                "role": construct.get("role"),
                "temporal_status": construct.get("temporal_status"),
                "is_outcome": bool(construct.get("is_outcome", False)),
                "reference_indicator": reference_indicator_lookup.get(construct_name),
                "indicators": [
                    _build_indicator_anchor(
                        indicator_name,
                        indicator_lookup,
                        indicator_audits,
                        model_clock=model_clock,
                        is_reference=indicator_name
                        == reference_indicator_lookup.get(construct_name),
                        has_distribution_decision_card=indicator_name in ambiguous_indicator_names,
                    )
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ],
            }
        )
    return cards


def build_prior_cards(causal_spec: dict, skeleton: Stage4Skeleton) -> list[dict[str, Any]]:
    """Build compact prompt-local prior cards for every deterministic parameter."""
    surface_index = build_stage4_parameter_surface_index(causal_spec, skeleton)
    return [surface.to_prior_card() for surface in surface_index.surfaces]


def _build_indicator_anchor(
    indicator_name: str | None,
    indicator_lookup: dict[str, dict[str, Any]],
    indicator_audits: dict[str, dict[str, Any]] | None,
    *,
    model_clock: str | None,
    is_reference: bool,
    has_distribution_decision_card: bool,
) -> dict[str, Any] | None:
    if not indicator_name:
        return None

    indicator = indicator_lookup.get(indicator_name, {})
    profile = ((indicator_audits or {}).get(indicator_name) or {}).get("profile") or {}
    return {
        "indicator": indicator_name,
        "construct": indicator.get("construct_name"),
        "measurement_dtype": indicator.get("measurement_dtype"),
        "construct_polarity": indicator.get("construct_polarity"),
        "how_to_measure": indicator.get("how_to_measure"),
        "aggregation": indicator.get("aggregation"),
        "observation_window": indicator.get("observation_window"),
        "effective_window": indicator.get("observation_window") or model_clock,
        "is_reference": is_reference,
        "has_distribution_decision_card": has_distribution_decision_card,
        "profile": _compact_profile(profile),
    }


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    if not profile:
        return None

    compact: dict[str, Any] = {}
    for key in (
        "n_obs",
        "mean",
        "std",
        "q25",
        "q50",
        "q75",
        "min",
        "max",
        "time_coverage_ratio",
        "max_gap_ratio",
        "duplicate_pct",
        "n_unparseable_timestamps",
        "zero_fraction",
        "variance_to_mean_ratio",
    ):
        value = profile.get(key)
        if value is not None:
            compact[key] = value

    support_flags = [
        flag_name
        for flag_name in ("is_nonnegative", "is_unit_interval", "looks_integer_valued")
        if profile.get(flag_name) is not None
    ]
    for flag_name in support_flags:
        compact[flag_name] = profile[flag_name]
    return compact or None
