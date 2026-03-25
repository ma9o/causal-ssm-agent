"""Stage 4 deterministic model skeleton and prompt context helpers.

Pre-computes everything that follows directly from the causal spec without
LLM judgment: parameter enumeration, deterministic likelihood choices,
and compact prompt-local context cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from causal_ssm_agent.distributions import VALID_LIKELIHOODS_FOR_DTYPE
from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.orchestrator.schemas_model import VALID_LINKS_FOR_DISTRIBUTION
from causal_ssm_agent.utils.causal_spec import (
    get_estimation_edges,
    get_estimation_state_order,
    get_indicators,
    get_induced_dependencies,
    get_latent_constructs,
    get_outcome_name,
)


@dataclass(frozen=True)
class Stage4Skeleton:
    """Deterministic Stage 4 decision surface derived from the causal spec."""

    resolved_likelihoods: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_indicators: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    loading_params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_params(self) -> list[dict[str, Any]]:
        """Return the full final parameter inventory, including loadings."""
        return [*self.parameters, *self.loading_params]

    @property
    def final_parameter_names(self) -> list[str]:
        """Return the final parameter names in deterministic order."""
        return [param["name"] for param in self.all_params]


def derive_deterministic_spec(causal_spec: dict) -> Stage4Skeleton:
    """Pre-compute all deterministic parts of the stage-4 model skeleton."""
    retained_state_order = get_estimation_state_order(causal_spec)
    retained_edges = get_estimation_edges(causal_spec)
    indicators = get_indicators(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec)
    }
    retained_constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]

    indicators_per_construct = _indicators_per_construct(indicators)
    reference_indicator_lookup = {
        construct: indicator_names[0]
        for construct, indicator_names in indicators_per_construct.items()
        if indicator_names
    }

    resolved_likelihoods: list[dict[str, Any]] = []
    ambiguous_indicators: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    loading_params: list[dict[str, Any]] = []

    # --- Likelihoods ---
    for indicator in indicators:
        name = indicator["name"]
        dtype = indicator.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, ())

        if len(valid_dists) == 1:
            dist = next(iter(valid_dists))
            valid_links = VALID_LINKS_FOR_DISTRIBUTION[dist]
            if len(valid_links) == 1:
                link = next(iter(valid_links))
                resolved_likelihoods.append(
                    {
                        "variable": name,
                        "distribution": dist.value,
                        "link": link.value,
                        "reasoning": f"{dtype} dtype -> {dist.value} / {link.value}",
                    }
                )
            else:
                ambiguous_indicators.append(
                    {
                        "variable": name,
                        "dtype": dtype,
                        "fixed_distribution": dist.value,
                        "valid_links": sorted(link_fn.value for link_fn in valid_links),
                    }
                )
        else:
            link_options: dict[str, list[str]] = {}
            for distribution in sorted(valid_dists, key=lambda item: item.value):
                links = VALID_LINKS_FOR_DISTRIBUTION[distribution]
                link_options[distribution.value] = sorted(link_fn.value for link_fn in links)
            ambiguous_indicators.append(
                {
                    "variable": name,
                    "dtype": dtype,
                    "valid_distributions": sorted(dist.value for dist in valid_dists),
                    "link_options": link_options,
                }
            )

    # --- Autoregressive parameters ---
    for construct in retained_constructs:
        if (
            construct.get("temporal_status") == "time_varying"
            and construct.get("role") == "endogenous"
        ):
            construct_name = construct["name"]
            parameters.append(
                {
                    "name": f"rho_{construct_name}",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": f"AR(1) discrete-time persistence for {construct_name}",
                    "construct": construct_name,
                }
            )

    # --- Fixed effects ---
    for edge in retained_edges:
        cause = edge["cause"]
        effect = edge["effect"]
        parameters.append(
            {
                "name": f"beta_{cause}_{effect}",
                "role": "fixed_effect",
                "constraint": "none",
                "description": f"Effect of {cause} on {effect}",
                "cause": cause,
                "effect": effect,
                "lagged": edge.get("lagged", True),
            }
        )

    # --- Residual SDs ---
    for construct in retained_constructs:
        construct_name = construct["name"]
        parameters.append(
            {
                "name": f"sigma_{construct_name}",
                "role": "residual_sd",
                "constraint": "positive",
                "description": f"Residual/innovation SD for {construct_name}",
                "construct": construct_name,
            }
        )

    # --- Loadings ---
    reference_set: set[str] = set()
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if (
            not construct_name
            or construct_name not in indicators_per_construct
            or len(indicators_per_construct[construct_name]) <= 1
        ):
            continue

        if construct_name not in reference_set:
            reference_set.add(construct_name)
            continue

        reference_indicator = reference_indicator_lookup.get(construct_name)
        loading_params.append(
            {
                "name": f"lambda_{indicator['name']}_{construct_name}",
                "role": "loading",
                "constraint": "positive",
                "description": f"Factor loading for {indicator['name']} on {construct_name}",
                "indicator": indicator["name"],
                "construct": construct_name,
                "reference_indicator": reference_indicator,
            }
        )

    # --- Correlations from marginalized confounders ---
    for dependency in get_induced_dependencies(causal_spec):
        construct_1, construct_2 = dependency["between"]
        dependency_kind = dependency["kind"]
        parameter_name = (
            f"cor_{construct_1}_{construct_2}"
            if dependency_kind == "innovation_correlation"
            else f"cor0_{construct_1}_{construct_2}"
        )
        role = (
            "correlation"
            if dependency_kind == "innovation_correlation"
            else "initial_state_correlation"
        )
        parameters.append(
            {
                "name": parameter_name,
                "role": role,
                "constraint": "correlation",
                "description": (
                    f"{dependency_kind.replace('_', ' ')} between {construct_1} and {construct_2} "
                    f"(source confounders: {', '.join(dependency['source_confounders'])})"
                ),
                "construct_1": construct_1,
                "construct_2": construct_2,
                "dependency_kind": dependency_kind,
                "source_confounders": dependency["source_confounders"],
            }
        )

    return Stage4Skeleton(
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        parameters=parameters,
        loading_params=loading_params,
    )


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
    indicator_lookup = {indicator["name"]: indicator for indicator in get_indicators(causal_spec)}

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
    retained_state_order = get_estimation_state_order(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec)
    }
    constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]
    indicators = get_indicators(causal_spec)
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    indicators_per_construct = _indicators_per_construct(indicators)
    reference_indicator_lookup = {
        construct: indicator_names[0]
        for construct, indicator_names in indicators_per_construct.items()
        if indicator_names
    }
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
                        is_reference=indicator_name
                        == reference_indicator_lookup.get(construct_name),
                        has_distribution_decision_card=indicator_name in ambiguous_indicator_names,
                    )
                    for indicator_name in indicators_per_construct.get(construct_name, [])
                ],
            }
        )
    return cards


def build_prior_cards(skeleton: Stage4Skeleton) -> list[dict[str, Any]]:
    """Build compact prompt-local prior cards for every deterministic parameter."""
    cards: list[dict[str, Any]] = []
    for parameter in skeleton.all_params:
        role = parameter["role"]
        card: dict[str, Any] = {
            "parameter": parameter["name"],
            "role": role,
            "constraint": parameter["constraint"],
        }
        if role in {"ar_coefficient", "residual_sd"}:
            construct_name = parameter["construct"]
            card["structural_context"] = {"construct": construct_name}
        elif role == "fixed_effect":
            cause = parameter["cause"]
            effect = parameter["effect"]
            card["structural_context"] = {
                "cause": cause,
                "effect": effect,
                "lagged": parameter.get("lagged", True),
            }
        elif role == "loading":
            construct_name = parameter["construct"]
            indicator_name = parameter["indicator"]
            reference_indicator = parameter.get("reference_indicator")
            card["structural_context"] = {
                "construct": construct_name,
                "indicator": indicator_name,
                "reference_indicator": reference_indicator,
            }
        elif role in {"correlation", "initial_state_correlation"}:
            construct_1 = parameter["construct_1"]
            construct_2 = parameter["construct_2"]
            card["structural_context"] = {
                "construct_1": construct_1,
                "construct_2": construct_2,
                "dependency_kind": parameter["dependency_kind"],
                "source_confounders": parameter["source_confounders"],
            }
        else:
            card["structural_context"] = {}

        cards.append(card)

    return cards


def _indicators_per_construct(indicators: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if construct_name:
            grouped.setdefault(construct_name, []).append(indicator["name"])
    return grouped


def _build_indicator_anchor(
    indicator_name: str | None,
    indicator_lookup: dict[str, dict[str, Any]],
    indicator_audits: dict[str, dict[str, Any]] | None,
    *,
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
        "how_to_measure": indicator.get("how_to_measure"),
        "aggregation": indicator.get("aggregation"),
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
        "min",
        "max",
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
