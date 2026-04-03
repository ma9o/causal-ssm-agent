"""Stage 4 prompts: frontier-reduced model specification and prior elicitation.

The Stage 4 prompt builders narrow the LLM context to one active decision scope
at a time.

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py, and prior families
in causal_ssm_agent.distributions.PriorDistributionFamily
"""

from typing import Any

from causal_ssm_agent.distributions import (
    PRIOR_PARAMETER_GUIDANCE_ROWS,
    render_dynamic_prior_scale_guidance,
    render_lagged_beta_authored_interval_guidance,
    render_observation_distribution_guidance_bullets,
    render_observation_link_guidance_bullets,
    render_prior_distribution_guidance_bullets,
)
from causal_ssm_agent.orchestrator.stage4_feedback import (
    Stage4ScopeSnapshot,
    render_stage4_validation_feedback,
)

OBSERVATION_DISTRIBUTION_GUIDANCE_BULLETS = render_observation_distribution_guidance_bullets()
OBSERVATION_LINK_GUIDANCE_BULLETS = render_observation_link_guidance_bullets()
PRIOR_DISTRIBUTION_GUIDANCE_BULLETS = render_prior_distribution_guidance_bullets()
DYNAMIC_PRIOR_SCALE_GUIDANCE = render_dynamic_prior_scale_guidance()
LAGGED_BETA_AUTHORED_INTERVAL_GUIDANCE = render_lagged_beta_authored_interval_guidance()
PRIOR_SOURCE_GUIDANCE = """If you include non-empty `sources`, each entry must be an object with this shape:
```json
{{
  "title": "Source title",
  "snippet": "Relevant excerpt supporting the prior",
  "url": "https://example.org/paper",
  "effect_size": "β=0.21",
  "study_interval_days": 7.0
}}
```

Only `title` and `snippet` are required. Do not use raw strings or ad hoc keys such as `citation`, `finding`, `study_type`, or `notes`. If you are unsure, use `"sources": []`. `study_interval_days` belongs inside each source entry; `reference_interval_days` belongs on the prior itself."""


def format_loading_params(loading_params: list[dict]) -> str:
    """Format deterministic loading orientations derived upstream."""
    if not loading_params:
        return "\n(no multi-indicator constructs)\n"
    lines = [
        "",
        "These loading orientations are fixed from Stage 1b indicator polarity. "
        "They are part of the locked model form and are not a decision surface in Stage 4.",
        "",
    ]
    for lp in loading_params:
        lines.append(
            f"- `{lp['name']}`: `{lp['indicator']}` on `{lp['construct']}` (`{lp['constraint']}`)"
        )
    lines.append("")
    return "\n".join(lines)


def _format_profile_summary(profile: dict | None) -> str:
    """Format compact empirical profile text."""
    if not profile:
        return "no empirical profile"

    fields: list[str] = [f"n={profile.get('n_obs', 0)}"]
    for key, label in (
        ("mean", "mean"),
        ("std", "std"),
        ("q25", "q25"),
        ("q50", "q50"),
        ("q75", "q75"),
        ("min", "min"),
        ("max", "max"),
    ):
        value = profile.get(key)
        if value is not None:
            fields.append(f"{label}={value:.3g}")
    if profile.get("time_coverage_ratio") is not None:
        fields.append(f"coverage={profile['time_coverage_ratio']:.0%}")
    if profile.get("max_gap_ratio") is not None:
        fields.append(f"max_gap={profile['max_gap_ratio']:.3g}x")
    if profile.get("duplicate_pct") is not None:
        fields.append(f"dups={profile['duplicate_pct']:.1%}")
    if profile.get("n_unparseable_timestamps") is not None:
        fields.append(f"bad_ts={int(profile['n_unparseable_timestamps'])}")
    if profile.get("zero_fraction") is not None:
        fields.append(f"zero_frac={profile['zero_fraction']:.2%}")
    if profile.get("variance_to_mean_ratio") is not None:
        fields.append(f"var/mean={profile['variance_to_mean_ratio']:.3g}")

    support_flags = []
    if profile.get("is_nonnegative"):
        support_flags.append("nonnegative")
    if profile.get("is_unit_interval"):
        support_flags.append("unit_interval")
    if profile.get("looks_integer_valued"):
        support_flags.append("integer_like")
    if support_flags:
        fields.append("support=" + ",".join(support_flags))
    return "; ".join(fields)


def _format_window_summary(window: str | None) -> str:
    """Render the effective support window for prompt tables."""
    return window or "-"


def _format_selected_likelihood(item: dict | None) -> str:
    """Render the currently selected likelihood when available."""
    if not item:
        return "-"
    distribution = item.get("selected_distribution")
    link = item.get("selected_link")
    if distribution and link:
        return f"`{distribution}` / `{link}`"
    return "-"


def format_model_topology(model_topology: dict) -> str:
    """Format compact fixed model topology context."""
    if not model_topology:
        return "(none)"

    lines = [
        f"- model_clock: `{model_topology.get('model_clock') or 'unknown'}`",
        f"- model_interval_days: `{model_topology.get('model_interval_days')}`",
        f"- outcome: `{model_topology.get('outcome') or 'unknown'}`",
        "",
        "### Latent Edges",
        "",
    ]
    edges = model_topology.get("latent_edges") or []
    if not edges:
        lines.append("(none)")
        return "\n".join(lines)

    lines.extend(
        [
            "| Cause | Effect | Lagged | Description |",
            "|-------|--------|--------|-------------|",
        ]
    )
    for edge in edges:
        lines.append(
            "| {cause} | {effect} | {lagged} | {description} |".format(
                cause=edge["cause"],
                effect=edge["effect"],
                lagged="yes" if edge.get("lagged", True) else "no",
                description=edge.get("description") or "-",
            )
        )
    return "\n".join(lines)


def format_distribution_cards(distribution_cards: list[dict]) -> str:
    """Format cards for ambiguous indicator likelihood decisions."""
    if not distribution_cards:
        return "(none — all indicator likelihoods were deterministic)"

    lines: list[str] = [
        "| Variable | Construct | Dtype | Aggregation | Window | Current Choice | Options | Empirical Profile | Issues | How to Measure |",
        "|----------|-----------|-------|-------------|--------|----------------|---------|-------------------|--------|----------------|",
    ]
    for card in distribution_cards:
        option_parts = []
        for option in card.get("options", []):
            links = option.get("links") or []
            if len(links) == 1:
                option_parts.append(f"`{option['distribution']}` → `{links[0]}` (auto)")
            else:
                option_parts.append(
                    f"`{option['distribution']}` → {', '.join(f'`{link}`' for link in links)}"
                )
        options_str = "; ".join(option_parts) if option_parts else "-"

        issues = card.get("validation_issues") or []
        issues_str = "; ".join(issues) if issues else "none"

        lines.append(
            "| {variable} | {construct} | {dtype} | {aggregation} | {window} | {current_choice} | {options} | {profile} | {issues} | {how} |".format(
                variable=card["variable"],
                construct=card.get("construct") or "unknown",
                dtype=card.get("measurement_dtype") or "unknown",
                aggregation=card.get("aggregation") or "unknown",
                window=_format_window_summary(card.get("effective_window")).replace("|", "/"),
                current_choice=_format_selected_likelihood(card).replace("|", "/"),
                options=options_str.replace("|", "/"),
                profile=_format_profile_summary(card.get("profile")).replace("|", "/"),
                issues=issues_str.replace("|", "/"),
                how=(card.get("how_to_measure") or "-").replace("|", "/"),
            )
        )

    return "\n".join(lines)


def format_construct_scale_cards(construct_scale_cards: list[dict]) -> str:
    """Format one scale card per construct."""
    if not construct_scale_cards:
        return "(none)"

    lines: list[str] = [
        "- These cards summarize observed indicators attached to each construct.",
        "- Use them to understand semantics and rough observed scale, but do not copy raw indicator means or `log(mean(indicator))` into latent `t0_mean_*` priors unless the construct is explicitly identified on that observed scale.",
        "",
    ]
    for card in construct_scale_cards:
        lines.extend(
            [
                f"### `{card['construct']}`",
                (
                    f"- role: `{card.get('role') or 'unknown'}`; "
                    f"temporal_status: `{card.get('temporal_status') or 'unknown'}`; "
                    f"outcome: `{'yes' if card.get('is_outcome') else 'no'}`"
                ),
                f"- description: {card.get('description') or '-'}",
                f"- reference_indicator: `{card.get('reference_indicator') or 'none'}`",
                "",
            ]
        )

        indicators = card.get("indicators") or []
        if not indicators:
            lines.append("(no indicators)")
            lines.append("")
            continue

        if len(indicators) == 1:
            indicator = indicators[0]
            if indicator.get("has_distribution_decision_card"):
                selected_likelihood = _format_selected_likelihood(indicator)
                details = (
                    "see distribution decision card"
                    if selected_likelihood == "-"
                    else (
                        f"likelihood={selected_likelihood}; "
                        f"{_format_profile_summary(indicator.get('profile'))}; "
                        f"how={indicator.get('how_to_measure') or '-'}"
                    )
                )
                lines.append(
                    "- indicator: `{indicator}`; dtype: `{dtype}`; aggregation: "
                    "`{aggregation}`; window: `{window}`; reference: `{reference}`; details: {details}".format(
                        indicator=indicator["indicator"],
                        dtype=indicator.get("measurement_dtype") or "unknown",
                        aggregation=indicator.get("aggregation") or "unknown",
                        window=_format_window_summary(indicator.get("effective_window")),
                        reference="yes" if indicator.get("is_reference") else "no",
                        details=details,
                    )
                )
            else:
                details = (
                    f"window={_format_window_summary(indicator.get('effective_window'))}; "
                    f"likelihood={_format_selected_likelihood(indicator)}; "
                    f"{_format_profile_summary(indicator.get('profile'))}; "
                    f"how={indicator.get('how_to_measure') or '-'}"
                )
                lines.append(
                    "- indicator: `{indicator}`; dtype: `{dtype}`; aggregation: "
                    "`{aggregation}`; reference: `{reference}`; details: {details}".format(
                        indicator=indicator["indicator"],
                        dtype=indicator.get("measurement_dtype") or "unknown",
                        aggregation=indicator.get("aggregation") or "unknown",
                        reference="yes" if indicator.get("is_reference") else "no",
                        details=details,
                    )
                )
            lines.append("")
            continue

        lines.extend(
            [
                "| Indicator | Dtype | Aggregation | Window | Likelihood | Reference | Details |",
                "|-----------|-------|-------------|--------|------------|-----------|---------|",
            ]
        )
        for indicator in indicators:
            if indicator.get("has_distribution_decision_card"):
                details = (
                    "see distribution decision card"
                    if not indicator.get("selected_distribution")
                    else _format_profile_summary(indicator.get("profile"))
                )
            else:
                details = (
                    f"{_format_profile_summary(indicator.get('profile'))}; "
                    f"how={indicator.get('how_to_measure') or '-'}"
                )
            lines.append(
                "| {indicator} | {dtype} | {aggregation} | {window} | {likelihood} | {reference} | {details} |".format(
                    indicator=indicator["indicator"],
                    dtype=indicator.get("measurement_dtype") or "unknown",
                    aggregation=indicator.get("aggregation") or "unknown",
                    window=_format_window_summary(indicator.get("effective_window")).replace(
                        "|", "/"
                    ),
                    likelihood=_format_selected_likelihood(indicator).replace("|", "/"),
                    reference="yes" if indicator.get("is_reference") else "no",
                    details=details.replace("|", "/"),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_structural_context(structural_context: dict) -> str:
    """Format compact structural context for a prior card."""
    if not structural_context:
        return "-"
    if "cause" in structural_context and "effect" in structural_context:
        relation = "lagged" if structural_context.get("lagged", True) else "same_interval"
        return (
            f"cause=`{structural_context['cause']}`; "
            f"effect=`{structural_context['effect']}`; "
            f"relation=`{relation}`"
        )
    if "construct_1" in structural_context and "construct_2" in structural_context:
        return (
            f"construct_1=`{structural_context['construct_1']}`; "
            f"construct_2=`{structural_context['construct_2']}`; "
            f"dependency_kind=`{structural_context.get('dependency_kind')}`; "
            "source_confounders=`{}`".format(
                ",".join(structural_context.get("source_confounders") or [])
            )
        )
    if "indicator" in structural_context:
        return (
            f"construct=`{structural_context.get('construct')}`; "
            f"indicator=`{structural_context.get('indicator')}`; "
            f"reference_indicator=`{structural_context.get('reference_indicator')}`"
        )
    if "construct" in structural_context:
        return f"construct=`{structural_context['construct']}`"
    return ", ".join(f"{key}=`{value}`" for key, value in structural_context.items())


def format_prior_cards(prior_cards: list[dict]) -> str:
    """Format compact prior cards grouped by role."""
    if not prior_cards:
        return "(none)"
    groups: dict[str, list[dict]] = {}
    for card in prior_cards:
        groups.setdefault(card["role"], []).append(card)

    lines: list[str] = []
    accepted_prior_cards = [card for card in prior_cards if card.get("accepted_prior") is not None]
    if accepted_prior_cards:
        lines.extend(
            [
                "#### Current Accepted Priors",
                "",
                "| Parameter | Accepted Prior |",
                "|-----------|----------------|",
            ]
        )
        for card in accepted_prior_cards:
            lines.append(
                "| {parameter} | {accepted_prior} |".format(
                    parameter=card["parameter"],
                    accepted_prior=_format_authored_prior_summary(card.get("accepted_prior")),
                )
            )
        lines.append("")

    ar_cards = groups.get("ar_coefficient") or []
    if ar_cards:
        lines.extend(
            [
                "#### AR Coefficients",
                "",
                "| Parameter | Construct | Constraint |",
                "|-----------|-----------|------------|",
            ]
        )
        for card in ar_cards:
            lines.append(
                "| {parameter} | {construct} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=(card.get("structural_context") or {}).get("construct", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    fixed_effect_cards = groups.get("fixed_effect") or []
    if fixed_effect_cards:
        lines.extend(
            [
                "#### Fixed Effects",
                "",
                "| Parameter | Cause | Effect | Relation | Model Interval (days) | Feedback Loop | Constraint |",
                "|-----------|-------|--------|----------|-----------------------|---------------|------------|",
            ]
        )
        for card in fixed_effect_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {cause} | {effect} | {relation} | {interval_days} | {feedback_loop} | {constraint} |".format(
                    parameter=card["parameter"],
                    cause=structural_context.get("cause", "-"),
                    effect=structural_context.get("effect", "-"),
                    relation="lagged"
                    if structural_context.get("lagged", True)
                    else "same_interval",
                    interval_days=structural_context.get("expected_lag_days", "-"),
                    feedback_loop="yes" if structural_context.get("feedback_loop") else "no",
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    residual_sd_cards = groups.get("residual_sd") or []
    if residual_sd_cards:
        lines.extend(
            [
                "#### Residual SDs",
                "",
                "| Parameter | Construct | Constraint |",
                "|-----------|-----------|------------|",
            ]
        )
        for card in residual_sd_cards:
            lines.append(
                "| {parameter} | {construct} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=(card.get("structural_context") or {}).get("construct", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    measurement_error_cards = groups.get("measurement_error_sd") or []
    if measurement_error_cards:
        lines.extend(
            [
                "#### Measurement-Error SDs",
                "",
                "| Parameter | Construct | Indicator | Constraint |",
                "|-----------|-----------|-----------|------------|",
            ]
        )
        for card in measurement_error_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {construct} | {indicator} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=structural_context.get("construct", "-"),
                    indicator=structural_context.get("indicator", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    initial_state_mean_cards = groups.get("initial_state_mean") or []
    if initial_state_mean_cards:
        lines.extend(
            [
                "#### Initial-State Means",
                "",
                "| Parameter | Construct | Constraint |",
                "|-----------|-----------|------------|",
            ]
        )
        for card in initial_state_mean_cards:
            lines.append(
                "| {parameter} | {construct} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=(card.get("structural_context") or {}).get("construct", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    initial_state_sd_cards = groups.get("initial_state_sd") or []
    if initial_state_sd_cards:
        lines.extend(
            [
                "#### Initial-State SDs",
                "",
                "| Parameter | Construct | Constraint |",
                "|-----------|-----------|------------|",
            ]
        )
        for card in initial_state_sd_cards:
            lines.append(
                "| {parameter} | {construct} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=(card.get("structural_context") or {}).get("construct", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    loading_cards = groups.get("loading") or []
    if loading_cards:
        lines.extend(
            [
                "#### Loadings",
                "",
                "| Parameter | Construct | Indicator | Reference Indicator | Constraint |",
                "|-----------|-----------|-----------|---------------------|------------|",
            ]
        )
        for card in loading_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {construct} | {indicator} | {reference_indicator} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct=structural_context.get("construct", "-"),
                    indicator=structural_context.get("indicator", "-"),
                    reference_indicator=structural_context.get("reference_indicator", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    observation_hyperparameter_cards = [
        *(groups.get("observation_hyperparameter") or []),
        *(groups.get("observation_hyperparameter_positive") or []),
    ]
    if observation_hyperparameter_cards:
        lines.extend(
            [
                "#### Observation Hyperparameters",
                "",
                "| Parameter | Families | Indicators | Constraint |",
                "|-----------|----------|------------|------------|",
            ]
        )
        for card in observation_hyperparameter_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {families} | {indicators} | {constraint} |".format(
                    parameter=card["parameter"],
                    families=", ".join(
                        structural_context.get("activation_distribution_families") or ["-"]
                    ),
                    indicators=", ".join(structural_context.get("indicator_names") or ["-"]),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    correlation_cards = groups.get("correlation") or []
    if correlation_cards:
        lines.extend(
            [
                "#### Innovation Correlations",
                "",
                "| Parameter | Construct 1 | Construct 2 | Source Confounders | Constraint |",
                "|-----------|-------------|-------------|--------------------|------------|",
            ]
        )
        for card in correlation_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {construct_1} | {construct_2} | {source_confounders} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct_1=structural_context.get("construct_1", "-"),
                    construct_2=structural_context.get("construct_2", "-"),
                    source_confounders=", ".join(
                        structural_context.get("source_confounders") or ["-"]
                    ),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    initial_state_correlation_cards = groups.get("initial_state_correlation") or []
    if initial_state_correlation_cards:
        lines.extend(
            [
                "#### Initial-State Correlations",
                "",
                "| Parameter | Construct 1 | Construct 2 | Source Confounders | Constraint |",
                "|-----------|-------------|-------------|--------------------|------------|",
            ]
        )
        for card in initial_state_correlation_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {construct_1} | {construct_2} | {source_confounders} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct_1=structural_context.get("construct_1", "-"),
                    construct_2=structural_context.get("construct_2", "-"),
                    source_confounders=", ".join(
                        structural_context.get("source_confounders") or ["-"]
                    ),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    other_roles = [
        role
        for role in groups
        if role
        not in {
            "ar_coefficient",
            "fixed_effect",
            "residual_sd",
            "measurement_error_sd",
            "initial_state_mean",
            "initial_state_sd",
            "loading",
            "observation_hyperparameter",
            "observation_hyperparameter_positive",
            "correlation",
            "initial_state_correlation",
        }
    ]
    for role in sorted(other_roles):
        lines.extend(
            [
                f"#### {role}",
                "",
                "| Parameter | Constraint | Structural Context |",
                "|-----------|------------|--------------------|",
            ]
        )
        for card in groups[role]:
            lines.append(
                "| {parameter} | {constraint} | {structural_context} |".format(
                    parameter=card["parameter"],
                    constraint=card["constraint"],
                    structural_context=_format_structural_context(
                        card.get("structural_context") or {}
                    ),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Frontier-reduced prompts
# ---------------------------------------------------------------------------


def _join_sections(sections: list[str]) -> str:
    """Join prompt sections while dropping empty content."""
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _format_markdown_section(title: str, body: str) -> str:
    """Render a markdown section only when the body is non-empty."""
    content = body.strip()
    if not content:
        return ""
    return f"### {title}\n\n{content}"


def _summarize_scope_names(names: tuple[str, ...]) -> str:
    """Render a compact preview of scope parameter names."""
    if not names:
        return "(none)"
    return ", ".join(f"`{name}`" for name in names)


def _format_authored_prior_summary(prior: dict[str, Any] | None) -> str:
    """Format one accepted prior into a compact single-line summary."""
    if not isinstance(prior, dict):
        return "-"
    distribution = prior.get("distribution")
    params = prior.get("params")
    if not isinstance(distribution, str):
        return "-"
    if not isinstance(params, dict) or not params:
        return distribution
    ordered_parts = [f"{key}={params[key]}" for key in sorted(params)]
    return f"{distribution}({', '.join(ordered_parts)})"


def _format_stage4_scope_snapshot(snapshot: Stage4ScopeSnapshot) -> str:
    """Render the typed prompt-visible scope snapshot."""
    visible_parameter_names = snapshot.visible_parameter_names
    if not visible_parameter_names and snapshot.loading_params:
        visible_parameter_names = tuple(
            item["name"] for item in snapshot.loading_params if isinstance(item.get("name"), str)
        )
    lines = ["## Scope Snapshot", ""]
    if snapshot.editable_parameter_names:
        lines.append(
            f"- editable parameters: {_summarize_scope_names(snapshot.editable_parameter_names)}"
        )
    if visible_parameter_names:
        lines.append(f"- visible parameters: {_summarize_scope_names(visible_parameter_names)}")
    if snapshot.coupled_parameter_names:
        lines.append(
            "- coupled parameters outside this local edit scope: "
            f"{_summarize_scope_names(snapshot.coupled_parameter_names)}"
        )
    if len(lines) == 2:
        lines.append("- this block has no prior-parameter edit surface.")
    return "\n".join(lines)


def _format_latest_validation_state(snapshot: Stage4ScopeSnapshot) -> str:
    """Render a typed validation summary alongside the full validator text."""
    packet = snapshot.latest_validation
    lines = [
        "## Latest Validation State",
        "",
        f"- status: `{packet.status}`",
        f"- summary: {packet.summary}",
    ]
    if packet.failing_parameters:
        lines.append(
            f"- failing parameters: {_summarize_scope_names(packet.failing_parameters)}"
        )
    if packet.coupled_parameters:
        lines.append(
            f"- coupled parameters: {_summarize_scope_names(packet.coupled_parameters)}"
        )
    if packet.global_failure_sites:
        lines.append(
            f"- global failure sites: {_summarize_scope_names(packet.global_failure_sites)}"
        )
    return "\n".join(lines)


def _format_coupled_prior_cards(prior_cards: list[dict[str, Any]]) -> str:
    """Format accepted priors that are visible for coupling context only."""
    if not prior_cards:
        return "(none)"
    lines = [
        "| Parameter | Structural Context | Accepted Prior |",
        "|-----------|--------------------|----------------|",
    ]
    for card in prior_cards:
        lines.append(
            "| {parameter} | {structural_context} | {accepted_prior} |".format(
                parameter=card["parameter"],
                structural_context=_format_structural_context(card.get("structural_context") or {}),
                accepted_prior=_format_authored_prior_summary(card.get("accepted_prior")),
            )
        )
    return "\n".join(lines)


def _render_scope_parameter_guidance(parameter_guidance_prefixes: tuple[str, ...]) -> str:
    """Render only the prior-parameter guidance rows relevant to the active scope."""
    if not parameter_guidance_prefixes:
        return ""

    lines = [
        "| Type | Typical Distribution | Typical Range | Scale |",
        "|---|---|---|---|",
    ]
    for row in PRIOR_PARAMETER_GUIDANCE_ROWS:
        if not row.parameter_type.startswith(parameter_guidance_prefixes):
            continue
        lines.append(
            f"| {row.parameter_type} | {row.typical_distribution} | {row.typical_range} | {row.scale} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _render_stage4_guidance_section(
    section_key: str,
    *,
    parameter_guidance_prefixes: tuple[str, ...] = (),
) -> str:
    """Render one named system-prompt guidance section."""
    if section_key == "observation_distribution_guidance":
        return (
            "## Observation Distribution Guidance\n\n" + OBSERVATION_DISTRIBUTION_GUIDANCE_BULLETS
        )
    if section_key == "link_function_rules":
        return (
            "## Link Function Rules\n\n"
            "Most distributions have exactly one valid link (auto-determined). "
            "You only choose when multiple are valid:\n" + OBSERVATION_LINK_GUIDANCE_BULLETS
        )
    if section_key == "prior_distribution_types":
        return "## Prior Distribution Types\n\n" + PRIOR_DISTRIBUTION_GUIDANCE_BULLETS
    if section_key == "parameter_guidance":
        parameter_guidance = _render_scope_parameter_guidance(parameter_guidance_prefixes)
        return (
            "## Parameter Guidance for This Scope\n\n" + parameter_guidance
            if parameter_guidance
            else ""
        )
    if section_key == "measurement_prior_guidance":
        return (
            "## Measurement Prior Guidance\n\n"
            "- Use the construct scale card to anchor plausible indicator-to-construct magnitude.\n"
            "- Respect the fixed loading orientation already locked for this block.\n"
            "- For `obs_sd_*`, larger scales shift more variance into indicator noise instead of latent structure.\n"
            "- For `obs_*` observation hyperparameters, calibrate tails, dispersion, concentration, or thresholds to the locked likelihood family only."
        )
    if section_key == "continuous_time_dynamics":
        return "## Continuous-Time Dynamics\n\n" + DYNAMIC_PRIOR_SCALE_GUIDANCE
    if section_key == "latent_initial_state_guidance":
        return (
            "## Initial-State Scale Discipline\n\n"
            "- `t0_mean_*` and `t0_sd_*` live on the latent state scale.\n"
            "- Do not set `t0_mean_*` to the raw reference-indicator mean or `log(mean(indicator))` just because the indicator uses an identity or log link.\n"
            "- Default to weakly informative latent-scale priors such as `Normal(0, 1)` and `HalfNormal(1)` unless the construct is explicitly identified on an observed scale."
        )
    if section_key == "dynamics_budget_discipline":
        return (
            "## Dynamics Budget Discipline\n\n"
            "- These priors determine the damping available for later incoming lagged effects.\n"
            "- Avoid near-unit-root persistence or overly wide uncertainty unless strong evidence "
            "supports it.\n"
            "- Leave enough conservative decay that plausible incoming effects can still fit "
            "inside the compiled drift budget.\n"
            "- Treat the reported headroom as advisory stability guidance rather than a formal "
            "acceptance rule.\n"
            "- If the validator reports a partial drift failure at this stage, tighten the active "
            "dynamics priors toward faster decay before moving on."
        )
    if section_key == "effect_row_budget_discipline":
        return (
            "## Effect Row Budget Discipline\n\n"
            "- The user prompt reports a compiled continuous-time drift budget for the active "
            "target row.\n"
            "- Treat the conservative row budget and remaining headroom as advisory stability "
            "guidance, not as a mechanical acceptance rule.\n"
            "- Use the guidance to keep means and uncertainty modest; do not aim to spend the full "
            "headroom.\n"
            "- In dense SCC rows or when the Parameter Prior Cards mark `Feedback Loop` as `yes`, "
            "start from tightly zero-centered priors with modest uncertainty, often around "
            "`Normal(0, 0.1-0.2)` unless strong longitudinal evidence supports more.\n"
            "- Prefer shrinkage toward zero when evidence is sparse, mixed, or indirect.\n"
            "- If several incoming effects are plausible, distribute modest effects across them "
            "instead of making one edge dominate without strong support.\n"
            "- If validator feedback reports a partial drift failure, repair this row by "
            "shrinking effect means and/or scales."
        )
    if section_key == "lagged_effect_interval_guidance":
        return "## Lagged Effect Interval Guidance\n\n" + LAGGED_BETA_AUTHORED_INTERVAL_GUIDANCE
    raise ValueError(f"Unknown Stage 4 guidance section {section_key!r}")


def _format_effect_prior_budget_discipline() -> str:
    """Render the dynamic budget discipline section for effect-prior blocks."""
    return (
        "## Effect-Block Stability Discipline\n\n"
        "- The frontier status above reports the compiled continuous-time drift budget for this "
        "target row.\n"
        "- Treat the remaining headroom as advisory stability telemetry for the full row in this "
        "block.\n"
        "- Keep the row conservative rather than trying to use all reported headroom.\n"
        "- In dense feedback rows, start from tightly zero-centered effects with modest "
        "uncertainty, often around `Normal(0, 0.1-0.2)` unless strong longitudinal evidence "
        "supports more.\n"
        "- If the Parameter Prior Cards mark `Feedback Loop` as `yes`, be more conservative than "
        "you would be for a comparable acyclic lagged effect.\n"
        "- Prefer smaller means and tighter scales when the literature is weak or mixed.\n"
        "- If multiple incoming effects are plausible, spread modest effects across them instead "
        "of forcing one dominant coefficient without strong evidence.\n"
        "- If the latest validator feedback reports a partial drift failure, revise this row by "
        "shrinking effect means and/or scales."
    )


def build_stage4_system_prompt(
    *,
    system_task: str,
    guidance_section_keys: tuple[str, ...],
    parameter_guidance_prefixes: tuple[str, ...] = (),
    enabled_tool_names: tuple[str, ...] = ("validate_model",),
) -> str:
    """Build the scope-local Stage 4 system prompt for the active frontier."""
    sections = [
        (
            "You are a Bayesian statistician completing one active Stage 4 prompt scope for "
            "causal inference via a continuous-time state-space model (CT-SSM).\n\n"
            "Most of the specification has already been determined from the causal structure. "
            "Work only on the active scope shown in the user message. Do not add or remove "
            "constructs, edges, indicators, or parameters."
        ),
        (
            "## What Is Already Fixed\n\n"
            "- deterministic likelihoods where dtype leaves no ambiguity\n"
            "- final parameter inventory implied by the causal structure\n"
            "- construct scale cards and empirical profiles prepared by the pipeline\n"
            "- accepted upstream decisions preserved server-side unless the validator reopens them"
        ),
        "## Active Task\n\n" + system_task,
    ]

    sections.extend(
        _render_stage4_guidance_section(
            section_key,
            parameter_guidance_prefixes=parameter_guidance_prefixes,
        )
        for section_key in guidance_section_keys
    )
    if "search_literature" in enabled_tool_names:
        sections.append(
            "## Literature Evidence\n\n"
            "- Use `search_literature` selectively when empirical effect-size evidence matters.\n"
            "- If multiple active-scope parameters still need literature support, batch those "
            "`search_literature` calls in the same turn.\n"
            "- Always pass an active-scope `parameter_name` when calling the tool.\n"
            "- Anchor priors on larger longitudinal evidence when available.\n"
            "- If evidence is heterogeneous or indirect, widen the prior.\n"
            "- Do not paraphrase the same search for the same parameter across extra turns.\n"
            "- If direct evidence remains sparse but a conservative prior is already justified, "
            "stop searching and submit `validate_model`."
        )
    if "elicit_prior_gmm" in enabled_tool_names:
        sections.append(
            "## Robust Prior Elicitation\n\n"
            "- If `elicit_prior_gmm` is available, use it only for an active-scope parameter.\n"
            "- Treat it as optional support for difficult prior judgments, not a substitute for reasoning."
        )

    available_tools = [
        "- `validate_model`: submit only the active scope using the block-local contract.",
    ]
    if "search_literature" in enabled_tool_names:
        available_tools.append(
            "- `search_literature`: fetch empirical effect-size evidence for an active-scope "
            "parameter; you may call it multiple times in one turn when several active-scope "
            "parameters remain unresolved."
        )
    if "elicit_prior_gmm" in enabled_tool_names:
        available_tools.append(
            "- `elicit_prior_gmm`: run robust paraphrased elicitation for one active-scope parameter."
        )
    sections.append(
        "## Tool Contract\n\n"
        "Use `validate_model` with exactly this outer shape:\n"
        "```json\n"
        '{\n  "block_id": "...",\n  "block_kind": "...",\n  "proposal": { ... }\n}\n'
        "```\n\n"
        "Available tools on this scope:\n"
        + "\n".join(available_tools)
        + "\n\nDo not submit decisions or priors for any other block. "
        "After a rejection, read the validator feedback and resubmit only the active block.\n\n"
        'Once you get "VALID", STOP immediately and output nothing else.'
    )
    return _join_sections(sections)


def build_stage4_user_prompt(
    *,
    question: str,
    snapshot: Stage4ScopeSnapshot,
) -> str:
    """Build the scope-local Stage 4 user prompt for the active frontier."""
    sections = [
        "## Research Question\n\n" + question,
        "## Fixed Model Context\n\n## Model Topology\n\n" + format_model_topology(snapshot.model_topology),
        "## Frontier Status\n\n" + snapshot.frontier_status,
        (
            "## Active Block\n\n"
            f"- `id`: `{snapshot.block_id}`\n"
            f"- `kind`: `{snapshot.block_kind}`\n"
            f"- `label`: {snapshot.block_label}\n\n"
            f"{snapshot.block_instructions}"
        ),
        _format_stage4_scope_snapshot(snapshot),
    ]
    if snapshot.block_kind == "effect_prior":
        sections.append(_format_effect_prior_budget_discipline())

    if snapshot.distribution_cards:
        sections.append(
            _format_markdown_section(
                "Distribution Decision Cards",
                format_distribution_cards(snapshot.distribution_cards),
            )
        )
    if snapshot.loading_params:
        sections.append(
            _format_markdown_section(
                "Loading Orientation",
                format_loading_params(snapshot.loading_params),
            )
        )
    if snapshot.construct_scale_cards:
        sections.append(
            _format_markdown_section(
                "Construct Scale Cards",
                format_construct_scale_cards(snapshot.construct_scale_cards),
            )
        )
    if snapshot.prior_cards:
        sections.append(
            _format_markdown_section(
                "Parameter Prior Cards",
                format_prior_cards(snapshot.prior_cards),
            )
        )
    if snapshot.coupled_prior_cards:
        sections.append(
            _format_markdown_section(
                "Accepted Coupled Priors Outside This Edit Scope",
                _format_coupled_prior_cards(snapshot.coupled_prior_cards),
            )
        )
    if any(
        card.get("role") in {"initial_state_mean", "initial_state_sd"}
        for card in snapshot.prior_cards
    ):
        sections.append(
            "### Initial-State Scale Note\n\n"
            "- `t0_mean_*` and `t0_sd_*` are latent-state priors.\n"
            "- Do not anchor them to the raw reference-indicator mean or `log(mean(indicator))` unless the construct is explicitly identified on that observed scale."
        )

    submission_parts = ["## Submission Contract\n\n" + snapshot.submission_example]
    if snapshot.include_prior_source_guidance:
        submission_parts.append(PRIOR_SOURCE_GUIDANCE.replace("{{", "{").replace("}}", "}"))
    sections.append(_join_sections(submission_parts))
    sections.append(_format_latest_validation_state(snapshot))
    sections.append(
        "## Latest Validator Feedback\n\n"
        + render_stage4_validation_feedback(snapshot.latest_validation)
    )
    return _join_sections(sections)
