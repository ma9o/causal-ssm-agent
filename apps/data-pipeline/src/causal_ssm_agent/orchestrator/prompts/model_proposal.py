"""Stage 4 prompts: frontier-reduced model specification and prior elicitation.

The Stage 4 prompt builders narrow the LLM context to one active decision scope
at a time.

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py, and prior families
in causal_ssm_agent.distributions.PriorDistributionFamily
"""

from causal_ssm_agent.distributions import (
    PRIOR_PARAMETER_GUIDANCE_ROWS,
    render_dynamic_prior_scale_guidance,
    render_lagged_beta_authored_interval_guidance,
    render_observation_distribution_guidance_bullets,
    render_observation_link_guidance_bullets,
    render_prior_distribution_guidance_bullets,
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
    """Format loading parameters needing constraint decisions."""
    if not loading_params:
        return "\n(no multi-indicator constructs — skip this section)\n"
    lines = [
        "",
        "Decide `positive` (reference/sign identification) or `none` (if negative "
        "loadings are plausible) for each loading below. Richer indicator/reference "
        "context is repeated in the loading prior cards.",
        "",
    ]
    for lp in loading_params:
        selected_constraint = lp.get("selected_constraint")
        selected_suffix = f" (selected: `{selected_constraint}`)" if selected_constraint else ""
        lines.append(
            f"- `{lp['name']}`: `{lp['indicator']}` on `{lp['construct']}`{selected_suffix}"
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

    lines: list[str] = []
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
            "loading",
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
            "- Respect the accepted loading/sign decision already locked for this block."
        )
    if section_key == "continuous_time_dynamics":
        return "## Continuous-Time Dynamics\n\n" + DYNAMIC_PRIOR_SCALE_GUIDANCE
    if section_key == "lagged_effect_interval_guidance":
        return "## Lagged Effect Interval Guidance\n\n" + LAGGED_BETA_AUTHORED_INTERVAL_GUIDANCE
    raise ValueError(f"Unknown Stage 4 guidance section {section_key!r}")


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
    model_topology: dict,
    frontier_status: str,
    block_id: str,
    block_kind: str,
    block_label: str,
    block_instructions: str,
    distribution_cards: list[dict],
    loading_params: list[dict],
    construct_scale_cards: list[dict],
    prior_cards: list[dict],
    submission_example: str,
    latest_feedback: str,
    include_prior_source_guidance: bool,
) -> str:
    """Build the scope-local Stage 4 user prompt for the active frontier."""
    sections = [
        "## Research Question\n\n" + question,
        "## Fixed Model Context\n\n## Model Topology\n\n" + format_model_topology(model_topology),
        "## Frontier Status\n\n" + frontier_status,
        (
            "## Active Block\n\n"
            f"- `id`: `{block_id}`\n"
            f"- `kind`: `{block_kind}`\n"
            f"- `label`: {block_label}\n\n"
            f"{block_instructions}"
        ),
    ]

    if distribution_cards:
        sections.append(
            _format_markdown_section(
                "Distribution Decision Cards",
                format_distribution_cards(distribution_cards),
            )
        )
    if loading_params:
        sections.append(
            _format_markdown_section("Loading Constraints", format_loading_params(loading_params))
        )
    if construct_scale_cards:
        sections.append(
            _format_markdown_section(
                "Construct Scale Cards",
                format_construct_scale_cards(construct_scale_cards),
            )
        )
    if prior_cards:
        sections.append(
            _format_markdown_section("Parameter Prior Cards", format_prior_cards(prior_cards))
        )

    submission_parts = ["## Submission Contract\n\n" + submission_example]
    if include_prior_source_guidance:
        submission_parts.append(PRIOR_SOURCE_GUIDANCE.replace("{{", "{").replace("}}", "}"))
    sections.append(_join_sections(submission_parts))
    sections.append("## Latest Validator Feedback\n\n" + latest_feedback)
    return _join_sections(sections)
