"""Stage 4 prompts: Model Specification & Prior Elicitation.

The AGENTIC_SYSTEM / AGENTIC_USER prompts drive the single-conversation
agentic flow (``orchestrator/stage4.py``).

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py, and prior families
in causal_ssm_agent.distributions.PriorDistributionFamily
"""

from causal_ssm_agent.distributions import PriorDistributionFamily

PRIOR_DISTRIBUTION_CHOICE_LIST = "|".join(family.value for family in PriorDistributionFamily)


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
        lines.append(f"- `{lp['name']}`: `{lp['indicator']}` on `{lp['construct']}`")
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
        ("min", "min"),
        ("max", "max"),
    ):
        value = profile.get(key)
        if value is not None:
            fields.append(f"{label}={value:.3g}")
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
        "| Variable | Construct | Dtype | Aggregation | Options | Empirical Profile | Issues | How to Measure |",
        "|----------|-----------|-------|-------------|---------|-------------------|--------|----------------|",
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
            "| {variable} | {construct} | {dtype} | {aggregation} | {options} | {profile} | {issues} | {how} |".format(
                variable=card["variable"],
                construct=card.get("construct") or "unknown",
                dtype=card.get("measurement_dtype") or "unknown",
                aggregation=card.get("aggregation") or "unknown",
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
                lines.append(
                    "- indicator: `{indicator}`; reference: `{reference}`; details: "
                    "see distribution decision card".format(
                        indicator=indicator["indicator"],
                        reference="yes" if indicator.get("is_reference") else "no",
                    )
                )
            else:
                details = (
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
                "| Indicator | Dtype | Aggregation | Reference | Details |",
                "|-----------|-------|-------------|-----------|---------|",
            ]
        )
        for indicator in indicators:
            if indicator.get("has_distribution_decision_card"):
                details = "see distribution decision card"
            else:
                details = (
                    f"{_format_profile_summary(indicator.get('profile'))}; "
                    f"how={indicator.get('how_to_measure') or '-'}"
                )
            lines.append(
                "| {indicator} | {dtype} | {aggregation} | {reference} | {details} |".format(
                    indicator=indicator["indicator"],
                    dtype=indicator.get("measurement_dtype") or "unknown",
                    aggregation=indicator.get("aggregation") or "unknown",
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
            f"marginalized_confounder=`{structural_context.get('marginalized_confounder')}`"
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
                "| Parameter | Cause | Effect | Relation | Constraint |",
                "|-----------|-------|--------|----------|------------|",
            ]
        )
        for card in fixed_effect_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {cause} | {effect} | {relation} | {constraint} |".format(
                    parameter=card["parameter"],
                    cause=structural_context.get("cause", "-"),
                    effect=structural_context.get("effect", "-"),
                    relation="lagged"
                    if structural_context.get("lagged", True)
                    else "same_interval",
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
                "#### Correlations",
                "",
                "| Parameter | Construct 1 | Construct 2 | Marginalized Confounder | Constraint |",
                "|-----------|-------------|-------------|-------------------------|------------|",
            ]
        )
        for card in correlation_cards:
            structural_context = card.get("structural_context") or {}
            lines.append(
                "| {parameter} | {construct_1} | {construct_2} | {confounder} | {constraint} |".format(
                    parameter=card["parameter"],
                    construct_1=structural_context.get("construct_1", "-"),
                    construct_2=structural_context.get("construct_2", "-"),
                    confounder=structural_context.get("marginalized_confounder", "-"),
                    constraint=card["constraint"],
                )
            )
        lines.append("")

    other_roles = [
        role
        for role in groups
        if role not in {"ar_coefficient", "fixed_effect", "residual_sd", "loading", "correlation"}
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
# Agentic prompts (single-conversation flow)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM = """\
You are a Bayesian statistician completing a model specification and eliciting \
priors for causal inference via a continuous-time state-space model (CT-SSM).

Most of the specification has already been determined from the causal structure. \
Your job is to provide the decisions that require statistical judgment, and to \
propose priors for every parameter.

Use the fixed model context in the user message for reasoning, but do not \
rewrite it as if it were undecided. Do not add or remove constructs, edges, \
indicators, or parameters.

## Part 1 — Model Specification Decisions

### What Has Been Pre-Computed

The following are already determined and shown in the user message:
- **Final parameter inventory**, enumerated once in the parameter prior cards
- **Deterministic likelihoods** are omitted from the decision cards because they \
require no judgment
- **Construct scale cards** summarize indicator semantics and data scale once per construct
- **Parameter constraints** based on role (ar → unit_interval, fixed_effect → none, \
residual_sd → positive)

### What You Decide

1. **Distribution + link** for indicators with ambiguous dtypes (continuous, count, \
categorical). Choose based on the distribution decision cards and domain knowledge.

2. **Loading constraints**: For each loading parameter, decide `positive` (sign \
identification) or `none` (if negative loadings are theoretically plausible).

### Distribution Guidelines

- `gaussian`: Continuous unbounded data, approximately symmetric
- `student_t`: Continuous data with heavy tails or outliers
- `gamma`: Positive continuous data (reaction times, durations)
- `beta`: Proportions/rates in (0, 1)
- `poisson`: Count data (low counts, rare events, variance ≈ mean)
- `negative_binomial`: Overdispersed count data (variance > mean)
- `bernoulli`: Binary outcomes (logit or probit link)

### Link Function Rules

Most distributions have exactly one valid link (auto-determined). You only choose \
when multiple are valid:
- **bernoulli**: `logit` (default) or `probit`
- **gamma**: `log` (default) or `inverse`
- **beta**: `logit` (default) or `probit`

## Part 2 — Prior Elicitation

For EVERY parameter, propose a prior distribution.

### Prior Distribution Types
- **Normal(mu, sigma)**: Unconstrained effects (can be positive or negative)
- **HalfNormal(sigma)**: Positive-only parameters (variances, SDs)
- **Beta(alpha, beta)**: Parameters in [0, 1] (probabilities)
- **Uniform(lower, upper)**: When you want to bound the parameter
- **TruncatedNormal(mu, sigma, lower, upper)**: Bounded with a center
- **Gamma(concentration, rate)**: Positive-only parameters when a right-skewed prior is more plausible

### Parameter Guidelines by Type

| Type | Typical Distribution | Typical Range | Scale |
|------|---------------------|---------------|-------|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] | Discrete-time |
| rho (AR coefficient) | Beta(2, 2) | [0, 1] | Discrete-time persistence |
| sigma (residual SD) | HalfNormal(1) | [0, 5] | Data scale |
| lambda (loading) | HalfNormal(1) | [0, 3] | Data scale |
| cor (correlation) | TruncatedNormal(0, 0.3, -1, 1) | [-1, 1] | Innovation correlation |
| tau (random SD) | HalfNormal(0.5) | [0, 2] | Data scale |

Both beta and rho priors should be on the **discrete-time scale**. They are \
automatically converted to continuous-time rates internally.

### Literature Evidence
- If you have access to the `search_literature` tool, use it for key causal \
effects where empirical evidence matters. Not every parameter needs a search — \
AR coefficients, residual SDs, and loadings typically use standard defaults.
- When calling `search_literature`, always pass the `parameter_name` of the \
parameter you are searching for.
- Anchor priors on meta-analyses or large longitudinal studies when available.
- If evidence is heterogeneous, use wider priors.

### Continuous-Time Dynamics

Time is measured in fractional days. AR coefficients represent discrete-time \
persistence per observation interval, in (0, 1). The model interval is shown in \
the fixed model context. The system handles CT conversion.

## Tools

- `validate_model`: Stateful validator. It retains accepted model decisions and \
valid priors across retries. It rejects mixed decision+prior submissions, large \
prior dumps, and unchanged resubmissions. Start by validating the model spec, \
then add priors. After any failure, resubmit only the fields you changed.
- `search_literature` (if available): Search for empirical effect sizes. Use \
selectively for parameters where domain knowledge is uncertain.
- `elicit_prior_gmm` (if available): Run robust paraphrased elicitation for a \
single parameter. Returns an aggregated prior estimate.

## Workflow

1. Review the model topology, distribution decision cards, construct scale cards, and parameter prior cards
2. Optionally search literature for key causal effect parameters
3. Submit the full `distribution_choices` and `loading_constraints` first to lock the model spec
4. Do not include priors in that same `validate_model` call
5. Add priors incrementally via `validate_model` in small batches of at most 8 priors
6. If validation fails, read the feedback, fix the issues, and resubmit only the changed fields
7. Once you get "VALID", STOP immediately — do not output anything else
"""

AGENTIC_USER = """\
## Research Question

{question}

## Fixed Model Context

## Model Topology

{model_topology}

## Your Decisions

### 1. Distribution Decision Cards

Only indicators shown below need a distribution/link choice. Indicators not shown \
already have deterministic likelihoods.

{distribution_cards}

### 2. Loading Constraints
{loading_params}

### 3. Construct Scale Cards

Use these cards for construct semantics and data-scale anchoring. Single-indicator \
constructs are summarized inline. If a construct references a `distribution \
decision card`, the detailed measurement text and empirical profile are already \
shown in Section 1.

{construct_scale_cards}

### 4. Parameter Prior Cards

Provide exactly one prior for EVERY parameter below. The inventory is grouped by \
role to avoid repetition.

{prior_cards}

---

`validate_model` is stateful. You do not need to resend unchanged fields after a rejection.
It rejects mixed decision+prior updates, large prior batches, and unchanged accepted fields.

Typical sequence:

1. First validate the model spec only:
```json
{{
  "distribution_choices": [
    {{"variable": "...", "distribution": "...", "link": "...", "reasoning": "..."}}
  ],
  "loading_constraints": [
    {{"parameter": "...", "constraint": "positive|none", "reasoning": "..."}}
  ]
}}
```

2. Then add priors in focused updates. Priors must be sent without model decisions, and each call may include at most 8 priors:
```json
{{
  "priors": {{
    "parameter_name": {{
      "parameter": "parameter_name",
      "distribution": "__PRIOR_DISTRIBUTION_CHOICE_LIST__",
      "params": {{"mu": 0.3, "sigma": 0.15}},
      "sources": [],
      "reasoning": "Justification for the prior",
      "reference_interval_days": 7.0
    }}
  }}
}}
```

Never combine priors with model decisions in the same tool call. After a failure, only resend the priors or model decisions you changed.

Only include `reference_interval_days` when the literature evidence is expressed \
on a different observation interval than the model interval shown in Model \
Topology. Include a prior for EVERY parameter listed above.
"""

AGENTIC_USER = AGENTIC_USER.replace(
    "__PRIOR_DISTRIBUTION_CHOICE_LIST__",
    PRIOR_DISTRIBUTION_CHOICE_LIST,
)
