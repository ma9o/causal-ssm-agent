"""Stage 4 prompts: Model Specification Proposal.

The LLM receives a pre-computed skeleton (parameters, deterministic likelihoods)
and only provides genuine decisions: distribution choices for ambiguous indicators,
loading constraints, search contexts, and reasoning.

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py
"""

SYSTEM = """\
You are a Bayesian statistician completing a model specification for causal inference.

Most of this specification has already been determined from the causal structure. \
Your job is to provide ONLY the decisions that require statistical judgment.

## What Has Been Pre-Computed

The following are already determined and shown in the user message:
- **All parameters** (enumerated from the DAG: one AR per time-varying endogenous construct, \
one fixed effect per edge, one residual SD per construct, loadings for multi-indicator constructs)
- **Deterministic likelihoods** (e.g., ordinal → ordered_logistic / cumulative_logit)
- **Parameter constraints** based on role (ar → correlation, fixed_effect → none, residual_sd → positive)

## What You Decide

1. **Distribution + link** for indicators with ambiguous dtypes (continuous, count, \
categorical). Choose based on the data summary and domain knowledge.

2. **Loading constraints**: For each loading parameter, decide `positive` (sign identification) \
or `none` (if negative loadings are theoretically plausible).

3. **Search contexts**: For EVERY parameter, write a search query that would find \
relevant effect sizes in the academic literature (meta-analyses, systematic reviews, \
large longitudinal studies). This is used for prior elicitation.

## Distribution Guidelines

- `gaussian`: Continuous unbounded data, approximately symmetric
- `student_t`: Continuous data with heavy tails or outliers
- `gamma`: Positive continuous data (reaction times, durations)
- `beta`: Proportions/rates in (0, 1)
- `poisson`: Count data (low counts, rare events, variance ≈ mean)
- `negative_binomial`: Overdispersed count data (variance > mean)
- `bernoulli`: Binary outcomes (logit or probit link)

## Link Function Rules

Most distributions have exactly one valid link (auto-determined). You only choose \
when multiple are valid:
- **bernoulli**: `logit` (default) or `probit` (threshold/latent variable interpretation)
- **gamma**: `log` (default, ensures positive mean) or `inverse` (canonical, \
direct reciprocal relationship)
- **beta**: `logit` (default) or `probit`

## Continuous-Time Dynamics

The underlying model is a continuous-time state-space model (CT-SSM). Time is \
measured in fractional days. AR coefficients represent discrete-time persistence \
per observation interval, in (−1, 1). Positive = smooth persistence; negative = \
oscillatory dynamics (common in homeostatic/feedback systems). The system handles \
the CT conversion automatically.

## Validation Tool

You have access to `validate_model_spec` tool. Use it to validate your JSON. \
Keep validating until you get "VALID".

IMPORTANT: Once you get "VALID", STOP. Do not output anything else — the validated \
result is already saved by the tool.
"""

USER = """\
## Research Question

{question}

## Pre-Computed Model Skeleton

### Resolved Likelihoods (fully deterministic — do not change)

{resolved_likelihoods}

### All Parameters (enumerated from DAG — do not add or remove)

{parameters}

## Your Decisions

### 1. Distribution Choices

For each indicator below, choose the appropriate distribution and link function.

{ambiguous_indicators}

### 2. Loading Constraints
{loading_params}

### 3. Search Contexts

For EVERY parameter listed above, provide a search query for finding relevant \
effect sizes in the academic literature.

## Data Summary

{data_summary}

---

Output your decisions as JSON:
```json
{{
  "distribution_choices": [
    {{"variable": "...", "distribution": "...", "link": "...", "reasoning": "..."}}
  ],
  "loading_constraints": [
    {{"parameter": "...", "constraint": "positive|none", "reasoning": "..."}}
  ],
  "search_contexts": {{
    "parameter_name": "search query for literature"
  }},
  "reasoning": "Overall justification for model design choices"
}}
```
"""


def format_resolved_likelihoods(resolved: list[dict]) -> str:
    """Format pre-computed likelihoods for the prompt."""
    if not resolved:
        return "(none — all indicators require your decision)"
    lines = [
        "| Variable | Distribution | Link | Reason |",
        "|----------|-------------|------|--------|",
    ]
    for rl in resolved:
        lines.append(
            f"| {rl['variable']} | {rl['distribution']} | {rl['link']} | {rl['reasoning']} |"
        )
    return "\n".join(lines)


def format_ambiguous_indicators(ambiguous: list[dict]) -> str:
    """Format indicators needing LLM distribution choices."""
    if not ambiguous:
        return "(none — all distributions were determined by dtype)"
    lines = []
    for ai in ambiguous:
        var = ai["variable"]
        dtype = ai["dtype"]
        if "fixed_distribution" in ai:
            dist = ai["fixed_distribution"]
            links = ", ".join(ai["valid_links"])
            lines.append(
                f"- **{var}** (dtype={dtype}): distribution is `{dist}` — choose link: {links}"
            )
        else:
            dists = ", ".join(ai["valid_distributions"])
            lines.append(f"- **{var}** (dtype={dtype}): choose distribution from: {dists}")
            link_opts = ai.get("link_options", {})
            for d, links in link_opts.items():
                if len(links) == 1:
                    lines.append(f"  - if `{d}` → link is `{links[0]}` (auto)")
                else:
                    lines.append(f"  - if `{d}` → choose link: {', '.join(links)}")
    return "\n".join(lines)


def format_parameters(parameters: list[dict]) -> str:
    """Format pre-computed parameters for the prompt."""
    if not parameters:
        return "(none)"
    lines = [
        "| Name | Role | Constraint | Description |",
        "|------|------|-----------|-------------|",
    ]
    for p in parameters:
        constraint = p["constraint"]
        if p["role"] == "loading":
            constraint += " (you decide)"
        lines.append(f"| {p['name']} | {p['role']} | {constraint} | {p['description']} |")
    return "\n".join(lines)


def format_loading_params(loading_params: list[dict]) -> str:
    """Format loading parameters needing constraint decisions."""
    if not loading_params:
        return "\n(no multi-indicator constructs — skip this section)\n"
    lines = [
        "",
        "For each loading below, decide `positive` (reference/sign identification) "
        "or `none` (if negative loadings are plausible).",
        "",
        "| Parameter | Indicator | Construct |",
        "|-----------|-----------|-----------|",
    ]
    for lp in loading_params:
        lines.append(f"| {lp['name']} | {lp['indicator']} | {lp['construct']} |")
    lines.append("")
    return "\n".join(lines)


# Keep legacy formatters for backward compatibility with other callers
def format_constructs(causal_spec: dict) -> str:
    """Format constructs for the prompt."""
    from causal_ssm_agent.utils.causal_spec import get_constructs

    lines = []
    for construct in get_constructs(causal_spec):
        name = construct.get("name", "?")
        role = construct.get("role", "?")
        temporal = construct.get("temporal_status", "?")
        gran = construct.get("temporal_scale", "N/A")
        outcome = " [OUTCOME]" if construct.get("is_outcome") else ""
        desc = construct.get("description", "")
        lines.append(f"- **{name}**: {role}, {temporal}, granularity={gran}{outcome}")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_edges(causal_spec: dict) -> str:
    """Format causal edges for the prompt."""
    from causal_ssm_agent.utils.causal_spec import get_edges

    lines = []
    for edge in get_edges(causal_spec):
        cause = edge.get("cause", "?")
        effect = edge.get("effect", "?")
        lagged = "lagged" if edge.get("lagged", True) else "contemporaneous"
        desc = edge.get("description", "")
        lines.append(f"- {cause} → {effect} ({lagged})")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_indicators(causal_spec: dict) -> str:
    """Format indicators for the prompt."""
    from causal_ssm_agent.utils.causal_spec import get_indicators

    lines = []
    for indicator in get_indicators(causal_spec):
        name = indicator.get("name", "?")
        construct = indicator.get("construct_name", "?")
        dtype = indicator.get("measurement_dtype", "?")
        agg = indicator.get("aggregation", "?")
        lines.append(f"- **{name}**: measures {construct}")
        lines.append(f"  dtype={dtype}, aggregation={agg}")
    return "\n".join(lines)
