"""Stage 4 prompts: Model Specification & Prior Elicitation.

The AGENTIC_SYSTEM / AGENTIC_USER prompts drive the single-conversation
agentic flow (``orchestrator/stage4.py``).

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py
"""

import json


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


def format_full_causal_spec(causal_spec: dict) -> str:
    """Format the full causal spec as JSON prompt context without truncation."""
    if not causal_spec:
        return "(none)"
    return f"```json\n{json.dumps(causal_spec, indent=2)}\n```"


# ---------------------------------------------------------------------------
# Agentic prompts (single-conversation flow)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM = """\
You are a Bayesian statistician completing a model specification and eliciting \
priors for causal inference via a continuous-time state-space model (CT-SSM).

Most of the specification has already been determined from the causal structure. \
Your job is to provide the decisions that require statistical judgment, and to \
propose priors for every parameter.

The user message intentionally separates:
- **Full model context**: the complete causal model from stages 1a/1b, provided so you can reason with the whole model
- **Decision surface**: the limited set of items you are allowed to choose

Do not rewrite or reinterpret the full model as if it were undecided. Do not add \
or remove constructs, edges, indicators, or parameters.

## Part 1 — Model Specification Decisions

### What Has Been Pre-Computed

The following are already determined and shown in the user message:
- **All parameters** (enumerated from the DAG: one AR per time-varying endogenous \
construct, one fixed effect per edge, one residual SD per construct, loadings for \
multi-indicator constructs)
- **Deterministic likelihoods** (e.g., ordinal → ordered_logistic / cumulative_logit)
- **Parameter constraints** based on role (ar → unit_interval, fixed_effect → none, \
residual_sd → positive)

### What You Decide

1. **Distribution + link** for indicators with ambiguous dtypes (continuous, count, \
categorical). Choose based on the data summary and domain knowledge.

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

### Parameter Guidelines by Type

| Type | Typical Distribution | Typical Range | Scale |
|------|---------------------|---------------|-------|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] | Discrete-time |
| rho (AR coefficient) | Beta(2, 2) | [0, 1] | Discrete-time persistence |
| sigma (residual SD) | HalfNormal(1) | [0, 5] | Data scale |
| lambda (loading) | HalfNormal(1) | [0, 3] | Data scale |
| tau (random SD) | HalfNormal(0.5) | [0, 2] | Data scale |

Both beta and rho priors should be on the **discrete-time scale**. They are \
automatically converted to continuous-time rates internally.

### Literature Evidence
- If you have access to the `search_literature` tool, use it for key causal \
effects where empirical evidence matters. Not every parameter needs a search — \
AR coefficients, residual SDs, and loadings typically use standard defaults.
- Anchor priors on meta-analyses or large longitudinal studies when available.
- If evidence is heterogeneous, use wider priors.

### Continuous-Time Dynamics

Time is measured in fractional days. AR coefficients represent discrete-time \
persistence per observation interval, in (0, 1). The system handles CT conversion.

## Tools

- `validate_model`: Submit your model spec decisions and ALL priors. Returns \
"VALID" on success or detailed feedback on failure. Fix issues and resubmit.
- `search_literature` (if available): Search for empirical effect sizes. Use \
selectively for parameters where domain knowledge is uncertain.
- `elicit_prior_gmm` (if available): Run robust paraphrased elicitation for a \
single parameter. Returns an aggregated prior estimate.

## Workflow

1. Review the pre-computed skeleton and data summary
2. Optionally search literature for key causal effect parameters
3. Submit everything via `validate_model`
4. If validation fails, read the feedback, fix the issues, and resubmit
5. Once you get "VALID", STOP immediately — do not output anything else

IMPORTANT: Once validate_model returns "VALID", STOP.
"""

AGENTIC_USER = """\
## Research Question

{question}

## Full Causal Model Context

Read this full Stage 1a/1b model for context. It is provided so you can see the \
entire latent model and measurement model, not so you can modify them.

{full_causal_model}

## Fixed Model Skeleton

These items have already been derived from the full causal model. Use them for \
context, but do not change them.

### Resolved Likelihoods (fully deterministic — do not change)

{resolved_likelihoods}

### All Parameters (enumerated from DAG — do not add or remove)

{parameters}

## Your Decisions

Only the items below require judgment from you. The rest of the model is context.

### 1. Distribution Choices

For each indicator below, choose the appropriate distribution and link function.

{ambiguous_indicators}

### 2. Loading Constraints
{loading_params}

### 3. Priors

Provide a prior for EVERY parameter listed in "All Parameters". The parameter \
list is fixed; only the prior choices are yours.

## Data Summary

{data_summary}

---

Submit your decisions and priors via the `validate_model` tool as a single JSON:
```json
{{
  "distribution_choices": [
    {{"variable": "...", "distribution": "...", "link": "...", "reasoning": "..."}}
  ],
  "loading_constraints": [
    {{"parameter": "...", "constraint": "positive|none", "reasoning": "..."}}
  ],
  "priors": {{
    "parameter_name": {{
      "parameter": "parameter_name",
      "distribution": "Normal|HalfNormal|Beta|Uniform|TruncatedNormal",
      "params": {{"mu": 0.3, "sigma": 0.15}},
      "sources": [],
      "reasoning": "Justification for the prior"
    }}
  }}
}}
```

Include a prior for EVERY parameter listed above.
"""
