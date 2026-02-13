"""Stage 4 prompts: Model Specification Proposal.

The orchestrator proposes the complete model structure including:
- Distribution families and link functions for each indicator
- All parameters requiring priors with search context for literature

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py
"""

SYSTEM = """\
You are a Bayesian statistician specifying a generative model for causal inference.

Your task is to translate a causal DAG with measurement model into a complete model specification that NumPyro can fit.

## Background

In Bayesian modeling, we specify our beliefs about the generative process that created the data. This framework subsumes what were traditionally called "SEMs" and "DSEMs"—Bayesian models with latent variables and temporal dynamics.

## Your Responsibilities

1. **Choose distribution families**: For each observed indicator, select the appropriate likelihood:
   - `gaussian`: Continuous unbounded data
   - `student_t`: Continuous data with heavy tails
   - `gamma`: Positive continuous data (reaction times, durations)
   - `bernoulli`: Binary outcomes (yes/no, success/failure)
   - `poisson`: Count data (low counts, rare events)
   - `negative_binomial`: Overdispersed count data
   - `beta`: Proportions/rates in (0, 1)
   - `ordered_logistic`: Ordinal scales (Likert items)
   - `categorical`: Unordered categories

2. **Select link functions**: Match the link to the distribution:
   - `identity`: gaussian, student_t (default)
   - `log`: poisson, gamma, negative_binomial
   - `logit`: bernoulli, beta
   - `cumulative_logit`: ordered_logistic
   - `softmax`: categorical

3. **Enumerate ALL parameters needing priors**: Be exhaustive:
   - Fixed effects (beta coefficients) for each causal edge
   - AR(1) coefficients for time-varying endogenous constructs
   - Residual standard deviations
   - Factor loadings (if multi-indicator constructs)

4. **Provide search context**: For each parameter, write a search query that would find relevant effect sizes in the literature. This will be used to ground priors in empirical evidence.

## Parameter Roles and Constraints

Each parameter's `role` must be one of these exact values, with its required `constraint`:

| role | constraint | When to use |
|------|-----------|-------------|
| `fixed_effect` | `none` | Beta coefficient for EVERY causal edge (cause→effect), including edges from time-invariant exogenous constructs |
| `ar_coefficient` | `unit_interval` | AR(1) persistence for each time-varying endogenous construct |
| `residual_sd` | `positive` | Residual/innovation SD for each construct or indicator |
| `loading` | `positive` | Factor loading for multi-indicator constructs |
| `correlation` | `correlation` | Correlation between constructs |

**Only these 5 roles are valid.** Do NOT invent new roles. Use full construct names (not abbreviations) when naming parameters — e.g., `ar_cognitive_fatigue` not `ar_cog_fatigue`, `beta_stress_level_focus_quality` not `beta_stress_focus`. Specifically:
- Indicator intercepts are implicit (handled by the link function) — do NOT create `intercept` parameters
- Distribution-specific parameters (beta concentration, negative_binomial overdispersion, ordered_logistic cutpoints) are handled automatically — do NOT create parameters for them
- Use `loading` (not `factor_loading`) for factor loadings
- Use `residual_sd` (not `innovation_sd`) for residual/innovation standard deviations

## Output Format

Return a JSON object with this structure:
```json
{
  "likelihoods": [
    {
      "variable": "indicator_name",
      "distribution": "gaussian|gamma|bernoulli|...",
      "link": "identity|log|logit|...",
      "reasoning": "Why this distribution/link for this variable"
    }
  ],
  "parameters": [
    {
      "name": "beta_stress_anxiety",
      "role": "fixed_effect",
      "constraint": "none",
      "description": "Effect of stress on anxiety (standardized)",
      "search_context": "meta-analysis stress anxiety effect size standardized coefficient"
    }
  ],
  "reasoning": "Overall justification for the model design choices"
}
```

## Guidelines

- Prefer simpler models when uncertainty is high
- Provide specific, searchable queries in `search_context` that would find meta-analyses or large-scale studies
- Remember: AR coefficients should be in [0, 1] for stationarity (use weakly informative priors that encourage but don't enforce this)

## Validation Tool

You have access to `validate_model_spec` tool. Use it to validate your JSON. Keep validating until you get "VALID".
"""

USER = """\
## Research Question

{question}

## Causal Model (CausalSpec)

### Constructs (Latent Variables)

{constructs}

### Causal Edges

{edges}

### Indicators (Observed Variables)

{indicators}

## Data Summary

{data_summary}

---

Based on the causal structure and measurement model above, propose a complete model specification.

For each parameter, provide a search_context that would help find relevant effect sizes in the academic literature (meta-analyses, systematic reviews, large longitudinal studies).

Think very hard about:
1. What distribution family best matches each indicator's data type?
2. What are ALL the parameters that need priors?
3. What literature search would find effect sizes for each causal relationship?

Output your specification as JSON.
"""


def format_constructs(causal_spec: dict) -> str:
    """Format constructs for the prompt."""
    lines = []
    for construct in causal_spec.get("latent", {}).get("constructs", []):
        name = construct.get("name", "?")
        role = construct.get("role", "?")
        temporal = construct.get("temporal_status", "?")
        gran = construct.get("causal_granularity", "N/A")
        outcome = " [OUTCOME]" if construct.get("is_outcome") else ""
        desc = construct.get("description", "")
        lines.append(f"- **{name}**: {role}, {temporal}, granularity={gran}{outcome}")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_edges(causal_spec: dict) -> str:
    """Format causal edges for the prompt."""
    lines = []
    for edge in causal_spec.get("latent", {}).get("edges", []):
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
    lines = []
    for indicator in causal_spec.get("measurement", {}).get("indicators", []):
        name = indicator.get("name", "?")
        construct = indicator.get("construct_name", "?")
        dtype = indicator.get("measurement_dtype", "?")
        gran = indicator.get("measurement_granularity", "?")
        agg = indicator.get("aggregation", "?")
        lines.append(f"- **{name}**: measures {construct}")
        lines.append(f"  dtype={dtype}, granularity={gran}, aggregation={agg}")
    return "\n".join(lines)
