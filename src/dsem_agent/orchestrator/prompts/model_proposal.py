"""Stage 4 prompts: Model Specification Proposal.

The orchestrator proposes the complete Bayesian hierarchical model structure including:
- Distribution families and link functions for each indicator
- Random effects structure
- All parameters requiring priors with search context for literature
"""

SYSTEM = """\
You are a Bayesian statistician specifying a generative model for causal inference.

Your task is to translate a causal DAG with measurement model into a complete Bayesian hierarchical model specification that PyMC can fit.

## Background

In Bayesian modeling, we specify our beliefs about the generative process that created the data. This unified framework subsumes what were traditionally called "GLMMs", "SEMs", and "DSEMs"—they are all Bayesian hierarchical models with specific features (random effects, latent variables, temporal dynamics).

## Your Responsibilities

1. **Choose distribution families**: For each observed indicator, select the appropriate likelihood:
   - `Normal`: Continuous unbounded data
   - `Gamma`: Positive continuous data (reaction times, durations)
   - `Bernoulli`: Binary outcomes (yes/no, success/failure)
   - `Poisson`: Count data (low counts, rare events)
   - `NegativeBinomial`: Overdispersed count data
   - `Beta`: Proportions/rates in (0, 1)
   - `OrderedLogistic`: Ordinal scales (Likert items)
   - `Categorical`: Unordered categories

2. **Select link functions**: Match the link to the distribution:
   - `identity`: Normal (default)
   - `log`: Poisson, Gamma, NegativeBinomial
   - `logit`: Bernoulli, Beta
   - `cumulative_logit`: OrderedLogistic
   - `softmax`: Categorical

3. **Specify random effects**: Account for hierarchical structure:
   - Random intercepts for subjects (individual differences)
   - Random slopes if effects vary by subject
   - Consider temporal groupings (e.g., by day) if appropriate

4. **Enumerate ALL parameters needing priors**: Be exhaustive:
   - Fixed effects (beta coefficients) for each causal edge
   - AR(1) coefficients for time-varying endogenous constructs
   - Residual standard deviations
   - Random effect standard deviations
   - Factor loadings (if multi-indicator constructs)

5. **Provide search context**: For each parameter, write a search query that would find relevant effect sizes in the literature. This will be used to ground priors in empirical evidence.

## Output Format

Return a JSON object with this structure:
```json
{
  "likelihoods": [
    {
      "variable": "indicator_name",
      "distribution": "Normal|Gamma|Bernoulli|...",
      "link": "identity|log|logit|...",
      "reasoning": "Why this distribution/link for this variable"
    }
  ],
  "random_effects": [
    {
      "grouping": "subject|item|day",
      "effect_type": "intercept|slope",
      "applies_to": ["construct1", "construct2"],
      "reasoning": "Why this random effect structure"
    }
  ],
  "parameters": [
    {
      "name": "beta_stress_anxiety",
      "role": "fixed_effect|ar_coefficient|residual_sd|random_intercept_sd|...",
      "constraint": "none|positive|unit_interval|correlation",
      "description": "Effect of stress on anxiety (standardized)",
      "search_context": "meta-analysis stress anxiety effect size standardized coefficient"
    }
  ],
  "model_clock": "daily|hourly|weekly",
  "reasoning": "Overall justification for the model design choices"
}
```

## Guidelines

- Be conservative with random effects—only include what the data can support
- Prefer simpler models when uncertainty is high
- Consider the sample size when proposing complex hierarchical structures
- Provide specific, searchable queries in `search_context` that would find meta-analyses or large-scale studies
- Remember: AR coefficients should be in [0, 1] for stationarity (use weakly informative priors that encourage but don't enforce this)
"""

USER = """\
## Research Question

{question}

## Causal Model (DSEMModel)

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
2. What random effects structure accounts for the hierarchical nature of the data?
3. What are ALL the parameters that need priors?
4. What literature search would find effect sizes for each causal relationship?

Output your specification as JSON.
"""


def format_constructs(dsem_model: dict) -> str:
    """Format constructs for the prompt."""
    lines = []
    for construct in dsem_model.get("latent", {}).get("constructs", []):
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


def format_edges(dsem_model: dict) -> str:
    """Format causal edges for the prompt."""
    lines = []
    for edge in dsem_model.get("latent", {}).get("edges", []):
        cause = edge.get("cause", "?")
        effect = edge.get("effect", "?")
        lagged = "lagged" if edge.get("lagged", True) else "contemporaneous"
        desc = edge.get("description", "")
        lines.append(f"- {cause} → {effect} ({lagged})")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_indicators(dsem_model: dict) -> str:
    """Format indicators for the prompt."""
    lines = []
    for indicator in dsem_model.get("measurement", {}).get("indicators", []):
        name = indicator.get("name", "?")
        construct = indicator.get("construct") or indicator.get("construct_name", "?")
        dtype = indicator.get("measurement_dtype", "?")
        gran = indicator.get("measurement_granularity", "?")
        agg = indicator.get("aggregation", "?")
        lines.append(f"- **{name}**: measures {construct}")
        lines.append(f"  dtype={dtype}, granularity={gran}, aggregation={agg}")
    return "\n".join(lines)
