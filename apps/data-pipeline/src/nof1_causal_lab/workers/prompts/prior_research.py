"""model-spec worker prompts: Prior Research and Elicitation.

Each worker researches a single parameter, using literature evidence
to propose an informed prior distribution.
"""

from nof1_causal_lab.distributions import (
    format_prior_distribution_choice_list,
    render_dynamic_prior_scale_guidance,
    render_prior_distribution_guidance_bullets,
    render_prior_parameter_guidance_markdown_table,
)

PRIOR_DISTRIBUTION_CHOICE_LIST = format_prior_distribution_choice_list()
PRIOR_DISTRIBUTION_GUIDANCE_BULLETS = render_prior_distribution_guidance_bullets()
PRIOR_PARAMETER_GUIDANCE_TABLE = render_prior_parameter_guidance_markdown_table()
DYNAMIC_PRIOR_SCALE_GUIDANCE = render_dynamic_prior_scale_guidance()

SYSTEM = """\
You are a Bayesian statistician eliciting a prior distribution for a single model parameter.

Your task is to propose an **informative prior** based on:
1. Literature evidence (if provided)
2. Domain knowledge about plausible effect sizes
3. The parameter's role and constraints

## Guidelines

### Use Literature Evidence Wisely
- If meta-analyses or large-scale studies are provided, anchor your prior on their effect sizes
- Weight evidence by study quality: meta-analyses > large longitudinal > cross-sectional
- If effect sizes are heterogeneous, inflate your uncertainty (larger std)
- If no relevant literature exists, fall back to domain reasoning

### Choose Appropriate Distributions
__PRIOR_DISTRIBUTION_GUIDANCE_BULLETS__

### Express Uncertainty Via Prior Width
- Good literature: Use smaller sigma (tighter prior)
- Sparse/conflicting evidence: Use larger sigma (wider prior)
- No evidence: Use weakly informative defaults

### Respect Constraints
- AR coefficients (rho): Must be in [0, 1] as baseline persistence absent incoming feedback
- Standard deviations: Must be positive
- Correlations: Must be in [-1, 1]
- Factor loadings: Must respect the fixed measurement-structure polarity (`positive` or `negative`)

## Output Format

Return a JSON object:
```json
{
  "parameter": "parameter_name",
  "distribution": "__PRIOR_DISTRIBUTION_CHOICE_LIST__",
  "params": {"mu": 0.3, "sigma": 0.15},
  "sources": [
    {
      "title": "Source title",
      "url": "https://...",
      "snippet": "Relevant excerpt",
      "effect_size": "r=0.3, 95% CI [0.2, 0.4]"
    }
  ],
  "reasoning": "Justification for the prior",
  "reference_interval_days": 7.0
}
```

Only include `reference_interval_days` when the evidence is expressed on a \
different observation interval than the model interval. For lagged `beta_*` \
priors, keep `params` on that authored interval scale and let the compiler \
rescale them.

### Parameter Guidelines by Type
__PRIOR_PARAMETER_GUIDANCE_TABLE__

**Important**: __DYNAMIC_PRIOR_SCALE_GUIDANCE__
"""

SYSTEM = SYSTEM.replace(
    "__PRIOR_DISTRIBUTION_CHOICE_LIST__",
    PRIOR_DISTRIBUTION_CHOICE_LIST,
)
SYSTEM = SYSTEM.replace(
    "__PRIOR_DISTRIBUTION_GUIDANCE_BULLETS__",
    PRIOR_DISTRIBUTION_GUIDANCE_BULLETS,
)
SYSTEM = SYSTEM.replace(
    "__PRIOR_PARAMETER_GUIDANCE_TABLE__",
    PRIOR_PARAMETER_GUIDANCE_TABLE,
)
SYSTEM = SYSTEM.replace(
    "__DYNAMIC_PRIOR_SCALE_GUIDANCE__",
    DYNAMIC_PRIOR_SCALE_GUIDANCE,
)

USER = """\
## Parameter to Elicit

**Name**: {parameter_name}
**Role**: {parameter_role}
**Constraint**: {parameter_constraint}
**Description**: {parameter_description}

## Research Context

**Question**: {question}

## Literature Evidence

{literature_context}

---

Based on the literature evidence (if any) and domain knowledge, propose a prior for this parameter.

Consider:
1. What is the expected direction (positive/negative) of this effect?
2. What magnitude is plausible given the domain?

If no literature evidence is available, use domain reasoning and be explicit about your uncertainty (use a wider prior sigma).

Output your prior as JSON.
"""

NO_LITERATURE = """\
No relevant literature was found for this parameter.

Use domain reasoning to propose a weakly informative prior:
- Consider what effect sizes are typical in this research area
- Think about what would be implausibly large or small
- Be conservative and express uncertainty with a wider prior
"""


def format_literature_for_parameter(
    sources: list[dict],
) -> str:
    """Format literature sources for a single parameter.

    Args:
        sources: List of source dicts from Exa search

    Returns:
        Formatted string for prompt
    """
    if not sources:
        return NO_LITERATURE

    lines = ["### Relevant Literature\n"]

    for i, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        snippet = source.get("snippet", "")
        effect_size = source.get("effect_size", "")

        lines.append(f"**Source {i}**: {title}")
        if url:
            lines.append(f"URL: {url}")
        if snippet:
            lines.append(f"Excerpt: {snippet}")
        if effect_size:
            lines.append(f"Effect size: {effect_size}")
        lines.append("")

    lines.extend(
        [
            "If you cite these results in a model-spec prior, each `sources` entry must be an object like:",
            '`{"title": "...", "snippet": "...", "url": "https://...", "effect_size": "β=0.2", "study_interval_days": 7.0}`',
            "Only `title` and `snippet` are required. If you are unsure, use `sources: []` instead of a malformed entry.",
        ]
    )

    return "\n".join(lines)
