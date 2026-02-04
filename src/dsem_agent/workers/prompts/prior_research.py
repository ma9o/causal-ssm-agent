"""Stage 4 worker prompts: Prior Research and Elicitation.

Each worker researches a single parameter, using literature evidence
to propose an informed prior distribution.
"""

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
- **Normal(mu, sigma)**: Unconstrained effects (can be positive or negative)
- **HalfNormal(sigma)**: Positive-only parameters (variances, SDs)
- **Beta(alpha, beta)**: Parameters in [0, 1] (probabilities)
- **Uniform(lower, upper)**: When you want to bound the parameter
- **TruncatedNormal(mu, sigma, lower, upper)**: Bounded with a center

### Express Uncertainty Honestly
- Confident (good literature): Use smaller sigma (tighter prior)
- Uncertain (sparse/conflicting evidence): Use larger sigma (wider prior)
- Very uncertain (no evidence): Use weakly informative defaults

### Respect Constraints
- AR coefficients (rho): Must be in [0, 1] for stationarity
- Standard deviations: Must be positive
- Correlations: Must be in [-1, 1]
- Factor loadings: Typically positive by convention

## Output Format

Return a JSON object:
```json
{
  "parameter": "parameter_name",
  "distribution": "Normal|HalfNormal|Beta|Uniform|TruncatedNormal",
  "params": {"mu": 0.3, "sigma": 0.15},
  "sources": [
    {
      "title": "Source title",
      "url": "https://...",
      "snippet": "Relevant excerpt",
      "effect_size": "r=0.3, 95% CI [0.2, 0.4]"
    }
  ],
  "confidence": 0.7,
  "reasoning": "Justification for the prior"
}
```

### Parameter Guidelines by Type

| Parameter Type | Typical Distribution | Typical Range |
|---------------|---------------------|---------------|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] |
| rho (AR coefficient) | Beta(2, 2) or Uniform(0, 1) | [0, 1] |
| sigma (residual SD) | HalfNormal(1) | [0, 5] |
| lambda (loading) | HalfNormal(1) | [0, 3] |
| tau (random SD) | HalfNormal(0.5) | [0, 2] |
"""

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
3. How confident are you in this prior?

If no literature evidence is available, use domain reasoning and be explicit about your uncertainty.

Output your prior as JSON.
"""

NO_LITERATURE = """\
No relevant literature was found for this parameter.

Use domain reasoning to propose a weakly informative prior:
- Consider what effect sizes are typical in this research area
- Think about what would be implausibly large or small
- Be conservative and express uncertainty with a wider prior
"""

# AutoElicit-style paraphrase templates for prior elicitation
# Each template provides a different framing to reduce anchoring bias
PARAPHRASE_TEMPLATES = [
    # 0: Baseline (standard prompt)
    """\
Based on the literature evidence (if any) and domain knowledge, propose a prior for this parameter.

Consider:
1. What is the expected direction (positive/negative) of this effect?
2. What magnitude is plausible given the domain?
3. How confident are you in this prior?

If no literature evidence is available, use domain reasoning and be explicit about your uncertainty.

Output your prior as JSON.""",
    # 1: Uncertainty emphasis
    """\
Focus on expressing your uncertainty about this parameter's value.

Think about:
1. What is the widest reasonable range of values?
2. Where would you place most probability mass?
3. What would surprise you if you observed it?

Be honest about what you don't know. Wider priors are better than overconfident narrow ones.

Output your prior as JSON.""",
    # 2: Direction focus
    """\
First, think about the DIRECTION of this effect before its magnitude.

Ask yourself:
1. Should this effect be positive, negative, or could it go either way?
2. What theoretical or empirical reasons support this direction?
3. Only after establishing direction, what magnitude seems reasonable?

Output your prior as JSON.""",
    # 3: Zero anchor
    """\
Start from the null hypothesis: assume this effect is zero.

Now consider:
1. What evidence would move you away from zero?
2. How far from zero do you expect this effect to be?
3. Is the evidence strong enough to rule out zero?

A prior centered near zero expresses skepticism. Move away only with good reason.

Output your prior as JSON.""",
    # 4: Magnitude thinking
    """\
Think in terms of practical significance, not just statistical significance.

Consider:
1. What effect size would be "small" in this domain?
2. What would be "medium" or "large"?
3. Where does your best guess fall on this scale?

Cohen's conventions (d=0.2, 0.5, 0.8) may or may not apply. Use domain-specific reasoning.

Output your prior as JSON.""",
    # 5: Conservative framing
    """\
Take a skeptical, conservative stance on this parameter.

Ask:
1. What if the true effect is smaller than studies suggest?
2. Publication bias may inflate reported effect sizes. Account for this.
3. What prior would you defend to a skeptical reviewer?

Err on the side of wider, more conservative priors.

Output your prior as JSON.""",
    # 6: Optimistic framing
    """\
Assume the literature evidence is reliable and well-measured.

Consider:
1. If studies are accurate, what does the evidence suggest?
2. Where do multiple sources converge?
3. What prior honors the empirical evidence?

You can be more confident if evidence is consistent across studies.

Output your prior as JSON.""",
    # 7: Practical significance
    """\
Think about what effect size would matter in practice.

Consider:
1. What is the smallest effect that would be practically meaningful?
2. Is the expected effect large enough to matter for interventions?
3. Would policymakers or practitioners care about an effect of this size?

Connect statistical effect to real-world impact.

Output your prior as JSON.""",
    # 8: Cross-study variation
    """\
Focus on how much this effect varies across studies and populations.

Think about:
1. Do different studies report similar or different effect sizes?
2. What explains this heterogeneity (populations, methods, contexts)?
3. Your prior should reflect this between-study variation.

Higher heterogeneity → wider prior uncertainty.

Output your prior as JSON.""",
    # 9: Bound checking
    """\
Think about the logical and empirical bounds on this parameter.

Consider:
1. What are the hard constraints (e.g., must be positive, must be < 1)?
2. What values would be implausibly large or small?
3. What does your prior imply about extreme values?

Check that your prior assigns near-zero probability to impossible values.

Output your prior as JSON.""",
]


def generate_paraphrased_prompts(
    parameter_name: str,
    parameter_role: str,
    parameter_constraint: str,
    parameter_description: str,
    question: str,
    literature_context: str,
    n_paraphrases: int = 10,
) -> list[str]:
    """Generate N paraphrased prompts for prior elicitation.

    Args:
        parameter_name: Name of the parameter
        parameter_role: Role of the parameter in the model
        parameter_constraint: Constraint on the parameter
        parameter_description: Description of the parameter
        question: Research question
        literature_context: Formatted literature evidence
        n_paraphrases: Number of paraphrases to generate (max 10)

    Returns:
        List of formatted prompts using different paraphrase templates
    """
    n = min(n_paraphrases, len(PARAPHRASE_TEMPLATES))

    prompts = []
    base_context = f"""\
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

"""
    for i in range(n):
        prompts.append(base_context + PARAPHRASE_TEMPLATES[i])

    return prompts


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

    return "\n".join(lines)
