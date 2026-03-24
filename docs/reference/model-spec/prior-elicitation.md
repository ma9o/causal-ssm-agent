# ModelSpec: Prior Elicitation

This page explains the LLM-assisted part of Stage 4.

## Part 2: LLM-Assisted Prior Elicitation

For parameters not fully determined by rules, Stage 4 uses LLM elicitation informed by recent prior-elicitation literature.

### 2.1 What the LLM Specifies

| Parameter type | LLM provides | Rule constraint |
|---|---|---|
| Cross-lag `beta` | `PriorProposal` with exact prior family name, params, reasoning, sources | `none` |
| AR `rho` | `PriorProposal` with exact prior family name, params, reasoning, optional `reference_interval_days` | `unit_interval` |
| Residual `sigma` | `PriorProposal` with exact prior family name, params, reasoning | `positive` |

The `distribution` field uses the exact canonical `PriorDistributionFamily` names documented in [Supported Prior Distribution Families](./prior-distribution-families.md). Aliases are not accepted.

### 2.2 Elicitation Protocol (AutoElicit-style, optional)

Following Capstick et al. (2024), Stage 4 can optionally use paraphrased prompting to reduce brittle overconfidence from any one prompt wording. When `stage4_prior_elicitation.paraphrasing.enabled = true`, the agent receives an `elicit_prior_gmm` tool that supports robust repeated elicitation for a single parameter.

When paraphrased elicitation is enabled, Stage 4 can use multiple paraphrases to reduce brittle overconfidence from any one prompt wording:

1. Generate `N` paraphrased task descriptions
2. For each paraphrase, elicit prior parameters from the LLM
3. Aggregate those responses into a pooled prior or a mixture-of-Gaussians representation

Default behavior keeps paraphrased prompting disabled for cost reasons, so the common path remains a single direct elicitation per parameter.

### 2.3 Prompt Structure

```json
{
  "parameter": "beta_stress_sleep",
  "distribution": "Normal",
  "params": {"mu": 0.2, "sigma": 0.15},
  "sources": [
    {
      "title": "Meta-analysis title",
      "url": "https://...",
      "snippet": "Relevant excerpt",
      "effect_size": "r=0.2"
    }
  ],
  "reasoning": "Justification for the chosen prior distribution and parameters",
  "reference_interval_days": 30
}
```

The worker prompt provides:

1. Parameter name, role, and constraint
2. Structural/domain context from Stage 4
3. Literature evidence, if available
4. The exact allowed prior-family names and parameter-guidance table

### 2.4 Aggregation Strategy

When paraphrasing is enabled, Stage 4 aggregates elicited prior summaries `{(mu_k, sigma_k)}` in one of two ways:

1. **Simple aggregation:** Use the mean of means and a pooled SD, where the final uncertainty reflects both within-prompt uncertainty and between-prompt disagreement
   - `mu_pooled = mean(mu_k)`
   - `sigma_pooled = sqrt(mean(sigma_k^2) + var(mu_k))`
2. **Mixture model:** Fit a Gaussian mixture when responses are clearly multimodal

**References for LLM-assisted prior elicitation**

- Capstick et al. (2024). *AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling.* arXiv: [2411.17284](https://arxiv.org/abs/2411.17284).
- Chen et al. (2025). *LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models.* arXiv: [2508.08300](https://arxiv.org/abs/2508.08300).
- Huang (2025). *LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation.* arXiv: [2508.03766](https://arxiv.org/abs/2508.03766).
- Riegler et al. (2025). *Using large language models to suggest informative prior distributions in Bayesian regression analysis.* *Scientific Reports*. DOI: [10.1038/s41598-025-18425-9](https://www.nature.com/articles/s41598-025-18425-9).
- Selby et al. (2024). *Had Enough of Experts? Elicitation and Evaluation of Bayesian Priors from Large Language Models.* NeurIPS BDU Workshop.

