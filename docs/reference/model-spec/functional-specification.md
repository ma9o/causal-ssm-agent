# ModelSpec: Functional Specification

`ModelSpec` is the Stage 4 domain primitive that turns the causal-and-measurement handoff into a fitting-ready functional specification. The authoritative schema lives in [Stage 4](../../pipeline/04-model-specification-priors.md). The detailed rule set lives in [parameters-likelihoods-and-priors.md](parameters-likelihoods-and-priors.md). The elicitation workflow lives in [prior-elicitation.md](prior-elicitation.md).

## Terminology

See [../terminology.md](../terminology.md) for the terminology conventions used in these docs. Stage 4 receives the topological structure from Stage 1a and Stage 1b and translates it into a functional specification: the regression equations, distributions, and priors needed to fit the model in NumPyro.

## Two-Part Architecture

Stage 4 combines two mechanisms:

1. Rule-based specification that constrains the space of valid models
2. LLM-assisted prior elicitation for quantities that still require domain judgment

### Why Not Fully Rule-Based?

Effect sizes are fundamentally domain-specific. A beta of `-0.3` between stress and mood may be plausible; the same effect between weather and stock prices is much less so. Rules can constrain the *form* of the prior or model, but not the *content* of what a reasonable effect size should be.

### Why Not Fully LLM-Based?

LLMs can produce invalid statistical objects such as negative variances or improper distributions. Rule-based guardrails ensure that the output remains a valid Bayesian model.

### The Hybrid Approach

```text
CausalSpec
    |
    +--> rule-based engine
    |      - link functions
    |      - AR(1) structure
    |      - coefficient bounds
    |      - measurement-model constraints
    |
    +--> LLM prior elicitor
           - effect size means
           - uncertainty
           - domain reasoning
    |
    +--> aggregation layer
           - pooled priors
           - mixture-of-Gaussians when needed
           - constraint checks
    |
    v
ModelSpec + PriorProposal
```

## Output

Stage 4 produces a `ModelSpec` plus `PriorProposal` objects. The user-facing schema lives in `apps/data-pipeline/src/causal_ssm_agent/orchestrator/schemas_model.py`. Those outputs are then consumed by the SSM compilation pipeline described in [../compilation.md](../compilation.md) to build a NumPyro-ready `SSMModel`.
