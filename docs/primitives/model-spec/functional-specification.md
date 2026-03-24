# ModelSpec: Functional Specification

This page describes how Stage 4 translates the causal DAG, or topological structure, into a fully specified NumPyro/JAX state-space model, or functional specification. The approach combines rule-based constraints with LLM-assisted prior elicitation.

Within the pipeline artifact lineage, this is the semantic bridge from [CausalSpec](../../pipeline/01b-measurement-identifiability.md#causalspec) to [ModelSpec](../../pipeline/04-model-specification-priors.md#modelspec) plus priors. The authoritative output schema remains [Stage 4](../../pipeline/04-model-specification-priors.md). This page owns the semantic explanation of that output before pure compilation.

For the cross-cutting pipeline map, see [../../concepts/pipeline-dimensions.md](../../concepts/pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [../../concepts/artifact-index.md](../../concepts/artifact-index.md).

## Terminology

See [../../concepts/causal-modeling-terminology.md](../../concepts/causal-modeling-terminology.md) for the terminology conventions used in these docs. Stage 4 receives the topological structure from Stage 1a and Stage 1b and translates it into a functional specification: the regression equations, distributions, and priors needed to fit the model in NumPyro.

## Two-Part Architecture

Stage 4 combines two mechanisms:

1. Rule-based specification that constrains the space of valid models
2. LLM-assisted prior elicitation for quantities that still require domain judgment

The detailed rule set lives in [parameters-likelihoods-and-priors.md](parameters-likelihoods-and-priors.md). The elicitation workflow lives in [prior-elicitation.md](prior-elicitation.md).

## Output

Stage 4 produces a `ModelSpec` plus `PriorProposal` objects. The user-facing schema lives in `apps/data-pipeline/src/causal_ssm_agent/orchestrator/schemas_model.py`. Those outputs are then consumed by the SSM compilation pipeline described in [../../model-runtime/compilation.md](../../model-runtime/compilation.md) to build a NumPyro-ready `SSMModel`.

## Stage 4b: Parametric Identifiability

After Stage 4 produces the model specification, **Stage 4b** runs pre-fit parametric identifiability diagnostics before handing off to Stage 5 inference. This catches structural non-identifiability and weakly informed parameters early, before spending compute on expensive MCMC or SVI.

Stage 4b applies a cascade of diagnostics:

1. **T-rule (conservative counting screen):** Compares free parameters to a conservative lower bound on available observed moment conditions. Failure raises a warning about likely overparameterization but does not, by itself, halt the pipeline.
2. **Output sensitivity analysis:** Perturbs each parameter and measures the effect on model outputs. Parameters with near-zero sensitivity are structurally non-identifiable because the data cannot distinguish different values.
3. **Profile likelihood:** For each parameter, optimizes over all other parameters to trace out the profile-likelihood surface. A flat profile indicates non-identifiability; a bounded but shallow profile indicates weak identifiability.

The checked-in Stage 4b runtime currently covers the three diagnostics above. Post-fit power-scaling sensitivity belongs to Stage 5b after fitting, and SBC is not part of the current Stage 4b flow.

These diagnostics target a different failure mode from Stage 1b. Stage 1b asks whether the causal effect is identified from the graph and measurement assumptions. Stage 4b asks whether the chosen parametric form is likely to be recoverable from the available data.

The checked-in implementation lives in `apps/data-pipeline/src/causal_ssm_agent/flows/stages/stage4b_parametric_id.py` and `apps/data-pipeline/src/causal_ssm_agent/utils/parametric_id.py`.

**Reference:** Raue et al. (2009). *Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood.* *Bioinformatics*.

## Implementation Notes

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

## Model Validation: Predictive Checks

Bayesian model validation uses predictive checks at two points in the workflow. These replace a CFA-first workflow with a unified generative approach.

### Prior Predictive Checks (Stage 4)

**When:** After prior elicitation and before fitting to data.

**What:** Simulate data from the generative model using only priors:

1. Sample parameters from their prior distributions
2. Generate implied indicator values through the measurement model
3. Check whether the simulated data are consistent with domain expectations

**Purpose:** Validate that priors plus model structure produce plausible data. This catches priors that are too wide, priors that are too narrow, and structural combinations that imply unreasonable variance patterns.

Examples include absurd values such as loadings of `+/-30`, overly concentrated priors that prevent learning from data, or implied variance structures that are unreasonable for the domain.

**If check fails:** Iterate on prior specification before proceeding to MCMC or more expensive inference.

### Posterior Predictive Checks (Stage 5)

**When:** After fitting the model to extracted data.

**What:** Simulate data from the fitted model and compare to actual data:

1. Sample parameters from the posterior distribution
2. Generate replicated datasets through the full model
3. Compare summary statistics such as means, variances, and quantiles between real and replicated data

**Purpose:** Validate that the fitted model captures relevant aspects of the data. This catches measurement-model misspecification, structural misspecification such as missing edges or wrong functional form, and distribution misspecification such as heavy tails or multimodality.

**Interpretation:** A Bayesian p-value near `0.5` indicates good fit in the sense that replicated data are hard to distinguish from observed data, whereas values near `0` or `1` indicate systematic misfit.

**If check fails:** Revise the model structure, re-fit, and re-check.

### Why Not CFA First?

Traditional SEM validates the measurement model via CFA before fitting the structural model. In Bayesian workflow terms, however, the full generative model is specified and fit together, and prior plus posterior predictive checks replace the separate CFA validation step. This is the workflow perspective argued by Betancourt (2018), Gabry et al. (2019), and Gelman et al. (2020).

**References for predictive-check workflow**

- Anderson, J. C., & Gerbing, D. W. (1988). *Structural equation modeling in practice: A review and recommended two-step approach.* *Psychological Bulletin*, 103(3), 411-423.
- Betancourt, M. (2018). *Towards a Principled Bayesian Workflow.*
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). *Visualization in Bayesian Workflow.* *JRSS-A*, 182(2), 389-402.
- Gelman et al. (2020). *Bayesian Workflow.* arXiv: [2011.01808](https://arxiv.org/abs/2011.01808).

## Future Considerations (ModelSpec-Related)

The following are explicitly not assumed and may be added in future versions:

- **Non-linear relationships:** Currently all structural effects are linear in parameters
- **General non-Gaussian latent dynamics:** Student-t process noise is supported via the particle-filter backend, but more general non-Gaussian dynamics such as jump-diffusion or switching regimes are not
- **Time-varying parameters:** Currently all causal coefficients are time-invariant
- **Random slopes:** Currently only random intercepts, not person-specific effect sizes
- **Cross-level interactions:** Currently between-person variables do not moderate within-person effects
