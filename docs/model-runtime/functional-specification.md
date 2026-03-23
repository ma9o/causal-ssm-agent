# Functional Specification (Stage 4)

This document describes how Stage 4 translates the causal DAG (topological structure) into a fully specified NumPyro/JAX state-space model (functional specification). The approach combines rule-based constraints with LLM-assisted prior elicitation.

Within the pipeline artifact lineage, this document explains the transition from [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) to [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) plus priors. For the cross-cutting pipeline map, see [../concepts/pipeline-dimensions.md](../concepts/pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [../concepts/artifact-index.md](../concepts/artifact-index.md).

---

## Terminology

See AGENTS.md for terminology conventions. Stage 4 receives the topological structure from Stage 1a/1b and translates it into a functional specification: the regression equations, distributions, and priors needed to fit the model in NumPyro.

---

## Two-Part Architecture

### Part 1: Rule-Based Specification (Guardrails)

Deterministic rules that enforce modeling assumptions and constrain the space of valid models.

**1.1 Link Functions from Indicator dtype**

| `measurement_dtype` | Default distribution | Link | Alternatives |
|---------------------|---------------------|------|--------------|
| `continuous` | Gaussian | identity | Student-t, Gamma (log), Beta (logit) |
| `binary` | Bernoulli | logit | — |
| `count` | Poisson | log | Negative Binomial (log) |
| `ordinal` | OrderedLogistic | cumulative logit | — |
| `categorical` | Categorical | softmax | — |

The default distribution is selected automatically from `measurement_dtype`. Alternative distributions for the same dtype can be specified explicitly via per-indicator entries in the `likelihoods` field of `ModelSpec`.

**1.2 Temporal Structure:** AR(1) for all endogenous time-varying constructs. See [../concepts/assumptions.md](../concepts/assumptions.md) A3.

**1.3 Measurement Model Structure:** Single-indicator constructs fix λ=1; multi-indicator constructs use CFA with first loading fixed. See [../concepts/assumptions.md](../concepts/assumptions.md) A6/A9.

**1.4 Cross-Timescale Aggregation**

When cause and effect operate at different granularities:
- Finer → Coarser (e.g., hourly → daily): Aggregate cause using indicator's `aggregation` field
- Coarser → Finer (e.g., weekly → daily): Broadcast coarser to all finer time points

#### 1.5 Parameter Roles and Constraints

Each parameter in the SSM has a **role** (its function in the model) and a **constraint** (its domain restriction). These are enforced by construction — the prior distribution family guarantees the constraint.

**Roles**

| Role | Symbol | Meaning | Appears in |
|------|--------|---------|------------|
| `ar_coefficient` | ρ | Autoregressive persistence of a latent state | Diagonal of **A** |
| `fixed_effect` | β | Cross-lag causal effect between constructs | Off-diagonal of **A** |
| `residual_sd` | σ | Scale of the innovation (process noise) | Diagonal of **G** |
| `loading` | λ | Factor loading mapping latent → observed | Measurement model |
| `correlation` | Ω | Off-diagonal correlation between residuals | Noise covariance |

**Constraints**

| Constraint | Domain | Typical prior families |
|------------|--------|----------------------|
| `none` | (−∞, +∞) | Normal |
| `positive` | (0, +∞) | HalfNormal, HalfCauchy, Exponential, LogNormal |
| `unit_interval` | [0, 1] | Beta, Uniform(0,1) |
| `correlation` | [−1, 1] | LKJCholesky, Uniform(−1,1) |

**Role → Constraint mapping**

| Role | Default constraint | Rationale |
|------|-------------------|-----------|
| `ar_coefficient` | `correlation` | Orchestrator elicits ρ ∈ (−1, 1) in DT terms; `SSMModelBuilder` transforms to CT drift diagonal via `−log(|ρ|)/dt` (magnitude), then the model enforces negativity via `−|x|` for stability. Note: the `|ρ|` means negative DT persistence (oscillatory dynamics) maps to the same CT drift as positive — real-valued OU processes cannot represent oscillatory AR. |
| `fixed_effect` | `none` | Effect sizes can be positive or negative |
| `residual_sd` | `positive` | Standard deviations are non-negative by definition |
| `loading` | `positive` or `none` | LLM decides: `positive` for reference indicator (sign identification), `none` if negative loadings are plausible |
| `correlation` | `correlation` | Bounded by definition |

---

### Part 2: LLM-Assisted Prior Elicitation

For parameters not fully determined by rules, we use LLM elicitation following recent literature.

**2.1 What the LLM Specifies**

| Parameter | LLM provides | Rule constraint |
|-----------|--------------|-----------------|
| Cross-lag β | Mean, SD | None (domain knowledge) |
| AR ρ | Mean, SD | Bounded to (−1, 1) for stationarity |
| Residual σ² | Scale | Must be positive (Exponential/HalfNormal) |

**2.2 Elicitation Protocol (AutoElicit-style, optional)**

Based on Capstick et al. (2024), Stage 4 can optionally use paraphrased prompting to handle LLM overconfidence. When `stage4_prior_elicitation.paraphrasing.enabled=true`, the agent receives an `elicit_prior_gmm` tool that:

1. Generate N paraphrased task descriptions (N=10-100)
2. For each paraphrase, elicit prior parameters from LLM
3. Aggregate into mixture-of-Gaussians: p(β) = Σ π_k · N(μ_k, σ_k)

**Default behavior:** Paraphrased prompting is disabled by default for cost, so the common path is still a single direct elicitation per parameter (Section 2.3).

**2.3 Prompt Structure**

```
You are an expert in {domain} providing prior beliefs for a Bayesian model.

Context: We are estimating the causal effect of {cause} on {effect}.
- {cause}: {description of cause construct}
- {effect}: {description of effect construct}
- Temporal relationship: {lagged/contemporaneous}
- Data context: {brief description of study/data}

Question: What is your prior belief about the regression coefficient β_{effect}_{cause}?

Provide:
1. Your best guess (mean)
2. Your uncertainty (standard deviation)
3. Brief reasoning (1-2 sentences)

Output as JSON: {"mean": X, "std": Y, "reasoning": "..."}
```

**2.4 Aggregation Strategy**

When paraphrasing is enabled, Stage 4 aggregates N elicited priors {(μ_k, σ_k)}:

1. **Simple aggregation**: Use mean of means, pooled SD
   - μ_pooled = mean(μ_k)
   - σ_pooled = sqrt(mean(σ_k²) + var(μ_k))

2. **Mixture model**: Fit K-component GMM (if responses are multimodal)

---

## Output

Stage 4 produces a `ModelSpec` (see `orchestrator/schemas_model.py`) consumed by the SSM compilation pipeline (see [compilation.md](compilation.md)) to build a NumPyro-ready `SSMModel`.

---

## Stage 4b: Parametric Identifiability

After Stage 4 produces the model specification, **Stage 4b** runs pre-fit parametric identifiability diagnostics before handing off to Stage 5 (inference). This catches structural non-identifiability and weakly informed parameters early -- before spending compute on expensive MCMC/SVI.

Stage 4b applies a cascade of diagnostics:

1. **T-rule (conservative counting screen):** Compares free parameters to a conservative lower bound on available observed moment conditions. Failure raises a warning about likely overparameterization but does not, by itself, halt the pipeline.
2. **Output sensitivity analysis:** Perturbs each parameter and measures the effect on model outputs. Parameters with near-zero sensitivity are structurally non-identifiable (the data cannot distinguish different values).
3. **Profile likelihood:** For each parameter, optimizes over all other parameters to trace out the profile likelihood surface. A flat profile indicates non-identifiability; a bounded but shallow profile indicates weak identifiability (Raue et al., 2009).

The checked-in Stage 4b runtime currently covers the three diagnostics above. Post-fit power-scaling sensitivity belongs to Stage 5b after fitting, and SBC is not part of the current Stage 4b flow.

See `flows/stages/stage4b_parametric_id.py` and `utils/parametric_id.py` for implementation.

---

## Literature

See [literature.md](../literature.md) for the full bibliography. Key papers for Stage 4: AutoElicit (Capstick et al. 2024), LLM-BI (Chen et al. 2025), LLM-Prior (Huang 2025).

---

## Implementation Notes

### Why Not Fully Rule-Based?

Effect sizes are fundamentally domain-specific. A β = -0.3 between stress and mood is plausible; between weather and stock prices, less so. Rules can constrain the *form* (Normal, bounded) but not the *content* (what's a reasonable effect size).

### Why Not Fully LLM-Based?

LLMs can produce invalid statistical objects (negative variances, improper distributions). Rule-based guardrails ensure the output is always a valid NumPyro model.

### The Hybrid Approach

```
┌─────────────────────────────────────────────────┐
│  DAG + Measurement Model (from Stage 1a/1b)    │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Rule-Based Engine     │
          │  - Link functions       │
          │  - AR(1) structure      │
          │  - Coefficient bounds   │
          │  - Measurement model    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   LLM Prior Elicitor    │
          │  - Effect size means    │
          │  - Uncertainty (SD)     │
          │  - Domain reasoning     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Aggregation Layer     │
          │  - Mixture-of-Gaussians │
          │  - Constraint checking  │
          └────────────┬────────────┘
                       │
                       ▼
             SSMSpec (NumPyro-ready)
```

---

## Model Validation: Predictive Checks

Bayesian model validation uses predictive checks at two points in the workflow. These replace frequentist CFA-style validation with a unified generative approach.

### Prior Predictive Checks (Stage 4)

**When:** After prior elicitation, before fitting to data.

**What:** Simulate data from the generative model using only priors (no observed data):
1. Sample parameters from their prior distributions (loadings, AR coefficients, effect sizes)
2. Generate implied indicator values through the measurement model
3. Check if simulated data is consistent with domain expectations

**Purpose:** Validate that priors + model structure produce plausible data. Catches:
- Priors too wide (absurd values like loadings of ±30)
- Priors too narrow (artificially constrained, won't learn from data)
- Structural misspecification (implied variance structure is unreasonable)

**If check fails:** Iterate on prior specification before proceeding to MCMC.

### Posterior Predictive Checks (Stage 5)

**When:** After fitting the model to extracted data.

**What:** Simulate data from the fitted model and compare to actual data:
1. Sample parameters from the posterior distribution
2. Generate replicated datasets through the full model
3. Compare summary statistics (mean, variance, quantiles) between real and replicated data

**Purpose:** Validate that the fitted model captures relevant aspects of the data. Catches:
- Measurement model misspecification (indicators don't reflect constructs)
- Structural misspecification (missing edges, wrong functional form)
- Distribution misspecification (heavy tails, multimodality)

**Interpretation:** A Bayesian p-value near 0.5 indicates good fit (replicated data indistinguishable from observed); near 0 or 1 indicates systematic misfit.

**If check fails:** Revise model structure, re-fit, and re-check.

### Why Not CFA First?

Traditional SEM validates the measurement model via CFA before fitting the structural model (Anderson & Gerbing, 1988). In Bayesian workflow, the full generative model is specified and fit together; prior/posterior predictive checks replace the separate CFA validation step. See Gelman et al. (2020), Betancourt (2018).
