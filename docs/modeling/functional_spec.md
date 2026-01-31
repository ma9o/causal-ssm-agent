# Functional Specification (Stage 4)

This document describes how Stage 4 translates the causal DAG (topological structure) into a fully specified PyMC model (functional specification). The approach combines rule-based constraints with LLM-assisted prior elicitation.

---

## Terminology

Per CLAUDE.md, we distinguish:

| Concept | Term | Stage |
|---------|------|-------|
| DAG encoding parent-child relationships | **Topological structure** | Stage 1a/1b |
| Mathematical form of causal mechanisms | **Functional specification** | Stage 4 |

Stage 4 bridges these: given the DAG, it specifies the regression equations, distributions, and priors needed to fit the model in PyMC.

---

## Two-Part Architecture

### Part 1: Rule-Based Specification (Guardrails)

Deterministic rules that enforce modeling assumptions and constrain the space of valid models.

**1.1 Link Functions from Indicator dtype**

| `measurement_dtype` | Distribution | Link | PyMC |
|---------------------|--------------|------|------|
| `continuous` | Gaussian | identity | `pm.Normal` |
| `binary` | Bernoulli | logit | `pm.Bernoulli(logit_p=...)` |
| `count` | Poisson | log | `pm.Poisson(mu=pm.math.exp(...))` |
| `ordinal` | OrderedLogistic | cumulative logit | `pm.OrderedLogistic` |
| `categorical` | Categorical | softmax | `pm.Categorical` |

**1.2 Temporal Structure (from A3 Markov assumption)**

All endogenous time-varying constructs receive AR(1):
```
Construct_t = ρ · Construct_{t-1} + Σ β_j · Parent_j + ε_t
```

Where:
- ρ ∈ [0, 1] for stability (enforced via prior bounds)
- β_j are cross-lag coefficients for each causal edge
- ε_t ~ N(0, σ²) is the structural residual

**1.3 Measurement Model Structure (from A6/A9)**

| Indicator count | Structure | Loadings |
|-----------------|-----------|----------|
| Single (=1) | Construct ≡ Indicator | λ = 1 (fixed) |
| Multiple (≥2) | CFA structure | λ_1 = 1, λ_2+ estimated |

The measurement equation follows the standard factor analysis form:

```
x_i = τ_i + λ_i · ξ + ε_i
```

Where x is the observed indicator, τ is the intercept, λ is the factor loading, ξ is the latent construct, and ε is measurement error.

**PyMC Implementation Pattern** (from [PyMC CFA/SEM Example](https://www.pymc.io/projects/examples/en/latest/case_studies/CFA_SEM.html)):

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model():
    # Estimate all loadings with weakly informative prior
    lambdas_ = pm.Normal("lambdas_raw", mu=1, sigma=10, dims="indicators")

    # Fix first loading to 1 for scale identification
    lambdas = pm.Deterministic(
        "lambdas",
        pt.set_subtensor(lambdas_[0], 1.0),
        dims="indicators"
    )
```

Key points:
- First loading fixed to 1.0 establishes the measurement scale
- Remaining loadings estimated freely with Normal(1, 10) prior
- The closer loadings are to 1, the better the indicators measure the same construct
- If loadings deviate substantially from 1, consider whether indicators belong together

**1.4 Cross-Timescale Aggregation**

When cause and effect operate at different granularities:
- Finer → Coarser (e.g., hourly → daily): Aggregate cause using indicator's `aggregation` field
- Coarser → Finer (e.g., weekly → daily): Broadcast coarser to all finer time points

**1.5 Coefficient Bounds**

| Parameter | Constraint | Rationale |
|-----------|------------|-----------|
| AR coefficient ρ | [0, 1] | Stationarity; negative AR rare in behavioral data |
| Factor loadings λ | [0, ∞) | Sign convention: all loadings positive |
| Residual variance σ² | (0, ∞) | Must be positive |

---

### Part 2: LLM-Assisted Prior Elicitation

For parameters not fully determined by rules, we use LLM elicitation following recent literature.

**2.1 What the LLM Specifies**

| Parameter | LLM provides | Rule constraint |
|-----------|--------------|-----------------|
| Cross-lag β | Mean, SD | None (domain knowledge) |
| AR ρ | Mean, SD | Bounded to [0, 1] |
| Residual σ² | Scale | Must be positive (Exponential/HalfNormal) |

**2.2 Elicitation Protocol (AutoElicit-style)**

Based on Capstick et al. (2024), we use paraphrased prompting:

1. Generate N paraphrased task descriptions (N=10-100)
2. For each paraphrase, elicit prior parameters from LLM
3. Aggregate into mixture-of-Gaussians: p(β) = Σ π_k · N(μ_k, σ_k)

This handles LLM overconfidence by capturing variance across phrasings.

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

From N elicited priors {(μ_k, σ_k)}:

1. **Simple aggregation**: Use mean of means, pooled SD
   - μ_pooled = mean(μ_k)
   - σ_pooled = sqrt(mean(σ_k²) + var(μ_k))

2. **Mixture model**: Fit K-component GMM (if responses are multimodal)

---

## Output Schema

Stage 4 produces a `ModelSpec` dict:

```python
{
    "constructs": {
        "mood": {
            "type": "endogenous",
            "temporal": "time_varying",
            "granularity": "daily",
            "ar_prior": {"dist": "Uniform", "lower": 0, "upper": 1},
        },
        ...
    },
    "edges": {
        "β_mood_stress": {
            "cause": "stress",
            "effect": "mood",
            "lagged": True,
            "prior": {"dist": "Normal", "mean": -0.3, "std": 0.2},
        },
        ...
    },
    "measurement": {
        "hrv": {
            "construct": "stress",
            "dtype": "continuous",
            "link": "identity",
            "loading_prior": {"dist": "HalfNormal", "sigma": 1},
        },
        ...
    },
    "residuals": {
        "σ²_mood": {"dist": "Exponential", "scale": 1},
        ...
    },
    "time_index": "daily",  # Model clock (finest endogenous outcome)
}
```

---

## Literature Foundation

### LLM-Assisted Prior Elicitation

| Paper | Key Contribution |
|-------|------------------|
| [LLM-BI](https://arxiv.org/abs/2508.08300) (2025) | Full model specification (priors + likelihood) from NL |
| [AutoElicit](https://arxiv.org/abs/2411.17284) (2024) | Paraphrased prompting + mixture aggregation |
| [LLM-Prior](https://arxiv.org/abs/2508.03766) (2025) | Coupling LLM with tractable generative model |
| [Riegler et al.](https://www.nature.com/articles/s41598-025-18425-9) (2025) | Tested Claude/Gemini on real datasets with reflection |

### Key Findings

1. **Low-data regime**: LLM priors most beneficial when training data is scarce
2. **Paraphrasing handles overconfidence**: Variance across phrasings captures uncertainty
3. **Rule constraints improve reliability**: Restricting allowed distributions helps
4. **Clinical validation**: LLM priors reduced required sample sizes in trials

### Limitations

- LLM priors may not match "true" internal knowledge (Selby et al., 2024)
- Performance is task-dependent
- No replacement for genuine domain expertise when available

---

## Implementation Notes

### Why Not Fully Rule-Based?

Effect sizes are fundamentally domain-specific. A β = -0.3 between stress and mood is plausible; between weather and stock prices, less so. Rules can constrain the *form* (Normal, bounded) but not the *content* (what's a reasonable effect size).

### Why Not Fully LLM-Based?

LLMs can produce invalid statistical objects (negative variances, improper distributions). Rule-based guardrails ensure the output is always a valid PyMC model.

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
              PyMC-Ready ModelSpec
```

---

## References

### Bayesian SEM in PyMC

- PyMC Development Team. [Confirmatory Factor Analysis and Structural Equation Models in Psychometrics](https://www.pymc.io/projects/examples/en/latest/case_studies/CFA_SEM.html). PyMC Example Gallery.

### LLM-Assisted Prior Elicitation

- Capstick, A., et al. (2024). AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling. arXiv:2411.17284.
- Huang, Y. (2025). LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation. arXiv:2508.03766.
- Chen, Z., et al. (2025). LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models. arXiv:2508.08300.
- Riegler, M., et al. (2025). Using large language models to suggest informative prior distributions in Bayesian regression analysis. Scientific Reports.
- Selby, J., et al. (2024). Had Enough of Experts? Elicitation and Evaluation of Bayesian Priors from Large Language Models. NeurIPS BDU Workshop.
