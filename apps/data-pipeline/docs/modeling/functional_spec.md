# Functional Specification (Stage 4)

This document describes how Stage 4 translates the causal DAG (topological structure) into a fully specified NumPyro/JAX state-space model (functional specification). The approach combines rule-based constraints with LLM-assisted prior elicitation.

---

## Terminology

See AGENTS.md for terminology conventions. Stage 4 receives the topological structure from Stage 1a/1b (see [pipeline.md](../reference/pipeline.md)) and translates it into a functional specification: the regression equations, distributions, and priors needed to fit the model in NumPyro.

---

## Two-Part Architecture

### Part 1: Rule-Based Specification (Guardrails)

Deterministic rules that enforce modeling assumptions and constrain the space of valid models.

**1.1 Link Functions from Indicator dtype**

| `measurement_dtype` | Distribution | Link | NumPyro |
|---------------------|--------------|------|---------|
| `continuous` | Gaussian | identity | `numpyro.distributions.Normal` |
| `binary` | Bernoulli | logit | `numpyro.distributions.Bernoulli(logits=...)` |
| `count` | Poisson | log | `numpyro.distributions.Poisson(rate=jnp.exp(...))` |
| `ordinal` | OrderedLogistic | cumulative logit | `numpyro.distributions.OrderedLogistic` |
| `categorical` | Categorical | softmax | `numpyro.distributions.Categorical` |

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

**NumPyro Implementation Pattern:**

```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model():
    # Estimate all loadings with weakly informative prior
    lambdas_raw = numpyro.sample("lambdas_raw", dist.Normal(1.0, 10.0).expand([n_indicators]))

    # Fix first loading to 1 for scale identification
    lambdas = numpyro.deterministic(
        "lambdas",
        lambdas_raw.at[0].set(1.0),
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
| `ar_coefficient` | `correlation` | Stationarity requires ρ ∈ (−1, 1); negative values indicate oscillatory dynamics |
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

Stage 4 produces a model specification dict that is consumed by `SSMModelBuilder` to construct an `SSMSpec`/`SSMModel`:

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

## Stage 4b: Parametric Identifiability

After Stage 4 produces the model specification, **Stage 4b** runs pre-fit parametric identifiability diagnostics before handing off to Stage 5 (inference). This catches structural non-identifiability, boundary identifiability, and weakly informed parameters early -- before spending compute on expensive MCMC/SVI.

Stage 4b computes the Fisher information matrix at prior draws and checks:
- **Rank deficiency:** Structurally non-identifiable parameters (zero Fisher information)
- **Boundary identifiability:** Intermittent rank deficiency across draws
- **Weak contraction:** Parameters with low expected prior-to-posterior contraction

See `src/causal_ssm_agent/flows/stages/stage4b_parametric_id.py` and `src/causal_ssm_agent/utils/parametric_id.py` for implementation.

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

**Implementation:**
```python
# NumPyro pattern — uses numpyro.infer.Predictive
from numpyro.infer import Predictive

predictive = Predictive(model_fn, num_samples=1000)
prior_pred = predictive(rng_key)

# Check: are simulated indicator values in plausible range?
# Check: do simulated effect sizes match domain expectations?
```

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

**Implementation:**
```python
# NumPyro pattern — uses numpyro.infer.Predictive with posterior samples
from numpyro.infer import Predictive

predictive = Predictive(model_fn, posterior_samples=samples)
posterior_pred = predictive(rng_key)

# Compare: observed vs replicated variance, means, correlations
# Bayesian p-value: proportion of replicates where test stat > observed
```

**Interpretation:**
- p-value near 0.5: Good fit (replicated data indistinguishable from observed)
- p-value near 0 or 1: Systematic misfit (model fails to capture some aspect)

**If check fails:** Revise model structure, re-fit, and re-check.

### Why Not CFA First?

Traditional SEM uses a two-step approach (Anderson & Gerbing, 1988): validate the measurement model via CFA, then fit the structural model. This is a frequentist computational convenience.

In Bayesian workflow (Gelman et al., 2020; Betancourt, 2018):
- The full generative model (measurement + structural) is specified and fit together
- Prior predictive checks validate the model specification before seeing data
- Posterior predictive checks validate model adequacy after fitting
- There is no separate "measurement validation" step

The question "is this indicator a good proxy?" is answered by the posterior predictive distribution, not by a pre-fitting validation gate.

### References

- Gelman, A., et al. (2020). Bayesian Workflow. arXiv:2011.01808.
- Betancourt, M. (2018). Towards a Principled Bayesian Workflow. https://betanalpha.github.io/
- Gabry, J., et al. (2019). Visualization in Bayesian Workflow. JRSS-A, 182(2), 389-402.
- Stan Development Team. Posterior and Prior Predictive Checks. https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html

---

## References

### Bayesian SEM

- PyMC Development Team. [Confirmatory Factor Analysis and Structural Equation Models in Psychometrics](https://www.pymc.io/projects/examples/en/latest/case_studies/CFA_SEM.html). PyMC Example Gallery. (Reference for CFA/SEM theory; implementation uses NumPyro/JAX.)

### LLM-Assisted Prior Elicitation

- Capstick, A., et al. (2024). AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling. arXiv:2411.17284.
- Huang, Y. (2025). LLM-Prior: A Framework for Knowledge-Driven Prior Elicitation and Aggregation. arXiv:2508.03766.
- Chen, Z., et al. (2025). LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models. arXiv:2508.08300.
- Riegler, M., et al. (2025). Using large language models to suggest informative prior distributions in Bayesian regression analysis. Scientific Reports.
- Selby, J., et al. (2024). Had Enough of Experts? Elicitation and Evaluation of Bayesian Priors from Large Language Models. NeurIPS BDU Workshop.
