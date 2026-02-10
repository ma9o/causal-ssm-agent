# Theory

This document covers general principles of Bayesian causal inference theory that inform the framework's design. 

---

## 1. Locality of Identification

**Principle:** The identifiability of a causal effect P(y|do(x)) depends only on the graph structure restricted to the ancestors of the outcome. Adding variables elsewhere in the graph cannot break identification of an effect that was already identifiable.

### Formal Statement

Let G be a causal graph. The identifiability of P(y|do(x)) depends only on the induced subgraph G[An(Y)], where An(Y) denotes the ancestors of Y (including Y itself). Specifically:

1. Variables that are not ancestors of Y are irrelevant to identification of P(y|do(x))
2. If P(y|do(x)) is identifiable in G, it remains identifiable in any extension G' ⊇ G that preserves the ancestral structure of Y
3. Non-identifiability of P(w|do(z)) for some other pair (Z,W) does not affect identifiability of P(y|do(x))

### Why This Works

The ID algorithm (Shpitser & Pearl, 2006, 2008) identifies causal effects by:

1. Restricting to ancestors of Y (Line 2 of the algorithm)
2. Decomposing into c-components (maximal confounded subsets)
3. Checking for "hedges" — specific structures that block identification

As Shpitser & Pearl (2008) state:

> "To identify the causal effect on Y, it is sufficient to restrict our attention on the parts of the model ancestral to Y. One intuitive argument for this is that descendants of Y can be viewed as 'noisy versions' of Y and so any information they may impart which may be helpful for identification is already present in Y. On the other hand, variables which are neither ancestors nor descendants of Y lie outside the relevant causal chain entirely, and have no useful information to contribute."

Since the algorithm never examines parts of the graph outside An(Y), additions there cannot introduce new hedges or c-component structures that would block identification.

### Practical Implication

When the framework calls `identify_effect(T, O)` for each treatment-outcome pair, each check is independent. If we check effects {(A→B), (B→C), (A→C)} and find (B→C) non-identifiable but (A→B) and (A→C) identifiable, we can proceed with estimating those two effects. The non-identifiability doesn't "infect" the identifiable queries.

This is why Stage 1b can check identification per-effect rather than requiring the entire model to be "fully identified."

### References

- Shpitser, I., & Pearl, J. (2006). Identification of joint interventional distributions in recursive semi-Markovian causal models. *AAAI*.
- Shpitser, I., & Pearl, J. (2008). Complete identification methods for the causal hierarchy. *JMLR*, 9, 1941-1979.
- Tian, J., & Pearl, J. (2002). A general identification condition for causal effects. *AAAI*.

---

## 2. What Correlated Errors Represent

**Principle:** When latent variable models capture unobserved confounders via error covariances, the covariance parameter represents residual dependence — not an identified causal effect of the confounder.

### The Setup

Consider the canonical unobserved confounding scenario:

```
    U
   ↙ ↘
  X   Y
```

Where U is unobserved. In a linear SEM, we might represent this as:

```
X = βUX · U + εX
Y = βXY · X + βUY · U + εY
```

If we cannot measure U, we observe only:

```
X = εX'
Y = βXY · X + εY'
```

Where εX' and εY' are correlated (Cov(εX', εY') = ψXY ≠ 0).

### The Identification Problem

The covariance parameter ψXY is a function of three quantities:

- βUX (effect of U on X)
- βUY (effect of U on Y)
- Var(U)

From ψXY alone, these three quantities cannot be disentangled. The observed covariance is:

```
ψXY = βUX · βUY · Var(U)
```

This is one equation with three unknowns. You can think causally — "some unmeasured stuff is driving both X and Y" — but you cannot identify the *magnitude* of U's effect on either variable.

### What This Means

| What you CAN do | What you CANNOT do |
|-----------------|-------------------|
| Acknowledge confounding exists | Estimate U's causal effect on X or Y |
| Bound the bias under assumptions | Claim a specific confounder strength |
| Perform sensitivity analysis | Identify the full structural model |
| Identify X→Y via other strategies (IV, front-door) | Decompose variance into "due to U" vs "due to X" |

### When U Becomes Identifiable

The causal effect of (or through) U *can* be identified when U has **proxy variables** — observed indicators that are caused by U. This is exactly the measurement model setup:

```
    U
   ↙↓↘
  I1 I2 I3
```

With sufficient indicators satisfying independence and rank conditions, U's relationship to other variables becomes identified through the shared variance among indicators. This is the bridge between:

- Classical factor analysis (measurement)
- Proximal causal inference (Tchetgen Tchetgen et al., 2020)
- The CFA-then-SEM workflow this framework implements

As Miao, Geng & Tchetgen Tchetgen (2018) show: "with at least two independent proxy variables satisfying a rank condition, causal effects are nonparametrically identified — even when the measurement error mechanism itself is not identified."

### Why IV / Front-Door Work Differently

Instrumental variables and front-door adjustment identify X→Y *despite* U, not by estimating U. They work by finding paths that "route around" the confounding:

**IV:** Find Z such that Z→X and Z⊥Y|U (exclusion). The effect of Z on Y must go through X.

**Front-door:** Find M such that X→M→Y and U doesn't confound X→M or M→Y directly.

In both cases, U gets "marginalized out" of the estimand. We never learn what U's effect is — we learn X's effect on Y by exploiting the graph structure.

### Practical Implication

When the framework encounters an unobserved confounder without indicators:

1. **It does not try to estimate U's effect** — this would be non-identified
2. **It checks if the target effect X→Y is identified** via backdoor, front-door, or IV
3. **If not identified**, it prompts for proxy indicators to try to resolve the confounding
4. **If still non-identifiable**, the effect is flagged in the model output (future: sensitivity analysis could bound the bias)

The correlated-error representation is useful for *acknowledging* and *bounding* confounding, not for *resolving* it without additional structure.

For a worked example demonstrating why marginalization matters for MCMC convergence, see [`notebooks/frontdoor_pymc_demo.ipynb`](../../notebooks/frontdoor_pymc_demo.ipynb).

### References

- D'Amour, A., Ding, P., Feller, A., Lei, L., & Sekhon, J. (2019). On multi-cause causal inference with unobserved confounding. *AISTATS*.
- Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993.
- Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., & Miao, W. (2020). An introduction to proximal causal inference. *arXiv:2009.10982*.
- De Stavola, B. L., Daniel, R. M., Cole, S. R., Hernán, M. A., & Ioannidis, J. P. (2015). Mediation analysis with intermediate confounding. *American Journal of Epidemiology*, 181(1), 64-80.

---

## 3. Measurement Model Identification Enables Causal Identification

**Principle:** Once the measurement model is identified (via CFA for multi-indicator constructs, or by assumption for single-indicator constructs), constructs can be treated as effectively observed for the purpose of causal identification via the structural model.

### The Workflow

The framework implements a two-stage workflow grounded in the structural equation modeling (SEM) tradition, following the approach established by **Anderson & Gerbing (1988)**:

1. **Stage 1a (Latent Model):** The orchestrator LLM proposes a theoretical causal DAG over constructs based on domain knowledge alone—no data. This separates theoretical reasoning from operationalization.

2. **Stage 1b (Measurement Model with Identifiability):** Given data, the orchestrator proposes observed indicators for each construct. Indicators follow the reflective measurement model. Constructs may have one indicator (measurement error absorbed) or multiple indicators (CFA identification). After proposing measurements, y0 checks identification of target causal effects using Pearl's ID algorithm. If effects are non-identifiable, the orchestrator is prompted to propose proxies for blocking confounders.

3. **Stage 2 (Worker Extraction):** Worker LLMs process data chunks in parallel to extract indicator values.

4. **Stage 3 (Extraction Validation):** Validates that worker extraction produced usable data. For multi-indicator constructs, CFA can validate that proposed indicators load on their intended constructs.

This two-step approach—measurement model first, structural model second—is the standard methodology in SEM research. As Anderson & Gerbing (1988) argue, validating the measurement model is a necessary prerequisite to interpreting structural relationships:

> "We present a comprehensive, two-step modeling approach that employs a series of nested models and sequential chi-square difference tests. We discuss the comparative advantages of this approach over a one-step approach."

The recent **Structural After Measurement (SAM)** approach by Rosseel & Loh (2024), implemented in lavaan's `sam()` function, formalizes this workflow with proper two-step standard errors that account for uncertainty from the measurement stage when estimating structural parameters.

### The Core Logic

Under the **pure indicators assumption** (no direct Indicator→Indicator edges; all covariance between indicators flows through constructs), once the measurement model is identified:

- The construct covariance matrix becomes identified from observed indicator covariances
- This construct covariance matrix serves as "data" for the structural model
- Pearl-style identification criteria (backdoor, front-door, do-calculus) apply to the structural DAG

This is the logic underlying all latent variable SEM since LISREL. The framework makes this logic explicit by separating the stages and connecting them to modern causal identification theory.

### The Anderson & Gerbing Two-Step Approach

The canonical reference for this workflow is **Anderson & Gerbing (1988)**, which established that:

1. The measurement model should be validated first via CFA before testing structural hypotheses
2. Poor measurement model fit invalidates any conclusions from the structural model
3. Separating the steps allows diagnosing whether problems stem from measurement or structure

This paper has been cited over 37,000 times and remains the methodological standard in psychology, management, and the social sciences.

### Bridging SEM and Pearl's Causal Framework

**Bollen & Pearl (2013)** establish the bridge between SEM and Pearl's causal framework:

> "SEM is an inference engine that takes in two inputs, qualitative causal assumptions and empirical data, and derives two logical consequences of these inputs: quantitative causal conclusions and statistical measures of fit for the testable implications of the assumptions."

The measurement model provides the mapping from observables to constructs; the structural model encodes causal assumptions among constructs.

**Kuroki & Pearl (2014)** show that causal effects can be recovered from proxy variables of unmeasured confounders under specific graphical conditions. This establishes that indicators of latent variables can serve the same role as proxies in causal identification.

**Miao, Geng & Tchetgen Tchetgen (2018)** generalize this result: with at least two independent proxy variables satisfying a rank condition, causal effects are nonparametrically identified—even when the measurement error mechanism itself is not identified. The rank condition maps directly onto CFA identification conditions.

### Connection to Proximal Causal Inference

The **proximal causal inference (PCI)** framework (Tchetgen Tchetgen et al., 2020) provides the modern synthesis. PCI requires:

1. **Treatment-inducing confounding proxies:** Variables related to treatment only through unmeasured confounders
2. **Outcome-inducing confounding proxies:** Variables related to outcomes only through unmeasured confounders
3. **Completeness conditions:** The proxies must vary sufficiently relative to the confounder's variability

Under a reflective measurement model, indicators of a latent confounder naturally partition into these categories. The ≥3 indicators requirement ensures the measurement model contributes sufficient information to the joint likelihood that posteriors on latent constructs concentrate around data-informed values rather than reflecting prior assumptions (Bollen, 1989; see Levy & Mislevy, 2016 for Bayesian treatment).

As noted in **"Demystifying Proximal Causal Inference" (2024)**:

> "It may be natural to think of U as a latent factor or a set of latent factors that are measured by a collection of indicators, and these indicators could be used to form our sets of proxies."

### References

- Anderson, J. C., & Gerbing, D. W. (1988). Structural equation modeling in practice: A review and recommended two-step approach. *Psychological Bulletin*, 103(3), 411-423.
- Bollen, K. A., & Pearl, J. (2013). Eight myths about causality and structural equation models. In *Handbook of Causal Analysis for Social Research* (pp. 301-328). Springer.
- Kuroki, M., & Pearl, J. (2014). Measurement bias and effect restoration in causal inference. *Biometrika*, 101(2), 423-437.
- Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993.
- Rosseel, Y., & Loh, W. W. (2024). Structural-after-measurement (SAM) approach to structural equation modeling. *Psychological Methods*.
- Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., & Miao, W. (2020). An introduction to proximal causal inference. *arXiv:2009.10982*.

---

## 4. Bayesian Unification of GLM, SEM, and DSEM

**Principle:** GLM/GLMM from biostatistics, SEM from psychometrics, and DSEM for intensive longitudinal data are all special cases of Bayesian hierarchical models. The key insight crystallized in Skrondal & Rabe-Hesketh's 2004 GLLAMM framework: **random effects ARE latent variables**—both are unobserved quantities drawn from population distributions and estimated through identical computational machinery.

### The Generative Modeling Perspective

The Bayesian perspective, articulated most influentially by Andrew Gelman and Richard McElreath, reveals that the GLM/SEM distinction is artificial. Both involve:

- Latent (unobserved) parameters varying across units
- Population distributions describing this variation
- Observation models linking parameters to data
- Prior information constraining estimation

As McElreath emphasizes in *Statistical Rethinking*: the question shifts from "which technique should I use?" to "what is the generative process that created my data?" Once you specify your beliefs about data generation—distributions, dependencies, hierarchical structure—the Bayesian framework handles estimation uniformly regardless of whether the model resembles a traditional mixed-effects regression, factor analysis, or time-series model.

### The Mathematical Equivalences

| Traditional Framing | Bayesian Hierarchical Equivalent |
|---------------------|----------------------------------|
| Confirmatory factor analysis | Hierarchical model where indicators depend on latent factors |
| Random intercepts model | Hierarchical model where observations depend on group-specific intercepts |
| Latent growth model | Simultaneously a repeated-measures multilevel model and an SEM |
| Random slopes | Latent factors with covariate-dependent loadings |

In Bayesian computation, both random effects and latent variables are treated identically—as parameters sampled from conditional posteriors.

### Priors as Soft Structural Constraints

Traditional SEM fixes a factor loading or latent variance to set scale. Bayesian priors provide "soft" constraints:

- **Informative priors** encode structural assumptions
- **Shrinkage priors** (horseshoe, LASSO-type) regularize while allowing exploration
- **Small-variance priors** allow cross-loadings while constraining toward zero

Muthén & Asparouhov (2012) proposed "replacing parameter specifications of exact zeros with approximate zeros based on informative, small-variance priors"—a more theoretically honest approach than assuming exact zeros.

### Non-Centered Parameterization

For hierarchical models with varying effects, non-centered parameterization is critical. Instead of sampling θ ~ Normal(μ_θ, σ_θ) directly (which creates funnel geometries that challenge MCMC), sample θ_raw ~ Normal(0, 1) and transform: θ = μ_θ + σ_θ × θ_raw. This dramatically improves sampling efficiency and is essential for DSEM models with many person-specific random effects.

### DSEM-Specific Prior Recommendations

For temporal parameters in DSEM, the methodological literature suggests:

| Parameter | Recommended Prior | Rationale |
|-----------|-------------------|-----------|
| AR coefficients | Normal(0, 0.5) | Encourages stationarity without hard constraint |
| Cross-lagged effects | Normal(0, 0.3-0.5) | Cross-lagged effects rarely exceed ±0.5 |
| Random effect SDs | Half-Cauchy(0, 2.5) | Weakly informative, critical for small N |
| Correlations | LKJ(η=2) | Slight shrinkage toward zero |

Stan documentation advises against hard stationarity constraints: "If the data are not well fit by a stationary model it is best to know this." Weakly informative priors that encourage but don't enforce stationarity provide better diagnostics.

### Sample Size Requirements

From the methodological literature on Bayesian DSEM:

- **N > 100 persons, T > 50 timepoints:** Excellent convergence and unbiased parameter recovery
- **N = 50-100, T = 30-50:** Works with slight variance bias
- **N < 50:** Requires informative priors for stable estimation

Bayesian DSEM's key advantage over frequentist alternatives: MCMC handles many random effects where maximum likelihood becomes intractable.

### Connection to This Framework

The framework implements this unified perspective:

1. **DAG specification layer:** LLM proposes construct-level causal structure from domain knowledge
2. **Identification layer:** y0 applies do-calculus to derive estimands
3. **Estimation layer:** NumPyro/JAX builds Bayesian hierarchical state-space models with appropriate priors
4. **Uncertainty communication:** Full posterior distributions, not point estimates

The theoretical unification—random effects as latent variables, both subsumed by Bayesian hierarchical models—has practical consequences: you're no longer choosing between techniques but specifying generative processes. The question shifts from "should I use mixed models or SEM?" to "what distributional assumptions and dependency structure match my understanding of how these data arose?"

### References

- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
- Muthén, B. (2002). Beyond SEM: General latent variable modeling. *Behaviormetrika*, 29(1), 81-117.
- Muthén, B., & Asparouhov, T. (2012). Bayesian structural equation modeling: A more flexible representation of substantive theory. *Psychological Methods*, 17(3), 313-335.
- Skrondal, A., & Rabe-Hesketh, S. (2004). *Generalized Latent Variable Modeling*. Chapman & Hall/CRC.
- Stan Development Team. (2024). Stan User's Guide: Time-Series Models.

