# Modeling Assumptions

This document enumerates the core modeling assumptions underlying the causal-agent framework. Each assumption constrains what can be modeled and has implications for interpretation.

---

## A1. Reflective Measurement Model

**Assumption:** All latent constructs use a reflective (effect-indicator) measurement model, not a formative (causal-indicator) model.

**Definition:** In a reflective model, the latent variable is the common cause of its indicators. Causality flows from construct to indicators:

```
Latent Construct → Indicator₁
                 → Indicator₂
                 → Indicator₃
```

In a formative model, indicators cause the construct (e.g., SES is "formed" by income, education, occupation). We do not support formative measurement.

**Implications:**
- Indicators of the same latent construct should be correlated (they share a common cause)
- Removing an indicator does not change the definition of the construct
- The latent construct exists independently of its specific operationalization
- No causal edges from indicators to their latent construct

**Justification:** Reflective models are standard in psychological and behavioral SEM. They align with classical test theory where observed scores reflect true scores plus error. Formative models require different identification constraints and are conceptually suited to composite indices (HDI, SES) rather than theoretical constructs.

**Reference:** Diamantopoulos, A., & Siguaw, J. A. (2006). Formative versus reflective indicators in organizational measure development. *British Journal of Management*.

---

## A2. Latent Time-Varying Constructs Require Indicators

**Assumption:** Latent time-varying constructs must have at least one observed indicator. Pure state-space models (latent dynamics with no indicators) are not supported.

**Definition:**
- **Supported:** Latent "stress" (time-varying) with indicators [HRV, cortisol, self-report]
- **Not supported:** Latent "true_mood" (time-varying) with no indicators, identified only via temporal dynamics

**Implications:**
- Every latent construct in the structural model must be operationalized into observables
- Identification comes from the measurement model (factor loadings), not from dynamics alone
- We avoid the process-vs-measurement variance disentanglement problem inherent in state-space models

**Justification:** State-space models require specifying priors on initial state distribution, process variance, and measurement variance. These interact non-trivially and are difficult to disentangle without strong domain knowledge. By requiring indicators, we use the well-understood factor-analytic identification strategy instead.

---

## A3. Markov Property for Temporal Dynamics

**Assumption:** All endogenous time-varying constructs follow first-order Markov dynamics. The state at t-1 is a sufficient statistic for all prior history.

**Definition:** For any endogenous variable Y:
```
P(Yₜ | Yₜ₋₁, Yₜ₋₂, ..., Y₁) = P(Yₜ | Yₜ₋₁)
```

**Implications:**
- AR(1) captures the relevant temporal dependence
- Higher-order lags (AR(2), AR(3), etc.) are not modeled
- Cross-lagged effects use lag-1 at the native granularity
- Residual autocorrelation suggests missing cross-lags or unmeasured confounders, not higher-order AR

**Justification:** The Markov property is a parsimony constraint with asymmetric costs. Including unnecessary AR(1) wastes one parameter (coefficient ≈ 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default AR(1) is the conservative choice.

---

## A4. Acyclicity Within Time Slice

**Assumption:** Contemporaneous causal relationships (within the same time index) must form a directed acyclic graph (DAG).

**Definition:** If we consider only edges where lag = 0, the resulting graph must have no cycles.

**Implications:**
- Feedback loops must be modeled via lagged edges (across time)
- Contemporaneous relationships represent instantaneous causation or common response to unmodeled causes
- Standard DAG-based identification algorithms apply within each time slice

**Justification:** Cyclic contemporaneous relationships are not identified without additional constraints (instrumental variables, non-Gaussianity). Requiring acyclicity simplifies identification while allowing feedback dynamics through the temporal structure.

---

## A5. Random Effects as Exogenous Between-Person Heterogeneity

**Assumption:** Random effects (latent time-invariant variables) capture stable between-person differences. They are exogenous by construction—no modeled causes within the system.

**Implications:**
- Random effects partition variance into between-person and within-person components
- They prevent conflation of between and within effects (Simpson's paradox)
- They cannot be predicted by other variables in the model

**Justification:** In intensive longitudinal data, ignoring between-person heterogeneity biases within-person effect estimates. Random intercepts are the minimal adjustment for this confound.

---

## A6. Measurement Error Absorbed into Residuals (for Observed Variables)

**Assumption:** For variables treated as observed (not latent), we do not explicitly model measurement error. It is absorbed into residual variance.

**Definition:** For any observed variable X:
```
X_observed = X_true + ε  (where ε is absorbed into model residual)
```

**Implications:**
- Estimated coefficients may be attenuated (biased toward zero) if measurement error is substantial
- "Observed" means data exists, not perfect measurement
- Results should be interpreted as relationships between measured quantities

**Note:** This assumption applies to the final observed indicators. Latent constructs (when used) DO separate measurement error from construct variance via the factor model—that's the point of A1 and A2.

**Justification:** Explicit measurement error modeling for every observed variable requires known reliability coefficients. The framework prioritizes accessibility: users can model relationships with available data and acknowledge measurement limitations in interpretation.

---

# A7. Measurement Model Identification Enables Causal Identification

**Assumption:** Once the measurement model is identified (via CFA), latent constructs can be treated as effectively observed for the purpose of causal identification via the structural model.

---

## The Workflow

The framework implements a two-stage workflow grounded in the structural equation modeling (SEM) tradition, following the approach established by **Anderson & Gerbing (1988)**:

1. **Stage 1a (Latent Causal Structure):** The orchestrator LLM proposes a theoretical causal DAG over latent constructs based on domain knowledge alone—no data. This separates theoretical reasoning from operationalization.

2. **Stage 1b (Measurement Model):** Given data, the orchestrator proposes observed indicators for each latent construct. Each latent must have ≥1 indicator satisfying the reflective measurement model (A1).

3. **Stage 3 (CFA Validation):** Confirmatory factor analysis validates that proposed indicators load on their intended latents and that the measurement model is identified.

4. **Stage 3 (Causal Identification):** DoWhy checks identification of target causal effects on the latent DAG, treating latents with validated measurement models as effectively observed.

This two-step approach—measurement model first, structural model second—is the standard methodology in SEM research. As Anderson & Gerbing (1988) argue, validating the measurement model is a necessary prerequisite to interpreting structural relationships:

> "We present a comprehensive, two-step modeling approach that employs a series of nested models and sequential chi-square difference tests. We discuss the comparative advantages of this approach over a one-step approach."

The recent **Structural After Measurement (SAM)** approach by Rosseel & Loh (2024), implemented in lavaan's `sam()` function, formalizes this workflow with proper two-step standard errors that account for uncertainty from the measurement stage when estimating structural parameters.

---

## Theoretical Justification

### The Core Logic

Under the **pure indicators assumption** (no direct M→M edges; all covariance between indicators flows through latents), once the measurement model is identified:

- The latent covariance matrix becomes identified from observed indicator covariances
- This latent covariance matrix serves as "data" for the structural model
- Pearl-style identification criteria (backdoor, front-door, do-calculus) apply to the latent DAG

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

The measurement model provides the mapping from observables to latents; the structural model encodes causal assumptions among latents.

**Kuroki & Pearl (2014)** show that causal effects can be recovered from proxy variables of unmeasured confounders under specific graphical conditions. This establishes that indicators of latent variables can serve the same role as proxies in causal identification.

**Miao, Geng & Tchetgen Tchetgen (2018)** generalize this result: with at least two independent proxy variables satisfying a rank condition, causal effects are nonparametrically identified—even when the measurement error mechanism itself is not identified. The rank condition maps directly onto CFA identification conditions.

### Connection to Proximal Causal Inference

The **proximal causal inference (PCI)** framework (Tchetgen Tchetgen et al., 2020) provides the modern synthesis. PCI requires:

1. **Treatment-inducing confounding proxies:** Variables related to treatment only through unmeasured confounders
2. **Outcome-inducing confounding proxies:** Variables related to outcomes only through unmeasured confounders
3. **Completeness conditions:** The proxies must vary sufficiently relative to the confounder's variability

Under a reflective measurement model, indicators of a latent confounder naturally partition into these categories. The CFA identification conditions (≥3 indicators per latent, or ≥2 with cross-latent correlations) correspond to the completeness conditions ensuring the latent is sufficiently well-measured.

As noted in **"Demystifying Proximal Causal Inference" (2024)**:

> "It may be natural to think of U as a latent factor or a set of latent factors that are measured by a collection of indicators, and these indicators could be used to form our sets of proxies."

---
## Future Considerations (Not Currently Assumed)

The following are explicitly NOT assumed and may be added in future versions:

- **Non-linear relationships:** Currently all structural effects are linear in parameters
- **Non-Gaussian distributions:** Currently residuals are assumed Gaussian
- **Time-varying parameters:** Currently all causal coefficients are time-invariant
- **Random slopes:** Currently only random intercepts, not person-specific effect sizes
- **Cross-level interactions:** Currently between-person variables do not moderate within-person effects
- **Formative measurement:** Currently only reflective models supported
