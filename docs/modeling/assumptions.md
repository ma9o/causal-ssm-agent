# Modeling Assumptions

This document enumerates the core modeling assumptions underlying the causal-agent framework. Each assumption constrains what can be modeled and has implications for interpretation.

---

## A1. Reflective Measurement Model

**Assumption:** All constructs with indicators use a reflective (effect-indicator) measurement model, not a formative (causal-indicator) model.

**Definition:** In a reflective model, the construct is the common cause of its indicators. Causality flows from construct to indicators:

```
Construct → Indicator₁
          → Indicator₂
          → Indicator₃
```

In a formative model, indicators cause the construct (e.g., SES is "formed" by income, education, occupation). We do not support formative measurement.

**Implications:**
- Indicators of the same construct should be correlated (they share a common cause)
- Removing an indicator does not change the definition of the construct
- The construct exists independently of its specific operationalization
- No causal edges from indicators to their parent construct

**Justification:** Reflective models are standard in psychological and behavioral SEM. They align with classical test theory where observed scores reflect true scores plus error. Formative models require different identification constraints and are conceptually suited to composite indices (HDI, SES) rather than theoretical constructs.

**Reference:** Diamantopoulos, A., & Siguaw, J. A. (2006). Formative versus reflective indicators in organizational measure development. *British Journal of Management*.

---

## A3. Markov Property for Temporal Dynamics

**Assumption:** All endogenous time-varying constructs follow first-order Markov dynamics. The state at t-1 is a sufficient statistic for all prior history.

**Definition:** For any endogenous construct C:
```
P(Cₜ | Cₜ₋₁, Cₜ₋₂, ..., C₁) = P(Cₜ | Cₜ₋₁)
```

**Implications:**
- AR(1) captures the relevant temporal dependence for constructs
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

**Assumption:** Random effects (time-invariant constructs without indicators) capture stable between-person differences. They are exogenous by construction—no modeled causes within the system.

**Implications:**
- Random effects partition variance into between-person and within-person components
- They prevent conflation of between and within effects (Simpson's paradox)
- They cannot be predicted by other variables in the model

**Justification:** In intensive longitudinal data, ignoring between-person heterogeneity biases within-person effect estimates. Random intercepts are the minimal adjustment for this confound.

---

## A6. Measurement Error Handling Depends on Indicator Count

**Assumption:** How measurement error is handled depends on whether a construct has one or multiple indicators.

**Definition:**
- **Multi-indicator constructs (≥2):** Measurement error is separated from construct variance via factor analysis (CFA). The construct is identified through shared variance among indicators.
- **Single-indicator constructs (=1):** Measurement error is absorbed into the structural residual (see A9). No separation is possible.

**Implications:**
- Multi-indicator constructs yield unattenuated coefficient estimates (measurement error partitioned out)
- Single-indicator constructs may have attenuated coefficients (biased toward zero)
- Both are "observed" for the purpose of causal identification—the distinction is about precision, not identifiability

**Justification:** Separating measurement error from true construct variance requires multiple indicators to identify the factor structure. With a single indicator, the two sources of variance are fundamentally confounded without external reliability information.

---

## A7. Measurement Model Identification Enables Causal Identification

**Assumption:** Once the measurement model is identified (via CFA for multi-indicator constructs, or by assumption for single-indicator constructs), constructs can be treated as effectively observed for the purpose of causal identification via the structural model.

See "The Workflow" and "Theoretical Justification" sections below for the full rationale.

---

## A8. Indicator Residuals Are Temporally Independent

**Assumption:** Measurement error in indicators is iid across time. All temporal dependence in observed indicator series is attributed to the construct's dynamics.

**Definition:** For any indicator I of construct C:
```
Iₜ = λ · Cₜ + εₜ
εₜ ~ N(0, σ²), independent across t
```

**Implications:**
- Indicators do not receive AR structure; only constructs do
- Residual autocorrelation in indicators suggests model misspecification
- Possible causes: construct granularity too coarse, missing cross-loadings, systematic measurement dynamics

**Justification:** Separating construct dynamics from indicator dynamics requires strong identification constraints. By attributing all temporal structure to the construct, we maintain a clean separation between "what's happening" (construct dynamics) and "how we see it" (measurement model). This is the default in DSEM implementations.

As Asparouhov, Hamaker & Muthén (2018) note in the foundational DSEM paper:

> "The measurement errors are assumed to be uncorrelated across time... If the residuals are correlated across time, this would indicate that the latent variable does not fully account for the dynamics in the observed variables."

**Relaxation (not currently supported):** AR in indicator residuals is possible but introduces identification challenges. Mplus DSEM allows this via the `RESIDUAL` option. Future versions may support this with appropriate constraints.

**Reference:** Asparouhov, T., Hamaker, E. L., & Muthén, B. (2018). Dynamic structural equation models. *Structural Equation Modeling: A Multidisciplinary Journal*, 25(3), 359-388.

---

## A9. Single-Indicator Constructs Absorb Measurement Error

**Assumption:** When a construct has exactly one indicator, the indicator is treated as identical to the construct. Measurement error is absorbed into the structural residual.

**Definition:** For a single-indicator construct:
```
Constructₜ ≡ Indicatorₜ
```

Conceptually, this collapses:
```
Constructₜ = structural dynamics + structural error
Indicatorₜ = λ · Constructₜ + measurement error
```

Into:
```
Indicatorₜ = structural dynamics + combined error
```

Where λ is fixed to 1 and measurement error merges with structural error.

**Implications:**
- Coefficient estimates may be attenuated (biased toward zero) if measurement error is substantial
- No separation of true construct variance from measurement noise
- This is a pragmatic choice, not an assertion that measurement is perfect

**Justification:** Single-indicator identification of separate measurement and structural variance is impossible without external information (e.g., known reliability coefficients). As Bollen (1989) establishes:

> "With a single indicator, the factor loading and error variance are not identified without additional constraints. The common solution is to fix the loading to unity and the error variance to zero, effectively equating the indicator with the latent variable."

**Recommendation:** When substantively important, prefer multiple indicators per construct to enable measurement error separation. Single-indicator constructs are appropriate for (a) well-validated scales with known high reliability, or (b) exploratory analysis where attenuation bias is acceptable.

**Reference:** Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. (Chapter 7: The Measurement Model)

---

## The Workflow

The framework implements a two-stage workflow grounded in the structural equation modeling (SEM) tradition, following the approach established by **Anderson & Gerbing (1988)**:

1. **Stage 1a (Latent Model):** The orchestrator LLM proposes a theoretical causal DAG over constructs based on domain knowledge alone—no data. This separates theoretical reasoning from operationalization.

2. **Stage 1b (Measurement Model):** Given data, the orchestrator proposes observed indicators for each construct. Indicators follow the reflective measurement model (A1). Constructs may have one indicator (A9 applies) or multiple indicators (CFA identification).

3. **Stage 3 (CFA Validation):** For multi-indicator constructs, confirmatory factor analysis validates that proposed indicators load on their intended constructs and that the measurement model is identified.

4. **Stage 3 (Causal Identification):** DoWhy checks identification of target causal effects on the structural model, treating observed constructs (those with indicators) as identifiable.

This two-step approach—measurement model first, structural model second—is the standard methodology in SEM research. As Anderson & Gerbing (1988) argue, validating the measurement model is a necessary prerequisite to interpreting structural relationships:

> "We present a comprehensive, two-step modeling approach that employs a series of nested models and sequential chi-square difference tests. We discuss the comparative advantages of this approach over a one-step approach."

The recent **Structural After Measurement (SAM)** approach by Rosseel & Loh (2024), implemented in lavaan's `sam()` function, formalizes this workflow with proper two-step standard errors that account for uncertainty from the measurement stage when estimating structural parameters.

---

## Theoretical Justification

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

Under a reflective measurement model, indicators of a latent confounder naturally partition into these categories. The CFA identification conditions (≥3 indicators per construct, or ≥2 with cross-construct correlations) correspond to the completeness conditions ensuring the construct is sufficiently well-measured.

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
- **Indicator AR:** Currently indicator residuals are iid; correlated residuals not supported (see A8)