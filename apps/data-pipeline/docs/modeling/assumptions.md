# Modeling Assumptions

This document enumerates the core modeling assumptions underlying the causal-ssm-agent framework. Each assumption constrains what can be modeled and has implications for interpretation.

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

<!-- A2 is intentionally absent. It was removed during an early revision; numbering is kept stable to avoid breaking cross-references in code and other docs. -->

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

## A3a. Latent Confounders Have Bounded Temporal Reach

**Assumption:** Unobserved constructs follow the same first-order Markov dynamics as observed constructs. Latent confounding has maximum latency of 1 timestep.

**Definition:** A latent confounder U can create confounding via:
- U_t (contemporaneous): U_t → X_t, U_t → Y_t
- U_{t-1} (lagged): U_{t-1} → X_t, U_{t-1} → Y_t

But NOT via U_{t-2} or earlier, because U_{t-1} d-separates U_{t-2} from current effects under the Markov property.

**Implications:**
- A 2-timestep graph segment suffices for identification checking
- Confounding "memory" is bounded by the Markov property
- No need to reason about arbitrarily long confounding paths through time
- The ID algorithm (Shpitser-Pearl) can be applied to a finite unrolled graph

**Justification:** This is the natural extension of A3 to unobserved constructs. If observed constructs are Markov, it's parsimonious to assume latent constructs are too. Without this assumption, identification becomes undecidable—you'd need to consider confounding paths of arbitrary length through time.

**Theoretical foundation:** Jahn, Karnik & Schulman (2025) prove that for periodic causal graphs with width w (variables per timestep) and latency L (max lag), running the ID algorithm on a segment of size O(w × L) suffices to decide identifiability. Under A3 + A3a, L = 1, so a 2-timestep segment suffices.

**Note on latent AR dynamics:** The paper works with ADMGs where latent confounders are represented as bidirected edges, not as explicit nodes with their own temporal dynamics. This is the standard causal inference approach: latent variables are "marginalized out" into bidirected edges that represent the confounding relationship. The internal dynamics of latent confounders (e.g., their AR(1) structure) are irrelevant for identification - what matters is *where* the confounding appears (which observed variables at which timesteps are affected), not how the latent evolves internally. Our implementation follows this convention: AR(1) edges are only added for observed constructs, while unobserved constructs contribute only their confounding edges to the identification graph.

**Reference:** Jahn, E., Karnik, K., & Schulman, L. J. (2025). Causal Identification in Time Series Models. arXiv:2504.20172.

---

## A4. Acyclicity Within Time Slice

**Assumption:** Contemporaneous causal relationships (within the same time index) must form a directed acyclic graph (DAG).

**Definition:** If we consider only edges where lag = 0, the resulting graph must have no cycles.

**Implications:**
- Feedback loops must be modeled via lagged edges (across time)
- Contemporaneous relationships represent instantaneous causation or common response to unmodeled causes
- Standard DAG-based identification algorithms apply within each time slice

**Justification:** Cyclic contemporaneous relationships are not identified without additional constraints (instrumental variables, non-Gaussianity). Requiring acyclicity simplifies identification while allowing feedback dynamics through the temporal structure.

**Identification implication:** When checking causal identifiability, we unroll the temporal graph to 2 timesteps (per A3a) and apply the Shpitser-Pearl ID algorithm to this finite graph. This correctly handles lagged confounding—an unobserved U_{t-1} affecting both X_t and Y_t blocks identification of X_t → Y_t. The unrolled DAG is projected to an ADMG internally for the ID algorithm.

**Reference:** Asparouhov, T., Hamaker, E. L., & Muthén, B. (2018). Dynamic structural equation models. *Structural Equation Modeling*, 25(3), 359-388. https://doi.org/10.1080/10705511.2017.1406803

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

**Assumption:** Once the measurement model is identified (via CFA for multi-indicator constructs, or by assumption for single-indicator constructs), constructs can be treated as effectively observed for the purpose of causal identification via the latent model.

See theory.md §3 "Measurement Model Enables Causal Identification" for the full rationale.

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

**Justification:** Separating construct dynamics from indicator dynamics requires strong identification constraints. By attributing all temporal structure to the construct, we maintain a clean separation between "what's happening" (construct dynamics) and "how we see it" (measurement model). This is the default in dynamic SSM implementations.

As Asparouhov, Hamaker & Muthén (2018) note:

> "The measurement errors are assumed to be uncorrelated across time... If the residuals are correlated across time, this would indicate that the latent variable does not fully account for the dynamics in the observed variables."

**Relaxation (not currently supported):** AR in indicator residuals is possible but introduces identification challenges. Mplus allows this via the `RESIDUAL` option. Future versions may support this with appropriate constraints.

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

## Future Considerations (Not Currently Assumed)

The following are explicitly NOT assumed and may be added in future versions:

- **Non-linear relationships:** Currently all structural effects are linear in parameters
- **Non-Gaussian distributions:** Poisson, Student-t, Gamma, Bernoulli, Negative Binomial, and Beta are supported in the observation model (see `DistributionFamily` in `src/causal_ssm_agent/orchestrator/schemas_model.py` and the emission functions in `src/causal_ssm_agent/models/likelihoods/emissions.py`). Additional families may be added.
- **Time-varying parameters:** Currently all causal coefficients are time-invariant
- **Random slopes:** Currently only random intercepts, not person-specific effect sizes
- **Cross-level interactions:** Currently between-person variables do not moderate within-person effects
- **Formative measurement:** Currently only reflective models supported
- **Indicator AR:** Currently indicator residuals are iid; correlated residuals not supported (see A8)