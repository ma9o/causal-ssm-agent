# Modeling Assumptions

This document enumerates the core modeling assumptions underlying the causal-ssm-agent framework. Each assumption constrains what can be modeled and has implications for interpretation.

These assumptions cut across the full pipeline: they shape the [`LatentModel`](../pipeline/01a-latent-model.md#latentmodel), [measurement model](../pipeline/01b-measurement-identifiability.md#measurement-model), [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec), functional specification, and estimation runtime. For the cross-cutting pipeline map, see [pipeline-dimensions.md](pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [artifact-index.md](artifact-index.md).

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

**Note on latent AR dynamics:** The identification literature is often expressed in ADMG terms, where latent confounders appear as bidirected edges after projection. That is an internal representation only. Our user-facing contract stays as a DAG with explicit latent confounder nodes. For identification, we unroll that DAG to two timesteps and then project internally to an ADMG for y0's ID algorithm. The internal dynamics of latent confounders are still not the key object for identification; what matters is which observed variables they jointly affect at which timesteps.

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

**References:**
- Asparouhov, T., Hamaker, E. L., & Muthén, B. (2018). Dynamic structural equation models. *Structural Equation Modeling*, 25(3), 359-388. https://doi.org/10.1080/10705511.2017.1406803
- Shpitser, I., & Pearl, J. (2006). *Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models.* AAAI.

---

## A5. Time-Invariant Latents as Subject-Level Static States

**Assumption:** Time-invariant constructs capture stable subject-level differences over the modeled window. They may be exogenous or endogenous, but any modeled causes of a time-invariant construct must themselves be time-invariant.

**Definition:** A time-invariant latent is implemented as a quasi-constant state: its drift diagonal is set to ≈0 (−1e−6) and its diffusion to ≈0, so η_i(t) ≈ η_i(0) throughout the time series.

**Implications:**
- Time-invariant latents act as subject-specific static states, absorbing stable baseline differences
- They may serve as stable covariates or as static outcomes/traits explained by other stable constructs
- They cannot have time-varying parents, because a within-person changing cause cannot determine a within-person fixed child
- If a time-invariant construct has parents, those parents must also be time-invariant
- They affect time-varying constructs at every timestep (see A3a unrolling)

**Note on hierarchical modeling:** The current implementation fits each subject independently—there is no cross-subject shrinkage or hierarchical prior on these intercepts. True random effects (in the SEM/multilevel sense) would require a hierarchical model where subject-level parameters are drawn from a population distribution. This is not currently supported.

**Justification:** In intensive longitudinal data, ignoring stable individual differences biases within-person effect estimates. Static subject-level states are the minimal adjustment for this confound. Allowing time-invariant constructs to depend on other time-invariant constructs still preserves that interpretation while ruling out incoherent within-person arrows into fixed variables.

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

**Rationale:** Under the pure indicators assumption (no direct Indicator→Indicator edges), the construct covariance matrix becomes identified from observed indicator covariances via CFA. This matrix then serves as "data" for the latent model, and Pearl-style identification criteria apply to the construct-level DAG. This is the logic underlying all latent variable SEM since LISREL — the framework makes it explicit by separating the stages.

**References:**
- Anderson, J. C., & Gerbing, D. W. (1988). Structural equation modeling in practice: A review and recommended two-step approach. *Psychological Bulletin*, 103(3), 411-423.
- Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993.

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
- **General non-Gaussian latent dynamics:** Student-t process noise is supported (via the particle filter backend with `diffusion_dist="student_t"`), but more general non-Gaussian dynamics (e.g., jump-diffusion, switching regimes) are not
- **Time-varying parameters:** Currently all causal coefficients are time-invariant
- **Random slopes:** Currently only random intercepts, not person-specific effect sizes
- **Cross-level interactions:** Currently between-person variables do not moderate within-person effects
- **Formative measurement:** Currently only reflective models supported
- **Indicator AR:** Currently indicator residuals are iid; correlated residuals not supported (see A8)
