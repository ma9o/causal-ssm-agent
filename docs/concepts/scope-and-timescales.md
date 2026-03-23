# Scope and Timescales

This framework models dynamics of time-varying constructs with optional time-invariant covariates. This is a **causal effect estimation** framework.

Within the pipeline, this document constrains the objects that appear in the [`LatentModel`](../pipeline/01a-latent-model.md#latentmodel), [measurement model](../pipeline/01b-measurement-identifiability.md#measurement-model), and [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec). For the cross-cutting pipeline map, see [pipeline-dimensions.md](pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [artifact-index.md](artifact-index.md).

---

## Ontology

**Constructs** are theoretical entities in the causal model (stress, mood, cognitive load). They live in the latent model.

**Indicators** are observed data (HRV readings, self-report scores, cortisol levels). They live in the measurement model and reflect their parent construct via factor loadings.

---

## Construct Dimensions

Constructs are classified along two dimensions:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| **Role** | Exogenous / Endogenous | Whether construct receives causal edges from other constructs |
| **Temporal** | Time-varying / Time-invariant | Whether construct changes within person over time |

This yields four construct types:

| Role | Temporal | AR Structure | Example |
|------|----------|--------------|---------|
| Exogenous | Time-varying | None (conditioned on) | Weather, day of week |
| Exogenous | Time-invariant | None (conditioned on) | Age, gender, person intercept |
| Endogenous | Time-varying | AR(1) | Mood, stress, sleep quality |
| Endogenous | Time-invariant | None | Baseline severity, stable trait outcome |

Edge restriction: time-invariant constructs may only have time-invariant parents.
A time-varying construct cannot cause a time-invariant construct, because the
child is fixed within person over the modeled window.

---

## Autoregressive Structure

**Endogenous time-varying constructs** receive AR(1). See assumptions.md A3.

**Indicators** do not receive AR structure. All temporal dependence in indicator series is attributed to the construct's dynamics. Indicator residuals are assumed iid (A8).

**Exogenous constructs** do not receive AR structure—we condition on their values.

---

## Identification

Identifiability is checked by y0 in Stage 1b, not enforced at the schema level. See [assumptions.md](assumptions.md) A3a/A7 for the temporal unrolling strategy.

Each treatment-outcome pair is checked independently. Non-identifiability of one effect does not affect identifiability of others, because the ID algorithm (Shpitser & Pearl, 2006) restricts attention to ancestors of the outcome — additions elsewhere cannot introduce new blocking structures.

---

## Temporal Granularity

Constructs have an associated time granularity: `hourly`, `daily`, `weekly`, `monthly`, `yearly`, or `None` (time-invariant).

### Model Clock

The model operates at the finest endogenous outcome granularity. If the finest endogenous construct is daily, the model's time index is daily.

### Aggregation at Indicator Level

Raw data may be finer-grained than the indicator's target granularity. The measurement model specifies an aggregation function for each indicator, defining how raw observations collapse to the construct's causal timescale. Different aggregations encode different substantive meanings:

- Mean: average level matters
- Sum: cumulative amount matters
- Max/Min: extremes matter
- Last/First: most recent or earliest state matters
- Variance/Std: instability itself matters
- Median, Skew, Kurtosis, Entropy: distributional shape matters
- Percentiles (p10, p25, p75, p90, p99): tail behavior matters
- Range, IQR, CV: spread relative to level matters
- Instability (MSSD): mean squared successive differences
- Trend: OLS slope over time within aggregation window

---

## Cross-Timescale Rules

### Same-Timescale Edges

Two valid lag values under the Markov property:

- **Lag = 0:** Contemporaneous effect within the same time index
- **Lag = 1 granularity unit:** Lagged effect from t-1 to t

Higher-order lags (t-2, t-3, ...) are not permitted. Under Markovian dynamics, t-1 is a sufficient statistic for all prior history. Information from t-2 is already propagated through the AR(1) path.

### Cross-Timescale Edges

**Contemporaneous edges (lag=0) are prohibited.** "Simultaneous" is undefined when constructs operate at different grains.

### Coarser Cause → Finer Effect

Lag must equal exactly one unit of the coarser construct's granularity.

**Justification (Markov property):** The AR(1) structure on the coarser construct means its value at t-1 is a sufficient statistic for prior history. Reaching back further is redundant—that information is already propagated through the coarser construct's own autoregressive path.

**Example:** Weekly stress → daily mood requires lag = 168 hours (one week). Last week's stress affects this week's daily mood. Stress from two weeks ago affects last week's stress, which affects this week—the effect is mediated, not direct.

### Finer Cause → Coarser Effect

Lag must equal exactly one unit of the coarser (effect) construct's granularity. Additionally, an aggregation function specifies how fine-grained observations collapse to the coarser outcome's timescale.

**Example:** Hourly steps → daily mood requires lag = 24 hours (one day). Yesterday's hourly steps (aggregated to a daily value) affect today's mood.

---

## Out of Scope

**Trajectory estimation for unmeasured constructs.** Every construct must have at least one indicator. Latent state filtering is used internally for likelihood computation, but the framework's outputs are causal effect estimates, not state trajectories.
