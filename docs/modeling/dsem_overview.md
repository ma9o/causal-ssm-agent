# DSEM Overview

## Variable Taxonomy

Variables are classified along three dimensions:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| Role | Endogenous / Exogenous | Whether variable receives causal edges from system |
| Observability | Observed / Latent | Whether variable is directly measured |
| Temporal status | Time-varying / Time-invariant | Whether variable changes within person over time |

---

## Measurement Philosophy

**Observed** means data exists, not perfect measurement. Outcomes and inputs may be proxies of underlying constructs (e.g., "mood" measured via app ratings, "stress" inferred from heart rate variability). Measurement error is absorbed into residual variance—we model relationships between measured quantities, not "true" latent constructs. This is a pragmatic choice: proxy-based inference is useful even when imperfect.

**Latent** means the construct itself is not directly measured. Two cases exist:

1. **Random effects (Type 4):** No indicators whatsoever. Identified purely from the variance structure of repeated observations. They capture stable between-person heterogeneity that we infer exists but never directly observe.

2. **Latent time-varying constructs (Type 7):** Have observed indicators via a reflective measurement model. The construct is latent but identified through its indicators (see below).

---

## Latent Time-Varying Constructs (Type 7 with Indicators)

Latent time-varying constructs are supported when operationalized through a **reflective measurement model**. This means:

1. The latent construct has one or more observed indicators
2. Causality flows from latent → indicators (not reverse)
3. Indicators are interchangeable manifestations of the underlying construct

**Example:**
```
Latent "stress" (daily) ──→ observed "self_reported_stress"
                        ├──→ observed "heart_rate_variability"
                        └──→ observed "cortisol_level"
```

**Two-Stage Specification:**
- **Stage 1a (Latent Model):** LLM proposes theoretical constructs and causal structure from domain knowledge alone (no data). All constructs are tagged as latent.
- **Stage 1b (Measurement Model):** LLM sees data and operationalizes each latent into observed indicators. All outputs are tagged as observed.

**What this enables:**
- Modeling theoretical constructs that aren't directly measurable
- Multiple imperfect measurements of the same underlying construct
- Separation of measurement error from true construct variance (via factor model)

**What this does NOT enable:**
- State-space models where latent states have no indicators (see scope.md)

---

## Autoregressive Structure

### Rule

All endogenous time-varying variables receive AR(1) at their native timescale by default.

### Justification (Markov Property)

Under the Markov assumption, the state at t-1 is a sufficient statistic for all prior history. Conditioning on Y_{t-1} renders Y_{t-2}, Y_{t-3}, ... conditionally independent of Y_t. Therefore:

- AR(1) captures the relevant temporal dependence
- Higher-order lags add parameters without explanatory benefit under Markovian dynamics
- If residual autocorrelation persists, this suggests missing cross-lags or unmeasured confounders—not higher-order AR

### Cost Asymmetry

Including unnecessary AR(1) wastes one parameter (coefficient ≈ 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default inclusion is the conservative choice.

### Exogenous Variables

No AR structure modeled. We condition on observed values; their temporal structure is irrelevant to the causal model.

---

## Temporal Granularity

Variables have an associated time granularity: `hourly`, `daily`, `weekly`, `monthly`, `yearly`, or `None` (time-invariant).

### Model Clock

The model operates at the finest endogenous outcome granularity. If the finest endogenous variable is daily, the model's time index is daily.

### Aggregation at Dimension Level

Raw data may be finer-grained than the dimension's specified granularity. The orchestrator specifies an aggregation function defining how raw observations collapse to the dimension's timescale. Different aggregations encode different substantive meanings:

- Mean: average level matters
- Sum: cumulative amount matters
- Max/Min: extremes matter
- Last: most recent state matters
- Variance: instability itself matters
- Custom: domain-specific aggregations (rolling means, exponential decay, quantiles, etc.)

---

## Cross-Timescale Rules

### Same-Timescale Edges

Two valid lag values under the Markov property:

- **Lag = 0:** Contemporaneous effect within the same time index
- **Lag = 1 granularity unit:** Lagged effect from t-1 to t

Higher-order lags (t-2, t-3, ...) are not permitted. Under Markovian dynamics, t-1 is a sufficient statistic for all prior history. Information from t-2 is already propagated through the AR(1) path.

### Cross-Timescale Edges

**Contemporaneous edges (lag=0) are prohibited.** "Simultaneous" is undefined when variables operate at different grains.

### Coarser Cause → Finer Effect

Lag must equal exactly one unit of the coarser variable's granularity.

**Justification (Markov property):** The AR(1) structure on the coarser variable means its value at t-1 is a sufficient statistic for prior history. Reaching back further is redundant—that information is already propagated through the coarser variable's own autoregressive path.

**Example:** Weekly stress → daily mood requires lag = 168 hours (one week). Last week's stress affects this week's daily mood. Stress from two weeks ago affects last week's stress, which affects this week—the effect is mediated, not direct.

### Finer Cause → Coarser Effect

Lag must equal exactly one unit of the coarser (effect) variable's granularity. Additionally, an aggregation function specifies how fine-grained observations collapse to the coarser outcome's timescale.

**Example:** Hourly steps → daily mood requires lag = 24 hours (one day). Yesterday's hourly steps (aggregated to a daily value) affect today's mood.

---

## Interpretation Guidance

Effects are estimated as relationships between **observed** quantities. Measurement error is absorbed into residual variance. Interpret:

- AR coefficients as inertia in the measured variable
- Cross-lag coefficients as predictive relationships between measured variables
- Random effects as stable between-person differences in measured baselines

Do not interpret as effects on "true" latent constructs unless measurement properties are well-established and error is known to be small.
