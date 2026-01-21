# DSEM Overview

## Construct Taxonomy

Constructs are classified along two dimensions:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| Role | Endogenous / Exogenous | Whether construct receives causal edges from other constructs |
| Temporal status | Time-varying / Time-invariant | Whether construct changes within person over time |

This yields four construct types:

| Role | Temporal | AR Structure | Example |
|------|----------|--------------|---------|
| Exogenous | Time-varying | None (conditioned on) | Weather, day of week |
| Exogenous | Time-invariant | None (conditioned on) | Age, gender, person intercept |
| Endogenous | Time-varying | AR(1) | Mood, stress, sleep quality |
| Endogenous | Time-invariant | None | Single-occasion outcome |

---

## Constructs and Indicators

**Constructs** are theoretical entities in the causal model (stress, mood, cognitive load). They live in the latent model and represent what we're reasoning about causally.

**Indicators** are observed measurements (HRV readings, self-report scores, cortisol levels). They live in the measurement model and reflect their parent construct.

**Key insight:** Whether a construct has indicators is not a property of the construct itself. It's determined by the measurement model:
- A construct with indicators can be identified through those measurements
- A construct without indicators may still be valid in the DAG, but whether the target causal effect is identifiable depends on the graph structure

Identification is checked by DoWhy in Stage 3, not enforced at the schema level.

---

## Two-Stage Specification

**Stage 1a (Latent Model):** The LLM proposes theoretical constructs and causal structure from domain knowledge alone (no data). This defines what constructs exist and how they relate causally.

**Stage 1b (Measurement Model):** The LLM sees data and proposes indicators for each construct. Each indicator specifies:
- `how_to_measure`: extraction instructions
- `measurement_dtype`: continuous, binary, count, ordinal, categorical
- `measurement_granularity`: resolution of raw measurements
- `aggregation`: how to collapse to the construct's causal timescale

A construct may have zero, one, or multiple indicators. Multiple indicators enable measurement error separation via factor models.

---

## Autoregressive Structure

### Rule

All endogenous time-varying constructs receive AR(1) at their native timescale.

### Justification (Markov Property)

Under the Markov assumption, the state at t-1 is a sufficient statistic for all prior history. Conditioning on Y_{t-1} renders Y_{t-2}, Y_{t-3}, ... conditionally independent of Y_t. Therefore:

- AR(1) captures the relevant temporal dependence
- Higher-order lags add parameters without explanatory benefit under Markovian dynamics
- If residual autocorrelation persists, this suggests missing cross-lags or unmeasured confounders—not higher-order AR

### Cost Asymmetry

Including unnecessary AR(1) wastes one parameter (coefficient ≈ 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default inclusion is the conservative choice.

### Exogenous Constructs

No AR structure modeled. We condition on observed values; their temporal structure is irrelevant to the causal model.

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

**Contemporaneous edges (lag=0) are prohibited.** "Simultaneous" is undefined when constructs operate at different grains.

### Coarser Cause → Finer Effect

Lag must equal exactly one unit of the coarser construct's granularity.

**Justification (Markov property):** The AR(1) structure on the coarser construct means its value at t-1 is a sufficient statistic for prior history. Reaching back further is redundant—that information is already propagated through the coarser construct's own autoregressive path.

**Example:** Weekly stress → daily mood requires lag = 168 hours (one week). Last week's stress affects this week's daily mood. Stress from two weeks ago affects last week's stress, which affects this week—the effect is mediated, not direct.

### Finer Cause → Coarser Effect

Lag must equal exactly one unit of the coarser (effect) construct's granularity. Additionally, an aggregation function specifies how fine-grained observations collapse to the coarser outcome's timescale.

**Example:** Hourly steps → daily mood requires lag = 24 hours (one day). Yesterday's hourly steps (aggregated to a daily value) affect today's mood.

---

## Interpretation Guidance

Effects are estimated as relationships between constructs as measured through their indicators. Measurement error in indicators is absorbed into residual variance. Interpret:

- AR coefficients as inertia in the construct
- Cross-lag coefficients as causal relationships between constructs
- Random effects as stable between-person differences in baselines

Causal interpretation requires that the DAG correctly captures the true causal structure and that all relevant confounders are included.
