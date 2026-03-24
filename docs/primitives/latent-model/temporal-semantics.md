# LatentModel: Temporal Semantics

This page covers the time semantics owned by the construct-level graph. Observation windows, aggregation, and `model_clock` live in the [MeasurementModel](../measurement-model/windows-and-aggregation.md).

## Temporal Granularity

Constructs have an associated time granularity: `hourly`, `daily`, `weekly`, `monthly`, `yearly`, or `None` for time-invariant constructs.

## Autoregressive Structure

**Endogenous time-varying constructs** receive AR(1). See [A3](assumptions.md).

**Indicators** do not receive AR structure. All temporal dependence in indicator series is attributed to the construct's dynamics. Indicator residuals are assumed iid under [A8](../measurement-model/assumptions.md).

**Exogenous constructs** do not receive AR structure; we condition on their values.

## Same-Timescale Edges

Two valid lag values exist under the Markov property:

- **Lag = 0:** Contemporaneous effect within the same time index
- **Lag = 1 granularity unit:** Lagged effect from `t - 1` to `t`

Higher-order lags (`t - 2`, `t - 3`, and so on) are not permitted. Under Markovian dynamics, `t - 1` is a sufficient statistic for all prior history. Information from `t - 2` is already propagated through the AR(1) path.

## Cross-Timescale Edges

### Contemporaneous cross-timescale edges

**Contemporaneous edges (`lag = 0`) are prohibited.** "Simultaneous" is undefined when constructs operate at different grains.

### Coarser Cause -> Finer Effect

Lag must equal exactly one unit of the coarser construct's granularity.

**Justification (Markov property):** The AR(1) structure on the coarser construct means its value at `t - 1` is a sufficient statistic for prior history. Reaching back further is redundant; that information is already propagated through the coarser construct's own autoregressive path.

**Example:** Weekly stress -> daily mood requires lag = 168 hours (one week). Last week's stress affects this week's daily mood. Stress from two weeks ago affects last week's stress, which affects this week; the effect is mediated, not direct.

### Finer Cause -> Coarser Effect

Lag must equal exactly one unit of the coarser effect granularity. Additionally, later stages must specify how fine-grained observations collapse to the coarser outcome timescale.

**Example:** Hourly steps -> daily mood requires lag = 24 hours (one day). Yesterday's hourly steps, after the Stage 1b aggregation rule is applied, affect today's mood.

## Downstream Consequence

These temporal rules constrain the `LatentModel` itself. Stage 1b then turns them into concrete indicator windows, Stage 4 turns them into parameterized model structure, and the estimation runtime turns elapsed `dt` into continuous-to-discrete transitions.
