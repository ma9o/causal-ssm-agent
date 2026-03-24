# LatentModel: Temporal Semantics

This page covers the time semantics owned by the construct-level graph. Observation windows, aggregation, and `model_clock` live in the [MeasurementModel](../measurement-model/windows-and-aggregation.md).

## Temporal Granularity

All time-varying constructs currently share a single timescale determined by the `model_clock` on the [MeasurementModel](../measurement-model/windows-and-aggregation.md). Time-invariant constructs have no temporal granularity.

## Autoregressive Structure

**Endogenous time-varying constructs** receive AR(1). See [A3](assumptions.md).

**Indicators** do not receive AR structure. All temporal dependence in indicator series is attributed to the construct's dynamics. Indicator residuals are assumed iid under [A8](../measurement-model/assumptions.md).

**Exogenous constructs** do not receive AR structure; we condition on their values.

## Edge Lag Rules

Two valid lag values exist under the [Markov property](assumptions.md):

- **Lag = 0:** Contemporaneous effect within the same time index. Under [A4b](assumptions.md#a4b-endogenous-time-varying-directed-effects-are-drift-mediated), this is not a valid encoding for edges between two endogenous time-varying constructs.
- **Lag = 1 model-clock tick:** Lagged effect from `t - 1` to `t`

Higher-order lags (`t - 2`, `t - 3`, and so on) are not permitted. Under Markovian dynamics, `t - 1` is a sufficient statistic for all prior history. Information from `t - 2` is already propagated through the AR(1) path.

## Downstream Consequence

These temporal rules constrain the `LatentModel` itself. Stage 1b then turns them into concrete indicator windows, Stage 4 turns them into parameterized model structure, and the estimation runtime turns elapsed `dt` into continuous-to-discrete transitions.
