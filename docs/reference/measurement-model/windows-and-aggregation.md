# MeasurementModel: Windows and Aggregation

This page explains how the measurement layer turns messy raw observations into construct-aligned indicator values.

## Observation Windows

An indicator value is defined over an explicit support window. The `MeasurementModel` owns that support semantics before any extraction code runs.

Examples:

- "Average heart rate over the previous day"
- "Number of production incidents during the previous week"
- "Teacher feedback sentiment in the current grading period"

[Stage 2](../../pipeline/02-indicator-extraction.md) later materializes `support_start`, `support_end`, and `anchor_time` for each extracted row, but the window meaning starts here.

## Model Clock

The model operates at the finest endogenous outcome granularity. If the finest endogenous construct is daily, the model's time index is daily.

`model_clock` is therefore the observation-window width later used by extraction and discretization steps as their shared clock.

## Aggregation at Indicator Level

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

These choices are substantive, not just technical. A daily mean mood score and a daily max mood spike encode different theories of what matters.

## Relationship to Temporal Causation

Edge lag rules are owned by the [LatentModel](../latent-model/temporal-semantics.md). The `MeasurementModel` makes those causal commitments operational by specifying the aggregation and observation windows needed to align indicators to the shared `model_clock`.

Examples:

- In healthcare, hourly monitor readings may be aggregated into a daily instability indicator.
- In software engineering, per-incident severity may be aggregated into a weekly service-burden indicator.
- In education, daily homework events may be aggregated into a weekly study-consistency indicator.
