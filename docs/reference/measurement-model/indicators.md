# MeasurementModel: Indicators

`MeasurementModel` is the domain primitive that explains how constructs are observed in data. It owns indicator semantics, support windows, aggregation, and `model_clock`. The authoritative schema lives in [Stage 1b](../../pipeline/01b-measurement-identifiability.md).

## Constructs vs Indicators

A construct is the theoretical variable in the causal graph. An indicator is the observed manifestation used to measure that construct in data.

Examples:

- In healthcare, `Medication Adherence` may be measured by pharmacy refill gaps and pill-count compliance.
- In software engineering, `Incident Load` may be measured by alert count and pages acknowledged.
- In education, `Student Engagement` may be measured by attendance, assignment completion, and classroom participation.

## Indicator Semantics

The schema for `Indicator` is defined in [Stage 1b](../../pipeline/01b-measurement-identifiability.md). Semantically, each indicator answers four questions:

| Question | Field family | Meaning |
|---|---|---|
| What is being measured? | `name`, `construct_name`, `how_to_measure` | The indicator's substantive meaning |
| What kind of value is produced? | `measurement_dtype`, `ordinal_levels` | The support and category semantics of the measurement |
| Where does the value come from? | `source_columns`, `extraction_mode` | Whether extraction is computed directly or requires semantic interpretation |
| Over what support is it defined? | `aggregation`, `observation_window` | How raw observations become one indicator value |

## Extraction Modes

`MeasurementModel` owns whether an indicator is extracted by direct computation or by semantic interpretation:

- `computed`: the value can be obtained by deterministic transformation or aggregation over raw columns.
- `semantic`: the value requires LLM interpretation over a support window before it becomes a numeric indicator.

Stage 2 executes those modes, but the choice belongs here in the measurement definition.

## Measurement Dtype

`measurement_dtype` is a semantic commitment about the support of the observed variable. It is not yet a full likelihood choice.

| Dtype | Meaning | Typical downstream consequence |
|---|---|---|
| `continuous` | Real-valued measurement | Gaussian-family default in Stage 4 |
| `binary` | Two-state measurement | Bernoulli-family default |
| `count` | Non-negative event count | Poisson-family default |
| `ordinal` | Ordered categories | Ordered logistic default |
| `categorical` | Unordered categories | Categorical-family default |

Stage 4 consumes these dtype semantics when constructing the `ModelSpec`.

## Observation Windows and Model Clock

An indicator value is defined over an explicit support window. The `MeasurementModel` owns that support semantics before any extraction code runs.

Examples:

- "Average heart rate over the previous day"
- "Number of production incidents during the previous week"
- "Teacher feedback sentiment in the current grading period"

[Stage 2](../../pipeline/02-indicator-extraction.md) later materializes `support_start`, `support_end`, and `anchor_time` for each extracted row, but the window meaning starts here.

`model_clock` is the shared observation-window width that later governs extraction, discretization, and the default lag unit for construct-level edges.

## Aggregation at Indicator Level

Raw data may be finer-grained than an indicator's target granularity. The `MeasurementModel` therefore specifies an aggregation for each indicator, defining how raw observations collapse to the construct's causal timescale.

Different aggregations encode different substantive meanings:

- Mean: average level matters
- Sum: cumulative amount matters
- Max/Min: extremes matter
- Last/First: most recent or earliest state matters
- Variance/Std: instability itself matters
- Median, Skew, Kurtosis, Entropy: distributional shape matters
- Percentiles (p10, p25, p75, p90, p99): tail behavior matters
- Range, IQR, CV: spread relative to level matters
- Instability (MSSD): mean squared successive differences
- Trend: OLS slope over time within the aggregation window

These choices are substantive, not just technical. A daily mean mood score and a daily max mood spike encode different theories of what matters.

## Derived Observation Semantics

The `MeasurementModel` does not store per-row timestamps itself, but it fully determines the row-level support semantics that [Stage 2](../../pipeline/02-indicator-extraction.md) materializes:

- `support_kind`
- `summary_operator`
- `anchor_policy`
- `support_start`
- `support_end`
- `anchor_time`

With the current operator set, `first` anchors at `support_start`. All other supported operators anchor at `support_end`.

## Relationship to Temporal Causation

Edge lag rules are owned by the [`LatentModel`](../latent-model/constructs-and-edges.md#edge-lag-rules). The `MeasurementModel` makes those causal commitments operational by specifying the aggregation and support-window semantics needed to align indicators to the shared `model_clock`.

Examples:

- In healthcare, hourly monitor readings may be aggregated into a daily instability indicator.
- In software engineering, per-incident severity may be aggregated into a weekly service-burden indicator.
- In education, daily homework events may be aggregated into a weekly study-consistency indicator.
