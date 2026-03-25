# MeasurementModel: Indicators

This page explains how a `MeasurementModel` operationalizes constructs in data: indicator semantics, support windows, aggregation, and `model_clock`. For the emitted Stage 1b contract, see [Stage 1b](../../pipeline/01b-measurement-identifiability.md).

For the construct/indicator ontology—what a construct is, what an indicator is, and how they relate—see [Constructs and Edges: Ontology](../latent-model/constructs-and-edges.md#ontology).

## Indicator Semantics

The schema for `Indicator` is defined in [Stage 1b](../../pipeline/01b-measurement-identifiability.md). Semantically, each indicator answers four questions:

| Question | Field family | Meaning |
|---|---|---|
| What is being measured? | `name`, `construct_name`, `how_to_measure` | The indicator's substantive meaning |
| What kind of value is produced? | `measurement_dtype`, `ordinal_levels` | The support and category semantics of the measurement |
| Where does the value come from? | `source_columns`, `extraction_mode` | Whether extraction is computed directly or requires semantic interpretation |
| Over what support is it defined? | `aggregation`, `observation_window` | How raw observations become one indicator value |

## Extraction Modes

`MeasurementModel` specifies whether an indicator is extracted by direct computation or by semantic interpretation:

- `computed`: the value can be obtained by deterministic transformation or aggregation over raw columns.
- `semantic`: the value requires LLM interpretation over a support window before it becomes a numeric indicator.

Stage 2 executes those modes, but the choice belongs here in the measurement definition.

## Measurement Dtype

`measurement_dtype` is a semantic commitment about the support of the observed variable. It is not yet a full likelihood choice.

| Dtype | Meaning |
|---|---|
| `continuous` | Real-valued measurement |
| `binary` | Two-state measurement |
| `count` | Non-negative event count |
| `ordinal` | Ordered categories |
| `categorical` | Unordered categories |

Stage 4 consumes these dtype semantics when constructing the `ModelSpec`. For the distribution and link-function mapping from dtype, see [Link Functions from Indicator dtype](../model-spec/parameters-likelihoods-and-priors.md#11-link-functions-from-indicator-dtype).

## Observation Windows and Model Clock

An indicator value is defined over an explicit support window. The `MeasurementModel` sets that support semantics before any extraction code runs.

Examples:

- "Average heart rate over the previous day"
- "Number of production incidents during the previous week"
- "Teacher feedback sentiment in the current grading period"

[Stage 2](../../pipeline/02-indicator-extraction.md) later materializes `support_start`, `support_end`, and `anchor_time` for each extracted row, but the window meaning starts here.

`model_clock` is the shared observation-window width that later governs extraction, discretization, and the default lag unit for construct-level edges.

## Aggregation at Indicator Level

Raw data may be finer-grained than an indicator's target granularity. The `MeasurementModel` therefore specifies an aggregation for each indicator, defining how raw observations collapse to the construct's causal timescale.

The current end-to-end measurement stack supports these summary operators:

- `mean`: average level matters
- `sum`: cumulative amount matters
- `count`: event frequency matters
- `last`: the most recent state in the support window matters
- `first`: the earliest state in the support window matters
- `std`: within-window instability matters

These choices are substantive, not just technical. A daily mean mood score and a daily end-of-day mood score encode different theories of what matters.

Other operators such as `min`, `max`, `median`, percentiles, or `trend` may appear in parser utilities or older prose, but they are not currently supported as end-to-end measurement-model operators.

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
