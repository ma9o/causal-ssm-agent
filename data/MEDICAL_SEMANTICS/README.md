# MEDICAL_SEMANTICS Fixture

This fixture exists to lock down Stage 2 observation semantics for a mixed
computed/semantic measurement model.

The checked-in expected full Stage 2 tables live in:

- [`expected-stage2-raw-data.csv`](/Users/ma9o/Desktop/causal-ssm-agent/trees/main/data/MEDICAL_SEMANTICS/expected-stage2-raw-data.csv)
- [`expected-stage2-model-data.csv`](/Users/ma9o/Desktop/causal-ssm-agent/trees/main/data/MEDICAL_SEMANTICS/expected-stage2-model-data.csv)

Use those artifacts for exact row/value regression checks; this README explains
the meaning of the contract.

## Stage 2 Contract

Stage 2 should leave two parquet artifacts in [`run/`](/Users/ma9o/Desktop/causal-ssm-agent/trees/main/data/MEDICAL_SEMANTICS/run):

- `stage2-raw-data.parquet`: canonical long-format observation rows.
- `stage2-model-data.parquet`: the same observation rows after numeric encoding
  for inference.

For this fixture, both artifacts should cover the same 320 observations:

- 16 indicators
- 20 non-empty daily support windows
- exactly one row per `(indicator, support_start)` pair
- null `value` is allowed, but support metadata must still be present

Stage 2 only emits support windows that contain at least one source event. For
this fixture the expected support starts are:

- `2025-03-03T00:00:00`
- `2025-03-04T00:00:00`
- `2025-03-05T00:00:00`
- `2025-03-06T00:00:00`
- `2025-03-07T00:00:00`
- `2025-03-08T00:00:00`
- `2025-03-09T00:00:00`
- `2025-03-10T00:00:00`
- `2025-03-11T00:00:00`
- `2025-03-12T00:00:00`
- `2025-03-13T00:00:00`
- `2025-03-14T00:00:00`
- `2025-03-15T00:00:00`
- `2025-03-16T00:00:00`
- `2025-03-17T00:00:00`
- `2025-03-18T00:00:00`
- `2025-03-19T00:00:00`
- `2025-03-23T00:00:00`
- `2025-03-29T00:00:00`
- `2025-03-31T00:00:00`

All indicators in this fixture use `observation_window=1d`, so:

- `support_end = support_start + 1d`
- `anchor_time = support_end`

That last rule is intentional: every indicator here uses `anchor_policy =
support_end`.

## Raw Artifact

`stage2-raw-data.parquet` should have these columns:

- `indicator`
- `value`
- `anchor_time`
- `support_kind`
- `summary_operator`
- `anchor_policy`
- `observation_window`
- `support_start`
- `support_end`

Expected types:

- `value` is a string or null
- ordinal values should be numeric codes serialized as strings (`0..K-1` in
  `ordinal_levels` order), not label text
- time/support fields are populated for every row
- semantics columns are populated for every row

## Model Artifact

`stage2-model-data.parquet` should preserve the same row coverage and support
metadata, but with inference-ready types:

- `value` is `Float64` or null
- `anchor_time`, `support_start`, and `support_end` are datetime columns

Encoding expectations:

- binary indicators become `0.0` / `1.0`
- ordinal indicators are already integer-coded in the raw artifact using the
  declared `ordinal_levels` order
- categorical indicators are label-encoded when present
- `patient_baseline` is expected to remain null in this fixture because no
  daily window contains an extractable baseline summary

## Observation Semantics

| Indicator | Extraction | Dtype | Aggregation | Support kind | Anchor policy |
| --- | --- | --- | --- | --- | --- |
| `daily_pain` | computed | continuous | `mean` | `interval` | `support_end` |
| `daily_fatigue` | semantic | ordinal | `last` | `point` | `support_end` |
| `missed_doses` | semantic | count | `sum` | `interval` | `support_end` |
| `glucose_std` | computed | continuous | `std` | `interval` | `support_end` |
| `glucose_out_of_range` | semantic | count | `sum` | `interval` | `support_end` |
| `inhaler_usage` | computed | count | `sum` | `interval` | `support_end` |
| `low_spo2` | semantic | binary | `last` | `point` | `support_end` |
| `sleep_duration` | computed | continuous | `sum` | `interval` | `support_end` |
| `sleep_rating` | semantic | ordinal | `last` | `point` | `support_end` |
| `psychological_distress_flags` | semantic | binary | `last` | `point` | `support_end` |
| `hrv_measure` | computed | continuous | `mean` | `interval` | `support_end` |
| `daily_steps` | computed | count | `sum` | `interval` | `support_end` |
| `environmental_conditions` | semantic | binary | `last` | `point` | `support_end` |
| `fever` | semantic | binary | `last` | `point` | `support_end` |
| `infection_mentions` | semantic | binary | `last` | `point` | `support_end` |
| `patient_baseline` | semantic | categorical | `last` | `point` | `support_end` |

The critical semantic distinction is:

- `point` indicators still retain the full daily support window, but the
  measurement is attached to the window end because they use `last`
- `interval` indicators summarize the full daily window and are also anchored at
  the window end

## User-Facing JSON Note

[`run/stage-2.json`](/Users/ma9o/Desktop/causal-ssm-agent/trees/main/data/MEDICAL_SEMANTICS/run/stage-2.json)
contains only a sample of extracted rows (`combined_extractions_sample`). It is
not the full Stage 2 table and is biased toward the first persisted rows, which
for this fixture are mostly computed indicators.

## Validation Failures

The fixture should be treated as broken if any of the following happen:

- missing `anchor_time`, `support_start`, or `support_end`
- duplicate `(indicator, support_start)` pairs
- disagreement between Stage 1b semantics and Stage 2 semantics columns
- raw/model coverage mismatch
- any `observation_window` other than `1d`
