# Data Contract

Stage 0 is intentionally lenient about what you upload. It can ingest a single raw file or a zip bundle of related files, as long as ingestion can normalize them into one typed raw dataframe.

The preferred downstream shape is a timestamped event table. More formally, this is a longitudinal observational dataset. In practice, that means:

- each row is one event, measurement, note, status change, or timepoint
- one primary time column orders the rows
- columns contain raw values or context needed to interpret them later
- multiple source files are fine if they can be joined or concatenated onto the same analysis timeline

This is much better than an aggregate analytics table because the pipeline needs raw observed evidence that can later be interpreted, windowed, and aligned.

Example:

- `vitals.csv`, `med_admin.csv`, and `journal.jsonl` can be bundled into one zip archive and normalized into one raw dataframe keyed by `timestamp`, with sparse columns such as `heart_rate_bpm`, `medication_name`, `dose_mg`, and `note_text`

## What Raw Fields Matter

| Field type | Why it matters | Typical examples |
| --- | --- | --- |
| Primary time column | Required for bucketing events into support windows | `timestamp`, `date`, `created_at` |
| Direct measurement columns | Feed `computed` indicators | `glucose_mg_dl`, `steps`, `spo2_pct` |
| Semantic evidence columns | Feed `semantic` indicators | `note_text`, `search_query`, `event_name` |
| Context and filter columns | Help interpret values, units, and event types | `unit`, `status`, `event_type` |

## Timestamp Contract

- Include at least one parseable date or datetime column.
- Best practice is a single column named `timestamp` using ISO 8601 UTC, for example `2025-03-03T10:00:00Z`.
- Date-only values such as `2025-03-03` are acceptable when daily granularity is enough.
- If multiple datetime-like columns exist, the pipeline prefers common names such as `timestamp`, `time`, `date`, `datetime`, `created_at`, `ts`, `dt`, and `updated_at`.
- Avoid mixed timezones or mixed string formats inside the same column.

The measurement model later chooses a `model_clock` and per-indicator `observation_window`. The pipeline then buckets raw events into support windows and attaches each extracted value to an `anchor_time`.

In current semantics, `first` anchors at `support_start`. All other supported operators anchor at `support_end`.

## How Multiple Sources Are Aligned

- The upload boundary is one ingestible unit. If your raw data comes from several related exports, bundle them into one archive so they can be joined or concatenated into one timestamped table during ingestion.
- Downstream alignment happens on time, not on source-specific row IDs. Indicators are extracted over shared support windows such as daily or monthly windows.
- The canonical extracted row is an observation row with `indicator`, `value`, `anchor_time`, `support_start`, and `support_end`.
- Indicators with different raw cadences can coexist as long as they live on the same primary time axis.
- The extracted timeline is sparse: windows with no source events are omitted, and emitted windows can still contain null values for particular indicators.

Example:

- a daily symptom journal, hourly wearable export, and medication log can coexist in one upload as long as they all describe the same unit on a compatible time axis; Stage 0 normalizes them into one raw dataframe, and later stages align them by support window rather than requiring one source-specific row ID

## Missingness Contract

- `null` means the pipeline did not have enough relevant observed evidence for that indicator in that support window.
- `0` means the relevant source field was observed and the observed evidence implied a zero or negative result.
- A window containing only unrelated events should usually yield `null`, not `0`, for semantic indicators.
- If a computed indicator depends on raw columns that are absent from the dataframe, it cannot be produced deterministically.

This distinction is intentional. The pipeline treats "no evidence" differently from "observed zero."

## `computed` Versus `semantic` Indicators

These are pipeline terms, not just informal descriptions. Stage 1b defines `Indicator.extraction_mode` as `computed` or `semantic`; see [../pipeline/01b-measurement-identifiability.md](../pipeline/01b-measurement-identifiability.md). Stage 2 then defines the corresponding extraction paths; see [../pipeline/02-indicator-extraction.md](../pipeline/02-indicator-extraction.md).

| Mode | Use when | How it works |
| --- | --- | --- |
| `computed` | The raw columns already contain the needed value, or it can be derived deterministically from known raw fields | Direct aggregation or another deterministic formula; no LLM interpretation |
| `semantic` | The value must be inferred from text, mixed event context, or other non-direct evidence inside a support window | An LLM worker reads the window and returns one scalar per indicator |

Both modes produce the same downstream observation-row contract.

Supported aggregation operators are currently `first`, `last`, `sum`, `count`, `mean`, and `std`.

- `first` and `last` create point-style measurements.
- `sum`, `count`, `mean`, and `std` create interval summaries.
- `ordinal` indicators currently support only `first` or `last`.
- `mean` and `std` require `continuous` measurements.
- `count` requires `measurement_dtype="count"`.
- `sum` requires `continuous` or `count` measurements.

## Minimum Viable Dataset

At minimum, a useful upload has:

- one file or zip bundle that Stage 0 can normalize into one raw dataframe
- one parseable primary time column
- at least one raw field for the outcome side of the question
- at least one raw field for a plausible driver, treatment, or exposure
- enough context columns to interpret those fields

One minimal but valid shape looks like this:

```csv
timestamp,event_type,glucose_mg_dl,medication_status,symptom_note
2025-03-03T08:10:00Z,glucose_reading,185,,
2025-03-03T09:00:00Z,medication,,taken,
2025-03-03T21:15:00Z,journal,,,"short of breath and slept badly"
```

Why this is enough:

- every row has a timestamp
- `glucose_mg_dl` can drive computed indicators
- `medication_status` and `symptom_note` can drive LLM-extracted indicators
- sparse rows are acceptable because alignment happens by support window, not by requiring every column on every row

Another valid shape is a zip bundle where one file contains physiological measurements, another contains medication events, and another contains text notes, as long as ingestion can normalize them onto one shared timeline.

What is usually not enough:

- data with no parseable time column
- an aggregate report with totals but no per-period timestamps
- unrelated files that cannot be reconciled onto one analysis timeline during ingestion
