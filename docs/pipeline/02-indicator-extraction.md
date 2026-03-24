# Stage 2: Indicator Extraction

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| llm+grounding | No | No | [Observation rows](#observation-row) and [model-ready data](#model-ready-data) |

Extracts numeric indicator values from raw data by routing each indicator through either a deterministic Polars aggregation or a parallel LLM worker, then annotates the merged output into canonical [observation rows](#observation-row) with explicit [support-window semantics](../reference/measurement-model/windows-and-aggregation.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question—provides temporal and semantic context for LLM workers |
| `stage0.result` | [Stage 0](00-ingestion.md) | Ingested dataframe (wide-format parquet) plus column descriptions |
| `stage1b.result` | [Stage 1b](01b-measurement-identifiability.md) | [`CausalSpec`](01b-measurement-identifiability.md#causalspec) with indicators and extraction modes |

Stage 1b specified *what* to measure and *how*; Stage 2 carries out those instructions against the raw data. This is the first point where indicator definitions are evaluated over actual values.

## Process

Stage 2 splits indicators into two extraction paths based on each indicator's `extraction_mode`, runs them concurrently, then merges and annotates the results into a canonical observation-row table.

**Indicator routing.** Each indicator defined in the [`CausalSpec`](01b-measurement-identifiability.md#causalspec) carries an `extraction_mode`—either `"computed"` (a deterministic aggregation that Polars can evaluate mechanically) or `"semantic"` (requiring an LLM to interpret unstructured text). The indicator list is split by mode and both paths run in parallel:

- **Computed path.** Indicators with `extraction_mode="computed"` are aggregated directly via Polars expressions. For each computed indicator, Polars truncates the raw time column by the indicator's effective [observation window](../reference/measurement-model/windows-and-aggregation.md) (explicit if set, otherwise the `model_clock`), groups by the resulting tick boundary, and applies the indicator's aggregation function. Computed rules—multi-column expressions specified as an AST—are compiled into Polars expressions and evaluated within the same window groups. This path produces long-format rows of `(indicator, value, timestamp)` in ~50 ms.

- **Semantic path.** Indicators with `extraction_mode="semantic"` require LLM interpretation. Extraction chunks are prepared deterministically: semantic indicators are grouped by their effective observation window, the raw DataFrame is projected to only the `source_columns` referenced by those indicators, [bucketed](../reference/measurement-model/windows-and-aggregation.md) into support windows via clock truncation, chunked into batches, and formatted as LLM-readable markdown showing timestamped events within each window. Events are truncated per window when they exceed a configurable cap (preserving the first and last events with uniform sampling in between).
Each chunk is dispatched to a parallel LLM worker. The worker receives the formatted window text, the research question, and the indicator definitions (name, dtype, summary operator, support kind, window, and `how_to_measure` instructions). It reads the events, interprets them against each indicator's `how_to_measure` instructions, and submits its extractions via a `validate_extractions` tool call. The validation tool checks:
    - *Indicator names* exist in the `CausalSpec`
    - *Support-window starts* match the expected boundaries for this chunk
    - *Dtype conformance*: extracted values match the indicator's `measurement_dtype` (continuous, binary, count, ordinal, categorical)
    - *No duplicate `(window_start, indicator)` pairs* within the chunk
    - *Ordinal bounds*: ordinal codes fall within `0..len(ordinal_levels) − 1`

**Annotation.** Both paths emit raw `(indicator, value, timestamp)` tuples where `timestamp` is the support-window start. The annotation step joins these rows with indicator metadata from the `CausalSpec` to derive the canonical [observation-row](#observation-row) fields: [`support_kind`](../reference/measurement-model/windows-and-aggregation.md) (point or interval, determined by the aggregation and `measurement_dtype`), `summary_operator` (the aggregation function name), `anchor_policy` (`support_start` for `first`, `support_end` for all others), `observation_window`, and the realized `support_start` / `support_end` / `anchor_time` timestamps. These fields are not free parameters—they are derived deterministically from the measurement model.

**Materialization.** The annotated observation rows are encoded in place: non-continuous types are cast to Float64, ISO strings are parsed to native datetimes, rows with null `anchor_time` are dropped, and the result is sorted by `(indicator, anchor_time)`. The single [model-ready table](#model-ready-data) is persisted as `stage2-model-data.parquet`.

## Outputs

| Output | Type | Description |
|---|---|---|
| `data_for_model` | [Model-ready data](#model-ready-data) | Numerically encoded observation table for downstream fitting |

The public stage payload exposes per-worker execution summaries (`workers`: status, extraction count, window count, and error if any) and may include `llm_trace` as runtime provenance for the UI. The data table is persisted as a parquet sidecar file rather than serialized into the web payload. The stage outcome is `"success"` if at least one observation row was extracted.

## Definitions

### Observation Row

An observation row is the canonical extracted indicator datum. It owns:

| Field | Type | Description |
|---|---|---|
| `indicator` | string | Indicator name, referencing the [measurement model](01b-measurement-identifiability.md#measurement-model) |
| `value` | Float64 | Extracted value (numerically encoded; non-continuous types label-encoded) |
| `anchor_time` | ISO datetime | Latent-grid attachment time—the timestamp downstream models use for this observation |
| `support_kind` | `"point"` \| `"interval"` | Whether the measurement is point-local (`first`/`last`) or an interval summary (`sum`/`count`/`mean`/`std`) |
| `summary_operator` | string | The aggregation applied within the support window |
| `anchor_policy` | `"support_start"` \| `"support_end"` | Which support boundary `anchor_time` corresponds to |
| `observation_window` | duration string | The window width (e.g. `"1d"`, `"1w"`) over which the value was measured or aggregated |
| `support_start` | ISO datetime | Start of the realized support window |
| `support_end` | ISO datetime | End of the realized support window (`support_start` + `observation_window`) |

`support_kind`, `summary_operator`, and `anchor_policy` are derived deterministically from each indicator's `aggregation` and `measurement_dtype` via the [observation semantics](../reference/measurement-model/windows-and-aggregation.md). They are not free parameters—the measurement model fully determines them.

### Model-Ready Data

Model-ready data is the numerically encoded table derived from observation rows for downstream fitting backends. It shares the observation-row schema but with `value` cast to Float64 and non-continuous dtypes encoded:

- **binary**: `true`/`yes`/`1` → 1.0, `false`/`no`/`0` → 0.0
- **ordinal**: label-encoded by the indicator's `ordinal_levels` order
- **categorical**: integer label-encoded (sorted categories)
- **continuous** / **count**: no-op (already numeric)

Timestamps are parsed to native datetime objects. Rows with null `anchor_time` are dropped. The table is sorted by `(indicator, anchor_time)`.

This is the fitting contract used by [Stage 4](04-model-specification-priors.md) onward—it is not just the same rows in another file.

Example: for a study of classroom interventions and student learning where Stage 1b defined computed indicators like "mean daily attendance rate" (`computed`, continuous, `mean`, `1w`) and semantic indicators like "teacher-reported engagement level" (`semantic`, ordinal, `last`, `1w`), Stage 2 would aggregate attendance records via Polars into weekly means while dispatching weekly narrative logs to LLM workers that extract ordinal engagement codes. Both paths produce observation rows anchored at week boundaries with explicit support windows.
