# Stage 2: Indicator Extraction

Extracts numeric indicator values from raw data using direct Polars aggregation or parallel LLM workers.

## At a Glance

| Property | Value |
|---|---|
| Type | Hybrid |
| Interactive | No |
| Gate | No |
| Produces | Canonical observation rows and model-ready data |

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | Pipeline request | Provides temporal and semantic context |
| `stage0.result` | Stage 0 | Raw dataframe plus column descriptions |
| `stage1b.result` | Stage 1b | `CausalSpec` with indicators and extraction modes |
| `root_run_id` | Orchestrator runtime | Prefect run ID for worker progress events |
| `max_windows` | Pipeline config | Cap on extraction support windows |

## Process

1. Group indicators by extraction mode and observation window.
2. Run the computed path with direct Polars aggregation for `extraction_mode="computed"`.
3. Run the semantic path with concurrent LLM workers over support windows.
4. Merge both paths into canonical observation rows `{indicator, value, anchor_time, support_start, support_end}`.
5. Encode non-continuous types and sort by indicator then anchor time.

## Outputs

| Output | Type | Description |
|---|---|---|
| `workers` | `list[WorkerStatus]` | Per-worker status |
| `combined_extractions_sample` | `list[{indicator, value, anchor_time}]` | First extracted rows |
| `per_indicator_counts` | `dict[str, int]` | Count per indicator |
| `llm_trace` | `LLMTrace?` | Sampled worker trace |

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `WorkerStatus` | `{worker_id, status, n_extractions, n_windows, error}` | Runtime status for semantic workers |
| Observation row | `{indicator, value, anchor_time, support_start, support_end}` | Canonical extracted datum |

## Related Docs

- [../concepts/scope-and-timescales.md](../concepts/scope-and-timescales.md)
- [../concepts/artifact-glossary.md](../concepts/artifact-glossary.md)
- [../runtime/persistence-and-exposure.md](../runtime/persistence-and-exposure.md)
