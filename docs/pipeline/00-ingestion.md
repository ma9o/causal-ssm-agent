# Stage 0: Agentic Data Ingestion

| Modality | Interactive | Gate | Produces |
|---|---|---|---|
| Semantic | No | No | [`Raw Dataframe`](#raw-dataframe) |

Normalizes the latest uploaded raw export into one typed Polars dataframe with human-readable descriptions for every column.

## Inputs

| Input | Source | Description |
|---|---|---|
| `workspace_id` | Pipeline request | Identifies the workspace. Stage 0 scans `data/{workspace_id}/input/` and selects the most recent non-hidden file. |

## Process

1. Scans `data/{workspace_id}/input/` and ingests the most recent non-hidden file. The question is stored in `data/{workspace_id}/query.txt`, and stage outputs land in `data/{workspace_id}/run/`
2. Run a sandboxed agentic ingestion loop with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`, requiring the agent to end with exactly one non-empty Polars dataframe plus a human-readable description for every dataframe column.
3. Persist the full dataframe as a parquet sidecar and expose only serializable plumbing fields in the persisted stage payload; the web layer derives convenience summary fields from the parquet sidecar when rendering results.

If a dataset spans several related raw files, they should usually be bundled into one archive. Stage 0 reads one latest uploaded file, so multi-file datasets need to arrive as one ingestible unit.

For what makes a useful upload, see [../guides/data_contract.md](../guides/data_contract.md).

## Outputs

| Output | Type | Description |
|---|---|---|
| `raw_dataframe` | `polars.DataFrame` | Normalized typed observed-data table persisted as the Stage 0 parquet artifact |

Additionally the stage payload includes:

- `column_descriptions` is the persisted JSON projection of per-column descriptions, stored as `{name, description}` entries.
- `llm_trace` is optional runtime provenance for the UI.

For how the persisted payload, restored runtime state, and web-facing projection differ, see [execution-semantics.md](../reference/execution-semantics.md#2-persistence-and-exposure-boundary).

## Definitions

### Raw Dataframe

The raw dataframe includes:

- the full typed dataframe that downstream stages read from parquet
- the row grain chosen during ingestion, typically one raw event, observation, or timepoint
- the column set and dtypes chosen during ingestion
- the column-level descriptions attached to that dataframe

The agent may rename, cast, join, or concatenate related raw files to produce this table, but it still ends with one coherent observed-data dataframe.

Example: a ZIP containing `tickets.csv` and `deploys.csv` may be normalized into one dataframe with columns such as `timestamp`, `event_type`, `ticket_count`, `service_name`, `deploy_status`, and `incident_note`, where each row is one raw event on the shared timeline.
