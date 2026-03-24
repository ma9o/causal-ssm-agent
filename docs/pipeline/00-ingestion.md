# Stage 0: Agentic Data Ingestion

| Type | Interactive | Gate | Produces |
|---|---|---|---|
| llm | No | No | [`Raw Dataframe`](#raw-dataframe) plus column descriptions |

Normalizes the latest uploaded raw export into one typed Polars dataframe with human-readable descriptions for every column.

## Inputs

| Input | Source | Description |
|---|---|---|
| `workspace_id` | Pipeline request | Identifies the workspace. Stage 0 scans `data/{workspace_id}/input/` and selects the most recent non-hidden file. |

For workspace layout and raw file placement, see [../guides/data_workflow.md](../guides/data_workflow.md).

## Process

1. Select the latest non-hidden uploaded file for the workspace.
2. If the file is a ZIP archive, extract it into a prepared directory; otherwise copy it unchanged into that directory.
3. Run one agentic ingestion loop with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`.
4. Require the agent to end with exactly one non-empty Polars dataframe plus a human-readable description for every dataframe column.
5. Persist the full dataframe as a parquet sidecar and expose only a summary, schema, and trace in the public stage payload.

If a dataset spans several related raw files, they should usually be bundled into one archive. Stage 0 reads one latest uploaded file, so multi-file datasets need to arrive as one ingestible unit.

For what makes a useful upload, see [../guides/data_contract.md](../guides/data_contract.md).

Stage 0 is a normalization step, not a modeling step. It decides how raw files become one coherent observed table. It does not define constructs, indicators, support-window semantics, validation findings, or model-ready encodings.

## Outputs

Stage 0's real downstream output is the `Raw Dataframe`; the public web payload is only a lightweight summary containing `source_label`, `n_records`, `n_columns`, `date_range`, a display `sample`, `column_descriptions`, and optional `llm_trace`. For how that public summary differs from the persisted parquet artifact, see [../runtime/persistence-and-exposure.md](../runtime/persistence-and-exposure.md).

## Definitions

### Raw Dataframe

The raw dataframe is the normalized Stage 0 dataframe produced from the raw upload. It owns:

- the full typed dataframe that downstream stages read from parquet
- the row grain chosen during ingestion, typically one raw event, observation, or timepoint
- the column set and dtypes chosen during ingestion
- the source label and column-level descriptions attached to that dataframe

Stage 0 may rename, cast, join, or concatenate related raw files to produce this table, but it still ends with one coherent observed-data dataframe.

Example: a ZIP containing `vitals.csv` and `med_admin.csv` may be normalized into one dataframe with columns such as `timestamp`, `event_type`, `heart_rate_bpm`, `medication_name`, `dose_mg`, and `note_text`, where each row is one raw event on the shared timeline.
