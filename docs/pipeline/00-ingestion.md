# Stage 0: Agentic Data Ingestion

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | No | `raw_dataframe` |

Normalizes the latest uploaded raw export into one typed Polars dataframe.

## Inputs

| Input | Source | Description |
|---|---|---|
| File upload | User | Single file or a zip bundle containing the raw data |
| `workspace_id` | Pipeline request | Identifies the workspace. Stage 0 scans `data/{workspace_id}/input/` and selects the most recent non-hidden file. |

The ingestion agent can normalize most tabular or semi-structured formats as long as the data has a time dimension. Other columns can feed either [computed or semantic](01b-measurement-structure-identifiability.md#extraction-modes) indicators downstream.

## Process

A sandboxed agentic ingestion loop with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`.

### Example

A ZIP containing `tickets.csv` and `deploys.csv` may be normalized into one dataframe with columns such as `timestamp`, `event_type`, `ticket_count`, `service_name`, `deploy_status`, and `incident_note`, where each row is one raw event on the shared timeline.

## Outputs

| Output | Type | Description |
|---|---|---|
| `raw_dataframe` | `polars.DataFrame` | Normalized typed observed-data table indexed by `timestamp`. May be wide (multiple columns) or long (event log format), depending on the raw data structure. |
| `column_descriptions` | `list[dict{name, description}]` | Human-readable descriptions for columns when the agent provides them via `submit_table`; may be empty when Stage 0 completes from the captured dataframe alone |
| `llm_trace` | `LLMTrace` | Conversation trace for UI provenance and debugging |
