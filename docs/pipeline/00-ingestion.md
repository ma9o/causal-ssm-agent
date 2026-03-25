# Stage 0: Agentic Data Ingestion

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | No | [`Raw Dataframe`](#raw-dataframe) |

Normalizes the latest uploaded raw export into one typed Polars dataframe with human-readable descriptions for every column.

## Inputs

| Input | Source | Description |
|---|---|---|
| File upload | User | Single file or a zip bundle containing the raw data |
| `workspace_id` | Pipeline request | Identifies the workspace. Stage 0 scans `data/{workspace_id}/input/` and selects the most recent non-hidden file. |

The ingestion agent can normalize most tabular or semi-structured formats as long as the data has a time dimension. Other columns can feed either [computed or semantic](01b-measurement-identifiability.md#extraction-modes) indicators downstream.

## Process

A sandboxed agentic ingestion loop with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`, requiring the agent to end with exactly one non-empty Polars dataframe plus a human-readable description for every dataframe column.

### Example

A ZIP containing `tickets.csv` and `deploys.csv` may be normalized into one dataframe with columns such as `timestamp`, `event_type`, `ticket_count`, `service_name`, `deploy_status`, and `incident_note`, where each row is one raw event on the shared timeline.


## Outputs

| Output | Type | Description |
|---|---|---|
| `raw_dataframe` | `polars.DataFrame` | Normalized typed observed-data table persisted indexed by `timestamp` |
| `column_descriptions` | `list[dict{name, description}]` | Human-readable descriptions for each column in the dataframe, derived from the agent's reasoning about the raw data |
| `llm_trace` | `LLMTrace` | Conversation trace for UI provenance and debugging |

### Raw Dataframe

Dynamic artifact containing all extracted data in a single Polars dataframe, indexed by a `timestamp` column that the ingestion agent identifies and normalizes. The dataframe may be wide (multiple columns) or long (event log format), depending on the raw data structure and what the ingestion agent determines is most appropriate for downstream processing. Each column has an associated human-readable description to provide semantic context for later stages.

