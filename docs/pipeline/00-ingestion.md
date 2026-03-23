# Stage 0: Agentic Data Ingestion

Parses arbitrary user-uploaded files into a typed Polars dataframe with column-level metadata.

## At a Glance

| Property | Value |
|---|---|
| Type | Semantic |
| Interactive | No |
| Gate | No |
| Produces | Typed dataframe summary plus column descriptions |

## Inputs

| Input | Source | Description |
|---|---|---|
| `workspace_id` | Pipeline request | Identifies the workspace; the stage discovers the latest uploaded file under `data/{workspace_id}/input/`. |

## Process

1. Locate and extract the uploaded file, including ZIP archives.
2. Run an agentic LLM conversation with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`.
3. Parse the source into a Polars dataframe and require human-readable column metadata.
4. Validate that both the dataframe and column descriptions are present before finalizing.

## Outputs

| Output | Type | Description |
|---|---|---|
| `source_label` | `str` | Human-readable data source name |
| `n_records` | `int` | Row count |
| `n_columns` | `int` | Column count |
| `date_range` | `{start, end}` | Temporal extent |
| `sample` | `list[dict]` | First rows of the ingested table |
| `column_descriptions` | `list[{name, dtype, description}]` | Per-column metadata |
| `llm_trace` | `LLMTrace?` | Tool-call history |

## Related Docs

- [../concepts/artifact-glossary.md](../concepts/artifact-glossary.md)
- [../runtime/persistence-and-exposure.md](../runtime/persistence-and-exposure.md)
