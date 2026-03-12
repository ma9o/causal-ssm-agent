"""Stage 0: Agentic ingestion core logic.

An LLM agent explores a prepared input directory, writes Python code to parse
the contents, and produces a single Polars DataFrame. Code execution happens
inside a Modal CPU sandbox for isolation.
"""

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from causal_ssm_agent.utils.llm import GenerateFn

from .stage0_tools import ModalCodeSandbox, make_ingestion_tools

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Output of the agentic ingestion stage."""

    dataframe: pl.DataFrame
    source_label: str
    column_descriptions: dict[str, str] = field(default_factory=dict)
    llm_trace: dict | None = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a data ingestion specialist. You have been given uploaded input that \
has been staged in a directory. The directory may contain extracted archive \
contents or raw files copied directly. Your task is to explore the contents, \
understand the data formats, and write Python code to parse everything into a \
single Polars DataFrame.

## Available Tools

- **list_files(path)** — List directory contents (sizes, types)
- **read_file_sample(path, n_lines)** — Peek at the first N lines of a file
- **execute_python(code)** — Run Python code; assign your result to `result_df`
- **submit_table(source_label, column_descriptions_json)** — Finalize the result

## Workflow

1. Start by calling `list_files()` to see the input structure
2. Use `read_file_sample()` to understand file formats
3. Write Python code with `execute_python()` to parse the data
4. Iterate until `result_df` looks correct
5. Call `submit_table()` with a source label and column descriptions

## Code Environment

Your code runs inside an isolated sandbox container. \
Available in the namespace:
- `polars` / `pl` — Polars library
- `csv`, `json`, `re`, `math`, `io`, `datetime` — standard library
- `Path` — pathlib.Path
- `DATA_DIR` — string path to the prepared input directory root

Common patterns:
```python
df = pl.read_csv(Path(DATA_DIR) / "file.csv")
df = pl.read_excel(Path(DATA_DIR) / "file.xlsx")
data = json.loads(Path(Path(DATA_DIR) / "file.json").read_text())
df = pl.DataFrame(data)
```

## Guidelines

- Produce a SINGLE wide-format DataFrame (one row per observation/timepoint)
- Include a date/timestamp column if temporal data exists
- Use clean column names (lowercase, underscores, no spaces)
- Cast numeric columns to appropriate types (Float64, Int64)
- Handle encoding issues gracefully (try utf-8, then latin-1)
- Drop empty or irrelevant columns
- If multiple files contain related data, join or concatenate them

## Important

- Assign your final DataFrame to `result_df`
- Once `submit_table` returns "VALID", STOP — the result is saved
"""

USER_PROMPT = """\
The uploaded input files have been staged and are available via DATA_DIR.

Explore the contents and parse all relevant data into a single Polars DataFrame.
"""

FINALIZE_PROMPT = """\
The dataframe has already been created successfully and is stored in memory.

Do not call `execute_python` again unless the dataframe itself is wrong.
Call `submit_table()` exactly once with:
- a concise human-readable `source_label`
- a JSON object with descriptions for EVERY column in the dataframe

Current schema:
{schema}

Sample rows:
{sample}
"""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _has_submission_metadata(capture: dict) -> bool:
    df = capture.get("dataframe")
    if df is None or df.is_empty():
        return False

    source_label = capture.get("source_label")
    column_descriptions = capture.get("column_descriptions")
    return (
        bool(source_label)
        and isinstance(column_descriptions, dict)
        and set(column_descriptions) == set(df.columns)
    )


def _format_finalize_prompt(df: pl.DataFrame) -> str:
    schema_lines = [f"- {col}: {df.schema[col]}" for col in df.columns]
    return FINALIZE_PROMPT.format(
        schema="\n".join(schema_lines),
        sample=df.head(5),
    )


async def run_agentic_ingestion(
    extract_dir: Path,
    generate: GenerateFn,
) -> IngestionResult:
    """Run the agentic ingestion loop.

    Spins up a Modal CPU sandbox, then lets the LLM agent explore the
    prepared input directory using tools and produce a Polars DataFrame.

    Args:
        extract_dir: Root directory of the prepared input files.
        generate: Async generate function (from make_generate_fn).

    Returns:
        IngestionResult with the parsed DataFrame and metadata.

    Raises:
        ValueError: If the agent did not produce a valid table.
    """
    with ModalCodeSandbox(extract_dir) as sandbox:
        tools, capture = make_ingestion_tools(extract_dir, sandbox)
        submit_tool = next(tool for tool in tools if tool.name == "submit_table")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        await generate(messages, tools)

        df = capture.get("dataframe")
        if df is not None and not df.is_empty() and not _has_submission_metadata(capture):
            await generate(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _format_finalize_prompt(df)},
                ],
                [submit_tool],
            )

    # Extract result from capture
    df = capture.get("dataframe")
    if df is None or df.is_empty():
        raise ValueError("Ingestion agent did not produce a valid DataFrame")
    if not _has_submission_metadata(capture):
        raise ValueError("Ingestion agent produced a DataFrame but did not finalize it")

    source_label = capture["source_label"]
    column_descriptions = capture["column_descriptions"]

    return IngestionResult(
        dataframe=df,
        source_label=source_label,
        column_descriptions=column_descriptions,
    )
