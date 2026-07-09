"""ingestion shared contracts, prompts, and input staging helpers."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import polars as pl

from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import input_dir


@dataclass
class IngestionResult:
    """Output of the agentic ingestion stage."""

    dataframe: pl.DataFrame
    column_descriptions: dict[str, str] = field(default_factory=dict)
    llm_trace_ref: str | None = None


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
- **submit_table(column_descriptions_json)** — Finalize the result

## Workflow

1. Start by calling `list_files()` to see the input structure
2. Use `read_file_sample()` to understand file formats
3. Write Python code with `execute_python()` to parse the data
4. Iterate until `result_df` looks correct
5. Call `submit_table()` with column descriptions

## Code Environment

Your code runs in the local pipeline process. \
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
- The primary temporal axis MUST be a Datetime-typed column named exactly \
`timestamp`. Rename the source time column if necessary.
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


def _find_raw_input(workspace_id: str) -> str:
    """Find the most recent uploaded file for a workspace."""
    user_dir = input_dir(workspace_id)
    if not storage.exists(user_dir):
        raise FileNotFoundError(f"No raw data directory: {user_dir}")

    entries = storage.listdir(user_dir)
    files: list[tuple[str, float]] = []
    for entry in entries:
        name = entry.rsplit("/", 1)[-1]
        if name.startswith("."):
            continue
        info = storage.file_info(entry)
        if info.get("type") == "file":
            mtime = info.get("last_modified", info.get("LastModified", info.get("mtime", 0)))
            if hasattr(mtime, "timestamp"):
                mtime = mtime.timestamp()
            files.append((entry, float(mtime)))

    if not files:
        raise FileNotFoundError(f"No files in {user_dir}")

    files.sort(key=lambda item: item[1], reverse=True)
    return files[0][0]


def _prepare_raw_input(raw_path: Path, dest_dir: Path) -> Path:
    """Prepare an uploaded file tree for the ingestion agent."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_zipfile(raw_path):
        with ZipFile(raw_path, "r") as archive:
            archive.extractall(dest_dir)
        return dest_dir

    shutil.copy2(raw_path, dest_dir / raw_path.name)
    return dest_dir
