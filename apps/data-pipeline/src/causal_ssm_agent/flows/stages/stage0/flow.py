"""Stage 0: Agentic ingestion logic and Prefect entrypoint.

An LLM agent explores a prepared input directory, writes Python code to parse
the contents, and produces a single Polars DataFrame. Code execution happens
inside a Modal CPU sandbox for isolation.
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.data import input_dir
from causal_ssm_agent.utils.llm import GenerateFn, LLMStageContext
from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key

from .tools import ModalCodeSandbox, make_ingestion_tools

logger = get_prefect_logger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Output of the agentic ingestion stage."""

    dataframe: pl.DataFrame
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
- **submit_table(column_descriptions_json)** — Finalize the result

## Workflow

1. Start by calling `list_files()` to see the input structure
2. Use `read_file_sample()` to understand file formats
3. Write Python code with `execute_python()` to parse the data
4. Iterate until `result_df` looks correct
5. Call `submit_table()` with column descriptions

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

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        await generate(messages, tools)

    # Extract result from capture
    df = capture.get("dataframe")
    if df is None or df.is_empty():
        raise ValueError("Ingestion agent did not produce a valid DataFrame")

    column_descriptions = capture.get("column_descriptions")
    if column_descriptions is None:
        column_descriptions = {}
    elif not isinstance(column_descriptions, dict) or set(column_descriptions) != set(df.columns):
        raise ValueError("Ingestion agent produced invalid column descriptions")

    return IngestionResult(
        dataframe=df,
        column_descriptions=column_descriptions,
    )


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


@task(cache_policy=INPUTS, persist_result=True, result_serializer="pickle")
async def agentic_ingest(
    workspace_id: str = "test_workspace", openrouter_api_key: str | None = None
) -> IngestionResult:
    """Run Stage 0 end to end for the latest uploaded file."""
    raw_storage_path = _find_raw_input(workspace_id)
    raw_name = raw_storage_path.rsplit("/", 1)[-1]
    logger.info("Ingesting %s for workspace %s", raw_name, workspace_id)

    config = get_config()
    with use_openrouter_api_key(openrouter_api_key):
        async with LLMStageContext("stage-0") as ctx:
            generate = ctx.make_generate(
                config.stage0_ingestion.llm.model,
                max_tool_turns=config.stage0_ingestion.max_tool_turns,
            )

            with tempfile.TemporaryDirectory(prefix="ingest_") as tmpdir:
                if storage.is_remote():
                    local_raw = Path(tmpdir) / "download" / raw_name
                    local_raw.parent.mkdir(parents=True, exist_ok=True)
                    storage.get_fs().get(raw_storage_path, str(local_raw))
                else:
                    local_raw = Path(raw_storage_path)

                extract_dir = _prepare_raw_input(local_raw, Path(tmpdir))
                result = await run_agentic_ingestion(extract_dir, generate)

            trace_out = ctx.finalize({})
            if "llm_trace" in trace_out:
                result.llm_trace = trace_out["llm_trace"]

            logger.info(
                "Ingested %d rows x %d columns",
                result.dataframe.shape[0],
                result.dataframe.shape[1],
            )
            return result
