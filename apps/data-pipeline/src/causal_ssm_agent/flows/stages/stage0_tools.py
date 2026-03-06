"""Stage 0: Agentic ingestion tools.

Provides the LLM agent with tools to explore a zip archive and
produce a single Polars DataFrame from arbitrary file contents.
"""

import csv
import datetime
import io
import json
import math
import re
import traceback
from pathlib import Path

import polars as pl
from inspect_ai.tool import Tool, tool

# Curated namespace for LLM-generated code execution.
# Only safe, data-oriented modules are exposed.
_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "isinstance": isinstance,
    "type": type,
    "None": None,
    "True": True,
    "False": False,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "open": open,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "Exception": Exception,
}


def _make_safe_globals(data_dir: Path) -> dict:
    """Build the globals dict injected into exec() for LLM code."""
    return {
        "__builtins__": _SAFE_BUILTINS,
        "pl": pl,
        "polars": pl,
        "csv": csv,
        "json": json,
        "Path": Path,
        "datetime": datetime,
        "re": re,
        "math": math,
        "io": io,
        "DATA_DIR": str(data_dir),
    }


def _format_df_preview(df: pl.DataFrame, max_rows: int = 5) -> str:
    """Format a DataFrame schema + sample for LLM feedback."""
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"]
    lines.append("Schema:")
    for col in df.columns:
        lines.append(f"  {col}: {df.schema[col]}")
    lines.append(f"\nFirst {min(max_rows, len(df))} rows:")
    lines.append(str(df.head(max_rows)))
    return "\n".join(lines)


def _safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve a user-provided path safely within the base directory."""
    resolved = (base / user_path).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"Path traversal blocked: {user_path}")
    return resolved


def make_ingestion_tools(extract_dir: Path) -> tuple[list[Tool], dict]:
    """Create the toolset for the agentic ingestion agent.

    Args:
        extract_dir: Root directory of the extracted zip contents.

    Returns:
        Tuple of (tools_list, capture_dict). After the agent loop,
        check capture["dataframe"] for the final DataFrame and
        capture["column_descriptions"] for per-column descriptions.
    """
    capture: dict = {}

    @tool
    def list_files():
        """List files in the extracted archive."""

        async def execute(path: str = ".") -> str:
            """
            List files and directories at the given path within the archive.

            Args:
                path: Relative path within the archive (default: root).

            Returns:
                Formatted directory listing with file sizes and types.
            """
            try:
                target = _safe_resolve(extract_dir, path)
            except ValueError as e:
                return str(e)

            if not target.exists():
                return f"Path not found: {path}"
            if not target.is_dir():
                return f"Not a directory: {path}"

            entries = []
            for item in sorted(target.iterdir()):
                rel = item.relative_to(extract_dir)
                if item.is_dir():
                    n_children = sum(1 for _ in item.iterdir())
                    entries.append(f"  [dir]  {rel}/  ({n_children} items)")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    entries.append(f"  [file] {rel}  ({size_str})")

            if not entries:
                return f"Empty directory: {path}"
            return "\n".join(entries)

        return execute

    @tool
    def read_file_sample():
        """Read a sample of lines from a file to understand its format."""

        async def execute(path: str, n_lines: int = 50) -> str:
            """
            Read the first N lines of a file to understand its structure.

            Args:
                path: Relative path to the file within the archive.
                n_lines: Number of lines to read (default: 50).

            Returns:
                File contents (first N lines) or a description for binary files.
            """
            try:
                target = _safe_resolve(extract_dir, path)
            except ValueError as e:
                return str(e)

            if not target.exists():
                return f"File not found: {path}"
            if target.is_dir():
                return f"Is a directory, not a file: {path}"

            suffix = target.suffix.lower()

            # Binary formats: return metadata instead of contents
            if suffix in (".xlsx", ".xls", ".parquet", ".feather", ".arrow"):
                size = target.stat().st_size
                return (
                    f"Binary file: {target.name} ({size} bytes)\n"
                    f"Type: {suffix}\n"
                    f"Use pl.read_excel(Path(DATA_DIR) / '{path}') for Excel files\n"
                    f"Use pl.read_parquet(Path(DATA_DIR) / '{path}') for Parquet files"
                )

            # Text-based files: read first N lines
            for encoding in ("utf-8", "latin-1"):
                try:
                    with target.open(encoding=encoding) as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= n_lines:
                                break
                            lines.append(line.rstrip("\n"))
                    total_lines = (
                        f"(showing first {n_lines} lines)" if len(lines) == n_lines else ""
                    )
                    header = f"File: {path} (encoding: {encoding}) {total_lines}\n"
                    return header + "\n".join(lines)
                except UnicodeDecodeError:
                    continue

            return f"Could not read file {path} with utf-8 or latin-1 encoding"

        return execute

    @tool
    def execute_python():
        """Execute Python code to parse files into a Polars DataFrame."""

        async def execute(code: str) -> str:
            """
            Execute Python code that produces a Polars DataFrame.

            The code must assign its result to a variable named `result_df`.
            Available in the namespace: polars (as `pl`), csv, json, Path,
            datetime, re, math, io. DATA_DIR points to the extracted archive.

            Args:
                code: Python code to execute.

            Returns:
                DataFrame schema and sample rows on success, or traceback on error.
            """
            safe_globals = _make_safe_globals(extract_dir)
            local_ns: dict = {}

            try:
                exec(code, safe_globals, local_ns)
            except Exception:
                return f"Execution error:\n{traceback.format_exc()}"

            result_df = local_ns.get("result_df")
            if result_df is None:
                available = [k for k in local_ns if not k.startswith("_")]
                return (
                    "No `result_df` variable found after execution.\n"
                    f"Variables defined: {available}\n"
                    "Assign your final DataFrame to `result_df`."
                )

            if not isinstance(result_df, pl.DataFrame):
                return (
                    f"`result_df` is {type(result_df).__name__}, not a Polars DataFrame.\n"
                    "Use pl.DataFrame(...) or pl.read_csv(...) to create one."
                )

            if result_df.is_empty():
                return "Warning: `result_df` is empty (0 rows). Check your parsing logic."

            # Store for later submission
            capture["dataframe"] = result_df
            return f"Success!\n\n{_format_df_preview(result_df)}"

        return execute

    @tool
    def submit_table():
        """Validate and finalize the ingested DataFrame with column descriptions."""

        async def execute(source_label: str, column_descriptions_json: str) -> str:
            """
            Finalize the ingested DataFrame and provide column metadata.

            Call this after you have a good DataFrame from execute_python.

            Args:
                source_label: A short human-readable label for the data source
                    (e.g., "Medical records from Doctolib", "Fitness tracker export").
                column_descriptions_json: JSON object mapping column names to
                    descriptions, e.g. {"date": "Appointment date", "ldl": "LDL cholesterol mg/dL"}.

            Returns:
                "VALID" on success, or validation errors.
            """
            df = capture.get("dataframe")
            if df is None:
                return (
                    "No DataFrame available. Run execute_python first "
                    "and assign your result to `result_df`."
                )

            if df.is_empty():
                return "DataFrame is empty (0 rows). Parse more data before submitting."

            # Parse column descriptions
            try:
                col_descs = json.loads(column_descriptions_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON for column_descriptions: {e}"

            if not isinstance(col_descs, dict):
                return "column_descriptions_json must be a JSON object mapping column names to descriptions."

            # Check all columns have descriptions
            missing = [c for c in df.columns if c not in col_descs]
            if missing:
                return f"Missing descriptions for columns: {missing}"

            extra = [c for c in col_descs if c not in df.columns]
            if extra:
                return f"Descriptions for non-existent columns: {extra}"

            # Store final result
            capture["source_label"] = source_label
            capture["column_descriptions"] = col_descs
            return "VALID"

        return execute

    tools = [list_files(), read_file_sample(), execute_python(), submit_table()]
    return tools, capture
