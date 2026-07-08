"""Tool adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

import contextlib
import csv
import datetime
import io
import json
import math
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.temporal.llm_subroutine_storage import (
    read_subroutine_json,
    read_subroutine_pickle,
    write_subroutine_json,
    write_subroutine_pickle,
)
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from nof1_causal_lab.machine.temporal.messages import (
        HarnessToolRequest,
        LLMToolExecutionInput,
        LLMToolSpec,
    )

_RAW_EXEC_NAMESPACE_NAMES = frozenset(
    {
        "pl",
        "polars",
        "csv",
        "json",
        "Path",
        "datetime",
        "re",
        "math",
        "io",
        "DATA_DIR",
    }
)


def _validate_tool_payload(
    *,
    context_kind: str,
    context_ref: str,
    data: dict,
) -> tuple[dict | None, str]:
    if context_kind == "measurement_extraction":
        from nof1_causal_lab.workers.schemas import validate_worker_output

        spec = read_subroutine_json(context_ref)
        output, errors = validate_worker_output(
            data,
            spec["measurement_structure"],
            spec["window_starts"],
        )
        if errors:
            return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        if output is None:
            return None, "VALIDATION ERRORS:\n- validator returned no output"
        return data, "VALID"

    if context_kind == "latent_structure":
        from nof1_causal_lab.flows.transitions.latent_structure.grounding import (
            latent_structure_grounding,
        )

        return latent_structure_grounding(data)

    if context_kind == "measurement_structure":
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        context = read_subroutine_json(context_ref)
        return measurement_structure_grounding(data, context["latent_structure"])

    raise ValueError(f"unknown LLM subroutine context kind {context_kind!r}")


def _raw_data_context(context_ref: str) -> dict[str, Any]:
    return read_subroutine_json(context_ref)


def _read_raw_dataframe(dataframe_ref: str):
    import polars as pl

    with storage.open_file(dataframe_ref, "rb") as file:
        return pl.read_ipc(file)


def _write_raw_dataframe(dataframe_ref: str, dataframe: Any) -> None:
    with storage.open_file(dataframe_ref, "wb") as file:
        dataframe.write_ipc(file)


def _raw_python_output_prefix(stdout: str, stderr: str) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part.strip()]
    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


def _execute_python_locally(extract_dir: Path, code: str) -> tuple[str, Any | None]:
    import polars as pl

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    namespace = {
        "__builtins__": __builtins__,
        "pl": pl,
        "polars": pl,
        "csv": csv,
        "json": json,
        "Path": Path,
        "datetime": datetime,
        "re": re,
        "math": math,
        "io": io,
        "DATA_DIR": str(extract_dir),
    }
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, namespace)
    except Exception:  # noqa: BLE001 - execute_python reports code tracebacks as tool feedback.
        prefix = _raw_python_output_prefix(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        return f"{prefix}Execution error:\n{traceback.format_exc()}", None

    prefix = _raw_python_output_prefix(stdout_buffer.getvalue(), stderr_buffer.getvalue())
    result_df = namespace.get("result_df")
    if result_df is None:
        defined = [
            name
            for name in namespace
            if not name.startswith("_") and name not in _RAW_EXEC_NAMESPACE_NAMES
        ]
        return (
            f"{prefix}No `result_df` variable found after execution.\n"
            f"Variables defined: {defined}\n"
            "Assign your final DataFrame to `result_df`."
        ), None

    if not isinstance(result_df, pl.DataFrame):
        return (
            f"{prefix}`result_df` is {type(result_df).__name__}, not a Polars DataFrame.\n"
            "Use pl.DataFrame(...) or pl.read_csv(...) to create one."
        ), None

    if result_df.is_empty():
        return f"{prefix}Warning: `result_df` is empty (0 rows). Check your parsing logic.", None

    lines = [
        f"Shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns",
        "",
        "Schema:",
    ]
    for column in result_df.columns:
        lines.append(f"  {column}: {result_df.schema[column]}")
    sample = min(5, len(result_df))
    lines.append(f"\nFirst {sample} rows:")
    lines.append(str(result_df.head(sample)))
    return f"{prefix}Success!\n\n" + "\n".join(lines), result_df


def _execute_raw_data_list_files(context_ref: str, args: dict[str, Any]) -> tuple[str, str | None]:
    from nof1_causal_lab.flows.transitions.ingestion.tools import _safe_resolve

    context = _raw_data_context(context_ref)
    extract_dir = Path(context["extract_dir"]).resolve()
    path = str(args.get("path") or ".")
    try:
        target = _safe_resolve(extract_dir, path)
    except ValueError as exc:
        return str(exc), None

    if not target.exists():
        return f"Path not found: {path}", None
    if not target.is_dir():
        return f"Not a directory: {path}", None

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
        return f"Empty directory: {path}", None
    return "\n".join(entries), None


def _execute_raw_data_read_file_sample(
    context_ref: str, args: dict[str, Any]
) -> tuple[str, str | None]:
    from nof1_causal_lab.flows.transitions.ingestion.tools import _safe_resolve

    context = _raw_data_context(context_ref)
    extract_dir = Path(context["extract_dir"]).resolve()
    path = str(args["path"])
    n_lines = int(args.get("n_lines") or 50)
    try:
        target = _safe_resolve(extract_dir, path)
    except ValueError as exc:
        return str(exc), None

    if not target.exists():
        return f"File not found: {path}", None
    if target.is_dir():
        return f"Is a directory, not a file: {path}", None

    suffix = target.suffix.lower()
    if suffix in (".xlsx", ".xls", ".parquet", ".feather", ".arrow"):
        size = target.stat().st_size
        return (
            f"Binary file: {target.name} ({size} bytes)\n"
            f"Type: {suffix}\n"
            f"Use pl.read_excel(Path(DATA_DIR) / '{path}') for Excel files\n"
            f"Use pl.read_parquet(Path(DATA_DIR) / '{path}') for Parquet files",
            None,
        )

    for encoding in ("utf-8", "latin-1"):
        try:
            with target.open(encoding=encoding) as fh:
                lines = []
                for index, line in enumerate(fh):
                    if index >= n_lines:
                        break
                    lines.append(line.rstrip("\n"))
            total_lines = f"(showing first {n_lines} lines)" if len(lines) == n_lines else ""
            header = f"File: {path} (encoding: {encoding}) {total_lines}\n"
            return header + "\n".join(lines), None
        except UnicodeDecodeError:
            continue

    return f"Could not read file {path} with utf-8 or latin-1 encoding", None


def _execute_raw_data_python(context_ref: str, args: dict[str, Any]) -> tuple[str, str | None]:
    context = _raw_data_context(context_ref)
    code = str(args["code"])
    output, result_df = _execute_python_locally(Path(context["extract_dir"]).resolve(), code)
    if result_df is not None:
        _write_raw_dataframe(context["dataframe_ref"], result_df)
    return output, None


def _execute_raw_data_submit_table(
    context_ref: str,
    result_ref: str,
    args: dict[str, Any],
) -> tuple[str, str | None]:
    import polars as pl

    context = _raw_data_context(context_ref)
    dataframe_ref = context["dataframe_ref"]
    if not storage.exists(dataframe_ref):
        return (
            "No DataFrame available. Run execute_python first and assign your result to `result_df`.",
            None,
        )

    df = _read_raw_dataframe(dataframe_ref)
    if df.is_empty():
        return "DataFrame is empty (0 rows). Parse more data before submitting.", None

    try:
        col_descs = json.loads(str(args["column_descriptions_json"]))
    except json.JSONDecodeError as exc:
        return f"Invalid JSON for column_descriptions: {exc}", None

    if not isinstance(col_descs, dict):
        return (
            "column_descriptions_json must be a JSON object mapping column names to descriptions.",
            None,
        )

    if "timestamp" not in df.columns:
        return (
            "DataFrame must contain a Datetime column named 'timestamp' as the primary "
            "temporal axis. Rename or cast the time column in your execute_python code.",
            None,
        )
    if df.schema["timestamp"] not in (pl.Datetime, pl.Date):
        return (
            f"Column 'timestamp' has type {df.schema['timestamp']}; it must be Datetime "
            "or Date. Cast it in your execute_python code.",
            None,
        )

    missing = [column for column in df.columns if column not in col_descs]
    if missing:
        return f"Missing descriptions for columns: {missing}", None

    extra = [column for column in col_descs if column not in df.columns]
    if extra:
        return f"Descriptions for non-existent columns: {extra}", None

    write_subroutine_json(
        result_ref,
        {
            "dataframe_ref": dataframe_ref,
            "column_descriptions": col_descs,
        },
    )
    return "VALID", result_ref


def _save_model_spec_state(state_ref: str, state: Any) -> None:
    write_subroutine_pickle(state_ref, state)


def _load_model_spec_state(state_ref: str) -> Any:
    return read_subroutine_pickle(state_ref)


async def _execute_model_spec_search_literature(
    context_ref: str,
    args: dict[str, Any],
) -> tuple[str, str | None]:
    from nof1_causal_lab.flows.transitions.model_spec.tools import search_literature

    context = read_subroutine_json(context_ref)
    state_ref = context["state_ref"]
    state = _load_model_spec_state(state_ref)
    query = str(args.get("query") or "")
    parameter_name = str(args.get("parameter_name") or "")
    if not query:
        return "Error: query is required", None
    state.search_queries[parameter_name] = query
    cached = state.search_cache.get(query)
    if cached is not None:
        return cached, None
    result = await search_literature(query)
    state.search_cache[query] = result
    _save_model_spec_state(state_ref, state)
    return result, None


def _execute_model_spec_submit_construct(
    context_ref: str,
    args: dict[str, Any],
) -> tuple[str, str | None]:
    context = read_subroutine_json(context_ref)
    state_ref = context["state_ref"]
    state = _load_model_spec_state(state_ref)
    feedback = state.submit_construct(
        construct=str(args["construct"]),
        indicators=list(args["indicators"]),
        priors=dict(args["priors"]),
        accept=args.get("accept"),
    )
    _save_model_spec_state(state_ref, state)
    return feedback, None


async def execute_subroutine_tool(
    *,
    input: LLMToolExecutionInput | HarnessToolRequest,
    tool: LLMToolSpec,
    args: dict[str, Any],
    result_ref: str,
) -> tuple[str, str | None]:
    if tool.executor == "context_json_validation":
        data = json.loads(str(args[tool.param_name]))
        context_output, feedback = _validate_tool_payload(
            context_kind=input.context_kind,
            context_ref=input.context_ref,
            data=data,
        )
        if context_output is not None:
            write_subroutine_json(result_ref, context_output)
            return feedback, result_ref
        return feedback, None

    if tool.executor == "raw_data_list_files":
        return _execute_raw_data_list_files(input.context_ref, args)
    if tool.executor == "raw_data_read_file_sample":
        return _execute_raw_data_read_file_sample(input.context_ref, args)
    if tool.executor == "raw_data_execute_python":
        return _execute_raw_data_python(input.context_ref, args)
    if tool.executor == "raw_data_submit_table":
        return _execute_raw_data_submit_table(input.context_ref, result_ref, args)
    if tool.executor == "model_spec_submit_construct":
        return _execute_model_spec_submit_construct(input.context_ref, args)
    if tool.executor == "model_spec_search_literature":
        return await _execute_model_spec_search_literature(input.context_ref, args)

    raise ValueError(f"Unsupported tool executor: {tool.executor}")
