"""ingestion: Agentic ingestion tools.

Provides the LLM agent with tools to explore staged input files and
produce a single Polars DataFrame from arbitrary file contents.

Code execution runs inside a Modal CPU sandbox for isolation.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.utils.openrouter_client import Tool, tool

if TYPE_CHECKING:
    from pathlib import Path

    import modal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runner script executed inside the Modal sandbox.
#
# Reads user code from /tmp/user_code.py, executes it, and writes
# structured status to /tmp/status.json.  If a result_df DataFrame
# is produced, it is serialised to /tmp/result.ipc (Arrow IPC).
# ---------------------------------------------------------------------------

_SANDBOX_RUNNER = r"""
import polars as pl
import csv, json, re, math, io, datetime, traceback, sys
from pathlib import Path

DATA_DIR = "/data"

user_code = open("/tmp/user_code.py").read()

ns = {
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
    "DATA_DIR": DATA_DIR,
}

status = {}
try:
    exec(user_code, ns)
except Exception:
    status = {"s": "error", "tb": traceback.format_exc()}
else:
    result_df = ns.get("result_df")
    if result_df is not None and isinstance(result_df, pl.DataFrame):
        if result_df.is_empty():
            status = {"s": "empty"}
        else:
            result_df.write_ipc("/tmp/result.ipc")
            lines = [
                f"Shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns",
                "",
                "Schema:",
            ]
            for col in result_df.columns:
                lines.append(f"  {col}: {result_df.schema[col]}")
            sample = min(5, len(result_df))
            lines.append(f"\nFirst {sample} rows:")
            lines.append(str(result_df.head(sample)))
            status = {"s": "saved", "preview": "\n".join(lines)}
    elif result_df is not None:
        status = {"s": "not_df", "type": type(result_df).__name__}
    else:
        defined = [
            k
            for k in ns
            if not k.startswith("_")
            and k
            not in (
                "pl", "polars", "csv", "json", "Path",
                "datetime", "re", "math", "io", "DATA_DIR",
            )
        ]
        status = {"s": "no_result", "defined": defined}

with open("/tmp/status.json", "w") as f:
    json.dump(status, f)
""".lstrip()


# ---------------------------------------------------------------------------
# Modal CPU sandbox
# ---------------------------------------------------------------------------


def _make_sandbox_image():
    """Build a lightweight Modal image for data parsing."""
    import modal

    return modal.Image.debian_slim(python_version="3.12").pip_install(
        "polars", "openpyxl", "fastexcel"
    )


class ModalCodeSandbox:
    """Modal CPU sandbox for executing LLM-generated code safely.

    The prepared input directory is uploaded once on ``start()``. Each
    ``execute()`` call writes the user code into the sandbox and runs
    the runner script, returning feedback text and (optionally) the
    deserialised Polars DataFrame.
    """

    def __init__(self, extract_dir: Path, *, timeout: int = 600):
        self._extract_dir = extract_dir
        self._timeout = timeout
        self._sandbox: modal.Sandbox | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        import modal

        image = _make_sandbox_image()
        app = modal.App.lookup("nof1-causal-lab-ingestion", create_if_missing=True)

        self._sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=self._timeout,
            cpu=1,
            block_network=True,
        )

        # Upload the prepared input directory as a tarball and unpack it.
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(str(self._extract_dir), arcname=".")
        tar_bytes = tar_buf.getvalue()

        f = self._sandbox.open("/tmp/archive.tar.gz", "wb")
        f.write(tar_bytes)
        f.close()

        self._sandbox.exec("mkdir", "-p", "/data").wait()
        self._sandbox.exec("tar", "xzf", "/tmp/archive.tar.gz", "-C", "/data").wait()

        # Upload runner script (once).
        f = self._sandbox.open("/tmp/runner.py", "w")
        f.write(_SANDBOX_RUNNER)
        f.close()

        logger.info("Modal sandbox ready (timeout=%ds)", self._timeout)

    def terminate(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except (AttributeError, OSError, RuntimeError, ValueError):
                logger.warning("Failed to terminate sandbox", exc_info=True)
            self._sandbox = None

    def __enter__(self) -> ModalCodeSandbox:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.terminate()

    # -- code execution ------------------------------------------------------

    def execute(self, code: str) -> tuple[str, pl.DataFrame | None]:
        """Run *code* inside the sandbox.

        Returns:
            ``(feedback_text, result_df_or_none)``
        """
        assert self._sandbox is not None, "Sandbox not started"

        # Write user code.
        f = self._sandbox.open("/tmp/user_code.py", "w")
        f.write(code)
        f.close()

        # Run the runner.
        process = self._sandbox.exec("python", "/tmp/runner.py", timeout=120)
        process.wait()

        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Read structured status written by the runner.
        try:
            sf = self._sandbox.open("/tmp/status.json", "r")
            status: dict = json.loads(sf.read())
            sf.close()
        except (FileNotFoundError, OSError, ValueError) as exc:
            # Runner crashed before writing status (OOM, timeout, ...).
            logger.warning("Code execution runner crashed (%s: %s)", type(exc).__name__, exc)
            output = ""
            if stdout:
                output += stdout
            if stderr:
                output += f"\n{stderr}" if output else stderr
            return output or "Code execution failed (no output).", None

        return self._parse_status(status, stdout)

    # -- helpers -------------------------------------------------------------

    def _parse_status(self, status: dict, stdout: str) -> tuple[str, pl.DataFrame | None]:
        s = status.get("s")
        user_output = stdout.strip()
        prefix = f"{user_output}\n\n" if user_output else ""

        if s == "error":
            return f"{prefix}Execution error:\n{status['tb']}", None

        if s == "saved":
            assert self._sandbox is not None
            rf = self._sandbox.open("/tmp/result.ipc", "rb")
            ipc_bytes = rf.read()
            rf.close()
            result_df = pl.read_ipc(io.BytesIO(ipc_bytes))
            return f"{prefix}Success!\n\n{status['preview']}", result_df

        if s == "empty":
            return (
                f"{prefix}Warning: `result_df` is empty (0 rows). Check your parsing logic."
            ), None

        if s == "not_df":
            type_name = status.get("type", "unknown")
            return (
                f"{prefix}`result_df` is {type_name}, not a Polars DataFrame.\n"
                "Use pl.DataFrame(...) or pl.read_csv(...) to create one."
            ), None

        if s == "no_result":
            defined = status.get("defined", [])
            return (
                f"{prefix}No `result_df` variable found after execution.\n"
                f"Variables defined: {defined}\n"
                "Assign your final DataFrame to `result_df`."
            ), None

        return f"{prefix}Unexpected sandbox status.", None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve a user-provided path safely within the base directory."""
    base_resolved = base.resolve()
    resolved = (base / user_path).resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"Path traversal blocked: {user_path}")
    return resolved


def make_ingestion_tools(
    extract_dir_raw: Path, sandbox: ModalCodeSandbox
) -> tuple[list[Tool], dict]:
    """Create the toolset for the agentic ingestion agent.

    Args:
        extract_dir_raw: Root directory of the prepared input files.
        sandbox: Sandbox used to execute model-generated parsing code.

    Returns:
        Tuple of (tools_list, capture_dict). After the agent loop,
        check capture["dataframe"] for the final DataFrame and
        capture["column_descriptions"] for per-column descriptions.
    """
    # Resolve once to avoid macOS /var vs /private/var symlink mismatches
    extract_dir = extract_dir_raw.resolve()
    capture: dict = {}

    @tool
    def list_files():
        """List files in the prepared input directory."""

        async def execute(path: str = ".") -> str:
            """
            List files and directories at the given path within the input directory.

            Args:
                path: Relative path within the input directory (default: root).

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
                path: Relative path to the file within the input directory.
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
                    with target.open(encoding=encoding) as fh:
                        lines = []
                        for i, line in enumerate(fh):
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
        """Execute Python code in a Modal sandbox to parse files into a Polars DataFrame."""

        async def execute(code: str) -> str:
            """
            Execute Python code that produces a Polars DataFrame.

            The code runs inside an isolated Modal sandbox container.
            Assign your result to a variable named ``result_df``.
            Available in the namespace: polars (as ``pl``), csv, json, Path,
            datetime, re, math, io. ``DATA_DIR`` points to the prepared input directory.

            Args:
                code: Python code to execute.

            Returns:
                DataFrame schema and sample rows on success, or traceback on error.
            """
            output, result_df = sandbox.execute(code)
            if result_df is not None:
                capture["dataframe"] = result_df
            return output

        return execute

    @tool
    def submit_table():
        """Validate and finalize the ingested DataFrame with column descriptions."""

        async def execute(column_descriptions_json: str) -> str:
            """
            Finalize the ingested DataFrame and provide column metadata.

            Call this after you have a good DataFrame from execute_python.

            Args:
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

            # Require canonical timestamp column
            if "timestamp" not in df.columns:
                return (
                    "DataFrame must contain a Datetime column named 'timestamp' "
                    "as the primary temporal axis. Rename or cast the time column "
                    "in your execute_python code."
                )
            if df.schema["timestamp"] not in (pl.Datetime, pl.Date):
                return (
                    f"Column 'timestamp' has type {df.schema['timestamp']}; "
                    "it must be Datetime or Date. Cast it in your execute_python code."
                )

            # Check all columns have descriptions
            missing = [c for c in df.columns if c not in col_descs]
            if missing:
                return f"Missing descriptions for columns: {missing}"

            extra = [c for c in col_descs if c not in df.columns]
            if extra:
                return f"Descriptions for non-existent columns: {extra}"

            # Store final result
            capture["column_descriptions"] = col_descs
            return "VALID"

        return execute

    tools = [list_files(), read_file_sample(), execute_python(), submit_table()]
    return tools, capture
