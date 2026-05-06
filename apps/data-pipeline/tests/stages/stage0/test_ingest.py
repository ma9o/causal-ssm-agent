"""Tests for Stage 0 agentic ingestion tools and helpers."""

import asyncio
import csv
import datetime
import io
import json
import math
import os
import re
import traceback
import zipfile
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from causal_ssm_agent.flows.stages.stage0.tools import (
    _safe_resolve,
    make_ingestion_tools,
)

_MOCK_SANDBOX_EXECUTION_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    NameError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Mock sandbox that runs code locally (mirrors the real sandbox runner logic
# but avoids the Modal dependency in unit tests).
# ---------------------------------------------------------------------------


class _MockSandbox:
    """Local exec()-based sandbox for unit tests."""

    def __init__(self, extract_dir: Path):
        self._extract_dir = extract_dir

    def execute(self, code: str) -> tuple[str, pl.DataFrame | None]:
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
            "DATA_DIR": str(self._extract_dir),
        }
        try:
            exec(code, ns)
        except _MOCK_SANDBOX_EXECUTION_ERRORS:
            return f"Execution error:\n{traceback.format_exc()}", None

        result_df = ns.get("result_df")
        if result_df is None:
            available = [
                k
                for k in ns
                if not k.startswith("_")
                and k
                not in (
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
                )
            ]
            return (
                f"No `result_df` variable found after execution.\n"
                f"Variables defined: {available}\n"
                "Assign your final DataFrame to `result_df`."
            ), None

        if not isinstance(result_df, pl.DataFrame):
            return (
                f"`result_df` is {type(result_df).__name__}, not a Polars DataFrame.\n"
                "Use pl.DataFrame(...) or pl.read_csv(...) to create one."
            ), None

        if result_df.is_empty():
            return "Warning: `result_df` is empty (0 rows). Check your parsing logic.", None

        preview = f"Shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns"
        return f"Success!\n\n{preview}", result_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSafeResolve:
    def test_normal_path(self, tmp_path):
        child = tmp_path / "data.csv"
        child.touch()
        assert _safe_resolve(tmp_path, "data.csv") == child.resolve()

    def test_nested_path(self, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        child = nested / "data.csv"
        child.touch()
        assert _safe_resolve(tmp_path, "sub/data.csv") == child.resolve()

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal blocked"):
            _safe_resolve(tmp_path, "../../../etc/passwd")


class TestIngestionTools:
    """Test the individual tools returned by make_ingestion_tools."""

    @pytest.fixture
    def sample_archive(self, tmp_path):
        """Create a temp directory simulating an extracted zip with a CSV."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("timestamp,value,category\n2024-01-01,1.5,A\n2024-01-02,2.3,B\n")
        json_file = tmp_path / "meta.json"
        json_file.write_text(json.dumps({"source": "test"}))
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("hello\nworld\n")
        return tmp_path

    def _make_tools(self, extract_dir):
        sandbox = _MockSandbox(extract_dir)
        return make_ingestion_tools(extract_dir, cast("Any", sandbox))

    def test_list_files(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        list_tool = tools[0]
        result = _run(list_tool(path="."))
        assert "data.csv" in result
        assert "meta.json" in result
        assert "subdir/" in result

    def test_read_file_sample(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        read_tool = tools[1]
        result = _run(read_tool(path="data.csv", n_lines=10))
        assert "timestamp,value,category" in result
        assert "2024-01-01" in result

    def test_read_file_traversal_blocked(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        read_tool = tools[1]
        result = _run(read_tool(path="../../../etc/passwd"))
        assert "Path traversal blocked" in result

    def test_execute_python_success(self, sample_archive):
        tools, capture = self._make_tools(sample_archive)
        exec_tool = tools[2]
        code = (
            'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")\n'
            'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
        )
        result = _run(exec_tool(code=code))
        assert "Success" in result
        assert "dataframe" in capture
        assert isinstance(capture["dataframe"], pl.DataFrame)
        assert len(capture["dataframe"]) == 2

    def test_execute_python_no_result_df(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        exec_tool = tools[2]
        result = _run(exec_tool(code="x = 42"))
        assert "No `result_df` variable found" in result

    def test_execute_python_error(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        exec_tool = tools[2]
        result = _run(exec_tool(code="1/0"))
        assert "Execution error" in result
        assert "ZeroDivisionError" in result

    def test_submit_table_success(self, sample_archive):
        tools, capture = self._make_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(
            exec_tool(
                code=(
                    'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")\n'
                    'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
                )
            )
        )

        result = _run(
            submit_tool(
                column_descriptions_json=json.dumps(
                    {
                        "timestamp": "Timestamp of observation",
                        "value": "Numeric value",
                        "category": "Category label",
                    }
                ),
            )
        )
        assert result == "VALID"
        assert "timestamp" in capture["column_descriptions"]

    def test_submit_table_missing_timestamp_column(self, sample_archive):
        """submit_table rejects DataFrames without a 'timestamp' column."""
        tools, _ = self._make_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(exec_tool(code='result_df = pl.DataFrame({"id": [1], "value": [1.5]})'))

        result = _run(
            submit_tool(
                column_descriptions_json=json.dumps({"id": "Row id", "value": "Numeric value"}),
            )
        )
        assert "timestamp" in result.lower()

    def test_submit_table_rejects_string_timestamp(self, sample_archive):
        """submit_table rejects a 'timestamp' column that is not Datetime-typed."""
        tools, _ = self._make_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(exec_tool(code='result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")'))

        result = _run(
            submit_tool(
                column_descriptions_json=json.dumps(
                    {
                        "timestamp": "Timestamp",
                        "value": "Numeric value",
                        "category": "Category label",
                    }
                ),
            )
        )
        assert "Datetime" in result or "must be" in result.lower()

    def test_submit_table_missing_descriptions(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(
            exec_tool(
                code=(
                    'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")\n'
                    'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
                )
            )
        )

        result = _run(
            submit_tool(
                column_descriptions_json=json.dumps({"timestamp": "Timestamp"}),
            )
        )
        assert "Missing descriptions" in result

    def test_submit_table_no_dataframe(self, sample_archive):
        tools, _ = self._make_tools(sample_archive)
        submit_tool = tools[3]
        result = _run(
            submit_tool(
                column_descriptions_json="{}",
            )
        )
        assert "No DataFrame available" in result


class TestFindRawInput:
    def test_finds_most_recent_text_file_regardless_of_extension(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0.flow as mod
        from causal_ssm_agent.flows.stages.stage0.flow import _find_raw_input

        workspace_dir = tmp_path / "test_workspace"
        workspace_dir.mkdir()
        older = workspace_dir / "data.zip"
        newer = workspace_dir / "notes.txt"

        with zipfile.ZipFile(older, "w") as zf:
            zf.writestr("test.txt", "hello")
        newer.write_text("screen time, sleep quality\n")

        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_700_000_100, 1_700_000_100))

        monkeypatch.setattr(mod, "input_dir", lambda workspace_id: str(tmp_path / workspace_id))
        result = _find_raw_input("test_workspace")
        assert result.endswith("/notes.txt")

    def test_no_files_raises(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0.flow as mod
        from causal_ssm_agent.flows.stages.stage0.flow import _find_raw_input

        workspace_dir = tmp_path / "empty_workspace"
        workspace_dir.mkdir()

        monkeypatch.setattr(mod, "input_dir", lambda workspace_id: str(tmp_path / workspace_id))
        with pytest.raises(FileNotFoundError):
            _find_raw_input("empty_workspace")


class TestPrepareRawInput:
    def test_extracts_zip_archives(self, tmp_path):
        from causal_ssm_agent.flows.stages.stage0.flow import _prepare_raw_input

        raw_zip = tmp_path / "input.zip"
        with zipfile.ZipFile(raw_zip, "w") as zf:
            zf.writestr("nested/data.csv", "date,value\n2024-01-01,1\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_zip, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "nested" / "data.csv").read_text() == "date,value\n2024-01-01,1\n"

    def test_copies_non_archive_files(self, tmp_path):
        from causal_ssm_agent.flows.stages.stage0.flow import _prepare_raw_input

        raw_text = tmp_path / "input.txt"
        raw_text.write_text("line one\nline two\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_text, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "input.txt").read_text() == "line one\nline two\n"


class _MockSandboxContext:
    def __init__(self, extract_dir: Path, **_kwargs):
        self._sandbox = _MockSandbox(extract_dir)

    def __enter__(self):
        return self._sandbox

    def __exit__(self, exc_type, exc, tb):
        return None


class TestRunAgenticIngestion:
    def test_returns_finalize_metadata_when_submit_table_is_called(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0.flow as mod

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("timestamp,value,category\n2024-01-01,1.5,A\n2024-01-02,2.3,B\n")

        monkeypatch.setattr(mod, "ModalCodeSandbox", _MockSandboxContext)

        calls: list[list[str]] = []

        async def handler(tools, _user_message):
            calls.append([tool.name for tool in tools])
            tool_map = {tool.name: tool for tool in tools}
            await tool_map["execute_python"](
                code=(
                    'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")\n'
                    'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
                )
            )
            await tool_map["submit_table"](
                column_descriptions_json=json.dumps(
                    {
                        "timestamp": "Timestamp of observation",
                        "value": "Observed numeric value",
                        "category": "Category label",
                    }
                ),
            )
            return ""

        from tests.helpers import make_session_factory_from_handler

        factory = make_session_factory_from_handler(handler)
        result = _run(mod.run_agentic_ingestion(tmp_path, factory))

        assert calls == [["list_files", "read_file_sample", "execute_python", "submit_table"]]
        assert result.column_descriptions == {
            "timestamp": "Timestamp of observation",
            "value": "Observed numeric value",
            "category": "Category label",
        }

    def test_returns_dataframe_when_agent_never_finalizes(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0.flow as mod

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("timestamp,value\n2024-01-01,1.5\n")

        monkeypatch.setattr(mod, "ModalCodeSandbox", _MockSandboxContext)

        async def handler(tools, _user_message):
            tool_map = {tool.name: tool for tool in tools}
            if "execute_python" in tool_map:
                await tool_map["execute_python"](
                    code=(
                        'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")\n'
                        'result_df = result_df.with_columns(pl.col("timestamp").str.to_datetime())'
                    )
                )
            return ""

        from tests.helpers import make_session_factory_from_handler

        factory = make_session_factory_from_handler(handler)
        result = _run(mod.run_agentic_ingestion(tmp_path, factory))

        assert result.dataframe.shape == (1, 2)
        assert result.column_descriptions == {}
