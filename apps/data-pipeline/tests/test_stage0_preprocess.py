"""Tests for Stage 0 agentic ingestion tools and helpers."""

import asyncio
import json
import zipfile

import polars as pl
import pytest

from causal_ssm_agent.flows.stages.stage0_tools import (
    _format_df_preview,
    _safe_resolve,
    make_ingestion_tools,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


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


class TestFormatDfPreview:
    def test_basic_preview(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        preview = _format_df_preview(df, max_rows=2)
        assert "3 rows x 2 columns" in preview
        assert "a:" in preview
        assert "b:" in preview


class TestIngestionTools:
    """Test the individual tools returned by make_ingestion_tools."""

    @pytest.fixture()
    def sample_archive(self, tmp_path):
        """Create a temp directory simulating an extracted zip with a CSV."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("date,value,category\n2024-01-01,1.5,A\n2024-01-02,2.3,B\n")
        json_file = tmp_path / "meta.json"
        json_file.write_text(json.dumps({"source": "test"}))
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("hello\nworld\n")
        return tmp_path

    def test_list_files(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        list_tool = tools[0]
        result = _run(list_tool(path="."))
        assert "data.csv" in result
        assert "meta.json" in result
        assert "subdir/" in result

    def test_read_file_sample(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        read_tool = tools[1]
        result = _run(read_tool(path="data.csv", n_lines=10))
        assert "date,value,category" in result
        assert "2024-01-01" in result

    def test_read_file_traversal_blocked(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        read_tool = tools[1]
        result = _run(read_tool(path="../../../etc/passwd"))
        assert "Path traversal blocked" in result

    def test_execute_python_success(self, sample_archive):
        tools, capture = make_ingestion_tools(sample_archive)
        exec_tool = tools[2]
        code = 'result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")'
        result = _run(exec_tool(code=code))
        assert "Success" in result
        assert "dataframe" in capture
        assert isinstance(capture["dataframe"], pl.DataFrame)
        assert len(capture["dataframe"]) == 2

    def test_execute_python_no_result_df(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        exec_tool = tools[2]
        result = _run(exec_tool(code="x = 42"))
        assert "No `result_df` variable found" in result

    def test_execute_python_error(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        exec_tool = tools[2]
        result = _run(exec_tool(code="1/0"))
        assert "Execution error" in result
        assert "ZeroDivisionError" in result

    def test_submit_table_success(self, sample_archive):
        tools, capture = make_ingestion_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(exec_tool(
            code='result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")'
        ))

        result = _run(submit_tool(
            source_label="Test data",
            column_descriptions_json=json.dumps({
                "date": "Date of observation",
                "value": "Numeric value",
                "category": "Category label",
            }),
        ))
        assert result == "VALID"
        assert capture["source_label"] == "Test data"
        assert "date" in capture["column_descriptions"]

    def test_submit_table_missing_descriptions(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        exec_tool = tools[2]
        submit_tool = tools[3]

        _run(exec_tool(
            code='result_df = pl.read_csv(Path(DATA_DIR) / "data.csv")'
        ))

        result = _run(submit_tool(
            source_label="Test",
            column_descriptions_json=json.dumps({"date": "Date"}),
        ))
        assert "Missing descriptions" in result

    def test_submit_table_no_dataframe(self, sample_archive):
        tools, _ = make_ingestion_tools(sample_archive)
        submit_tool = tools[3]
        result = _run(submit_tool(
            source_label="Test",
            column_descriptions_json="{}",
        ))
        assert "No DataFrame available" in result


class TestMapColumnsToIndicators:
    """Test the column-to-indicator mapping function."""

    def test_basic_mapping(self):
        from causal_ssm_agent.flows.pipeline import map_columns_to_indicators

        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ldl_cholesterol": [4.1, 3.8, 4.0],
            "systolic_bp": [120.0, 118.0, 122.0],
            "irrelevant_col": ["a", "b", "c"],
        })
        causal_spec = {
            "measurement": {
                "indicators": [
                    {"name": "ldl_cholesterol", "construct_name": "cardiovascular"},
                    {"name": "systolic_bp", "construct_name": "cardiovascular"},
                ]
            }
        }
        result = map_columns_to_indicators(df, causal_spec)
        assert set(result.columns) == {"indicator", "value", "timestamp"}
        assert result["indicator"].n_unique() == 2
        assert len(result) == 6  # 3 rows x 2 indicators

    def test_no_matching_columns_raises(self):
        from causal_ssm_agent.flows.pipeline import map_columns_to_indicators

        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        causal_spec = {
            "measurement": {
                "indicators": [{"name": "nonexistent", "construct_name": "c"}]
            }
        }
        with pytest.raises(ValueError, match="No indicator columns found"):
            map_columns_to_indicators(df, causal_spec)

    def test_detects_timestamp_column(self):
        from causal_ssm_agent.flows.pipeline import map_columns_to_indicators

        df = pl.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "hr": [72.0, 75.0],
        })
        causal_spec = {
            "measurement": {
                "indicators": [{"name": "hr", "construct_name": "health"}]
            }
        }
        result = map_columns_to_indicators(df, causal_spec)
        assert "timestamp" in result.columns
        assert result["timestamp"][0] == "2024-01-01"


class TestFindRawInput:
    def test_finds_zip(self, tmp_path):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod
        from causal_ssm_agent.flows.stages.stage0_preprocess import _find_raw_input

        user_dir = tmp_path / "test_user"
        user_dir.mkdir()
        zip_path = user_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "hello")

        # Monkeypatch RAW_DIR
        original = mod.RAW_DIR
        mod.RAW_DIR = tmp_path
        try:
            result = _find_raw_input("test_user")
            assert result.name == "data.zip"
        finally:
            mod.RAW_DIR = original

    def test_no_files_raises(self, tmp_path):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod
        from causal_ssm_agent.flows.stages.stage0_preprocess import _find_raw_input

        user_dir = tmp_path / "empty_user"
        user_dir.mkdir()

        original = mod.RAW_DIR
        mod.RAW_DIR = tmp_path
        try:
            with pytest.raises(FileNotFoundError):
                _find_raw_input("empty_user")
        finally:
            mod.RAW_DIR = original
