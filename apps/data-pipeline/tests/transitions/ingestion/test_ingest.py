"""Tests for shared ingestion staging helpers."""

import os
import zipfile

import pytest

from nof1_causal_lab.flows.transitions.ingestion.tools import _safe_resolve


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

    def test_sibling_prefix_traversal_blocked(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        sibling = tmp_path / "base_evil"
        sibling.write_text("outside")

        with pytest.raises(ValueError, match="Path traversal blocked"):
            _safe_resolve(base, "../base_evil")


class TestFindRawInput:
    def test_finds_most_recent_text_file_regardless_of_extension(self, tmp_path, monkeypatch):
        import nof1_causal_lab.flows.transitions.ingestion.flow as mod
        from nof1_causal_lab.flows.transitions.ingestion.flow import _find_raw_input

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
        import nof1_causal_lab.flows.transitions.ingestion.flow as mod
        from nof1_causal_lab.flows.transitions.ingestion.flow import _find_raw_input

        workspace_dir = tmp_path / "empty_workspace"
        workspace_dir.mkdir()

        monkeypatch.setattr(mod, "input_dir", lambda workspace_id: str(tmp_path / workspace_id))
        with pytest.raises(FileNotFoundError):
            _find_raw_input("empty_workspace")


class TestPrepareRawInput:
    def test_extracts_zip_archives(self, tmp_path):
        from nof1_causal_lab.flows.transitions.ingestion.flow import _prepare_raw_input

        raw_zip = tmp_path / "input.zip"
        with zipfile.ZipFile(raw_zip, "w") as zf:
            zf.writestr("nested/data.csv", "date,value\n2024-01-01,1\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_zip, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "nested" / "data.csv").read_text() == "date,value\n2024-01-01,1\n"

    def test_copies_non_archive_files(self, tmp_path):
        from nof1_causal_lab.flows.transitions.ingestion.flow import _prepare_raw_input

        raw_text = tmp_path / "input.txt"
        raw_text.write_text("line one\nline two\n")

        prepared_dir = tmp_path / "prepared"
        result = _prepare_raw_input(raw_text, prepared_dir)

        assert result == prepared_dir
        assert (prepared_dir / "input.txt").read_text() == "line one\nline two\n"
