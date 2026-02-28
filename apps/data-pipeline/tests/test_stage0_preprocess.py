"""Tests for Stage 0 preprocess.

Covers: _extract_location, _process_activity, _records_to_lines,
        _sample_records, _compute_date_range, _find_raw_input,
        _parse_json, _parse_takeout_zip.
"""

import json
from datetime import UTC, datetime
from zipfile import ZipFile

import pytest

from causal_ssm_agent.flows.stages.stage0_preprocess import (
    TAKEOUT_ZIP_PATH,
    _compute_date_range,
    _extract_location,
    _find_raw_input,
    _parse_json,
    _parse_takeout_zip,
    _process_activity,
    _records_to_lines,
    _sample_records,
)


def _make_raw_entries(n: int = 30) -> list[dict]:
    """Create synthetic Google Takeout MyActivity entries."""
    entries = []
    for i in range(n):
        dt = datetime(
            2024,
            1 + (i * 6) // n,
            1 + i % 28,
            8 + i % 12,
            tzinfo=UTC,
        )
        entries.append(
            {
                "title": f"Searched for topic {i}",
                "time": dt.isoformat(),
            }
        )
    return entries


# =============================================================================
# _extract_location
# =============================================================================


class TestExtractLocation:
    def test_valid_center_coordinates(self):
        entry = {
            "locationInfos": [
                {"url": "https://maps.google.com?center=40.7128,-74.0060&zoom=14"}
            ]
        }
        assert _extract_location(entry) == "40.7128,-74.0060"

    def test_negative_coordinates(self):
        entry = {
            "locationInfos": [
                {"url": "https://maps.google.com?center=-33.8688,151.2093"}
            ]
        }
        assert _extract_location(entry) == "-33.8688,151.2093"

    def test_no_location_infos_key(self):
        assert _extract_location({}) is None

    def test_empty_location_infos(self):
        assert _extract_location({"locationInfos": []}) is None

    def test_url_without_center_param(self):
        entry = {"locationInfos": [{"url": "https://maps.google.com?q=NYC"}]}
        assert _extract_location(entry) is None

    def test_no_url_key(self):
        entry = {"locationInfos": [{"name": "some place"}]}
        assert _extract_location(entry) is None


# =============================================================================
# _process_activity
# =============================================================================


class TestProcessActivity:
    def test_parses_entries(self):
        entries = _make_raw_entries(5)
        records = _process_activity(entries)
        assert len(records) == 5
        assert all("datetime" in r for r in records)
        assert all("content" in r for r in records)

    def test_sorts_by_datetime(self):
        entries = _make_raw_entries(10)
        records = _process_activity(entries)
        datetimes = [r["datetime"] for r in records]
        assert datetimes == sorted(datetimes)

    def test_skips_entries_without_title_or_time(self):
        entries = [
            {"title": "", "time": "2024-01-01T00:00:00Z"},
            {"title": "Searched for x", "time": ""},
            {"title": "Searched for y", "time": "2024-01-02T00:00:00Z"},
        ]
        records = _process_activity(entries)
        assert len(records) == 1

    def test_classifies_activity_types(self):
        entries = [
            {"title": "Searched for cats", "time": "2024-01-01T00:00:00Z"},
            {"title": "Visited example.com", "time": "2024-01-01T01:00:00Z"},
            {"title": "Viewed a page", "time": "2024-01-01T02:00:00Z"},
            {"title": "Something else", "time": "2024-01-01T03:00:00Z"},
        ]
        records = _process_activity(entries)
        types = [r["activity_type"] for r in records]
        assert types == ["search", "visit", "view", "other"]


class TestSampleRecords:
    def test_returns_n_samples(self):
        records = _process_activity(_make_raw_entries(100))
        sample = _sample_records(records, n=15)
        assert len(sample) == 15

    def test_returns_all_when_fewer_than_n(self):
        records = _process_activity(_make_raw_entries(5))
        sample = _sample_records(records, n=15)
        assert len(sample) == 5

    def test_empty_records(self):
        assert _sample_records([]) == []

    def test_sample_preserves_all_record_keys(self):
        records = _process_activity(_make_raw_entries(20))
        sample = _sample_records(records, n=5)
        for entry in sample:
            assert set(entry.keys()) == set(records[0].keys())

    def test_sample_values_are_strings_or_none(self):
        records = _process_activity(_make_raw_entries(20))
        sample = _sample_records(records, n=5)
        for entry in sample:
            for v in entry.values():
                assert v is None or isinstance(v, str)

    def test_evenly_spaced(self):
        records = _process_activity(_make_raw_entries(100))
        sample = _sample_records(records, n=3)
        assert len(sample) == 3
        # First should be from the beginning, last from the end
        assert sample[0]["datetime"] == records[0]["datetime"].isoformat()
        assert sample[-1]["datetime"] == records[-1]["datetime"].isoformat()


class TestComputeDateRange:
    def test_returns_start_and_end(self):
        records = _process_activity(_make_raw_entries(10))
        dr = _compute_date_range(records)
        assert "start" in dr
        assert "end" in dr
        assert dr["start"] <= dr["end"]

    def test_empty_records(self):
        dr = _compute_date_range([])
        assert dr == {"start": "", "end": ""}

    def test_format_is_iso_date(self):
        records = _process_activity(_make_raw_entries(5))
        dr = _compute_date_range(records)
        # Should parse as date
        datetime.strptime(dr["start"], "%Y-%m-%d")
        datetime.strptime(dr["end"], "%Y-%m-%d")


# =============================================================================
# _records_to_lines
# =============================================================================


class TestRecordsToLines:
    def test_basic_line_format(self):
        records = [
            {
                "datetime": datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
                "activity_type": "search",
                "content": "python",
                "location": None,
            }
        ]
        lines = _records_to_lines(records)
        assert len(lines) == 1
        assert "[search]" in lines[0]
        assert "python" in lines[0]
        assert "2024-01-15" in lines[0]

    def test_includes_location(self):
        records = [
            {
                "datetime": datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
                "activity_type": "visit",
                "content": "coffee shop",
                "location": "40.71,-74.00",
            }
        ]
        lines = _records_to_lines(records)
        assert "@ 40.71,-74.00" in lines[0]
        assert "[visit]" in lines[0]

    def test_omits_location_when_none(self):
        records = [
            {
                "datetime": datetime(2024, 1, 15, tzinfo=UTC),
                "activity_type": "search",
                "content": "test",
                "location": None,
            }
        ]
        lines = _records_to_lines(records)
        assert "@" not in lines[0]

    def test_empty_records(self):
        assert _records_to_lines([]) == []

    def test_multiple_records(self):
        records = [
            {
                "datetime": datetime(2024, 1, 15, 10, tzinfo=UTC),
                "activity_type": "search",
                "content": "a",
                "location": None,
            },
            {
                "datetime": datetime(2024, 1, 15, 11, tzinfo=UTC),
                "activity_type": "visit",
                "content": "b",
                "location": None,
            },
        ]
        lines = _records_to_lines(records)
        assert len(lines) == 2
        assert "[search]" in lines[0]
        assert "[visit]" in lines[1]


# =============================================================================
# _find_raw_input
# =============================================================================


class TestFindRawInput:
    def test_finds_json_file(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        user_dir = tmp_path / "user1"
        user_dir.mkdir()
        (user_dir / "data.json").write_text("[]")
        assert _find_raw_input("user1") == user_dir / "data.json"

    def test_finds_zip_file(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        user_dir = tmp_path / "user1"
        user_dir.mkdir()
        (user_dir / "archive.zip").write_text("fake")
        assert _find_raw_input("user1") == user_dir / "archive.zip"

    def test_prefers_json_over_zip(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        user_dir = tmp_path / "user1"
        user_dir.mkdir()
        (user_dir / "data.json").write_text("[]")
        (user_dir / "archive.zip").write_text("fake")
        result = _find_raw_input("user1")
        assert result.suffix == ".json"

    def test_missing_user_dir(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="No raw data directory"):
            _find_raw_input("nonexistent")

    def test_empty_user_dir(self, tmp_path, monkeypatch):
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        user_dir = tmp_path / "user1"
        user_dir.mkdir()
        with pytest.raises(FileNotFoundError, match=r"No \.json or \.zip files"):
            _find_raw_input("user1")


# =============================================================================
# _parse_json
# =============================================================================


class TestParseJson:
    def test_parses_valid_json(self, tmp_path):
        entries = [
            {"title": "Searched for hello", "time": "2024-01-01T00:00:00Z"},
            {"title": "Visited example.com", "time": "2024-01-02T00:00:00Z"},
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(entries))
        records = _parse_json(f)
        assert len(records) == 2
        assert records[0]["activity_type"] == "search"
        assert records[1]["activity_type"] == "visit"

    def test_empty_json_raises(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("[]")
        with pytest.raises(ValueError, match="Empty JSON"):
            _parse_json(f)


# =============================================================================
# Integration
# =============================================================================


class TestPreprocessRawInputIntegration:
    """Integration test using a synthetic JSON file in tmp_path."""

    def test_full_preprocess(self, tmp_path, monkeypatch):
        """Create a synthetic JSON, monkeypatch RAW_DIR, and verify the result."""
        import causal_ssm_agent.flows.stages.stage0_preprocess as mod

        # Create user dir with a JSON file
        user_dir = tmp_path / "test_user"
        user_dir.mkdir()
        entries = _make_raw_entries(30)
        json_file = user_dir / "MyActivity.json"
        json_file.write_text(json.dumps(entries))

        # Monkeypatch RAW_DIR to point to tmp_path
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)

        # Call the underlying logic (not the Prefect task wrapper)
        raw_path = mod._find_raw_input("test_user")
        records = mod._parse_json(raw_path)
        lines = mod._records_to_lines(records)
        sample = mod._sample_records(records)
        date_range = mod._compute_date_range(records)

        assert len(lines) == 30
        assert len(sample) == 15
        assert date_range["start"]
        assert date_range["end"]

        # Verify the full result shape
        from causal_ssm_agent.flows.stages.stage0_preprocess import PreprocessResult

        result = PreprocessResult(
            lines=lines,
            n_records=len(records),
            date_range=date_range,
            sample=sample,
        )
        assert result["n_records"] == 30
        assert len(result["sample"]) == 15
        assert len(result["lines"]) == 30


# =============================================================================
# _parse_takeout_zip
# =============================================================================


def _create_takeout_zip(zip_path, entries):
    """Create a synthetic Google Takeout zip archive."""
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(TAKEOUT_ZIP_PATH, json.dumps(entries))


class TestParseTakeoutZip:
    def test_parses_valid_zip(self, tmp_path):
        entries = _make_raw_entries(10)
        zip_path = tmp_path / "takeout.zip"
        _create_takeout_zip(zip_path, entries)

        records = _parse_takeout_zip(zip_path)
        assert len(records) == 10
        assert all("datetime" in r for r in records)
        assert all("activity_type" in r for r in records)

    def test_sorts_results_by_datetime(self, tmp_path):
        entries = _make_raw_entries(20)
        zip_path = tmp_path / "takeout.zip"
        _create_takeout_zip(zip_path, entries)

        records = _parse_takeout_zip(zip_path)
        datetimes = [r["datetime"] for r in records]
        assert datetimes == sorted(datetimes)

    def test_not_a_zip_raises(self, tmp_path):
        fake = tmp_path / "not_a_zip.zip"
        fake.write_text("this is not a zip file")
        with pytest.raises(ValueError, match="not a valid zip"):
            _parse_takeout_zip(fake)

    def test_missing_expected_path_raises(self, tmp_path):
        zip_path = tmp_path / "takeout.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("wrong/path.json", "[]")
        with pytest.raises(ValueError, match="not found in archive"):
            _parse_takeout_zip(zip_path)

    def test_empty_json_in_zip_raises(self, tmp_path):
        zip_path = tmp_path / "takeout.zip"
        _create_takeout_zip(zip_path, [])
        with pytest.raises(ValueError, match="Empty JSON"):
            _parse_takeout_zip(zip_path)
