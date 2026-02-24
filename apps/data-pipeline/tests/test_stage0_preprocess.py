"""Tests for Stage 0 preprocess."""

import json
from datetime import UTC, datetime

from causal_ssm_agent.flows.stages.stage0_preprocess import (
    _compute_date_range,
    _process_activity,
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
            source_type="google-takeout-my-activity",
            source_label="Google Takeout \u2014 My Activity",
            n_records=len(records),
            date_range=date_range,
            sample=sample,
        )
        assert result["source_type"] == "google-takeout-my-activity"
        assert result["n_records"] == 30
        assert len(result["sample"]) == 15
        assert len(result["lines"]) == 30
