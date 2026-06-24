"""Tests for the evaluation data plane (fetched-dataset cache)."""

from __future__ import annotations

from pathlib import Path

from evaluation.data.datasets import cached_download


def test_cached_download_is_idempotent(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    cache = tmp_path / "cache"

    first = cached_download(src.as_uri(), "x.txt", cache_dir=cache)
    assert first.read_text() == "hello"
    assert first.parent == cache

    # Second call with a bogus URL must reuse the cached file, not re-fetch.
    second = cached_download("file:///definitely/missing", "x.txt", cache_dir=cache)
    assert second == first
    assert second.read_text() == "hello"


def test_questions_live_in_the_data_plane():
    # The eval questions are data — they live under evaluation/data/questions,
    # not interleaved with the inspect_evals task code.
    data_questions = Path(__file__).resolve().parents[2] / "evaluation" / "data" / "questions"
    assert (data_questions / "1_resolve-errors-faster" / "question.yaml").exists()
