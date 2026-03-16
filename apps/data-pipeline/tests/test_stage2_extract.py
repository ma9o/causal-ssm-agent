"""Tests for stage 2 worker extraction flow helpers."""

import asyncio
import logging
from types import SimpleNamespace

import polars as pl

from causal_ssm_agent.flows.stages import stage2_extract


class _FakeFuture:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


def _run(coro):
    return asyncio.run(coro)


def test_collect_batch_results_logs_completion_order_but_preserves_worker_order(
    monkeypatch, caplog
):
    future0 = _FakeFuture(
        {"dataframe": [{"indicator": "a"}], "n_extractions": 1, "status": "completed"}
    )
    future1 = _FakeFuture(
        {"dataframe": [{"indicator": "b"}], "n_extractions": 2, "status": "completed"}
    )

    def _as_completed(futures):
        assert len(list(futures)) == 2
        return iter([future1, future0])

    monkeypatch.setattr(stage2_extract, "as_completed", _as_completed)
    logger = logging.getLogger("test_stage2_extract")

    with caplog.at_level(logging.INFO, logger=logger.name):
        rows, statuses, n_total, sampled_trace = stage2_extract._collect_batch_results(
            futures=[future0, future1],
            batch_indices=[0, 1],
            batch_n_ticks=[5, 7],
            logger=logger,
            completed_before=5,
            total_chunks=7,
        )

    assert rows == [{"indicator": "a"}, {"indicator": "b"}]
    assert statuses == [
        {"worker_id": 0, "status": "completed", "n_extractions": 1, "n_ticks": 5},
        {"worker_id": 1, "status": "completed", "n_extractions": 2, "n_ticks": 7},
    ]
    assert n_total == 3
    assert sampled_trace is None
    assert "worker 1 completed (progress=6/7" in caplog.text
    assert "worker 0 completed (progress=7/7" in caplog.text


def test_collect_batch_results_records_failures(monkeypatch):
    future0 = _FakeFuture(error=RuntimeError("timeout"))
    future1 = _FakeFuture({"dataframe": [], "n_extractions": 0, "status": "completed"})

    def _as_completed(futures):
        assert len(list(futures)) == 2
        return iter([future0, future1])

    monkeypatch.setattr(stage2_extract, "as_completed", _as_completed)

    rows, statuses, n_total, sampled_trace = stage2_extract._collect_batch_results(
        futures=[future0, future1],
        batch_indices=[0, 1],
        batch_n_ticks=[3, 4],
        logger=logging.getLogger("test_stage2_extract"),
        completed_before=0,
        total_chunks=2,
    )

    assert rows == []
    assert statuses == [
        {
            "worker_id": 0,
            "status": "failed",
            "n_extractions": 0,
            "n_ticks": 3,
            "error": "timeout",
        },
        {"worker_id": 1, "status": "completed", "n_extractions": 0, "n_ticks": 4},
    ]
    assert n_total == 0
    assert sampled_trace is None


def test_extract_tick_chunk_task_uses_stage2_generate_config(monkeypatch, caplog):
    import causal_ssm_agent.utils.causal_spec as causal_spec_mod
    import causal_ssm_agent.utils.config as config_mod
    import causal_ssm_agent.utils.llm as llm_mod
    import causal_ssm_agent.workers.core as worker_core

    logger = logging.getLogger("test_stage2_extract")
    generate_config = SimpleNamespace(max_tokens=1234, reasoning_effort="medium")
    captured: dict[str, object] = {}

    monkeypatch.setattr(stage2_extract, "get_run_logger", lambda: logger)
    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(stage2_workers=SimpleNamespace(model="mock-stage2-model")),
    )
    monkeypatch.setattr(
        causal_spec_mod,
        "get_indicators",
        lambda _causal_spec: [{"name": "indicator_a"}, {"name": "indicator_b"}],
    )
    monkeypatch.setattr(llm_mod, "get_generate_config", lambda: generate_config)

    def fake_make_generate_fn(model_name, config=None, **_kwargs):
        captured["model_name"] = model_name
        captured["generate_config"] = config
        return "mock-generate"

    async def fake_run_worker_extraction(**kwargs):
        captured["worker_kwargs"] = kwargs
        return SimpleNamespace(
            output=SimpleNamespace(extractions=[{"indicator": "indicator_a"}]),
            dataframe=pl.DataFrame(
                [{"indicator": "indicator_a", "value": "1.0", "timestamp": "2024-01-01"}]
            ),
        )

    monkeypatch.setattr(llm_mod, "make_generate_fn", fake_make_generate_fn)
    monkeypatch.setattr(worker_core, "run_worker_extraction", fake_run_worker_extraction)

    tick_text = "## Tick: 2024-01-01\n\n08:00  event1\n09:00  event2"
    tick_ids = ["2024-01-01"]

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = _run(
            stage2_extract.extract_tick_chunk_task.fn(
                tick_text=tick_text,
                tick_ids=tick_ids,
                chunk_idx=3,
                question="Does treatment affect outcome?",
                causal_spec={"measurement": {"model_clock": "1d", "indicators": []}},
            )
        )

    assert result == {
        "dataframe": [
            {
                "indicator": "indicator_a",
                "value": "1.0",
                "timestamp": "2024-01-01",
            }
        ],
        "n_extractions": 1,
        "status": "completed",
    }
    assert captured["model_name"] == "mock-stage2-model"
    assert captured["generate_config"] is generate_config
    worker_kwargs = captured["worker_kwargs"]
    assert isinstance(worker_kwargs, dict)
    assert worker_kwargs["tick_text"] == tick_text
    assert worker_kwargs["tick_ids"] == tick_ids
    assert worker_kwargs["question"] == "Does treatment affect outcome?"
    assert worker_kwargs["causal_spec"] == {"measurement": {"model_clock": "1d", "indicators": []}}
    assert worker_kwargs["generate"] == "mock-generate"
    assert worker_kwargs["logger"] is logger
    assert "max_tokens=1234" in caplog.text
    assert "reasoning_effort=medium" in caplog.text


# ══════════════════════════════════════════════════════════════════════════════
# _project_to_source_columns tests
# ══════════════════════════════════════════════════════════════════════════════


def test_project_keeps_only_source_columns():
    df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "heart_rate": [70, 75],
            "steps": [5000, 8000],
            "mood": ["good", "bad"],
            "irrelevant": [1, 2],
        }
    )
    indicators = [
        {"name": "hr", "source_columns": ["heart_rate", "timestamp"]},
        {"name": "activity", "source_columns": ["steps"]},
    ]
    result = stage2_extract._project_to_source_columns(df, indicators)
    assert set(result.columns) == {"timestamp", "heart_rate", "steps"}


def test_project_no_source_columns_returns_full_df():
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    indicators = [{"name": "x"}]
    result = stage2_extract._project_to_source_columns(df, indicators)
    assert result.columns == df.columns


def test_project_missing_columns_warns(caplog):
    df = pl.DataFrame({"a": [1], "b": [2]})
    indicators = [{"name": "x", "source_columns": ["a", "nonexistent"]}]
    with caplog.at_level(logging.WARNING):
        result = stage2_extract._project_to_source_columns(df, indicators)
    assert "a" in result.columns
    assert "nonexistent" not in result.columns
    assert "nonexistent" in caplog.text
