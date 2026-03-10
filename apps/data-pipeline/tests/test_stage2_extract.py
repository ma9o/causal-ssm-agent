"""Tests for stage 2 worker extraction flow helpers."""

import logging

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


def _chunk(value: str) -> pl.DataFrame:
    return pl.DataFrame({"value": [value]})


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
        rows, statuses, n_total = stage2_extract._collect_batch_results(
            futures=[future0, future1],
            batch_indices=[0, 1],
            batch_chunks=[_chunk("first"), _chunk("second")],
            logger=logger,
            completed_before=5,
            total_chunks=7,
        )

    assert rows == [{"indicator": "a"}, {"indicator": "b"}]
    assert statuses == [
        {"worker_id": 0, "status": "completed", "n_extractions": 1, "chunk_size": 1},
        {"worker_id": 1, "status": "completed", "n_extractions": 2, "chunk_size": 1},
    ]
    assert n_total == 3
    assert "worker 1 completed (progress=6/7" in caplog.text
    assert "worker 0 completed (progress=7/7" in caplog.text


def test_collect_batch_results_records_failures(monkeypatch):
    future0 = _FakeFuture(error=RuntimeError("timeout"))
    future1 = _FakeFuture({"dataframe": [], "n_extractions": 0, "status": "completed"})

    def _as_completed(futures):
        assert len(list(futures)) == 2
        return iter([future0, future1])

    monkeypatch.setattr(stage2_extract, "as_completed", _as_completed)

    rows, statuses, n_total = stage2_extract._collect_batch_results(
        futures=[future0, future1],
        batch_indices=[0, 1],
        batch_chunks=[_chunk("first"), _chunk("second")],
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
            "chunk_size": 1,
            "error": "timeout",
        },
        {"worker_id": 1, "status": "completed", "n_extractions": 0, "chunk_size": 1},
    ]
    assert n_total == 0
