"""Tests for stage 2 worker extraction flow helpers."""

import logging
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from nof1_causal_lab.flows.stages.stage2 import flow as stage2_extract
from tests.helpers import run_async as _run


class _FakeFuture:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


def _require_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _require_list(value: object) -> list[Any]:
    assert isinstance(value, list)
    return value


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
            batch_n_windows=[5, 7],
            logger=logger,
            completed_before=5,
            total_chunks=7,
        )

    assert rows == [{"indicator": "a"}, {"indicator": "b"}]
    assert statuses == [
        {"worker_id": 0, "status": "completed", "n_extractions": 1, "n_windows": 5},
        {"worker_id": 1, "status": "completed", "n_extractions": 2, "n_windows": 7},
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
        batch_n_windows=[3, 4],
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
            "n_windows": 3,
            "error": "timeout",
        },
        {"worker_id": 1, "status": "completed", "n_extractions": 0, "n_windows": 4},
    ]
    assert n_total == 0
    assert sampled_trace is None


def test_extract_window_chunk_task_uses_stage2_generate_config(monkeypatch, caplog):
    import nof1_causal_lab.utils.causal_spec as causal_spec_mod
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.workers.core as worker_core
    from nof1_causal_lab.utils.agent_session import StageSessionFactory
    from nof1_causal_lab.utils.config import (
        ClaudeCodeDefaults,
        CodexDefaults,
        EmbeddedLLMDefaults,
        LLMDefaults,
        StageLLMConfig,
    )

    logger = logging.getLogger("test_stage2_extract")
    captured: dict[str, object] = {}

    stage2_llm = StageLLMConfig(harness="none", model="openrouter/mock-stage2-model")
    llm_defaults = LLMDefaults(
        embedded=EmbeddedLLMDefaults(max_tokens=1234, timeout=900, reasoning_effort="medium"),
        claude_code=ClaudeCodeDefaults(),
        codex=CodexDefaults(),
    )

    monkeypatch.setattr(stage2_extract, "get_run_logger", lambda: logger)
    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                llm=stage2_llm,
                max_tool_turns=55,
                worker_timeout=120,
            ),
            llm=llm_defaults,
        ),
    )
    monkeypatch.setattr(
        causal_spec_mod,
        "get_indicators",
        lambda _causal_spec: [{"name": "indicator_a"}, {"name": "indicator_b"}],
    )

    async def fake_run_worker_extraction(**kwargs):
        captured["worker_kwargs"] = kwargs
        return SimpleNamespace(
            output=SimpleNamespace(extractions=[{"indicator": "indicator_a"}]),
            dataframe=pl.DataFrame(
                [{"indicator": "indicator_a", "value": "1.0", "timestamp": "2024-01-01"}]
            ),
        )

    monkeypatch.setattr(worker_core, "run_worker_extraction", fake_run_worker_extraction)

    window_text = "## Window Start: 2024-01-01\n\n08:00  event1\n09:00  event2"
    window_starts = ["2024-01-01"]

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = _run(
            stage2_extract.extract_window_chunk_task.fn(
                window_text=window_text,
                window_starts=window_starts,
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
    worker_kwargs = _require_mapping(captured["worker_kwargs"])
    assert worker_kwargs["window_text"] == window_text
    assert worker_kwargs["window_starts"] == window_starts
    assert worker_kwargs["question"] == "Does treatment affect outcome?"
    assert worker_kwargs["causal_spec"] == {"measurement": {"model_clock": "1d", "indicators": []}}
    assert worker_kwargs["logger"] is logger
    factory = worker_kwargs["session_factory"]
    assert isinstance(factory, StageSessionFactory)
    # Worker timeout is applied as an override on the stage_llm inside the factory.
    assert factory._stage_llm.model == "openrouter/mock-stage2-model"
    assert factory._stage_llm.timeout == 120
    assert factory._max_tool_turns == 55
    assert "timeout=120s" in caplog.text
    assert "mock-stage2-model" in caplog.text


def test_extract_window_chunk_task_emits_running_stage2_worker_and_snapshot_events(monkeypatch):
    import nof1_causal_lab.utils.causal_spec as causal_spec_mod
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.workers.core as worker_core
    from nof1_causal_lab.utils.config import (
        ClaudeCodeDefaults,
        CodexDefaults,
        EmbeddedLLMDefaults,
        LLMDefaults,
        StageLLMConfig,
    )

    worker_events: list[dict[str, object]] = []
    snapshot_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        stage2_extract, "get_run_logger", lambda: logging.getLogger("stage2-events")
    )
    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                llm=StageLLMConfig(harness="none", model="openrouter/mock-stage2-model"),
                max_tool_turns=40,
                worker_timeout=120,
            ),
            llm=LLMDefaults(
                embedded=EmbeddedLLMDefaults(),
                claude_code=ClaudeCodeDefaults(),
                codex=CodexDefaults(),
            ),
        ),
    )
    monkeypatch.setattr(
        causal_spec_mod,
        "get_indicators",
        lambda _causal_spec: [{"name": "indicator_a"}],
    )

    async def fake_run_worker_extraction(**_kwargs):
        return SimpleNamespace(
            output=SimpleNamespace(extractions=[]),
            dataframe=pl.DataFrame([]),
        )

    monkeypatch.setattr(worker_core, "run_worker_extraction", fake_run_worker_extraction)
    monkeypatch.setattr(
        stage2_extract,
        "emit_stage2_worker_event",
        lambda resource_run_id, **payload: worker_events.append(
            {"resource_run_id": resource_run_id, **payload}
        ),
    )
    monkeypatch.setattr(
        stage2_extract,
        "_get_stage2_progress_tracker",
        lambda _root_run_id: SimpleNamespace(
            mark_running=lambda _worker_id: {
                "total_workers": 8,
                "pending_workers": 7,
                "running_workers": 1,
                "completed_workers": 0,
                "failed_workers": 0,
            }
        ),
    )
    monkeypatch.setattr(
        stage2_extract,
        "_emit_stage2_snapshot",
        lambda resource_run_id, snapshot: snapshot_events.append(
            {"resource_run_id": resource_run_id, **snapshot}
        ),
    )

    _run(
        stage2_extract.extract_window_chunk_task.fn(
            window_text="## Window Start: 2024-01-01",
            window_starts=["2024-01-01"],
            chunk_idx=7,
            question="Q",
            causal_spec={"measurement": {"model_clock": "1d", "indicators": []}},
            root_run_id="root-123",
        )
    )

    assert worker_events == [
        {
            "resource_run_id": "root-123",
            "worker_id": 7,
            "state": "running",
            "n_windows": 1,
        }
    ]
    assert snapshot_events == [
        {
            "resource_run_id": "root-123",
            "total_workers": 8,
            "pending_workers": 7,
            "running_workers": 1,
            "completed_workers": 0,
            "failed_workers": 0,
        }
    ]


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


def test_project_missing_columns_warns(caplog):
    df = pl.DataFrame({"a": [1], "b": [2]})
    indicators = [{"name": "x", "source_columns": ["a", "nonexistent"]}]
    with caplog.at_level(logging.WARNING):
        result = stage2_extract._project_to_source_columns(df, indicators)
    assert "a" in result.columns
    assert "nonexistent" not in result.columns
    assert "nonexistent" in caplog.text


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_stage2_extraction_core_accepts_injected_semantic_chunk_runner(
    monkeypatch,
):
    import nof1_causal_lab.utils.data as data_mod
    import nof1_causal_lab.workers.windows as windows_mod

    raw_df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01T08:00:00", "2024-01-15T08:00:00"],
            "stress_score": [4.0, 5.0],
            "sleep_hours": [7.0, 6.0],
        }
    )

    bucket_windows: list[str] = []

    def fake_bucket_by_clock(df: pl.DataFrame, model_clock: str, time_col: str):
        bucket_windows.append(model_clock)
        return [(f"{model_clock}-window", df)]

    monkeypatch.setattr(data_mod, "bucket_by_clock", fake_bucket_by_clock)
    monkeypatch.setattr(windows_mod, "chunk_windows", lambda ticks, _chunk_size: [ticks])
    monkeypatch.setattr(
        windows_mod,
        "format_window_chunk",
        lambda chunk, _time_col, _display_cols, _max_events: f"chunk:{chunk[0][0]}",
    )

    captured_runner: dict[str, object] = {}

    async def fake_semantic_chunk_runner(**kwargs):
        captured_runner.update(kwargs)
        return [], [], 0, None

    causal_spec = {
        "latent": {"constructs": [], "edges": []},
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average stress score in the window",
                    "aggregation": "mean",
                    "source_columns": ["timestamp", "stress_score"],
                },
                {
                    "name": "monthly_sleep_hours",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average sleep hours over the last month",
                    "aggregation": "mean",
                    "observation_window": "1mo",
                    "source_columns": ["timestamp", "sleep_hours"],
                },
            ],
        },
    }

    result = _run(
        stage2_extract.run_stage2_extraction_core(
            raw_df=raw_df,
            question="Does stress affect sleep?",
            causal_spec=causal_spec,
            stage2_workers=SimpleNamespace(
                windows_per_chunk=8,
                max_events_per_window=50,
                max_concurrent_workers=2,
                max_rpm=0,
            ),
            semantic_chunk_runner=fake_semantic_chunk_runner,
        )
    )

    assert bucket_windows == ["1d", "1mo"]
    chunk_contexts = _require_list(captured_runner["chunk_contexts"])
    assert [
        _require_mapping(_require_mapping(ctx)["measurement"])["indicators"][0]["name"]
        for ctx in chunk_contexts
    ] == [
        "stress_score",
        "monthly_sleep_hours",
    ]
    assert captured_runner["chunk_texts"] == ["chunk:1d-window", "chunk:1mo-window"]
    assert result["observation_rows"] == []


def test_run_semantic_chunks_prefect_emits_stage2_plan_worker_and_snapshot_events(monkeypatch):
    plan_events: list[dict[str, object]] = []
    worker_events: list[dict[str, object]] = []
    snapshot_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        stage2_extract,
        "emit_stage2_plan_event",
        lambda resource_run_id, **payload: plan_events.append(
            {"resource_run_id": resource_run_id, **payload}
        ),
    )
    monkeypatch.setattr(
        stage2_extract,
        "emit_stage2_worker_event",
        lambda resource_run_id, **payload: worker_events.append(
            {"resource_run_id": resource_run_id, **payload}
        ),
    )
    monkeypatch.setattr(
        stage2_extract,
        "_emit_stage2_snapshot",
        lambda resource_run_id, snapshot: snapshot_events.append(
            {"resource_run_id": resource_run_id, **snapshot}
        ),
    )

    future_ok = _FakeFuture(
        {"dataframe": [{"indicator": "a"}], "n_extractions": 2, "status": "completed"}
    )
    future_fail = _FakeFuture(error=RuntimeError("timeout"))

    def fake_map(chunk_texts, **_kwargs):
        assert chunk_texts == ["chunk-0", "chunk-1"]
        return [future_ok, future_fail]

    monkeypatch.setattr(stage2_extract.extract_window_chunk_task, "map", fake_map)
    monkeypatch.setattr(stage2_extract, "as_completed", lambda futures: iter(futures))

    rows, statuses, n_total, sampled_trace = _run(
        stage2_extract._run_semantic_chunks_prefect(
            chunk_texts=["chunk-0", "chunk-1"],
            chunk_window_starts=[["2024-01-01"], ["2024-01-02"]],
            chunk_contexts=[{"measurement": {}}, {"measurement": {}}],
            question="Q",
            root_run_id="root-456",
            openrouter_api_key=None,
            max_concurrent_workers=6,
            max_rpm=450,
        )
    )

    assert plan_events == [
        {
            "resource_run_id": "root-456",
            "total_workers": 2,
            "max_concurrent_workers": 6,
            "max_rpm": 450,
        }
    ]
    assert snapshot_events == [
        {
            "resource_run_id": "root-456",
            "total_workers": 2,
            "pending_workers": 2,
            "running_workers": 0,
            "completed_workers": 0,
            "failed_workers": 0,
        },
        {
            "resource_run_id": "root-456",
            "total_workers": 2,
            "pending_workers": 1,
            "running_workers": 0,
            "completed_workers": 1,
            "failed_workers": 0,
        },
        {
            "resource_run_id": "root-456",
            "total_workers": 2,
            "pending_workers": 0,
            "running_workers": 0,
            "completed_workers": 1,
            "failed_workers": 1,
        },
    ]
    assert worker_events == [
        {
            "resource_run_id": "root-456",
            "worker_id": 0,
            "state": "completed",
            "n_windows": 1,
            "n_extractions": 2,
            "n_llm_calls": None,
        },
        {
            "resource_run_id": "root-456",
            "worker_id": 1,
            "state": "failed",
            "n_windows": 1,
            "error": "timeout",
        },
    ]
    assert rows == [{"indicator": "a"}]
    assert statuses == [
        {"worker_id": 0, "status": "completed", "n_extractions": 2, "n_windows": 1},
        {
            "worker_id": 1,
            "status": "failed",
            "n_extractions": 0,
            "n_windows": 1,
            "error": "timeout",
        },
    ]
    assert n_total == 2
    assert sampled_trace is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stage2_extraction_flow_buckets_semantic_indicators_by_observation_window(
    monkeypatch, tmp_path
):
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.utils.data as data_mod
    import nof1_causal_lab.workers.windows as windows_mod

    raw_df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01T08:00:00", "2024-01-15T08:00:00"],
            "stress_score": [4.0, 5.0],
            "sleep_hours": [7.0, 6.0],
        }
    )
    raw_path = tmp_path / "input.parquet"
    raw_df.write_parquet(raw_path)

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                windows_per_chunk=8,
                max_events_per_window=50,
                max_concurrent_workers=2,
                max_rpm=0,
            )
        ),
    )

    bucket_windows: list[str] = []

    def fake_bucket_by_clock(df: pl.DataFrame, model_clock: str, time_col: str):
        bucket_windows.append(model_clock)
        return [(f"{model_clock}-window", df)]

    monkeypatch.setattr(data_mod, "bucket_by_clock", fake_bucket_by_clock)
    monkeypatch.setattr(windows_mod, "chunk_windows", lambda ticks, _chunk_size: [ticks])
    monkeypatch.setattr(
        windows_mod,
        "format_window_chunk",
        lambda chunk, _time_col, _display_cols, _max_events: f"chunk:{chunk[0][0]}",
    )

    captured_contexts: list[dict] = []

    def fake_map(chunk_texts, **kwargs):
        captured_contexts.extend(kwargs["causal_spec"])
        return [
            _FakeFuture({"dataframe": [], "n_extractions": 0, "status": "completed"})
            for _ in chunk_texts
        ]

    monkeypatch.setattr(stage2_extract.extract_window_chunk_task, "map", fake_map)
    monkeypatch.setattr(stage2_extract, "as_completed", lambda futures: iter(futures))

    causal_spec = {
        "latent": {"constructs": [], "edges": []},
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average stress score in the window",
                    "aggregation": "mean",
                    "source_columns": ["timestamp", "stress_score"],
                },
                {
                    "name": "monthly_sleep_hours",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average sleep hours over the last month",
                    "aggregation": "mean",
                    "observation_window": "1mo",
                    "source_columns": ["timestamp", "sleep_hours"],
                },
            ],
        },
    }

    result = _run(
        stage2_extract.stage2_extraction_flow.fn(
            raw_df_path=str(raw_path),
            question="Does stress affect sleep?",
            causal_spec=causal_spec,
        )
    )

    assert bucket_windows == ["1d", "1mo"]
    assert [ctx["measurement"]["indicators"][0]["name"] for ctx in captured_contexts] == [
        "stress_score",
        "monthly_sleep_hours",
    ]
    assert result["observation_rows"] == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stage2_extraction_flow_annotates_medical_imaging_monthly_summary_support_window(
    monkeypatch, tmp_path
):
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.workers.windows as windows_mod

    raw_df = pl.DataFrame(
        {
            "timestamp": ["2024-01-05T08:00:00Z", "2024-01-22T14:00:00Z"],
            "report_text": [
                "Chest CT follow-up scheduled for this month.",
                "Radiology summary: stable pulmonary nodule on January CT.",
            ],
        }
    )
    raw_path = tmp_path / "input.parquet"
    raw_df.write_parquet(raw_path)

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                windows_per_chunk=8,
                max_events_per_window=50,
                max_concurrent_workers=2,
                max_rpm=0,
            )
        ),
    )
    monkeypatch.setattr(windows_mod, "chunk_windows", lambda ticks, _chunk_size: [ticks])
    monkeypatch.setattr(
        windows_mod,
        "format_window_chunk",
        lambda chunk, _time_col, _display_cols, _max_events: f"chunk:{chunk[0][0]}",
    )

    def fake_map(chunk_texts, **kwargs):
        assert chunk_texts == ["chunk:2024-01-01T00:00:00+00:00"]
        assert kwargs["window_starts"] == [["2024-01-01T00:00:00+00:00"]]
        return [
            _FakeFuture(
                {
                    "dataframe": [
                        {
                            "indicator": "monthly_ct_impression",
                            "value": "stable_nodule",
                            "timestamp": "2024-01-01T00:00:00+00:00",
                        }
                    ],
                    "n_extractions": 1,
                    "status": "completed",
                }
            )
        ]

    monkeypatch.setattr(stage2_extract.extract_window_chunk_task, "map", fake_map)
    monkeypatch.setattr(stage2_extract, "as_completed", lambda futures: iter(futures))

    causal_spec = {
        "latent": {"constructs": [], "edges": []},
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "monthly_ct_impression",
                    "measurement_dtype": "categorical",
                    "how_to_measure": (
                        "Within each month, scan report_text for an explicit monthly chest CT "
                        "summary mention and extract the summarized impression directly."
                    ),
                    "aggregation": "last",
                    "observation_window": "1mo",
                    "source_columns": ["timestamp", "report_text"],
                    "extraction_mode": "semantic",
                },
            ],
        },
    }

    result = _run(
        stage2_extract.stage2_extraction_flow.fn(
            raw_df_path=str(raw_path),
            question="How do imaging findings evolve over time?",
            causal_spec=causal_spec,
        )
    )

    rows = pl.DataFrame(result["observation_rows"])
    assert result["n_total_extractions"] == 1
    assert rows.height == 1

    imaging = rows.row(0, named=True)
    assert imaging["indicator"] == "monthly_ct_impression"
    assert imaging["value"] == "stable_nodule"
    assert imaging["support_kind"] == "point"
    assert imaging["summary_operator"] == "last"
    assert imaging["anchor_policy"] == "support_end"
    assert imaging["observation_window"] == "1mo"
    assert imaging["support_start"] == "2024-01-01T00:00:00"
    assert imaging["support_end"] == "2024-02-01T00:00:00"
    assert imaging["anchor_time"] == "2024-02-01T00:00:00"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stage2_extraction_flow_annotates_semantic_rows_into_canonical_observation_rows(
    monkeypatch, tmp_path
):
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.workers.windows as windows_mod

    raw_df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01T08:00:00Z"],
            "stress_score": [4.0],
            "mood_label": ["good"],
        }
    )
    raw_path = tmp_path / "input.parquet"
    raw_df.write_parquet(raw_path)

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                windows_per_chunk=8,
                max_events_per_window=50,
                max_concurrent_workers=2,
                max_rpm=0,
            )
        ),
    )
    monkeypatch.setattr(windows_mod, "chunk_windows", lambda ticks, _chunk_size: [ticks])
    monkeypatch.setattr(
        windows_mod,
        "format_window_chunk",
        lambda chunk, _time_col, _display_cols, _max_events: f"chunk:{chunk[0][0]}",
    )

    def fake_map(chunk_texts, **kwargs):
        assert chunk_texts == ["chunk:2024-01-01T00:00:00+00:00"]
        assert kwargs["window_starts"] == [["2024-01-01T00:00:00+00:00"]]
        return [
            _FakeFuture(
                {
                    "dataframe": [
                        {
                            "indicator": "closing_mood",
                            "value": "good",
                            "timestamp": "2024-01-01T00:00:00+00:00",
                        }
                    ],
                    "n_extractions": 1,
                    "status": "completed",
                }
            )
        ]

    monkeypatch.setattr(stage2_extract.extract_window_chunk_task, "map", fake_map)
    monkeypatch.setattr(stage2_extract, "as_completed", lambda futures: iter(futures))

    causal_spec = {
        "latent": {"constructs": [], "edges": []},
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average stress score in the window",
                    "aggregation": "mean",
                    "source_columns": ["stress_score"],
                    "extraction_mode": "computed",
                },
                {
                    "name": "closing_mood",
                    "measurement_dtype": "ordinal",
                    "how_to_measure": "Take the last mood label in the window",
                    "aggregation": "last",
                    "ordinal_levels": ["bad", "good"],
                    "source_columns": ["timestamp", "mood_label"],
                    "extraction_mode": "semantic",
                },
            ],
        },
    }

    result = _run(
        stage2_extract.stage2_extraction_flow.fn(
            raw_df_path=str(raw_path),
            question="Does stress affect mood?",
            causal_spec=causal_spec,
        )
    )

    rows = pl.DataFrame(result["observation_rows"]).sort("indicator")
    assert result["n_total_extractions"] == 2
    assert result["worker_statuses"] == [
        {
            "worker_id": 0,
            "status": "completed",
            "n_extractions": 1,
            "n_windows": 1,
        }
    ]
    assert rows.height == 2
    assert rows["indicator"].to_list() == ["closing_mood", "stress_score"]

    mood = rows.filter(pl.col("indicator") == "closing_mood")
    assert mood["value"][0] == "good"
    assert mood["support_kind"][0] == "point"
    assert mood["summary_operator"][0] == "last"
    assert mood["anchor_policy"][0] == "support_end"
    assert mood["observation_window"][0] == "1d"
    assert mood["support_start"][0] == "2024-01-01T00:00:00"
    assert mood["support_end"][0] == "2024-01-02T00:00:00"
    assert mood["anchor_time"][0] == "2024-01-02T00:00:00"

    stress = rows.filter(pl.col("indicator") == "stress_score")
    assert stress["value"][0] == "4.0"
    assert stress["support_kind"][0] == "interval"
    assert stress["summary_operator"][0] == "mean"
    assert stress["anchor_policy"][0] == "support_end"
    assert stress["observation_window"][0] == "1d"
    assert stress["support_start"][0] == "2024-01-01T00:00:00"
    assert stress["support_end"][0] == "2024-01-02T00:00:00"
    assert stress["anchor_time"][0] == "2024-01-02T00:00:00"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_stage2_extraction_flow_merges_computed_rule_rows_with_semantic_rows(monkeypatch, tmp_path):
    import nof1_causal_lab.utils.config as config_mod
    import nof1_causal_lab.workers.windows as windows_mod

    raw_df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01T08:00:00Z", "2024-01-01T10:00:00Z"],
            "spo2_pct": [95.0, 91.0],
            "mood_label": ["good", "bad"],
        }
    )
    raw_path = tmp_path / "input.parquet"
    raw_df.write_parquet(raw_path)

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            stage2_workers=SimpleNamespace(
                windows_per_chunk=8,
                max_events_per_window=50,
                max_concurrent_workers=2,
                max_rpm=0,
            )
        ),
    )
    monkeypatch.setattr(windows_mod, "chunk_windows", lambda ticks, _chunk_size: [ticks])
    monkeypatch.setattr(
        windows_mod,
        "format_window_chunk",
        lambda chunk, _time_col, _display_cols, _max_events: f"chunk:{chunk[0][0]}",
    )

    def fake_map(chunk_texts, **kwargs):
        assert chunk_texts == ["chunk:2024-01-01T00:00:00+00:00"]
        assert kwargs["window_starts"] == [["2024-01-01T00:00:00+00:00"]]
        return [
            _FakeFuture(
                {
                    "dataframe": [
                        {
                            "indicator": "closing_mood",
                            "value": "bad",
                            "timestamp": "2024-01-01T00:00:00+00:00",
                        }
                    ],
                    "n_extractions": 1,
                    "status": "completed",
                }
            )
        ]

    monkeypatch.setattr(stage2_extract.extract_window_chunk_task, "map", fake_map)
    monkeypatch.setattr(stage2_extract, "as_completed", lambda futures: iter(futures))

    causal_spec = {
        "latent": {"constructs": [], "edges": []},
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "low_spo2",
                    "measurement_dtype": "binary",
                    "how_to_measure": "Deterministically flag any low SpO2 in the window",
                    "aggregation": "last",
                    "source_columns": ["spo2_pct"],
                    "computed_rule": {
                        "window_expr": "1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)"
                    },
                    "extraction_mode": "computed",
                },
                {
                    "name": "closing_mood",
                    "measurement_dtype": "ordinal",
                    "how_to_measure": "Take the last mood label in the window",
                    "aggregation": "last",
                    "ordinal_levels": ["bad", "good"],
                    "source_columns": ["timestamp", "mood_label"],
                    "extraction_mode": "semantic",
                },
            ],
        },
    }

    result = _run(
        stage2_extract.stage2_extraction_flow.fn(
            raw_df_path=str(raw_path),
            question="Does oxygen saturation affect mood?",
            causal_spec=causal_spec,
        )
    )

    rows = pl.DataFrame(result["observation_rows"]).sort("indicator")
    assert result["n_total_extractions"] == 2
    assert rows["indicator"].to_list() == ["closing_mood", "low_spo2"]

    low_spo2 = rows.filter(pl.col("indicator") == "low_spo2")
    assert low_spo2["value"][0] == "1"
    assert low_spo2["support_kind"][0] == "point"
    assert low_spo2["summary_operator"][0] == "last"
    assert low_spo2["anchor_policy"][0] == "support_end"
    assert low_spo2["support_start"][0] == "2024-01-01T00:00:00"
    assert low_spo2["support_end"][0] == "2024-01-02T00:00:00"
    assert low_spo2["anchor_time"][0] == "2024-01-02T00:00:00"
