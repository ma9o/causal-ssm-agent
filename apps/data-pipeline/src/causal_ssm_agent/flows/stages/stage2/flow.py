"""Stage 2: Hybrid computed/semantic extraction entrypoints."""

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import polars as pl
from prefect import flow, get_run_logger, task
from prefect.futures import as_completed

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.runtime_events import (
    emit_nested_stage_running_event,
    emit_stage2_plan_event,
    emit_stage2_worker_event,
)
from causal_ssm_agent.flows.stages.stage2.execution import collect_batch_results
from causal_ssm_agent.flows.stages.stage2.planning import (
    chunk_log_label,
    prepare_semantic_chunks,
    project_to_source_columns,
)
from causal_ssm_agent.flows.stages.stage2.progress import (
    clear_stage2_progress_tracker,
    emit_stage2_snapshot,
    get_stage2_progress_tracker,
    register_stage2_progress_tracker,
)

logger = get_prefect_logger(__name__)

SemanticChunkRunner = Callable[..., Awaitable[tuple[list[dict], list[dict], int, dict | None]]]


def _register_stage2_progress_tracker(
    root_run_id: str,
    *,
    total_workers: int,
) -> Any:
    return register_stage2_progress_tracker(root_run_id, total_workers=total_workers)


def _get_stage2_progress_tracker(root_run_id: str) -> Any | None:
    return get_stage2_progress_tracker(root_run_id)


def _clear_stage2_progress_tracker(root_run_id: str) -> None:
    clear_stage2_progress_tracker(root_run_id)


def _emit_stage2_snapshot(root_run_id: str, snapshot: dict[str, int]) -> None:
    emit_stage2_snapshot(root_run_id, snapshot)


def _project_to_source_columns(df: pl.DataFrame, indicators: list[dict]) -> pl.DataFrame:
    return project_to_source_columns(df, indicators)


def _chunk_log_label(chunk_idx: int, n_windows: int, n_events: int) -> str:
    return chunk_log_label(chunk_idx, n_windows, n_events)


def _collect_batch_results(
    *,
    futures: Sequence[Any],
    batch_indices: Sequence[int],
    batch_n_windows: Sequence[int],
    logger: Any,
    completed_before: int,
    total_chunks: int,
    root_run_id: str | None = None,
) -> tuple[list[dict], list[dict], int, dict | None]:
    return collect_batch_results(
        futures=futures,
        batch_indices=batch_indices,
        batch_n_windows=batch_n_windows,
        logger=logger,
        completed_before=completed_before,
        total_chunks=total_chunks,
        as_completed_fn=as_completed,
        root_run_id=root_run_id,
        emit_worker_event=emit_stage2_worker_event,
        get_tracker=_get_stage2_progress_tracker,
        emit_snapshot=_emit_stage2_snapshot,
    )


def _prepare_semantic_chunks(
    *,
    raw_df: pl.DataFrame,
    semantic_inds: list[dict],
    causal_spec: dict,
    model_clock: str,
    time_col: str,
    windows_per_chunk: int,
    max_events_per_window: int,
    max_windows: int | None,
) -> tuple[list[str], list[list[str]], list[dict]]:
    return prepare_semantic_chunks(
        raw_df=raw_df,
        semantic_inds=semantic_inds,
        causal_spec=causal_spec,
        model_clock=model_clock,
        time_col=time_col,
        windows_per_chunk=windows_per_chunk,
        max_events_per_window=max_events_per_window,
        max_windows=max_windows,
    )


async def _run_semantic_chunks_prefect(
    *,
    chunk_texts: list[str],
    chunk_window_starts: list[list[str]],
    chunk_contexts: list[dict],
    question: str,
    root_run_id: str | None,
    openrouter_api_key: str | None,
    max_concurrent_workers: int,
    max_rpm: int,
) -> tuple[list[dict], list[dict], int, dict | None]:
    """Execute semantic chunks through the existing Prefect worker path."""
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.utils.openrouter_client import RpmLimiter, set_limiter

    all_indices = list(range(len(chunk_texts)))
    all_n_windows = [len(ids) for ids in chunk_window_starts]

    logger.info(
        "Stage 2: %d semantic chunks of up to %d windows each (max_concurrent_workers=%d, max_rpm=%d)",
        len(chunk_texts),
        max(all_n_windows) if all_n_windows else 0,
        max_concurrent_workers,
        max_rpm,
    )

    if max_rpm:
        set_limiter("llm", RpmLimiter(max_rpm))

    try:
        if root_run_id:
            emit_stage2_plan_event(
                root_run_id,
                total_workers=len(chunk_texts),
                max_concurrent_workers=max_concurrent_workers,
                max_rpm=max_rpm,
            )
            tracker = _register_stage2_progress_tracker(
                root_run_id,
                total_workers=len(chunk_texts),
            )
            _emit_stage2_snapshot(root_run_id, tracker.snapshot())
        results = extract_window_chunk_task.map(
            chunk_texts,
            window_starts=chunk_window_starts,
            chunk_idx=all_indices,
            question=unmapped(question),
            causal_spec=chunk_contexts,
            root_run_id=unmapped(root_run_id),
            openrouter_api_key=unmapped(openrouter_api_key),
        )
        return _collect_batch_results(
            futures=results,
            batch_indices=all_indices,
            batch_n_windows=all_n_windows,
            logger=logger,
            completed_before=0,
            total_chunks=len(chunk_texts),
            root_run_id=root_run_id,
        )
    finally:
        if root_run_id:
            _clear_stage2_progress_tracker(root_run_id)
        set_limiter("llm", None)


async def run_stage2_extraction_core(
    *,
    raw_df: pl.DataFrame,
    question: str,
    causal_spec: dict,
    stage2_workers: Any,
    root_run_id: str | None = None,
    max_windows: int | None = None,
    openrouter_api_key: str | None = None,
    semantic_chunk_runner: SemanticChunkRunner | None = None,
) -> dict:
    """Shared Stage 2 extraction helper for flows and evals.

    This is the deterministic orchestration core for:
    1. splitting computed vs semantic indicators
    2. computing direct indicators via Polars
    3. preparing semantic support-window chunks
    4. delegating semantic execution to an injected backend
    5. annotating canonical observation-row support metadata
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.data import ObservationRecord, annotate_observation_rows

    semantic_chunk_runner = semantic_chunk_runner or _run_semantic_chunks_prefect

    all_indicators = get_indicators(causal_spec)
    time_col = "timestamp"
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "1d")
    logger.info("Stage 2: time_col='%s', model_clock='%s'", time_col, model_clock)

    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [
        i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"
    ]
    logger.info(
        "Stage 2: %d computed + %d semantic indicators",
        len(computed_inds),
        len(semantic_inds),
    )

    computed_dicts: list[dict] = []
    if computed_inds:
        from causal_ssm_agent.utils.aggregations import compute_indicators

        computed_df = compute_indicators(raw_df, computed_inds, model_clock, time_col)
        computed_dicts = computed_df.to_dicts()
        logger.info(
            "Stage 2: computed %d indicator(s) via Polars (%d rows)",
            len(computed_inds),
            len(computed_df),
        )

    semantic_dicts: list[dict] = []
    worker_statuses: list[dict] = []
    sampled_llm_trace: dict | None = None
    n_semantic_total = 0

    if semantic_inds:
        chunk_texts, chunk_window_starts, chunk_contexts = _prepare_semantic_chunks(
            raw_df=raw_df,
            semantic_inds=semantic_inds,
            causal_spec=causal_spec,
            model_clock=model_clock,
            time_col=time_col,
            windows_per_chunk=stage2_workers.windows_per_chunk,
            max_events_per_window=stage2_workers.max_events_per_window,
            max_windows=max_windows,
        )

        if chunk_texts:
            (
                semantic_dicts,
                worker_statuses,
                n_semantic_total,
                sampled_llm_trace,
            ) = await semantic_chunk_runner(
                chunk_texts=chunk_texts,
                chunk_window_starts=chunk_window_starts,
                chunk_contexts=chunk_contexts,
                question=question,
                root_run_id=root_run_id,
                openrouter_api_key=openrouter_api_key,
                max_concurrent_workers=stage2_workers.max_concurrent_workers,
                max_rpm=stage2_workers.max_rpm,
            )

    all_dicts = computed_dicts + semantic_dicts
    n_total = len(computed_dicts) + n_semantic_total
    logger.info(
        "Stage 2: %d total extractions (%d computed, %d semantic from %d workers)",
        n_total,
        len(computed_dicts),
        n_semantic_total,
        len(worker_statuses),
    )

    observation_rows = cast(
        "list[ObservationRecord]",
        annotate_observation_rows(pl.DataFrame(all_dicts), causal_spec).to_dicts(),
    )

    result = {
        "observation_rows": observation_rows,
        "worker_statuses": worker_statuses,
        "n_total_extractions": n_total,
    }
    if sampled_llm_trace is not None:
        result["llm_trace"] = sampled_llm_trace
    return result


@task(
    retries=2,
    retry_delay_seconds=10,
    timeout_seconds=300,
    task_run_name="extract-windows-{chunk_idx}",
)
async def extract_window_chunk_task(
    window_text: str,
    window_starts: list[str],
    chunk_idx: int,
    question: str,
    causal_spec: dict,
    root_run_id: str | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """Extract indicator values from a chunk of support windows.

    Args:
        window_text: Pre-formatted text of support-window events.
        window_starts: Expected support-window starts in this chunk.
        chunk_idx: Index of this chunk (for logging/naming).
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.

    Returns:
        Dict with 'dataframe' (as list of dicts for serialization),
        'n_extractions', and 'status'.
    """
    from dataclasses import replace

    from causal_ssm_agent.utils.agent_session import StageSessionFactory
    from causal_ssm_agent.utils.causal_spec import get_indicators
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key
    from causal_ssm_agent.workers.core import run_worker_extraction

    run_logger = get_run_logger()
    config = get_config()
    # Stage 2 workers override only the per-call timeout (prevents hung calls).
    # Everything else inherits the embedded LLM defaults.
    stage2_llm = replace(
        config.stage2_workers.llm,
        timeout=config.stage2_workers.worker_timeout,
    )
    indicator_count = len(get_indicators(causal_spec))
    n_events = window_text.count("\n")
    chunk_label = _chunk_log_label(chunk_idx, len(window_starts), n_events)

    run_logger.info(
        "[%s] Starting extraction with %d windows, %d indicators using model %s (timeout=%ds)",
        chunk_label,
        len(window_starts),
        indicator_count,
        stage2_llm.model,
        stage2_llm.timeout,
    )
    if root_run_id:
        emit_stage2_worker_event(
            root_run_id,
            worker_id=chunk_idx,
            state="running",
            n_windows=len(window_starts),
        )
        tracker = _get_stage2_progress_tracker(root_run_id)
        if tracker is not None:
            _emit_stage2_snapshot(root_run_id, tracker.mark_running(chunk_idx))

    with use_openrouter_api_key(openrouter_api_key):
        factory = StageSessionFactory(
            stage2_llm,
            config.llm,
            stage_id=f"stage-2/chunk-{chunk_idx}",
            max_tool_turns=config.stage2_workers.max_tool_turns,
        )

        started_at = perf_counter()
        result = await run_worker_extraction(
            window_text=window_text,
            window_starts=window_starts,
            question=question,
            causal_spec=causal_spec,
            session_factory=factory,
            logger=run_logger,
            call_label=chunk_label,
        )

        elapsed = perf_counter() - started_at
        run_logger.info(
            "[%s] Finished in %.1fs with %d extractions and %d output rows",
            chunk_label,
            elapsed,
            len(result.output.extractions),
            result.dataframe.height,
        )

        result_dict: dict = {
            "dataframe": result.dataframe.to_dicts(),
            "n_extractions": len(result.output.extractions),
            "status": "completed",
        }
        if factory.accumulated_trace.messages:
            result_dict["llm_trace"] = factory.accumulated_trace.model_dump(mode="json")
        return result_dict


@flow(
    name="stage2-worker-extraction",
    log_prints=True,
    persist_result=True,
    result_serializer="json",
)
async def stage2_extraction_flow(
    raw_df_path: str,
    question: str,
    causal_spec: dict,
    root_run_id: str | None = None,
    max_windows: int | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """Stage 2: Extract indicator values via hybrid computed/semantic paths.

    1. Splits indicators by extraction_mode (computed vs semantic)
    2. Computed path: direct Polars aggregation (~50ms)
    3. Semantic path: support-window LLM workers
    4. Merges results into canonical observation rows with support metadata

    Args:
        raw_df_path: Path to the raw wide-format DataFrame from Stage 0 ingestion.
        question: The causal research question.
        causal_spec: Full CausalSpec dict with measurement model.

    Returns:
        Dict with 'observation_rows' (long-format observation rows as list of dicts),
        'worker_statuses', and 'n_total_extractions'.
    """
    from causal_ssm_agent.utils.config import get_config

    if root_run_id:
        emit_nested_stage_running_event(root_run_id, "stage-2")

    config = get_config()
    raw_df = pl.read_parquet(Path(raw_df_path))
    return await run_stage2_extraction_core(
        raw_df=raw_df,
        question=question,
        causal_spec=causal_spec,
        stage2_workers=config.stage2_workers,
        root_run_id=root_run_id,
        max_windows=max_windows,
        openrouter_api_key=openrouter_api_key,
        semantic_chunk_runner=_run_semantic_chunks_prefect,
    )
