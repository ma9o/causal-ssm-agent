"""Stage 2 deterministic planning helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.flows import get_prefect_logger

if TYPE_CHECKING:
    import polars as pl

logger = get_prefect_logger(__name__)


def project_to_source_columns(df: pl.DataFrame, indicators: list[dict]) -> pl.DataFrame:
    """Project DataFrame to only the columns referenced by indicators."""
    source_cols: set[str] = set()
    for indicator in indicators:
        source_cols.update(indicator.get("source_columns", []))

    if not source_cols:
        return df

    missing = source_cols - set(df.columns)
    if missing:
        logger.warning(
            "Stage 2: source_columns not found in DataFrame, skipping them: %s",
            sorted(missing),
        )
    keep = [column for column in df.columns if column in source_cols]
    if not keep:
        return df

    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.info(
            "Stage 2: projected %d→%d columns (dropped %d)",
            len(df.columns),
            len(keep),
            dropped,
        )
    return df.select(keep)


def group_indicators_by_window(
    indicators: list[dict],
    model_clock: str,
) -> list[tuple[str, list[dict]]]:
    from nof1_causal_lab.utils.causal_spec import get_effective_observation_window

    grouped: dict[str, list[dict]] = {}
    for indicator in indicators:
        window = get_effective_observation_window(indicator, model_clock) or model_clock
        grouped.setdefault(window, []).append(indicator)
    return sorted(grouped.items(), key=lambda item: item[0])


def chunk_log_label(chunk_idx: int, n_windows: int, n_events: int) -> str:
    return f"stage2 chunk={chunk_idx} windows={n_windows} events={n_events}"


def prepare_semantic_chunks(
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
    """Prepare semantic extraction chunks without executing them."""
    from nof1_causal_lab.utils.causal_spec import make_extraction_context
    from nof1_causal_lab.utils.data import bucket_by_clock
    from nof1_causal_lab.workers.windows import chunk_windows, format_window_chunk

    chunk_texts: list[str] = []
    chunk_window_starts: list[list[str]] = []
    chunk_contexts: list[dict] = []

    for observation_window, semantic_group in group_indicators_by_window(
        semantic_inds, model_clock
    ):
        semantic_spec = {
            **causal_spec,
            "measurement": {**causal_spec.get("measurement", {}), "indicators": semantic_group},
        }
        extraction_ctx = make_extraction_context(semantic_spec)

        projected = project_to_source_columns(raw_df, semantic_group)
        if time_col not in projected.columns:
            projected = projected.with_columns(raw_df[time_col])

        windows = bucket_by_clock(projected, observation_window, time_col)
        logger.info(
            "Stage 2: bucketed %d rows into %d support windows (window=%s, indicators=%d)",
            len(projected),
            len(windows),
            observation_window,
            len(semantic_group),
        )

        if max_windows is not None and len(windows) > max_windows:
            logger.warning(
                "Stage 2: free-tier window cap active for window=%s — truncating %d windows to most recent %d",
                observation_window,
                len(windows),
                max_windows,
            )
            windows = windows[-max_windows:]

        if not windows:
            continue

        display_cols = [column for column in projected.columns if column != time_col]
        chunks = chunk_windows(windows, windows_per_chunk)
        for chunk in chunks:
            chunk_texts.append(
                format_window_chunk(chunk, time_col, display_cols, max_events_per_window)
            )
            chunk_window_starts.append([window_start for window_start, _ in chunk])
            chunk_contexts.append(extraction_ctx)

    return chunk_texts, chunk_window_starts, chunk_contexts
