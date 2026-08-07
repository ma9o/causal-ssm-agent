"""extraction deterministic planning helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def project_to_source_columns(
    df: pl.DataFrame,
    indicators: list[UncheckedJsonObject],
) -> pl.DataFrame:
    """Project DataFrame to only the columns referenced by indicators."""
    source_cols: set[str] = set()
    for indicator in indicators:
        source_cols.update(indicator.get("source_columns", []))

    if not source_cols:
        return df

    missing = source_cols - set(df.columns)
    if missing:
        logger.warning(
            "extraction: source_columns not found in DataFrame, skipping them: %s",
            sorted(missing),
        )
    keep = [column for column in df.columns if column in source_cols]
    if not keep:
        return df

    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.info(
            "extraction: projected %d→%d columns (dropped %d)",
            len(df.columns),
            len(keep),
            dropped,
        )
    return df.select(keep)


def group_indicators_by_window(
    indicators: list[UncheckedJsonObject],
    model_clock: str,
) -> list[tuple[str, list[UncheckedJsonObject]]]:
    from nof1_causal_lab.utils.causal_design import get_effective_observation_window

    grouped: dict[str, list[UncheckedJsonObject]] = {}
    for indicator in indicators:
        window = get_effective_observation_window(indicator, model_clock) or model_clock
        grouped.setdefault(window, []).append(indicator)
    return sorted(grouped.items(), key=lambda item: item[0])


def prepare_semantic_chunks(
    *,
    raw_df: pl.DataFrame,
    semantic_inds: list[UncheckedJsonObject],
    measurement_structure: UncheckedJsonObject,
    model_clock: str,
    time_col: str,
    windows_per_chunk: int,
    max_events_per_window: int,
    max_windows: int | None,
) -> tuple[list[str], list[list[str]], list[UncheckedJsonObject]]:
    """Prepare semantic extraction chunks without executing them."""
    from nof1_causal_lab.utils.causal_design import make_measurement_extraction_context
    from nof1_causal_lab.utils.data import bucket_by_clock
    from nof1_causal_lab.workers.windows import chunk_windows, format_window_chunk

    chunk_texts: list[str] = []
    chunk_window_starts: list[list[str]] = []
    chunk_contexts: list[UncheckedJsonObject] = []

    for observation_window, semantic_group in group_indicators_by_window(
        semantic_inds, model_clock
    ):
        semantic_spec = {
            **measurement_structure,
            "indicators": semantic_group,
        }
        extraction_ctx = make_measurement_extraction_context(semantic_spec)

        projected = project_to_source_columns(raw_df, semantic_group)
        if time_col not in projected.columns:
            projected = projected.with_columns(raw_df[time_col])

        windows = bucket_by_clock(projected, observation_window, time_col)
        logger.info(
            "extraction: bucketed %d rows into %d support windows (window=%s, indicators=%d)",
            len(projected),
            len(windows),
            observation_window,
            len(semantic_group),
        )

        if max_windows is not None and len(windows) > max_windows:
            logger.warning(
                "extraction: free-tier window cap active for window=%s — truncating %d windows to most recent %d",
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
