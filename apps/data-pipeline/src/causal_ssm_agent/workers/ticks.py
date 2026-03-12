"""Tick formatting and chunking for tick-based extraction.

Converts bucketed DataFrames into LLM-ready text and groups ticks
into chunks for parallel worker calls.
"""

import random

import polars as pl


def chunk_ticks(
    ticks: list[tuple[str, pl.DataFrame]],
    ticks_per_chunk: int,
) -> list[list[tuple[str, pl.DataFrame]]]:
    """Group ticks into chunks of N ticks each.

    Args:
        ticks: List of (tick_id, events_df) from bucket_by_clock.
        ticks_per_chunk: Maximum ticks per chunk.

    Returns:
        List of chunks, each a list of (tick_id, events_df).
    """
    if not ticks:
        return []
    return [ticks[i : i + ticks_per_chunk] for i in range(0, len(ticks), ticks_per_chunk)]


def format_tick_chunk(
    chunk: list[tuple[str, pl.DataFrame]],
    time_col: str,
    display_cols: list[str] | None = None,
    max_events_per_tick: int = 300,
) -> str:
    """Format a chunk of ticks as text for the LLM prompt.

    Each tick is rendered as a section with chronological event lines.
    Events exceeding max_events_per_tick are truncated with boundary
    preservation (first 10 + last 10 + sampled middle).

    Args:
        chunk: List of (tick_id, events_df) for this chunk.
        time_col: Name of the time column (used for chronological ordering).
        display_cols: Columns to show. If None, shows all non-time columns.
        max_events_per_tick: Maximum events to show per tick before truncation.

    Returns:
        Formatted text ready for the LLM prompt.
    """
    sections = []

    for tick_id, events_df in chunk:
        # Sort chronologically
        events_df = events_df.sort(time_col)

        # Determine columns to display
        cols = display_cols or [c for c in events_df.columns if c != time_col]
        if not cols:
            cols = events_df.columns

        # Format events as CSV lines with time prefix
        lines = []
        for row in events_df.iter_rows(named=True):
            # Extract sub-tick time for ordering context
            time_val = row.get(time_col)
            time_str = _format_time_within_tick(time_val)

            # Format remaining columns
            parts = []
            for col in cols:
                val = row.get(col)
                if val is not None:
                    parts.append(str(val))
            line = f"{time_str}  {', '.join(parts)}" if time_str else ", ".join(parts)
            lines.append(line)

        # Truncate if needed
        n_total = len(lines)
        if n_total > max_events_per_tick:
            lines = _truncate_events(lines, max_events_per_tick, n_total)

        # Build section
        header = f"## Tick: {tick_id}"
        sections.append(header + "\n\n" + "\n".join(lines))

    return "\n\n".join(sections)


def _format_time_within_tick(time_val) -> str:
    """Format a datetime value as HH:MM for display within a tick."""
    if time_val is None:
        return ""
    if hasattr(time_val, "strftime"):
        return time_val.strftime("%H:%M")
    return ""


def _truncate_events(
    lines: list[str],
    max_events: int,
    n_total: int,
) -> list[str]:
    """Truncate event lines preserving temporal boundaries.

    Strategy: keep first 10 + last 10, uniform sample from middle.
    """
    n_boundary = min(10, max_events // 3)
    n_middle = max_events - 2 * n_boundary

    head = lines[:n_boundary]
    tail = lines[-n_boundary:]
    middle_pool = lines[n_boundary : -n_boundary or None]

    if n_middle > 0 and middle_pool:
        sampled = sorted(
            random.sample(middle_pool, min(n_middle, len(middle_pool))),
            key=lambda x: middle_pool.index(x),
        )
    else:
        sampled = []

    note = f"(showing {max_events} of {n_total} events, sampled)"
    return [note, *head, *sampled, *tail]
