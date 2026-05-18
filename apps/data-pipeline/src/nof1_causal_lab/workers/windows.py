"""Support-window formatting and chunking for worker extraction.

Converts bucketed DataFrames into LLM-ready text and groups support windows
into chunks for parallel worker calls.
"""

import polars as pl


def chunk_windows(
    windows: list[tuple[str, pl.DataFrame]],
    windows_per_chunk: int,
) -> list[list[tuple[str, pl.DataFrame]]]:
    """Group support windows into chunks of N windows each.

    Args:
        windows: List of (window_start, events_df) from bucket_by_clock.
        windows_per_chunk: Maximum windows per chunk.

    Returns:
        List of chunks, each a list of (window_start, events_df).
    """
    if not windows:
        return []
    return [windows[i : i + windows_per_chunk] for i in range(0, len(windows), windows_per_chunk)]


def format_window_chunk(
    chunk: list[tuple[str, pl.DataFrame]],
    time_col: str,
    display_cols: list[str] | None = None,
    max_events_per_window: int = 300,
) -> str:
    """Format a chunk of support windows as text for the LLM prompt.

    Each support window is rendered as a section with chronological event lines.
    Events exceeding max_events_per_window are truncated with boundary
    preservation (first 10 + last 10 + sampled middle).

    Args:
        chunk: List of (window_start, events_df) for this chunk.
        time_col: Name of the time column (used for chronological ordering).
        display_cols: Columns to show. If None, shows all non-time columns.
        max_events_per_window: Maximum events to show per support window before truncation.

    Returns:
        Formatted text ready for the LLM prompt.
    """
    sections = []

    for window_start, events_df in chunk:
        # Sort chronologically
        events_df = events_df.sort(time_col)

        # Determine columns to display
        cols = display_cols or [c for c in events_df.columns if c != time_col]
        if not cols:
            cols = events_df.columns

        # Format events as CSV lines with time prefix
        lines = []
        for row in events_df.iter_rows(named=True):
            # Extract within-window time for ordering context
            time_val = row.get(time_col)
            time_str = _format_time_within_window(time_val)

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
        if n_total > max_events_per_window:
            lines = _truncate_events(lines, max_events_per_window, n_total)

        # Build section
        header = f"## Window Start: {window_start}"
        sections.append(header + "\n\n" + "\n".join(lines))

    return "\n\n".join(sections)


def _format_time_within_window(time_val) -> str:
    """Format a datetime value as HH:MM for display within a support window."""
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

    sampled = _sample_middle_events(middle_pool, n_middle)

    note = f"(showing {max_events} of {n_total} events, sampled)"
    return [note, *head, *sampled, *tail]


def _sample_middle_events(middle_pool: list[str], n_middle: int) -> list[str]:
    """Select a deterministic, evenly spaced sample from the middle of a window."""
    if n_middle <= 0 or not middle_pool:
        return []
    if n_middle >= len(middle_pool):
        return list(middle_pool)

    segment_size = len(middle_pool) / n_middle
    sampled: list[str] = []
    for i in range(n_middle):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        idx = start if end <= start else (start + end - 1) // 2
        sampled.append(middle_pool[idx])
    return sampled
