"""Shared Stage 4 text-formatting helpers."""

from __future__ import annotations


def summarize_stage4_names(names: list[str], *, limit: int = 8) -> str:
    """Render a compact preview of Stage 4 names."""
    if not names:
        return "(none)"
    preview = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) <= limit:
        return preview
    return f"{preview}, ... (+{len(names) - limit} more)"
