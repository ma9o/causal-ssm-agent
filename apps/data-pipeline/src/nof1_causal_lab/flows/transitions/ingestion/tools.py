"""Shared ingestion tool helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve a user-provided path within the staged input directory."""
    base_resolved = base.resolve()
    resolved = (base / user_path).resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"Path traversal blocked: {user_path}")
    return resolved
