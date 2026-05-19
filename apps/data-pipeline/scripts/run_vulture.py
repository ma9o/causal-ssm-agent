"""Run vulture with notebooks (.ipynb) converted to .py first.

vulture only scans .py files. This wrapper extracts code cells from every
notebook under ``notebooks/`` into a temporary ``.vulture_cache/`` tree of
``.py`` shadows, then runs vulture against the config paths plus the cache,
then removes the cache.

Identifiers wrapped in backticks inside markdown cells (e.g. `foo_bar`) are
emitted as phantom references so functions documented as swap-in hooks in
notebook prose are not flagged as dead.

Usage:
    cd apps/data-pipeline
    uv run python scripts/run_vulture.py                # standard run
    uv run python scripts/run_vulture.py --make-whitelist > vulture_whitelist.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
CACHE_DIR = REPO_ROOT / ".vulture_cache"
PYPROJECT = REPO_ROOT / "pyproject.toml"

BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def _extract_backtick_idents(text: str) -> set[str]:
    refs: set[str] = set()
    for span in BACKTICK_SPAN.findall(text):
        refs.update(IDENT.findall(span))
    return refs


def _convert_notebooks() -> set[str]:
    """Convert .ipynb code cells to .py shadows; return markdown identifier refs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    markdown_refs: set[str] = set()
    for nb_path in NOTEBOOKS_DIR.rglob("*.ipynb"):
        out_path = CACHE_DIR / nb_path.relative_to(NOTEBOOKS_DIR).with_suffix(".py")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nb = nbformat.read(nb_path, as_version=4)
        sources: list[str] = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                sources.append(cell.source)
            elif cell.cell_type == "markdown":
                markdown_refs.update(_extract_backtick_idents(cell.source))
        out_path.write_text("\n\n".join(sources))
    return markdown_refs


def _write_markdown_phantom(refs: set[str]) -> None:
    if not refs:
        return
    body = "\n".join(f"_ = {name}" for name in sorted(refs))
    (CACHE_DIR / "_notebook_markdown_refs.py").write_text(body + "\n")


def main() -> int:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    try:
        refs = _convert_notebooks()
        _write_markdown_phantom(refs)
        config = tomllib.loads(PYPROJECT.read_text())
        paths = config["tool"]["vulture"]["paths"]
        result = subprocess.run(
            ["vulture", *paths, CACHE_DIR.name, *sys.argv[1:]],
            cwd=REPO_ROOT,
            check=False,
        )
        return result.returncode
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
