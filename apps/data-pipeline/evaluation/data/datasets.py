"""External-dataset acquisition for the evaluation harness.

The code-vs-data split:

* **Generators are code** — synthetic fixtures live in ``evaluation/fixtures``.
* **Static input fixtures are data** — golden artifacts and eval questions live
  under ``evaluation/data`` (e.g. ``evaluation/data/questions``).
* **Fetched external datasets** (EpiCF, PhysioNet, the causal chambers, ...) are
  downloaded on demand and cached under ``evaluation/data/cache`` (gitignored)
  via :func:`cached_download` — never committed to the code tree.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"


def cached_download(url: str, filename: str, *, cache_dir: Path | None = None) -> Path:
    """Fetch ``url`` into the cache as ``filename`` once; reuse it thereafter.

    Idempotent: an existing non-empty cached file is returned without re-fetching,
    so the same external dataset is downloaded at most once per machine.
    """
    dest_dir = cache_dir or CACHE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())
    return dest
