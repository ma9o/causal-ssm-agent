"""Run-level artifact persistence for the causal inference pipeline.

Run-dir I/O that remains outside the versioned artifact store: stage-4's
private compile-cache sidecar. Everything the machine owns lives in
nof1_causal_lab.machine.store.
"""

from __future__ import annotations

import logging
from typing import Any

import cloudpickle

from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import runs_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename constants for run artifacts
# ---------------------------------------------------------------------------

STAGE4_JAX_CACHE_FILENAMES = ("stage4-jax-cache.tar.gz",)
STAGE4_JAX_CACHE_METADATA_FILENAMES = ("stage4-jax-cache-metadata.json",)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_run_dir(workspace_id: str) -> str:
    """Return the run directory, creating it if needed."""
    path = runs_dir(workspace_id)
    storage.makedirs(path)
    return path


def existing_run_dir(workspace_id: str) -> str:
    """Return the run directory, raising if it doesn't exist."""
    path = runs_dir(workspace_id)
    if not storage.exists(path):
        raise FileNotFoundError(f"No results directory found for workspace_id {workspace_id}")
    return path


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def load_parquet(path: str) -> Any:
    """Read a Polars DataFrame from a parquet path."""
    import polars as pl

    return pl.read_parquet(path, storage_options=storage.polars_storage_options())


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------


def load_pickle(path: str) -> Any:
    """Unpickle a value from storage."""
    with storage.open_file(path, "rb") as f:
        return cloudpickle.load(f)


def load_json(path: str) -> Any:
    """Read a JSON value from storage."""
    return storage.read_json(path)


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def find_run_artifact(workspace_id: str, filenames: tuple[str, ...]) -> str:
    """Return the path of the first existing artifact from a list of candidates."""
    run_dir = existing_run_dir(workspace_id)
    for filename in filenames:
        path = storage.join(run_dir, filename)
        if storage.exists(path):
            return path
    expected = ", ".join(filenames)
    raise FileNotFoundError(f"None of [{expected}] exist for workspace_id {workspace_id}")
