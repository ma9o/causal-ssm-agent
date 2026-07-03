"""Run-level artifact persistence for the causal inference pipeline.

Run-dir I/O that remains outside the versioned artifact store: the
web-facing stage JSON projection and stage-4's private compile-cache /
checkpoint machinery. Everything the machine owns lives in
nof1_causal_lab.machine.store.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import cloudpickle

from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import runs_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename constants for run artifacts
# ---------------------------------------------------------------------------

STAGE4_JAX_CACHE_FILENAMES = ("stage4-jax-cache.tar.gz",)
STAGE4_JAX_CACHE_METADATA_FILENAMES = ("stage4-jax-cache-metadata.json",)
STAGE4_CHECKPOINT_DIRNAME = "stage-4-checkpoints"
STAGE4_CHECKPOINT_CURSOR_FILENAME = "cursor.json"
STAGE4_DONE_CHECKPOINT_CACHE_KEY = "__done__"

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


# ---------------------------------------------------------------------------
# JSON artifact I/O
# ---------------------------------------------------------------------------


def save_json(value: Any, workspace_id: str, filename: str) -> str:
    """Write a JSON-serializable value into the run directory."""
    path = storage.join(ensure_run_dir(workspace_id), filename)
    storage.write_text(path, json.dumps(value))
    return path


def load_json(path: str) -> Any:
    """Read a JSON value from storage."""
    return storage.read_json(path)


def _stage4_checkpoint_dir(workspace_id: str, *, create: bool) -> str:
    """Return the Stage 4 checkpoint directory."""
    run_dir = ensure_run_dir(workspace_id) if create else existing_run_dir(workspace_id)
    return storage.join(run_dir, STAGE4_CHECKPOINT_DIRNAME)


def _stage4_checkpoint_cursor_path(workspace_id: str, *, create: bool) -> str:
    """Return the Stage 4 checkpoint cursor path."""
    return storage.join(
        _stage4_checkpoint_dir(workspace_id, create=create),
        STAGE4_CHECKPOINT_CURSOR_FILENAME,
    )


def _stage4_checkpoint_cache_key(runtime: Any) -> str:
    """Return the persisted Stage 4 cache key for one runtime wait-state."""
    domain = getattr(runtime, "domain", None)
    if domain is None:
        raise TypeError("Stage 4 checkpoint payload is not a Stage4Runtime")
    if bool(getattr(domain, "done", False)):
        return STAGE4_DONE_CHECKPOINT_CACHE_KEY
    block_id = getattr(domain, "active_block_id", None)
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("Stage 4 checkpoint requires an active block or a done state")
    return block_id


def _stage4_checkpoint_path_for_cache_key(
    workspace_id: str,
    cache_key: str,
    *,
    create: bool,
) -> str:
    """Return the Stage 4 checkpoint path for one block or done cache key."""
    filename = f"{quote(cache_key, safe='')}.pkl"
    return storage.join(_stage4_checkpoint_dir(workspace_id, create=create), filename)


def _stage4_checkpoint_cursor_payload(runtime: Any) -> dict[str, Any]:
    """Build the Stage 4 cursor payload for one persisted runtime."""
    cache_key = _stage4_checkpoint_cache_key(runtime)
    if cache_key == STAGE4_DONE_CHECKPOINT_CACHE_KEY:
        return {"kind": "done"}
    return {"kind": "block", "block_id": cache_key}


def save_stage4_checkpoint(runtime: Any, workspace_id: str) -> str:
    """Persist the latest Stage 4 runtime at its active block or done cache key."""
    cache_key = _stage4_checkpoint_cache_key(runtime)
    checkpoint_path = _stage4_checkpoint_path_for_cache_key(
        workspace_id,
        cache_key,
        create=True,
    )
    with storage.open_file(checkpoint_path, "wb") as f:
        cloudpickle.dump(runtime, f)
    cursor_path = _stage4_checkpoint_cursor_path(workspace_id, create=True)
    storage.write_text(cursor_path, json.dumps(_stage4_checkpoint_cursor_payload(runtime)))
    return checkpoint_path


def stage4_checkpoint_exists(workspace_id: str) -> bool:
    """Return ``True`` if a Stage 4 resume cursor is persisted for ``workspace_id``."""
    cursor_path = _stage4_checkpoint_cursor_path(workspace_id, create=False)
    return storage.exists(cursor_path)


def load_stage4_checkpoint(workspace_id: str) -> Any:
    """Load the persisted Stage 4 runtime addressed by the cursor file."""
    cursor_path = _stage4_checkpoint_cursor_path(workspace_id, create=False)
    if not storage.exists(cursor_path):
        raise FileNotFoundError(
            f"No Stage 4 checkpoint cursor found for workspace_id {workspace_id}"
        )
    cursor = storage.read_json(cursor_path)
    if not isinstance(cursor, dict):
        raise TypeError(f"Stage 4 checkpoint cursor for workspace_id {workspace_id} is not a dict")
    if cursor.get("kind") == "done":
        cache_key = STAGE4_DONE_CHECKPOINT_CACHE_KEY
    elif cursor.get("kind") == "block" and isinstance(cursor.get("block_id"), str):
        cache_key = str(cursor["block_id"])
    else:
        raise ValueError(f"Stage 4 checkpoint cursor for workspace_id {workspace_id} is invalid")
    return load_pickle(_stage4_checkpoint_path_for_cache_key(workspace_id, cache_key, create=False))


def clear_stage4_checkpoint(workspace_id: str) -> None:
    """Remove the Stage 4 resume cursor while retaining per-block cache files."""
    checkpoint_dir = _stage4_checkpoint_dir(workspace_id, create=False)
    if not storage.exists(checkpoint_dir):
        return
    cursor_path = _stage4_checkpoint_cursor_path(workspace_id, create=False)
    if storage.exists(cursor_path):
        storage.remove(cursor_path)


# ---------------------------------------------------------------------------
# Public JSON payloads (web-facing results)
# ---------------------------------------------------------------------------


def load_public_payload(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted web-facing stage payload."""
    path = storage.join(existing_run_dir(workspace_id), f"{stage_id}.json")
    if not storage.exists(path):
        raise FileNotFoundError(
            f"No public stage payload found for {stage_id} in workspace_id {workspace_id}"
        )
    payload = storage.read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Persisted payload for {stage_id} in workspace_id {workspace_id} is not a dict"
        )
    return payload


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
