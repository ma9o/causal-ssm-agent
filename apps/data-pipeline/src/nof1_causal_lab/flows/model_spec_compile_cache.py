"""model-spec model-lock JAX compilation cache sidecar management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Any

from typing_extensions import TypeIs

from nof1_causal_lab.utils import storage

from .run_store import (
    MODEL_SPEC_JAX_CACHE_FILENAMES,
    MODEL_SPEC_JAX_CACHE_METADATA_FILENAMES,
    ensure_run_dir,
    find_run_artifact,
    load_json,
)

logger = logging.getLogger(__name__)

_MODEL_SPEC_COMPILE_CACHE_SCHEMA_VERSION = 1
_MODEL_SPEC_COMPILE_CACHE_WAIT_TIMEOUT_SECONDS = 3600


def _archive_path(workspace_id: str) -> str:
    return storage.join(ensure_run_dir(workspace_id), MODEL_SPEC_JAX_CACHE_FILENAMES[0])


def load_model_spec_compile_cache_metadata(workspace_id: str) -> dict[str, Any] | None:
    """Load model-spec compile-cache metadata, if present."""
    try:
        path = find_run_artifact(workspace_id, MODEL_SPEC_JAX_CACHE_METADATA_FILENAMES)
    except FileNotFoundError:
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def compiled_ssm_topology_fingerprint(compiled_ssm: dict[str, Any]) -> str:
    """Hash the topology-defining portion of a compiled SSM artifact."""
    spec_payload = compiled_ssm.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("Compiled artifact is missing required 'spec'")
    return hashlib.sha256(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metadata_matches(
    metadata: dict[str, Any] | None, topology_fingerprint: str
) -> TypeIs[dict[str, Any]]:
    return (
        isinstance(metadata, dict)
        and metadata.get("topology_fingerprint") == topology_fingerprint
        and metadata.get("schema_version") == _MODEL_SPEC_COMPILE_CACHE_SCHEMA_VERSION
    )


def _archive_exists(workspace_id: str) -> bool:
    return storage.exists(_archive_path(workspace_id))


def _jax_persistent_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "nof1-causal-lab" / "jax"
    override = os.getenv("JAX_COMPILATION_CACHE_DIR")
    if isinstance(override, str) and override:
        cache_dir = Path(override)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _restore_cache_archive(workspace_id: str) -> bool:
    if not _archive_exists(workspace_id):
        return False
    cache_dir = _jax_persistent_cache_dir()
    with (
        storage.open_file(_archive_path(workspace_id), "rb") as src,
        tarfile.open(fileobj=src, mode="r:gz") as archive,
    ):
        archive.extractall(cache_dir)
    return True


def _wait_for_pending_compile_cache(metadata: dict[str, Any]) -> bool:
    function_call_id = metadata.get("function_call_id")
    if not isinstance(function_call_id, str) or not function_call_id:
        return False

    import modal
    from modal.exception import Error as ModalError

    try:
        modal.FunctionCall.from_id(function_call_id).get(
            timeout=_MODEL_SPEC_COMPILE_CACHE_WAIT_TIMEOUT_SECONDS
        )
    except (ModalError, TimeoutError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("model-spec compile cache warmup wait failed: %s", exc)
        return False
    return True


def restore_model_spec_compile_cache(
    workspace_id: str | None,
    compiled_ssm: dict[str, Any] | None,
    *,
    wait_for_pending: bool,
) -> bool:
    """Restore a topology-matched compile cache sidecar into the local JAX cache dir."""
    if workspace_id is None or compiled_ssm is None:
        return False

    topology_fingerprint = compiled_ssm_topology_fingerprint(compiled_ssm)
    metadata = load_model_spec_compile_cache_metadata(workspace_id)
    if not _metadata_matches(metadata, topology_fingerprint):
        return False

    if metadata.get("status") == "pending" and wait_for_pending:
        _wait_for_pending_compile_cache(metadata)
        metadata = load_model_spec_compile_cache_metadata(workspace_id)

    if not _metadata_matches(metadata, topology_fingerprint):
        return False
    if metadata.get("status") != "ready":
        return False

    restored = _restore_cache_archive(workspace_id)
    if restored:
        logger.info(
            "Restored model-spec compile cache for topology %s",
            topology_fingerprint[:12],
        )
    return restored
