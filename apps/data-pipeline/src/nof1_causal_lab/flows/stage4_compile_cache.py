"""Stage 4 model-lock JAX compilation cache sidecar management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.utils import storage

from .run_store import (
    STAGE2_MODEL_PARQUET_FILENAMES,
    STAGE4_JAX_CACHE_FILENAMES,
    STAGE4_JAX_CACHE_METADATA_FILENAMES,
    ensure_run_dir,
    find_run_artifact,
    load_json,
    load_parquet,
    save_json,
)

logger = get_prefect_logger(__name__)

_STAGE4_COMPILE_CACHE_SCHEMA_VERSION = 1
_STAGE4_COMPILE_CACHE_ACCELERATOR = "modal-a100-80gb"
_STAGE4_COMPILE_CACHE_WAIT_TIMEOUT_SECONDS = 3600


def _archive_path(workspace_id: str) -> str:
    return storage.join(ensure_run_dir(workspace_id), STAGE4_JAX_CACHE_FILENAMES[0])


def _build_metadata(
    *,
    status: str,
    topology_fingerprint: str,
    function_call_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": _STAGE4_COMPILE_CACHE_SCHEMA_VERSION,
        "status": status,
        "topology_fingerprint": topology_fingerprint,
        "accelerator": _STAGE4_COMPILE_CACHE_ACCELERATOR,
    }
    if function_call_id is not None:
        payload["function_call_id"] = function_call_id
    if error is not None:
        payload["error"] = error
    return payload


def load_stage4_compile_cache_metadata(workspace_id: str) -> dict[str, Any] | None:
    """Load Stage 4 compile-cache metadata, if present."""
    try:
        path = find_run_artifact(workspace_id, STAGE4_JAX_CACHE_METADATA_FILENAMES)
    except FileNotFoundError:
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def save_stage4_compile_cache_metadata(
    workspace_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist Stage 4 compile-cache metadata."""
    save_json(metadata, workspace_id, STAGE4_JAX_CACHE_METADATA_FILENAMES[0])
    return metadata


def compiled_ssm_topology_fingerprint(compiled_ssm: dict[str, Any]) -> str:
    """Hash the topology-defining portion of a compiled SSM artifact."""
    spec_payload = compiled_ssm.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("Compiled artifact is missing required 'spec'")
    return hashlib.sha256(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metadata_matches(metadata: dict[str, Any] | None, topology_fingerprint: str) -> bool:
    return (
        isinstance(metadata, dict)
        and metadata.get("topology_fingerprint") == topology_fingerprint
        and metadata.get("schema_version") == _STAGE4_COMPILE_CACHE_SCHEMA_VERSION
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


def _reset_cache_dir(cache_dir: Path) -> None:
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _archive_cache_dir(cache_dir: Path, workspace_id: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with tarfile.open(tmp.name, mode="w:gz") as archive:
            for item in sorted(cache_dir.iterdir(), key=lambda path: path.name):
                archive.add(item, arcname=item.name)
        with (
            Path(tmp.name).open("rb") as src,
            storage.open_file(_archive_path(workspace_id), "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)


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
            timeout=_STAGE4_COMPILE_CACHE_WAIT_TIMEOUT_SECONDS
        )
    except (ModalError, TimeoutError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("Stage 4 compile cache warmup wait failed: %s", exc)
        return False
    return True


def restore_stage4_compile_cache(
    workspace_id: str | None,
    compiled_ssm: dict[str, Any] | None,
    *,
    wait_for_pending: bool,
) -> bool:
    """Restore a topology-matched compile cache sidecar into the local JAX cache dir."""
    if workspace_id is None or compiled_ssm is None:
        return False

    topology_fingerprint = compiled_ssm_topology_fingerprint(compiled_ssm)
    metadata = load_stage4_compile_cache_metadata(workspace_id)
    if not _metadata_matches(metadata, topology_fingerprint):
        return False

    if metadata.get("status") == "pending" and wait_for_pending:
        _wait_for_pending_compile_cache(metadata)
        metadata = load_stage4_compile_cache_metadata(workspace_id)

    if not _metadata_matches(metadata, topology_fingerprint):
        return False
    if metadata.get("status") != "ready":
        return False

    restored = _restore_cache_archive(workspace_id)
    if restored:
        logger.info(
            "Restored Stage 4 compile cache for topology %s",
            topology_fingerprint[:12],
        )
    return restored


def dispatch_stage4_model_compile_warmup(
    workspace_id: str,
    model_spec: dict[str, Any],
    causal_spec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Launch the Modal A100 warmup job once the Stage 4 model spec is locked."""
    if not storage.is_remote():
        return None

    from nof1_causal_lab.models.ssm_compiler import compile_ssm_artifact_with_default_priors

    warmup_artifact = compile_ssm_artifact_with_default_priors(model_spec, causal_spec=causal_spec)
    topology_fingerprint = compiled_ssm_topology_fingerprint(warmup_artifact)
    current = load_stage4_compile_cache_metadata(workspace_id)
    if _metadata_matches(current, topology_fingerprint):
        status = current.get("status")
        if status == "ready" and _archive_exists(workspace_id):
            return current
        if status == "pending" and isinstance(current.get("function_call_id"), str):
            return current

    pending = save_stage4_compile_cache_metadata(
        workspace_id,
        _build_metadata(status="pending", topology_fingerprint=topology_fingerprint),
    )

    from nof1_causal_lab.flows.modal_runners import spawn_stage4_model_compile_warmup

    function_call_id = spawn_stage4_model_compile_warmup(
        workspace_id=workspace_id,
        model_spec=model_spec,
        causal_spec=causal_spec,
        topology_fingerprint=topology_fingerprint,
    )
    current = load_stage4_compile_cache_metadata(workspace_id)
    if _metadata_matches(current, topology_fingerprint) and current.get("status") == "pending":
        pending = save_stage4_compile_cache_metadata(
            workspace_id,
            _build_metadata(
                status="pending",
                topology_fingerprint=topology_fingerprint,
                function_call_id=function_call_id,
            ),
        )
    return pending


def warm_stage4_compile_cache_artifact(
    *,
    workspace_id: str,
    model_spec: dict[str, Any],
    causal_spec: dict[str, Any] | None,
    topology_fingerprint: str,
) -> dict[str, Any]:
    """Build and persist the Stage 4 compile-cache sidecar on a Modal A100 worker."""
    from nof1_causal_lab.models.ssm_compiler import compile_ssm_artifact_with_default_priors

    try:
        compiled_ssm = compile_ssm_artifact_with_default_priors(
            model_spec,
            causal_spec=causal_spec,
        )
        actual_fingerprint = compiled_ssm_topology_fingerprint(compiled_ssm)
        if actual_fingerprint != topology_fingerprint:
            raise ValueError(
                "Stage 4 compile-cache warmup topology mismatch: "
                f"expected {topology_fingerprint}, got {actual_fingerprint}"
            )

        data_for_model = load_parquet(
            find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
        )
        _reset_cache_dir(_jax_persistent_cache_dir())
        _warm_compiled_ssm_runtime(compiled_ssm, data_for_model)
        _archive_cache_dir(_jax_persistent_cache_dir(), workspace_id)
        return save_stage4_compile_cache_metadata(
            workspace_id,
            _build_metadata(status="ready", topology_fingerprint=topology_fingerprint),
        )
    except Exception as exc:
        save_stage4_compile_cache_metadata(
            workspace_id,
            _build_metadata(
                status="failed",
                topology_fingerprint=topology_fingerprint,
                error=str(exc),
            ),
        )
        raise


def _warm_compiled_ssm_runtime(compiled_ssm: dict[str, Any], data_for_model: Any) -> None:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax.flatten_util import ravel_pytree

    from nof1_causal_lab.models.ssm.inference.utils import _build_eval_fns, _discover_sites
    from nof1_causal_lab.models.ssm_builder import prepare_model_runtime

    runtime = prepare_model_runtime(data_for_model=data_for_model, compiled_ssm=compiled_ssm)
    backend = runtime.model.make_likelihood_backend()
    site_info = _discover_sites(
        runtime.model,
        runtime.observations,
        runtime.times,
        random.PRNGKey(0),
        backend,
    )
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    z0 = jnp.zeros_like(flat_example)
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        runtime.model,
        runtime.observations,
        runtime.times,
        site_info,
        unravel_fn,
        backend,
    )

    def _neg_log_posterior(z):
        return -(log_lik_fn(z) + log_prior_unc_fn(z))

    value_and_grad = jax.jit(jax.value_and_grad(_neg_log_posterior))

    jax.device_get(log_prior_unc_fn(z0))
    jax.device_get(log_lik_fn(z0))
    jax.device_get(value_and_grad(z0))
