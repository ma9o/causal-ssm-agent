"""Run-level artifact persistence for the causal inference pipeline.

Centralises all file-system I/O for pipeline run artifacts:
parquet DataFrames, pickled objects, stage snapshots, and public
JSON payloads.  Every module that needs to save or load run artifacts
should import from here instead of duplicating path logic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.data import runs_dir

from . import get_prefect_logger

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Filename constants for run artifacts
# ---------------------------------------------------------------------------

STAGE0_PARQUET_FILENAMES = ("stage0-raw-input.parquet", "stage2-raw-input.parquet")
STAGE2_RAW_PARQUET_FILENAMES = ("stage2-raw-data.parquet",)
STAGE2_MODEL_PARQUET_FILENAMES = ("stage2-model-data.parquet",)
STAGE5B_PICKLE_FILENAMES = ("stage5b-fitted-result.pkl",)

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


def save_parquet(df: Any, workspace_id: str, filename: str) -> str:
    """Write a Polars DataFrame to parquet in the run directory."""
    path = storage.join(ensure_run_dir(workspace_id), filename)
    if storage.is_remote():
        with storage.get_fs().open(path, "wb") as f:
            df.write_parquet(f)
    else:
        df.write_parquet(path)
    return path


def load_parquet(path: str) -> Any:
    """Read a Polars DataFrame from a parquet path."""
    import polars as pl

    return pl.read_parquet(path, storage_options=storage.polars_storage_options())


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------


def save_pickle(value: Any, workspace_id: str, filename: str) -> str:
    """Pickle a value into the run directory."""
    path = storage.join(ensure_run_dir(workspace_id), filename)
    with storage.open_file(path, "wb") as f:
        cloudpickle.dump(value, f)
    return path


def load_pickle(path: str) -> Any:
    """Unpickle a value from storage."""
    with storage.open_file(path, "rb") as f:
        return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Stage snapshots (full internal state)
# ---------------------------------------------------------------------------


def save_stage_snapshot(stage_id: str, state: dict[str, Any], workspace_id: str) -> None:
    """Persist full stage state (result + web + gate) for resume."""
    path = storage.join(ensure_run_dir(workspace_id), f"{stage_id}-state.pkl")
    with storage.open_file(path, "wb") as f:
        cloudpickle.dump(state, f)


def load_stage_snapshot(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load a previously saved stage snapshot."""
    path = storage.join(existing_run_dir(workspace_id), f"{stage_id}-state.pkl")
    if not storage.exists(path):
        raise FileNotFoundError(
            f"No stage snapshot found for {stage_id} in workspace_id {workspace_id}"
        )
    with storage.open_file(path, "rb") as f:
        return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Public JSON payloads (web-facing results)
# ---------------------------------------------------------------------------


def _unwrap_persisted_result(raw: Any) -> Any:
    """Strip Prefect's result wrapper if present."""
    if isinstance(raw, dict) and "result" in raw:
        raw = raw["result"]
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def load_public_payload(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted web-facing stage payload."""
    path = storage.join(existing_run_dir(workspace_id), f"{stage_id}.json")
    if not storage.exists(path):
        raise FileNotFoundError(
            f"No public stage payload found for {stage_id} in workspace_id {workspace_id}"
        )
    raw = storage.read_json(path)
    payload = _unwrap_persisted_result(raw)
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


# ---------------------------------------------------------------------------
# Stage lifecycle helpers
# ---------------------------------------------------------------------------


def stage_state(
    result: dict[str, Any],
    web: dict[str, Any],
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical stage state dict."""
    state: dict[str, Any] = {"result": result, "web": web}
    if gate is not None:
        state["gate"] = gate
    return state


def finalize_stage(
    stage_id: str,
    result: dict[str, Any],
    workspace_id: str,
    *,
    extras: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    contract: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Build web payload, persist it, save snapshot, return stage state.

    Combines the former ``_web_payload`` + ``_stage_state`` +
    ``_finalize_stage_state`` into a single call.
    """
    from .stages import persist_web_result
    from .stages.contracts import STAGE_CONTRACTS, StageId

    stage_contract = contract or STAGE_CONTRACTS[cast("StageId", stage_id)]
    contract_fields = set(stage_contract.model_fields.keys())
    web = {k: v for k, v in result.items() if k in contract_fields}
    if extras:
        web.update(extras)
    web = persist_web_result(stage_id, web, workspace_id)

    state = stage_state(result, web, gate=gate)
    save_stage_snapshot(stage_id, state, workspace_id)
    return state


# ---------------------------------------------------------------------------
# Task result helpers
# ---------------------------------------------------------------------------


def unwrap_task_result(task_or_value: Any) -> Any:
    """Extract the result from a Prefect task return, or pass through raw values.

    Prefect tasks may return either a raw value or a future-like object with a
    ``.result()`` method.  This helper normalises both to a plain value.
    """
    if hasattr(task_or_value, "result"):
        return task_or_value.result()
    return task_or_value
