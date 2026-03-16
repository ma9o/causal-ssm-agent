"""Run-level artifact persistence for the causal inference pipeline.

Centralises all file-system I/O for pipeline run artifacts:
parquet DataFrames, pickled objects, stage snapshots, and public
JSON payloads.  Every module that needs to save or load run artifacts
should import from here instead of duplicating path logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cloudpickle

from causal_ssm_agent.utils.data import runs_dir

from . import get_prefect_logger

logger = get_prefect_logger(__name__)

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


def ensure_run_dir(user_id: str) -> Path:
    """Return the run directory, creating it if needed."""
    path = runs_dir(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def existing_run_dir(user_id: str) -> Path:
    """Return the run directory, raising if it doesn't exist."""
    path = runs_dir(user_id)
    if not path.exists():
        raise FileNotFoundError(f"No results directory found for user_id {user_id}")
    return path


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def save_parquet(df: Any, user_id: str, filename: str) -> str:
    """Write a Polars DataFrame to parquet in the run directory."""
    path = ensure_run_dir(user_id) / filename
    df.write_parquet(path)
    return str(path)


def load_parquet(path: str) -> Any:
    """Read a Polars DataFrame from a parquet path."""
    import polars as pl

    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------


def save_pickle(value: Any, user_id: str, filename: str) -> str:
    """Pickle a value into the run directory."""
    path = ensure_run_dir(user_id) / filename
    with path.open("wb") as f:
        cloudpickle.dump(value, f)
    return str(path)


def load_pickle(path: str) -> Any:
    """Unpickle a value from disk."""
    with Path(path).open("rb") as f:
        return cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Stage snapshots (full internal state)
# ---------------------------------------------------------------------------


def save_stage_snapshot(stage_id: str, state: dict[str, Any], user_id: str) -> None:
    """Persist full stage state (result + web + gate) for resume."""
    path = ensure_run_dir(user_id) / f"{stage_id}-state.pkl"
    with path.open("wb") as f:
        cloudpickle.dump(state, f)


def load_stage_snapshot(user_id: str, stage_id: str) -> dict[str, Any]:
    """Load a previously saved stage snapshot."""
    path = existing_run_dir(user_id) / f"{stage_id}-state.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No stage snapshot found for {stage_id} in user_id {user_id}")
    with path.open("rb") as f:
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


def load_public_payload(user_id: str, stage_id: str) -> dict[str, Any]:
    """Load a persisted web-facing stage payload."""
    path = existing_run_dir(user_id) / f"{stage_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No public stage payload found for {stage_id} in user_id {user_id}"
        )
    with path.open() as f:
        raw = json.load(f)
    payload = _unwrap_persisted_result(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Persisted payload for {stage_id} in user_id {user_id} is not a dict")
    return payload


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def find_run_artifact(user_id: str, filenames: tuple[str, ...]) -> str:
    """Return the path of the first existing artifact from a list of candidates."""
    run_dir = existing_run_dir(user_id)
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            return str(path)
    expected = ", ".join(filenames)
    raise FileNotFoundError(f"None of [{expected}] exist for user_id {user_id}")


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
    user_id: str,
    *,
    extras: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build web payload, persist it, save snapshot, return stage state.

    Combines the former ``_web_payload`` + ``_stage_state`` +
    ``_finalize_stage_state`` into a single call.
    """
    from .stages import persist_web_result
    from .stages.contracts import STAGE_CONTRACTS, StageId

    contract_fields = set(STAGE_CONTRACTS[cast("StageId", stage_id)].model_fields.keys())
    web = {k: v for k, v in result.items() if k in contract_fields}
    if extras:
        web.update(extras)
    web = persist_web_result(stage_id, web, user_id)

    state = stage_state(result, web, gate=gate)
    save_stage_snapshot(stage_id, state, user_id)
    return state
