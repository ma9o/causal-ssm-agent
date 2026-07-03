"""Web-facing stage payload projection.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[workspace_id]/[stage]. This is a read
model derived from the versioned artifact store — never a source of
truth.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import runs_dir

from .stage_contracts import _validate_stage_model

logger = logging.getLogger(__name__)


def _normalize_nonfinite_json_values(value: Any) -> Any:
    """Replace non-finite numeric values with null-compatible ``None`` recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _normalize_nonfinite_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_nonfinite_json_values(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_nonfinite_json_values(item) for item in value]
    return value


def _persist_payload(stage_id: str, payload: dict, workspace_id: str) -> None:
    """Write a validated JSON payload to the well-known stage result path."""
    path = storage.join(runs_dir(workspace_id), f"{stage_id}.json")
    storage.makedirs(runs_dir(workspace_id))
    storage.write_text(path, json.dumps(payload, allow_nan=False))
    logger.debug("Persisted %s result to %s", stage_id, path)


def persist_validated_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
    """Validate a raw dict and persist the stage's public web payload."""
    model = _validate_stage_model(stage_id, data)
    payload = _normalize_nonfinite_json_values(model.model_dump(mode="json"))
    _persist_payload(stage_id, payload, workspace_id)

    summary = getattr(model, "summary_message", None)
    if callable(summary):
        logger.info(summary())
    return payload


def persist_web_patch(stage_id: str, patch: dict, workspace_id: str) -> dict:
    """Merge a web-payload patch, validate it, and persist it."""
    from .run_store import load_public_payload

    current = load_public_payload(workspace_id, stage_id)
    return persist_validated_web_result(stage_id, {**current, **patch}, workspace_id)
