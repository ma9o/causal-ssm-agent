"""Generic web-result persistence task.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[workspace_id]/[stage].
"""

import json

from prefect import task

from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.data import runs_dir

from .. import get_prefect_logger
from .contracts import _validate_stage_model

logger = get_prefect_logger(__name__)


def persist_validated_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
    """Validate and persist a stage's public web payload."""
    model = _validate_stage_model(stage_id, data)
    payload = model.model_dump(mode="json")

    path = storage.join(runs_dir(workspace_id), f"{stage_id}.json")
    storage.makedirs(runs_dir(workspace_id))
    storage.write_text(path, json.dumps(payload))

    logger.debug("Persisted %s result to %s", stage_id, path)
    level, summary = model.summarize()
    logger.log(level, summary)
    return payload


def persist_web_patch(stage_id: str, patch: dict, workspace_id: str) -> dict:
    """Merge a web-payload patch, validate it, persist it, and refresh the snapshot web state."""
    from ..run_store import load_public_payload, load_stage_snapshot, save_stage_snapshot

    current = load_public_payload(workspace_id, stage_id)
    payload = persist_validated_web_result(stage_id, {**current, **patch}, workspace_id)

    try:
        snapshot = load_stage_snapshot(workspace_id, stage_id)
    except FileNotFoundError:
        return payload

    snapshot["web"] = payload
    save_stage_snapshot(stage_id, snapshot, workspace_id)
    return payload


@task(
    task_run_name="persist-{stage_id}",
)
def persist_web_result(stage_id: str, data: dict, workspace_id: str) -> dict:
    """Persist stage result for web frontend consumption.

    Writes validated data as JSON to ``data/{workspace_id}/run/{stage_id}.json``
    via the storage backend (local filesystem or R2).

    Args:
        stage_id: Stage identifier (e.g. "stage-0", "stage-4").
        data: Web-shaped dict matching the frontend's StageXData contract.
        workspace_id: Workspace ID used for result storage path.

    Returns:
        Validated stage payload dict.
    """
    return persist_validated_web_result(stage_id, data, workspace_id)
