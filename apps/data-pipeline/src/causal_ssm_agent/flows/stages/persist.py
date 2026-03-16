"""Generic web-result persistence task.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[user_id]/[stage].
"""

import json

from prefect import task

from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.data import runs_dir

from .. import get_prefect_logger
from .contracts import _validate_stage_model

logger = get_prefect_logger(__name__)


@task(
    task_run_name="persist-{stage_id}",
)
def persist_web_result(stage_id: str, data: dict, user_id: str) -> dict:
    """Persist stage result for web frontend consumption.

    Writes validated data as JSON to ``data/{user_id}/run/{stage_id}.json``
    via the storage backend (local filesystem or R2).

    Args:
        stage_id: Stage identifier (e.g. "stage-0", "stage-4").
        data: Web-shaped dict matching the frontend's StageXData contract.
        user_id: User ID used for result storage path.

    Returns:
        Validated stage payload dict.
    """
    model = _validate_stage_model(stage_id, data)
    payload = model.model_dump(mode="json")

    path = storage.join(runs_dir(user_id), f"{stage_id}.json")
    storage.makedirs(runs_dir(user_id))
    storage.write_text(path, json.dumps(payload))

    logger.debug("Persisted %s result to %s", stage_id, path)
    level, summary = model.summarize()
    logger.log(level, summary)
    return payload
