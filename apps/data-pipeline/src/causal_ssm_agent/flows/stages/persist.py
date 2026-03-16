"""Generic web-result persistence task.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[user_id]/[stage].
"""

from prefect import task

from causal_ssm_agent.utils.data import DATA_DIR

from .. import get_prefect_logger
from .contracts import _validate_stage_model

logger = get_prefect_logger(__name__)


@task(
    result_storage=DATA_DIR,
    result_serializer="json",
    result_storage_key="{parameters[user_id]}/run/{parameters[stage_id]}.json",
    task_run_name="persist-{stage_id}",
    refresh_cache=True,
)
def persist_web_result(stage_id: str, data: dict, user_id: str) -> dict:
    """Persist stage result for web frontend consumption.

    Uses Prefect's result persistence to write the data as JSON
    to ``data/{user_id}/run/{stage_id}.json``.

    Args:
        stage_id: Stage identifier (e.g. "stage-0", "stage-4").
        data: Web-shaped dict matching the frontend's StageXData contract.
        user_id: User ID used for result storage path.

    Returns:
        Validated stage payload dict (Prefect serialises the return value).
    """
    model = _validate_stage_model(stage_id, data)
    logger.debug("Persisting %s result to data/%s/run/%s.json", stage_id, user_id, stage_id)
    level, summary = model.summarize()
    logger.log(level, summary)
    return model.model_dump(mode="json")
