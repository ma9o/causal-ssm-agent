"""Generic web-result persistence task.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[runId]/[stage].
"""

from prefect import task

from .contracts import validate_stage_payload


@task(
    result_serializer="json",
    result_storage_key="{flow_run.id}/{parameters[stage_id]}.json",
    task_run_name="persist-{stage_id}",
)
def persist_web_result(stage_id: str, data: dict) -> dict:
    """Persist stage result for web frontend consumption.

    Uses Prefect's result persistence to write the data as JSON
    to ``results/{flow_run.id}/{stage_id}.json``.

    Args:
        stage_id: Stage identifier (e.g. "stage-0", "stage-4").
        data: Web-shaped dict matching the frontend's StageXData contract.

    Returns:
        Validated stage payload dict (Prefect serialises the return value).
    """
    return validate_stage_payload(stage_id, data)
