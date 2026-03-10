"""Generic web-result persistence task.

Writes stage results as JSON to a well-known path so the web frontend
can fetch them via /api/results/[runId]/[stage].
"""

import logging

from prefect import task

from .. import get_prefect_logger
from .contracts import validate_stage_payload

logger = get_prefect_logger(__name__)


def _count_validation_issues(report: dict) -> tuple[int, int]:
    issues = report.get("issues", []) or []
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return error_count, warning_count


def _stage_summary(stage_id: str, data: dict) -> tuple[int, str]:
    outcome = data.get("outcome", "success")
    level = logging.WARNING if outcome in {"warn", "fail"} else logging.INFO

    if stage_id == "stage-0":
        date_range = data.get("date_range", {})
        return (
            logging.INFO,
            f"Stage 0 summary: source={data.get('source_label', 'unknown')} "
            f"records={data.get('n_records', 0)} columns={data.get('n_columns', 0)} "
            f"date_range={date_range.get('start') or '?'}..{date_range.get('end') or '?'}",
        )

    if stage_id == "stage-1a":
        latent = data.get("latent_model", {})
        return (
            logging.INFO,
            f"Stage 1a summary: constructs={len(latent.get('constructs', []))} "
            f"edges={len(latent.get('edges', []))} "
            f"treatments={len(data.get('treatments', []))} "
            f"outcome={data.get('outcome_name', '') or 'unknown'}",
        )

    if stage_id == "stage-1b":
        causal_spec = data.get("causal_spec", {})
        latent = causal_spec.get("latent", {})
        measurement = causal_spec.get("measurement", {})
        non_identifiable = (
            causal_spec.get("identifiability", {}).get("non_identifiable_treatments", {}) or {}
        )
        return (
            level,
            f"Stage 1b summary: constructs={len(latent.get('constructs', []))} "
            f"indicators={len(measurement.get('indicators', []))} "
            f"filtered_treatments={len(non_identifiable)} outcome={outcome}",
        )

    if stage_id == "stage-2":
        workers = data.get("workers", [])
        completed = sum(1 for worker in workers if worker.get("status") == "completed")
        failed = sum(1 for worker in workers if worker.get("status") == "failed")
        return (
            level,
            f"Stage 2 summary: workers={len(workers)} completed={completed} "
            f"failed={failed} sample_rows={len(data.get('combined_extractions_sample', []))} "
            f"indicators={len(data.get('per_indicator_counts', {}))} outcome={outcome}",
        )

    if stage_id == "stage-3":
        report = data.get("validation_report", {})
        errors, warnings = _count_validation_issues(report)
        return (
            level,
            f"Stage 3 summary: is_valid={report.get('is_valid', False)} "
            f"issues={len(report.get('issues', []) or [])} "
            f"errors={errors} warnings={warnings} outcome={outcome}",
        )

    if stage_id == "stage-4":
        model_spec = data.get("model_spec", {})
        return (
            logging.INFO,
            f"Stage 4 summary: parameters={len(model_spec.get('parameters', []))} "
            f"likelihoods={len(model_spec.get('likelihoods', []))} "
            f"priors={len(data.get('priors', {}))} "
            f"prior_predictive_channels={len(data.get('prior_predictive_samples', {}) or {})}",
        )

    if stage_id == "stage-4b":
        parametric_id = data.get("parametric_id", {})
        summary = parametric_id.get("summary", {})
        t_rule = parametric_id.get("t_rule", {})
        return (
            level,
            f"Stage 4b summary: checked={parametric_id.get('checked', False)} "
            f"t_rule={'pass' if t_rule.get('satisfies', True) else 'fail'} "
            f"structural_issues={len(summary.get('structural_issues', []) or [])} "
            f"boundary_issues={len(summary.get('boundary_issues', []) or [])} "
            f"weak_params={len(summary.get('weak_params', []) or [])} outcome={outcome}",
        )

    if stage_id == "stage-5":
        power_scaling = data.get("power_scaling", [])
        ps_issues = sum(
            1
            for item in power_scaling
            if item.get("diagnosis") in {"prior_dominated", "prior_data_conflict"}
        )
        ppc_warnings = len(data.get("ppc", {}).get("per_variable_warnings", []) or [])
        inference = data.get("inference_metadata", {})
        return (
            level,
            f"Stage 5 summary: method={inference.get('method', 'unknown')} "
            f"samples={inference.get('n_samples', '?')} "
            f"power_scaling_issues={ps_issues} ppc_warnings={ppc_warnings} outcome={outcome}",
        )

    if stage_id == "stage-6":
        interventions = data.get("intervention_results", [])
        warnings = sum(
            1
            for result in interventions
            if result.get("warning")
            or result.get("ppc_warnings")
            or result.get("prior_sensitivity_warning")
        )
        return (
            level,
            f"Stage 6 summary: treatments_ranked={len(interventions)} "
            f"warnings={warnings} outcome={outcome}",
        )

    return logging.INFO, f"{stage_id} summary: persisted result payload"


@task(
    result_serializer="json",
    result_storage_key="{parameters[run_id]}/{parameters[stage_id]}.json",
    task_run_name="persist-{stage_id}",
)
def persist_web_result(stage_id: str, data: dict, run_id: str) -> dict:
    """Persist stage result for web frontend consumption.

    Uses Prefect's result persistence to write the data as JSON
    to ``results/{run_id}/{stage_id}.json``.

    Args:
        stage_id: Stage identifier (e.g. "stage-0", "stage-4").
        data: Web-shaped dict matching the frontend's StageXData contract.
        run_id: Root pipeline flow run identifier used for result storage.

    Returns:
        Validated stage payload dict (Prefect serialises the return value).
    """
    validated = validate_stage_payload(stage_id, data)
    logger.debug("Persisting %s result to results/%s/%s.json", stage_id, run_id, stage_id)
    level, summary = _stage_summary(stage_id, validated)
    logger.log(level, summary)
    return validated
