"""Stage 5a: SVI preflight helpers."""

from __future__ import annotations

from typing import Any

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.run_store import load_parquet, unwrap_task_result

logger = get_prefect_logger(__name__)


def build_stage5a_svi_attempts() -> list[dict[str, Any]]:
    """Return the universal cheap SVI preflight ladder.

    Stage 5a always runs, but it should not depend on one brittle optimizer
    configuration. Use a bounded attempt ladder that keeps the stage cheap and
    non-blocking while giving harder models a second, more stable try.
    """
    from causal_ssm_agent.utils.config import get_config

    svi_config = get_config().inference.svi
    sample_count = min(250, get_config().inference.num_samples)

    return [
        {
            "method": "svi",
            "guide_type": "delta",
            "learning_rate": min(float(svi_config.learning_rate), 1e-4),
            "num_steps": min(int(svi_config.num_steps), 150),
            "num_samples": sample_count,
        },
        {
            "method": "svi",
            "guide_type": "normal",
            "learning_rate": min(float(svi_config.learning_rate), 1e-4),
            "init_scale": 0.01,
            "num_steps": min(int(svi_config.num_steps), 500),
            "num_samples": sample_count,
        },
        {
            "method": "svi",
            "guide_type": "mvn",
            "learning_rate": min(float(svi_config.learning_rate), 3e-4),
            "init_scale": 0.01,
            "num_steps": min(int(svi_config.num_steps), 750),
            "num_samples": sample_count,
        },
    ]


def run_stage5a_preflight(
    stage4: dict,
    stage2: dict,
    workspace_id: str,
) -> dict:
    """Run Stage 5a SVI preflight with a bounded attempt ladder."""
    from causal_ssm_agent.flows.stages.stage5b.fit import fit_model

    data_for_model = load_parquet(stage2["_data_for_model_path"])

    total_duration = 0.0
    attempts = build_stage5a_svi_attempts()
    fitted_result: dict[str, Any] | None = None

    for index, svi_config in enumerate(attempts, start=1):
        logger.info(
            "Stage 5a SVI attempt %d/%d: guide=%s lr=%s steps=%s samples=%s",
            index,
            len(attempts),
            svi_config.get("guide_type"),
            svi_config.get("learning_rate"),
            svi_config.get("num_steps"),
            svi_config.get("num_samples"),
        )
        fitted = fit_model(
            stage4.get("_compiled_ssm"),
            data_for_model,
            sampler_config=svi_config,
            workspace_id=workspace_id,
            wait_for_compile_cache=False,
        )
        fitted_result = unwrap_task_result(fitted)
        total_duration += float(fitted_result.get("duration_seconds", 0.0))
        if fitted_result.get("fitted", False):
            break

        logger.warning(
            "Stage 5a SVI attempt %d/%d failed: %s",
            index,
            len(attempts),
            fitted_result.get("error", "unknown"),
        )

    if not fitted_result or not fitted_result.get("fitted", False):
        return {
            "inference_metadata": {
                "method": "svi",
                "n_samples": 0,
                "duration_seconds": total_duration,
            },
            "svi_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
            "outcome": "warn",
        }

    return {
        "inference_metadata": {
            "method": "svi",
            "n_samples": int(fitted_result.get("n_samples", 0)),
            "duration_seconds": total_duration,
        },
        "svi_diagnostics": fitted_result.get("svi_diagnostics"),
        "posterior_marginals": fitted_result.get("posterior_marginals"),
        "posterior_pairs": fitted_result.get("posterior_pairs"),
        "outcome": "success",
    }
