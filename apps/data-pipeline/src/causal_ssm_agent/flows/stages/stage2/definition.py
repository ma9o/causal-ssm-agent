"""Stage 2 pipeline definition."""

from __future__ import annotations

import os

from causal_ssm_agent.flows.stage_runtime import PipelineContext, StageDefinition
from causal_ssm_agent.flows.stages.stage2.contracts import Stage2Contract


def _bind_stage2(ctx: PipelineContext, states: dict) -> dict:
    from causal_ssm_agent.utils.config import get_config

    return {
        "question": ctx.question,
        "stage0": states["stage-0"],
        "stage1b": states["stage-1b"],
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
        "max_windows": None
        if ctx.openrouter_access_mode in {"user", "local"}
        or os.environ.get("DEPLOYMENT_ENV") != "production"
        else get_config().stage2_workers.max_free_windows,
    }


def build_stage2_definition() -> StageDefinition:
    from causal_ssm_agent.flows import dag

    return StageDefinition(
        stage_id="stage-2",
        depends_on=frozenset({"stage-0", "stage-1b"}),
        contract=Stage2Contract,
        bind_inputs=_bind_stage2,
        runner=dag.stage2,
        question_required=True,
    )
