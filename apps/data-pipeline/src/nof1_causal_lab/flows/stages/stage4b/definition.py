"""Stage 4b pipeline definition."""

from __future__ import annotations

from causal_ssm_agent.flows.stage_runtime import PipelineContext, StageDefinition
from causal_ssm_agent.flows.stages.stage4b.contracts import Stage4bContract


def _bind_stage4b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
    }


def build_stage4b_definition() -> StageDefinition:
    from causal_ssm_agent.flows import dag

    return StageDefinition(
        stage_id="stage-4b",
        depends_on=frozenset({"stage-4", "stage-2"}),
        contract=Stage4bContract,
        bind_inputs=_bind_stage4b,
        runner=dag.stage4b,
    )
