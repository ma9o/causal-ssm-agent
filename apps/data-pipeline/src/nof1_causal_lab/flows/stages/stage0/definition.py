"""Stage 0 pipeline definition."""

from __future__ import annotations

from causal_ssm_agent.flows.stage_runtime import PipelineContext, StageDefinition
from causal_ssm_agent.flows.stages.stage0.contracts import Stage0Contract


def _bind_stage0(ctx: PipelineContext, _states: dict) -> dict:
    return {"workspace_id": ctx.workspace_id}


def build_stage0_definition() -> StageDefinition:
    from causal_ssm_agent.flows import dag

    return StageDefinition(
        stage_id="stage-0",
        depends_on=frozenset(),
        contract=Stage0Contract,
        bind_inputs=_bind_stage0,
        runner=dag.stage0,
    )
