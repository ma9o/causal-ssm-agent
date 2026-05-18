"""Stage 3 pipeline definition."""

from __future__ import annotations

from nof1_causal_lab.flows.stage_runtime import PipelineContext, StageDefinition
from nof1_causal_lab.flows.stages.stage3.contracts import Stage3Contract


def _bind_stage3(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage1b": states["stage-1b"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
    }


def build_stage3_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-3",
        depends_on=frozenset({"stage-1b", "stage-2"}),
        contract=Stage3Contract,
        bind_inputs=_bind_stage3,
        runner=dag.stage3,
    )
