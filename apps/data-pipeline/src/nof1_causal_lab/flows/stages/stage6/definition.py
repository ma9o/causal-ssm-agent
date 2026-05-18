"""Stage 6 pipeline definition."""

from __future__ import annotations

from nof1_causal_lab.flows.stage_runtime import PipelineContext, StageDefinition
from nof1_causal_lab.flows.stages.stage6.contracts import Stage6Contract


def _bind_stage6(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage5b": states["stage-5b"],
        "stage1b": states["stage-1b"],
        "workspace_id": ctx.workspace_id,
    }


def build_stage6_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-6",
        depends_on=frozenset({"stage-5b", "stage-1b"}),
        contract=Stage6Contract,
        bind_inputs=_bind_stage6,
        runner=dag.stage6,
    )
