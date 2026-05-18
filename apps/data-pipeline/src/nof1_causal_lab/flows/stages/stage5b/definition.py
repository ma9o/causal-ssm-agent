"""Stage 5b pipeline definition."""

from __future__ import annotations

from nof1_causal_lab.flows.stage_runtime import PipelineContext, StageDefinition
from nof1_causal_lab.flows.stages.stage5b.contracts import Stage5bContract


def _bind_stage5b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
        "inference_method": ctx.inference_method,
    }


def build_stage5b_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-5b",
        depends_on=frozenset({"stage-4", "stage-2"}),
        contract=Stage5bContract,
        bind_inputs=_bind_stage5b,
        runner=dag.stage5b,
    )
