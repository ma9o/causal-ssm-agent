"""Stage 5a pipeline definition."""

from __future__ import annotations

from nof1_causal_lab.flows.stage_runtime import PipelineContext, StageDefinition
from nof1_causal_lab.flows.stages.stage5a.contracts import Stage5aContract


def _bind_stage5a(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
    }


def build_stage5a_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-5a",
        depends_on=frozenset({"stage-4", "stage-2"}),
        contract=Stage5aContract,
        bind_inputs=_bind_stage5a,
        runner=dag.stage5a,
        skip_restore=True,
    )
