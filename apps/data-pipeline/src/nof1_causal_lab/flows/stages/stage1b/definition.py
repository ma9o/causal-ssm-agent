"""Stage 1b pipeline definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from causal_ssm_agent.flows.stage_runtime import (
    PipelineContext,
    StageDefinition,
    StageOverrideAdapter,
)
from causal_ssm_agent.flows.stages.stage1b.contracts import Stage1bContract

if TYPE_CHECKING:
    from causal_ssm_agent.flows.contracts_base import BaseStageContract


def _bind_stage1b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage0": states["stage-0"],
        "stage1a": states["stage-1a"],
        "workspace_id": ctx.workspace_id,
    }


def _coerce_override_stage1b(payload: dict[str, Any]) -> dict[str, Any]:
    causal_spec = payload.get("causal_spec")
    if not isinstance(causal_spec, dict):
        raise ValueError("Stage 1b replay requires a 'causal_spec' object")

    editable = {"causal_spec": causal_spec}
    if "llm_trace" in payload:
        editable["llm_trace"] = payload.get("llm_trace")
    if "outcome" in payload:
        editable["outcome"] = payload.get("outcome")
    if "fail_reason" in payload:
        editable["fail_reason"] = payload.get("fail_reason")
    return editable


def _materialize_override_stage1b(
    editable: dict[str, Any],
    _ctx: PipelineContext,
    states: dict[str, BaseStageContract],
) -> BaseStageContract:
    from causal_ssm_agent.flows.stages.stage1b.result import finalize_stage1b_result

    stage1a = states.get("stage-1a")
    latent_model = stage1a.latent_model.model_dump() if stage1a else None  # type: ignore[union-attr]
    finalized = finalize_stage1b_result(dict(editable), latent_model=latent_model)
    fields = set(Stage1bContract.model_fields.keys())
    return Stage1bContract.model_validate(
        {key: value for key, value in finalized.items() if key in fields}
    )


def build_stage1b_definition() -> StageDefinition:
    from causal_ssm_agent.flows import dag

    return StageDefinition(
        stage_id="stage-1b",
        depends_on=frozenset({"stage-0", "stage-1a"}),
        contract=Stage1bContract,
        bind_inputs=_bind_stage1b,
        runner=dag.stage1b,
        question_required=True,
        override_eligible=True,
        override_adapter=StageOverrideAdapter(
            coerce_editable=_coerce_override_stage1b,
            materialize=_materialize_override_stage1b,
        ),
    )
