"""Stage 1a pipeline definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.flows.stage_runtime import (
    PipelineContext,
    StageDefinition,
    StageOverrideAdapter,
)
from nof1_causal_lab.flows.stages.stage1a.contracts import Stage1aContract

if TYPE_CHECKING:
    from nof1_causal_lab.flows.contracts_base import BaseStageContract


def _bind_stage1a(ctx: PipelineContext, _states: dict) -> dict:
    return {"question": ctx.question}


def _coerce_override_stage1a(payload: dict[str, Any]) -> dict[str, Any]:
    latent_model = payload.get("latent_model")
    if not isinstance(latent_model, dict):
        raise ValueError("Stage 1a replay requires a 'latent_model' object")

    editable = {"latent_model": latent_model}
    if "llm_trace" in payload:
        editable["llm_trace"] = payload.get("llm_trace")
    if "outcome" in payload:
        editable["outcome"] = payload.get("outcome")
    if "fail_reason" in payload:
        editable["fail_reason"] = payload.get("fail_reason")
    return editable


def _materialize_override_stage1a(
    editable: dict[str, Any],
    _ctx: PipelineContext,
    _states: dict[str, BaseStageContract],
) -> BaseStageContract:
    return Stage1aContract.model_validate(editable)


def build_stage1a_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-1a",
        depends_on=frozenset(),
        contract=Stage1aContract,
        bind_inputs=_bind_stage1a,
        runner=dag.stage1a,
        question_required=True,
        override_eligible=True,
        override_adapter=StageOverrideAdapter(
            coerce_editable=_coerce_override_stage1a,
            materialize=_materialize_override_stage1a,
        ),
    )
