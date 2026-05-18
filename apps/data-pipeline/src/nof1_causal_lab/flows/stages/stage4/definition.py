"""Stage 4 pipeline definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.flows.stage_runtime import (
    PipelineContext,
    StageDefinition,
    StageOverrideAdapter,
)
from nof1_causal_lab.flows.stages.stage4.contracts import Stage4Contract

if TYPE_CHECKING:
    from nof1_causal_lab.flows.contracts_base import BaseStageContract


def _emit_stage4_initial_replay_state(inputs: dict[str, Any]) -> None:
    from nof1_causal_lab.flows.runtime_events import (
        emit_stage4_graph_event,
        emit_stage4_snapshot_event,
    )
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_runtime_projections import (
        project_stage4_initial_state,
    )

    root_run_id = inputs["root_run_id"]
    if not isinstance(root_run_id, str) or not root_run_id:
        raise ValueError("Stage 4 initial replay emission requires a non-empty root_run_id")

    stage1b = inputs["stage1b"]
    causal_spec_dict = stage1b.causal_spec.model_dump()
    graph, snapshot = project_stage4_initial_state(causal_spec_dict)
    emit_stage4_graph_event(root_run_id, graph=graph)
    emit_stage4_snapshot_event(root_run_id, snapshot=snapshot)


def _bind_stage4(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage1b": states["stage-1b"],
        "stage2": states["stage-2"],
        "stage3": states["stage-3"],
        "enable_literature": ctx.lit_enabled,
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
    }


def _coerce_override_stage4(payload: dict[str, Any]) -> dict[str, Any]:
    from nof1_causal_lab.flows.stages.stage4.assembly import coerce_stage4_override_payload

    return coerce_stage4_override_payload(payload)


def _materialize_override_stage4(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, BaseStageContract],
) -> BaseStageContract:
    from nof1_causal_lab.flows.run_store import (
        STAGE2_MODEL_PARQUET_FILENAMES,
        find_run_artifact,
        load_parquet,
        save_json,
    )
    from nof1_causal_lab.flows.stages.stage4.assembly import materialize_stage4_result

    stage1b = states["stage-1b"]
    stage3 = states["stage-3"]
    data_for_model_path = find_run_artifact(ctx.workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    authored = dict(editable)
    materialized = materialize_stage4_result(
        model_spec=authored["model_spec"],
        authored_priors=authored["authored_priors"],
        data_for_model=load_parquet(data_for_model_path),
        indicator_audits={k: v.model_dump() for k, v in stage3.indicators.items()},  # type: ignore[union-attr]
        causal_spec=stage1b.causal_spec.model_dump(),  # type: ignore[union-attr]
        llm_trace=authored.get("llm_trace"),
    )

    compiled_ssm = materialized.pop("_compiled_ssm", None)
    if compiled_ssm is not None:
        save_json(compiled_ssm, ctx.workspace_id, "stage4-compiled-ssm.json")

    if compiled_ssm is not None:
        materialized["outcome"] = "success"
    else:
        materialized["outcome"] = "fail"
        materialized["fail_reason"] = "model_compile_failed"

    fields = set(Stage4Contract.model_fields.keys())
    return Stage4Contract.model_validate(
        {key: value for key, value in materialized.items() if key in fields}
    )


def build_stage4_definition() -> StageDefinition:
    from nof1_causal_lab.flows import dag

    return StageDefinition(
        stage_id="stage-4",
        depends_on=frozenset({"stage-1b", "stage-2", "stage-3"}),
        contract=Stage4Contract,
        bind_inputs=_bind_stage4,
        runner=dag.stage4,
        question_required=True,
        override_eligible=True,
        override_adapter=StageOverrideAdapter(
            coerce_editable=_coerce_override_stage4,
            materialize=_materialize_override_stage4,
        ),
        before_run=_emit_stage4_initial_replay_state,
    )
