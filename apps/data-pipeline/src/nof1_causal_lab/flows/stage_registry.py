"""Stage registry: assemble stage-owned definitions into one execution DAG."""

from __future__ import annotations

import graphlib
import os
from dataclasses import replace

from . import get_prefect_logger
from .contracts_base import BaseStageContract
from .run_store import load_public_payload, load_stage_snapshot
from .stage_contracts import INTERACTIVE_STAGES
from .stage_runtime import (
    OpenRouterAccessMode,
    PipelineContext,
    StageDefinition,
    StageOverrideAdapter,
    run_stage_flow,
)
from .stages.stage0.definition import build_stage0_definition
from .stages.stage1a.definition import build_stage1a_definition
from .stages.stage1b.definition import build_stage1b_definition
from .stages.stage2.definition import build_stage2_definition
from .stages.stage3.definition import build_stage3_definition
from .stages.stage4.definition import build_stage4_definition
from .stages.stage5b.definition import build_stage5b_definition
from .stages.stage6.definition import build_stage6_definition

logger = get_prefect_logger(__name__)


def load_stage_state(
    workspace_id: str,
    stage_id: str,
) -> BaseStageContract:
    """Load a stage state, preferring snapshot then falling back to web JSON."""
    defn = get_stage_registry()[stage_id]
    try:
        snapshot = load_stage_snapshot(workspace_id, stage_id)
        if isinstance(snapshot, BaseStageContract):
            return snapshot
    except FileNotFoundError:
        pass

    web = load_public_payload(workspace_id, stage_id)
    return defn.contract.model_validate(web)


def _build_registry() -> dict[str, StageDefinition]:
    registry = {
        defn.stage_id: defn
        for defn in (
            build_stage0_definition(),
            build_stage1a_definition(),
            build_stage1b_definition(),
            build_stage2_definition(),
            build_stage3_definition(),
            build_stage4_definition(),
            build_stage5b_definition(),
            build_stage6_definition(),
        )
    }

    for stage_id in INTERACTIVE_STAGES:
        defn = registry[stage_id]
        if defn.override_eligible and defn.override_adapter is None:
            raise ValueError(
                f"Interactive stage {stage_id} must declare override materialization "
                "via StageOverrideAdapter"
            )

    if os.environ.get("DEPLOYMENT_ENV") == "production":
        from . import dag
        from .modal_runners import modal_stage4_runner, modal_stage5b_runner

        base_stage4_bind = registry["stage-4"].bind_inputs
        base_stage5b_bind = registry["stage-5b"].bind_inputs

        async def _run_stage4_modal_or_local(
            question: str,
            stage1b: BaseStageContract,
            stage2: BaseStageContract,
            stage3: BaseStageContract,
            enable_literature: bool,
            workspace_id: str,
            openrouter_access_mode: OpenRouterAccessMode | None,
            root_run_id: str | None,
        ) -> BaseStageContract:
            if openrouter_access_mode == "local":
                return await dag.stage4(
                    question,
                    stage1b,
                    stage2,
                    stage3,
                    enable_literature,
                    workspace_id=workspace_id,
                    root_run_id=root_run_id,
                )
            return await modal_stage4_runner(
                question,
                stage1b,
                stage2,
                stage3,
                enable_literature,
                workspace_id=workspace_id,
                root_run_id=root_run_id,
            )

        def _bind_stage4_modal_or_local(ctx: PipelineContext, states: dict) -> dict:
            base = base_stage4_bind(ctx, states)
            base["openrouter_access_mode"] = ctx.openrouter_access_mode
            return base

        registry["stage-5b"] = replace(
            registry["stage-5b"],
            bind_inputs=base_stage5b_bind,
            runner=modal_stage5b_runner,
        )
        registry["stage-4"] = replace(
            registry["stage-4"],
            bind_inputs=_bind_stage4_modal_or_local,
            runner=_run_stage4_modal_or_local,
        )

    return registry


_registry: dict[str, StageDefinition] | None = None
_execution_order: tuple[str, ...] | None = None


def _ensure_initialized() -> None:
    global _registry, _execution_order
    if _registry is not None:
        return
    _registry = _build_registry()
    dep_graph = {defn.stage_id: set(defn.depends_on) for defn in _registry.values()}
    _execution_order = tuple(graphlib.TopologicalSorter(dep_graph).static_order())


def get_stage_registry() -> dict[str, StageDefinition]:
    _ensure_initialized()
    assert _registry is not None
    return _registry


def get_execution_order() -> tuple[str, ...]:
    _ensure_initialized()
    assert _execution_order is not None
    return _execution_order


__all__ = [
    "OpenRouterAccessMode",
    "PipelineContext",
    "StageDefinition",
    "StageOverrideAdapter",
    "get_execution_order",
    "get_stage_registry",
    "load_stage_state",
    "run_stage_flow",
]
