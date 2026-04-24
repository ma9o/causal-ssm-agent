"""Generic stage runtime types and execution combinators."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key

from .run_store import finalize_stage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .contracts_base import BaseStageContract

OpenRouterAccessMode = Literal["user", "anonymous", "local"]


@dataclass(frozen=True)
class PipelineContext:
    """Non-stage runtime values threaded through the pipeline."""

    workspace_id: str
    prefect_run_id: str
    question: str | None
    lit_enabled: bool
    inference_method: str | None
    supported_overrides: dict[str, dict]
    openrouter_api_key: str | None
    openrouter_access_mode: OpenRouterAccessMode | None


@dataclass(frozen=True)
class StageOverrideAdapter:
    """Stage-owned replay adapter: public/editable payload -> contract."""

    coerce_editable: Callable[[dict[str, Any]], dict[str, Any]]
    materialize: Callable[
        [dict[str, Any], PipelineContext, dict[str, BaseStageContract]],
        BaseStageContract,
    ]


@dataclass(frozen=True)
class StageDefinition:
    """Declarative stage metadata with behavior carried as first-class functions."""

    stage_id: str
    depends_on: frozenset[str]
    contract: type[BaseStageContract]
    bind_inputs: Callable[[PipelineContext, dict[str, BaseStageContract]], dict[str, Any]]
    runner: Callable[..., BaseStageContract | Awaitable[BaseStageContract]]
    question_required: bool = False
    override_eligible: bool = False
    override_adapter: StageOverrideAdapter | None = None
    before_run: Callable[[dict[str, Any]], None] | None = None
    skip_restore: bool = False


async def run_stage_flow(
    defn: StageDefinition,
    ctx: PipelineContext,
    stage_states: dict[str, BaseStageContract],
    *,
    finalize: Callable[[str, BaseStageContract, str], BaseStageContract] = finalize_stage,
) -> BaseStageContract:
    """Execute a single stage: bind inputs, run, persist, finalize."""

    override_payload = (
        ctx.supported_overrides.get(defn.stage_id) if defn.override_eligible else None
    )

    if override_payload is not None and defn.override_adapter is not None:
        editable = defn.override_adapter.coerce_editable(dict(override_payload))
        contract = defn.override_adapter.materialize(editable, ctx, stage_states)
    elif override_payload is not None:
        raise ValueError(
            f"Stage {defn.stage_id} received an override without an explicit materialization policy"
        )
    else:
        inputs = defn.bind_inputs(ctx, stage_states)
        if defn.before_run is not None:
            defn.before_run(inputs)
        with use_openrouter_api_key(ctx.openrouter_api_key):
            contract = defn.runner(**inputs)
            if inspect.isawaitable(contract):
                contract = await contract

    return finalize(defn.stage_id, contract, ctx.workspace_id)
