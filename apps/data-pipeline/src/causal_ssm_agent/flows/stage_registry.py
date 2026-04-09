"""Stage registry: declarative stage metadata and a generic execution combinator.

Each stage is defined once in STAGE_REGISTRY.  Execution order is derived from
the ``depends_on`` DAG via topological sort — no manual index.  The pipeline
runner is a fold over ``EXECUTION_ORDER``.

All per-stage callbacks share a uniform signature (ctx, states, etc.) even when
a specific stage doesn't use every argument.  ARG001 is suppressed file-wide.
"""
# ruff: noqa: ARG001

from __future__ import annotations

import graphlib
import inspect
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key

from . import get_prefect_logger
from .run_store import (
    finalize_stage,
    load_public_payload,
    load_stage_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .stage_contracts import BaseStageContract

logger = get_prefect_logger(__name__)
OpenRouterAccessMode = Literal["user", "anonymous", "local"]


# ═══════════════════════════════════════════════════════════════════════════════
# Core dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


def _emit_stage4_initial_replay_state(inputs: dict[str, Any]) -> None:
    """Emit the initial Stage 4 graph/snapshot before heavy startup work."""
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_runtime_projections import (
        project_stage4_initial_state,
    )

    from .runtime_events import (
        emit_stage4_graph_event,
        emit_stage4_snapshot_event,
    )

    root_run_id = inputs["root_run_id"]
    if not isinstance(root_run_id, str) or not root_run_id:
        raise ValueError("Stage 4 initial replay emission requires a non-empty root_run_id")

    stage1b = inputs["stage1b"]
    causal_spec_dict = stage1b.causal_spec.model_dump()

    graph, snapshot = project_stage4_initial_state(causal_spec_dict)
    emit_stage4_graph_event(root_run_id, graph=graph)
    emit_stage4_snapshot_event(root_run_id, snapshot=snapshot)


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
    """Declarative stage metadata with behavior carried as first-class functions.

    The pipeline becomes a fold over stage definitions, with execution order
    derived from the dependency graph.
    """

    stage_id: str
    depends_on: frozenset[str]
    contract: type[BaseStageContract]

    # (PipelineContext, stage_states) -> kwargs for runner
    bind_inputs: Callable[[PipelineContext, dict[str, BaseStageContract]], dict[str, Any]]

    # Bare stage computation function — returns a contract instance
    runner: Callable[..., BaseStageContract | Awaitable[BaseStageContract]]

    question_required: bool = False
    override_eligible: bool = False

    override_adapter: StageOverrideAdapter | None = None

    # True for stage-5a (best-effort preflight, no restore on resume)
    skip_restore: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Generic combinator
# ═══════════════════════════════════════════════════════════════════════════════


async def run_stage_flow(
    defn: StageDefinition,
    ctx: PipelineContext,
    stage_states: dict[str, BaseStageContract],
) -> BaseStageContract:
    """Execute a single stage: bind inputs, run, persist, finalize."""

    # Check for override
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
        if defn.stage_id == "stage-4":
            _emit_stage4_initial_replay_state(inputs)
        with use_openrouter_api_key(ctx.openrouter_api_key):
            contract = defn.runner(**inputs)
            if inspect.isawaitable(contract):
                contract = await contract

    # Persist web JSON and save snapshot
    return finalize_stage(defn.stage_id, contract, ctx.workspace_id)


def load_stage_state(
    workspace_id: str,
    stage_id: str,
    prior_states: dict[str, BaseStageContract] | None = None,
) -> BaseStageContract:
    """Load a stage state, preferring snapshot then falling back to web JSON."""
    from .stage_contracts import BaseStageContract as _BaseStageContract

    defn = get_stage_registry()[stage_id]
    try:
        snapshot = load_stage_snapshot(workspace_id, stage_id)
        if isinstance(snapshot, _BaseStageContract):
            return snapshot
    except FileNotFoundError:
        pass

    web = load_public_payload(workspace_id, stage_id)
    return defn.contract.model_validate(web)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage bind_inputs
# ═══════════════════════════════════════════════════════════════════════════════


def _bind_stage0(ctx: PipelineContext, states: dict) -> dict:
    return {
        "workspace_id": ctx.workspace_id,
    }


def _bind_stage1a(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
    }


def _bind_stage1b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage0": states["stage-0"],
        "stage1a": states["stage-1a"],
        "workspace_id": ctx.workspace_id,
    }


def _bind_stage2(ctx: PipelineContext, states: dict) -> dict:
    from causal_ssm_agent.utils.config import get_config

    return {
        "question": ctx.question,
        "stage0": states["stage-0"],
        "stage1b": states["stage-1b"],
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
        "max_windows": None
        if ctx.openrouter_access_mode in {"user", "local"}
        or os.environ.get("DEPLOYMENT_ENV") != "production"
        else get_config().stage2_workers.max_free_windows,
    }


def _bind_stage3(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage1b": states["stage-1b"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
    }


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


def _bind_stage4b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
    }


def _bind_stage5a(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
    }


def _bind_stage5b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"],
        "stage2": states["stage-2"],
        "workspace_id": ctx.workspace_id,
        "inference_method": ctx.inference_method,
    }


def _bind_stage6(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage5b": states["stage-5b"],
        "stage1b": states["stage-1b"],
        "workspace_id": ctx.workspace_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage override preparers
# ═══════════════════════════════════════════════════════════════════════════════


def _coerce_override_stage1a(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a replay payload and keep only authored stage-1a fields."""
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


def _materialize_override_identity(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, BaseStageContract],
) -> BaseStageContract:
    """Use the authored editable payload directly as the runtime result."""
    from .stage_contracts import Stage1aContract

    return Stage1aContract.model_validate(editable)


def _coerce_override_stage1b(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a replay payload and keep only authored stage-1b fields."""
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


def _coerce_override_stage4(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a replay payload and keep only authored stage-4 fields."""
    from .stages.stage4.assembly import coerce_stage4_override_payload

    return coerce_stage4_override_payload(payload)


def _materialize_override_stage1b(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, BaseStageContract],
) -> BaseStageContract:
    """Materialize a stage-1b override via the same derived-field finalizer as normal runs."""
    from .stage_contracts import Stage1bContract
    from .stages.stage1b.result import finalize_stage1b_result

    stage1a = states.get("stage-1a")
    latent_model = stage1a.latent_model.model_dump() if stage1a else None  # type: ignore[union-attr]
    finalized = finalize_stage1b_result(dict(editable), latent_model=latent_model)
    fields = set(Stage1bContract.model_fields.keys())
    return Stage1bContract.model_validate({k: v for k, v in finalized.items() if k in fields})


def _materialize_override_stage4(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, BaseStageContract],
) -> BaseStageContract:
    """Prepare a stage-4 override via the same stage-owned finalizer as normal runs."""
    from .run_store import (
        STAGE2_MODEL_PARQUET_FILENAMES,
        find_run_artifact,
        load_parquet,
        save_json,
    )
    from .stage_contracts import Stage4Contract
    from .stages.stage4.assembly import materialize_stage4_result

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

    # Save compiled SSM artifact
    compiled_ssm = materialized.pop("_compiled_ssm", None)
    if compiled_ssm is not None:
        save_json(compiled_ssm, ctx.workspace_id, "stage4-compiled-ssm.json")

    if compiled_ssm is not None:
        materialized["outcome"] = "success"
    else:
        materialized["outcome"] = "fail"
        materialized["fail_reason"] = "model_compile_failed"

    fields = set(Stage4Contract.model_fields.keys())
    return Stage4Contract.model_validate({k: v for k, v in materialized.items() if k in fields})


# ═══════════════════════════════════════════════════════════════════════════════
# Stage registry
# ═══════════════════════════════════════════════════════════════════════════════


def _build_registry() -> dict[str, StageDefinition]:
    """Build the stage registry with lazy imports to avoid circular dependencies."""
    from dataclasses import replace

    from . import dag
    from .stage_contracts import INTERACTIVE_STAGES, STAGE_CONTRACTS

    registry = {
        "stage-0": StageDefinition(
            stage_id="stage-0",
            depends_on=frozenset(),
            contract=STAGE_CONTRACTS["stage-0"],
            bind_inputs=_bind_stage0,
            runner=dag.stage0,
        ),
        "stage-1a": StageDefinition(
            stage_id="stage-1a",
            depends_on=frozenset(),
            contract=STAGE_CONTRACTS["stage-1a"],
            bind_inputs=_bind_stage1a,
            runner=dag.stage1a,
            question_required=True,
            override_eligible=True,
            override_adapter=StageOverrideAdapter(
                coerce_editable=_coerce_override_stage1a,
                materialize=_materialize_override_identity,
            ),
        ),
        "stage-1b": StageDefinition(
            stage_id="stage-1b",
            depends_on=frozenset({"stage-0", "stage-1a"}),
            contract=STAGE_CONTRACTS["stage-1b"],
            bind_inputs=_bind_stage1b,
            runner=dag.stage1b,
            question_required=True,
            override_eligible=True,
            override_adapter=StageOverrideAdapter(
                coerce_editable=_coerce_override_stage1b,
                materialize=_materialize_override_stage1b,
            ),
        ),
        "stage-2": StageDefinition(
            stage_id="stage-2",
            depends_on=frozenset({"stage-0", "stage-1b"}),
            contract=STAGE_CONTRACTS["stage-2"],
            bind_inputs=_bind_stage2,
            runner=dag.stage2,
            question_required=True,
        ),
        "stage-3": StageDefinition(
            stage_id="stage-3",
            depends_on=frozenset({"stage-1b", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-3"],
            bind_inputs=_bind_stage3,
            runner=dag.stage3,
        ),
        "stage-4": StageDefinition(
            stage_id="stage-4",
            depends_on=frozenset({"stage-1b", "stage-2", "stage-3"}),
            contract=STAGE_CONTRACTS["stage-4"],
            bind_inputs=_bind_stage4,
            runner=dag.stage4,
            question_required=True,
            override_eligible=True,
            override_adapter=StageOverrideAdapter(
                coerce_editable=_coerce_override_stage4,
                materialize=_materialize_override_stage4,
            ),
        ),
        "stage-4b": StageDefinition(
            stage_id="stage-4b",
            depends_on=frozenset({"stage-4", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-4b"],
            bind_inputs=_bind_stage4b,
            runner=dag.stage4b,
        ),
        "stage-5a": StageDefinition(
            stage_id="stage-5a",
            depends_on=frozenset({"stage-4", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-5a"],
            bind_inputs=_bind_stage5a,
            runner=dag.stage5a,
            skip_restore=True,
        ),
        "stage-5b": StageDefinition(
            stage_id="stage-5b",
            depends_on=frozenset({"stage-4", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-5b"],
            bind_inputs=_bind_stage5b,
            runner=dag.stage5b,
        ),
        "stage-6": StageDefinition(
            stage_id="stage-6",
            depends_on=frozenset({"stage-5b", "stage-1b"}),
            contract=STAGE_CONTRACTS["stage-6"],
            bind_inputs=_bind_stage6,
            runner=dag.stage6,
        ),
    }

    for stage_id in INTERACTIVE_STAGES:
        defn = registry[stage_id]
        if defn.override_eligible and defn.override_adapter is None:
            raise ValueError(
                f"Interactive stage {stage_id} must declare override materialization "
                "via StageOverrideAdapter"
            )

    # In production, offload stages 4 and 5b to Modal.
    # Only explicit local mode keeps stage 4 in-process.
    if os.environ.get("DEPLOYMENT_ENV") == "production":
        from . import dag
        from .modal_runners import (
            modal_stage4_runner,
            modal_stage5b_runner,
        )

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

        def _bind_stage5b_modal(ctx: PipelineContext, states: dict) -> dict:
            return _bind_stage5b(ctx, states)

        def _bind_stage4_modal_or_local(ctx: PipelineContext, states: dict) -> dict:
            base = _bind_stage4(ctx, states)
            base["openrouter_access_mode"] = ctx.openrouter_access_mode
            return base

        registry["stage-5b"] = replace(
            registry["stage-5b"],
            bind_inputs=_bind_stage5b_modal,
            runner=modal_stage5b_runner,
        )
        registry["stage-4"] = replace(
            registry["stage-4"],
            bind_inputs=_bind_stage4_modal_or_local,
            runner=_run_stage4_modal_or_local,
        )

    return registry


# Lazily initialized at first access
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
