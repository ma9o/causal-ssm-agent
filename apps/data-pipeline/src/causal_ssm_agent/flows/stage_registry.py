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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..utils.openrouter_client import use_openrouter_api_key
from . import get_prefect_logger
from .run_store import (
    STAGE0_PARQUET_FILENAMES,
    STAGE2_MODEL_PARQUET_FILENAMES,
    STAGE4_COMPILED_SSM_FILENAMES,
    STAGE5B_PICKLE_FILENAMES,
    finalize_stage,
    find_run_artifact,
    load_json,
    load_public_payload,
    load_stage_snapshot,
    save_json,
    save_parquet,
    save_pickle,
    stage_state,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

logger = get_prefect_logger(__name__)
OpenRouterAccessMode = Literal["user", "trial"]


# ═══════════════════════════════════════════════════════════════════════════════
# Core dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


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
class StageMaterializer:
    """Restore/persist/finalize behavior for a stage as one cohesive concern."""

    restore: Callable[[str, dict, dict[str, dict]], dict] = field(
        default_factory=lambda: _restore_default
    )
    persist: Callable[[dict, str], dict] = field(default_factory=lambda: _persist_noop)
    finalize_extras: Callable[[dict, str], dict[str, Any]] = field(
        default_factory=lambda: _finalize_noop
    )


@dataclass(frozen=True)
class StageOverrideAdapter:
    """Stage-owned replay adapter: public/editable payload -> runtime result."""

    coerce_editable: Callable[[dict[str, Any]], dict[str, Any]]
    materialize: Callable[[dict[str, Any], PipelineContext, dict[str, dict]], dict[str, Any]]


@dataclass(frozen=True)
class StageDefinition:
    """Declarative stage metadata with behavior carried as first-class functions.

    The pipeline becomes a fold over stage definitions, with execution order
    derived from the dependency graph.
    """

    stage_id: str
    depends_on: frozenset[str]
    contract: type[BaseModel]

    # (PipelineContext, stage_states) -> kwargs for runner
    bind_inputs: Callable[[PipelineContext, dict[str, dict]], dict[str, Any]]

    # Bare stage computation function
    runner: Callable[..., dict | Awaitable[dict]]

    materializer: StageMaterializer = field(default_factory=StageMaterializer)

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
    stage_states: dict[str, dict],
) -> dict[str, Any]:
    """Execute a single stage: bind inputs, run, persist, finalize."""

    # Check for override
    override_payload = (
        ctx.supported_overrides.get(defn.stage_id) if defn.override_eligible else None
    )

    if override_payload is not None and defn.override_adapter is not None:
        editable = defn.override_adapter.coerce_editable(dict(override_payload))
        result = defn.override_adapter.materialize(editable, ctx, stage_states)
    elif override_payload is not None:
        raise ValueError(
            f"Stage {defn.stage_id} received an override without an explicit materialization policy"
        )
    else:
        inputs = defn.bind_inputs(ctx, stage_states)
        with use_openrouter_api_key(ctx.openrouter_api_key):
            result = defn.runner(**inputs)
            if inspect.isawaitable(result):
                result = await result

    # Persist artifacts (save_parquet, save_pickle, etc.)
    result = defn.materializer.persist(result, ctx.workspace_id)
    extras = defn.materializer.finalize_extras(result, ctx.workspace_id)

    # Finalize (validate contract, persist JSON, save snapshot)
    state = finalize_stage(
        defn.stage_id,
        result,
        ctx.workspace_id,
        extras=extras or None,
        contract=defn.contract,
    )

    return state


def load_stage_state(
    workspace_id: str,
    stage_id: str,
    prior_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a stage state snapshot, reconstructing from public payloads when needed."""
    prior_states = prior_states or {}
    defn = get_stage_registry()[stage_id]
    try:
        snapshot = load_stage_snapshot(workspace_id, stage_id)
        web = snapshot.get("web") or load_public_payload(workspace_id, stage_id)
        restored = defn.materializer.restore(workspace_id, web, prior_states)
        result = dict(snapshot.get("result", {}) or {})
        result.update(restored)
        return stage_state(result, web)
    except FileNotFoundError:
        logger.info(
            "Reconstructing %s state from public payloads for workspace_id %s",
            stage_id,
            workspace_id,
        )

    web = load_public_payload(workspace_id, stage_id)
    result = defn.materializer.restore(workspace_id, web, prior_states)
    return stage_state(result, web)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers for persist / restore / log
# ═══════════════════════════════════════════════════════════════════════════════


def _persist_noop(result: dict, workspace_id: str) -> dict:
    return result


def _finalize_noop(result: dict, workspace_id: str) -> dict[str, Any]:
    return {}


def _restore_default(workspace_id: str, web: dict, prior_states: dict) -> dict:
    return dict(web)


def _column_descriptions_from_web(web: dict[str, Any]) -> dict[str, str]:
    column_descriptions = web.get("column_descriptions", [])
    if not isinstance(column_descriptions, list):
        return {}
    return {
        str(item.get("name")): str(item.get("description", ""))
        for item in column_descriptions
        if isinstance(item, dict) and item.get("name")
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage persist callbacks
# ═══════════════════════════════════════════════════════════════════════════════


def _persist_stage0(result: dict, workspace_id: str) -> dict:
    raw_df = result.pop("_df")
    result["_df_path"] = save_parquet(raw_df, workspace_id, "stage0-raw-input.parquet")
    return result


def _persist_stage2(result: dict, workspace_id: str) -> dict:
    data_for_model = result.pop("_data_for_model")
    result["_data_for_model_row_count"] = len(data_for_model)
    result["_data_for_model_path"] = save_parquet(
        data_for_model, workspace_id, "stage2-model-data.parquet"
    )
    return result


def _finalize_stage2_extras(result: dict, workspace_id: str) -> dict[str, Any]:
    row_count = int(result.get("_data_for_model_row_count", 0))
    if row_count > 0:
        return {"outcome": "success"}
    return {
        "outcome": "fail",
        "fail_reason": "no_observations_extracted",
    }


def _finalize_stage4_extras(result: dict, workspace_id: str) -> dict[str, Any]:
    if result.get("_compiled_ssm") is not None:
        return {"outcome": "success"}
    return {
        "outcome": "fail",
        "fail_reason": "model_compile_failed",
    }


def _persist_stage5b(result: dict, workspace_id: str) -> dict:
    fitted_artifact = result.pop("_fitted_artifact")
    result["_fitted_result_path"] = save_pickle(
        fitted_artifact, workspace_id, "stage5b-fitted-result.pkl"
    )
    return result


def _persist_stage4(result: dict, workspace_id: str) -> dict:
    compiled_ssm = result.get("_compiled_ssm")
    if compiled_ssm is not None:
        result["_compiled_ssm_path"] = save_json(
            compiled_ssm, workspace_id, "stage4-compiled-ssm.json"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage restore callbacks
# ═══════════════════════════════════════════════════════════════════════════════


def _restore_stage0(workspace_id: str, web: dict, prior_states: dict) -> dict:
    result = dict(web)
    result["_df_path"] = find_run_artifact(workspace_id, STAGE0_PARQUET_FILENAMES)
    result["_column_descriptions"] = _column_descriptions_from_web(web)
    return result


def _restore_stage2(workspace_id: str, web: dict, prior_states: dict) -> dict:
    workers = list(web.get("workers", []) or [])
    result = dict(web)
    result["workers"] = workers
    result["_data_for_model_path"] = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    return result


def _restore_stage4(workspace_id: str, web: dict, prior_states: dict) -> dict:
    result = dict(web)
    result["authored_priors"] = dict(web.get("authored_priors", {}) or {})
    stage1b_state = prior_states.get("stage-1b")
    if stage1b_state is not None:
        result.setdefault("_causal_spec", stage1b_state["result"]["causal_spec"])
    try:
        compiled_ssm_path = find_run_artifact(workspace_id, STAGE4_COMPILED_SSM_FILENAMES)
    except FileNotFoundError:
        return result
    result["_compiled_ssm_path"] = compiled_ssm_path
    result["_compiled_ssm"] = load_json(compiled_ssm_path)
    return result


def _restore_stage4b(workspace_id: str, web: dict, prior_states: dict) -> dict:
    return {
        "parametric_id": web.get("parametric_id", {}),
        "inference_structure": web.get("inference_structure"),
    }


def _restore_stage5b(workspace_id: str, web: dict, prior_states: dict) -> dict:
    power_scaling = list(web.get("power_scaling", []) or [])
    return {
        "outcome": web.get("outcome", "success"),
        "power_scaling": power_scaling,
        "ppc": dict(web.get("ppc", {}) or {}),
        "inference_metadata": dict(web.get("inference_metadata", {}) or {}),
        "mcmc_diagnostics": web.get("mcmc_diagnostics"),
        "svi_diagnostics": web.get("svi_diagnostics"),
        "smc_diagnostics": web.get("smc_diagnostics"),
        "loo_diagnostics": web.get("loo_diagnostics"),
        "posterior_marginals": web.get("posterior_marginals"),
        "posterior_pairs": web.get("posterior_pairs"),
        "_fitted_result_path": find_run_artifact(workspace_id, STAGE5B_PICKLE_FILENAMES),
    }


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
        "stage0": states["stage-0"]["result"],
        "stage1a": states["stage-1a"]["result"],
    }


def _bind_stage2(ctx: PipelineContext, states: dict) -> dict:
    from .stages.stage2_extract import MAX_FREE_WINDOWS

    return {
        "question": ctx.question,
        "stage0": states["stage-0"]["result"],
        "stage1b": states["stage-1b"]["result"],
        "root_run_id": ctx.prefect_run_id,
        "max_windows": None if ctx.openrouter_access_mode == "user" or os.environ.get("DEPLOYMENT_ENV") != "production" else MAX_FREE_WINDOWS,
    }


def _bind_stage3(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage1b": states["stage-1b"]["result"],
        "stage2": states["stage-2"]["result"],
    }


def _bind_stage4(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage1b": states["stage-1b"]["result"],
        "stage2": states["stage-2"]["result"],
        "stage3": states["stage-3"]["result"],
        "enable_literature": ctx.lit_enabled,
        "workspace_id": ctx.workspace_id,
        "root_run_id": ctx.prefect_run_id,
    }


def _bind_stage4b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"]["result"],
        "stage2": states["stage-2"]["result"],
        "root_run_id": ctx.prefect_run_id,
    }


def _bind_stage5a(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"]["result"],
        "stage2": states["stage-2"]["result"],
    }


def _bind_stage5b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"]["result"],
        "stage2": states["stage-2"]["result"],
        "inference_method": ctx.inference_method,
    }


def _bind_stage6(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage5b": states["stage-5b"]["result"],
        "stage1b": states["stage-1b"]["result"],
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
    states: dict[str, dict],
) -> dict[str, Any]:
    """Use the authored editable payload directly as the runtime result."""
    return dict(editable)


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
    from .stages.stage4_assembly import coerce_stage4_override_payload

    return coerce_stage4_override_payload(payload)


def _materialize_override_stage1b(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, dict],
) -> dict[str, Any]:
    """Materialize a stage-1b override via the same derived-field finalizer as normal runs."""
    from . import dag

    latent_model = ((states.get("stage-1a") or {}).get("result") or {}).get("latent_model")
    return dag.finalize_stage1b_result(dict(editable), latent_model=latent_model)


def _materialize_override_stage4(
    editable: dict[str, Any],
    ctx: PipelineContext,
    states: dict[str, dict],
) -> dict[str, Any]:
    """Prepare a stage-4 override via the same stage-owned finalizer as normal runs."""
    from .run_store import load_parquet
    from .stages.stage4_assembly import materialize_stage4_result

    stage1b_result = states["stage-1b"]["result"]
    stage2_result = states["stage-2"]["result"]
    stage3_result = states["stage-3"]["result"]
    authored = dict(editable)
    return materialize_stage4_result(
        model_spec=authored["model_spec"],
        authored_priors=authored["authored_priors"],
        data_for_model=load_parquet(stage2_result["_data_for_model_path"]),
        indicator_audits=stage3_result["indicators"],
        causal_spec=stage1b_result["causal_spec"],
        llm_trace=authored.get("llm_trace"),
    )
# ═══════════════════════════════════════════════════════════════════════════════
# Stage registry
# ═══════════════════════════════════════════════════════════════════════════════


def _build_registry() -> dict[str, StageDefinition]:
    """Build the stage registry with lazy imports to avoid circular dependencies."""
    from dataclasses import replace

    from . import dag
    from .stages.contracts import INTERACTIVE_STAGES, STAGE_CONTRACTS

    registry = {
        "stage-0": StageDefinition(
            stage_id="stage-0",
            depends_on=frozenset(),
            contract=STAGE_CONTRACTS["stage-0"],
            bind_inputs=_bind_stage0,
            runner=dag.stage0,
            materializer=StageMaterializer(
                restore=_restore_stage0,
                persist=_persist_stage0,
            ),
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
            materializer=StageMaterializer(
                restore=_restore_stage2,
                persist=_persist_stage2,
                finalize_extras=_finalize_stage2_extras,
            ),
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
            materializer=StageMaterializer(
                restore=_restore_stage4,
                persist=_persist_stage4,
                finalize_extras=_finalize_stage4_extras,
            ),
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
            materializer=StageMaterializer(restore=_restore_stage4b),
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
            materializer=StageMaterializer(
                restore=_restore_stage5b,
                persist=_persist_stage5b,
            ),
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

    # In production, offload stages 4 and 5b to Modal
    if os.environ.get("DEPLOYMENT_ENV") == "production":
        from . import dag
        from .modal_runners import (
            modal_stage4_runner,
            modal_stage5b_runner,
            persist_noop,
        )

        async def _run_stage4_modal_or_local(
            question: str,
            stage1b: dict,
            stage2: dict,
            stage3: dict,
            enable_literature: bool,
            workspace_id: str,
            openrouter_access_mode: OpenRouterAccessMode | None,
            root_run_id: str | None,
        ) -> dict:
            if openrouter_access_mode == "user":
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
            base = _bind_stage5b(ctx, states)
            base["workspace_id"] = ctx.workspace_id
            return base

        def _bind_stage4_modal_or_local(ctx: PipelineContext, states: dict) -> dict:
            base = _bind_stage4(ctx, states)
            base["openrouter_access_mode"] = ctx.openrouter_access_mode
            return base

        registry["stage-5b"] = replace(
            registry["stage-5b"],
            bind_inputs=_bind_stage5b_modal,
            runner=modal_stage5b_runner,
            materializer=StageMaterializer(
                restore=_restore_stage5b,
                persist=persist_noop,
            ),
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
