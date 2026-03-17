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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import get_prefect_logger
from .run_store import (
    STAGE0_PARQUET_FILENAMES,
    STAGE2_MODEL_PARQUET_FILENAMES,
    STAGE2_RAW_PARQUET_FILENAMES,
    STAGE5B_PICKLE_FILENAMES,
    finalize_stage,
    find_run_artifact,
    load_public_payload,
    load_stage_snapshot,
    save_parquet,
    save_pickle,
    stage_state,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

logger = get_prefect_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Core dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PipelineContext:
    """Non-stage runtime values threaded through the pipeline."""

    user_id: str
    prefect_run_id: str
    question: str | None
    gates_overridden: bool
    lit_enabled: bool
    inference_method: str | None
    supported_overrides: dict[str, dict]


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

    # (result, stage_states, gates_overridden) -> gate_result | None
    gate: Callable[[dict, dict[str, dict], bool], dict] | None = None

    materializer: StageMaterializer = field(default_factory=StageMaterializer)

    # Error message for gate failure (receives gate_result)
    gate_error: Callable[[dict], str] | None = None

    # (web_dict) -> None; logs completion summary
    log_summary: Callable[[dict], None] = field(default_factory=lambda: _log_noop)

    question_required: bool = False
    override_eligible: bool = False

    # (override_payload, ctx, stage_states) -> prepared result
    prepare_override: Callable[[dict, PipelineContext, dict[str, dict]], dict] | None = None

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
    """Execute a single stage: bind inputs, run, persist, gate, finalize."""

    # Check for override
    override_payload = (
        ctx.supported_overrides.get(defn.stage_id) if defn.override_eligible else None
    )

    if override_payload is not None and defn.prepare_override is not None:
        result = defn.prepare_override(override_payload, ctx, stage_states)
    elif override_payload is not None:
        result = dict(override_payload)
    else:
        inputs = defn.bind_inputs(ctx, stage_states)
        result = defn.runner(**inputs)
        if inspect.isawaitable(result):
            result = await result

    # Persist artifacts (save_parquet, save_pickle, etc.)
    result = defn.materializer.persist(result, ctx.user_id)
    extras = defn.materializer.finalize_extras(result, ctx.user_id)

    # Gate check
    gate_result = None
    if defn.gate is not None:
        gate_result = defn.gate(result, stage_states, ctx.gates_overridden)
        gate_extras = _gate_extras(defn, gate_result)
        extras.update(gate_extras)

    # Finalize (validate contract, persist JSON, save snapshot)
    state = finalize_stage(
        defn.stage_id,
        result,
        ctx.user_id,
        extras=extras or None,
        gate=gate_result,
        contract=defn.contract,
    )

    # Raise on hard gate failure
    if (
        gate_result is not None
        and defn.gate_error is not None
        and gate_result.get("gate_failed")
        and not gate_result.get("gate_overridden")
    ):
        raise RuntimeError(defn.gate_error(gate_result))

    # Log summary
    defn.log_summary(state["web"])

    return state


def load_stage_state(
    user_id: str,
    stage_id: str,
    prior_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a stage state snapshot, reconstructing from public payloads when needed."""
    try:
        return load_stage_snapshot(user_id, stage_id)
    except FileNotFoundError:
        logger.info(
            "Reconstructing %s state from public payloads for user_id %s",
            stage_id,
            user_id,
        )

    prior_states = prior_states or {}
    web = load_public_payload(user_id, stage_id)
    defn = get_stage_registry()[stage_id]

    result = defn.materializer.restore(user_id, web, prior_states)

    gate_result = None
    if defn.gate is not None:
        gate_result = defn.gate(result, prior_states, bool(web.get("gate_overridden")))

    return stage_state(result, web, gate=gate_result)


def _gate_extras(defn: StageDefinition, gate_result: dict) -> dict[str, Any]:
    """Build finalization extras from a gate result."""
    extras: dict[str, Any] = {}
    outcome = gate_result.get("outcome") or gate_result.get("web_outcome")
    if outcome:
        extras["outcome"] = outcome
    if gate_result.get("gate_overridden") and defn.gate_error is not None:
        extras["gate_overridden"] = {"reason": defn.gate_error(gate_result)}
    return extras


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers for persist / restore / log
# ═══════════════════════════════════════════════════════════════════════════════


def _persist_noop(result: dict, user_id: str) -> dict:
    return result


def _finalize_noop(result: dict, user_id: str) -> dict[str, Any]:
    return {}


def _restore_default(user_id: str, web: dict, prior_states: dict) -> dict:
    return dict(web)


def _log_noop(web: dict) -> None:
    pass


def _column_descriptions_from_web(web: dict[str, Any]) -> dict[str, str]:
    column_descriptions = web.get("column_descriptions", [])
    if not isinstance(column_descriptions, list):
        return {}
    return {
        str(item.get("name")): str(item.get("description", ""))
        for item in column_descriptions
        if isinstance(item, dict) and item.get("name")
    }


def _power_scaling_list_to_result(entries: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis = {
        str(entry.get("parameter")): str(entry.get("diagnosis"))
        for entry in entries
        if entry.get("parameter") is not None and entry.get("diagnosis") is not None
    }
    prior_sensitivity = {
        str(entry.get("parameter")): float(entry.get("prior_sensitivity", 0.0))
        for entry in entries
        if entry.get("parameter") is not None
    }
    likelihood_sensitivity = {
        str(entry.get("parameter")): float(entry.get("likelihood_sensitivity", 0.0))
        for entry in entries
        if entry.get("parameter") is not None
    }
    psis_k_hat = {
        str(entry.get("parameter")): float(entry.get("psis_k_hat", 0.0))
        for entry in entries
        if entry.get("parameter") is not None and entry.get("psis_k_hat") is not None
    }
    return {
        "checked": bool(entries),
        "diagnosis": diagnosis,
        "prior_sensitivity": prior_sensitivity,
        "likelihood_sensitivity": likelihood_sensitivity,
        "psis_k_hat": psis_k_hat,
    }


def _validation_issue_counts(report: dict[str, Any]) -> tuple[int, int]:
    issues = report.get("issues", []) or []
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return error_count, warning_count


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage persist callbacks
# ═══════════════════════════════════════════════════════════════════════════════


def _persist_stage0(result: dict, user_id: str) -> dict:
    raw_df = result.pop("_df")
    result["_df_path"] = save_parquet(raw_df, user_id, "stage0-raw-input.parquet")
    return result


def _persist_stage2(result: dict, user_id: str) -> dict:
    raw_data = result.pop("_raw_data")
    data_for_model = result.pop("_data_for_model")
    result["_raw_data_row_count"] = len(raw_data)
    result["_raw_data_path"] = save_parquet(raw_data, user_id, "stage2-raw-data.parquet")
    result["_data_for_model_path"] = save_parquet(
        data_for_model, user_id, "stage2-model-data.parquet"
    )
    return result


def _finalize_stage2_extras(result: dict, user_id: str) -> dict[str, Any]:
    row_count = int(result.get("_raw_data_row_count", 0))
    return {"outcome": "success" if row_count > 0 else "fail"}


def _persist_stage5b(result: dict, user_id: str) -> dict:
    fitted_artifact = result.pop("_fitted_artifact")
    result["_fitted_result_path"] = save_pickle(
        fitted_artifact, user_id, "stage5b-fitted-result.pkl"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage restore callbacks
# ═══════════════════════════════════════════════════════════════════════════════


def _restore_stage0(user_id: str, web: dict, prior_states: dict) -> dict:
    result = dict(web)
    result["_df_path"] = find_run_artifact(user_id, STAGE0_PARQUET_FILENAMES)
    result["_column_descriptions"] = _column_descriptions_from_web(web)
    return result


def _restore_stage1b(user_id: str, web: dict, prior_states: dict) -> dict:
    # Gate is reconstructed separately by the combinator via defn.gate
    return dict(web)


def _restore_stage2(user_id: str, web: dict, prior_states: dict) -> dict:
    workers = list(web.get("workers", []) or [])
    result = dict(web)
    result["workers"] = workers
    result["_worker_statuses"] = workers
    result["_raw_data_path"] = find_run_artifact(user_id, STAGE2_RAW_PARQUET_FILENAMES)
    result["_data_for_model_path"] = find_run_artifact(user_id, STAGE2_MODEL_PARQUET_FILENAMES)
    return result


def _restore_stage4(user_id: str, web: dict, prior_states: dict) -> dict:
    result = dict(web)
    stage1b_state = prior_states.get("stage-1b")
    if stage1b_state is not None:
        result.setdefault("causal_spec", stage1b_state["result"]["causal_spec"])
    return result


def _restore_stage4b(user_id: str, web: dict, prior_states: dict) -> dict:
    return {"parametric_id": web.get("parametric_id", {})}


def _restore_stage5b(user_id: str, web: dict, prior_states: dict) -> dict:
    power_scaling = list(web.get("power_scaling", []) or [])
    return {
        "outcome": web.get("outcome", "success"),
        "power_scaling": power_scaling,
        "_ps_result": _power_scaling_list_to_result(power_scaling),
        "_ppc_result": dict(web.get("ppc", {}) or {}),
        "ppc": dict(web.get("ppc", {}) or {}),
        "inference_metadata": dict(web.get("inference_metadata", {}) or {}),
        "mcmc_diagnostics": web.get("mcmc_diagnostics"),
        "svi_diagnostics": web.get("svi_diagnostics"),
        "loo_diagnostics": web.get("loo_diagnostics"),
        "posterior_marginals": web.get("posterior_marginals"),
        "posterior_pairs": web.get("posterior_pairs"),
        "_fitted_result_path": find_run_artifact(user_id, STAGE5B_PICKLE_FILENAMES),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage bind_inputs
# ═══════════════════════════════════════════════════════════════════════════════


def _bind_stage0(ctx: PipelineContext, states: dict) -> dict:
    return {"user_id": ctx.user_id}


def _bind_stage1a(ctx: PipelineContext, states: dict) -> dict:
    return {"question": ctx.question}


def _bind_stage1b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage0": states["stage-0"]["result"],
        "stage1a": states["stage-1a"]["result"],
    }


def _bind_stage2(ctx: PipelineContext, states: dict) -> dict:
    return {
        "question": ctx.question,
        "stage0": states["stage-0"]["result"],
        "stage1b": states["stage-1b"]["result"],
        "root_run_id": ctx.prefect_run_id,
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
        "enable_literature": ctx.lit_enabled,
    }


def _bind_stage4b(ctx: PipelineContext, states: dict) -> dict:
    return {
        "stage4": states["stage-4"]["result"],
        "stage2": states["stage-2"]["result"],
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
        "stage5b": states["stage-5b"]["result"],
        "stage1a": states["stage-1a"]["result"],
        "stage1b": states["stage-1b"]["result"],
        "stage1b_gate": states["stage-1b"]["gate"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage gate adapters
# ═══════════════════════════════════════════════════════════════════════════════


def _gate_stage1b(result: dict, states: dict, gates_overridden: bool) -> dict:
    from .dag import stage1b_gate

    return stage1b_gate(states["stage-1a"]["result"], result, gates_overridden)


def _gate_stage4b(result: dict, states: dict, gates_overridden: bool) -> dict:
    from .dag import stage4b_gate

    return stage4b_gate(result, gates_overridden)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage gate error messages
# ═══════════════════════════════════════════════════════════════════════════════


def _gate_error_stage1b(gate_result: dict) -> str:
    return (
        "No identifiable treatment effects remain after filtering. "
        "All treatments are blocked by unobserved confounders."
    )


def _gate_error_stage4b(gate_result: dict) -> str:
    t_rule = gate_result.get("t_rule") or {}
    return (
        f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
        f"> {t_rule.get('n_moments')} moment conditions. "
        "Model is provably non-identified. Halting pipeline."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage override preparers
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_override_stage4(payload: dict, ctx: PipelineContext, states: dict) -> dict:
    """Prepare a stage-4 override: inject causal_spec, compile if needed."""
    from .run_store import load_parquet, unwrap_task_result

    result = dict(payload)
    stage1b_result = states["stage-1b"]["result"]
    stage2_result = states["stage-2"]["result"]
    result.setdefault("causal_spec", stage1b_result["causal_spec"])

    if "_compiled_ssm" not in result:
        from .stages.stage4_model import compile_model_task

        compile_task = compile_model_task(
            result.get("model_spec", {}),
            result.get("priors", {}),
            load_parquet(stage2_result["_data_for_model_path"]),
            causal_spec=result["causal_spec"],
        )
        compile_result = unwrap_task_result(compile_task)
        compiled_ssm = compile_result.pop("compiled_ssm", None)
        result.setdefault("model_info", compile_result)
        if compiled_ssm is not None:
            result["_compiled_ssm"] = compiled_ssm

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-stage log summaries
# ═══════════════════════════════════════════════════════════════════════════════


def _log_stage0(web: dict) -> None:
    date_range = web.get("date_range", {})
    logger.info(
        "Stage 0 complete: source=%s records=%d columns=%d date_range=%s..%s",
        web.get("source_label", "unknown"),
        web.get("n_records", 0),
        web.get("n_columns", 0),
        date_range.get("start") or "?",
        date_range.get("end") or "?",
    )


def _log_stage1a(web: dict) -> None:
    latent_model = web.get("latent_model", {})
    logger.info(
        "Stage 1a complete: constructs=%d edges=%d treatments=%d outcome=%s",
        len(latent_model.get("constructs", [])),
        len(latent_model.get("edges", [])),
        len(web.get("treatments", [])),
        web.get("outcome_name", "") or "unknown",
    )


def _log_stage1b(web: dict) -> None:
    causal_spec = web.get("causal_spec", {})
    latent = causal_spec.get("latent", {})
    measurement = causal_spec.get("measurement", {})
    logger.info(
        "Stage 1b complete: constructs=%d indicators=%d",
        len(latent.get("constructs", [])),
        len(measurement.get("indicators", [])),
    )


def _log_stage2(web: dict) -> None:
    logger.info(
        "Stage 2 complete: outcome=%s",
        web.get("outcome", "success"),
    )


def _log_stage3(web: dict) -> None:
    report = web.get("validation_report", {})
    error_count, warning_count = _validation_issue_counts(report)
    logger.info(
        "Stage 3 complete: is_valid=%s issues=%d errors=%d warnings=%d outcome=%s",
        report.get("is_valid", False),
        len(report.get("issues", []) or []),
        error_count,
        warning_count,
        web.get("outcome", "success"),
    )


def _log_stage4(web: dict) -> None:
    model_spec = web.get("model_spec", {})
    validation = web.get("validation", {})
    model_info = web.get("model_info", {})
    logger.info(
        "Stage 4 complete: parameters=%d likelihoods=%d priors=%d validation_ok=%s model_built=%s",
        len(model_spec.get("parameters", [])),
        len(model_spec.get("likelihoods", [])),
        len(web.get("priors", {})),
        validation.get("is_valid", False),
        model_info.get("model_built", False),
    )


def _log_stage4b(web: dict) -> None:
    parametric_id = web.get("parametric_id") or {}
    t_rule = parametric_id.get("t_rule") or {}
    logger.info(
        "Stage 4b complete: checked=%s t_rule=%s(%s/%s) outcome=%s",
        parametric_id.get("checked", False),
        "pass" if t_rule.get("satisfies", True) else "fail",
        t_rule.get("n_free_params", "?"),
        t_rule.get("n_moments", "?"),
        web.get("outcome", "success"),
    )


def _log_stage5a(web: dict) -> None:
    logger.info(
        "Stage 5a complete: svi_converged=%s outcome=%s",
        web.get("svi_diagnostics") is not None,
        web.get("outcome", "success"),
    )


def _log_stage5b(web: dict) -> None:
    ps_list = web.get("power_scaling", [])
    ps_issues = sum(
        1
        for entry in ps_list
        if entry.get("diagnosis") in {"prior_dominated", "prior_data_conflict"}
    )
    ppc_warnings = len((web.get("ppc") or {}).get("per_variable_warnings") or [])
    logger.info(
        "Stage 5b complete: method=%s power_scaling_issues=%d ppc_warnings=%d outcome=%s",
        (web.get("inference_metadata") or {}).get("method", "unknown"),
        ps_issues,
        ppc_warnings,
        web.get("outcome", "success"),
    )


def _log_stage6(web: dict) -> None:
    intervention_results = web.get("intervention_results", [])
    warning_count = sum(
        1
        for r in intervention_results
        if r.get("warning") or r.get("ppc_warnings") or r.get("prior_sensitivity_warning")
    )
    logger.info(
        "Stage 6 complete: treatments_ranked=%d warnings=%d outcome=%s",
        len(intervention_results),
        warning_count,
        web.get("outcome", "success"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage registry
# ═══════════════════════════════════════════════════════════════════════════════


def _build_registry() -> dict[str, StageDefinition]:
    """Build the stage registry with lazy imports to avoid circular dependencies."""
    from . import dag
    from .stages.contracts import STAGE_CONTRACTS

    return {
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
            log_summary=_log_stage0,
        ),
        "stage-1a": StageDefinition(
            stage_id="stage-1a",
            depends_on=frozenset(),
            contract=STAGE_CONTRACTS["stage-1a"],
            bind_inputs=_bind_stage1a,
            runner=dag.stage1a,
            question_required=True,
            override_eligible=True,
            log_summary=_log_stage1a,
        ),
        "stage-1b": StageDefinition(
            stage_id="stage-1b",
            depends_on=frozenset({"stage-0", "stage-1a"}),
            contract=STAGE_CONTRACTS["stage-1b"],
            bind_inputs=_bind_stage1b,
            runner=dag.stage1b,
            gate=_gate_stage1b,
            gate_error=_gate_error_stage1b,
            materializer=StageMaterializer(restore=_restore_stage1b),
            question_required=True,
            override_eligible=True,
            log_summary=_log_stage1b,
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
            log_summary=_log_stage2,
        ),
        "stage-3": StageDefinition(
            stage_id="stage-3",
            depends_on=frozenset({"stage-1b", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-3"],
            bind_inputs=_bind_stage3,
            runner=dag.stage3,
            log_summary=_log_stage3,
        ),
        "stage-4": StageDefinition(
            stage_id="stage-4",
            depends_on=frozenset({"stage-1b", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-4"],
            bind_inputs=_bind_stage4,
            runner=dag.stage4,
            materializer=StageMaterializer(restore=_restore_stage4),
            question_required=True,
            override_eligible=True,
            prepare_override=_prepare_override_stage4,
            log_summary=_log_stage4,
        ),
        "stage-4b": StageDefinition(
            stage_id="stage-4b",
            depends_on=frozenset({"stage-4", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-4b"],
            bind_inputs=_bind_stage4b,
            runner=dag.stage4b,
            gate=_gate_stage4b,
            gate_error=_gate_error_stage4b,
            materializer=StageMaterializer(restore=_restore_stage4b),
            log_summary=_log_stage4b,
        ),
        "stage-5a": StageDefinition(
            stage_id="stage-5a",
            depends_on=frozenset({"stage-4", "stage-2"}),
            contract=STAGE_CONTRACTS["stage-5a"],
            bind_inputs=_bind_stage5a,
            runner=dag.stage5a,
            skip_restore=True,
            log_summary=_log_stage5a,
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
            log_summary=_log_stage5b,
        ),
        "stage-6": StageDefinition(
            stage_id="stage-6",
            depends_on=frozenset({"stage-5b", "stage-1a", "stage-1b"}),
            contract=STAGE_CONTRACTS["stage-6"],
            bind_inputs=_bind_stage6,
            runner=dag.stage6,
            log_summary=_log_stage6,
        ),
    }


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


@property
def _get_registry():
    _ensure_initialized()
    return _registry


class _RegistryAccessor:
    """Module-level accessor for lazily-initialized registry and execution order."""

    @staticmethod
    def get_registry() -> dict[str, StageDefinition]:
        _ensure_initialized()
        assert _registry is not None
        return _registry

    @staticmethod
    def get_execution_order() -> tuple[str, ...]:
        _ensure_initialized()
        assert _execution_order is not None
        return _execution_order


# Public API
STAGES = _RegistryAccessor()


def get_stage_registry() -> dict[str, StageDefinition]:
    return STAGES.get_registry()


def get_execution_order() -> tuple[str, ...]:
    return STAGES.get_execution_order()
