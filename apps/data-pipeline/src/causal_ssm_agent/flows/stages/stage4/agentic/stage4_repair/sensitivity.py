"""Jacobian-sensitivity repair localization for Stage 4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .helpers import _find_block_for_parameter, _parameter_construct_names
from .planning import build_repair_plan
from .prior import _advance_repair_scope, _build_scope_candidates
from .types import RepairReasons, Stage4FailureLocalization

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair.types import (
        ResolvedRepairPlan,
    )
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import Stage4Runtime
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation

_SENSITIVITY_LOADING_MIN = 0.1
_SENSITIVITY_DIRECT_LOADING_COVERAGE = 0.6
_SENSITIVITY_MAX_DIRECT_PARAMETERS = 4
_SENSITIVITY_MAX_SUPPORTING_PARAMETERS = 8
_SENSITIVITY_DRIFT_BLOCK_KINDS = frozenset({"dynamics_prior", "effect_prior"})


def _failing_sensitivity_directions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return weak normalized directions that should block Stage 4 completion."""
    return sorted(
        [
            direction
            for direction in payload.get("weak_directions", [])
            if isinstance(direction, dict) and direction.get("status") == "fail"
        ],
        key=lambda item: (
            float(item.get("normalized_singular_value", float("inf"))),
            int(item.get("index", 0)),
        ),
    )


def _resolve_loading_parameter_name(
    plan: Stage4Plan,
    loading: dict[str, Any],
) -> str | None:
    """Resolve one weak-direction loading to a Stage 4 semantic parameter name."""
    for key in ("interpretable_parameter", "parameter"):
        candidate = loading.get(key)
        if not isinstance(candidate, str):
            continue
        if _find_block_for_parameter(plan, candidate) is not None:
            return candidate
    return None


def _ranked_direction_parameters(
    plan: Stage4Plan,
    direction: dict[str, Any],
) -> list[tuple[str, float]]:
    """Return unique semantic parameter hints ranked by absolute loading."""
    ranked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for loading in direction.get("top_loadings", []):
        if not isinstance(loading, dict):
            continue
        parameter_name = _resolve_loading_parameter_name(plan, loading)
        if parameter_name is None or parameter_name in seen:
            continue
        try:
            abs_loading = float(loading.get("abs_loading", abs(float(loading["loading"]))))
        except (KeyError, TypeError, ValueError):
            continue
        if abs_loading < _SENSITIVITY_LOADING_MIN:
            continue
        ranked.append((parameter_name, abs_loading))
        seen.add(parameter_name)
    return ranked


def _partition_direction_parameters(
    ranked_parameters: list[tuple[str, float]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split dominant and supporting parameters using cumulative normalized loadings."""
    if not ranked_parameters:
        return (), ()

    total_weight = sum(weight * weight for _, weight in ranked_parameters)
    direct: list[str] = []
    supporting: list[str] = []
    cumulative_weight = 0.0

    for parameter_name, weight in ranked_parameters:
        if len(direct) < _SENSITIVITY_MAX_DIRECT_PARAMETERS and (
            not direct
            or cumulative_weight / max(total_weight, 1e-12) < _SENSITIVITY_DIRECT_LOADING_COVERAGE
        ):
            direct.append(parameter_name)
            cumulative_weight += weight * weight
            continue
        if len(supporting) < _SENSITIVITY_MAX_SUPPORTING_PARAMETERS:
            supporting.append(parameter_name)

    return tuple(direct), tuple(supporting)


def _format_sensitivity_reason(
    direction: dict[str, Any],
    *,
    direct_parameters: tuple[str, ...],
    supporting_parameters: tuple[str, ...],
) -> str:
    """Build one concrete repair reason from the weakest normalized direction."""
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_text import summarize_stage4_names

    try:
        normalized_sv = f"{float(direction.get('normalized_singular_value')):.3g}"
    except (TypeError, ValueError):
        normalized_sv = "unknown"

    reason = (
        "Jacobian sensitivity found weak normalized direction "
        f"{direction.get('index')} (normalized singular value={normalized_sv})"
    )
    if direct_parameters:
        reason += f" dominated by {summarize_stage4_names(list(direct_parameters))}"
    if supporting_parameters:
        reason += (
            f"; secondary coupled parameters: {summarize_stage4_names(list(supporting_parameters))}"
        )
    return reason + "."


def _localize_sensitivity_failure(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureLocalization:
    """Project the weakest failing normalized sensitivity direction into Stage 4 scope hints."""
    payload = validation.sensitivity_payload or {}
    fail_directions = _failing_sensitivity_directions(payload)
    if not fail_directions:
        raise ValueError("Stage 4 sensitivity routing requires a failing weak direction")

    direction = fail_directions[0]
    ranked_parameters = _ranked_direction_parameters(plan, direction)
    direct_parameters, supporting_parameters = _partition_direction_parameters(ranked_parameters)
    parameter_hints = tuple(dict.fromkeys([*direct_parameters, *supporting_parameters]))
    construct_names = tuple(
        dict.fromkeys(_parameter_construct_names(plan.repair_topology, parameter_hints))
    )
    owner_blocks = tuple(
        block
        for parameter_name in parameter_hints
        if (block := _find_block_for_parameter(plan, parameter_name)) is not None
    )
    drift_related = any(block.kind in _SENSITIVITY_DRIFT_BLOCK_KINDS for block in owner_blocks)
    diagnostic_code = "jacobian_sensitivity_drift" if drift_related else "jacobian_sensitivity"
    reason = _format_sensitivity_reason(
        direction,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
    )
    has_global_failure = not parameter_hints
    return Stage4FailureLocalization(
        failure_family=(
            (diagnostic_code, f"direction:{int(direction.get('index', 0))}"),
            tuple(sorted(construct_names)),
            tuple(sorted(parameter_hints)),
        ),
        diagnostic_codes=(diagnostic_code,),
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        manifest_names=(),
        construct_names=construct_names,
        validator_repair_scope=None,
        validator_parameter_hints=(),
        pathology_certificate=None,
        has_global_failure=has_global_failure,
        issues_text=reason.lower(),
        reasons=RepairReasons(
            default=reason,
            support=None,
            drift=reason if drift_related else None,
            validator=None,
            global_=reason if has_global_failure else None,
        ),
    )


def classify_sensitivity_failure_blocks(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    validation: AssemblyValidation | None,
    runtime: Stage4Runtime,
) -> ResolvedRepairPlan:
    """Route Jacobian-sensitivity failures through the existing deterministic repair ladder."""
    if validation is None or not validation.has_sensitivity_failure:
        raise ValueError(
            "Stage 4 sensitivity repair classification requires a failing sensitivity payload"
        )

    localization = _localize_sensitivity_failure(plan, validation)
    candidates = _build_scope_candidates(plan, localization)
    chosen_scope = _advance_repair_scope(
        runtime,
        failure_family=localization.failure_family,
        candidates=candidates,
    )
    if chosen_scope is not None:
        return build_repair_plan(plan, chosen_scope)

    raise ValueError(
        "Stage 4 could not derive a concrete Jacobian-sensitivity repair scope "
        f"for {active_block.id!r}"
    )
