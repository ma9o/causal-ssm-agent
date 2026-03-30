"""Stage 4 repair routing helpers.

Owns diagnostic-to-repair attribution and loop-guard bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import PriorValidationResult

    from .stage4 import Stage4RepairRoute, Stage4Runtime
    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan


def _find_block_for_parameter(
    plan: Stage4Plan,
    parameter_name: str,
) -> Stage4FrontierBlock | None:
    """Map a validation parameter name back to the narrowest frontier block."""
    if parameter_name.startswith("scale_"):
        indicator_name = parameter_name.removeprefix("scale_")
        measurement_block_id = plan.indicator_to_measurement_block_id.get(indicator_name)
        if measurement_block_id is not None:
            return plan.get_block(measurement_block_id)
    block_id = plan.parameter_to_block_id.get(parameter_name)
    return None if block_id is None else plan.get_block(block_id)


def _repair_route(
    *,
    kind: str,
    block_ids: tuple[str, ...],
    reason: str,
    diagnostic_codes: tuple[str, ...] = (),
    related_parameters: tuple[str, ...] = (),
) -> Stage4RepairRoute:
    """Build a deterministic Stage 4 repair route."""
    from .stage4 import Stage4RepairRoute

    return Stage4RepairRoute(
        kind=kind,
        block_ids=block_ids,
        reason=reason,
        diagnostic_codes=diagnostic_codes,
        related_parameters=related_parameters,
    )


def _classify_compile_failure_route(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    feedback: str | None,
) -> Stage4RepairRoute:
    """Route compile failures back to the smallest matching block."""
    text = feedback or ""
    for block in plan.all_blocks:
        for token in [*block.variable_names, *block.parameter_names]:
            if token and token in text:
                return _repair_route(
                    kind="compile_local",
                    block_ids=(block.id,),
                    reason="compile failure mentions an active-scope token",
                )
    return _repair_route(
        kind="compile_active_block",
        block_ids=(active_block.id,),
        reason="compile failure could not be localized more narrowly",
    )


def _all_dynamics_block_ids(plan: Stage4Plan) -> tuple[str, ...]:
    """Return all dynamics-prior block ids in plan order."""
    return tuple(block.id for block in plan.prior_blocks if block.kind == "dynamics_prior")


def _ordered_block_ids(
    plan: Stage4Plan,
    block_ids: set[str],
) -> tuple[str, ...]:
    """Return block ids in deterministic plan order."""
    return tuple(block.id for block in plan.all_blocks if block.id in block_ids)


def _prior_failure_signature(validation: AssemblyValidation | None) -> tuple[Any, ...]:
    """Build a stable signature for the current invalid PP diagnostics."""
    if validation is None:
        return ()

    failed = [result for result in validation.prior_predictive_diagnostics if not result.is_valid]
    signature: list[tuple[Any, ...]] = []
    for result in failed:
        related_parameters = tuple(
            sorted(
                dict.fromkeys(
                    result.related_parameters or ([result.parameter] if result.parameter else [])
                )
            )
        )
        supporting_codes = tuple(sorted(dict.fromkeys(result.supporting_codes or [])))
        signature.append((result.code, result.parameter, related_parameters, supporting_codes))
    return tuple(sorted(signature))


def _record_prior_failure_signature(
    runtime: Stage4Runtime,
    *,
    block_id: str,
    signature: tuple[Any, ...],
) -> int:
    """Record one PP failure signature for a block and return the repeat count."""
    previous = runtime.prior_failure_signatures.get(block_id)
    if previous == signature:
        repeat_count = runtime.prior_failure_repeat_counts.get(block_id, 0) + 1
    else:
        repeat_count = 1
    runtime.prior_failure_signatures[block_id] = signature
    runtime.prior_failure_repeat_counts[block_id] = repeat_count
    return repeat_count


def _clear_prior_failure_signature(runtime: Stage4Runtime, block_id: str) -> None:
    """Clear any remembered PP failure loop state for a block."""
    runtime.prior_failure_signatures.pop(block_id, None)
    runtime.prior_failure_repeat_counts.pop(block_id, None)


def _block_ids_for_repair_scope(
    plan: Stage4Plan,
    repair_scope: Any,
) -> tuple[str, ...]:
    """Map a structured repair scope to concrete Stage 4 blocks."""
    if repair_scope is None:
        return ()

    if getattr(repair_scope, "kind", None) != "dynamics_scc":
        return ()

    construct_names = tuple(getattr(repair_scope, "construct_names", ()) or ())
    if not construct_names:
        return _all_dynamics_block_ids(plan)

    block_ids = {
        plan.dynamics_block_id_by_construct[name]
        for name in construct_names
        if name in plan.dynamics_block_id_by_construct
    }
    if not block_ids:
        return _all_dynamics_block_ids(plan)
    return _ordered_block_ids(plan, block_ids)


def _global_prior_review_block_ids(plan: Stage4Plan) -> tuple[str, ...]:
    """Return the exceptional post-prior global repair block, if configured."""
    if plan.prior_review_block is None:
        return ()
    return (plan.prior_review_block.id,)


def _compile_diagnostics_for_supporting_codes(
    validation: AssemblyValidation,
    *,
    supporting_codes: set[str],
) -> list[PriorValidationResult]:
    """Return compile warnings referenced by a failing PP diagnostic."""
    if not supporting_codes:
        return []
    return [
        diagnostic
        for diagnostic in validation.compile_diagnostics
        if diagnostic.code in supporting_codes
    ]


def _bundle_hint_parameters(
    plan: Stage4Plan,
    validation: AssemblyValidation,
    failed: list[PriorValidationResult],
) -> tuple[str, ...]:
    """Recover deterministic parameter hints for a bounded repair bundle."""
    parameter_names: list[str] = []

    supporting_codes = {code for result in failed for code in result.supporting_codes if code}
    for diagnostic in _compile_diagnostics_for_supporting_codes(
        validation,
        supporting_codes=supporting_codes,
    ):
        for parameter_name in diagnostic.related_parameters or (
            [diagnostic.parameter] if diagnostic.parameter else []
        ):
            if _find_block_for_parameter(plan, parameter_name) is not None:
                parameter_names.append(parameter_name)

    if parameter_names:
        return tuple(sorted(dict.fromkeys(parameter_names)))

    for diagnostic in validation.prior_predictive_diagnostics:
        if not diagnostic.is_valid:
            continue
        if diagnostic.severity != "warning":
            continue
        for parameter_name in diagnostic.related_parameters or (
            [diagnostic.parameter] if diagnostic.parameter else []
        ):
            if _find_block_for_parameter(plan, parameter_name) is not None:
                parameter_names.append(parameter_name)

    return tuple(sorted(dict.fromkeys(parameter_names)))


def _synthesize_repair_bundle_block_ids(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    *,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Build a bounded multi-block repair bundle from direct-writer hints."""
    seed_block_ids: set[str] = set()
    seed_constructs: set[str] = set()

    for parameter_name in parameter_names:
        block = _find_block_for_parameter(plan, parameter_name)
        if block is None:
            continue
        seed_block_ids.add(block.id)
        seed_constructs.update(block.construct_names)

    if not seed_block_ids or not seed_constructs:
        return ()

    closed_constructs: set[str] = set()
    for construct_name in seed_constructs:
        closed_constructs.update(
            plan.scc_construct_names_by_construct.get(construct_name, (construct_name,))
        )

    bundle_block_ids = set(seed_block_ids)
    for construct_name in closed_constructs:
        dynamics_block_id = plan.dynamics_block_id_by_construct.get(construct_name)
        if dynamics_block_id is not None:
            bundle_block_ids.add(dynamics_block_id)

    if active_block.id in bundle_block_ids and len(bundle_block_ids) > 1:
        bundle_block_ids.remove(active_block.id)

    return _ordered_block_ids(plan, bundle_block_ids)


def _classify_prior_failure_blocks(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    validation: AssemblyValidation | None,
) -> Stage4RepairRoute:
    """Route prior-validation failures back to the smallest repairable scope."""
    if validation is None:
        return _repair_route(
            kind="active_block_fallback",
            block_ids=(active_block.id,),
            reason="missing validation payload",
        )

    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    failed = [result for result in validation.prior_predictive_diagnostics if not result.is_valid]
    repair_block_ids: set[str] = set()
    route_codes = tuple(sorted(dict.fromkeys(result.code for result in failed if result.code)))
    route_parameters = tuple(
        sorted(
            dict.fromkeys(
                parameter_name
                for result in failed
                for parameter_name in (
                    result.related_parameters or ([result.parameter] if result.parameter else [])
                )
                if parameter_name
            )
        )
    )
    for result in failed:
        repair_block_ids.update(
            _block_ids_for_repair_scope(plan, getattr(result, "repair_scope", None))
        )
    if repair_block_ids:
        return _repair_route(
            kind="repair_scope",
            block_ids=_ordered_block_ids(plan, repair_block_ids),
            reason="diagnostic supplied an explicit repair scope",
            diagnostic_codes=route_codes,
            related_parameters=route_parameters,
        )

    local_block_ids: set[str] = set()
    for result in failed:
        for parameter_name in (result.parameter, *result.related_parameters):
            block = _find_block_for_parameter(plan, parameter_name)
            if block is not None:
                local_block_ids.add(block.id)
    if local_block_ids:
        return _repair_route(
            kind="direct_writer_blocks",
            block_ids=_ordered_block_ids(plan, local_block_ids),
            reason="diagnostic related_parameters map directly to Stage 4 blocks",
            diagnostic_codes=route_codes,
            related_parameters=route_parameters,
        )

    bundle_hint_parameters = _bundle_hint_parameters(plan, validation, failed)
    bundle_block_ids = _synthesize_repair_bundle_block_ids(
        plan,
        active_block,
        parameter_names=bundle_hint_parameters,
    )
    if bundle_block_ids:
        return _repair_route(
            kind="bounded_repair_bundle",
            block_ids=bundle_block_ids,
            reason=(
                "global PP failure is generic, but supporting diagnostics bound a "
                "smaller dependency-closed repair bundle"
            ),
            diagnostic_codes=route_codes,
            related_parameters=tuple(
                sorted(dict.fromkeys([*route_parameters, *bundle_hint_parameters]))
            ),
        )

    issues_text = " ".join(result.issue or "" for result in failed).lower()
    if "support check" in issues_text or "outside support" in issues_text:
        for indicator_name, block_id in plan.indicator_to_decision_block_id.items():
            if indicator_name in issues_text:
                return _repair_route(
                    kind="likelihood_support",
                    block_ids=(block_id,),
                    reason="global support failure names an indicator likelihood",
                    diagnostic_codes=route_codes,
                    related_parameters=route_parameters,
                )
        for block in plan.model_blocks:
            if block.kind == "indicator_decision":
                return _repair_route(
                    kind="likelihood_support",
                    block_ids=(block.id,),
                    reason="support failure requires indicator-decision repair",
                    diagnostic_codes=route_codes,
                    related_parameters=route_parameters,
                )

    if any(result.parameter == "dynamics_stability" for result in failed):
        dynamics_block_ids = _all_dynamics_block_ids(plan)
        if dynamics_block_ids:
            return _repair_route(
                kind="dynamics_family",
                block_ids=dynamics_block_ids,
                reason="dynamics stability failure requires the dynamics block family",
                diagnostic_codes=route_codes,
                related_parameters=route_parameters,
            )

    prior_review_block_ids = _global_prior_review_block_ids(plan)
    if prior_review_block_ids:
        return _repair_route(
            kind="global_prior_review",
            block_ids=prior_review_block_ids,
            reason="global PP failure could not be localized to a bounded repair scope",
            diagnostic_codes=route_codes,
            related_parameters=route_parameters,
        )

    global_failures = [result for result in failed if result.parameter in GLOBAL_FAILURE_SITES]
    if global_failures:
        details = "; ".join(
            f"{result.code}:{','.join(result.related_parameters or [result.parameter])}"
            for result in global_failures
        )
        raise ValueError(
            "Unattributed global prior-predictive failure cannot be repaired by reopening "
            f"the active block {active_block.id!r}. Details: {details}"
        )

    return _repair_route(
        kind="active_block_fallback",
        block_ids=(active_block.id,),
        reason="non-global failure could not be localized more narrowly",
        diagnostic_codes=route_codes,
        related_parameters=route_parameters,
    )
