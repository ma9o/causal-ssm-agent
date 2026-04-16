"""Prior-predictive repair scope synthesis and routing for Stage 4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .helpers import (
    _certificate_improved,
    _materialize_scope_candidate,
    _parameter_construct_names,
    _require_reason,
    _validator_scope_construct_names,
)
from .localization import _localize_prior_failure
from .planning import build_repair_plan
from .types import (
    _DRIFT_RELATED_CODES,
    _GLOBAL_REVIEW_SCOPE_RANK,
    _MAX_SCOPE_ATTEMPTS,
    _VALIDATOR_SCOPE_RANK,
    ResolvedRepairPlan,
    ResolvedRepairScope,
    Stage4FailureLocalization,
    Stage4ScopeCandidateSpec,
    Stage4ScopeCandidateStrategy,
)

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import Stage4Runtime
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation


def _support_failure_scope(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> Stage4ScopeCandidateSpec | None:
    """Localize support violations to indicator-decision blocks when possible."""
    if (
        "support check" not in localization.issues_text
        and "outside support" not in localization.issues_text
    ):
        return None

    topology = plan.repair_topology
    for indicator_name, block_id in topology.indicator_to_decision_block_id.items():
        if indicator_name in localization.issues_text:
            return Stage4ScopeCandidateSpec(
                scope_kind="likelihood_support",
                scope_rank=0,
                reason=_require_reason(
                    localization.reasons.support,
                    context="likelihood support repair",
                ),
                prompt_block_hints=(block_id,),
                scope_token=block_id,
            )

    for block in plan.model_blocks:
        if block.kind == "indicator_decision":
            return Stage4ScopeCandidateSpec(
                scope_kind="likelihood_support",
                scope_rank=0,
                reason=_require_reason(
                    localization.reasons.support,
                    context="likelihood support repair",
                ),
                prompt_block_hints=(block.id,),
                scope_token=block.id,
            )
    return None


def _is_drift_related(localization: Stage4FailureLocalization) -> bool:
    """Whether the failure should be repaired through drift-structured scopes."""
    if localization.validator_repair_scope is not None:
        return True
    if any(code in _DRIFT_RELATED_CODES for code in localization.diagnostic_codes):
        return True
    return any(
        parameter_name.startswith("beta_") for parameter_name in localization.parameter_hints
    )


def _support_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Emit support-driven candidate scopes, if any."""
    support_scope = _support_failure_scope(plan, localization)
    return () if support_scope is None else (support_scope,)


def _validator_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Emit validator-owned candidate scopes."""
    del plan
    if localization.validator_repair_scope is not None:
        return (
            Stage4ScopeCandidateSpec(
                scope_kind="validator_scope",
                scope_rank=_VALIDATOR_SCOPE_RANK,
                reason=_require_reason(
                    localization.reasons.validator,
                    localization.reasons.drift,
                    context="validator_scope",
                ),
                construct_names=_validator_scope_construct_names(
                    localization.validator_repair_scope
                ),
            ),
        )
    return ()


def _drift_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Emit drift-structured candidate scopes."""
    if _is_drift_related(localization) and localization.parameter_hints:
        return (
            Stage4ScopeCandidateSpec(
                scope_kind="local_drift_motif",
                scope_rank=0,
                reason=_require_reason(
                    localization.reasons.drift,
                    localization.reasons.validator,
                    context="local_drift_motif",
                ),
                parameter_names=localization.parameter_hints,
                construct_names=_parameter_construct_names(
                    plan.repair_topology, localization.parameter_hints
                ),
            ),
            Stage4ScopeCandidateSpec(
                scope_kind="reciprocal_pair",
                scope_rank=1,
                reason=_require_reason(
                    localization.reasons.drift,
                    localization.reasons.validator,
                    context="reciprocal_pair",
                ),
                parameter_names=tuple(
                    sorted(
                        {
                            *localization.parameter_hints,
                            *(
                                plan.repair_topology.reciprocal_parameter_by_parameter.get(
                                    parameter_name
                                )
                                for parameter_name in localization.parameter_hints
                            ),
                        }
                        - {None}
                    )
                ),
                construct_names=_parameter_construct_names(
                    plan.repair_topology, localization.parameter_hints
                ),
            ),
            Stage4ScopeCandidateSpec(
                scope_kind="scc_drift_subsystem",
                scope_rank=2,
                reason=_require_reason(
                    localization.reasons.drift,
                    localization.reasons.validator,
                    context="scc_drift_subsystem",
                ),
                parameter_names=localization.parameter_hints,
                construct_names=localization.construct_names,
            ),
        )
    return ()


def _direct_writer_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Emit direct-writer candidate scopes for non-drift failures."""
    del plan
    if not _is_drift_related(localization) and localization.direct_parameters:
        return (
            Stage4ScopeCandidateSpec(
                scope_kind="direct_writer_blocks",
                scope_rank=0,
                reason=_require_reason(
                    localization.reasons.default,
                    context="direct_writer_blocks",
                ),
                parameter_names=localization.direct_parameters,
                construct_names=localization.construct_names,
            ),
        )
    return ()


def _scale_fallback_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Escalate local scale mismatch repairs to the full prior review when needed."""
    del plan
    if "scale_mismatch" not in localization.diagnostic_codes:
        return ()
    return (
        Stage4ScopeCandidateSpec(
            scope_kind="global_prior_review",
            scope_rank=1,
            reason=_require_reason(
                localization.reasons.default,
                context="scale_mismatch global prior review",
            ),
            scope_token="prior_system",
        ),
    )


def _global_scope_candidate_specs(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> tuple[Stage4ScopeCandidateSpec, ...]:
    """Emit whole-system candidate scopes for unattributed global failures."""
    del plan
    if localization.has_global_failure:
        return (
            Stage4ScopeCandidateSpec(
                scope_kind="global_prior_review",
                scope_rank=_GLOBAL_REVIEW_SCOPE_RANK,
                reason=_require_reason(
                    localization.reasons.global_,
                    localization.reasons.default,
                    context="global_prior_review",
                ),
                parameter_names=localization.parameter_hints,
                construct_names=localization.construct_names,
                scope_token="prior_system",
            ),
        )
    return ()


_SCOPE_CANDIDATE_STRATEGIES: tuple[Stage4ScopeCandidateStrategy, ...] = (
    Stage4ScopeCandidateStrategy(
        name="support",
        build_specs=_support_scope_candidate_specs,
        stop_after_match=True,
    ),
    Stage4ScopeCandidateStrategy(
        name="validator",
        build_specs=_validator_scope_candidate_specs,
    ),
    Stage4ScopeCandidateStrategy(
        name="drift",
        build_specs=_drift_scope_candidate_specs,
    ),
    Stage4ScopeCandidateStrategy(
        name="direct_writer",
        build_specs=_direct_writer_scope_candidate_specs,
    ),
    Stage4ScopeCandidateStrategy(
        name="scale_fallback",
        build_specs=_scale_fallback_scope_candidate_specs,
    ),
    Stage4ScopeCandidateStrategy(
        name="global",
        build_specs=_global_scope_candidate_specs,
    ),
)


def _build_scope_candidates(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> list[ResolvedRepairScope]:
    """Build ordered deterministic scope candidates for this failure family."""
    candidates: list[ResolvedRepairScope] = []
    seen_scope_keys: set[str] = set()
    for strategy in _SCOPE_CANDIDATE_STRATEGIES:
        emitted = False
        for spec in strategy.build_specs(plan, localization):
            if (
                not spec.parameter_names
                and not spec.construct_names
                and not spec.prompt_block_hints
                and spec.scope_token is None
            ):
                continue
            scope = _materialize_scope_candidate(localization, spec)
            if scope.scope_key in seen_scope_keys:
                continue
            seen_scope_keys.add(scope.scope_key)
            candidates.append(scope)
            emitted = True
        if emitted and strategy.stop_after_match:
            break
    return candidates


def _advance_repair_scope(
    runtime: Stage4Runtime,
    *,
    failure_family: tuple[Any, ...],
    candidates: list[ResolvedRepairScope],
) -> ResolvedRepairScope | None:
    """Advance monotonically through the deterministic scope ladder."""
    if not candidates:
        return None

    ordered_candidates = [
        candidate
        for _index, candidate in sorted(
            enumerate(candidates),
            key=lambda item: (item[1].scope_rank, item[0]),
        )
    ]
    campaign = runtime.domain.repair_campaign
    if campaign is None or campaign.failure_family_key != failure_family:
        return ordered_candidates[0]

    current_scope = next(
        (
            candidate
            for candidate in ordered_candidates
            if candidate.scope_key == campaign.scope_key
        ),
        None,
    )
    if (
        current_scope is not None
        and campaign.attempts_at_scope < _MAX_SCOPE_ATTEMPTS
        and (
            current_scope.pathology_certificate is None
            or campaign.best_certificate is None
            or _certificate_improved(
                current_scope.pathology_certificate,
                campaign.best_certificate,
            )
        )
    ):
        return current_scope

    for candidate in ordered_candidates:
        if candidate.scope_rank > campaign.scope_rank:
            return candidate
    return None


def classify_prior_failure_blocks(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    validation: AssemblyValidation | None,
    runtime: Stage4Runtime,
) -> ResolvedRepairPlan:
    """Route prior-validation failures to the smallest monotone repair scope."""
    if validation is None:
        raise ValueError(
            f"Stage 4 prior failure classification requires a validation payload for "
            f"{active_block.id!r}"
        )

    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    localization = _localize_prior_failure(plan, validation)
    candidates = _build_scope_candidates(plan, localization)
    chosen_scope = _advance_repair_scope(
        runtime,
        failure_family=localization.failure_family,
        candidates=candidates,
    )
    if chosen_scope is not None:
        return build_repair_plan(plan, chosen_scope)

    failed = [result for result in validation.prior_predictive_diagnostics if not result.is_valid]
    global_failures = [result for result in failed if result.parameter in GLOBAL_FAILURE_SITES]
    if global_failures:
        details = "; ".join(
            f"{result.code}:{','.join(result.related_parameters or [result.parameter])}"
            for result in global_failures
        )
        raise ValueError(
            "Stage 4 exhausted the deterministic repair-scope ladder for an unattributed "
            f"global prior-predictive failure. Details: {details}"
        )

    raise ValueError(
        "Stage 4 could not derive a concrete structural repair scope from the failing "
        f"diagnostics for {active_block.id!r}"
    )
