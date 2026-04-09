"""Stage 4 repair routing helpers.

Owns failure localization, structural scope resolution, and monotone
campaign-aware repair escalation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan, Stage4RepairTopology
    from .stage4_state import Stage4RepairCampaignState, Stage4Runtime


_MAX_SCOPE_ATTEMPTS = 2
_GLOBAL_REVIEW_SCOPE_RANK = 3
_VALIDATOR_SCOPE_RANK = 2
_DRIFT_RELATED_CODES = frozenset(
    {
        "dt_ct_approximation_warning",
        "partial_dynamics_budget_exhausted",
        "partial_dynamics_row_budget_exceeded",
        "partial_dynamics_stability",
        "partial_row_budget_exceeded",
    }
)

Stage4ValidationOutcome = Literal[
    "accepted",
    "compile_error",
    "prior_predictive_failure",
]


@dataclass(frozen=True)
class RepairReasons:
    """Candidate repair reasons extracted from diagnostic evidence."""

    default: str | None
    support: str | None
    drift: str | None
    validator: str | None
    global_: str | None


@dataclass(frozen=True)
class Stage4FailureLocalization:
    """Localized evidence for one Stage 4 prior-validation failure family."""

    failure_family: tuple[Any, ...]
    diagnostic_codes: tuple[str, ...]
    direct_parameters: tuple[str, ...]
    supporting_parameters: tuple[str, ...]
    construct_names: tuple[str, ...]
    validator_repair_scope: PriorRepairScope | None
    pathology_certificate: PriorPathologyCertificate | None
    has_global_failure: bool
    issues_text: str
    reasons: RepairReasons

    @property
    def parameter_hints(self) -> tuple[str, ...]:
        """Return deterministic seed parameters for scope synthesis."""
        return tuple(dict.fromkeys([*self.direct_parameters, *self.supporting_parameters]))


@dataclass(frozen=True)
class Stage4FailureEvidence:
    """Normalized diagnostic evidence for one failed prior-predictive validation."""

    topology: Stage4RepairTopology
    failed_diagnostics: tuple[PriorValidationResult, ...]
    supporting_compile_diagnostics: tuple[PriorValidationResult, ...]
    diagnostic_codes: tuple[str, ...]
    supporting_codes: tuple[str, ...]
    global_failure_sites: frozenset[str]

    @property
    def all_reason_diagnostics(self) -> tuple[PriorValidationResult, ...]:
        """Return diagnostics eligible to contribute user-facing repair reasons."""
        return (*self.failed_diagnostics, *self.supporting_compile_diagnostics)


@dataclass(frozen=True)
class ResolvedRepairScope:
    """Deterministic structural repair scope independent of prompt blocks."""

    scope_kind: str
    scope_rank: int
    scope_key: str
    reason: str
    failure_family: tuple[Any, ...]
    parameter_names: tuple[str, ...] = ()
    construct_names: tuple[str, ...] = ()
    prompt_block_hints: tuple[str, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()
    pathology_certificate: PriorPathologyCertificate | None = None


@dataclass(frozen=True)
class Stage4RepairScopeStrategy:
    """Strategy for projecting one structural repair scope into prompt execution."""

    scope_kind: str
    resolve_prompt_block_ids: Callable[[Stage4Plan, ResolvedRepairScope], tuple[str, ...]]
    project_prompt_block: Callable[
        [Stage4Plan, Stage4FrontierBlock, ResolvedRepairScope],
        Stage4FrontierBlock | None,
    ]
    uses_repair_campaign: bool = False


@dataclass(frozen=True)
class Stage4ScopeCandidateSpec:
    """Candidate structural scope emitted before prompt-block projection."""

    scope_kind: str
    scope_rank: int
    reason: str
    parameter_names: tuple[str, ...] = ()
    construct_names: tuple[str, ...] = ()
    prompt_block_hints: tuple[str, ...] = ()
    scope_token: str | None = None


@dataclass(frozen=True)
class Stage4ScopeCandidateStrategy:
    """Strategy for emitting candidate repair scopes from localized evidence."""

    name: str
    build_specs: Callable[
        [Stage4Plan, Stage4FailureLocalization],
        tuple[Stage4ScopeCandidateSpec, ...],
    ]
    stop_after_match: bool = False


@dataclass(frozen=True)
class ResolvedRepairPlan:
    """Prompt execution plan for one resolved structural repair scope."""

    scope: ResolvedRepairScope
    prompt_blocks: tuple[Stage4FrontierBlock, ...]
    requires_barrier_validation: bool = False
    uses_repair_campaign: bool = False

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.id for block in self.prompt_blocks)


@dataclass(frozen=True)
class Stage4ValidationOutcomeDecision:
    """Typed classification for one validation outcome."""

    outcome: Stage4ValidationOutcome
    repair_plan: ResolvedRepairPlan | None = None


@dataclass(frozen=True)
class Stage4PriorRepairDecision:
    """Typed reducer decision for a prior submission after repair routing."""

    repair_plan: ResolvedRepairPlan | None
    accepted_block_id: str | None
    route_kind: Literal["accepted", "repair_single", "repair_multi", "rejected"]

    @property
    def promote_campaign_feedback(self) -> bool:
        """Whether reducer feedback should surface campaign-wide routing."""
        return self.route_kind == "repair_multi"


def _find_block_for_parameter(
    plan: Stage4Plan,
    parameter_name: str,
) -> Stage4FrontierBlock | None:
    """Map a validation parameter name back to the narrowest prompt block."""
    topology = plan.repair_topology
    if parameter_name.startswith("scale_"):
        indicator_name = parameter_name.removeprefix("scale_")
        measurement_block_id = topology.get_measurement_block_id(indicator_name)
        if measurement_block_id is not None:
            return plan.get_block(measurement_block_id)
        for construct_name, indicator_names in topology.indicator_names_by_construct.items():
            if indicator_name not in indicator_names:
                continue
            dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
            if dynamics_block_id is not None:
                return plan.get_block(dynamics_block_id)
            break
    block_id = topology.get_parameter_block_id(parameter_name)
    return None if block_id is None else plan.get_block(block_id)


def _ordered_block_ids(plan: Stage4Plan, block_ids: set[str]) -> tuple[str, ...]:
    """Return block ids in deterministic plan order."""
    return tuple(block.id for block in plan.all_blocks if block.id in block_ids)


def _resolved_repair_scope(
    *,
    scope_kind: str,
    scope_rank: int,
    reason: str,
    failure_family: tuple[Any, ...],
    parameter_names: tuple[str, ...] = (),
    construct_names: tuple[str, ...] = (),
    prompt_block_hints: tuple[str, ...] = (),
    diagnostic_codes: tuple[str, ...] = (),
    pathology_certificate: PriorPathologyCertificate | None = None,
    scope_token: str | None = None,
) -> ResolvedRepairScope:
    """Build a deterministic resolved repair scope."""
    scope_parts = [*construct_names, *parameter_names]
    scope_suffix = (
        "|".join(dict.fromkeys(scope_parts)) if scope_parts else (scope_token or "global")
    )
    scope_key = f"{scope_kind}:{scope_suffix}"
    return ResolvedRepairScope(
        scope_kind=scope_kind,
        scope_rank=scope_rank,
        scope_key=scope_key,
        reason=reason,
        failure_family=failure_family,
        parameter_names=parameter_names,
        construct_names=construct_names,
        prompt_block_hints=prompt_block_hints,
        diagnostic_codes=diagnostic_codes,
        pathology_certificate=pathology_certificate,
    )


def _materialize_scope_candidate(
    localization: Stage4FailureLocalization,
    spec: Stage4ScopeCandidateSpec,
) -> ResolvedRepairScope:
    """Materialize one candidate scope spec with localization-owned metadata."""
    return _resolved_repair_scope(
        scope_kind=spec.scope_kind,
        scope_rank=spec.scope_rank,
        reason=spec.reason,
        failure_family=localization.failure_family,
        parameter_names=spec.parameter_names,
        construct_names=spec.construct_names,
        prompt_block_hints=spec.prompt_block_hints,
        diagnostic_codes=localization.diagnostic_codes,
        pathology_certificate=localization.pathology_certificate,
        scope_token=spec.scope_token,
    )


def _identity_prompt_block_projection(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Keep the authored prompt block unchanged for this repair scope."""
    del plan, scope
    return block


def _narrow_effect_prompt_block_to_scc(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Project a structural drift scope onto the narrowest authored effect prompt."""
    if block.kind != "effect_prior":
        return block

    allowed_constructs = set(scope.construct_names)
    if not allowed_constructs:
        return block

    topology = plan.repair_topology
    allowed_parameter_names = tuple(
        parameter_name
        for parameter_name in block.parameter_names
        if set(topology.parameter_construct_names.get(parameter_name, ())).issubset(
            allowed_constructs
        )
    )
    if not allowed_parameter_names:
        return None
    if allowed_parameter_names == block.parameter_names:
        return block

    prompt_construct_names = tuple(
        construct_name
        for construct_name in block.construct_names
        if construct_name
        in {
            related_construct
            for parameter_name in allowed_parameter_names
            for related_construct in topology.parameter_construct_names.get(parameter_name, ())
        }
    )
    prompt_variable_names = tuple(
        indicator_name
        for construct_name in prompt_construct_names
        for indicator_name in topology.indicator_names_by_construct.get(construct_name, ())
    )
    return replace(
        block,
        label=f"{block.label} (internal SCC parameters only)",
        construct_names=prompt_construct_names,
        variable_names=prompt_variable_names,
        parameter_names=allowed_parameter_names,
        expand_neighbor_topology=False,
    )


def classify_validation_outcome(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    validation: AssemblyValidation | None,
    runtime: Stage4Runtime,
    *,
    feedback: str | None,
    include_prior_predictive: bool = True,
) -> Stage4ValidationOutcomeDecision:
    """Classify validation into acceptance or a concrete repair route."""
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        return Stage4ValidationOutcomeDecision(
            outcome="compile_error",
            repair_plan=classify_compile_failure_route(
                plan,
                active_block,
                getattr(validation, "compile_error", None) or feedback,
            ),
        )
    if (
        include_prior_predictive
        and validation is not None
        and getattr(validation, "pp_checked", False)
        and getattr(validation, "pp_valid", True) is False
    ):
        return Stage4ValidationOutcomeDecision(
            outcome="prior_predictive_failure",
            repair_plan=classify_prior_failure_blocks(
                plan,
                active_block,
                validation,
                runtime,
            ),
        )
    return Stage4ValidationOutcomeDecision(outcome="accepted")


def resolve_prior_repair_decision(
    *,
    active_block: Stage4FrontierBlock,
    repair_plan: ResolvedRepairPlan | None,
    campaign: Stage4RepairCampaignState | None,
    stage_output_present: bool,
) -> Stage4PriorRepairDecision:
    """Decide whether the active prior block remains accepted while routing repair."""
    if not stage_output_present:
        return Stage4PriorRepairDecision(
            repair_plan=repair_plan,
            accepted_block_id=None,
            route_kind="rejected",
        )
    if repair_plan is None:
        return Stage4PriorRepairDecision(
            repair_plan=None,
            accepted_block_id=active_block.id,
            route_kind="accepted",
        )

    widening_scope = (
        campaign is not None
        and repair_plan.scope.scope_rank > campaign.scope_rank
        and active_block.id in repair_plan.block_ids
    )
    accepted_block_id = None
    if active_block.id not in repair_plan.block_ids or (
        len(repair_plan.block_ids) > 1 and not widening_scope
    ):
        accepted_block_id = active_block.id

    route_kind = "repair_multi" if len(repair_plan.block_ids) > 1 else "repair_single"
    return Stage4PriorRepairDecision(
        repair_plan=repair_plan,
        accepted_block_id=accepted_block_id,
        route_kind=route_kind,
    )


def classify_compile_failure_route(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    feedback: str | None,
) -> ResolvedRepairPlan:
    """Route compile failures back to the smallest matching block."""
    text = feedback or ""
    failure_family = ("compile_failure", active_block.id)
    matching_block_ids = _compile_failure_matching_block_ids(plan, text)
    if matching_block_ids:
        chosen_block_id = _choose_compile_local_block_id(
            active_block=active_block,
            block_ids=matching_block_ids,
        )
        return build_repair_plan(
            plan,
            _resolved_repair_scope(
                scope_kind="compile_local",
                scope_rank=0,
                reason=_repair_reason_from_feedback(feedback),
                failure_family=failure_family,
                scope_token=chosen_block_id,
            ),
            prompt_block_ids=(chosen_block_id,),
            requires_barrier_validation=False,
        )
    if active_block.kind == "global_prior_review":
        return build_repair_plan(
            plan,
            _resolved_repair_scope(
                scope_kind="global_prior_review",
                scope_rank=_GLOBAL_REVIEW_SCOPE_RANK,
                reason=_repair_reason_from_feedback(feedback),
                failure_family=failure_family,
                scope_token="prior_system",
            ),
            requires_barrier_validation=False,
        )
    return build_repair_plan(
        plan,
        _resolved_repair_scope(
            scope_kind="compile_active_block",
            scope_rank=0,
            reason=_repair_reason_from_feedback(feedback),
            failure_family=failure_family,
            scope_token=active_block.id,
        ),
        prompt_block_ids=(active_block.id,),
        requires_barrier_validation=False,
    )


def _compile_failure_matching_block_ids(
    plan: Stage4Plan,
    text: str,
) -> tuple[str, ...]:
    """Return owner block ids for exact identifiers mentioned in compile feedback."""
    if not text:
        return ()

    topology = plan.repair_topology
    matched_block_ids: set[str] = set()

    for parameter_name in topology.parameter_to_block_id:
        if not _feedback_mentions_identifier(text, parameter_name):
            continue
        block = _find_block_for_parameter(plan, parameter_name)
        if block is not None:
            matched_block_ids.add(block.id)

    indicator_names = {
        *topology.indicator_to_decision_block_id,
        *topology.indicator_to_measurement_block_id,
    }
    for indicator_name in indicator_names:
        if not _feedback_mentions_identifier(text, indicator_name):
            continue
        block_id = topology.get_indicator_owner_block_id(indicator_name)
        if block_id is not None:
            matched_block_ids.add(block_id)

    return _ordered_block_ids(plan, matched_block_ids)


def _normalize_reason_text(text: str | None) -> str:
    """Collapse multiline validation text into one compact sentence."""
    return " ".join((text or "").split())


def _format_diagnostic_reason(result: PriorValidationResult) -> str | None:
    """Render one diagnostic into a user-facing repair reason."""
    issue = _normalize_reason_text(result.issue)
    suggested_adjustment = _normalize_reason_text(result.suggested_adjustment)
    if issue and suggested_adjustment:
        return f"{issue} Suggested fix: {suggested_adjustment}"
    return issue or suggested_adjustment or None


def _first_diagnostic_reason(
    diagnostics: Sequence[PriorValidationResult],
    *,
    predicate: Callable[[PriorValidationResult], bool] | None = None,
) -> str | None:
    """Return the first concrete reason matching the predicate."""
    for result in diagnostics:
        if predicate is not None and not predicate(result):
            continue
        reason = _format_diagnostic_reason(result)
        if reason:
            return reason
    return None


def _require_reason(*reasons: str | None, context: str) -> str:
    """Return the first concrete repair reason or raise."""
    for reason in reasons:
        if reason:
            return reason
    raise ValueError(f"Stage 4 repair classification requires a concrete reason for {context}")


def _repair_reason_from_feedback(feedback: str | None) -> str:
    """Extract a concrete repair reason from compile feedback."""
    compact = _normalize_reason_text(feedback)
    compact = re.sub(r"^(compile error:|validation errors:)\s*", "", compact, flags=re.IGNORECASE)
    return _require_reason(compact, context="compile failure")


def _feedback_mentions_identifier(text: str, identifier: str) -> bool:
    """Whether compile feedback mentions an identifier as a full token."""
    if not identifier:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    return bool(re.search(pattern, text))


def _choose_compile_local_block_id(
    *,
    active_block: Stage4FrontierBlock,
    block_ids: tuple[str, ...],
) -> str:
    """Choose the narrowest compile-local block, preferring the active owner when matched."""
    if active_block.id in block_ids:
        return active_block.id
    return block_ids[0]


def _all_dynamics_block_ids(plan: Stage4Plan) -> tuple[str, ...]:
    """Return all dynamics-prior block ids in plan order."""
    return tuple(block.id for block in plan.prior_blocks if block.kind == "dynamics_prior")


def _compile_diagnostics_for_supporting_codes(
    validation: AssemblyValidation,
    *,
    supporting_codes: set[str],
) -> list[PriorValidationResult]:
    """Return compile diagnostics referenced by a failing PP diagnostic."""
    if not supporting_codes:
        return []
    return [
        diagnostic
        for diagnostic in validation.compile_diagnostics
        if diagnostic.code in supporting_codes
    ]


def _certificate_order_key(certificate: PriorPathologyCertificate) -> tuple[float, float]:
    """Return a comparable severity key where lower means improvement."""
    secondary = (
        certificate.secondary_score if certificate.secondary_score is not None else float("inf")
    )
    return (certificate.primary_score, secondary)


def _certificate_improved(
    current: PriorPathologyCertificate | None,
    best_so_far: PriorPathologyCertificate | None,
) -> bool:
    """Whether the latest pathology certificate strictly improved at this scope."""
    if current is None or best_so_far is None:
        return False
    if current.kind != best_so_far.kind:
        return False
    return _certificate_order_key(current) < _certificate_order_key(best_so_far)


def _parameter_construct_names(
    topology: Stage4RepairTopology,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return ordered construct hints implied by semantic parameter names."""
    constructs: list[str] = []
    for parameter_name in parameter_names:
        constructs.extend(topology.parameter_construct_names.get(parameter_name, ()))
    return tuple(dict.fromkeys(constructs))


def _validator_scope_construct_names(repair_scope: PriorRepairScope | None) -> tuple[str, ...]:
    """Return construct names from a validator-owned repair scope."""
    if repair_scope is None:
        return ()
    return tuple(getattr(repair_scope, "construct_names", ()) or ())


def build_stage4_failure_evidence(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureEvidence:
    """Build the normalized evidence surface for a failed PP validation."""
    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    failed_diagnostics = tuple(
        result for result in validation.prior_predictive_diagnostics if not result.is_valid
    )
    diagnostic_codes = tuple(
        sorted(dict.fromkeys(result.code for result in failed_diagnostics if result.code))
    )
    supporting_codes = tuple(
        sorted({code for result in failed_diagnostics for code in result.supporting_codes if code})
    )
    supporting_compile_diagnostics = tuple(
        _compile_diagnostics_for_supporting_codes(
            validation,
            supporting_codes=set(supporting_codes),
        )
    )
    return Stage4FailureEvidence(
        topology=plan.repair_topology,
        failed_diagnostics=failed_diagnostics,
        supporting_compile_diagnostics=supporting_compile_diagnostics,
        diagnostic_codes=diagnostic_codes,
        supporting_codes=supporting_codes,
        global_failure_sites=frozenset(GLOBAL_FAILURE_SITES),
    )


def _diagnostic_parameter_names(
    plan: Stage4Plan,
    diagnostics: tuple[PriorValidationResult, ...],
) -> tuple[str, ...]:
    """Return authored parameter names referenced by diagnostics in sorted order."""
    return tuple(
        sorted(
            dict.fromkeys(
                parameter_name
                for result in diagnostics
                for parameter_name in (
                    result.related_parameters or ([result.parameter] if result.parameter else [])
                )
                if parameter_name and _find_block_for_parameter(plan, parameter_name) is not None
            )
        )
    )


def _validator_scope_from_failure_evidence(
    evidence: Stage4FailureEvidence,
) -> PriorRepairScope | None:
    """Return the first validator-owned repair scope from failed diagnostics."""
    return next(
        (
            result.repair_scope
            for result in evidence.failed_diagnostics
            if result.repair_scope is not None
        ),
        None,
    )


def _pathology_certificate_from_failure_evidence(
    evidence: Stage4FailureEvidence,
) -> PriorPathologyCertificate | None:
    """Return the dominant pathology certificate for same-scope retry gating."""
    failed_certificates = [
        result.pathology_certificate
        for result in evidence.failed_diagnostics
        if result.pathology_certificate is not None
    ]
    supporting_certificates = [
        diagnostic.pathology_certificate
        for diagnostic in evidence.supporting_compile_diagnostics
        if diagnostic.pathology_certificate is not None
    ]
    if failed_certificates:
        return max(failed_certificates, key=_certificate_order_key)
    if supporting_certificates:
        # Same-scope retry gating should prefer the certificate for the actual
        # failed PP pathology, not the supporting compile warning. Compile-side
        # certificates are only a fallback when PP has no comparable metric.
        return max(supporting_certificates, key=_certificate_order_key)
    return None


def _construct_names_from_failure_evidence(
    evidence: Stage4FailureEvidence,
    *,
    direct_parameters: tuple[str, ...],
    supporting_parameters: tuple[str, ...],
    validator_repair_scope: PriorRepairScope | None,
) -> tuple[str, ...]:
    """Return construct hints synthesized from authored parameters and validator scope."""
    construct_names = tuple(
        dict.fromkeys(
            [
                *_parameter_construct_names(evidence.topology, direct_parameters),
                *_parameter_construct_names(evidence.topology, supporting_parameters),
                *_validator_scope_construct_names(validator_repair_scope),
            ]
        )
    )
    if construct_names:
        return construct_names
    return tuple(
        dict.fromkeys(
            name
            for result in evidence.failed_diagnostics
            for parameter_name in (result.related_parameters or [])
            for name in evidence.topology.parameter_construct_names.get(parameter_name, ())
        )
    )


def _issues_text_from_failure_evidence(evidence: Stage4FailureEvidence) -> str:
    """Return the normalized lower-cased issue text across failed diagnostics."""
    return " ".join(result.issue or "" for result in evidence.failed_diagnostics).lower()


def _has_global_failure(evidence: Stage4FailureEvidence) -> bool:
    """Whether any failed diagnostic is a whole-system global failure."""
    return any(
        result.parameter in evidence.global_failure_sites for result in evidence.failed_diagnostics
    )


def _is_support_issue(result: PriorValidationResult) -> bool:
    """Whether the diagnostic describes a likelihood support violation."""
    issue = (result.issue or "").lower()
    return "support check" in issue or "outside support" in issue


def _reasons_from_failure_evidence(evidence: Stage4FailureEvidence) -> RepairReasons:
    """Extract user-facing repair reasons from the normalized evidence surface."""
    return RepairReasons(
        default=_first_diagnostic_reason(evidence.all_reason_diagnostics),
        support=_first_diagnostic_reason(
            evidence.all_reason_diagnostics,
            predicate=_is_support_issue,
        ),
        drift=_first_diagnostic_reason(
            evidence.all_reason_diagnostics,
            predicate=lambda result: (
                result.repair_scope is not None
                or result.code in _DRIFT_RELATED_CODES
                or any(
                    parameter_name.startswith("beta_")
                    for parameter_name in (
                        result.related_parameters
                        or ([result.parameter] if result.parameter else [])
                    )
                )
            ),
        ),
        validator=_first_diagnostic_reason(
            evidence.failed_diagnostics,
            predicate=lambda result: result.repair_scope is not None,
        ),
        global_=_first_diagnostic_reason(
            evidence.failed_diagnostics,
            predicate=lambda result: result.parameter in evidence.global_failure_sites,
        ),
    )


def _localize_prior_failure(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureLocalization:
    """Localize a failed PP validation into deterministic structural evidence."""
    evidence = build_stage4_failure_evidence(plan, validation)
    direct_parameters = _diagnostic_parameter_names(plan, evidence.failed_diagnostics)
    supporting_parameters = _diagnostic_parameter_names(
        plan,
        evidence.supporting_compile_diagnostics,
    )
    validator_repair_scope = _validator_scope_from_failure_evidence(evidence)
    construct_names = _construct_names_from_failure_evidence(
        evidence,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        validator_repair_scope=validator_repair_scope,
    )
    failure_family = (
        evidence.diagnostic_codes,
        evidence.supporting_codes,
        tuple(sorted(construct_names)),
    )
    return Stage4FailureLocalization(
        failure_family=failure_family,
        diagnostic_codes=evidence.diagnostic_codes,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        construct_names=construct_names,
        validator_repair_scope=validator_repair_scope,
        pathology_certificate=_pathology_certificate_from_failure_evidence(evidence),
        has_global_failure=_has_global_failure(evidence),
        issues_text=_issues_text_from_failure_evidence(evidence),
        reasons=_reasons_from_failure_evidence(evidence),
    )


def _block_ids_for_repair_scope(
    plan: Stage4Plan,
    repair_scope: PriorRepairScope | None,
) -> tuple[str, ...]:
    """Map a validator-owned repair scope to concrete prompt blocks."""
    if repair_scope is None:
        return ()
    if getattr(repair_scope, "kind", None) != "dynamics_scc":
        return ()
    construct_names = tuple(getattr(repair_scope, "construct_names", ()) or ())
    return _scc_drift_subsystem_block_ids(plan, construct_names)


def _local_drift_motif_block_ids(
    plan: Stage4Plan,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the smallest local drift motif for the seed parameters."""
    topology = plan.repair_topology
    bundle_block_ids: set[str] = set()
    constructs: list[str] = []
    for parameter_name in parameter_names:
        block = _find_block_for_parameter(plan, parameter_name)
        if block is None:
            continue
        bundle_block_ids.add(block.id)
        constructs.extend(
            topology.parameter_construct_names.get(parameter_name, block.construct_names)
        )

    for construct_name in dict.fromkeys(constructs):
        dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
        if dynamics_block_id is not None:
            bundle_block_ids.add(dynamics_block_id)

    return _ordered_block_ids(plan, bundle_block_ids)


def _reciprocal_pair_block_ids(
    plan: Stage4Plan,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return a reciprocal feedback-pair bundle when reverse edges exist."""
    topology = plan.repair_topology
    expanded_parameters = set(parameter_names)
    for parameter_name in parameter_names:
        reciprocal = topology.reciprocal_parameter_by_parameter.get(parameter_name)
        if reciprocal is not None:
            expanded_parameters.add(reciprocal)
    return _local_drift_motif_block_ids(plan, tuple(sorted(expanded_parameters)))


def _scc_drift_subsystem_block_ids(
    plan: Stage4Plan,
    construct_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the smallest SCC-closed drift subsystem for construct hints."""
    topology = plan.repair_topology
    if not construct_names:
        return _all_dynamics_block_ids(plan)

    closed_scc_ids: set[str] = set()
    for construct_name in construct_names:
        scc_id = topology.get_scc_id(construct_name)
        if scc_id is not None:
            closed_scc_ids.add(scc_id)

    if not closed_scc_ids:
        return ()

    bundle_block_ids: set[str] = set()
    for scc_id in closed_scc_ids:
        for construct_name in topology.scc_construct_names_by_id.get(scc_id, ()):
            dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
            if dynamics_block_id is not None:
                bundle_block_ids.add(dynamics_block_id)
        bundle_block_ids.update(topology.internal_effect_block_ids_by_scc_id.get(scc_id, ()))
    return _ordered_block_ids(plan, bundle_block_ids)


def _direct_writer_block_ids(
    plan: Stage4Plan,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return authored blocks that directly write the named parameters."""
    return _ordered_block_ids(
        plan,
        {
            block.id
            for parameter_name in parameter_names
            if (block := _find_block_for_parameter(plan, parameter_name)) is not None
        },
    )


def _global_prior_review_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Return the whole-system prior-review block when configured."""
    del scope
    prior_review_id = plan.prior_review_block_id
    return () if prior_review_id is None else (prior_review_id,)


def _prompt_hint_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Return prompt-block hints already embedded in the structural scope."""
    del plan
    return scope.prompt_block_hints


def _local_drift_motif_strategy_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Resolve the local drift motif for the scope's parameter hints."""
    return _local_drift_motif_block_ids(plan, scope.parameter_names)


def _reciprocal_pair_strategy_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Resolve the reciprocal-pair motif for the scope's parameter hints."""
    return _reciprocal_pair_block_ids(plan, scope.parameter_names)


def _scc_strategy_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Resolve the SCC-closed drift subsystem for the scope's construct hints."""
    return _scc_drift_subsystem_block_ids(plan, scope.construct_names)


def _direct_writer_strategy_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Resolve authored writer blocks for the scope's parameters."""
    return _direct_writer_block_ids(plan, scope.parameter_names)


_REPAIR_SCOPE_STRATEGIES: dict[str, Stage4RepairScopeStrategy] = {
    scope_kind: Stage4RepairScopeStrategy(
        scope_kind=scope_kind,
        resolve_prompt_block_ids=_prompt_hint_block_ids,
        project_prompt_block=_identity_prompt_block_projection,
        uses_repair_campaign=False,
    )
    for scope_kind in (
        "compile_local",
        "compile_active_block",
        "global_review",
        "likelihood_support",
        "model_spec_lock",
    )
}
_REPAIR_SCOPE_STRATEGIES.update(
    {
        "local_drift_motif": Stage4RepairScopeStrategy(
            scope_kind="local_drift_motif",
            resolve_prompt_block_ids=_local_drift_motif_strategy_block_ids,
            project_prompt_block=_identity_prompt_block_projection,
            uses_repair_campaign=True,
        ),
        "reciprocal_pair": Stage4RepairScopeStrategy(
            scope_kind="reciprocal_pair",
            resolve_prompt_block_ids=_reciprocal_pair_strategy_block_ids,
            project_prompt_block=_identity_prompt_block_projection,
            uses_repair_campaign=True,
        ),
        "scc_drift_subsystem": Stage4RepairScopeStrategy(
            scope_kind="scc_drift_subsystem",
            resolve_prompt_block_ids=_scc_strategy_block_ids,
            project_prompt_block=_narrow_effect_prompt_block_to_scc,
            uses_repair_campaign=True,
        ),
        "validator_scope": Stage4RepairScopeStrategy(
            scope_kind="validator_scope",
            resolve_prompt_block_ids=_scc_strategy_block_ids,
            project_prompt_block=_narrow_effect_prompt_block_to_scc,
            uses_repair_campaign=True,
        ),
        "direct_writer_blocks": Stage4RepairScopeStrategy(
            scope_kind="direct_writer_blocks",
            resolve_prompt_block_ids=_direct_writer_strategy_block_ids,
            project_prompt_block=_identity_prompt_block_projection,
            uses_repair_campaign=True,
        ),
        "global_prior_review": Stage4RepairScopeStrategy(
            scope_kind="global_prior_review",
            resolve_prompt_block_ids=_global_prior_review_block_ids,
            project_prompt_block=_identity_prompt_block_projection,
            uses_repair_campaign=True,
        ),
    }
)


def get_stage4_repair_scope_strategy(scope_kind: str) -> Stage4RepairScopeStrategy:
    """Return the repair-scope strategy registered for one structural scope kind."""
    strategy = _REPAIR_SCOPE_STRATEGIES.get(scope_kind)
    if strategy is None:
        raise ValueError(f"Unsupported Stage 4 repair scope kind {scope_kind!r}")
    return strategy


def build_repair_plan(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
    *,
    prompt_block_ids: tuple[str, ...] | None = None,
    requires_barrier_validation: bool | None = None,
) -> ResolvedRepairPlan:
    """Project a structural scope into the prompt blocks Stage 4 should run."""
    strategy = get_stage4_repair_scope_strategy(scope.scope_kind)
    if prompt_block_ids is None:
        prompt_block_ids = scope.prompt_block_hints or strategy.resolve_prompt_block_ids(
            plan, scope
        )

    prompt_blocks: list[Stage4FrontierBlock] = []
    for block_id in prompt_block_ids:
        block = plan.get_block(block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
        prompt_block = strategy.project_prompt_block(plan, block, scope)
        if prompt_block is not None:
            prompt_blocks.append(prompt_block)

    if requires_barrier_validation is None:
        requires_barrier_validation = len(prompt_blocks) > 1
    return ResolvedRepairPlan(
        scope=scope,
        prompt_blocks=tuple(prompt_blocks),
        requires_barrier_validation=requires_barrier_validation,
        uses_repair_campaign=strategy.uses_repair_campaign,
    )


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
    campaign = runtime.repair_campaign
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
