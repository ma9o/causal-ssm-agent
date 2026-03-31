"""Stage 4 repair routing helpers.

Owns failure localization, structural scope resolution, and monotone
campaign-aware repair escalation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan, Stage4RepairTopology
    from .stage4_state import Stage4Runtime


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

    @property
    def parameter_hints(self) -> tuple[str, ...]:
        """Return deterministic seed parameters for scope synthesis."""
        return tuple(dict.fromkeys([*self.direct_parameters, *self.supporting_parameters]))


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
class ResolvedRepairPlan:
    """Prompt execution plan for one resolved structural repair scope."""

    scope: ResolvedRepairScope
    prompt_blocks: tuple[Stage4FrontierBlock, ...]
    requires_barrier_validation: bool = False

    @property
    def scope_kind(self) -> str:
        return self.scope.scope_kind

    @property
    def scope_rank(self) -> int:
        return self.scope.scope_rank

    @property
    def scope_key(self) -> str:
        return self.scope.scope_key

    @property
    def reason(self) -> str:
        return self.scope.reason

    @property
    def failure_family(self) -> tuple[Any, ...]:
        return self.scope.failure_family

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return self.scope.diagnostic_codes

    @property
    def pathology_certificate(self) -> PriorPathologyCertificate | None:
        return self.scope.pathology_certificate

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.id for block in self.prompt_blocks)


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


def _effect_prompt_block_for_structural_scope(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Project a structural drift scope onto the narrowest authored effect prompt."""
    if block.kind != "effect_prior" or scope.scope_kind not in {
        "scc_drift_subsystem",
        "validator_scope",
    }:
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


def build_repair_plan(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
    *,
    prompt_block_ids: tuple[str, ...] | None = None,
    requires_barrier_validation: bool | None = None,
) -> ResolvedRepairPlan:
    """Project a structural scope into the prompt blocks Stage 4 should run."""
    if prompt_block_ids is None:
        if scope.prompt_block_hints:
            prompt_block_ids = scope.prompt_block_hints
        elif scope.scope_kind == "local_drift_motif":
            prompt_block_ids = _local_drift_motif_block_ids(plan, scope.parameter_names)
        elif scope.scope_kind == "reciprocal_pair":
            prompt_block_ids = _reciprocal_pair_block_ids(plan, scope.parameter_names)
        elif scope.scope_kind in {"scc_drift_subsystem", "validator_scope"}:
            prompt_block_ids = _scc_drift_subsystem_block_ids(plan, scope.construct_names)
        elif scope.scope_kind == "direct_writer_blocks":
            prompt_block_ids = _ordered_block_ids(
                plan,
                {
                    block.id
                    for parameter_name in scope.parameter_names
                    if (block := _find_block_for_parameter(plan, parameter_name)) is not None
                },
            )
        elif scope.scope_kind == "global_prior_review":
            prior_review_id = plan.prior_review_block_id
            prompt_block_ids = () if prior_review_id is None else (prior_review_id,)
        else:
            prompt_block_ids = ()

    prompt_blocks: list[Stage4FrontierBlock] = []
    for block_id in prompt_block_ids:
        block = plan.get_block(block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
        prompt_block = _effect_prompt_block_for_structural_scope(plan, block, scope)
        if prompt_block is not None:
            prompt_blocks.append(prompt_block)

    if requires_barrier_validation is None:
        requires_barrier_validation = len(prompt_blocks) > 1
    return ResolvedRepairPlan(
        scope=scope,
        prompt_blocks=tuple(prompt_blocks),
        requires_barrier_validation=requires_barrier_validation,
    )


def _classify_compile_failure_route(
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
                reason="compile failure names an owned parameter or indicator exactly",
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
                reason="compile failure during whole-system prior review requires another global pass",
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
            reason="compile failure could not be localized more narrowly",
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


def _localize_prior_failure(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureLocalization:
    """Localize a failed PP validation into deterministic structural evidence."""
    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    failed = [result for result in validation.prior_predictive_diagnostics if not result.is_valid]
    diagnostic_codes = tuple(sorted(dict.fromkeys(result.code for result in failed if result.code)))
    supporting_codes = {code for result in failed for code in result.supporting_codes if code}

    direct_parameters = tuple(
        sorted(
            dict.fromkeys(
                parameter_name
                for result in failed
                for parameter_name in (
                    result.related_parameters or ([result.parameter] if result.parameter else [])
                )
                if parameter_name and _find_block_for_parameter(plan, parameter_name) is not None
            )
        )
    )

    supporting_parameters = tuple(
        sorted(
            dict.fromkeys(
                parameter_name
                for diagnostic in _compile_diagnostics_for_supporting_codes(
                    validation,
                    supporting_codes=supporting_codes,
                )
                for parameter_name in (
                    diagnostic.related_parameters
                    or ([diagnostic.parameter] if diagnostic.parameter else [])
                )
                if parameter_name and _find_block_for_parameter(plan, parameter_name) is not None
            )
        )
    )

    validator_repair_scope = next(
        (result.repair_scope for result in failed if result.repair_scope is not None),
        None,
    )
    supporting_compile_diagnostics = _compile_diagnostics_for_supporting_codes(
        validation,
        supporting_codes=supporting_codes,
    )
    pp_certificates = [
        result.pathology_certificate
        for result in failed
        if result.pathology_certificate is not None
    ]
    supporting_compile_certificates = [
        diagnostic.pathology_certificate
        for diagnostic in supporting_compile_diagnostics
        if diagnostic.pathology_certificate is not None
    ]
    pathology_certificate = None
    if pp_certificates:
        pathology_certificate = max(pp_certificates, key=_certificate_order_key)
    elif supporting_compile_certificates:
        # Same-scope retry gating should prefer the certificate for the actual
        # failed PP pathology, not the supporting compile warning. Compile-side
        # certificates are only a fallback when PP has no comparable metric.
        pathology_certificate = max(
            supporting_compile_certificates,
            key=_certificate_order_key,
        )

    topology = plan.repair_topology
    construct_names = tuple(
        dict.fromkeys(
            [
                *_parameter_construct_names(topology, direct_parameters),
                *_parameter_construct_names(topology, supporting_parameters),
                *_validator_scope_construct_names(validator_repair_scope),
            ]
        )
    )
    if not construct_names:
        construct_names = tuple(
            dict.fromkeys(
                name
                for result in failed
                for parameter_name in (result.related_parameters or [])
                for name in topology.parameter_construct_names.get(parameter_name, ())
            )
        )

    failure_family = (
        tuple(diagnostic_codes),
        tuple(sorted(supporting_codes)),
        tuple(sorted(construct_names)),
    )
    issues_text = " ".join(result.issue or "" for result in failed).lower()
    has_global_failure = any(result.parameter in GLOBAL_FAILURE_SITES for result in failed)
    return Stage4FailureLocalization(
        failure_family=failure_family,
        diagnostic_codes=diagnostic_codes,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        construct_names=construct_names,
        validator_repair_scope=validator_repair_scope,
        pathology_certificate=pathology_certificate,
        has_global_failure=has_global_failure,
        issues_text=issues_text,
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


def _support_failure_scope(
    plan: Stage4Plan,
    localization: Stage4FailureLocalization,
) -> ResolvedRepairScope | None:
    """Localize support violations to indicator-decision blocks when possible."""
    if (
        "support check" not in localization.issues_text
        and "outside support" not in localization.issues_text
    ):
        return None

    topology = plan.repair_topology
    for indicator_name, block_id in topology.indicator_to_decision_block_id.items():
        if indicator_name in localization.issues_text:
            return _resolved_repair_scope(
                scope_kind="likelihood_support",
                scope_rank=0,
                reason="global support failure names an indicator likelihood",
                failure_family=localization.failure_family,
                prompt_block_hints=(block_id,),
                diagnostic_codes=localization.diagnostic_codes,
                scope_token=block_id,
            )

    for block in plan.model_blocks:
        if block.kind == "indicator_decision":
            return _resolved_repair_scope(
                scope_kind="likelihood_support",
                scope_rank=0,
                reason="support failure requires indicator-decision repair",
                failure_family=localization.failure_family,
                prompt_block_hints=(block.id,),
                diagnostic_codes=localization.diagnostic_codes,
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


def _build_scope_candidates(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    localization: Stage4FailureLocalization,
) -> list[ResolvedRepairScope]:
    """Build ordered deterministic scope candidates for this failure family."""
    del active_block
    candidates: list[ResolvedRepairScope] = []

    def add_candidate(
        *,
        scope_kind: str,
        scope_rank: int,
        reason: str,
        parameter_names: tuple[str, ...] = (),
        construct_names: tuple[str, ...] = (),
        scope_token: str | None = None,
    ) -> None:
        if not parameter_names and not construct_names and scope_token is None:
            return
        scope = _resolved_repair_scope(
            scope_kind=scope_kind,
            scope_rank=scope_rank,
            reason=reason,
            failure_family=localization.failure_family,
            parameter_names=parameter_names,
            construct_names=construct_names,
            diagnostic_codes=localization.diagnostic_codes,
            pathology_certificate=localization.pathology_certificate,
            scope_token=scope_token,
        )
        if any(existing.scope_key == scope.scope_key for existing in candidates):
            return
        candidates.append(scope)

    support_scope = _support_failure_scope(plan, localization)
    if support_scope is not None:
        candidates.append(support_scope)
        return candidates

    if localization.validator_repair_scope is not None:
        add_candidate(
            scope_kind="validator_scope",
            scope_rank=_VALIDATOR_SCOPE_RANK,
            reason="validator supplied a deterministic SCC repair scope",
            construct_names=_validator_scope_construct_names(localization.validator_repair_scope),
        )

    if _is_drift_related(localization) and localization.parameter_hints:
        add_candidate(
            scope_kind="local_drift_motif",
            scope_rank=0,
            reason="drift-related PP failure should first repair the local drift motif",
            parameter_names=localization.parameter_hints,
            construct_names=_parameter_construct_names(
                plan.repair_topology, localization.parameter_hints
            ),
        )

        add_candidate(
            scope_kind="reciprocal_pair",
            scope_rank=1,
            reason="reciprocal feedback should be repaired jointly before widening further",
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
        )

        add_candidate(
            scope_kind="scc_drift_subsystem",
            scope_rank=2,
            reason="persistent drift pathology requires an SCC-closed repair subsystem",
            parameter_names=localization.parameter_hints,
            construct_names=localization.construct_names,
        )
    elif localization.direct_parameters:
        add_candidate(
            scope_kind="direct_writer_blocks",
            scope_rank=0,
            reason="diagnostic related_parameters map directly to Stage 4 blocks",
            parameter_names=localization.direct_parameters,
            construct_names=localization.construct_names,
        )

    if localization.has_global_failure:
        add_candidate(
            scope_kind="global_prior_review",
            scope_rank=_GLOBAL_REVIEW_SCOPE_RANK,
            reason="global PP failure could not be localized to a smaller bounded scope",
            parameter_names=localization.parameter_hints,
            construct_names=localization.construct_names,
            scope_token="prior_system",
        )

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


def _classify_prior_failure_blocks(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    validation: AssemblyValidation | None,
    runtime: Stage4Runtime,
) -> ResolvedRepairPlan:
    """Route prior-validation failures to the smallest monotone repair scope."""
    if validation is None:
        return build_repair_plan(
            plan,
            _resolved_repair_scope(
                scope_kind="active_block_fallback",
                scope_rank=0,
                reason="missing validation payload",
                failure_family=("missing_validation", active_block.id),
                scope_token=active_block.id,
            ),
            prompt_block_ids=(active_block.id,),
            requires_barrier_validation=False,
        )

    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    localization = _localize_prior_failure(plan, validation)
    candidates = _build_scope_candidates(plan, active_block, localization)
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

    return build_repair_plan(
        plan,
        _resolved_repair_scope(
            scope_kind="active_block_fallback",
            scope_rank=0,
            reason="non-global failure could not be localized more narrowly",
            failure_family=localization.failure_family,
            diagnostic_codes=localization.diagnostic_codes,
            scope_token=active_block.id,
        ),
        prompt_block_ids=(active_block.id,),
        requires_barrier_validation=False,
    )
