"""Structural scope projection and prompt-block planning for Stage 4 repair."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .helpers import _find_block_for_parameter, _ordered_block_ids
from .types import ResolvedRepairPlan, ResolvedRepairScope, Stage4RepairScopeStrategy

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )


def _identity_prompt_block_projection(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Keep the authored prompt block unchanged for this repair scope."""
    del plan, scope
    return block


def _narrow_prompt_block_to_scope_parameters(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Project a prompt block onto the scoped subset of semantic parameters."""
    del plan
    if not scope.parameter_names:
        return block

    allowed_parameter_names = tuple(
        parameter_name
        for parameter_name in block.parameter_names
        if parameter_name in scope.parameter_names
    )
    if not allowed_parameter_names:
        return None
    return _replace_block_parameter_surface(
        block,
        allowed_parameter_names=allowed_parameter_names,
        label_suffix="repair-local parameters only",
    )


def _replace_block_parameter_surface(
    block: Stage4FrontierBlock,
    *,
    allowed_parameter_names: tuple[str, ...],
    label_suffix: str,
) -> Stage4FrontierBlock:
    """Return one prompt block with a narrowed parameter coverage surface."""
    if allowed_parameter_names == block.parameter_names:
        return block

    return replace(
        block,
        label=f"{block.label} ({label_suffix})",
        parameter_names=allowed_parameter_names,
        required_parameter_names=tuple(
            parameter_name
            for parameter_name in block.required_parameter_names
            if parameter_name in allowed_parameter_names
        ),
        optional_parameter_names=tuple(
            parameter_name
            for parameter_name in block.optional_parameter_names
            if parameter_name in allowed_parameter_names
        ),
        expand_neighbor_topology=False,
    )


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
        _replace_block_parameter_surface(
            block,
            allowed_parameter_names=allowed_parameter_names,
            label_suffix="internal SCC parameters only",
        ),
        construct_names=prompt_construct_names,
        variable_names=prompt_variable_names,
        expand_neighbor_topology=False,
    )


def _narrow_validator_prompt_block(
    plan: Stage4Plan,
    block: Stage4FrontierBlock,
    scope: ResolvedRepairScope,
) -> Stage4FrontierBlock | None:
    """Respect validator-local parameter hints before widening to SCC closure."""
    narrowed_block = _narrow_prompt_block_to_scope_parameters(plan, block, scope)
    if narrowed_block is None:
        return None
    return _narrow_effect_prompt_block_to_scc(plan, narrowed_block, scope)


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
        raise ValueError(
            "Stage 4 SCC drift routing requires construct-level attribution"
        )

    closed_scc_ids: set[str] = set()
    for construct_name in construct_names:
        scc_id = topology.get_scc_id(construct_name)
        if scc_id is not None:
            closed_scc_ids.add(scc_id)

    if not closed_scc_ids:
        constructs = ", ".join(construct_names)
        raise ValueError(
            "Stage 4 SCC drift routing could not map construct attribution to any SCC: "
            f"{constructs}"
        )

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


def _validator_strategy_block_ids(
    plan: Stage4Plan,
    scope: ResolvedRepairScope,
) -> tuple[str, ...]:
    """Preserve validator-local block and parameter hints before SCC escalation."""
    if scope.prompt_block_hints:
        return scope.prompt_block_hints
    if scope.parameter_names:
        return _direct_writer_block_ids(plan, scope.parameter_names)
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
            resolve_prompt_block_ids=_validator_strategy_block_ids,
            project_prompt_block=_narrow_validator_prompt_block,
            uses_repair_campaign=True,
        ),
        "direct_writer_blocks": Stage4RepairScopeStrategy(
            scope_kind="direct_writer_blocks",
            resolve_prompt_block_ids=_direct_writer_strategy_block_ids,
            project_prompt_block=_narrow_prompt_block_to_scope_parameters,
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
    if not prompt_block_ids:
        raise ValueError(
            "Stage 4 repair scope projection produced no prompt blocks for "
            f"{scope.scope_key!r}"
        )

    prompt_blocks: list[Stage4FrontierBlock] = []
    for block_id in prompt_block_ids:
        block = plan.get_block(block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
        prompt_block = strategy.project_prompt_block(plan, block, scope)
        if prompt_block is not None:
            prompt_blocks.append(prompt_block)

    if not prompt_blocks:
        raise ValueError(
            "Stage 4 repair scope projection removed every prompt block for "
            f"{scope.scope_key!r}"
        )
    if requires_barrier_validation is None:
        requires_barrier_validation = len(prompt_blocks) > 1
    return ResolvedRepairPlan(
        scope=scope,
        prompt_blocks=tuple(prompt_blocks),
        requires_barrier_validation=requires_barrier_validation,
        uses_repair_campaign=strategy.uses_repair_campaign,
    )
