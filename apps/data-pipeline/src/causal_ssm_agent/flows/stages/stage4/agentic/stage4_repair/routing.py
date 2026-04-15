"""Top-level Stage 4 validation routing decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compile import classify_compile_failure_route
from .prior import classify_prior_failure_blocks
from .sensitivity import classify_sensitivity_failure_blocks
from .types import (
    ResolvedRepairPlan,
    Stage4PriorRepairDecision,
    Stage4ValidationOutcomeDecision,
)

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import (
        Stage4RepairCampaignState,
        Stage4Runtime,
    )
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation


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
    if validation is not None and not validation.compile_ok:
        return Stage4ValidationOutcomeDecision(
            outcome="compile_error",
            repair_plan=classify_compile_failure_route(
                plan,
                active_block,
                validation.compile_error or feedback,
            ),
        )
    if (
        include_prior_predictive
        and validation is not None
        and validation.pp_checked
        and not validation.pp_valid
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
    if validation is not None and validation.has_sensitivity_failure:
        return Stage4ValidationOutcomeDecision(
            outcome="sensitivity_failure",
            repair_plan=classify_sensitivity_failure_blocks(
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
