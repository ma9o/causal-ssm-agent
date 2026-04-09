"""Stage 4 graph and snapshot projections for the web runtime."""

from __future__ import annotations

from typing import Any

from .stage4_block_specs import get_stage4_block_phase
from .stage4_navigation import (
    get_stage4_phase,
    make_stage4_runtime,
)
from .stage4_orchestrator import Stage4Plan, build_stage4_plan
from .stage4_skeleton import derive_deterministic_spec
from .stage4_state import (
    Stage4BlockCursor,
    Stage4DoneCursor,
    Stage4ModelSpecLockPendingCursor,
    Stage4RepairBarrierCursor,
    Stage4Runtime,
)


def project_stage4_graph(plan: Stage4Plan) -> dict[str, Any]:
    """Project the static Stage 4 graph topology from the immutable plan."""
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    prev_id: str | None = None

    for block in plan.model_blocks:
        nodes.append(
            {
                "id": block.id,
                "kind": block.kind,
                "label": block.label,
                "phase": get_stage4_block_phase(block.kind),
            }
        )
        if prev_id is not None:
            edges.append({"from": prev_id, "to": block.id, "kind": "forward"})
        prev_id = block.id

    lock_id = "__lock__"
    nodes.append(
        {
            "id": lock_id,
            "kind": "model_spec_lock",
            "label": "Lock Model Spec",
            "phase": "model_decisions",
        }
    )
    if prev_id is not None:
        edges.append({"from": prev_id, "to": lock_id, "kind": "phase_advance"})

    if plan.review_block is not None:
        nodes.append(
            {
                "id": plan.review_block.id,
                "kind": plan.review_block.kind,
                "label": plan.review_block.label,
                "phase": get_stage4_block_phase(plan.review_block.kind),
            }
        )
        edges.append({"from": lock_id, "to": plan.review_block.id, "kind": "phase_advance"})
        prev_id = plan.review_block.id
    else:
        prev_id = lock_id

    for index, block in enumerate(plan.prior_blocks):
        nodes.append(
            {
                "id": block.id,
                "kind": block.kind,
                "label": block.label,
                "phase": get_stage4_block_phase(block.kind),
            }
        )
        if index == 0:
            edges.append({"from": prev_id, "to": block.id, "kind": "phase_advance"})
        else:
            edges.append(
                {"from": plan.prior_blocks[index - 1].id, "to": block.id, "kind": "forward"}
            )

    last_prior_id = plan.prior_blocks[-1].id if plan.prior_blocks else prev_id

    if plan.prior_review_block is not None:
        nodes.append(
            {
                "id": plan.prior_review_block.id,
                "kind": plan.prior_review_block.kind,
                "label": plan.prior_review_block.label,
                "phase": get_stage4_block_phase(plan.prior_review_block.kind),
            }
        )
        if last_prior_id is not None:
            edges.append(
                {
                    "from": last_prior_id,
                    "to": plan.prior_review_block.id,
                    "kind": "repair_transition",
                }
            )

    repair_barrier_id = "__repair_barrier__"
    nodes.append(
        {
            "id": repair_barrier_id,
            "kind": "repair_barrier",
            "label": "Validate Repair Scope",
            "phase": "prior_blocks",
        }
    )
    if plan.prior_blocks and last_prior_id is not None:
        edges.append({"from": last_prior_id, "to": repair_barrier_id, "kind": "repair_transition"})
    if plan.prior_review_block is not None:
        edges.append(
            {
                "from": repair_barrier_id,
                "to": plan.prior_review_block.id,
                "kind": "repair_transition",
            }
        )

    done_id = "__done__"
    nodes.append({"id": done_id, "kind": "done", "label": "Done", "phase": "done"})
    edges.append({"from": last_prior_id, "to": done_id, "kind": "phase_advance"})
    edges.append({"from": repair_barrier_id, "to": done_id, "kind": "repair_transition"})
    if plan.prior_review_block is not None:
        edges.append({"from": plan.prior_review_block.id, "to": done_id, "kind": "phase_advance"})

    phases = [
        {"id": "model_decisions", "label": "Model Decisions"},
        {"id": "global_review", "label": "Global Review"},
        {"id": "prior_blocks", "label": "Prior Elicitation"},
        {"id": "global_prior_review", "label": "Prior Review"},
        {"id": "done", "label": "Complete"},
    ]
    return {"nodes": nodes, "edges": edges, "phases": phases}


def project_stage4_snapshot(plan: Stage4Plan, runtime: Stage4Runtime) -> dict[str, Any]:
    """Project a JSON-serializable Stage 4 runtime snapshot for the web UI."""
    cursor = runtime.cursor
    if isinstance(cursor, Stage4BlockCursor):
        cursor_dict: dict[str, Any] = {"kind": "block", "block_id": cursor.block_id}
    elif isinstance(cursor, Stage4ModelSpecLockPendingCursor):
        cursor_dict = {"kind": "model_spec_lock"}
    elif isinstance(cursor, Stage4RepairBarrierCursor):
        cursor_dict = {"kind": "repair_barrier", "scope_block_ids": list(cursor.scope_block_ids)}
    elif isinstance(cursor, Stage4DoneCursor):
        cursor_dict = {"kind": "done"}
    else:
        cursor_dict = {"kind": "unknown"}

    campaign = runtime.repair_campaign
    repair_dict: dict[str, Any] | None = None
    if campaign is not None:
        repair_dict = {
            "scope_kind": campaign.scope_kind,
            "scope_block_ids": list(campaign.scope_block_ids),
            "completed_block_ids": list(campaign.completed_block_ids),
        }

    return {
        "cursor": cursor_dict,
        "block_status": dict(runtime.block_status),
        "model_spec_locked": runtime.accepted.model_spec is not None,
        "repair_campaign": repair_dict,
        "phase": get_stage4_phase(runtime, plan=plan),
    }


def project_stage4_initial_state(
    causal_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the initial Stage 4 graph and snapshot before agent startup work begins."""
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    return project_stage4_graph(plan), project_stage4_snapshot(plan, runtime)
