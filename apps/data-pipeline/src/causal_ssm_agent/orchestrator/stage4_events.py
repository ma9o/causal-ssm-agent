"""Typed reducer events for Stage 4 orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stage4_feedback import Stage4ValidationPacket
    from .stage4_repair import ResolvedRepairPlan


@dataclass(frozen=True)
class Stage4AcceptedStatePersistedEvent:
    """Persist accepted Stage 4 output without advancing a prompt block."""

    stage_output: dict[str, Any]


@dataclass(frozen=True)
class Stage4BlockAcceptedEvent:
    """Accept one Stage 4 block and optionally persist its accepted output."""

    block_id: str
    transition_payload: dict[str, Any] | None = None
    distribution_choice: dict[str, Any] | None = None
    stage_output: dict[str, Any] | None = None


@dataclass(frozen=True)
class Stage4RepairPlannedEvent:
    """Route the reducer into a repair plan, optionally keeping one block accepted."""

    repair_plan: ResolvedRepairPlan
    accepted_block_id: str | None = None
    accepted_transition_payload: dict[str, Any] | None = None
    distribution_choice: dict[str, Any] | None = None
    stage_output: dict[str, Any] | None = None


@dataclass(frozen=True)
class Stage4BarrierValidationPassedEvent:
    """Commit a successful repair-campaign barrier validation."""

    representative_block_id: str
    success_packet: Stage4ValidationPacket


Stage4ReducerEvent = (
    Stage4AcceptedStatePersistedEvent
    | Stage4BlockAcceptedEvent
    | Stage4RepairPlannedEvent
    | Stage4BarrierValidationPassedEvent
)
