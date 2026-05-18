"""Aggregate stage-owned tool metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.flows.stages.stage0.contracts import (
    IS_INTERACTIVE_STAGE as STAGE0_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage0.contracts import (
    STAGE0_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.stages.stage1a.contracts import (
    IS_INTERACTIVE_STAGE as STAGE1A_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage1a.contracts import (
    STAGE1A_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.stages.stage1b.contracts import (
    IS_INTERACTIVE_STAGE as STAGE1B_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage1b.contracts import (
    STAGE1B_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.stages.stage2.contracts import (
    IS_INTERACTIVE_STAGE as STAGE2_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage2.contracts import (
    STAGE2_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.stages.stage4.contracts import (
    IS_INTERACTIVE_STAGE as STAGE4_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage4.contracts import (
    STAGE4_TOOL_CONTRACTS,
)
from nof1_causal_lab.flows.stages.stage6.contracts import (
    IS_INTERACTIVE_STAGE as STAGE6_INTERACTIVE,
)
from nof1_causal_lab.flows.stages.stage6.contracts import (
    STAGE6_TOOL_CONTRACTS,
)

if TYPE_CHECKING:
    from nof1_causal_lab.flows.contracts_base import ToolContract

STAGE_TOOLS: dict[str, list[ToolContract]] = {
    "stage-0": STAGE0_TOOL_CONTRACTS,
    "stage-1a": STAGE1A_TOOL_CONTRACTS,
    "stage-1b": STAGE1B_TOOL_CONTRACTS,
    "stage-2": STAGE2_TOOL_CONTRACTS,
    "stage-4": STAGE4_TOOL_CONTRACTS,
    "stage-6": STAGE6_TOOL_CONTRACTS,
}

INTERACTIVE_STAGES: frozenset[str] = frozenset(
    stage_id
    for stage_id, is_interactive in (
        ("stage-0", STAGE0_INTERACTIVE),
        ("stage-1a", STAGE1A_INTERACTIVE),
        ("stage-1b", STAGE1B_INTERACTIVE),
        ("stage-2", STAGE2_INTERACTIVE),
        ("stage-4", STAGE4_INTERACTIVE),
        ("stage-6", STAGE6_INTERACTIVE),
    )
    if is_interactive
)
