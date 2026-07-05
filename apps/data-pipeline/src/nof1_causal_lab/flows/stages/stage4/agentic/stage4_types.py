"""Shared Stage 4 runtime types used across the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    authored_priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None
